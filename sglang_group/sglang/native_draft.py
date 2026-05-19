"""SGLang-native draft backend for heterogeneous-vocabulary proposals.

This backend is intentionally independent from SGLang's built-in EAGLE worker:
EAGLE assumes target/draft scheduling state can be advanced together, while the
SGLANG_GROUP methods re-tokenize text between heterogeneous vocabularies. The
backend below uses SGLang's low-level ModelRunner for draft forward passes and
keeps proposal generation scratch-local so rejected draft tokens cannot corrupt
future draft state.
"""

from __future__ import annotations

import copy
import dataclasses
import logging
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from threading import RLock
from types import SimpleNamespace
from typing import Sequence

from .config import GroupSGLangConfig

logger = logging.getLogger(__name__)


class _ScratchTreeCache(SimpleNamespace):
    def supports_swa(self) -> bool:
        return False

    def supports_mamba(self) -> bool:
        return False

    def is_chunk_cache(self) -> bool:
        return False

    def is_tree_cache(self) -> bool:
        return True

    def evict(self, *args, **kwargs) -> None:
        return None

    def pretty_print(self) -> None:
        return None

    def available_and_evictable_str(self) -> str:
        allocator = self.token_to_kv_pool_allocator
        available = getattr(allocator, "available_size", lambda: "unknown")()
        return f"available={available}, evictable=0"


@dataclass
class _NativeDraftSnapshot:
    next_token_logits: object
    batch_attrs: dict[str, object]
    req_attrs: list[tuple[object, dict[str, object]]]
    allocator_state: object
    req_to_token_rows: list[tuple[int, object]]


@dataclass
class SGLangNativeDraftSession:
    backend: "SGLangNativeDraftBackend"
    batch: object
    next_token_logits: object
    rid: str
    accepted_input_ids: tuple[int, ...]
    _snapshot: _NativeDraftSnapshot | None = None

    def decode(self, token_id: int) -> object:
        self.next_token_logits = self.backend.decode(self, token_id)
        return self.next_token_logits

    def commit_tokens(self, token_ids: Sequence[int]) -> None:
        suffix = tuple(int(token_id) for token_id in token_ids)
        for token_id in suffix:
            self.decode(token_id)
        self.accepted_input_ids = self.accepted_input_ids + suffix
        self._validate_restored_context("commit")

    def begin_speculative(self) -> "SGLangNativeDraftSession":
        if self._snapshot is not None:
            raise RuntimeError("SGLang-native draft session already has an active snapshot.")
        self._snapshot = self._take_snapshot()
        return self

    def rollback_speculative(self) -> None:
        if self._snapshot is None:
            return
        snapshot = self._snapshot
        self._snapshot = None

        allocator = self.backend.model_runner.token_to_kv_pool_allocator
        restore_state = getattr(allocator, "restore_state", None)
        if not callable(restore_state):
            raise RuntimeError(
                "SGLang-native draft allocator cannot restore speculative state."
            )
        restore_state(snapshot.allocator_state)

        for name, value in snapshot.batch_attrs.items():
            setattr(self.batch, name, _clone_for_restore(value))
        for req, attrs in snapshot.req_attrs:
            for name, value in attrs.items():
                setattr(req, name, _clone_for_restore(value))
        req_to_token_pool = getattr(
            self.backend.model_runner, "req_to_token_pool", None
        )
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        if req_to_token is not None:
            for row_index, row_value in snapshot.req_to_token_rows:
                req_to_token[row_index].copy_(row_value)
        self.next_token_logits = _clone_for_restore(snapshot.next_token_logits)
        self._validate_restored_context("rollback")

    def _take_snapshot(self) -> _NativeDraftSnapshot:
        allocator = self.backend.model_runner.token_to_kv_pool_allocator
        backup_state = getattr(allocator, "backup_state", None)
        if not callable(backup_state):
            raise RuntimeError(
                "SGLang-native draft allocator cannot snapshot speculative state."
            )

        batch_attrs = _snapshot_batch_attrs(self.batch)
        req_attrs: list[tuple[object, dict[str, object]]] = []
        for req in getattr(self.batch, "reqs", []) or []:
            req_attrs.append((req, _snapshot_object_attrs(req)))

        req_to_token_rows = []
        req_to_token_pool = getattr(
            self.backend.model_runner, "req_to_token_pool", None
        )
        req_to_token = getattr(req_to_token_pool, "req_to_token", None)
        if req_to_token is not None:
            for req in getattr(self.batch, "reqs", []) or []:
                req_pool_idx = getattr(req, "req_pool_idx", None)
                if req_pool_idx is not None:
                    req_to_token_rows.append(
                        (int(req_pool_idx), req_to_token[int(req_pool_idx)].clone())
                    )

        return _NativeDraftSnapshot(
            next_token_logits=_clone_for_snapshot(self.next_token_logits),
            batch_attrs=batch_attrs,
            req_attrs=req_attrs,
            allocator_state=_clone_for_snapshot(backup_state()),
            req_to_token_rows=req_to_token_rows,
        )

    def _validate_restored_context(self, label: str) -> None:
        expected_len = len(self.accepted_input_ids)
        batch = self.batch
        seq_lens_cpu = getattr(batch, "seq_lens_cpu", None)
        if seq_lens_cpu is not None and len(seq_lens_cpu) > 0:
            actual = _int_index(seq_lens_cpu, 0)
            if actual != expected_len:
                raise RuntimeError(
                    "SGLang-native draft cache state mismatch after "
                    f"{label}: seq_lens_cpu={actual}, expected={expected_len}."
                )
        seq_lens = getattr(batch, "seq_lens", None)
        if seq_lens is not None and len(seq_lens) > 0:
            actual = _int_index(seq_lens, 0)
            if actual != expected_len:
                raise RuntimeError(
                    "SGLang-native draft cache state mismatch after "
                    f"{label}: seq_lens={actual}, expected={expected_len}."
                )
        for req in getattr(batch, "reqs", []) or []:
            committed = getattr(req, "kv_committed_len", expected_len)
            allocated = getattr(req, "kv_allocated_len", expected_len)
            if int(committed) != expected_len or int(allocated) != expected_len:
                raise RuntimeError(
                    "SGLang-native draft cache state mismatch after "
                    f"{label}: kv_committed_len={committed}, "
                    f"kv_allocated_len={allocated}, expected={expected_len}."
                )


class SGLangNativeDraftBackend:
    """Run the draft model through SGLang 0.5.9's ModelRunner.

    The backend keeps one accepted-context draft batch and snapshots allocator
    state before speculative decoding. Rejected speculative tokens are rolled
    back while accepted context can be extended incrementally on the next
    proposal.
    """

    def __init__(
        self,
        *,
        server_args: object,
        gpu_id: int,
        tp_rank: int,
        dp_rank: int | None,
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        config: GroupSGLangConfig,
        trust_remote_code: bool,
    ) -> None:
        if config.draft_device_map:
            raise ValueError(
                "SGLang-native draft backend does not support "
                "SGLANG_GROUP_DRAFT_DEVICE_MAP. Use backend=transformers for "
                "HF device_map placement, or place the SGLang worker with CUDA_VISIBLE_DEVICES."
            )

        self.config = config
        self.server_args = copy.copy(server_args)
        self.server_args.model_path = server_args.speculative_draft_model_path
        self.server_args.revision = getattr(
            server_args, "speculative_draft_model_revision", None
        ) or getattr(server_args, "revision", None)
        self.server_args.skip_tokenizer_init = True
        self.server_args.disable_cuda_graph = True
        self.server_args.disable_overlap_schedule = True
        # Do not inherit the target model quantization for a generic draft model.
        # If the draft checkpoint declares quantization in its config, SGLang can
        # still detect it. Use SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION to force one.
        self.server_args.quantization = config.native_draft_quantization
        self._configure_scratch_cache_size(server_args)
        if config.draft_dtype != "auto":
            self.server_args.dtype = _sglang_dtype_name(config.draft_dtype)

        self.target_tp_size = int(getattr(server_args, "tp_size", 1) or 1)
        self.draft_tp_mode = self._resolve_draft_tp_mode()
        if self.draft_tp_mode == "replica":
            self.server_args.tp_size = 1

        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.draft_tp_rank = 0 if self.draft_tp_mode == "replica" else tp_rank
        self.dp_rank = dp_rank
        self.moe_ep_rank = moe_ep_rank
        self.attn_cp_rank = attn_cp_rank
        self.moe_dp_rank = moe_dp_rank
        self.nccl_port = nccl_port

        self.tokenizer = self._load_tokenizer(
            server_args.speculative_draft_model_path,
            trust_remote_code=trust_remote_code,
        )
        self.draft_tp_group = self._create_draft_tp_group()
        if (
            self.draft_tp_mode == "replica"
            and self.target_tp_size > 1
            and self.draft_tp_group is None
        ):
            raise RuntimeError(
                "SGLANG_GROUP replicated draft TP group could not be created."
            )
        self.model_runner = self._load_model_runner()
        self.device = self.model_runner.device
        self._cached_session: SGLangNativeDraftSession | None = None
        self._cached_rid: str | None = None
        self._cached_input_ids: tuple[int, ...] = ()
        self._native_kv_cache_disabled_reason: str | None = None
        self._forward_lock: RLock | None = None
        self._forward_gate: object | None = None

        logger.info(
            "Initialized SGLang-native draft backend: draft=%s, device=%s, "
            "target_tp_size=%s, draft_tp_mode=%s, draft_tp_size=%s, "
            "draft_tp_rank=%s, cache_tokens=%s, max_requests=%s, kv_cache=%s, "
            "draft_tp_group_world_size=%s",
            server_args.speculative_draft_model_path,
            self.device,
            self.target_tp_size,
            self.draft_tp_mode,
            getattr(self.server_args, "tp_size", None),
            self.draft_tp_rank,
            getattr(self.server_args, "draft_runner_cache_size", None),
            getattr(self.server_args, "max_num_reqs", None),
            bool(config.native_draft_kv_cache),
            getattr(self.draft_tp_group, "world_size", None),
        )

    def _resolve_draft_tp_mode(self) -> str:
        requested = getattr(self.config, "native_draft_tp_mode", "auto")
        if requested != "auto":
            return requested
        if self.target_tp_size <= 1:
            return "independent"
        if self.config.ssd_mode == "async-hit":
            return "replica"
        return "independent"

    def _configure_scratch_cache_size(self, source_server_args: object) -> None:
        page_size = int(getattr(self.server_args, "page_size", 1) or 1)
        requested_tokens = self.config.native_draft_cache_tokens
        if requested_tokens is None and self.config.max_context_tokens is not None:
            draft_tokens = self.config.max_draft_tokens
            if draft_tokens is None:
                draft_tokens = getattr(
                    source_server_args,
                    "speculative_num_draft_tokens",
                    None,
                )
            requested_tokens = self.config.max_context_tokens + int(draft_tokens or 8) + 16

        if requested_tokens is not None:
            requested_tokens = _ceil_to_page(int(requested_tokens), page_size)
            source_tokens = getattr(
                source_server_args,
                "draft_runner_cache_size",
                requested_tokens,
            )
            if source_tokens:
                requested_tokens = min(requested_tokens, int(source_tokens))
            self.server_args.draft_runner_cache_size = requested_tokens

        max_requests = max(1, int(self.config.native_draft_max_requests))
        self.server_args.max_num_reqs = max_requests

    def _load_tokenizer(self, model_path: str, *, trust_remote_code: bool):
        try:
            from sglang.srt.utils.hf_transformers_utils import get_tokenizer

            return get_tokenizer(
                model_path,
                tokenizer_mode=getattr(self.server_args, "tokenizer_mode", "auto"),
                trust_remote_code=trust_remote_code,
                revision=getattr(self.server_args, "revision", None),
            )
        except Exception:
            logger.debug("Falling back to AutoTokenizer for draft tokenizer.", exc_info=True)
            from transformers import AutoTokenizer

            return AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=trust_remote_code,
                revision=getattr(self.server_args, "revision", None),
            )

    def _load_model_runner(self):
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.model_executor.model_runner import ModelRunner

        model_config = ModelConfig.from_server_args(
            self.server_args,
            model_path=self.server_args.model_path,
            model_revision=self.server_args.revision,
            is_draft_model=False,
        )
        with self._draft_tp_init_context():
            return ModelRunner(
                model_config=model_config,
                mem_fraction_static=self.server_args.mem_fraction_static,
                gpu_id=self.gpu_id,
                tp_rank=self.draft_tp_rank,
                tp_size=self.server_args.tp_size,
                moe_ep_rank=self.moe_ep_rank,
                moe_ep_size=self.server_args.ep_size,
                pp_rank=0,
                pp_size=1,
                nccl_port=self.nccl_port,
                dp_rank=self.dp_rank,
                attn_cp_rank=self.attn_cp_rank,
                moe_dp_rank=self.moe_dp_rank,
                server_args=self.server_args,
                is_draft_worker=True,
                req_to_token_pool=None,
                token_to_kv_pool_allocator=None,
            )

    @property
    def has_independent_tp_group(self) -> bool:
        return self.draft_tp_mode == "independent" and self.draft_tp_group is not None

    @property
    def uses_replicated_tp(self) -> bool:
        return self.draft_tp_mode == "replica" and (
            self.target_tp_size <= 1 or self.draft_tp_group is not None
        )

    def set_forward_lock(self, lock: RLock | None) -> None:
        self._forward_lock = lock

    def set_forward_gate(self, gate: object | None) -> None:
        self._forward_gate = gate

    def _create_draft_tp_group(self):
        if int(getattr(self.server_args, "tp_size", 1) or 1) <= 1:
            if self.target_tp_size > 1:
                return self._create_replicated_tp_group()
            return None
        return self._create_independent_tp_group()

    def _create_replicated_tp_group(self):
        try:
            import torch.distributed as dist
            from sglang.srt.distributed import get_tp_group
            from sglang.srt.distributed.parallel_state import (
                get_world_group,
                init_model_parallel_group,
            )

            if not dist.is_available() or not dist.is_initialized():
                return None

            target_tp_group = get_tp_group()
            world_group = get_world_group()
            backend = dist.get_backend(world_group.device_group)
            singleton_groups = [[int(rank)] for rank in target_tp_group.ranks]
            draft_tp_group = init_model_parallel_group(
                singleton_groups,
                target_tp_group.local_rank,
                backend,
                use_pynccl=True,
                use_custom_allreduce=False,
                use_message_queue_broadcaster=False,
                use_mscclpp_allreduce=False,
                use_torch_symm_mem_allreduce=False,
                group_name="sglang_group_draft_replica_tp",
                pynccl_use_current_stream=True,
            )
            barrier = getattr(draft_tp_group, "barrier", None)
            if callable(barrier):
                barrier()
            logger.info(
                "Created replicated SGLANG_GROUP draft TP group: ranks=%s",
                getattr(draft_tp_group, "ranks", None),
            )
            return draft_tp_group
        except Exception:
            logger.exception("Failed to create replicated SGLANG_GROUP draft TP group.")
            return None

    def _create_independent_tp_group(self):
        if int(getattr(self.server_args, "tp_size", 1) or 1) <= 1:
            return None

        try:
            import torch.distributed as dist
            from sglang.srt.distributed import get_tp_group
            from sglang.srt.distributed.parallel_state import (
                get_world_group,
                init_model_parallel_group,
            )

            if not dist.is_available() or not dist.is_initialized():
                return None

            target_tp_group = get_tp_group()
            world_group = get_world_group()
            backend = dist.get_backend(world_group.device_group)
            draft_tp_group = init_model_parallel_group(
                [list(target_tp_group.ranks)],
                target_tp_group.local_rank,
                backend,
                use_pynccl=True,
                use_custom_allreduce=False,
                use_message_queue_broadcaster=False,
                use_mscclpp_allreduce=False,
                use_torch_symm_mem_allreduce=False,
                group_name="sglang_group_draft_tp",
                pynccl_use_current_stream=True,
            )
            barrier = getattr(draft_tp_group, "barrier", None)
            if callable(barrier):
                barrier()
            logger.info(
                "Created independent SGLANG_GROUP draft TP group: ranks=%s",
                getattr(draft_tp_group, "ranks", None),
            )
            return draft_tp_group
        except Exception:
            logger.exception("Failed to create independent SGLANG_GROUP draft TP group.")
            return None

    def _draft_tp_init_context(self):
        if self.draft_tp_group is None:
            return nullcontext()
        return _patch_draft_parallel_groups(self.draft_tp_group)

    def clear(self) -> None:
        self._drop_cached_session()
        self._clear_pools()

    def evict(self, rids: Sequence[str]) -> bool:
        rid_set = {str(rid) for rid in rids}
        if self._cached_rid is not None and self._cached_rid in rid_set:
            self.clear()
            return True
        return False

    def cache_size(self) -> int:
        return 1 if self._cached_session is not None else 0

    def disable_kv_cache(self, reason: str) -> None:
        self._native_kv_cache_disabled_reason = reason
        self.clear()
        logger.warning(
            "Disabled SGLang-native accepted-context draft KV cache: %s. "
            "Future proposals will use safe rebuild.",
            reason,
        )

    def _drop_cached_session(self) -> None:
        self._cached_session = None
        self._cached_rid = None
        self._cached_input_ids = ()

    def _clear_pools(self) -> None:
        model_runner = getattr(self, "model_runner", None)
        req_to_token_pool = getattr(model_runner, "req_to_token_pool", None)
        clear_req_pool = getattr(req_to_token_pool, "clear", None)
        if callable(clear_req_pool):
            clear_req_pool()
        allocator = getattr(model_runner, "token_to_kv_pool_allocator", None)
        clear_allocator = getattr(allocator, "clear", None)
        if callable(clear_allocator):
            clear_allocator()

    def ensure_session(
        self,
        input_ids: Sequence[int],
        *,
        rid: str,
    ) -> tuple[SGLangNativeDraftSession, str]:
        ids = tuple(int(token_id) for token_id in input_ids)
        if not ids:
            raise ValueError("SGLang-native draft context must contain at least one token.")

        rid = str(rid)
        use_kv_cache = (
            self.config.enable_draft_cache and self.config.native_draft_kv_cache
            and getattr(self, "_native_kv_cache_disabled_reason", None) is None
        )
        if use_kv_cache and self._cached_session is not None:
            if self._cached_rid == rid and self._cached_input_ids == ids:
                return self._cached_session, "sglang-hit"
            if (
                self._cached_rid == rid
                and len(ids) > len(self._cached_input_ids)
                and ids[: len(self._cached_input_ids)] == self._cached_input_ids
            ):
                suffix = ids[len(self._cached_input_ids) :]
                try:
                    self._cached_session.commit_tokens(suffix)
                except Exception as exc:
                    self.disable_kv_cache(f"accepted suffix commit failed: {exc}")
                    session = self.prefill(ids, rid=rid)
                    return session, "sglang-rebuild"
                self._cached_input_ids = ids
                return self._cached_session, "sglang-extend"

        session = self.prefill(ids, rid=rid)
        if use_kv_cache:
            self._cached_session = session
            self._cached_rid = rid
            self._cached_input_ids = ids
        else:
            self._cached_session = None
            self._cached_rid = None
            self._cached_input_ids = ()
        return session, "sglang-rebuild"

    def prefill(self, input_ids: Sequence[int], *, rid: str) -> SGLangNativeDraftSession:
        import torch

        ids = tuple(int(token_id) for token_id in input_ids)
        if not ids:
            raise ValueError("SGLang-native draft context must contain at least one token.")

        with torch.no_grad():
            self._drop_cached_session()
            self._clear_pools()
            batch = self._make_extend_batch(list(ids), rid=rid)
            logits = self._forward_batch(batch)
        return SGLangNativeDraftSession(
            backend=self,
            batch=batch,
            next_token_logits=logits,
            rid=str(rid),
            accepted_input_ids=ids,
        )

    def decode(self, session: SGLangNativeDraftSession, token_id: int) -> object:
        import torch

        batch = session.batch
        with torch.no_grad():
            batch.output_ids = torch.tensor(
                [int(token_id)],
                dtype=torch.int64,
                device=self.device,
            )
            batch.prepare_for_decode()
            return self._forward_batch(batch)

    def _make_extend_batch(self, input_ids: list[int], *, rid: str):
        from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
        from sglang.srt.sampling.sampling_params import SamplingParams
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

        sampling_params = SamplingParams(temperature=0.0, max_new_tokens=1)
        req = Req(
            rid=f"sglang-group-draft-{rid}",
            origin_input_text="",
            origin_input_ids=input_ids,
            sampling_params=sampling_params,
        )
        req.fill_ids = req.origin_input_ids
        req.logprob_start_len = -1
        req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

        tree_cache = _ScratchTreeCache(
            page_size=self.model_runner.server_args.page_size,
            device=self.device,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            req_to_token_pool=self.model_runner.req_to_token_pool,
        )
        batch = ScheduleBatch.init_new(
            reqs=[req],
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()
        return batch

    def _forward_batch(self, batch: object) -> object:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        with self._locked_forward_context():
            with self._draft_tp_init_context():
                self._maybe_prepare_mlp_sync_batch(batch)
                model_worker_batch = batch.get_model_worker_batch()
                forward_batch = ForwardBatch.init_new(
                    model_worker_batch,
                    self.model_runner,
                )
                logits_output = self.model_runner.forward(forward_batch).logits_output
                return logits_output.next_token_logits

    @contextmanager
    def _locked_forward_context(self):
        if self._forward_gate is not None:
            draft_context = getattr(self._forward_gate, "draft_context", None)
            if callable(draft_context):
                with draft_context():
                    yield
                return
        if self._forward_lock is None:
            yield
            return
        with self._forward_lock:
            yield

    def _maybe_prepare_mlp_sync_batch(self, batch: object) -> None:
        try:
            from sglang.srt.managers.scheduler_dp_attn_mixin import (
                prepare_mlp_sync_batch_raw,
            )
            from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
        except Exception:
            return

        if not require_mlp_sync(self.model_runner.server_args):
            return

        prepare_mlp_sync_batch_raw(
            batch,
            dp_size=self.model_runner.server_args.dp_size,
            attn_tp_size=1,
            tp_group=self.model_runner.tp_group,
            get_idle_batch=None,
            disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
            require_mlp_tp_gather=require_mlp_tp_gather(self.model_runner.server_args),
            disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
            offload_tags=set(),
        )


@contextmanager
def _patch_draft_parallel_groups(tp_group):
    """Patch SGLang TP globals while constructing a draft ModelRunner.

    SGLang 0.5.9 only exposes a public patch helper for `_TP`. The draft
    ModelRunner also snapshots `_ATTN_TP` during construction, so patch both
    here and restore immediately after loading.
    """

    try:
        import sglang.srt.distributed.parallel_state as parallel_state
    except Exception:
        logger.exception("Failed to import SGLang parallel state for draft TP patch.")
        yield
        return

    if getattr(parallel_state, "_TP_STATE_PATCHED", False):
        raise RuntimeError("SGLang tensor parallel state is already patched.")

    old_tp = getattr(parallel_state, "_TP", None)
    old_attn_tp = getattr(parallel_state, "_ATTN_TP", None)
    parallel_state._TP_STATE_PATCHED = True
    parallel_state._TP = tp_group
    parallel_state._ATTN_TP = tp_group
    try:
        yield
    finally:
        parallel_state._ATTN_TP = old_attn_tp
        parallel_state._TP = old_tp
        parallel_state._TP_STATE_PATCHED = False


def _sglang_dtype_name(dtype_name: str) -> str:
    normalized = dtype_name.lower()
    mapping = {
        "float16": "float16",
        "fp16": "float16",
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float32": "float32",
        "fp32": "float32",
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported SGLANG_GROUP_DRAFT_DTYPE: {dtype_name}") from exc


def _ceil_to_page(value: int, page_size: int) -> int:
    if page_size <= 1:
        return value
    return ((value + page_size - 1) // page_size) * page_size


def _snapshot_batch_attrs(batch: object) -> dict[str, object]:
    excluded = {
        "reqs",
        "req_to_token_pool",
        "token_to_kv_pool_allocator",
        "tree_cache",
        "model_config",
    }
    names: set[str] = set()
    if dataclasses.is_dataclass(batch):
        names.update(field.name for field in dataclasses.fields(batch))
    try:
        names.update(vars(batch).keys())
    except TypeError:
        pass
    return {
        name: _clone_for_snapshot(getattr(batch, name))
        for name in sorted(names)
        if name not in excluded and hasattr(batch, name)
    }


def _snapshot_object_attrs(obj: object) -> dict[str, object]:
    try:
        attrs = vars(obj)
    except TypeError:
        return {}
    return {name: _clone_for_snapshot(value) for name, value in attrs.items()}


def _int_index(value, index: int) -> int:
    item = value[index]
    try:
        return int(item.item())
    except AttributeError:
        return int(item)


def _clone_for_snapshot(value):
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(_clone_for_snapshot(item) for item in value)
    if isinstance(value, dict):
        return {key: _clone_for_snapshot(item) for key, item in value.items()}
    try:
        return copy.copy(value)
    except Exception:
        return value


def _clone_for_restore(value):
    return _clone_for_snapshot(value)
