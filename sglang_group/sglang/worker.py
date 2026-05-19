"""SGLang spec-v1 worker for unified SGLANG_GROUP methods."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from time import monotonic
from typing import Optional

import torch

from sglang.srt.layers.utils.logprob import add_output_logprobs_for_spec_v1
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.ngram_info import NgramVerifyInput
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.speculative.spec_utils import generate_token_bitmask

try:
    from sglang.srt.observability.req_time_stats import set_time_batch
    from sglang.srt.observability.trace import get_global_tracing_enabled
except ModuleNotFoundError:

    def set_time_batch(*args, **kwargs) -> None:
        return None

    def get_global_tracing_enabled() -> bool:
        return False

from .async_proposer import AsyncProposalRequest, AsyncProposalService
from .candidates import build_linear_candidate_rows, build_tree_candidate_rows
from .config import GroupSGLangConfig
from .cuda_overlap import CudaStreamOverlapController
from .proposer import BaseProposal, HeterogeneousDraftProposer, SamplingRequest
from .verify_input import make_group_verify_input

logger = logging.getLogger(__name__)


@dataclass
class SGLangGroupWorkerStats:
    batches: int = 0
    verify_batches: int = 0
    target_only_batches: int = 0
    requests: int = 0
    proposed_target_tokens: int = 0
    accepted_draft_tokens: int = 0
    evicted_requests: int = 0
    itl_batches: int = 0
    slem_batches: int = 0
    tli_batches: int = 0
    proposal_cache_hits: int = 0
    proposal_cache_misses: int = 0
    proposal_cache_skips: int = 0
    accepted_on_proposal_cache_hit: int = 0
    accepted_on_proposal_cache_miss: int = 0
    draft_cache_hits: int = 0
    draft_cache_extensions: int = 0
    draft_cache_rebuilds: int = 0
    draft_prepare_wall_time_s: float = 0.0
    target_verify_wall_time_s: float = 0.0
    verify_postprocess_wall_time_s: float = 0.0
    target_only_wall_time_s: float = 0.0
    extend_wall_time_s: float = 0.0
    spec_skip_batches: int = 0

    def snapshot(self) -> dict[str, int | float]:
        data = self.__dict__.copy()
        proposed = self.proposed_target_tokens
        accepted = self.accepted_draft_tokens
        data["acceptance_rate"] = accepted / proposed if proposed else 0.0
        if self.verify_batches:
            data["draft_prepare_ms_per_verify_batch"] = (
                self.draft_prepare_wall_time_s * 1000.0 / self.verify_batches
            )
            data["target_verify_ms_per_verify_batch"] = (
                self.target_verify_wall_time_s * 1000.0 / self.verify_batches
            )
            data["verify_postprocess_ms_per_verify_batch"] = (
                self.verify_postprocess_wall_time_s * 1000.0 / self.verify_batches
            )
        return data


def _result_field_names() -> set[str]:
    return set(getattr(GenerationBatchResult, "__dataclass_fields__", {}))


def _make_generation_result(
    *,
    logits_output,
    next_token_ids,
    accepted_tokens: int = 0,
    accepted_per_req_cpu: list[int] | None = None,
    can_run_cuda_graph: bool = False,
    accept_lens=None,
) -> GenerationBatchResult:
    fields = _result_field_names()
    kwargs = {
        "logits_output": logits_output,
        "next_token_ids": next_token_ids,
        "can_run_cuda_graph": can_run_cuda_graph,
        "accept_lens": accept_lens,
    }
    if "num_correct_drafts" in fields:
        kwargs["num_correct_drafts"] = accepted_tokens
        kwargs["num_correct_drafts_per_req_cpu"] = accepted_per_req_cpu
    else:
        kwargs["num_accepted_tokens"] = accepted_tokens
        kwargs["accept_length_per_req_cpu"] = accepted_per_req_cpu
    return GenerationBatchResult(
        **{key: value for key, value in kwargs.items() if key in fields}
    )


def _result_accepted_tokens(result: object) -> int:
    if hasattr(result, "num_accepted_tokens"):
        return int(getattr(result, "num_accepted_tokens") or 0)
    return int(getattr(result, "num_correct_drafts", 0) or 0)


def _result_accepted_per_req(result: object) -> list[int] | None:
    if hasattr(result, "accept_length_per_req_cpu"):
        return getattr(result, "accept_length_per_req_cpu")
    return getattr(result, "num_correct_drafts_per_req_cpu", None)


def _flatten_next_token_ids(next_token_ids):
    if isinstance(next_token_ids, list):
        return torch.cat([item.reshape(-1) for item in next_token_ids])
    return next_token_ids.reshape(-1)


def _spec_tensor(spec_info: object, modern_name: str, legacy_name: str):
    if hasattr(spec_info, modern_name):
        return getattr(spec_info, modern_name)
    return getattr(spec_info, legacy_name)


def _accept_lengths(verify_input: object):
    if hasattr(verify_input, "num_correct_drafts"):
        return getattr(verify_input, "num_correct_drafts")
    return getattr(verify_input, "accept_length", None)


def _accept_lens_for_result(verify_input: object):
    if hasattr(verify_input, "num_accept_tokens"):
        return getattr(verify_input, "num_accept_tokens")
    if hasattr(verify_input, "accept_length"):
        return getattr(verify_input, "accept_length", None)
    return None


class SGLangGroupWorker:
    """SGLang worker integrating ITL, SLEM, and TLI proposal methods.

    `auto` mode is batch-level on SGLang 0.5.9: greedy batches use
    `itl-base-slem`, and sampling batches use `itl-base-tli` so the verifier has
    real draft probabilities.
    Target verification, KV slot allocation, request mutation, and output
    post-processing reuse SGLang's NGRAM spec-v1 verifier.
    """

    speculative_num_draft_tokens: int

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ) -> None:
        self.server_args = server_args
        self.target_worker = target_worker
        self.model_runner = target_worker.model_runner
        self.tp_rank = tp_rank
        self.page_size = server_args.page_size
        self.max_draft_token_num = int(server_args.speculative_num_draft_tokens)
        self.speculative_num_draft_tokens = self.max_draft_token_num
        self.device = f"cuda:{gpu_id}" if gpu_id >= 0 else "cuda"
        self._warned_sampling_method_fallback = False

        self.config = GroupSGLangConfig.from_env(default_draft_device=self.device)
        if self.config.disable_cuda_graph and hasattr(server_args, "disable_cuda_graph"):
            server_args.disable_cuda_graph = True

        native_backend = None
        if self.config.draft_backend == "sglang":
            from .native_draft import SGLangNativeDraftBackend

            native_backend = SGLangNativeDraftBackend(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                config=self.config,
                trust_remote_code=bool(server_args.trust_remote_code),
            )

        self._validate_async_hit_tp_safety(server_args, native_backend)

        self.proposer = HeterogeneousDraftProposer(
            draft_model_path=server_args.speculative_draft_model_path,
            target_tokenizer=target_worker.tokenizer,
            target_vocab_size=self._target_vocab_size(target_worker),
            config=self.config,
            trust_remote_code=bool(server_args.trust_remote_code),
            native_backend=native_backend,
        )
        self.stats = SGLangGroupWorkerStats()
        self.async_proposer = None
        self.cuda_overlap = None
        if self.config.ssd_mode != "off":
            self.cuda_overlap = CudaStreamOverlapController(
                enabled=self.config.cuda_overlap,
                device=self.device,
            )
            self.async_proposer = AsyncProposalService(
                self.proposer,
                max_workers=self.config.ssd_prefetch_workers,
                max_entries=self.config.ssd_max_prefetch,
                cuda_overlap=self.cuda_overlap,
            )
        self._last_candidate_rows = None
        self._last_metrics_log_time = monotonic()
        logger.info(
            "Initialized SGLANG_GROUP worker: draft=%s, method=%s, draft_backend=%s, "
            "max_draft_tokens=%s, ssd_mode=%s, tree_branch_factor=%s",
            server_args.speculative_draft_model_path,
            self.config.method,
            self.config.draft_backend,
            self.max_draft_token_num,
            self.config.ssd_mode,
            self.config.tree_branch_factor,
        )

    def clear_cache_pool(self):
        if self.async_proposer is not None:
            self.async_proposer.clear()
        else:
            self.proposer.clear()

    def update_weights_from_tensor(self, recv_req):
        return self.target_worker.update_weights_from_tensor(recv_req)

    def add_external_corpus(self, corpus_id: str, token_chunks: list[list[int]]) -> int:
        logger.warning("SGLANG_GROUP ignores NGRAM external corpus load: %s", corpus_id)
        return 0

    def commit_corpus_load(self, corpus_id: str, loaded_token_count: int) -> None:
        return None

    def remove_external_corpus(self, corpus_id: str) -> None:
        return None

    def list_external_corpora(self) -> dict[str, int]:
        return {}

    def _validate_async_hit_tp_safety(
        self,
        server_args: ServerArgs,
        native_backend: object | None,
    ) -> None:
        needs_independent_group = (
            self.config.ssd_mode == "async-hit"
            and self.config.draft_backend == "sglang"
            and int(getattr(server_args, "tp_size", 1) or 1) > 1
        )
        if not needs_independent_group:
            return
        if native_backend is not None and bool(
            getattr(native_backend, "has_independent_tp_group", False)
        ):
            logger.info(
                "SGLANG_GROUP async-hit is using an independent draft TP group "
                "for tp_size=%s.",
                getattr(server_args, "tp_size", 1),
            )
            return
        raise RuntimeError(
            "SGLANG_GROUP async-hit with draft_backend=sglang and tp_size > 1 "
            "requires an independent draft TP group. The group could not be "
            "created, so refusing to start instead of falling back or risking "
            "NCCL collective deadlock."
        )

    def post_process_batch_result_prefill(self, batch: ScheduleBatch, result) -> None:
        self._schedule_after_scheduler_commit(batch)

    def post_process_batch_result_decode(self, batch: ScheduleBatch, result) -> None:
        self._schedule_after_scheduler_commit(batch)

    def forward_batch_generation(self, batch: ScheduleBatch) -> GenerationBatchResult:
        return self._forward_batch_generation(batch, allow_mixed_async_hit=True)

    def _forward_batch_generation(
        self,
        batch: ScheduleBatch,
        *,
        allow_mixed_async_hit: bool,
    ) -> GenerationBatchResult:
        self.stats.batches += 1
        self.stats.requests += batch.batch_size()
        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            model_worker_batch = batch.get_model_worker_batch()
            extend_start = monotonic()
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
            self.stats.extend_wall_time_s += monotonic() - extend_start
            return _make_generation_result(
                logits_output=batch_result.logits_output,
                next_token_ids=batch_result.next_token_ids,
                accepted_tokens=0,
                can_run_cuda_graph=batch_result.can_run_cuda_graph,
            )

        method = self._method_for_batch(batch)
        if allow_mixed_async_hit:
            mixed_result = self._maybe_forward_mixed_async_hit(batch, method=method)
            if mixed_result is not None:
                return mixed_result

        set_time_batch(batch.reqs, "set_spec_draft_start_time", trace_only=True)
        draft_prepare_start = monotonic()
        spec_prepared = self._prepare_for_speculative_decoding(batch, method=method)
        self.stats.draft_prepare_wall_time_s += monotonic() - draft_prepare_start
        set_time_batch(batch.reqs, "set_spec_draft_end_time", trace_only=True)

        model_worker_batch = batch.get_model_worker_batch()
        spec_info = getattr(model_worker_batch, "spec_info", None)
        accepted_tokens = 0
        accept_lens = None
        accepted_per_req_cpu = None

        if model_worker_batch.forward_mode.is_target_verify():
            self.stats.verify_batches += 1
            if method == "itl":
                self.stats.itl_batches += 1
            elif method == "itl-base-slem":
                self.stats.slem_batches += 1
            else:
                self.stats.tli_batches += 1

            if batch.has_grammar:
                retrieve_next_token = _spec_tensor(
                    spec_info, "retrieve_next_token", "retrive_next_token"
                )
                retrieve_next_sibling = _spec_tensor(
                    spec_info, "retrieve_next_sibling", "retrive_next_sibling"
                )
                retrieve_next_token_cpu = retrieve_next_token.cpu()
                retrieve_next_sibling_cpu = retrieve_next_sibling.cpu()
                draft_tokens_cpu = spec_info.draft_token.view(
                    retrieve_next_token.shape
                ).cpu()

            set_time_batch(batch.reqs, "set_spec_verify_start_time", trace_only=True)
            target_verify_start = monotonic()
            batch_result = self.target_worker.forward_batch_generation(
                model_worker_batch,
                is_verify=True,
            )
            self.stats.target_verify_wall_time_s += monotonic() - target_verify_start
            logits_output, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.can_run_cuda_graph,
            )

            verify_postprocess_start = monotonic()
            verify_input: NgramVerifyInput = model_worker_batch.spec_info
            vocab_mask = None
            if batch.has_grammar:
                vocab_mask = generate_token_bitmask(
                    batch.reqs,
                    verify_input,
                    retrieve_next_token_cpu,
                    retrieve_next_sibling_cpu,
                    draft_tokens_cpu,
                    batch.sampling_info.vocab_size,
                )
                if vocab_mask is not None:
                    assert verify_input.grammar is not None
                    retrieve_next_token = _spec_tensor(
                        verify_input, "retrieve_next_token", "retrive_next_token"
                    )
                    vocab_mask = vocab_mask.to(retrieve_next_token.device)
                    batch.sampling_info.vocab_mask = None

            logits_output, next_token_ids, accepted_tokens = verify_input.verify(
                batch,
                logits_output,
                self.page_size,
                vocab_mask,
            )
            accept_lengths = _accept_lengths(verify_input)
            accepted_per_req_cpu = (
                accept_lengths.cpu().tolist() if accept_lengths is not None else None
            )
            if accepted_per_req_cpu is not None:
                self.stats.accepted_draft_tokens += sum(accepted_per_req_cpu)
                self._record_acceptance_by_cache(accepted_per_req_cpu)

            if get_global_tracing_enabled():
                for idx, req in enumerate(batch.reqs):
                    correct = (
                        accept_lengths[idx].item()
                        if accept_lengths is not None
                        else 0
                    )
                    if hasattr(req.time_stats, "set_spec_verify_end_time"):
                        req.time_stats.set_spec_verify_end_time(
                            num_correct_drafts=correct
                        )

            accept_lens = _accept_lens_for_result(verify_input)
            if batch.return_logprob:
                add_output_logprobs_for_spec_v1(batch, verify_input, logits_output)
            self.stats.verify_postprocess_wall_time_s += (
                monotonic() - verify_postprocess_start
            )
            self._schedule_async_prefetch(batch, method=method)
            self._evict_finished_requests(batch)
            self._maybe_log_metrics()
            batch.forward_mode = ForwardMode.DECODE
        else:
            self.stats.target_only_batches += 1
            if not spec_prepared:
                self.stats.spec_skip_batches += 1
            target_only_start = monotonic()
            batch_result = self.target_worker.forward_batch_generation(model_worker_batch)
            self.stats.target_only_wall_time_s += monotonic() - target_only_start
            logits_output, next_token_ids, can_run_cuda_graph = (
                batch_result.logits_output,
                batch_result.next_token_ids,
                batch_result.can_run_cuda_graph,
            )

        return _make_generation_result(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accepted_tokens=accepted_tokens,
            accepted_per_req_cpu=accepted_per_req_cpu,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=accept_lens,
        )

    def _maybe_forward_mixed_async_hit(
        self,
        batch: ScheduleBatch,
        *,
        method: str,
    ) -> GenerationBatchResult | None:
        if self.async_proposer is None or self.config.ssd_mode != "async-hit":
            return None
        if batch.return_logprob or batch.has_grammar or batch.return_hidden_states:
            return None

        ready_indices, miss_indices = self._async_hit_ready_miss_indices(
            batch,
            method=method,
        )
        if not ready_indices or not miss_indices:
            return None

        ready_batch = self._filtered_subbatch(batch, ready_indices)
        miss_batch = self._filtered_subbatch(batch, miss_indices)

        ready_result = self._forward_batch_generation(
            ready_batch,
            allow_mixed_async_hit=False,
        )
        miss_result = self._forward_batch_generation(
            miss_batch,
            allow_mixed_async_hit=False,
        )
        self.stats.batches -= 1
        self.stats.requests -= batch.batch_size()
        self._merge_subbatch_state(
            batch,
            [(ready_indices, ready_batch), (miss_indices, miss_batch)],
        )
        return self._merge_subbatch_results(
            batch,
            [(ready_indices, ready_result), (miss_indices, miss_result)],
        )

    def _async_hit_ready_miss_indices(
        self,
        batch: ScheduleBatch,
        *,
        method: str,
    ) -> tuple[list[int], list[int]]:
        assert self.async_proposer is not None
        ready_indices: list[int] = []
        miss_indices: list[int] = []
        max_target_tokens = self.max_draft_token_num - 1
        for idx, req in enumerate(batch.reqs):
            if getattr(req, "multimodal_inputs", None) is not None:
                miss_indices.append(idx)
                continue
            try:
                current_ids = self._current_target_ids(req)
                current_text = self._decode_target(current_ids)
                request = AsyncProposalRequest(
                    rid=str(req.rid),
                    current_text=current_text,
                    current_target_ids=current_ids,
                    max_target_tokens=max_target_tokens,
                    method=method,
                    sampling=self._sampling_request(batch, idx),
                )
                if self.async_proposer.has_ready(request):
                    ready_indices.append(idx)
                else:
                    miss_indices.append(idx)
            except Exception:
                logger.exception(
                    "SGLANG_GROUP SSD readiness check failed for request %s", req.rid
                )
                miss_indices.append(idx)
        return ready_indices, miss_indices

    def _filtered_subbatch(
        self,
        batch: ScheduleBatch,
        indices: list[int],
    ) -> ScheduleBatch:
        subbatch = replace(
            batch,
            reqs=list(batch.reqs),
            req_pool_indices=batch.req_pool_indices.clone(),
            seq_lens=batch.seq_lens.clone(),
            seq_lens_cpu=batch.seq_lens_cpu.clone(),
            orig_seq_lens=batch.orig_seq_lens.clone(),
            output_ids=batch.output_ids.clone() if batch.output_ids is not None else None,
            sampling_info=replace(batch.sampling_info),
            multimodal_inputs=(
                list(batch.multimodal_inputs)
                if batch.multimodal_inputs is not None
                else None
            ),
        )
        subbatch.filter_batch(keep_indices=indices)
        return subbatch

    def _merge_subbatch_state(
        self,
        batch: ScheduleBatch,
        subbatches: list[tuple[list[int], ScheduleBatch]],
    ) -> None:
        for indices, subbatch in subbatches:
            index_tensor = torch.tensor(indices, dtype=torch.int64, device=self.device)
            batch.seq_lens[index_tensor] = subbatch.seq_lens
            batch.orig_seq_lens[index_tensor] = subbatch.orig_seq_lens
            if batch.output_ids is not None and subbatch.output_ids is not None:
                batch.output_ids[index_tensor] = subbatch.output_ids
            for pos, index in enumerate(indices):
                batch.seq_lens_cpu[index] = subbatch.seq_lens_cpu[pos]
        batch.seq_lens_sum = int(batch.seq_lens.sum().item())
        batch.forward_mode = ForwardMode.DECODE
        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM

    def _merge_subbatch_results(
        self,
        batch: ScheduleBatch,
        results: list[tuple[list[int], GenerationBatchResult]],
    ) -> GenerationBatchResult:
        accepted_per_req = [0 for _ in batch.reqs]
        accepted_total = 0
        next_token_parts = []
        logits_output = None
        can_run_cuda_graph = True
        for indices, result in results:
            if logits_output is None:
                logits_output = result.logits_output
            can_run_cuda_graph = can_run_cuda_graph and bool(result.can_run_cuda_graph)
            accepted_total += _result_accepted_tokens(result)
            per_req = _result_accepted_per_req(result)
            if per_req is not None:
                for pos, index in enumerate(indices):
                    accepted_per_req[index] = int(per_req[pos])
            next_token_ids = getattr(result, "next_token_ids", None)
            if next_token_ids is not None:
                next_token_parts.append(_flatten_next_token_ids(next_token_ids))

        next_token_ids = (
            torch.cat(next_token_parts)
            if next_token_parts
            else torch.empty(0, dtype=torch.int64, device=self.device)
        )
        return _make_generation_result(
            logits_output=logits_output,
            next_token_ids=next_token_ids,
            accepted_tokens=accepted_total,
            accepted_per_req_cpu=accepted_per_req,
            can_run_cuda_graph=can_run_cuda_graph,
            accept_lens=None,
        )

    def _schedule_after_scheduler_commit(self, batch: ScheduleBatch) -> None:
        if self.async_proposer is None:
            return
        try:
            method = self._method_for_batch(batch)
        except Exception:
            logger.exception("SGLANG_GROUP SSD scheduler hook could not select method.")
            return
        self._schedule_async_prefetch(batch, method=method)

    def _method_for_batch(self, batch: ScheduleBatch) -> str:
        max_temperature = _max_sampling_temperature(batch.sampling_info)
        method = self.config.method_for_batch(
            is_all_greedy=batch.sampling_info.is_all_greedy,
            max_temperature=max_temperature,
        )
        if not batch.sampling_info.is_all_greedy and method in {"itl", "itl-base-slem"}:
            if not self._warned_sampling_method_fallback:
                logger.warning(
                    "SGLANG_GROUP routes non-greedy sampling through itl-base-tli for "
                    "correct draft probabilities. Use greedy decoding for "
                    "itl/itl-base-slem."
                )
                self._warned_sampling_method_fallback = True
            method = "itl-base-tli"
        return method

    def _prepare_for_speculative_decoding(self, batch: ScheduleBatch, *, method: str) -> bool:
        bs = batch.batch_size()
        candidate_rows = self._build_candidate_rows(batch, method=method)
        if not candidate_rows.rows:
            self._last_candidate_rows = candidate_rows
            return False
        self._last_candidate_rows = candidate_rows
        rows = candidate_rows.rows
        draft_token_num = candidate_rows.draft_token_num
        draft_token = torch.tensor(
            [token for row in rows for token in row],
            dtype=torch.int64,
            device=self.device,
        )

        (
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            positions,
            custom_mask,
        ) = self._build_verify_tree_tensors(batch, candidate_rows)

        batch.spec_algorithm = SpeculativeAlgorithm.NGRAM
        batch.forward_mode = ForwardMode.TARGET_VERIFY

        draft_probs = None
        if method == "itl-base-tli":
            draft_probs = self._stack_draft_probs(
                candidate_rows.draft_prob_rows,
                draft_token_num=draft_token_num,
            )

        batch.spec_info = make_group_verify_input(
            draft_token=draft_token,
            tree_mask=custom_mask,
            positions=positions,
            retrieve_index=retrieve_index,
            retrieve_next_token=retrieve_next_token,
            retrieve_next_sibling=retrieve_next_sibling,
            draft_token_num=draft_token_num,
            draft_probs=draft_probs,
            verify_backend=self.config.verify_backend,
        )
        batch.spec_info.prepare_for_verify(batch, self.page_size)
        return True

    def _build_candidate_rows(self, batch: ScheduleBatch, *, method: str):
        roots: list[int] = []
        target_rows: list[tuple[int, ...]] = []
        target_tree_rows: list[tuple[int, ...]] = []
        parent_tree_rows: list[tuple[int, ...]] = []
        draft_prob_rows: list[object | None] = []
        proposal_cache_events: list[str] = []
        draft_cache_events: list[str] = []
        proposal_methods: list[str] = []
        max_target_tokens = self.max_draft_token_num - 1
        async_misses = 0

        for idx, req in enumerate(batch.reqs):
            root = self._root_token(req)
            roots.append(root)
            proposal = BaseProposal(method, (), (), None, "skipped", 0)
            if getattr(req, "multimodal_inputs", None) is None:
                try:
                    current_ids = self._current_target_ids(req)
                    current_text = self._decode_target(current_ids)
                    async_request = AsyncProposalRequest(
                        rid=str(req.rid),
                        current_text=current_text,
                        current_target_ids=current_ids,
                        max_target_tokens=max_target_tokens,
                        method=method,
                        sampling=self._sampling_request(batch, idx),
                    )
                    if self.async_proposer is not None:
                        ready = self.async_proposer.get_ready(async_request)
                        if ready is not None:
                            proposal = ready
                        else:
                            async_misses += 1
                            self.async_proposer.submit(async_request)
                            if self.config.ssd_mode == "async-sync-fallback":
                                proposal = self.async_proposer.propose_sync(async_request)
                    else:
                        proposal = self.proposer.propose(
                            str(req.rid),
                            current_text,
                            current_ids,
                            max_target_tokens=max_target_tokens,
                            method=method,
                            sampling=async_request.sampling,
                        )
                except Exception:
                    logger.exception("SGLANG_GROUP proposal failed for request %s", req.rid)

            target_rows.append(proposal.target_token_ids[:max_target_tokens])
            tree_tokens = proposal.target_tree_token_ids or proposal.target_token_ids
            tree_parents = (
                proposal.target_tree_parent_indices
                if proposal.target_tree_parent_indices
                else tuple(range(len(tree_tokens)))
            )
            target_tree_rows.append(tuple(tree_tokens[:max_target_tokens]))
            parent_tree_rows.append(tuple(tree_parents[:max_target_tokens]))
            draft_prob_rows.append(proposal.draft_prob_rows)
            proposal_cache_events.append(proposal.proposal_cache_event)
            draft_cache_events.append(proposal.cache_event)
            proposal_methods.append(proposal.method)

        if (
            self.async_proposer is not None
            and self.config.ssd_mode == "async-hit"
            and async_misses > 0
        ):
            target_rows = [() for _ in target_rows]
            target_tree_rows = [() for _ in target_tree_rows]
            parent_tree_rows = [() for _ in parent_tree_rows]
            draft_prob_rows = [None for _ in draft_prob_rows]
            proposal_cache_events = ["async-miss" for _ in proposal_cache_events]
            draft_cache_events = ["async-miss" for _ in draft_cache_events]

        if self.config.tree_branch_factor > 1 and method != "itl-base-tli":
            candidate_rows = build_tree_candidate_rows(
                roots,
                target_tree_rows,
                parent_tree_rows,
                max_draft_token_num=self.max_draft_token_num,
                proposal_cache_events=proposal_cache_events,
                draft_cache_events=draft_cache_events,
                proposal_methods=proposal_methods,
            )
        else:
            candidate_rows = build_linear_candidate_rows(
                roots,
                target_rows,
                max_draft_token_num=self.max_draft_token_num,
                draft_prob_rows=draft_prob_rows if method == "itl-base-tli" else None,
                proposal_cache_events=proposal_cache_events,
                draft_cache_events=draft_cache_events,
                proposal_methods=proposal_methods,
            )
        self.stats.proposed_target_tokens += candidate_rows.proposed_target_tokens
        self._record_candidate_cache_events(candidate_rows)
        return candidate_rows

    def _build_verify_tree_tensors(self, batch: ScheduleBatch, candidate_rows):
        bs = batch.batch_size()
        width = candidate_rows.draft_token_num
        retrieve_index = torch.arange(
            bs * width,
            dtype=torch.int64,
            device=self.device,
        ).reshape(bs, width)
        retrieve_next_token = torch.full((bs, width), -1, dtype=torch.int64, device=self.device)
        retrieve_next_sibling = torch.full((bs, width), -1, dtype=torch.int64, device=self.device)

        positions_parts = []
        custom_mask_parts = []
        for row_idx in range(bs):
            parents = candidate_rows.parent_rows[row_idx]
            depths = candidate_rows.depth_rows[row_idx]
            for parent in range(width):
                children = [idx for idx, value in enumerate(parents) if value == parent]
                if children:
                    retrieve_next_token[row_idx, parent] = children[0]
                    for left, right in zip(children, children[1:]):
                        retrieve_next_sibling[row_idx, left] = right

            depth_tensor = torch.tensor(depths, dtype=torch.int64, device=self.device)
            positions_parts.append(batch.seq_lens[row_idx].to(torch.int64) + depth_tensor)

            prefix_len = _seq_len_cpu_item(batch.seq_lens_cpu, row_idx)
            prefix_mask = torch.ones(
                (width, max(prefix_len - 1, 0)),
                dtype=torch.bool,
                device=self.device,
            )
            tree_mask = _tree_ancestor_mask(parents, device=self.device)
            custom_mask_parts.append(torch.cat((prefix_mask, tree_mask), dim=1).flatten())

        positions = (
            torch.cat(positions_parts)
            if positions_parts
            else torch.empty(0, dtype=torch.int64, device=self.device)
        )
        custom_mask = (
            torch.cat(custom_mask_parts)
            if custom_mask_parts
            else torch.empty(0, dtype=torch.bool, device=self.device)
        )
        return retrieve_index, retrieve_next_token, retrieve_next_sibling, positions, custom_mask

    def _schedule_async_prefetch(self, batch: ScheduleBatch, *, method: str) -> None:
        if self.async_proposer is None:
            return
        max_target_tokens = self.max_draft_token_num - 1
        for idx, req in enumerate(batch.reqs):
            if req.finished() or getattr(req, "is_retracted", False):
                continue
            if getattr(req, "multimodal_inputs", None) is not None:
                continue
            try:
                current_ids = self._current_target_ids(req)
                current_text = self._decode_target(current_ids)
                self.async_proposer.submit(
                    AsyncProposalRequest(
                        rid=str(req.rid),
                        current_text=current_text,
                        current_target_ids=current_ids,
                        max_target_tokens=max_target_tokens,
                        method=method,
                        sampling=self._sampling_request(batch, idx),
                    )
                )
            except Exception:
                logger.exception(
                    "SGLANG_GROUP SSD prefetch failed for request %s", req.rid
                )

    def _stack_draft_probs(self, rows: tuple[object | None, ...], *, draft_token_num: int):
        if not rows or any(row is None for row in rows):
            return None

        stacked_rows = []
        for row in rows:
            assert row is not None
            if len(row) < draft_token_num:
                return None
            stacked_rows.append(torch.stack(list(row[:draft_token_num]), dim=0))
        return torch.stack(stacked_rows, dim=0).to(self.device, non_blocking=True)

    def _evict_finished_requests(self, batch: ScheduleBatch) -> None:
        finished_req_ids = [
            str(req.rid)
            for req in batch.reqs
            if req.finished() or getattr(req, "is_retracted", False)
        ]
        if finished_req_ids:
            if self.async_proposer is not None:
                self.async_proposer.evict(finished_req_ids)
            else:
                self.proposer.evict(finished_req_ids)
            self.stats.evicted_requests += len(finished_req_ids)

    def _record_candidate_cache_events(self, candidate_rows) -> None:
        for event in getattr(candidate_rows, "proposal_cache_events", ()) or ():
            if event in {"hit", "async-hit"}:
                self.stats.proposal_cache_hits += 1
            elif event == "miss":
                self.stats.proposal_cache_misses += 1
            elif event in {"skip", "async-miss"}:
                self.stats.proposal_cache_skips += 1

        for event in getattr(candidate_rows, "draft_cache_events", ()) or ():
            if event.startswith("proposal-"):
                continue
            if event.endswith("hit"):
                self.stats.draft_cache_hits += 1
            elif event.endswith("extend"):
                self.stats.draft_cache_extensions += 1
            elif event.endswith("rebuild"):
                self.stats.draft_cache_rebuilds += 1

    def _record_acceptance_by_cache(self, accepted_per_req_cpu: list[int]) -> None:
        candidate_rows = self._last_candidate_rows
        if candidate_rows is None:
            return
        events = getattr(candidate_rows, "proposal_cache_events", ()) or ()
        for event, accepted in zip(events, accepted_per_req_cpu):
            accepted_count = int(accepted)
            if event in {"hit", "async-hit"}:
                self.stats.accepted_on_proposal_cache_hit += accepted_count
            elif event == "miss":
                self.stats.accepted_on_proposal_cache_miss += accepted_count

    def _maybe_log_metrics(self) -> None:
        interval = self.config.metrics_log_interval
        if interval is None:
            return
        now = monotonic()
        if now - self._last_metrics_log_time < interval:
            return
        self._last_metrics_log_time = now
        logger.info(
            "SGLANG_GROUP metrics: worker=%s proposer=%s async=%s cache_size=%s "
            "proposal_cache_size=%s",
            self.stats.snapshot(),
            self.proposer.stats.snapshot(),
            self.async_proposer.snapshot() if self.async_proposer is not None else None,
            self.proposer.cache_size(),
            self.proposer.proposal_cache_size(),
        )

    def _current_target_ids(self, req: object) -> tuple[int, ...]:
        input_ids = list(getattr(req, "origin_input_ids_unpadded", None) or req.origin_input_ids)
        return tuple(int(token_id) for token_id in input_ids + list(req.output_ids))

    def _decode_target(self, token_ids: tuple[int, ...]) -> str:
        tokenizer = self.target_worker.tokenizer
        try:
            return tokenizer.decode(
                list(token_ids),
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(list(token_ids))

    def _sampling_request(self, batch: ScheduleBatch, index: int) -> SamplingRequest:
        sampling_info = batch.sampling_info
        if sampling_info.is_all_greedy:
            return SamplingRequest(temperature=0.0)
        return SamplingRequest(
            temperature=float(
                _tensor_item(getattr(sampling_info, "temperatures", None), index, 1.0)
            ),
            top_k=int(_tensor_item(getattr(sampling_info, "top_ks", None), index, -1)),
            top_p=float(_tensor_item(getattr(sampling_info, "top_ps", None), index, 1.0)),
        )

    @staticmethod
    def _root_token(req: object) -> int:
        if getattr(req, "output_ids", None):
            return int(req.output_ids[-1])
        return int(req.origin_input_ids[-1])

    @staticmethod
    def _target_vocab_size(target_worker: TpModelWorker) -> int:
        for obj in (
            getattr(target_worker, "model_config", None),
            getattr(getattr(target_worker, "model_runner", None), "model_config", None),
            getattr(getattr(target_worker, "model_runner", None), "model", None),
        ):
            value = getattr(obj, "vocab_size", None)
            if value is not None:
                return int(value)
        tokenizer = getattr(target_worker, "tokenizer", None)
        try:
            return len(tokenizer)
        except Exception:
            vocab = tokenizer.get_vocab()
            return max(vocab.values()) + 1


def _max_sampling_temperature(sampling_info: object) -> float | None:
    if getattr(sampling_info, "is_all_greedy", False):
        return 0.0
    temperatures = getattr(sampling_info, "temperatures", None)
    if temperatures is None:
        return None
    try:
        if hasattr(temperatures, "detach"):
            values = temperatures.detach().flatten().cpu().tolist()
        elif hasattr(temperatures, "flatten"):
            values = temperatures.flatten().tolist()
        else:
            values = list(temperatures)
    except Exception:
        return None
    if not values:
        return None
    return max(float(value) for value in values)


def _tensor_item(value, index: int, default):
    if value is None:
        return default
    try:
        item = value[index]
    except Exception:
        return default
    try:
        if hasattr(item, "numel") and item.numel() > 1:
            item = item.flatten()[0]
        return item.item()
    except Exception:
        return item


def _seq_len_cpu_item(value, index: int) -> int:
    item = value[index]
    try:
        return int(item.item())
    except AttributeError:
        return int(item)


def _tree_ancestor_mask(parent_row: tuple[int, ...], *, device: str):
    width = len(parent_row)
    rows = []
    for index in range(width):
        ancestors = set()
        cursor = index
        while cursor >= 0 and cursor not in ancestors:
            ancestors.add(cursor)
            cursor = int(parent_row[cursor])
        rows.append([column in ancestors for column in range(width)])
    return torch.tensor(rows, dtype=torch.bool, device=device)
