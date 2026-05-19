"""HF draft proposer for unified SGLANG_GROUP methods."""

from __future__ import annotations

import copy
import hashlib
import logging
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Sequence

from sglang_group.alignment import dynamic_token_warping
from sglang_group.core import (
    VocabIntersection,
    decode_ids,
    encode_text,
    slem_target_proxies_from_assistant_window,
)

from .config import GroupSGLangConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SamplingRequest:
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0


@dataclass(frozen=True)
class BaseProposal:
    method: str
    draft_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    draft_prob_rows: object | None
    cache_event: str
    draft_context_tokens: int
    alignment_cost: float | None = None
    proposal_cache_event: str = "skip"
    tokenizer_bridge: str | None = None
    cuda_event: object | None = None
    cuda_overlap: bool = False
    target_tree_token_ids: tuple[int, ...] = ()
    target_tree_parent_indices: tuple[int, ...] = ()


@dataclass(frozen=True)
class ProposalCacheKey:
    rid: str
    method: str
    target_hash: str
    target_len: int
    draft_context_hash: str
    draft_context_len: int
    max_target_tokens: int
    temperature: float
    top_k: int
    top_p: float
    add_special_tokens: bool
    assistant_lookbehind: int
    target_lookbehind: int
    tokenizer_bridge: str
    tree_branch_factor: int
    tree_max_depth: int | None


@dataclass
class DraftRequestState:
    rid: str
    text: str
    input_ids: tuple[int, ...]
    past_key_values: object
    next_token_logits: object


@dataclass
class DraftProposerStats:
    proposals: int = 0
    itl_proposals: int = 0
    slem_proposals: int = 0
    tli_proposals: int = 0
    proposed_target_tokens: int = 0
    proposed_draft_tokens: int = 0
    cache_hits: int = 0
    cache_extensions: int = 0
    cache_rebuilds: int = 0
    cache_evictions: int = 0
    empty_proposals: int = 0
    failed_proposals: int = 0
    proposal_cache_hits: int = 0
    proposal_cache_misses: int = 0
    proposal_cache_stores: int = 0
    proposal_cache_evictions: int = 0
    proposal_cache_skips: int = 0
    bridge_uag_translations: int = 0
    bridge_segment_translations: int = 0
    bridge_empty_translations: int = 0
    tli_intersection_size: int = 0
    tree_proposals: int = 0
    tree_nodes: int = 0

    def snapshot(self) -> dict[str, int]:
        return {
            "proposals": self.proposals,
            "itl_proposals": self.itl_proposals,
            "slem_proposals": self.slem_proposals,
            "tli_proposals": self.tli_proposals,
            "proposed_target_tokens": self.proposed_target_tokens,
            "proposed_draft_tokens": self.proposed_draft_tokens,
            "cache_hits": self.cache_hits,
            "cache_extensions": self.cache_extensions,
            "cache_rebuilds": self.cache_rebuilds,
            "cache_evictions": self.cache_evictions,
            "empty_proposals": self.empty_proposals,
            "failed_proposals": self.failed_proposals,
            "proposal_cache_hits": self.proposal_cache_hits,
            "proposal_cache_misses": self.proposal_cache_misses,
            "proposal_cache_stores": self.proposal_cache_stores,
            "proposal_cache_evictions": self.proposal_cache_evictions,
            "proposal_cache_skips": self.proposal_cache_skips,
            "bridge_uag_translations": self.bridge_uag_translations,
            "bridge_segment_translations": self.bridge_segment_translations,
            "bridge_empty_translations": self.bridge_empty_translations,
            "tli_intersection_size": self.tli_intersection_size,
            "tree_proposals": self.tree_proposals,
            "tree_nodes": self.tree_nodes,
        }


class HeterogeneousDraftProposer:
    """Generate ITL, SLEM, or TLI proposals from one ordinary HF draft model."""

    def __init__(
        self,
        *,
        draft_model_path: str,
        target_tokenizer: object,
        target_vocab_size: int,
        config: GroupSGLangConfig,
        trust_remote_code: bool,
        native_backend: object | None = None,
    ) -> None:
        import torch

        self.config = config
        self.target_tokenizer = target_tokenizer
        self.target_vocab_size = int(target_vocab_size)
        self.native_backend = native_backend

        if self.native_backend is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            self.draft_tokenizer = AutoTokenizer.from_pretrained(
                draft_model_path,
                trust_remote_code=trust_remote_code,
            )

            model_kwargs: dict[str, object] = {"trust_remote_code": trust_remote_code}
            if config.draft_dtype != "auto":
                model_kwargs["torch_dtype"] = _torch_dtype(torch, config.draft_dtype)
            else:
                model_kwargs["torch_dtype"] = "auto"
            if config.draft_device_map:
                model_kwargs["device_map"] = config.draft_device_map

            self.draft_model = AutoModelForCausalLM.from_pretrained(
                draft_model_path,
                **model_kwargs,
            )
            if not config.draft_device_map and config.draft_device:
                self.draft_model.to(config.draft_device)
            self.draft_model.eval()
        else:
            self.draft_tokenizer = self.native_backend.tokenizer
            self.draft_model = None

        self.intersection: VocabIntersection | None = None
        self._valid_assistant_ids = None
        self._valid_target_ids = None

        self._states: OrderedDict[str, DraftRequestState] = OrderedDict()
        self._proposal_cache: OrderedDict[ProposalCacheKey, BaseProposal] = OrderedDict()
        self.stats = DraftProposerStats()

    def propose(
        self,
        rid: str,
        current_text: str,
        current_target_ids: Sequence[int],
        *,
        max_target_tokens: int,
        method: str,
        sampling: SamplingRequest | None = None,
    ) -> BaseProposal:
        self.stats.proposals += 1
        if max_target_tokens <= 0:
            self.stats.empty_proposals += 1
            return BaseProposal(method, (), (), None, "disabled", 0)

        try:
            sampling_req = sampling or SamplingRequest()
            self._count_method_proposal(method)
            context_ids = tuple(self._context_ids(current_text))
            cache_key = self._proposal_cache_key(
                rid=rid,
                method=method,
                current_target_ids=current_target_ids,
                context_ids=context_ids,
                max_target_tokens=max_target_tokens,
                sampling=sampling_req,
            )
            if cache_key is not None:
                cached_proposal = self._proposal_cache.get(cache_key)
                if cached_proposal is not None:
                    self._proposal_cache.move_to_end(cache_key)
                    self.stats.proposal_cache_hits += 1
                    proposal = replace(
                        cached_proposal,
                        cache_event=f"proposal-hit/{cached_proposal.cache_event}",
                        proposal_cache_event="hit",
                    )
                    self._record_proposal(proposal)
                    return proposal
                self.stats.proposal_cache_misses += 1
            else:
                self.stats.proposal_cache_skips += 1

            if method == "itl":
                proposal = self._propose_itl(
                    rid,
                    current_text,
                    current_target_ids,
                    context_ids=context_ids,
                    max_target_tokens=max_target_tokens,
                )
            elif method == "itl-base-slem":
                proposal = self._propose_slem(
                    rid,
                    current_text,
                    current_target_ids,
                    context_ids=context_ids,
                    max_target_tokens=max_target_tokens,
                )
            elif method == "itl-base-tli":
                proposal = self._propose_tli(
                    rid,
                    current_text,
                    context_ids=context_ids,
                    max_target_tokens=max_target_tokens,
                    sampling=sampling_req,
                )
            else:
                raise ValueError(f"Unsupported SGLANG_GROUP method: {method}")

            proposal = replace(
                proposal,
                proposal_cache_event="miss" if cache_key is not None else "skip",
            )
            self._store_proposal_cache(cache_key, proposal)
            self._record_proposal(proposal)
            return proposal
        except Exception:
            self.stats.failed_proposals += 1
            raise

    def evict(self, rids: Sequence[str]) -> None:
        if self.native_backend is not None:
            evict = getattr(self.native_backend, "evict", None)
            if callable(evict) and evict(rids):
                self.stats.cache_evictions += 1
        for rid in rids:
            if self._states.pop(str(rid), None) is not None:
                self.stats.cache_evictions += 1
        evicted = 0
        rid_set = {str(rid) for rid in rids}
        for key in list(self._proposal_cache.keys()):
            if key.rid in rid_set:
                self._proposal_cache.pop(key, None)
                evicted += 1
        self.stats.proposal_cache_evictions += evicted

    def clear(self) -> None:
        evicted = len(self._states)
        self._states.clear()
        self.stats.cache_evictions += evicted
        proposal_evicted = len(self._proposal_cache)
        self._proposal_cache.clear()
        self.stats.proposal_cache_evictions += proposal_evicted
        if self.native_backend is not None:
            self.native_backend.clear()

    def cache_size(self) -> int:
        if self.native_backend is not None:
            cache_size = getattr(self.native_backend, "cache_size", None)
            if callable(cache_size):
                return int(cache_size())
        return len(self._states)

    def proposal_cache_size(self) -> int:
        return len(self._proposal_cache)

    def _propose_itl(
        self,
        rid: str,
        current_text: str,
        current_target_ids: Sequence[int],
        *,
        context_ids: Sequence[int] | None = None,
        max_target_tokens: int,
    ) -> BaseProposal:
        import torch

        state, cache_event = self._ensure_state(rid, current_text, context_ids=context_ids)
        max_draft_tokens = self.config.max_draft_tokens
        if max_draft_tokens is None:
            max_draft_tokens = max(max_target_tokens * 4, max_target_tokens + 4)

        draft_ids: list[int] = []
        proxy_ids: list[int] = []
        generation_ids = list(state.input_ids)
        generation_past = self._fork_past_key_values(state.past_key_values)
        logits = state.next_token_logits
        root_logits = logits
        context_len = len(state.input_ids)

        try:
            with torch.inference_mode():
                for _ in range(max_draft_tokens):
                    next_token = int(torch.argmax(logits, dim=-1)[0])
                    draft_ids.append(next_token)

                    proxy_ids = self._bridge_assistant_tokens_to_target(
                        current_target_ids=current_target_ids,
                        assistant_context_ids=state.input_ids,
                        assistant_new_ids=draft_ids,
                    )
                    if len(proxy_ids) >= max_target_tokens:
                        break

                    generation_ids.append(next_token)
                    context_len += 1
                    logits, generation_past = self._forward_one(
                        token_id=next_token,
                        full_ids=generation_ids,
                        context_len=context_len,
                        past_key_values=generation_past,
                    )
                    eos_token_id = getattr(self.draft_tokenizer, "eos_token_id", None)
                    if eos_token_id is not None and next_token == int(eos_token_id):
                        break
        finally:
            self._rollback_past_key_values(generation_past)

        proxy_ids = proxy_ids[:max_target_tokens]
        tree_token_ids, tree_parent_indices = self._tree_from_greedy_with_root_siblings(
            root_logits=root_logits,
            greedy_target_ids=proxy_ids,
            current_target_ids=current_target_ids,
            assistant_context_ids=state.input_ids,
            max_target_tokens=max_target_tokens,
        )
        return BaseProposal(
            method="itl",
            draft_token_ids=tuple(draft_ids),
            target_token_ids=tuple(int(token_id) for token_id in proxy_ids),
            draft_prob_rows=None,
            cache_event=cache_event,
            draft_context_tokens=len(state.input_ids),
            alignment_cost=self._alignment_cost(draft_ids, proxy_ids),
            tokenizer_bridge=self.config.tokenizer_bridge,
            target_tree_token_ids=tree_token_ids,
            target_tree_parent_indices=tree_parent_indices,
        )

    def _propose_slem(
        self,
        rid: str,
        current_text: str,
        current_target_ids: Sequence[int],
        *,
        context_ids: Sequence[int] | None = None,
        max_target_tokens: int,
    ) -> BaseProposal:
        import torch

        state, cache_event = self._ensure_state(rid, current_text, context_ids=context_ids)
        max_draft_tokens = self.config.max_draft_tokens
        if max_draft_tokens is None:
            max_draft_tokens = max(max_target_tokens * 4, max_target_tokens + 4)

        draft_ids: list[int] = []
        proxy_ids: tuple[int, ...] = ()
        generation_ids = list(state.input_ids)
        generation_past = self._fork_past_key_values(state.past_key_values)
        logits = state.next_token_logits
        root_logits = logits
        context_len = len(state.input_ids)

        try:
            with torch.inference_mode():
                for _ in range(max_draft_tokens):
                    next_token = int(torch.argmax(logits, dim=-1)[0])
                    draft_ids.append(next_token)
                    proxy_ids = slem_target_proxies_from_assistant_window(
                        target_tokenizer=self.target_tokenizer,
                        assistant_tokenizer=self.draft_tokenizer,
                        current_target_ids=current_target_ids,
                        assistant_context_ids=state.input_ids,
                        assistant_new_ids=draft_ids,
                        assistant_lookbehind=self.config.assistant_lookbehind,
                        target_lookbehind=self.config.target_lookbehind,
                        add_special_tokens=self.config.add_special_tokens,
                    )
                    if len(proxy_ids) >= max_target_tokens:
                        break

                    generation_ids.append(next_token)
                    context_len += 1
                    logits, generation_past = self._forward_one(
                        token_id=next_token,
                        full_ids=generation_ids,
                        context_len=context_len,
                        past_key_values=generation_past,
                    )
                    eos_token_id = getattr(self.draft_tokenizer, "eos_token_id", None)
                    if eos_token_id is not None and next_token == int(eos_token_id):
                        break
        finally:
            self._rollback_past_key_values(generation_past)

        target_token_ids = tuple(int(token_id) for token_id in proxy_ids[:max_target_tokens])
        tree_token_ids, tree_parent_indices = self._tree_from_greedy_with_root_siblings(
            root_logits=root_logits,
            greedy_target_ids=target_token_ids,
            current_target_ids=current_target_ids,
            assistant_context_ids=state.input_ids,
            max_target_tokens=max_target_tokens,
        )
        return BaseProposal(
            method="itl-base-slem",
            draft_token_ids=tuple(draft_ids),
            target_token_ids=target_token_ids,
            draft_prob_rows=None,
            cache_event=cache_event,
            draft_context_tokens=len(state.input_ids),
            tokenizer_bridge="uag",
            target_tree_token_ids=tree_token_ids,
            target_tree_parent_indices=tree_parent_indices,
        )

    def _propose_tli(
        self,
        rid: str,
        current_text: str,
        *,
        context_ids: Sequence[int] | None = None,
        max_target_tokens: int,
        sampling: SamplingRequest,
    ) -> BaseProposal:
        import torch

        state, cache_event = self._ensure_state(rid, current_text, context_ids=context_ids)
        self._ensure_intersection_tensors()
        assert self._valid_assistant_ids is not None
        assert self._valid_target_ids is not None

        max_draft_tokens = self.config.max_draft_tokens or max_target_tokens
        max_draft_tokens = min(max_draft_tokens, max_target_tokens)
        draft_ids: list[int] = []
        target_ids: list[int] = []
        prob_rows: list[object] = []

        generation_ids = list(state.input_ids)
        generation_past = self._fork_past_key_values(state.past_key_values)
        logits = state.next_token_logits
        context_len = len(state.input_ids)
        generator = self._tli_sampling_generator(
            rid=rid,
            context_ids=state.input_ids,
            max_draft_tokens=max_draft_tokens,
            sampling=sampling,
            device=logits.device,
        )

        try:
            with torch.inference_mode():
                for _ in range(max_draft_tokens):
                    target_probs, selected_assistant_id, selected_target_id = (
                        self._sample_tli_token(
                            logits,
                            sampling=sampling,
                            generator=generator,
                        )
                    )
                    prob_rows.append(target_probs)
                    draft_ids.append(selected_assistant_id)
                    target_ids.append(selected_target_id)

                    generation_ids.append(selected_assistant_id)
                    context_len += 1
                    logits, generation_past = self._forward_one(
                        token_id=selected_assistant_id,
                        full_ids=generation_ids,
                        context_len=context_len,
                        past_key_values=generation_past,
                    )
                    eos_token_id = getattr(self.draft_tokenizer, "eos_token_id", None)
                    if (
                        eos_token_id is not None
                        and selected_assistant_id == int(eos_token_id)
                    ):
                        break

                # The final slot is the proposal distribution after the last draft
                # token. SGLang's tree verifier uses it for the target-only bonus
                # position and requires a row-aligned probability tensor.
                if prob_rows:
                    target_probs, _, _ = self._sample_tli_token(
                        logits,
                        sampling=sampling,
                        generator=generator,
                        sample_token=False,
                    )
                    prob_rows.append(target_probs)
        finally:
            self._rollback_past_key_values(generation_past)

        return BaseProposal(
            method="itl-base-tli",
            draft_token_ids=tuple(draft_ids),
            target_token_ids=tuple(target_ids),
            draft_prob_rows=prob_rows,
            cache_event=cache_event,
            draft_context_tokens=len(state.input_ids),
            tokenizer_bridge="tli",
            target_tree_token_ids=tuple(target_ids),
            target_tree_parent_indices=_linear_tree_parent_indices(len(target_ids)),
        )

    def _sample_tli_token(
        self,
        logits,
        *,
        sampling: SamplingRequest,
        generator=None,
        sample_token: bool = True,
    ):
        import torch

        assert self._valid_assistant_ids is not None
        assert self._valid_target_ids is not None

        valid_assistant_ids = self._valid_assistant_ids
        valid_target_ids = self._valid_target_ids
        if valid_assistant_ids.device != logits.device:
            valid_assistant_ids = valid_assistant_ids.to(logits.device, non_blocking=True)
        if valid_target_ids.device != logits.device:
            valid_target_ids = valid_target_ids.to(logits.device, non_blocking=True)

        valid_logits = logits[0, valid_assistant_ids].float()
        temperature = max(float(sampling.temperature), 1e-5)
        scaled = valid_logits / temperature
        probs = torch.softmax(scaled, dim=-1)
        probs = _renormalize_top_k_top_p(probs, top_k=sampling.top_k, top_p=sampling.top_p)

        if not sample_token or sampling.temperature <= 0:
            index = int(torch.argmax(probs).item())
        else:
            index = int(
                torch.multinomial(probs, num_samples=1, generator=generator).item()
            )

        target_probs = torch.zeros(
            (self.target_vocab_size,),
            dtype=torch.float32,
            device=logits.device,
        )
        target_probs.scatter_(0, valid_target_ids, probs)
        return (
            target_probs,
            int(valid_assistant_ids[index].item()),
            int(valid_target_ids[index].item()),
        )

    def _tli_sampling_generator(
        self,
        *,
        rid: str,
        context_ids: Sequence[int],
        max_draft_tokens: int,
        sampling: SamplingRequest,
        device,
    ):
        import torch

        seed = _tli_sampling_seed(
            rid=rid,
            context_ids=context_ids,
            max_draft_tokens=max_draft_tokens,
            sampling=sampling,
        )
        try:
            generator = torch.Generator(device=device)
        except TypeError:
            generator = torch.Generator()
        generator.manual_seed(seed)
        return generator

    def _ensure_intersection_tensors(self) -> None:
        import torch

        if self.intersection is None:
            self.intersection = VocabIntersection.from_tokenizers(
                target_tokenizer=self.target_tokenizer,
                assistant_tokenizer=self.draft_tokenizer,
                target_vocab_size=self.target_vocab_size,
            )
            self.intersection.require_non_empty()
            if len(self.intersection.assistant_ids) < self.config.tli_min_intersection:
                raise ValueError(
                    "Token-level intersection is too small: "
                    f"{len(self.intersection.assistant_ids)} < "
                    f"{self.config.tli_min_intersection}."
                )
            logger.info(
                "Initialized TLI vocabulary intersection: assistant=%s target=%s common=%s",
                self.intersection.assistant_vocab_size,
                self.intersection.target_vocab_size,
                len(self.intersection.assistant_ids),
            )
            self.stats.tli_intersection_size = len(self.intersection.assistant_ids)

        if self._valid_assistant_ids is None or self._valid_target_ids is None:
            device = self._input_device()
            self._valid_assistant_ids = torch.tensor(
                self.intersection.assistant_ids,
                dtype=torch.long,
                device=device,
            )
            self._valid_target_ids = torch.tensor(
                self.intersection.target_ids,
                dtype=torch.long,
                device=device,
            )

    def _ensure_state(
        self,
        rid: str,
        current_text: str,
        *,
        context_ids: Sequence[int] | None = None,
    ) -> tuple[DraftRequestState, str]:
        import torch

        rid = str(rid)
        context_ids = tuple(
            int(token_id)
            for token_id in (
                self._context_ids(current_text) if context_ids is None else context_ids
            )
        )
        if not context_ids:
            raise ValueError("draft context must contain at least one token.")

        if self.native_backend is not None:
            ensure_session = getattr(self.native_backend, "ensure_session", None)
            if callable(ensure_session):
                session, cache_event = ensure_session(context_ids, rid=rid)
            else:
                session = self.native_backend.prefill(context_ids, rid=rid)
                cache_event = "sglang-rebuild"
            if cache_event.endswith("hit"):
                self.stats.cache_hits += 1
            elif cache_event.endswith("extend"):
                self.stats.cache_extensions += 1
            else:
                self.stats.cache_rebuilds += 1
            return (
                DraftRequestState(
                    rid=rid,
                    text=current_text,
                    input_ids=tuple(getattr(session, "accepted_input_ids", context_ids)),
                    past_key_values=session,
                    next_token_logits=session.next_token_logits,
                ),
                cache_event,
            )

        cached = self._states.get(rid) if self.config.enable_draft_cache else None
        if cached is not None and cached.input_ids == context_ids:
            self._states.move_to_end(rid)
            self.stats.cache_hits += 1
            return cached, "hit"

        if (
            cached is not None
            and cached.past_key_values is not None
            and len(context_ids) > len(cached.input_ids)
            and context_ids[: len(cached.input_ids)] == cached.input_ids
        ):
            suffix = context_ids[len(cached.input_ids) :]
            past_key_values = self._fork_past_key_values(cached.past_key_values)
            suffix_tensor = torch.tensor(
                [list(suffix)],
                dtype=torch.long,
                device=self._input_device(),
            )
            attention_mask = torch.ones(
                (1, len(context_ids)),
                dtype=torch.long,
                device=suffix_tensor.device,
            )
            with torch.inference_mode():
                outputs = self.draft_model(
                    input_ids=suffix_tensor,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            state = DraftRequestState(
                rid=rid,
                text=current_text,
                input_ids=context_ids,
                past_key_values=getattr(outputs, "past_key_values", None),
                next_token_logits=outputs.logits[:, -1, :],
            )
            self._store_state(state)
            self.stats.cache_extensions += 1
            return state, "extend"

        input_tensor = torch.tensor(
            [list(context_ids)],
            dtype=torch.long,
            device=self._input_device(),
        )
        attention_mask = torch.ones_like(input_tensor)
        with torch.inference_mode():
            outputs = self.draft_model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                use_cache=True,
            )
        state = DraftRequestState(
            rid=rid,
            text=current_text,
            input_ids=context_ids,
            past_key_values=getattr(outputs, "past_key_values", None),
            next_token_logits=outputs.logits[:, -1, :],
        )
        self._store_state(state)
        self.stats.cache_rebuilds += 1
        return state, "rebuild"

    def _forward_one(
        self,
        *,
        token_id: int,
        full_ids: Sequence[int],
        context_len: int,
        past_key_values: object,
    ):
        import torch

        if self.native_backend is not None:
            decode = getattr(past_key_values, "decode", None)
            if not callable(decode):
                raise ValueError("SGLang-native draft session is missing decode().")
            return decode(token_id), past_key_values

        if past_key_values is None:
            input_tensor = torch.tensor(
                [list(full_ids)],
                dtype=torch.long,
                device=self._input_device(),
            )
            attention_mask = torch.ones_like(input_tensor)
        else:
            input_tensor = torch.tensor(
                [[int(token_id)]],
                dtype=torch.long,
                device=self._input_device(),
            )
            attention_mask = torch.ones(
                (1, context_len),
                dtype=torch.long,
                device=input_tensor.device,
            )
        outputs = self.draft_model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        return outputs.logits[:, -1, :], getattr(outputs, "past_key_values", None)

    def _store_state(self, state: DraftRequestState) -> None:
        if self.native_backend is not None:
            return
        if not self.config.enable_draft_cache:
            return
        self._states[state.rid] = state
        self._states.move_to_end(state.rid)
        while len(self._states) > self.config.max_cached_requests:
            self._states.popitem(last=False)
            self.stats.cache_evictions += 1

    def _proposal_cache_key(
        self,
        *,
        rid: str,
        method: str,
        current_target_ids: Sequence[int],
        context_ids: Sequence[int],
        max_target_tokens: int,
        sampling: SamplingRequest,
    ) -> ProposalCacheKey | None:
        if not getattr(self.config, "enable_proposal_cache", True):
            return None
        # TLI proposals carry target-probability rows and may sample from the
        # intersection distribution. Keep them out of the proposal cache until
        # the verifier path has copy-on-read probability tensors.
        if method not in {"itl", "itl-base-slem"}:
            return None
        target_ids = tuple(int(token_id) for token_id in current_target_ids)
        draft_context_ids = tuple(int(token_id) for token_id in context_ids)
        return ProposalCacheKey(
            rid=str(rid),
            method=method,
            target_hash=_hash_ints(target_ids),
            target_len=len(target_ids),
            draft_context_hash=_hash_ints(draft_context_ids),
            draft_context_len=len(draft_context_ids),
            max_target_tokens=int(max_target_tokens),
            temperature=_normalize_float(sampling.temperature),
            top_k=int(sampling.top_k),
            top_p=_normalize_float(sampling.top_p),
            add_special_tokens=bool(self.config.add_special_tokens),
            assistant_lookbehind=int(self.config.assistant_lookbehind),
            target_lookbehind=int(self.config.target_lookbehind),
            tokenizer_bridge=str(self.config.tokenizer_bridge),
            tree_branch_factor=int(getattr(self.config, "tree_branch_factor", 1) or 1),
            tree_max_depth=getattr(self.config, "tree_max_depth", None),
        )

    def _store_proposal_cache(
        self,
        cache_key: ProposalCacheKey | None,
        proposal: BaseProposal,
    ) -> None:
        if cache_key is None or not proposal.target_token_ids:
            return
        self._proposal_cache[cache_key] = proposal
        self._proposal_cache.move_to_end(cache_key)
        self.stats.proposal_cache_stores += 1
        limit = int(getattr(self.config, "max_cached_proposals", 1024) or 0)
        if limit <= 0:
            self._proposal_cache.clear()
            return
        while len(self._proposal_cache) > limit:
            self._proposal_cache.popitem(last=False)
            self.stats.proposal_cache_evictions += 1

    def _bridge_assistant_tokens_to_target(
        self,
        *,
        current_target_ids: Sequence[int],
        assistant_context_ids: Sequence[int],
        assistant_new_ids: Sequence[int],
    ) -> tuple[int, ...]:
        if self.config.tokenizer_bridge == "segment":
            self.stats.bridge_segment_translations += 1
            target_ids = self._encode(
                self.target_tokenizer,
                self._decode(self.draft_tokenizer, assistant_new_ids),
            )
        else:
            self.stats.bridge_uag_translations += 1
            target_ids = slem_target_proxies_from_assistant_window(
                target_tokenizer=self.target_tokenizer,
                assistant_tokenizer=self.draft_tokenizer,
                current_target_ids=current_target_ids,
                assistant_context_ids=assistant_context_ids,
                assistant_new_ids=assistant_new_ids,
                assistant_lookbehind=self.config.assistant_lookbehind,
                target_lookbehind=self.config.target_lookbehind,
                add_special_tokens=self.config.add_special_tokens,
            )
        if not target_ids:
            self.stats.bridge_empty_translations += 1
        return tuple(int(token_id) for token_id in target_ids)

    def _tree_from_greedy_with_root_siblings(
        self,
        *,
        root_logits,
        greedy_target_ids: Sequence[int],
        current_target_ids: Sequence[int],
        assistant_context_ids: Sequence[int],
        max_target_tokens: int,
    ) -> tuple[tuple[int, ...], tuple[int, ...]]:
        branch_factor = int(getattr(self.config, "tree_branch_factor", 1) or 1)
        max_depth = getattr(self.config, "tree_max_depth", None)
        greedy_budget = int(max_target_tokens)
        if branch_factor > 1 and max_target_tokens > 1:
            sibling_budget = min(branch_factor - 1, max_target_tokens - 1)
            greedy_budget = max(1, max_target_tokens - sibling_budget)
        if max_depth is not None:
            max_depth = max(1, int(max_depth))
            greedy_budget = min(greedy_budget, max_depth)

        greedy = tuple(int(token_id) for token_id in greedy_target_ids[:greedy_budget])
        tokens: list[int] = list(greedy)
        parents: list[int] = list(_linear_tree_parent_indices(len(greedy)))

        if branch_factor <= 1 or max_target_tokens <= 1:
            return tuple(tokens), tuple(parents)

        budget = max(0, max_target_tokens - len(tokens))
        if budget <= 0:
            return tuple(tokens), tuple(parents)

        for assistant_id in self._topk_assistant_ids(root_logits, branch_factor):
            if not budget:
                break
            branch = self._bridge_assistant_tokens_to_target(
                current_target_ids=current_target_ids,
                assistant_context_ids=assistant_context_ids,
                assistant_new_ids=(assistant_id,),
            )
            if not branch:
                continue
            if greedy and int(branch[0]) == int(greedy[0]):
                continue

            parent_index = 0
            for token_id in branch:
                if not budget:
                    break
                tokens.append(int(token_id))
                parents.append(parent_index)
                parent_index = len(tokens)
                budget -= 1

        if len(tokens) > len(greedy):
            self.stats.tree_proposals += 1
            self.stats.tree_nodes += len(tokens)
        return tuple(tokens), tuple(parents)

    @staticmethod
    def _topk_assistant_ids(logits, k: int) -> tuple[int, ...]:
        import torch

        if k <= 0:
            return ()
        values = logits[0]
        k = min(int(k), int(values.numel()))
        if k <= 0:
            return ()
        return tuple(int(token_id) for token_id in torch.topk(values, k=k).indices.tolist())

    def _count_method_proposal(self, method: str) -> None:
        if method == "itl":
            self.stats.itl_proposals += 1
        elif method == "itl-base-slem":
            self.stats.slem_proposals += 1
        elif method == "itl-base-tli":
            self.stats.tli_proposals += 1

    def _record_proposal(self, proposal: BaseProposal) -> None:
        if not proposal.target_token_ids:
            self.stats.empty_proposals += 1
        self.stats.proposed_target_tokens += len(proposal.target_token_ids)
        self.stats.proposed_draft_tokens += len(proposal.draft_token_ids)

    def _context_ids(self, text: str) -> tuple[int, ...]:
        token_ids = encode_text(
            self.draft_tokenizer,
            text,
            add_special_tokens=self.config.add_special_tokens,
        )
        if self.config.max_context_tokens is not None:
            token_ids = token_ids[-self.config.max_context_tokens :]
        return token_ids

    def _input_device(self):
        if self.native_backend is not None:
            return self.native_backend.device
        try:
            return self.draft_model.device
        except AttributeError:
            return next(self.draft_model.parameters()).device

    def _fork_past_key_values(self, past_key_values):
        if past_key_values is None:
            return None
        if self.native_backend is not None:
            begin_speculative = getattr(past_key_values, "begin_speculative", None)
            if callable(begin_speculative):
                return begin_speculative()
            return past_key_values
        if not self.config.clone_draft_cache:
            return past_key_values
        return _clone_cache(past_key_values)

    def _rollback_past_key_values(self, past_key_values) -> None:
        if self.native_backend is None or past_key_values is None:
            return
        rollback = getattr(past_key_values, "rollback_speculative", None)
        if callable(rollback):
            try:
                rollback()
            except Exception as exc:
                disable_kv_cache = getattr(self.native_backend, "disable_kv_cache", None)
                if callable(disable_kv_cache):
                    disable_kv_cache(f"speculative rollback failed: {exc}")
                else:
                    clear = getattr(self.native_backend, "clear", None)
                    if callable(clear):
                        clear()
                logger.warning(
                    "SGLANG_GROUP disabled native draft KV reuse after rollback "
                    "failure; continuing with safe rebuild for future proposals.",
                    exc_info=True,
                )

    def _alignment_cost(
        self,
        draft_ids: Sequence[int],
        proxy_ids: Sequence[int],
    ) -> float | None:
        if not draft_ids or not proxy_ids:
            return None
        try:
            draft_strings = tuple(
                self._decode(self.draft_tokenizer, [token_id])
                for token_id in draft_ids
            )
            proxy_strings = tuple(
                self._decode(self.target_tokenizer, [token_id])
                for token_id in proxy_ids
            )
            alignment = dynamic_token_warping(
                draft_strings,
                proxy_strings,
                window=self.config.dtw_window,
            )
            return alignment.total_cost
        except Exception:
            return None

    @staticmethod
    def _encode(
        tokenizer: object,
        text: str,
        *,
        add_special_tokens: bool = False,
    ) -> list[int]:
        try:
            return list(tokenizer.encode(text, add_special_tokens=add_special_tokens))
        except TypeError:
            encoded = tokenizer(text, add_special_tokens=add_special_tokens)
            input_ids = encoded["input_ids"] if isinstance(encoded, dict) else encoded.input_ids
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return list(input_ids)

    @staticmethod
    def _decode(tokenizer: object, token_ids: Sequence[int]) -> str:
        ids = [int(token_id) for token_id in token_ids]
        try:
            return tokenizer.decode(
                ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
        except TypeError:
            return tokenizer.decode(ids)

    def debug_decode_target(self, token_ids: Sequence[int]) -> str:
        return decode_ids(self.target_tokenizer, token_ids)


def _clone_cache(value):
    return _clone_cache_for_reuse(value)


def _hash_ints(token_ids: Sequence[int]) -> str:
    digest = hashlib.blake2b(digest_size=16)
    for token_id in token_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    return digest.hexdigest()


def _tli_sampling_seed(
    *,
    rid: str,
    context_ids: Sequence[int],
    max_draft_tokens: int,
    sampling: SamplingRequest,
) -> int:
    """Return a stable proposal RNG seed shared by all TP ranks.

    Replicated draft runners must emit identical candidate tokens on every
    target TP rank. Python's hash() is process-randomized, so derive the seed
    from explicit request fields instead.
    """

    digest = hashlib.blake2b(digest_size=8)
    digest.update(str(rid).encode("utf-8", errors="surrogatepass"))
    digest.update(b"\0")
    for token_id in context_ids:
        digest.update(int(token_id).to_bytes(8, "little", signed=True))
    digest.update(int(max_draft_tokens).to_bytes(8, "little", signed=True))
    digest.update(repr(_normalize_float(sampling.temperature)).encode("ascii"))
    digest.update(int(sampling.top_k).to_bytes(8, "little", signed=True))
    digest.update(repr(_normalize_float(sampling.top_p)).encode("ascii"))
    return int.from_bytes(digest.digest(), "little") & ((1 << 63) - 1)


def _linear_tree_parent_indices(length: int) -> tuple[int, ...]:
    return tuple(range(max(0, int(length))))


def _normalize_float(value: float) -> float:
    return round(float(value), 8)


def _clone_nested_tensors(value):
    import torch

    if torch.is_tensor(value):
        return value.clone()
    if isinstance(value, tuple):
        return tuple(_clone_nested_tensors(item) for item in value)
    if isinstance(value, list):
        return [_clone_nested_tensors(item) for item in value]
    if isinstance(value, dict):
        return {key: _clone_nested_tensors(item) for key, item in value.items()}
    clone = getattr(value, "clone", None)
    if callable(clone):
        return clone()
    return value


def _clone_cache_for_reuse(value):
    try:
        return copy.deepcopy(value)
    except Exception:
        pass

    to_legacy_cache = getattr(value, "to_legacy_cache", None)
    from_legacy_cache = getattr(type(value), "from_legacy_cache", None)
    if callable(to_legacy_cache) and callable(from_legacy_cache):
        legacy_cache = _clone_nested_tensors(to_legacy_cache())
        try:
            return from_legacy_cache(legacy_cache)
        except Exception:
            logger.debug(
                "Could not clone HF cache object from legacy cache.",
                exc_info=True,
            )

    return _clone_nested_tensors(value)


def _torch_dtype(torch, dtype_name: str):
    normalized = dtype_name.lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported SGLANG_GROUP_DRAFT_DTYPE: {dtype_name}") from exc


def _renormalize_top_k_top_p(probs, *, top_k: int, top_p: float):
    import torch

    probs = probs.float()
    if top_k is not None and int(top_k) > 0 and int(top_k) < probs.numel():
        top_values, top_indices = torch.topk(probs, int(top_k))
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, top_indices, top_values)
        probs = filtered

    if top_p is not None and 0 < float(top_p) < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        keep = cumulative <= float(top_p)
        if keep.numel() > 0:
            keep[0] = True
        filtered = torch.zeros_like(probs)
        filtered.scatter_(0, sorted_indices[keep], sorted_probs[keep])
        probs = filtered

    total = probs.sum()
    if not torch.isfinite(total) or total <= 0:
        return torch.full_like(probs, 1.0 / probs.numel())
    return probs / total
