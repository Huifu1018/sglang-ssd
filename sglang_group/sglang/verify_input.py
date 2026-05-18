"""SGLang verify input carrying TLI draft probabilities."""

from __future__ import annotations

from typing import Optional

from .verifier import tree_greedy_verify


class GroupGreedyVerifyInputMixin:
    """Use SGLANG_GROUP's linear/tree verifier for greedy target verification."""

    def __init__(self, *args, verify_backend: str = "auto", **kwargs):
        super().__init__(*args, **kwargs)
        self.sglang_group_verify_backend = verify_backend
        self.sglang_group_verify_backend_used = None

    def _greedy_verify(self, batch, logits_output):
        backend = getattr(self, "sglang_group_verify_backend", "auto")
        if backend == "sglang":
            self.sglang_group_verify_backend_used = "sglang"
            return super()._greedy_verify(batch, logits_output)

        result = tree_greedy_verify(
            next_token_logits=logits_output.next_token_logits,
            draft_token=self.draft_token,
            draft_token_num=self.draft_token_num,
            retrieve_index=_spec_attr(self, "retrieve_index", "retrive_index"),
            retrieve_next_token=_spec_attr(self, "retrieve_next_token", "retrive_next_token"),
            retrieve_next_sibling=_spec_attr(self, "retrieve_next_sibling", "retrive_next_sibling"),
            backend=backend,
        )
        self.predict = result.predict
        self.accepted_indices = result.accepted_indices
        self.accept_length = result.accept_length
        self.accept_indices = result.accepted_indices
        self.num_correct_drafts = result.accept_length
        self.sglang_group_verify_backend_used = result.backend


class TliVerifyInputMixin:
    """Mixin overriding SGLang's sampling verifier with real TLI draft probs."""

    def __init__(self, *args, draft_probs=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.draft_probs = draft_probs

    def _sampling_verify(self, batch, logits_output, sampling_info):
        import torch
        import torch.nn.functional as F

        from sglang.srt.server_args import get_global_server_args
        from sglang.srt.speculative.ngram_info import (
            top_k_renorm_prob,
            top_p_renorm_prob,
            tree_speculative_sampling_target_only,
        )

        bs = batch.batch_size()
        candidates = self.draft_token.reshape(bs, self.draft_token_num)
        predict_shape = list(logits_output.next_token_logits.shape)[:-1]
        predict_shape[-1] += 1
        self.predict = torch.empty(predict_shape, dtype=torch.int32, device=self.device)

        # SGLang 0.5.9 uses accepted_indices/accept_length; newer builds use
        # accept_indices/num_correct_drafts. Set both names when possible.
        accepted_indices = torch.full(
            (bs, self.draft_token_num), -1, dtype=torch.int32, device=self.device
        )
        accept_length = torch.empty((bs,), dtype=torch.int32, device=self.device)
        self.accepted_indices = accepted_indices
        self.accept_length = accept_length
        self.accept_indices = accepted_indices
        self.num_correct_drafts = accept_length

        expanded_temperature = torch.repeat_interleave(
            sampling_info.temperatures, self.draft_token_num, dim=0
        )
        target_probs = F.softmax(
            logits_output.next_token_logits / expanded_temperature,
            dim=-1,
        )
        target_probs = top_k_renorm_prob(
            target_probs,
            torch.repeat_interleave(sampling_info.top_ks, self.draft_token_num, dim=0),
        )
        if sampling_info.need_top_p_sampling:
            target_probs = top_p_renorm_prob(
                target_probs,
                torch.repeat_interleave(sampling_info.top_ps, self.draft_token_num, dim=0),
            )
        target_probs = target_probs.reshape(bs, self.draft_token_num, -1)

        if self.draft_probs is None:
            draft_probs = torch.zeros(
                target_probs.shape,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            draft_probs = self.draft_probs.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            if tuple(draft_probs.shape) != tuple(target_probs.shape):
                raise ValueError(
                    "TLI draft probability shape mismatch: "
                    f"{tuple(draft_probs.shape)} != {tuple(target_probs.shape)}"
                )

        coins = torch.rand_like(candidates, dtype=torch.float32, device=self.device)
        coins_for_final_sampling = torch.rand((bs,), dtype=torch.float32, device=self.device)

        retrieve_index = _spec_attr(self, "retrieve_index", "retrive_index")
        retrieve_next_token = _spec_attr(self, "retrieve_next_token", "retrive_next_token")
        retrieve_next_sibling = _spec_attr(self, "retrieve_next_sibling", "retrive_next_sibling")

        tree_speculative_sampling_target_only(
            predicts=self.predict,
            accept_index=accepted_indices,
            accept_token_num=accept_length,
            candidates=candidates.to(torch.int64),
            retrive_index=retrieve_index.to(torch.int64),
            retrive_next_token=retrieve_next_token.to(torch.int64),
            retrive_next_sibling=retrieve_next_sibling.to(torch.int64),
            uniform_samples=coins,
            uniform_samples_for_final_sampling=coins_for_final_sampling,
            target_probs=target_probs,
            draft_probs=draft_probs,
            threshold_single=get_global_server_args().speculative_accept_threshold_single,
            threshold_acc=get_global_server_args().speculative_accept_threshold_acc,
            deterministic=True,
        )


def _spec_attr(obj: object, modern_name: str, legacy_name: str):
    if hasattr(obj, modern_name):
        return getattr(obj, modern_name)
    return getattr(obj, legacy_name)


def make_group_verify_input(
    *,
    draft_token,
    tree_mask,
    positions,
    retrieve_index,
    retrieve_next_token,
    retrieve_next_sibling,
    draft_token_num: int,
    draft_probs=None,
    verify_backend: str = "auto",
    grammar: Optional[object] = None,
):
    """Create a SGLANG_GROUP verify input for the installed SGLang build."""

    from sglang.srt.speculative.ngram_info import NgramVerifyInput

    if draft_probs is None:
        class GroupVerifyInput(GroupGreedyVerifyInputMixin, NgramVerifyInput):
            pass

        verify_cls = GroupVerifyInput
    else:
        class GroupVerifyInput(TliVerifyInputMixin, GroupGreedyVerifyInputMixin, NgramVerifyInput):
            pass

        verify_cls = GroupVerifyInput

    kwargs = {"grammar": grammar, "verify_backend": verify_backend}
    if draft_probs is not None:
        kwargs["draft_probs"] = draft_probs
    return verify_cls(
        draft_token,
        tree_mask,
        positions,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        draft_token_num,
        **kwargs,
    )


def make_tli_verify_input(**kwargs):
    """Backward-compatible helper for tests/importers."""

    return make_group_verify_input(**kwargs)
