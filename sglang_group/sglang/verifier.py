"""SGLANG_GROUP verifier fast paths."""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LinearGreedyVerifyResult:
    predict: object
    accepted_indices: object
    accept_length: object
    backend: str


def tree_greedy_verify(
    *,
    next_token_logits,
    draft_token,
    draft_token_num: int,
    retrieve_index=None,
    retrieve_next_token=None,
    retrieve_next_sibling=None,
    backend: str = "auto",
) -> LinearGreedyVerifyResult:
    """Verify a SGLANG_GROUP tree without the generic upstream verifier.

    The accepted index chain contains the current node index at each accepted
    depth. The token appended at each step is ``target_predict[current_node]``.
    If that token matches a child candidate, the verifier advances to the child;
    otherwise it stops after appending the target bonus token.
    """

    import torch

    normalized = (backend or "auto").lower()
    width = int(draft_token_num)
    if width <= 0:
        raise ValueError("draft_token_num must be positive.")

    flat_predict = torch.argmax(next_token_logits, dim=-1).to(torch.int32)
    total = int(flat_predict.numel())
    if total % width != 0:
        raise ValueError(
            "next_token_logits first dimension must be divisible by draft_token_num: "
            f"{total} % {width}"
        )
    batch_size = total // width
    predict = torch.empty((total + 1,), dtype=torch.int32, device=flat_predict.device)
    predict[:total].copy_(flat_predict)
    predict[total] = -1

    candidates = draft_token.reshape(batch_size, width).to(torch.int32)
    target_predict = flat_predict.reshape(batch_size, width)
    if retrieve_next_token is None or retrieve_next_sibling is None:
        return linear_greedy_verify(
            next_token_logits=next_token_logits,
            draft_token=draft_token,
            draft_token_num=draft_token_num,
            backend=backend,
        )

    next_token = retrieve_next_token.to(device=flat_predict.device, dtype=torch.int32)
    next_sibling = retrieve_next_sibling.to(device=flat_predict.device, dtype=torch.int32)
    if retrieve_index is None:
        retrieve_index = torch.arange(
            batch_size * width,
            dtype=torch.int64,
            device=flat_predict.device,
        ).reshape(batch_size, width)
    if normalized in {"auto", "triton"} and getattr(flat_predict, "is_cuda", False):
        try:
            accepted_indices, accept_length = _cuda_tree_accept(
                predict,
                target_predict,
                candidates,
                retrieve_index,
                next_token,
                next_sibling,
            )
            return LinearGreedyVerifyResult(
                predict=predict,
                accepted_indices=accepted_indices,
                accept_length=accept_length,
                backend="sgl-kernel-tree",
            )
        except Exception:
            if normalized == "triton":
                raise
            logger.debug("CUDA tree verifier failed; falling back to torch.", exc_info=True)

    accepted_indices, accept_length = _torch_tree_accept(
        target_predict,
        candidates,
        next_token,
        next_sibling,
    )
    return LinearGreedyVerifyResult(
        predict=predict,
        accepted_indices=accepted_indices,
        accept_length=accept_length,
        backend="torch",
    )


def linear_greedy_verify(
    *,
    next_token_logits,
    draft_token,
    draft_token_num: int,
    backend: str = "auto",
) -> LinearGreedyVerifyResult:
    """Verify a linear row without the generic tree verifier."""

    import torch

    normalized = (backend or "auto").lower()
    width = int(draft_token_num)
    if width <= 0:
        raise ValueError("draft_token_num must be positive.")

    flat_predict = torch.argmax(next_token_logits, dim=-1).to(torch.int32)
    total = int(flat_predict.numel())
    if total % width != 0:
        raise ValueError(
            "next_token_logits first dimension must be divisible by draft_token_num: "
            f"{total} % {width}"
        )
    batch_size = total // width
    predict = torch.empty((total + 1,), dtype=torch.int32, device=flat_predict.device)
    predict[:total].copy_(flat_predict)
    predict[total] = -1
    candidates = draft_token.reshape(batch_size, width).to(torch.int32)
    target_predict = flat_predict.reshape(batch_size, width)
    if normalized in {"auto", "triton"} and _can_use_triton(flat_predict):
        try:
            accepted_indices, accept_length = _triton_linear_accept(target_predict, candidates)
            return LinearGreedyVerifyResult(predict, accepted_indices, accept_length, "triton")
        except Exception:
            if normalized == "triton":
                raise
            logger.debug("Triton linear verifier failed; falling back to torch.", exc_info=True)

    accepted_indices, accept_length = _torch_linear_accept(target_predict, candidates)
    return LinearGreedyVerifyResult(predict, accepted_indices, accept_length, "torch")


def _torch_linear_accept(target_predict, candidates):
    import torch

    batch_size, width = target_predict.shape
    device = target_predict.device
    if width == 1:
        accept_length = torch.zeros((batch_size,), dtype=torch.int32, device=device)
    else:
        matches = target_predict[:, :-1] == candidates[:, 1:]
        prefix_matches = matches.to(torch.int32).cumprod(dim=1)
        accept_length = prefix_matches.sum(dim=1).to(torch.int32)

    offsets = torch.arange(width, dtype=torch.int32, device=device)
    base = torch.arange(batch_size, dtype=torch.int32, device=device).unsqueeze(1) * width
    keep = offsets.unsqueeze(0) <= accept_length.unsqueeze(1)
    accepted_indices = torch.where(keep, base + offsets.unsqueeze(0), -1)
    return accepted_indices.to(torch.int32), accept_length


def _torch_tree_accept(target_predict, candidates, next_token, next_sibling):
    import torch

    batch_size, width = target_predict.shape
    device = target_predict.device
    accepted_indices = torch.full((batch_size, width), -1, dtype=torch.int32, device=device)
    accept_length = torch.zeros((batch_size,), dtype=torch.int32, device=device)

    for row in range(batch_size):
        current = 0
        step = 0
        accepted_indices[row, step] = row * width + current
        while step + 1 < width:
            target = int(target_predict[row, current].item())
            child = int(next_token[row, current].item())
            matched = -1
            while child != -1:
                if int(candidates[row, child].item()) == target:
                    matched = child
                    break
                child = int(next_sibling[row, child].item())
            if matched == -1:
                break
            current = matched
            step += 1
            accepted_indices[row, step] = row * width + current
        accept_length[row] = step

    return accepted_indices, accept_length


def _can_use_triton(tensor) -> bool:
    if not getattr(tensor, "is_cuda", False):
        return False
    try:
        import triton  # noqa: F401
    except Exception:
        return False
    return True


def _triton_linear_accept(target_predict, candidates):
    import torch
    import triton

    batch_size, width = target_predict.shape
    accepted_indices = torch.empty(
        (batch_size, width),
        dtype=torch.int32,
        device=target_predict.device,
    )
    accept_length = torch.empty((batch_size,), dtype=torch.int32, device=target_predict.device)
    block = triton.next_power_of_2(width)
    _linear_accept_kernel[(batch_size,)](
        target_predict,
        candidates,
        accepted_indices,
        accept_length,
        width,
        BLOCK=block,
    )
    return accepted_indices, accept_length


def _cuda_tree_accept(
    predict,
    target_predict,
    candidates,
    retrieve_index,
    next_token,
    next_sibling,
):
    import torch

    from sgl_kernel import verify_tree_greedy

    batch_size, width = target_predict.shape
    accepted_indices = torch.full(
        (batch_size, width),
        -1,
        dtype=torch.int32,
        device=target_predict.device,
    )
    accept_length = torch.empty((batch_size,), dtype=torch.int32, device=target_predict.device)
    verify_tree_greedy(
        predicts=predict,
        accept_index=accepted_indices,
        accept_token_num=accept_length,
        candidates=candidates,
        retrive_index=retrieve_index.to(torch.int64),
        retrive_next_token=next_token.to(torch.int64),
        retrive_next_sibling=next_sibling.to(torch.int64),
        target_predict=target_predict,
    )
    return accepted_indices, accept_length


def _triton_tree_accept(*args, **kwargs):
    raise NotImplementedError("tree verifier uses sgl_kernel on CUDA and torch fallback.")


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _linear_accept_kernel(
        target_predict,
        candidates,
        accepted_indices,
        accept_length,
        width: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK)
        mask = offsets < width
        row_base = row * width

        preds = tl.load(target_predict + row_base + offsets, mask=mask, other=-2)
        next_candidates = tl.load(
            candidates + row_base + offsets + 1,
            mask=offsets < width - 1,
            other=-3,
        )
        failed_offsets = tl.where(
            (offsets < width - 1) & (preds != next_candidates),
            offsets,
            width - 1,
        )
        accepted = tl.min(failed_offsets, axis=0)
        tl.store(accept_length + row, accepted)

        global_indices = row_base + offsets
        keep = offsets <= accepted
        tl.store(
            accepted_indices + row_base + offsets,
            tl.where(keep, global_indices, -1),
            mask=mask,
        )

except Exception:
    _linear_accept_kernel = None
