"""Candidate-row helpers for SGLang spec-v1 verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class CandidateRows:
    rows: tuple[tuple[int, ...], ...]
    draft_token_num: int
    proposed_target_tokens: int
    parent_rows: tuple[tuple[int, ...], ...] = ()
    depth_rows: tuple[tuple[int, ...], ...] = ()
    draft_prob_rows: tuple[object | None, ...] = ()
    proposal_cache_events: tuple[str, ...] = ()
    draft_cache_events: tuple[str, ...] = ()
    proposal_methods: tuple[str, ...] = ()

    @property
    def is_tree(self) -> bool:
        return bool(self.parent_rows)


def build_linear_candidate_rows(
    roots: Sequence[int],
    target_rows: Sequence[Sequence[int]],
    *,
    max_draft_token_num: int,
    draft_prob_rows: Sequence[object | None] | None = None,
    proposal_cache_events: Sequence[str] | None = None,
    draft_cache_events: Sequence[str] | None = None,
    proposal_methods: Sequence[str] | None = None,
) -> CandidateRows:
    """Build equal-width linear verify rows.

    SGLang's target verifier expects a single row width for the whole batch.
    The width is therefore clipped to the shortest real candidate row in the
    batch; no fake pad/eos candidates are introduced.
    """

    if max_draft_token_num <= 0:
        raise ValueError("max_draft_token_num must be positive.")
    if len(roots) != len(target_rows):
        raise ValueError("roots and target_rows must have the same length.")
    if draft_prob_rows is not None and len(draft_prob_rows) != len(roots):
        raise ValueError("draft_prob_rows must have the same length as roots.")
    if proposal_cache_events is not None and len(proposal_cache_events) != len(roots):
        raise ValueError("proposal_cache_events must have the same length as roots.")
    if draft_cache_events is not None and len(draft_cache_events) != len(roots):
        raise ValueError("draft_cache_events must have the same length as roots.")
    if proposal_methods is not None and len(proposal_methods) != len(roots):
        raise ValueError("proposal_methods must have the same length as roots.")

    max_target_tokens = max(0, max_draft_token_num - 1)
    raw_rows: list[tuple[int, ...]] = []
    proposed_target_tokens = 0
    for root, targets in zip(roots, target_rows):
        clipped = tuple(int(token_id) for token_id in targets[:max_target_tokens])
        proposed_target_tokens += len(clipped)
        raw_rows.append((int(root), *clipped))

    if not raw_rows:
        return CandidateRows(rows=(), draft_token_num=1, proposed_target_tokens=0)

    draft_token_num = max(1, min(len(row) for row in raw_rows))
    draft_token_num = min(draft_token_num, max_draft_token_num)
    rows = tuple(row[:draft_token_num] for row in raw_rows)

    clipped_prob_rows: tuple[object | None, ...] = ()
    if draft_prob_rows is not None:
        clipped_prob_rows = tuple(draft_prob_rows)

    parent_rows = tuple(
        tuple(-1 if index == 0 else index - 1 for index in range(draft_token_num))
        for _ in rows
    )
    depth_rows = tuple(tuple(range(draft_token_num)) for _ in rows)

    return CandidateRows(
        rows=rows,
        draft_token_num=draft_token_num,
        proposed_target_tokens=proposed_target_tokens,
        parent_rows=parent_rows,
        depth_rows=depth_rows,
        draft_prob_rows=clipped_prob_rows,
        proposal_cache_events=tuple(proposal_cache_events or ()),
        draft_cache_events=tuple(draft_cache_events or ()),
        proposal_methods=tuple(proposal_methods or ()),
    )


def build_tree_candidate_rows(
    roots: Sequence[int],
    target_tree_rows: Sequence[Sequence[int]],
    parent_tree_rows: Sequence[Sequence[int]],
    *,
    max_draft_token_num: int,
    draft_prob_rows: Sequence[object | None] | None = None,
    proposal_cache_events: Sequence[str] | None = None,
    draft_cache_events: Sequence[str] | None = None,
    proposal_methods: Sequence[str] | None = None,
) -> CandidateRows:
    if len(target_tree_rows) != len(roots):
        raise ValueError("target_tree_rows must have the same length as roots.")
    if len(parent_tree_rows) != len(roots):
        raise ValueError("parent_tree_rows must have the same length as roots.")
    linear_targets = []
    for targets, parents in zip(target_tree_rows, parent_tree_rows):
        if len(targets) != len(parents):
            raise ValueError("each target tree row must match its parent row length.")
        linear_targets.append(tuple(targets))

    if max_draft_token_num <= 0:
        raise ValueError("max_draft_token_num must be positive.")

    raw_rows: list[tuple[int, ...]] = []
    raw_parents: list[tuple[int, ...]] = []
    proposed_target_tokens = 0
    max_target_nodes = max(0, max_draft_token_num - 1)
    for root, targets, parents in zip(roots, linear_targets, parent_tree_rows):
        clipped_targets = tuple(int(token_id) for token_id in targets[:max_target_nodes])
        clipped_parents = tuple(int(parent) for parent in parents[: len(clipped_targets)])
        proposed_target_tokens += len(clipped_targets)
        raw_rows.append((int(root), *clipped_targets))
        raw_parents.append((-1, *clipped_parents))

    if not raw_rows:
        return CandidateRows(rows=(), draft_token_num=1, proposed_target_tokens=0)

    draft_token_num = max(1, min(len(row) for row in raw_rows))
    draft_token_num = min(draft_token_num, max_draft_token_num)
    rows = tuple(row[:draft_token_num] for row in raw_rows)
    parent_rows = tuple(_clip_parent_row(row[:draft_token_num]) for row in raw_parents)
    depth_rows = tuple(_depth_row(parents) for parents in parent_rows)

    return CandidateRows(
        rows=rows,
        draft_token_num=draft_token_num,
        proposed_target_tokens=proposed_target_tokens,
        parent_rows=parent_rows,
        depth_rows=depth_rows,
        draft_prob_rows=tuple(draft_prob_rows or ()),
        proposal_cache_events=tuple(proposal_cache_events or ()),
        draft_cache_events=tuple(draft_cache_events or ()),
        proposal_methods=tuple(proposal_methods or ()),
    )


def _clip_parent_row(parent_row: Sequence[int]) -> tuple[int, ...]:
    width = len(parent_row)
    clipped = []
    for index, parent in enumerate(parent_row):
        if index == 0:
            clipped.append(-1)
        elif parent < 0 or parent >= width:
            clipped.append(index - 1)
        else:
            clipped.append(int(parent))
    return tuple(clipped)


def _depth_row(parent_row: Sequence[int]) -> tuple[int, ...]:
    depths: list[int] = []
    for index, parent in enumerate(parent_row):
        if index == 0 or parent < 0:
            depths.append(0)
        else:
            depths.append(depths[int(parent)] + 1)
    return tuple(depths)
