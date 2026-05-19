"""Async proposal prefetch service for SSD-style SGLANG_GROUP execution."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from threading import Lock
from typing import Callable, Sequence

from .proposer import BaseProposal, SamplingRequest

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncProposalKey:
    rid: str
    method: str
    current_target_ids: tuple[int, ...]
    max_target_tokens: int
    temperature: float
    top_k: int
    top_p: float


@dataclass(frozen=True)
class AsyncProposalRequest:
    rid: str
    current_text: str | Callable[[], str]
    current_target_ids: tuple[int, ...]
    max_target_tokens: int
    method: str
    sampling: SamplingRequest

    @property
    def key(self) -> AsyncProposalKey:
        return AsyncProposalKey(
            rid=str(self.rid),
            method=self.method,
            current_target_ids=tuple(int(x) for x in self.current_target_ids),
            max_target_tokens=int(self.max_target_tokens),
            temperature=round(float(self.sampling.temperature), 6),
            top_k=int(self.sampling.top_k),
            top_p=round(float(self.sampling.top_p), 6),
        )

    def resolve_current_text(self) -> str:
        if callable(self.current_text):
            return str(self.current_text())
        return str(self.current_text)


@dataclass
class AsyncProposalStats:
    submitted: int = 0
    submit_skips: int = 0
    ready_hits: int = 0
    shift_hits: int = 0
    ready_misses: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    cache_evictions: int = 0
    evicted_by_request: int = 0
    sync_fallbacks: int = 0
    sync_directs: int = 0
    wait_hits: int = 0
    cuda_pending: int = 0
    stale_drops: int = 0
    latest_skips_due_to_inflight: int = 0
    proposal_runs: int = 0
    proposal_run_wall_time_s: float = 0.0
    proposal_max_run_wall_time_s: float = 0.0

    def snapshot(self) -> dict[str, object]:
        data = self.__dict__.copy()
        if self.proposal_runs:
            data["proposal_run_ms_avg"] = (
                self.proposal_run_wall_time_s * 1000.0 / self.proposal_runs
            )
            data["proposal_run_ms_max"] = self.proposal_max_run_wall_time_s * 1000.0
        return data


class AsyncProposalService:
    """Background proposal cache.

    The service owns the proposal cache and serializes access to the underlying
    proposer. This gives the worker a ready-proposal fast path while preserving a
    synchronous fallback path for correctness.
    """

    def __init__(
        self,
        proposer: object,
        *,
        max_workers: int = 1,
        max_entries: int = 256,
        cuda_overlap: object | None = None,
    ) -> None:
        self.proposer = proposer
        self.max_entries = max(1, int(max_entries))
        self.cuda_overlap = cuda_overlap
        self._executor = ThreadPoolExecutor(
            max_workers=max(1, int(max_workers)),
            thread_name_prefix="sglang-group-ssd",
        )
        self._state_lock = Lock()
        self._proposer_lock = Lock()
        self._cache: OrderedDict[AsyncProposalKey, BaseProposal] = OrderedDict()
        self._inflight: dict[AsyncProposalKey, Future[BaseProposal]] = {}
        self.stats = AsyncProposalStats()

    def get_ready(self, request: AsyncProposalRequest) -> BaseProposal | None:
        key = request.key
        self.collect_finished()
        with self._state_lock:
            proposal = self._cache.get(key)
            if proposal is not None:
                if not self._proposal_event_complete(proposal):
                    self.stats.ready_misses += 1
                    self.stats.cuda_pending += 1
                    return None
                self._cache.move_to_end(key)
                self.stats.ready_hits += 1
                return self._ready_proposal(proposal, "async-hit")

            shifted, pending = self._find_shifted_ready_locked(key)
            if shifted is not None:
                self.stats.ready_hits += 1
                self.stats.shift_hits += 1
                return shifted
            self.stats.ready_misses += 1
            if pending:
                self.stats.cuda_pending += 1
            return None

    def has_ready(self, request: AsyncProposalRequest) -> bool:
        key = request.key
        self.collect_finished()
        with self._state_lock:
            proposal = self._cache.get(key)
            if proposal is not None:
                return self._proposal_event_complete(proposal)
            shifted, _ = self._find_shifted_ready_locked(key)
            return shifted is not None

    def submit(self, request: AsyncProposalRequest) -> bool:
        key = request.key
        with self._state_lock:
            if key in self._cache or key in self._inflight:
                self.stats.submit_skips += 1
                return False
            future = self._executor.submit(self._run_proposal, request)
            self._inflight[key] = future
            self.stats.submitted += 1
            return True

    def submit_latest(self, request: AsyncProposalRequest) -> bool:
        """Submit only the newest prefix for one request.

        Exact matches are best, but a completed proposal for an older prefix can
        still be used if the target has consumed a prefix of its candidate row.
        Such shiftable cache entries are kept; entries that cannot match the
        newest target prefix are dropped.
        """

        key = request.key
        self.collect_finished()
        with self._state_lock:
            if key in self._cache or key in self._inflight:
                self.stats.submit_skips += 1
                return False

            for old_key in list(self._cache.keys()):
                if old_key.rid == key.rid:
                    old_proposal = self._cache.get(old_key)
                    if (
                        old_proposal is not None
                        and self._shift_proposal_for_key(old_key, old_proposal, key)
                        is not None
                    ):
                        continue
                    self._cache.pop(old_key, None)
                    self.stats.stale_drops += 1

            for old_key, future in list(self._inflight.items()):
                if old_key.rid != key.rid:
                    continue
                if future.cancel():
                    self._inflight.pop(old_key, None)
                    self.stats.cancelled += 1
                    self.stats.stale_drops += 1
                    continue

                # A stale proposal is already running. Queueing another one for
                # the same request usually makes async-hit fall behind further.
                self.stats.submit_skips += 1
                self.stats.latest_skips_due_to_inflight += 1
                return False

            future = self._executor.submit(self._run_proposal, request)
            self._inflight[key] = future
            self.stats.submitted += 1
            return True

    def propose_sync(self, request: AsyncProposalRequest) -> BaseProposal:
        key = request.key
        self.collect_finished()
        with self._state_lock:
            self.stats.sync_fallbacks += 1
            proposal = self._cache.get(key)
            if proposal is not None:
                self._cache.move_to_end(key)
                self.stats.ready_hits += 1
                self._wait_for_proposal_event(proposal)
                return self._ready_proposal(proposal, "async-hit")

            shifted, _ = self._find_shifted_ready_locked(key)
            if shifted is not None:
                self.stats.ready_hits += 1
                self.stats.shift_hits += 1
                return shifted

            future = self._inflight.get(key)

        if future is not None:
            try:
                proposal = future.result()
            except Exception:
                logger.exception(
                    "SGLANG_GROUP SSD async proposal failed before sync fallback: %s",
                    key,
                )
                with self._state_lock:
                    self._inflight.pop(key, None)
                    self.stats.failed += 1
                return self._run_proposal(request)

            with self._state_lock:
                self._inflight.pop(key, None)
                self._cache[key] = proposal
                self._cache.move_to_end(key)
                self.stats.completed += 1
                self.stats.wait_hits += 1
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
                    self.stats.cache_evictions += 1
            self._wait_for_proposal_event(proposal)
            return self._ready_proposal(proposal, "async-wait")

        proposal = self._run_proposal(request)
        self._wait_for_proposal_event(proposal)
        with self._state_lock:
            self._cache[key] = proposal
            self._cache.move_to_end(key)
            self.stats.sync_directs += 1
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
                self.stats.cache_evictions += 1
        return self._ready_proposal(proposal, "async-sync")

    def collect_finished(self) -> None:
        with self._state_lock:
            finished = [
                (key, future)
                for key, future in self._inflight.items()
                if future.done()
            ]
            for key, future in finished:
                self._inflight.pop(key, None)

        for key, future in finished:
            try:
                proposal = future.result()
            except Exception:
                logger.exception("SGLANG_GROUP SSD async proposal failed: %s", key)
                with self._state_lock:
                    self.stats.failed += 1
                continue

            with self._state_lock:
                self._cache[key] = proposal
                self._cache.move_to_end(key)
                self.stats.completed += 1
                while len(self._cache) > self.max_entries:
                    self._cache.popitem(last=False)
                    self.stats.cache_evictions += 1

    def evict(self, rids: Sequence[str]) -> None:
        rid_set = {str(rid) for rid in rids}
        with self._state_lock:
            for key in list(self._cache.keys()):
                if key.rid in rid_set:
                    self._cache.pop(key, None)
                    self.stats.evicted_by_request += 1
            for key, future in list(self._inflight.items()):
                if key.rid in rid_set:
                    if future.cancel():
                        self.stats.cancelled += 1
                    self._inflight.pop(key, None)

        with self._proposer_lock:
            evict = getattr(self.proposer, "evict", None)
            if callable(evict):
                evict(rids)

    def clear(self) -> None:
        with self._state_lock:
            for future in self._inflight.values():
                if future.cancel():
                    self.stats.cancelled += 1
            self._inflight.clear()
            self._cache.clear()

        with self._proposer_lock:
            clear = getattr(self.proposer, "clear", None)
            if callable(clear):
                clear()

    def shutdown(self) -> None:
        self.clear()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def drain(self, timeout_s: float = 5.0) -> None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            self.collect_finished()
            with self._state_lock:
                futures = list(self._inflight.values())
            if not futures:
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            wait(futures, timeout=min(0.05, remaining))

    def snapshot(self) -> dict[str, object]:
        self.collect_finished()
        with self._state_lock:
            data = self.stats.snapshot()
            data["cache_size"] = len(self._cache)
            data["inflight"] = len(self._inflight)
            if self.cuda_overlap is not None:
                snapshot = getattr(self.cuda_overlap, "snapshot", None)
                if callable(snapshot):
                    data["cuda_overlap"] = snapshot()
            return data

    def _run_proposal(self, request: AsyncProposalRequest) -> BaseProposal:
        def run() -> BaseProposal:
            with self._proposer_lock:
                return self.proposer.propose(
                    request.rid,
                    request.resolve_current_text(),
                    request.current_target_ids,
                    max_target_tokens=request.max_target_tokens,
                    method=request.method,
                    sampling=request.sampling,
                )

        start = time.monotonic()
        try:
            if self.cuda_overlap is not None:
                run_on_stream = getattr(self.cuda_overlap, "run", None)
                if callable(run_on_stream):
                    return run_on_stream(run)
            return run()
        finally:
            elapsed = time.monotonic() - start
            with self._state_lock:
                self.stats.proposal_runs += 1
                self.stats.proposal_run_wall_time_s += elapsed
                self.stats.proposal_max_run_wall_time_s = max(
                    self.stats.proposal_max_run_wall_time_s,
                    elapsed,
                )

    def _proposal_event_complete(self, proposal: BaseProposal) -> bool:
        if self.cuda_overlap is None:
            return True
        is_complete = getattr(self.cuda_overlap, "is_complete", None)
        if not callable(is_complete):
            return True
        return bool(is_complete(proposal))

    def _wait_for_proposal_event(self, proposal: BaseProposal) -> None:
        if self.cuda_overlap is None:
            return
        wait_for_event = getattr(self.cuda_overlap, "wait", None)
        if callable(wait_for_event):
            wait_for_event(proposal)

    @staticmethod
    def _ready_proposal(proposal: BaseProposal, prefix: str) -> BaseProposal:
        return replace(
            proposal,
            cache_event=f"{prefix}/{proposal.cache_event}",
            proposal_cache_event="async-hit",
        )

    def _find_shifted_ready_locked(
        self,
        key: AsyncProposalKey,
    ) -> tuple[BaseProposal | None, bool]:
        pending = False
        for old_key in list(reversed(self._cache)):
            proposal = self._cache[old_key]
            shifted = self._shift_proposal_for_key(old_key, proposal, key)
            if shifted is None:
                continue
            if not self._proposal_event_complete(proposal):
                pending = True
                continue
            self._cache.move_to_end(old_key)
            return shifted, pending
        return None, pending

    @staticmethod
    def _shift_proposal_for_key(
        old_key: AsyncProposalKey,
        proposal: BaseProposal,
        key: AsyncProposalKey,
    ) -> BaseProposal | None:
        if (
            old_key.rid != key.rid
            or old_key.method != key.method
            or old_key.max_target_tokens != key.max_target_tokens
            or old_key.temperature != key.temperature
            or old_key.top_k != key.top_k
            or old_key.top_p != key.top_p
        ):
            return None

        old_prefix = tuple(int(token_id) for token_id in old_key.current_target_ids)
        current = tuple(int(token_id) for token_id in key.current_target_ids)
        if len(current) <= len(old_prefix):
            return None
        if current[: len(old_prefix)] != old_prefix:
            return None

        target_token_ids = tuple(int(token_id) for token_id in proposal.target_token_ids)
        consumed_ids = current[len(old_prefix) :]
        consumed = len(consumed_ids)
        if consumed <= 0 or consumed >= len(target_token_ids):
            return None
        if target_token_ids[:consumed] != consumed_ids:
            return None

        remaining = target_token_ids[consumed:]
        draft_prob_rows = _shift_draft_prob_rows(proposal.draft_prob_rows, consumed)
        if proposal.draft_prob_rows is not None and draft_prob_rows is None:
            return None

        draft_token_ids = proposal.draft_token_ids
        if (
            proposal.tokenizer_bridge == "tli"
            or len(proposal.draft_token_ids) == len(target_token_ids)
        ):
            draft_token_ids = tuple(proposal.draft_token_ids[consumed:])

        return replace(
            proposal,
            draft_token_ids=tuple(int(token_id) for token_id in draft_token_ids),
            target_token_ids=remaining,
            draft_prob_rows=draft_prob_rows,
            cache_event=f"async-shift/{proposal.cache_event}",
            proposal_cache_event="async-hit",
            target_tree_token_ids=remaining,
            target_tree_parent_indices=tuple(range(len(remaining))),
        )


def _shift_draft_prob_rows(rows: object | None, consumed: int) -> object | None:
    if rows is None:
        return None
    try:
        if len(rows) <= consumed:  # type: ignore[arg-type]
            return None
        return rows[consumed:]  # type: ignore[index]
    except TypeError:
        return None
