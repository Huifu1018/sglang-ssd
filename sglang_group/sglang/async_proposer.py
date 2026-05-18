"""Async proposal prefetch service for SSD-style SGLANG_GROUP execution."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
from threading import Lock
from typing import Sequence

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
    current_text: str
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


@dataclass
class AsyncProposalStats:
    submitted: int = 0
    submit_skips: int = 0
    ready_hits: int = 0
    ready_misses: int = 0
    completed: int = 0
    failed: int = 0
    cancelled: int = 0
    cache_evictions: int = 0
    evicted_by_request: int = 0
    sync_fallbacks: int = 0
    wait_hits: int = 0
    cuda_pending: int = 0

    def snapshot(self) -> dict[str, object]:
        return self.__dict__.copy()


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
            if proposal is None:
                self.stats.ready_misses += 1
                return None
            if not self._proposal_event_complete(proposal):
                self.stats.ready_misses += 1
                self.stats.cuda_pending += 1
                return None
            self._cache.move_to_end(key)
            self.stats.ready_hits += 1
            return replace(
                proposal,
                cache_event=f"async-hit/{proposal.cache_event}",
                proposal_cache_event="async-hit",
            )

    def has_ready(self, request: AsyncProposalRequest) -> bool:
        key = request.key
        self.collect_finished()
        with self._state_lock:
            proposal = self._cache.get(key)
            if proposal is None:
                return False
            return self._proposal_event_complete(proposal)

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
                return replace(
                    proposal,
                    cache_event=f"async-hit/{proposal.cache_event}",
                    proposal_cache_event="async-hit",
                )
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
            return replace(
                proposal,
                cache_event=f"async-wait/{proposal.cache_event}",
                proposal_cache_event="async-hit",
            )

        proposal = self._run_proposal(request)
        self._wait_for_proposal_event(proposal)
        return proposal

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
                    request.current_text,
                    request.current_target_ids,
                    max_target_tokens=request.max_target_tokens,
                    method=request.method,
                    sampling=request.sampling,
                )

        if self.cuda_overlap is not None:
            run_on_stream = getattr(self.cuda_overlap, "run", None)
            if callable(run_on_stream):
                return run_on_stream(run)
        return run()

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
