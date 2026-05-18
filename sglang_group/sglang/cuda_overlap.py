"""CUDA stream/event helpers for async SGLANG_GROUP proposal overlap."""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from threading import Lock
from typing import Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class CudaOverlapStats:
    enqueued: int = 0
    unavailable: int = 0
    ready_hits: int = 0
    ready_misses: int = 0
    waits: int = 0

    def snapshot(self) -> dict[str, int]:
        return self.__dict__.copy()


class CudaStreamOverlapController:
    """Run draft proposal kernels on a non-default stream.

    The async proposer can finish its Python future before CUDA kernels on the
    draft stream have completed. The returned proposal therefore carries a CUDA
    event. Ready-only execution treats the proposal as usable only after the
    event has completed; sync fallback makes the current stream wait on it.
    """

    def __init__(self, *, enabled: bool, device: str | int | None) -> None:
        self.enabled = bool(enabled)
        self.device = device
        self.stats = CudaOverlapStats()
        self._stream = None
        self._torch = None
        self._lock = Lock()

    def run(self, fn: Callable[[], T]) -> T:
        if not self.enabled:
            return fn()
        torch = self._load_torch()
        if torch is None:
            return fn()

        stream = self._ensure_stream(torch)
        if stream is None:
            return fn()

        with torch.cuda.device(self._device_context_arg(torch)):
            with torch.cuda.stream(stream):
                result = fn()
                event = torch.cuda.Event(enable_timing=False)
                event.record(stream)
        self.stats.enqueued += 1
        return _attach_cuda_event(result, event)

    def is_complete(self, proposal: object) -> bool:
        event = getattr(proposal, "cuda_event", None)
        if event is None:
            self.stats.ready_hits += 1
            return True
        try:
            complete = bool(event.query())
        except Exception:
            logger.debug("CUDA proposal event query failed; treating as not ready.", exc_info=True)
            complete = False
        if complete:
            self.stats.ready_hits += 1
        else:
            self.stats.ready_misses += 1
        return complete

    def wait(self, proposal: object) -> None:
        event = getattr(proposal, "cuda_event", None)
        if event is None:
            return
        torch = self._load_torch()
        if torch is None:
            return
        try:
            with torch.cuda.device(self._device_context_arg(torch)):
                torch.cuda.current_stream().wait_event(event)
            self.stats.waits += 1
        except Exception:
            logger.debug("CUDA proposal stream wait failed; falling back to event sync.", exc_info=True)
            try:
                event.synchronize()
                self.stats.waits += 1
            except Exception:
                logger.exception("CUDA proposal event synchronization failed.")

    def snapshot(self) -> dict[str, int | bool | str | None]:
        data: dict[str, int | bool | str | None] = self.stats.snapshot()
        data["enabled"] = self.enabled
        data["device"] = None if self.device is None else str(self.device)
        data["stream_initialized"] = self._stream is not None
        return data

    def _load_torch(self):
        if self._torch is not None:
            return self._torch
        try:
            import torch
        except Exception:
            self.stats.unavailable += 1
            logger.debug("torch is unavailable; CUDA overlap disabled.", exc_info=True)
            return None
        if not getattr(torch, "cuda", None) or not torch.cuda.is_available():
            self.stats.unavailable += 1
            return None
        self._torch = torch
        return torch

    def _ensure_stream(self, torch):
        with self._lock:
            if self._stream is not None:
                return self._stream
            try:
                with torch.cuda.device(self._device_context_arg(torch)):
                    self._stream = torch.cuda.Stream()
            except Exception:
                self.stats.unavailable += 1
                logger.debug("Could not create SGLANG_GROUP draft CUDA stream.", exc_info=True)
                return None
            return self._stream

    def _device_context_arg(self, torch):
        if self.device is None:
            return torch.cuda.current_device()
        return self.device


class NullCudaOverlapController:
    enabled = False

    def run(self, fn: Callable[[], T]) -> T:
        return fn()

    def is_complete(self, proposal: object) -> bool:
        return True

    def wait(self, proposal: object) -> None:
        return None

    def snapshot(self) -> dict[str, object]:
        return {"enabled": False}


def _attach_cuda_event(result: T, event: object) -> T:
    if hasattr(result, "cuda_event"):
        try:
            return replace(result, cuda_event=event, cuda_overlap=True)
        except Exception:
            logger.debug("Could not attach CUDA event to proposal.", exc_info=True)
    return result
