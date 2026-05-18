"""Compatibility helpers for SGLang 0.5.9."""

from __future__ import annotations

import os
from collections.abc import MutableMapping
from pathlib import Path
from typing import Callable


LEGACY_PATCH_ENV = "SGLANG_GROUP_LEGACY_NGRAM_PATCH"
CHILD_BOOTSTRAP_ENV = "SGLANG_GROUP_CHILD_BOOTSTRAP"


def child_bootstrap_dir() -> Path:
    return Path(__file__).resolve().parent.parent / "_bootstrap"


def install_child_process_patch_hook(
    environ: MutableMapping[str, str] | None = None,
) -> Path:
    """Make spawned SGLang scheduler processes re-apply the legacy patch.

    SGLang 0.5.9 may create scheduler/model worker processes with Python spawn
    semantics. In that case, monkey patches applied in the launcher process are
    not inherited. We prepend a tiny sitecustomize directory to PYTHONPATH so
    child interpreters re-run patch_legacy_ngram_worker() at startup.
    """

    environ = os.environ if environ is None else environ
    bootstrap_dir = child_bootstrap_dir()
    sitecustomize = bootstrap_dir / "sitecustomize.py"
    if not sitecustomize.exists():
        raise RuntimeError(f"Missing sglang-group child bootstrap: {sitecustomize}")

    entries = [
        entry
        for entry in environ.get("PYTHONPATH", "").split(os.pathsep)
        if entry
    ]
    bootstrap_entry = str(bootstrap_dir)
    entries = [entry for entry in entries if entry != bootstrap_entry]
    entries.insert(0, bootstrap_entry)
    environ["PYTHONPATH"] = os.pathsep.join(entries)
    environ[CHILD_BOOTSTRAP_ENV] = "1"
    return bootstrap_dir


def has_native_sglang_group_algorithm() -> bool:
    """Return whether the installed SGLang source already knows SGLANG_GROUP."""

    try:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
    except Exception:
        return False
    return "SGLANG_GROUP" in getattr(SpeculativeAlgorithm, "__members__", {})


def has_native_custom_spec_registry() -> bool:
    try:
        import sglang.srt.speculative.spec_registry  # noqa: F401
    except Exception:
        return False
    return True


def patch_legacy_ngram_worker() -> bool:
    """Patch SGLang 0.5.9 to route NGRAM to SGLANG_GROUP on demand.

    SGLang 0.5.9 only accepts enum algorithm names. The launch wrapper rewrites
    `SGLANG_GROUP` to builtin `NGRAM` for argument parsing, then this patch swaps
    only the worker factory when `SGLANG_GROUP_LEGACY_NGRAM_PATCH=1`.
    """

    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    if getattr(SpeculativeAlgorithm, "_sglang_group_legacy_patch", False):
        return True

    original_create_worker: Callable = SpeculativeAlgorithm.create_worker

    def create_worker(self, server_args):
        if os.getenv(LEGACY_PATCH_ENV) == "1" and self == SpeculativeAlgorithm.NGRAM:
            from .worker import SGLangGroupWorker

            return SGLangGroupWorker
        return original_create_worker(self, server_args)

    SpeculativeAlgorithm.create_worker = create_worker
    SpeculativeAlgorithm._sglang_group_legacy_patch = True
    patch_scheduler_output_hooks()
    return True


def patch_scheduler_output_hooks() -> bool:
    """Patch scheduler output processing to notify draft workers after commits.

    The hook runs after prefill/decode output ids have been committed to
    requests. SGLANG_GROUP uses this to prefetch proposals for prefixes that are
    only visible after scheduler-side output processing, especially target-only
    fallbacks in async-hit mode.
    """

    try:
        from sglang.srt.managers.scheduler_output_processor_mixin import (
            SchedulerOutputProcessorMixin,
        )
    except Exception:
        return False

    if getattr(SchedulerOutputProcessorMixin, "_sglang_group_output_hook_patch", False):
        return True

    original_prefill = SchedulerOutputProcessorMixin.process_batch_result_prefill
    original_decode = SchedulerOutputProcessorMixin.process_batch_result_decode

    def process_batch_result_prefill(self, batch, result):
        ret = original_prefill(self, batch, result)
        _call_draft_worker_hook(self, "post_process_batch_result_prefill", batch, result)
        return ret

    def process_batch_result_decode(self, batch, result):
        ret = original_decode(self, batch, result)
        _call_draft_worker_hook(self, "post_process_batch_result_decode", batch, result)
        return ret

    SchedulerOutputProcessorMixin.process_batch_result_prefill = (
        process_batch_result_prefill
    )
    SchedulerOutputProcessorMixin.process_batch_result_decode = process_batch_result_decode
    SchedulerOutputProcessorMixin._sglang_group_output_hook_patch = True
    return True


def _call_draft_worker_hook(scheduler, hook_name: str, batch, result) -> None:
    draft_worker = getattr(scheduler, "draft_worker", None)
    hook = getattr(draft_worker, hook_name, None)
    if callable(hook):
        hook(batch, result)
