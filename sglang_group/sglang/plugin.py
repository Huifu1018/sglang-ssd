"""Optional native SGLang plugin entry point for SGLANG_GROUP."""

from __future__ import annotations

from sglang_group import SGLANG_GROUP_ALGORITHM

from .validation import validate_server_args


def activate() -> None:
    """Register SGLANG_GROUP when SGLang exposes a custom registry.

    SGLang 0.5.9 does not expose this registry; for that version the launch
    wrapper uses the legacy NGRAM patch instead.
    """

    from .compat import patch_legacy_ngram_worker, patch_scheduler_output_hooks

    try:
        from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
        from sglang.srt.speculative.spec_registry import CustomSpecAlgo, get_spec
    except ModuleNotFoundError:
        patch_legacy_ngram_worker()
        return

    patch_scheduler_output_hooks()

    if get_spec(SGLANG_GROUP_ALGORITHM) is not None:
        return

    class SGLangGroupSpecAlgo(CustomSpecAlgo):
        def is_ngram(self) -> bool:
            return True

        def supports_spec_v2(self) -> bool:
            return False

    @SpeculativeAlgorithm.register(
        SGLANG_GROUP_ALGORITHM,
        supports_overlap=False,
        validate_server_args=validate_server_args,
        spec_class=SGLangGroupSpecAlgo,
    )
    def _factory(server_args: object) -> type:
        from .worker import SGLangGroupWorker

        return SGLangGroupWorker
