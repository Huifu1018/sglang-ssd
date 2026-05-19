"""Runtime configuration for the SGLang SGLANG_GROUP worker."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_value(*names: str, default: str | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value is not None and value.strip() != "":
            return value
    return default


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


def _env_float(name: str, default: float | None) -> float | None:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    if value.strip().lower() in {"0", "false", "off", "none"}:
        return None
    parsed = float(value)
    if parsed == 0:
        return None
    if parsed < 0:
        raise ValueError(f"{name} must be positive when set.")
    return parsed


GROUP_METHODS = {"itl", "itl-base-slem", "itl-base-tli"}
GROUP_METHOD_ALIASES = {
    "token-itl": "itl",
    "token_itl": "itl",
    "tokentiming": "itl",
    "slem": "itl-base-slem",
    "base-slem": "itl-base-slem",
    "itl_base_slem": "itl-base-slem",
    "itl-base-slem": "itl-base-slem",
    "tli": "itl-base-tli",
    "base-tli": "itl-base-tli",
    "itl_base_tli": "itl-base-tli",
    "itl-base-tli": "itl-base-tli",
}
DRAFT_BACKENDS = {"transformers", "sglang"}
DRAFT_BACKEND_ALIASES = {
    "hf": "transformers",
    "huggingface": "transformers",
    "transformers": "transformers",
    "native": "sglang",
    "sglang": "sglang",
    "sglang-native": "sglang",
    "srt": "sglang",
}
SSD_MODES = {"off", "async-hit", "async-sync-fallback"}
SSD_MODE_ALIASES = {
    "0": "off",
    "false": "off",
    "none": "off",
    "off": "off",
    "hit": "async-hit",
    "async-hit": "async-hit",
    "async_hit": "async-hit",
    "ready": "async-hit",
    "ready-only": "async-hit",
    "ready_only": "async-hit",
    "async": "async-sync-fallback",
    "async-fallback": "async-sync-fallback",
    "async_fallback": "async-sync-fallback",
    "async-sync-fallback": "async-sync-fallback",
    "async_sync_fallback": "async-sync-fallback",
}
VERIFY_BACKENDS = {"auto", "sglang", "torch", "triton"}
VERIFY_BACKEND_ALIASES = {
    "auto": "auto",
    "default": "auto",
    "native": "auto",
    "sglang": "sglang",
    "upstream": "sglang",
    "torch": "torch",
    "pytorch": "torch",
    "triton": "triton",
    "cuda": "triton",
}
TOKENIZER_BRIDGES = {"uag", "segment"}
TOKENIZER_BRIDGE_ALIASES = {
    "uag": "uag",
    "slem": "uag",
    "lookbehind": "uag",
    "retokenize": "uag",
    "retokenise": "uag",
    "segment": "segment",
    "text": "segment",
}


def normalize_group_method(value: str, *, allow_auto: bool = False) -> str:
    method = value.strip().lower().replace("_", "-")
    if allow_auto and method == "auto":
        return method
    method = GROUP_METHOD_ALIASES.get(method, method)
    if method not in GROUP_METHODS:
        allowed = ["auto"] if allow_auto else []
        allowed.extend(sorted(GROUP_METHODS))
        raise ValueError(f"method must be one of: {', '.join(allowed)}.")
    return method


def normalize_draft_backend(value: str) -> str:
    backend = value.strip().lower().replace("_", "-")
    backend = DRAFT_BACKEND_ALIASES.get(backend, backend)
    if backend not in DRAFT_BACKENDS:
        raise ValueError(
            "draft backend must be one of: " + ", ".join(sorted(DRAFT_BACKENDS)) + "."
        )
    return backend


def normalize_ssd_mode(value: str) -> str:
    mode = value.strip().lower().replace("_", "-")
    mode = SSD_MODE_ALIASES.get(mode, mode)
    if mode not in SSD_MODES:
        raise ValueError(
            "ssd mode must be one of: " + ", ".join(sorted(SSD_MODES)) + "."
        )
    return mode


def normalize_verify_backend(value: str) -> str:
    backend = value.strip().lower().replace("_", "-")
    backend = VERIFY_BACKEND_ALIASES.get(backend, backend)
    if backend not in VERIFY_BACKENDS:
        raise ValueError(
            "verify backend must be one of: " + ", ".join(sorted(VERIFY_BACKENDS)) + "."
        )
    return backend


def normalize_tokenizer_bridge(value: str) -> str:
    bridge = value.strip().lower().replace("_", "-")
    bridge = TOKENIZER_BRIDGE_ALIASES.get(bridge, bridge)
    if bridge not in TOKENIZER_BRIDGES:
        raise ValueError(
            "tokenizer bridge must be one of: " + ", ".join(sorted(TOKENIZER_BRIDGES)) + "."
        )
    return bridge


@dataclass(frozen=True)
class GroupSGLangConfig:
    """Configuration read from environment variables.

    SGLang 0.5.9 does not expose plugin-owned CLI flags. The launch wrapper
    keeps the standard SGLang arguments and uses these environment variables for
    integration-specific behavior.
    """

    method: str = "auto"
    auto_greedy_method: str = "itl-base-slem"
    auto_mid_sampling_method: str = "itl-base-tli"
    auto_high_sampling_method: str = "itl-base-tli"
    auto_high_temp_threshold: float = 0.9
    draft_backend: str = "sglang"
    draft_device: str | None = None
    draft_device_map: str | None = None
    draft_dtype: str = "auto"
    native_draft_quantization: str | None = None
    native_draft_cache_tokens: int | None = None
    native_draft_max_requests: int = 1
    native_draft_kv_cache: bool = False
    dtw_window: int | None = 8
    max_draft_tokens: int | None = None
    max_context_tokens: int | None = None
    assistant_lookbehind: int = 10
    target_lookbehind: int = 10
    max_cached_requests: int = 256
    add_special_tokens: bool = False
    disable_cuda_graph: bool = True
    enable_draft_cache: bool = True
    enable_proposal_cache: bool = True
    clone_draft_cache: bool = True
    max_cached_proposals: int = 1024
    tli_min_intersection: int = 1
    metrics_log_interval: float | None = 60.0
    ssd_mode: str = "off"
    ssd_prefetch_workers: int = 1
    ssd_max_prefetch: int = 256
    cuda_overlap: bool = True
    verify_backend: str = "auto"
    tokenizer_bridge: str = "uag"
    tree_branch_factor: int = 1
    tree_max_depth: int | None = None

    @classmethod
    def from_env(cls, *, default_draft_device: str | None = None) -> "GroupSGLangConfig":
        method = normalize_group_method(
            _env_value("SGLANG_GROUP_METHOD", default="auto") or "auto",
            allow_auto=True,
        )
        auto_greedy_method = normalize_group_method(
            _env_value("SGLANG_GROUP_AUTO_GREEDY_METHOD", default="itl-base-slem")
            or "itl-base-slem"
        )
        auto_mid_sampling_method = normalize_group_method(
            _env_value(
                "SGLANG_GROUP_AUTO_MID_SAMPLING_METHOD",
                default="itl-base-tli",
            )
            or "itl-base-tli"
        )
        auto_high_sampling_method = normalize_group_method(
            _env_value(
                "SGLANG_GROUP_AUTO_HIGH_SAMPLING_METHOD",
                default="itl-base-tli",
            )
            or "itl-base-tli"
        )
        return cls(
            method=method,
            auto_greedy_method=auto_greedy_method,
            auto_mid_sampling_method=auto_mid_sampling_method,
            auto_high_sampling_method=auto_high_sampling_method,
            auto_high_temp_threshold=(
                _env_float("SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD", 0.9) or 0.9
            ),
            draft_backend=normalize_draft_backend(
                _env_value("SGLANG_GROUP_DRAFT_BACKEND", default="sglang")
                or "sglang"
            ),
            draft_device=os.getenv("SGLANG_GROUP_DRAFT_DEVICE", default_draft_device),
            draft_device_map=os.getenv("SGLANG_GROUP_DRAFT_DEVICE_MAP") or None,
            draft_dtype=os.getenv("SGLANG_GROUP_DRAFT_DTYPE", "auto"),
            native_draft_quantization=os.getenv(
                "SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION"
            )
            or None,
            native_draft_cache_tokens=_env_int(
                "SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS", None
            ),
            native_draft_max_requests=(
                _env_int("SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS", 1) or 1
            ),
            native_draft_kv_cache=_env_bool(
                "SGLANG_GROUP_ENABLE_NATIVE_DRAFT_KV_CACHE", False
            ),
            dtw_window=_env_int("SGLANG_GROUP_DTW_WINDOW", 8),
            max_draft_tokens=_env_int("SGLANG_GROUP_MAX_DRAFT_TOKENS", None),
            max_context_tokens=_env_int("SGLANG_GROUP_MAX_CONTEXT_TOKENS", None),
            assistant_lookbehind=_env_int("SGLANG_GROUP_ASSISTANT_LOOKBEHIND", 10) or 10,
            target_lookbehind=_env_int("SGLANG_GROUP_TARGET_LOOKBEHIND", 10) or 10,
            max_cached_requests=_env_int("SGLANG_GROUP_MAX_CACHED_REQUESTS", 256) or 256,
            add_special_tokens=_env_bool("SGLANG_GROUP_ADD_SPECIAL_TOKENS", False),
            disable_cuda_graph=_env_bool("SGLANG_GROUP_DISABLE_CUDA_GRAPH", True),
            enable_draft_cache=_env_bool("SGLANG_GROUP_ENABLE_DRAFT_CACHE", True),
            enable_proposal_cache=_env_bool("SGLANG_GROUP_ENABLE_PROPOSAL_CACHE", True),
            clone_draft_cache=_env_bool("SGLANG_GROUP_CLONE_DRAFT_CACHE", True),
            max_cached_proposals=(
                _env_int("SGLANG_GROUP_MAX_CACHED_PROPOSALS", 1024) or 1024
            ),
            tli_min_intersection=_env_int("SGLANG_GROUP_TLI_MIN_INTERSECTION", 1) or 1,
            metrics_log_interval=_env_float("SGLANG_GROUP_METRICS_LOG_INTERVAL", 60.0),
            ssd_mode=normalize_ssd_mode(
                _env_value("SGLANG_GROUP_SSD_MODE", default="off") or "off"
            ),
            ssd_prefetch_workers=(
                _env_int("SGLANG_GROUP_SSD_PREFETCH_WORKERS", 1) or 1
            ),
            ssd_max_prefetch=_env_int("SGLANG_GROUP_SSD_MAX_PREFETCH", 256) or 256,
            cuda_overlap=_env_bool("SGLANG_GROUP_ENABLE_CUDA_OVERLAP", True),
            verify_backend=normalize_verify_backend(
                _env_value("SGLANG_GROUP_VERIFY_BACKEND", default="auto") or "auto"
            ),
            tokenizer_bridge=normalize_tokenizer_bridge(
                _env_value("SGLANG_GROUP_TOKENIZER_BRIDGE", default="uag") or "uag"
            ),
            tree_branch_factor=(
                _env_int("SGLANG_GROUP_TREE_BRANCH_FACTOR", 1) or 1
            ),
            tree_max_depth=_env_int("SGLANG_GROUP_TREE_MAX_DEPTH", None),
        )

    def method_for_batch(
        self,
        *,
        is_all_greedy: bool,
        max_temperature: float | None = None,
    ) -> str:
        if self.method != "auto":
            return self.method
        if is_all_greedy or (max_temperature is not None and max_temperature <= 0):
            return self.auto_greedy_method
        if (
            max_temperature is not None
            and max_temperature >= self.auto_high_temp_threshold
        ):
            return self.auto_high_sampling_method
        return self.auto_mid_sampling_method
