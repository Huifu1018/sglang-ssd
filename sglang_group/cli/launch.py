"""Launch SGLang 0.5.9 with SGLANG_GROUP compatibility."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import MutableMapping

from sglang_group import SGLANG_GROUP_ALGORITHM
from sglang_group.sglang.compat import (
    LEGACY_PATCH_ENV,
    has_native_custom_spec_registry,
    has_native_sglang_group_algorithm,
    install_child_process_patch_hook,
    patch_legacy_ngram_worker,
)
from sglang_group.sglang.plugin import activate
from sglang_group.sglang.validation import validate_server_args


GROUP_VALUE_FLAGS = {
    "--sglang-group-method": "SGLANG_GROUP_METHOD",
    "--sglang-group-auto-greedy-method": "SGLANG_GROUP_AUTO_GREEDY_METHOD",
    "--sglang-group-auto-mid-sampling-method": "SGLANG_GROUP_AUTO_MID_SAMPLING_METHOD",
    "--sglang-group-auto-high-sampling-method": "SGLANG_GROUP_AUTO_HIGH_SAMPLING_METHOD",
    "--sglang-group-auto-high-temp-threshold": "SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD",
    "--sglang-group-draft-backend": "SGLANG_GROUP_DRAFT_BACKEND",
    "--sglang-group-draft-device": "SGLANG_GROUP_DRAFT_DEVICE",
    "--sglang-group-draft-device-map": "SGLANG_GROUP_DRAFT_DEVICE_MAP",
    "--sglang-group-draft-dtype": "SGLANG_GROUP_DRAFT_DTYPE",
    "--sglang-group-native-draft-quantization": "SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION",
    "--sglang-group-native-draft-cache-tokens": "SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS",
    "--sglang-group-native-draft-max-requests": "SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS",
    "--sglang-group-native-draft-tp-mode": "SGLANG_GROUP_NATIVE_DRAFT_TP_MODE",
    "--sglang-group-max-draft-tokens": "SGLANG_GROUP_MAX_DRAFT_TOKENS",
    "--sglang-group-max-context-tokens": "SGLANG_GROUP_MAX_CONTEXT_TOKENS",
    "--sglang-group-assistant-lookbehind": "SGLANG_GROUP_ASSISTANT_LOOKBEHIND",
    "--sglang-group-target-lookbehind": "SGLANG_GROUP_TARGET_LOOKBEHIND",
    "--sglang-group-dtw-window": "SGLANG_GROUP_DTW_WINDOW",
    "--sglang-group-max-cached-requests": "SGLANG_GROUP_MAX_CACHED_REQUESTS",
    "--sglang-group-max-cached-proposals": "SGLANG_GROUP_MAX_CACHED_PROPOSALS",
    "--sglang-group-tli-min-intersection": "SGLANG_GROUP_TLI_MIN_INTERSECTION",
    "--sglang-group-metrics-log-interval": "SGLANG_GROUP_METRICS_LOG_INTERVAL",
    "--sglang-group-ssd-mode": "SGLANG_GROUP_SSD_MODE",
    "--sglang-group-ssd-prefetch-workers": "SGLANG_GROUP_SSD_PREFETCH_WORKERS",
    "--sglang-group-ssd-max-prefetch": "SGLANG_GROUP_SSD_MAX_PREFETCH",
    "--sglang-group-verify-backend": "SGLANG_GROUP_VERIFY_BACKEND",
    "--sglang-group-tokenizer-bridge": "SGLANG_GROUP_TOKENIZER_BRIDGE",
    "--sglang-group-tree-branch-factor": "SGLANG_GROUP_TREE_BRANCH_FACTOR",
    "--sglang-group-tree-max-depth": "SGLANG_GROUP_TREE_MAX_DEPTH",
}

GROUP_BOOL_FLAGS = {
    "--no-sglang-group-draft-cache": ("SGLANG_GROUP_ENABLE_DRAFT_CACHE", "false"),
    "--no-sglang-group-proposal-cache": (
        "SGLANG_GROUP_ENABLE_PROPOSAL_CACHE",
        "false",
    ),
    "--no-sglang-group-cache-clone": ("SGLANG_GROUP_CLONE_DRAFT_CACHE", "false"),
    "--sglang-group-enable-native-draft-kv-cache": (
        "SGLANG_GROUP_ENABLE_NATIVE_DRAFT_KV_CACHE",
        "true",
    ),
    "--no-sglang-group-cuda-overlap": ("SGLANG_GROUP_ENABLE_CUDA_OVERLAP", "false"),
}


def _rewrite_algorithm(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    for index, item in enumerate(rewritten):
        if item == "--speculative-algorithm" and index + 1 < len(rewritten):
            if rewritten[index + 1].upper() == SGLANG_GROUP_ALGORITHM:
                rewritten[index + 1] = "NGRAM"
        elif item.startswith("--speculative-algorithm="):
            name = item.split("=", 1)[1]
            if name.upper() == SGLANG_GROUP_ALGORITHM:
                rewritten[index] = "--speculative-algorithm=NGRAM"
    return rewritten


def _uses_sglang_group(argv: list[str]) -> bool:
    for index, item in enumerate(argv):
        if item == "--speculative-algorithm" and index + 1 < len(argv):
            return argv[index + 1].upper() == SGLANG_GROUP_ALGORITHM
        if item.startswith("--speculative-algorithm="):
            return item.split("=", 1)[1].upper() == SGLANG_GROUP_ALGORITHM
    return False


def _consume_group_args(
    argv: list[str],
    *,
    environ: MutableMapping[str, str] | None = None,
) -> list[str]:
    environ = os.environ if environ is None else environ
    remaining: list[str] = []
    index = 0
    while index < len(argv):
        item = argv[index]
        if item in GROUP_BOOL_FLAGS:
            key, value = GROUP_BOOL_FLAGS[item]
            environ[key] = value
            index += 1
            continue

        if item in GROUP_VALUE_FLAGS:
            if index + 1 >= len(argv):
                raise SystemExit(f"{item} requires a value.")
            environ[GROUP_VALUE_FLAGS[item]] = argv[index + 1]
            index += 2
            continue

        if item.startswith("--") and "=" in item:
            flag, value = item.split("=", 1)
            if flag in GROUP_VALUE_FLAGS:
                environ[GROUP_VALUE_FLAGS[flag]] = value
                index += 1
                continue

        remaining.append(item)
        index += 1
    return remaining


def _ensure_legacy_ngram_flags(argv: list[str]) -> list[str]:
    rewritten = list(argv)
    if not _has_option(rewritten, "--speculative-ngram-max-bfs-breadth"):
        rewritten += ["--speculative-ngram-max-bfs-breadth", "1"]
    if not _has_option(rewritten, "--disable-cuda-graph"):
        rewritten.append("--disable-cuda-graph")
    if not _has_option(rewritten, "--disable-overlap-schedule"):
        rewritten.append("--disable-overlap-schedule")
    return rewritten


def _has_option(argv: list[str], option: str) -> bool:
    prefix = option + "="
    return any(item == option or item.startswith(prefix) for item in argv)


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in argv):
        parser = argparse.ArgumentParser(
            description=(
                "Launch SGLang with SGLANG_GROUP. Pass normal sglang.launch_server "
                "arguments; use --speculative-algorithm SGLANG_GROUP."
            ),
            add_help=True,
        )
        parser.add_argument(
            "--sglang-group-method",
            choices=["auto", "itl", "itl-base-slem", "itl-base-tli"],
            help="Unified speculative method. Default: auto.",
        )
        parser.add_argument(
            "--sglang-group-auto-high-temp-threshold",
            help="Temperature at or above which auto selects the high-temp method.",
        )
        parser.add_argument(
            "--sglang-group-draft-backend",
            choices=["transformers", "sglang"],
            help="Draft execution backend. Default: sglang.",
        )
        parser.add_argument(
            "--sglang-group-native-draft-quantization",
            help="Optional SGLang quantization override for backend=sglang.",
        )
        parser.add_argument(
            "--sglang-group-native-draft-cache-tokens",
            help="Optional draft KV pool token cap for backend=sglang.",
        )
        parser.add_argument(
            "--sglang-group-native-draft-max-requests",
            help="Draft request pool size for backend=sglang. Default: 1.",
        )
        parser.add_argument(
            "--sglang-group-native-draft-tp-mode",
            choices=["auto", "replica", "independent"],
            help=(
                "TP mode for backend=sglang draft. auto uses per-GPU replica "
                "for async-hit with target tp_size>1; independent uses a "
                "separate TP group. Default: auto."
            ),
        )
        parser.add_argument(
            "--sglang-group-enable-native-draft-kv-cache",
            action="store_true",
            help=(
                "Enable experimental accepted-context KV reuse for backend=sglang. "
                "Default is safe rebuild per proposal."
            ),
        )
        parser.add_argument(
            "--sglang-group-max-cached-proposals",
            help="Max deterministic proposal-result cache entries. Default: 1024.",
        )
        parser.add_argument(
            "--no-sglang-group-proposal-cache",
            action="store_true",
            help="Disable deterministic ITL/SLEM proposal-result caching.",
        )
        parser.add_argument(
            "--sglang-group-ssd-mode",
            choices=["off", "async-hit", "async-sync-fallback"],
            help=(
                "SSD-style async proposal mode. async-hit uses only ready "
                "background proposals; async-sync-fallback also fills misses "
                "synchronously."
            ),
        )
        parser.add_argument(
            "--sglang-group-ssd-prefetch-workers",
            help="Background proposal worker count. Default: 1.",
        )
        parser.add_argument(
            "--sglang-group-ssd-max-prefetch",
            help="Max completed async proposal entries. Default: 256.",
        )
        parser.add_argument(
            "--sglang-group-verify-backend",
            choices=["auto", "sglang", "torch", "triton"],
            help="Verifier backend. auto uses the SGLANG_GROUP linear fast path.",
        )
        parser.add_argument(
            "--sglang-group-tokenizer-bridge",
            choices=["uag", "segment"],
            help="How draft tokens are retokenized into target ids. Default: uag.",
        )
        parser.add_argument(
            "--sglang-group-tree-branch-factor",
            help="Root branch factor for tree candidates. Default: 1.",
        )
        parser.add_argument(
            "--sglang-group-tree-max-depth",
            help="Max verifier tree depth after the accepted root.",
        )
        parser.add_argument(
            "--no-sglang-group-cuda-overlap",
            action="store_true",
            help="Disable draft proposal CUDA stream/event overlap.",
        )
        parser.add_argument("sglang_args", nargs=argparse.REMAINDER)
        parser.parse_args(argv)
        return

    argv = _consume_group_args(argv)
    if _uses_sglang_group(argv):
        if has_native_sglang_group_algorithm():
            pass
        elif not has_native_custom_spec_registry():
            os.environ[LEGACY_PATCH_ENV] = "1"
            install_child_process_patch_hook()
            patch_legacy_ngram_worker()
            argv = _rewrite_algorithm(argv)
            argv = _ensure_legacy_ngram_flags(argv)
        else:
            activate()
    else:
        activate()

    from sglang.launch_server import run_server
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    server_args = prepare_server_args(argv)
    if (
        str(getattr(server_args, "speculative_algorithm", "")).upper()
        == SGLANG_GROUP_ALGORITHM
        or (
            str(getattr(server_args, "speculative_algorithm", "")).upper() == "NGRAM"
            and os.getenv(LEGACY_PATCH_ENV) == "1"
        )
    ):
        validate_server_args(server_args)

    try:
        run_server(server_args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)


if __name__ == "__main__":
    main()
