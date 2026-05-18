"""Source-level SGLang 0.5.9 integration patcher for SGLANG_GROUP."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path


SUPPORTED_SGLANG_VERSION = "0.5.9"
BACKUP_SUFFIX = ".sglang-group.bak"


@dataclass(frozen=True)
class SourcePatchReport:
    sglang_root: Path
    changed_files: tuple[Path, ...]
    already_integrated: bool
    dry_run: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "sglang_root": str(self.sglang_root),
            "changed_files": [str(path) for path in self.changed_files],
            "already_integrated": self.already_integrated,
            "dry_run": self.dry_run,
        }


def resolve_sglang_root(path: str | Path | None = None) -> Path:
    """Resolve a SGLang package root containing srt/speculative/spec_info.py."""

    candidates: list[Path] = []
    if path is not None:
        raw = Path(path).expanduser().resolve()
        candidates.extend([raw, raw / "sglang"])
    else:
        spec = importlib.util.find_spec("sglang")
        if spec and spec.submodule_search_locations:
            candidates.extend(
                Path(item).resolve() for item in spec.submodule_search_locations
            )

    for candidate in candidates:
        if _spec_info_path(candidate).exists() and _server_args_path(candidate).exists():
            return candidate

    hint = str(path) if path is not None else "installed Python environment"
    raise RuntimeError(f"Cannot find SGLang package root under {hint}.")


def is_source_integrated(path: str | Path | None = None) -> bool:
    root = resolve_sglang_root(path)
    spec_info = _spec_info_path(root).read_text(encoding="utf-8")
    server_args = _server_args_path(root).read_text(encoding="utf-8")
    output_processor = _output_processor_path(root).read_text(encoding="utf-8")
    return (
        "SGLANG_GROUP = auto()" in spec_info
        and "def is_sglang_group(self) -> bool:" in spec_info
        and "from sglang_group.sglang.worker import SGLangGroupWorker" in spec_info
        and '"SGLANG_GROUP"' in server_args
        and "post_process_batch_result_prefill" in output_processor
        and "post_process_batch_result_decode" in output_processor
    )


def apply_source_integration(
    path: str | Path | None = None,
    *,
    dry_run: bool = False,
) -> SourcePatchReport:
    """Patch SGLang 0.5.9 source so it accepts SGLANG_GROUP natively."""

    root = resolve_sglang_root(path)
    changed: list[Path] = []

    spec_info_path = _spec_info_path(root)
    server_args_path = _server_args_path(root)
    output_processor_path = _output_processor_path(root)

    spec_info = spec_info_path.read_text(encoding="utf-8")
    patched_spec_info = _patch_spec_info(spec_info)
    if patched_spec_info != spec_info:
        changed.append(spec_info_path)
        if not dry_run:
            _write_with_backup(spec_info_path, spec_info, patched_spec_info)

    server_args = server_args_path.read_text(encoding="utf-8")
    patched_server_args = _patch_server_args(server_args)
    if patched_server_args != server_args:
        changed.append(server_args_path)
        if not dry_run:
            _write_with_backup(server_args_path, server_args, patched_server_args)

    output_processor = output_processor_path.read_text(encoding="utf-8")
    patched_output_processor = _patch_output_processor(output_processor)
    if patched_output_processor != output_processor:
        changed.append(output_processor_path)
        if not dry_run:
            _write_with_backup(
                output_processor_path,
                output_processor,
                patched_output_processor,
            )

    return SourcePatchReport(
        sglang_root=root,
        changed_files=tuple(changed),
        already_integrated=not changed,
        dry_run=dry_run,
    )


def _patch_spec_info(text: str) -> str:
    patched = text
    if "SGLANG_GROUP = auto()" not in patched:
        patched = _replace_required(
            patched,
            "    NGRAM = auto()\n    NONE = auto()\n",
            "    NGRAM = auto()\n    SGLANG_GROUP = auto()\n    NONE = auto()\n",
            "SpeculativeAlgorithm enum",
        )

    if "def is_sglang_group(self) -> bool:" not in patched:
        patched = _replace_required(
            patched,
            (
                "    def is_ngram(self) -> bool:\n"
                "        return self == SpeculativeAlgorithm.NGRAM\n\n"
                "    def supports_spec_v2(self) -> bool:\n"
            ),
            (
                "    def is_ngram(self) -> bool:\n"
                "        return self in {\n"
                "            SpeculativeAlgorithm.NGRAM,\n"
                "            SpeculativeAlgorithm.SGLANG_GROUP,\n"
                "        }\n\n"
                "    def is_sglang_group(self) -> bool:\n"
                "        return self == SpeculativeAlgorithm.SGLANG_GROUP\n\n"
                "    def supports_spec_v2(self) -> bool:\n"
            ),
            "SpeculativeAlgorithm.is_ngram",
        )

    if "from sglang_group.sglang.worker import SGLangGroupWorker" not in patched:
        patched = _replace_required(
            patched,
            "        elif self.is_ngram():\n",
            (
                "        elif self.is_sglang_group():\n"
                "            if enable_overlap:\n"
                "                raise ValueError(\n"
                "                    f\"Speculative algorithm {self.name} does not support "
                "overlap worker creation.\"\n"
                "                )\n\n"
                "            from sglang_group.sglang.worker import SGLangGroupWorker\n\n"
                "            return SGLangGroupWorker\n"
                "        elif self.is_ngram():\n"
            ),
            "SpeculativeAlgorithm.create_worker",
        )
    return patched


def _patch_server_args(text: str) -> str:
    patched = text
    choices_with_group = (
        'choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", '
        '"NGRAM", "SGLANG_GROUP"],'
    )
    if choices_with_group not in patched:
        patched = _replace_required(
            patched,
            'choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"],',
            choices_with_group,
            "server_args speculative algorithm choices",
        )

    patched = _replace_if_present(
        patched,
        'elif self.speculative_algorithm != "NGRAM":',
        'elif self.speculative_algorithm not in ("NGRAM", "SGLANG_GROUP"):',
    )
    patched = _replace_if_present(
        patched,
        'if self.speculative_algorithm == "NGRAM":',
        'if self.speculative_algorithm in ("NGRAM", "SGLANG_GROUP"):',
    )
    patched = _replace_if_present(
        patched,
        'if self.speculative_algorithm not in ["NGRAM", None]:',
        'if self.speculative_algorithm not in ["NGRAM", "SGLANG_GROUP", None]:',
    )
    return patched


def _patch_output_processor(text: str) -> str:
    patched = text
    if "post_process_batch_result_prefill" not in patched:
        patched = _replace_required(
            patched,
            (
                "        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)\n\n"
                "        if self.current_scheduler_metrics_enabled:\n"
            ),
            (
                "        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)\n\n"
                "        draft_worker_hook = getattr(\n"
                "            getattr(self, \"draft_worker\", None),\n"
                "            \"post_process_batch_result_prefill\",\n"
                "            None,\n"
                "        )\n"
                "        if callable(draft_worker_hook):\n"
                "            draft_worker_hook(batch, result)\n\n"
                "        if self.current_scheduler_metrics_enabled:\n"
            ),
            "scheduler prefill output hook",
        )
    if "post_process_batch_result_decode" not in patched:
        patched = _replace_required(
            patched,
            (
                "        self.stream_output(batch.reqs, batch.return_logprob)\n"
                "        self.token_to_kv_pool_allocator.free_group_end()\n\n"
                "        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)\n"
            ),
            (
                "        self.stream_output(batch.reqs, batch.return_logprob)\n"
                "        self.token_to_kv_pool_allocator.free_group_end()\n\n"
                "        draft_worker_hook = getattr(\n"
                "            getattr(self, \"draft_worker\", None),\n"
                "            \"post_process_batch_result_decode\",\n"
                "            None,\n"
                "        )\n"
                "        if callable(draft_worker_hook):\n"
                "            draft_worker_hook(batch, result)\n\n"
                "        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)\n"
            ),
            "scheduler decode output hook",
        )
    return patched


def _replace_required(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        if new in text:
            return text
        raise RuntimeError(
            f"Cannot patch {label}; expected SGLang 0.5.9 snippet not found."
        )
    return text.replace(old, new, 1)


def _replace_if_present(text: str, old: str, new: str) -> str:
    if old not in text:
        return text
    return text.replace(old, new, 1)


def _write_with_backup(path: Path, original: str, patched: str) -> None:
    backup = path.with_name(path.name + BACKUP_SUFFIX)
    if not backup.exists():
        backup.write_text(original, encoding="utf-8")
    path.write_text(patched, encoding="utf-8")


def _spec_info_path(root: Path) -> Path:
    return root / "srt" / "speculative" / "spec_info.py"


def _server_args_path(root: Path) -> Path:
    return root / "srt" / "server_args.py"


def _output_processor_path(root: Path) -> Path:
    return root / "srt" / "managers" / "scheduler_output_processor_mixin.py"
