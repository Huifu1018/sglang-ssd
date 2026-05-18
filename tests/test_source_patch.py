import tempfile
import unittest
from pathlib import Path

from sglang_group.sglang.source_patch import (
    apply_source_integration,
    is_source_integrated,
    resolve_sglang_root,
)


SPEC_INFO_059 = '''from __future__ import annotations

from enum import Enum, auto


class SpeculativeAlgorithm(Enum):
    """Enumeration of speculative decoding algorithms."""

    EAGLE = auto()
    EAGLE3 = auto()
    STANDALONE = auto()
    NGRAM = auto()
    NONE = auto()

    def is_ngram(self) -> bool:
        return self == SpeculativeAlgorithm.NGRAM

    def supports_spec_v2(self) -> bool:
        return self.is_eagle() or self.is_standalone()

    def create_worker(self, server_args):
        enable_overlap = not server_args.disable_overlap_schedule
        if self.is_eagle():
            return "EAGLEWorker"
        elif self.is_standalone():
            return "StandaloneWorker"
        elif self.is_ngram():
            if enable_overlap:
                raise ValueError(
                    f"Speculative algorithm {self.name} does not support "
                    "overlap worker creation."
                )

            from sglang.srt.speculative.ngram_worker import NGRAMWorker

            return NGRAMWorker

        raise ValueError("Unreachable code path in create_worker.")
'''


SERVER_ARGS_059 = '''def reserve_memory(self):
    if self.speculative_algorithm is not None:
        if self.speculative_algorithm == "STANDALONE":
            reserved_mem += 6 * 1024
        elif self.speculative_algorithm != "NGRAM":
            reserved_mem += 2 * 1024

def handle_spec(self):
    if self.speculative_algorithm == "NGRAM":
        self.disable_overlap_schedule = True

def add_args(parser):
    parser.add_argument(
        "--speculative-algorithm",
        type=str,
        choices=["EAGLE", "EAGLE3", "NEXTN", "STANDALONE", "NGRAM"],
        help="Speculative algorithm.",
    )

def validate_lora(self):
    if self.speculative_algorithm not in ["NGRAM", None]:
        raise ValueError(
            "Currently LoRA is only compatible with NGRAM speculative decoding."
        )
'''


OUTPUT_PROCESSOR_059 = '''class SchedulerOutputProcessorMixin:
    def process_batch_result_prefill(self, batch, result):
        self.stream_output(batch.reqs, batch.return_logprob, skip_stream_req)

        if self.current_scheduler_metrics_enabled:
            self.log_prefill_stats(prefill_stats=batch.prefill_stats)

    def process_batch_result_decode(self, batch, result):
        self.stream_output(batch.reqs, batch.return_logprob)
        self.token_to_kv_pool_allocator.free_group_end()

        self.forward_ct_decode = (self.forward_ct_decode + 1) % (1 << 30)
'''


class SourcePatchTests(unittest.TestCase):
    def test_applies_source_integration_idempotently(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_fake_sglang_tree(Path(tmpdir))

            report = apply_source_integration(root)

            self.assertFalse(report.already_integrated)
            self.assertEqual(len(report.changed_files), 3)
            self.assertTrue(
                (
                    root / "srt" / "speculative" / "spec_info.py.sglang-group.bak"
                ).exists()
            )
            self.assertTrue(
                (root / "srt" / "server_args.py.sglang-group.bak").exists()
            )
            self.assertTrue(
                (
                    root
                    / "srt"
                    / "managers"
                    / "scheduler_output_processor_mixin.py.sglang-group.bak"
                ).exists()
            )
            self.assertTrue(is_source_integrated(root))

            second = apply_source_integration(root)
            self.assertTrue(second.already_integrated)
            self.assertEqual(second.changed_files, ())

    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = _write_fake_sglang_tree(Path(tmpdir))

            report = apply_source_integration(root, dry_run=True)

            self.assertFalse(report.already_integrated)
            self.assertEqual(len(report.changed_files), 3)
            self.assertFalse(is_source_integrated_no_raise(root))

    def test_resolves_repository_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            repo_root = Path(tmpdir)
            package_root = _write_fake_sglang_tree(repo_root / "sglang")

            self.assertEqual(resolve_sglang_root(repo_root), package_root.resolve())


def _write_fake_sglang_tree(root: Path) -> Path:
    spec_dir = root / "srt" / "speculative"
    spec_dir.mkdir(parents=True)
    managers_dir = root / "srt" / "managers"
    managers_dir.mkdir(parents=True)
    (spec_dir / "spec_info.py").write_text(SPEC_INFO_059, encoding="utf-8")
    (root / "srt" / "server_args.py").write_text(SERVER_ARGS_059, encoding="utf-8")
    (managers_dir / "scheduler_output_processor_mixin.py").write_text(
        OUTPUT_PROCESSOR_059,
        encoding="utf-8",
    )
    return root


def is_source_integrated_no_raise(root: Path) -> bool:
    try:
        return is_source_integrated(root)
    except RuntimeError:
        return False


if __name__ == "__main__":
    unittest.main()
