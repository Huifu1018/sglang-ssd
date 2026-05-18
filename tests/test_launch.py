import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from sglang_group.cli.launch import (
    _consume_group_args,
    _ensure_legacy_ngram_flags,
    _rewrite_algorithm,
    _uses_sglang_group,
)
from sglang_group.sglang.compat import (
    CHILD_BOOTSTRAP_ENV,
    child_bootstrap_dir,
    install_child_process_patch_hook,
)


class LaunchTests(unittest.TestCase):
    def test_detects_sglang_group_algorithm(self):
        argv = ["--model-path", "target", "--speculative-algorithm", "SGLANG_GROUP"]
        self.assertTrue(_uses_sglang_group(argv))

    def test_rewrites_for_sglang_059(self):
        argv = ["--speculative-algorithm=SGLANG_GROUP"]
        self.assertEqual(_rewrite_algorithm(argv), ["--speculative-algorithm=NGRAM"])

    def test_consumes_group_flags_into_environment(self):
        environ = {}
        argv = [
            "--model-path",
            "target",
            "--sglang-group-method",
            "itl",
            "--sglang-group-auto-high-temp-threshold=0.95",
            "--sglang-group-draft-backend",
            "sglang",
            "--sglang-group-native-draft-quantization=awq",
            "--sglang-group-native-draft-cache-tokens",
            "8192",
            "--sglang-group-native-draft-max-requests=2",
            "--sglang-group-max-cached-proposals",
            "16",
            "--sglang-group-metrics-log-interval=5",
            "--sglang-group-ssd-mode",
            "async-sync-fallback",
            "--sglang-group-ssd-prefetch-workers=2",
            "--sglang-group-ssd-max-prefetch",
            "32",
            "--sglang-group-verify-backend",
            "triton",
            "--sglang-group-tokenizer-bridge=uag",
            "--sglang-group-tree-branch-factor",
            "4",
            "--sglang-group-tree-max-depth=3",
            "--sglang-group-enable-native-draft-kv-cache",
            "--no-sglang-group-cuda-overlap",
            "--no-sglang-group-draft-cache",
            "--no-sglang-group-proposal-cache",
        ]

        remaining = _consume_group_args(argv, environ=environ)

        self.assertEqual(remaining, ["--model-path", "target"])
        self.assertEqual(environ["SGLANG_GROUP_METHOD"], "itl")
        self.assertEqual(environ["SGLANG_GROUP_AUTO_HIGH_TEMP_THRESHOLD"], "0.95")
        self.assertEqual(environ["SGLANG_GROUP_DRAFT_BACKEND"], "sglang")
        self.assertEqual(environ["SGLANG_GROUP_NATIVE_DRAFT_QUANTIZATION"], "awq")
        self.assertEqual(environ["SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS"], "8192")
        self.assertEqual(environ["SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS"], "2")
        self.assertEqual(environ["SGLANG_GROUP_MAX_CACHED_PROPOSALS"], "16")
        self.assertEqual(environ["SGLANG_GROUP_METRICS_LOG_INTERVAL"], "5")
        self.assertEqual(environ["SGLANG_GROUP_SSD_MODE"], "async-sync-fallback")
        self.assertEqual(environ["SGLANG_GROUP_SSD_PREFETCH_WORKERS"], "2")
        self.assertEqual(environ["SGLANG_GROUP_SSD_MAX_PREFETCH"], "32")
        self.assertEqual(environ["SGLANG_GROUP_VERIFY_BACKEND"], "triton")
        self.assertEqual(environ["SGLANG_GROUP_TOKENIZER_BRIDGE"], "uag")
        self.assertEqual(environ["SGLANG_GROUP_TREE_BRANCH_FACTOR"], "4")
        self.assertEqual(environ["SGLANG_GROUP_TREE_MAX_DEPTH"], "3")
        self.assertEqual(environ["SGLANG_GROUP_ENABLE_NATIVE_DRAFT_KV_CACHE"], "true")
        self.assertEqual(environ["SGLANG_GROUP_ENABLE_CUDA_OVERLAP"], "false")
        self.assertEqual(environ["SGLANG_GROUP_ENABLE_DRAFT_CACHE"], "false")
        self.assertEqual(environ["SGLANG_GROUP_ENABLE_PROPOSAL_CACHE"], "false")

    def test_adds_legacy_ngram_flags(self):
        argv = ["--speculative-algorithm=NGRAM"]
        rewritten = _ensure_legacy_ngram_flags(argv)

        self.assertIn("--speculative-ngram-max-bfs-breadth", rewritten)
        self.assertIn("--disable-cuda-graph", rewritten)
        self.assertIn("--disable-overlap-schedule", rewritten)

    def test_installs_child_process_patch_hook(self):
        environ = {"PYTHONPATH": os.pathsep.join(["/existing/path"])}

        bootstrap_dir = install_child_process_patch_hook(environ)
        install_child_process_patch_hook(environ)

        entries = environ["PYTHONPATH"].split(os.pathsep)
        self.assertEqual(bootstrap_dir, child_bootstrap_dir())
        self.assertEqual(entries[0], str(child_bootstrap_dir()))
        self.assertEqual(entries.count(str(child_bootstrap_dir())), 1)
        self.assertIn("/existing/path", entries)
        self.assertEqual(environ[CHILD_BOOTSTRAP_ENV], "1")

    def test_child_sitecustomize_applies_patch_on_spawn(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            spec_dir = root / "sglang" / "srt" / "speculative"
            spec_dir.mkdir(parents=True)
            for package_dir in (
                root / "sglang",
                root / "sglang" / "srt",
                spec_dir,
            ):
                (package_dir / "__init__.py").write_text("", encoding="utf-8")
            (spec_dir / "spec_info.py").write_text(
                textwrap.dedent(
                    """
                    class SpeculativeAlgorithm:
                        NGRAM = "NGRAM"

                        def create_worker(self, server_args):
                            return "ORIGINAL"
                    """
                ),
                encoding="utf-8",
            )

            env = os.environ.copy()
            env["SGLANG_GROUP_LEGACY_NGRAM_PATCH"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(
                [
                    str(child_bootstrap_dir()),
                    str(root),
                    str(Path(__file__).resolve().parents[1]),
                ]
            )
            result = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    (
                        "from sglang.srt.speculative.spec_info import "
                        "SpeculativeAlgorithm; "
                        "print(getattr(SpeculativeAlgorithm, "
                        "'_sglang_group_legacy_patch', False))"
                    ),
                ],
                check=True,
                capture_output=True,
                env=env,
                text=True,
            )

        self.assertEqual(result.stdout.strip(), "True")


if __name__ == "__main__":
    unittest.main()
