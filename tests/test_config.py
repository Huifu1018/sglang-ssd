import unittest
from unittest.mock import patch

from sglang_group.sglang.config import (
    GroupSGLangConfig,
    normalize_draft_backend,
    normalize_group_method,
    normalize_ssd_mode,
    normalize_tokenizer_bridge,
    normalize_verify_backend,
)


class ConfigTests(unittest.TestCase):
    def test_auto_method_selection(self):
        config = GroupSGLangConfig(method="auto")
        self.assertEqual(
            config.method_for_batch(is_all_greedy=True, max_temperature=0.0),
            "itl-base-slem",
        )
        self.assertEqual(
            config.method_for_batch(is_all_greedy=False, max_temperature=0.6),
            "itl-base-tli",
        )
        self.assertEqual(
            config.method_for_batch(is_all_greedy=False, max_temperature=1.0),
            "itl-base-tli",
        )

    def test_method_aliases(self):
        self.assertEqual(normalize_group_method("slem"), "itl-base-slem")
        self.assertEqual(normalize_group_method("tli"), "itl-base-tli")
        self.assertEqual(normalize_group_method("token_itl"), "itl")

    def test_draft_backend_aliases(self):
        self.assertEqual(normalize_draft_backend("hf"), "transformers")
        self.assertEqual(normalize_draft_backend("sglang-native"), "sglang")

    def test_ssd_mode_aliases(self):
        self.assertEqual(normalize_ssd_mode("off"), "off")
        self.assertEqual(normalize_ssd_mode("ready_only"), "async-hit")
        self.assertEqual(normalize_ssd_mode("async"), "async-sync-fallback")

    def test_verify_backend_aliases(self):
        self.assertEqual(normalize_verify_backend("cuda"), "triton")
        self.assertEqual(normalize_verify_backend("upstream"), "sglang")

    def test_tokenizer_bridge_aliases(self):
        self.assertEqual(normalize_tokenizer_bridge("slem"), "uag")
        self.assertEqual(normalize_tokenizer_bridge("text"), "segment")

    def test_default_draft_backend_is_sglang(self):
        config = GroupSGLangConfig.from_env()
        self.assertEqual(config.draft_backend, "sglang")
        self.assertFalse(config.native_draft_kv_cache)
        self.assertTrue(config.enable_proposal_cache)
        self.assertEqual(config.max_cached_proposals, 1024)
        self.assertTrue(config.cuda_overlap)
        self.assertEqual(config.verify_backend, "auto")
        self.assertEqual(config.tokenizer_bridge, "uag")
        self.assertEqual(config.tree_branch_factor, 1)
        self.assertIsNone(config.tree_max_depth)

    def test_env_validation(self):
        with patch.dict("os.environ", {"SGLANG_GROUP_METHOD": "bad"}):
            with self.assertRaises(ValueError):
                GroupSGLangConfig.from_env()

    def test_draft_backend_from_env(self):
        with patch.dict(
            "os.environ",
            {
                "SGLANG_GROUP_DRAFT_BACKEND": "sglang",
                "SGLANG_GROUP_NATIVE_DRAFT_CACHE_TOKENS": "4096",
                "SGLANG_GROUP_NATIVE_DRAFT_MAX_REQUESTS": "2",
                "SGLANG_GROUP_ENABLE_NATIVE_DRAFT_KV_CACHE": "1",
                "SGLANG_GROUP_ENABLE_PROPOSAL_CACHE": "0",
                "SGLANG_GROUP_MAX_CACHED_PROPOSALS": "12",
                "SGLANG_GROUP_METRICS_LOG_INTERVAL": "0",
                "SGLANG_GROUP_SSD_MODE": "async",
                "SGLANG_GROUP_SSD_PREFETCH_WORKERS": "2",
                "SGLANG_GROUP_SSD_MAX_PREFETCH": "32",
                "SGLANG_GROUP_ENABLE_CUDA_OVERLAP": "0",
                "SGLANG_GROUP_VERIFY_BACKEND": "cuda",
                "SGLANG_GROUP_TOKENIZER_BRIDGE": "text",
                "SGLANG_GROUP_TREE_BRANCH_FACTOR": "4",
                "SGLANG_GROUP_TREE_MAX_DEPTH": "3",
            },
        ):
            config = GroupSGLangConfig.from_env()
            self.assertEqual(config.draft_backend, "sglang")
            self.assertEqual(config.native_draft_cache_tokens, 4096)
            self.assertEqual(config.native_draft_max_requests, 2)
            self.assertTrue(config.native_draft_kv_cache)
            self.assertFalse(config.enable_proposal_cache)
            self.assertEqual(config.max_cached_proposals, 12)
            self.assertIsNone(config.metrics_log_interval)
            self.assertEqual(config.ssd_mode, "async-sync-fallback")
            self.assertEqual(config.ssd_prefetch_workers, 2)
            self.assertEqual(config.ssd_max_prefetch, 32)
            self.assertFalse(config.cuda_overlap)
            self.assertEqual(config.verify_backend, "triton")
            self.assertEqual(config.tokenizer_bridge, "segment")
            self.assertEqual(config.tree_branch_factor, 4)
            self.assertEqual(config.tree_max_depth, 3)


if __name__ == "__main__":
    unittest.main()
