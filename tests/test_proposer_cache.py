import unittest
from collections import OrderedDict
from types import MethodType

from sglang_group.sglang.config import GroupSGLangConfig
from sglang_group.sglang.proposer import (
    BaseProposal,
    DraftProposerStats,
    HeterogeneousDraftProposer,
    SamplingRequest,
    _clone_cache,
)


class TinyTokenizer:
    def __init__(self, vocab):
        self.vocab = dict(vocab)
        self.id_to_text = {idx: token for token, idx in self.vocab.items()}

    def decode(self, ids, **kwargs):
        return "".join(self.id_to_text[int(idx)] for idx in ids)

    def encode(self, text, add_special_tokens=False):
        ids = []
        cursor = 0
        pieces = sorted(
            ((token, idx) for token, idx in self.vocab.items()),
            key=lambda item: len(item[0]),
            reverse=True,
        )
        while cursor < len(text):
            for piece, idx in pieces:
                if text.startswith(piece, cursor):
                    ids.append(idx)
                    cursor += len(piece)
                    break
            else:
                cursor += 1
        return ids

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}


class ProposerCacheTests(unittest.TestCase):
    def test_hf_cache_object_is_cloned_not_reused(self):
        class FakeCache:
            def __init__(self, values=None):
                self.values = list(values or [1])

            def get_seq_length(self):
                return len(self.values)

            def to_legacy_cache(self):
                return tuple(self.values)

        cache = FakeCache()
        cloned = _clone_cache(cache)

        cache.values.append(2)

        self.assertIsInstance(cloned, FakeCache)
        self.assertIsNot(cloned, cache)
        self.assertEqual(cloned.get_seq_length(), 1)

    def test_legacy_cache_fallback_preserves_cache_api(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        class FakeCache:
            def __init__(self, legacy_cache=None):
                self.legacy_cache = legacy_cache
                self.source_tensor = torch.tensor([1])

            def __deepcopy__(self, memo):
                raise TypeError("force legacy fallback")

            def get_seq_length(self):
                return 1

            def to_legacy_cache(self):
                return ((self.source_tensor,),)

            @classmethod
            def from_legacy_cache(cls, legacy_cache):
                return cls(legacy_cache=legacy_cache)

        cache = FakeCache()
        cloned = _clone_cache(cache)

        self.assertIsInstance(cloned, FakeCache)
        self.assertEqual(cloned.get_seq_length(), 1)
        self.assertIsNot(cloned.legacy_cache[0][0], cache.source_tensor)
        self.assertEqual(int(cloned.legacy_cache[0][0][0].item()), 1)


class ProposalResultCacheTests(unittest.TestCase):
    def _new_proposer(self, config: GroupSGLangConfig | None = None):
        proposer = object.__new__(HeterogeneousDraftProposer)
        proposer.config = config or GroupSGLangConfig()
        proposer._proposal_cache = OrderedDict()
        proposer._states = OrderedDict()
        proposer.native_backend = None
        proposer.stats = DraftProposerStats()
        proposer._context_ids = MethodType(lambda self, text: (10, 20, 30), proposer)
        return proposer

    def test_slem_proposal_cache_hits_same_context(self):
        proposer = self._new_proposer()
        calls = {"count": 0}

        def fake_slem(
            self,
            rid,
            current_text,
            current_target_ids,
            *,
            context_ids=None,
            max_target_tokens,
        ):
            calls["count"] += 1
            return BaseProposal(
                "itl-base-slem",
                (101, 102),
                (201, 202),
                None,
                "rebuild",
                len(context_ids or ()),
            )

        proposer._propose_slem = MethodType(fake_slem, proposer)

        first = proposer.propose(
            "r0",
            "hello",
            (1, 2, 3),
            max_target_tokens=2,
            method="itl-base-slem",
            sampling=SamplingRequest(temperature=0.0),
        )
        second = proposer.propose(
            "r0",
            "hello",
            (1, 2, 3),
            max_target_tokens=2,
            method="itl-base-slem",
            sampling=SamplingRequest(temperature=0.0),
        )

        self.assertEqual(calls["count"], 1)
        self.assertEqual(first.proposal_cache_event, "miss")
        self.assertEqual(second.proposal_cache_event, "hit")
        self.assertEqual(second.target_token_ids, (201, 202))
        self.assertEqual(proposer.stats.proposal_cache_hits, 1)
        self.assertEqual(proposer.proposal_cache_size(), 1)

    def test_proposal_cache_skips_tli(self):
        proposer = self._new_proposer()
        calls = {"count": 0}

        def fake_tli(
            self,
            rid,
            current_text,
            *,
            context_ids=None,
            max_target_tokens,
            sampling,
        ):
            calls["count"] += 1
            return BaseProposal(
                "itl-base-tli",
                (101,),
                (201,),
                ["prob-row"],
                "rebuild",
                len(context_ids or ()),
            )

        proposer._propose_tli = MethodType(fake_tli, proposer)

        for _ in range(2):
            proposer.propose(
                "r0",
                "hello",
                (1, 2, 3),
                max_target_tokens=1,
                method="itl-base-tli",
                sampling=SamplingRequest(temperature=0.7),
            )

        self.assertEqual(calls["count"], 2)
        self.assertEqual(proposer.stats.proposal_cache_hits, 0)
        self.assertEqual(proposer.stats.proposal_cache_skips, 2)
        self.assertEqual(proposer.proposal_cache_size(), 0)

    def test_evict_removes_matching_proposal_cache_entries(self):
        proposer = self._new_proposer(GroupSGLangConfig(max_cached_proposals=4))

        def fake_itl(
            self,
            rid,
            current_text,
            current_target_ids,
            *,
            context_ids=None,
            max_target_tokens,
        ):
            return BaseProposal(
                "itl",
                (101,),
                (201,),
                None,
                "rebuild",
                len(context_ids or ()),
            )

        proposer._propose_itl = MethodType(fake_itl, proposer)
        proposer.propose("r0", "hello", (1,), max_target_tokens=1, method="itl")
        proposer.propose("r1", "hello", (1,), max_target_tokens=1, method="itl")

        proposer.evict(["r0"])

        self.assertEqual(proposer.proposal_cache_size(), 1)
        self.assertEqual(proposer.stats.proposal_cache_evictions, 1)

    def test_tokenizer_bridge_uses_uag_lookbehind_by_default(self):
        proposer = self._new_proposer(GroupSGLangConfig(tokenizer_bridge="uag"))
        proposer.target_tokenizer = TinyTokenizer({"hello": 1, " world": 2, "!": 3})
        proposer.draft_tokenizer = TinyTokenizer({"hello": 10, " world": 11, "!": 12})

        target_ids = proposer._bridge_assistant_tokens_to_target(
            current_target_ids=(1,),
            assistant_context_ids=(10,),
            assistant_new_ids=(11, 12),
        )

        self.assertEqual(target_ids, (2, 3))
        self.assertEqual(proposer.stats.bridge_uag_translations, 1)

    def test_tokenizer_bridge_segment_mode_retokenizes_new_text_only(self):
        proposer = self._new_proposer(GroupSGLangConfig(tokenizer_bridge="segment"))
        proposer.target_tokenizer = TinyTokenizer({"hello": 1, " world": 2, "!": 3})
        proposer.draft_tokenizer = TinyTokenizer({"hello": 10, " world": 11, "!": 12})

        target_ids = proposer._bridge_assistant_tokens_to_target(
            current_target_ids=(1,),
            assistant_context_ids=(10,),
            assistant_new_ids=(11, 12),
        )

        self.assertEqual(target_ids, (2, 3))
        self.assertEqual(proposer.stats.bridge_segment_translations, 1)

    def test_tree_builder_adds_root_sibling_from_topk(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        proposer = self._new_proposer(GroupSGLangConfig(tree_branch_factor=2))
        proposer.target_tokenizer = TinyTokenizer({"hello": 1, " world": 2, "!": 3})
        proposer.draft_tokenizer = TinyTokenizer({"hello": 10, " world": 11, "!": 12})
        logits = torch.zeros((1, 16))
        logits[0, 11] = 10
        logits[0, 12] = 9

        tokens, parents = proposer._tree_from_greedy_with_root_siblings(
            root_logits=logits,
            greedy_target_ids=(2,),
            current_target_ids=(1,),
            assistant_context_ids=(10,),
            max_target_tokens=3,
        )

        self.assertEqual(tokens, (2, 3))
        self.assertEqual(parents, (0, 0))
        self.assertEqual(proposer.stats.tree_proposals, 1)

    def test_tree_builder_reserves_budget_for_root_siblings(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        proposer = self._new_proposer(
            GroupSGLangConfig(tree_branch_factor=3, tree_max_depth=3)
        )
        proposer.target_tokenizer = TinyTokenizer(
            {"hello": 1, " world": 2, "!": 3, "?": 4}
        )
        proposer.draft_tokenizer = TinyTokenizer(
            {"hello": 10, " world": 11, "!": 12, "?": 13}
        )
        logits = torch.zeros((1, 16))
        logits[0, 11] = 10
        logits[0, 12] = 9
        logits[0, 13] = 8

        tokens, parents = proposer._tree_from_greedy_with_root_siblings(
            root_logits=logits,
            greedy_target_ids=(2, 2, 2, 2),
            current_target_ids=(1,),
            assistant_context_ids=(10,),
            max_target_tokens=4,
        )

        self.assertEqual(tokens, (2, 2, 3, 4))
        self.assertEqual(parents, (0, 1, 0, 0))
        self.assertEqual(proposer.stats.tree_proposals, 1)


if __name__ == "__main__":
    unittest.main()
