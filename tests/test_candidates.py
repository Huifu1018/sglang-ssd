import unittest

from sglang_group.sglang.candidates import (
    build_linear_candidate_rows,
    build_tree_candidate_rows,
)


class CandidateRowTests(unittest.TestCase):
    def test_builds_equal_width_rows(self):
        rows = build_linear_candidate_rows(
            [10, 20],
            [[11, 12, 13], [21]],
            max_draft_token_num=4,
        )
        self.assertEqual(rows.draft_token_num, 2)
        self.assertEqual(rows.rows, ((10, 11), (20, 21)))
        self.assertEqual(rows.proposed_target_tokens, 4)

    def test_rejects_mismatched_lengths(self):
        with self.assertRaises(ValueError):
            build_linear_candidate_rows([1], [[2]], max_draft_token_num=2, draft_prob_rows=[])

    def test_preserves_cache_metadata(self):
        rows = build_linear_candidate_rows(
            [10],
            [[11]],
            max_draft_token_num=2,
            proposal_cache_events=["hit"],
            draft_cache_events=["rebuild"],
            proposal_methods=["itl"],
        )

        self.assertEqual(rows.proposal_cache_events, ("hit",))
        self.assertEqual(rows.draft_cache_events, ("rebuild",))
        self.assertEqual(rows.proposal_methods, ("itl",))

    def test_linear_empty_candidate_rows_skip_verify(self):
        rows = build_linear_candidate_rows(
            [10],
            [[]],
            max_draft_token_num=4,
            proposal_cache_events=["async-miss"],
            draft_cache_events=["async-miss"],
            proposal_methods=["itl-base-slem"],
        )

        self.assertEqual(rows.rows, ())
        self.assertEqual(rows.draft_token_num, 1)
        self.assertEqual(rows.proposed_target_tokens, 0)
        self.assertEqual(rows.proposal_cache_events, ("async-miss",))
        self.assertEqual(rows.draft_cache_events, ("async-miss",))
        self.assertEqual(rows.proposal_methods, ("itl-base-slem",))

    def test_builds_tree_candidate_rows(self):
        rows = build_tree_candidate_rows(
            [5],
            [[7, 8, 9]],
            [[0, 1, 0]],
            max_draft_token_num=4,
        )

        self.assertTrue(rows.is_tree)
        self.assertEqual(rows.rows, ((5, 7, 8, 9),))
        self.assertEqual(rows.parent_rows, ((-1, 0, 1, 0),))
        self.assertEqual(rows.depth_rows, ((0, 1, 2, 1),))

    def test_tree_empty_candidate_rows_skip_verify(self):
        rows = build_tree_candidate_rows(
            [5],
            [[]],
            [[]],
            max_draft_token_num=4,
            proposal_cache_events=["async-miss"],
            draft_cache_events=["async-miss"],
            proposal_methods=["itl-base-slem"],
        )

        self.assertEqual(rows.rows, ())
        self.assertEqual(rows.draft_token_num, 1)
        self.assertEqual(rows.proposed_target_tokens, 0)
        self.assertEqual(rows.proposal_cache_events, ("async-miss",))
        self.assertEqual(rows.draft_cache_events, ("async-miss",))
        self.assertEqual(rows.proposal_methods, ("itl-base-slem",))


if __name__ == "__main__":
    unittest.main()
