import unittest

from sglang_group.sglang.verifier import linear_greedy_verify, tree_greedy_verify


class LinearVerifierTests(unittest.TestCase):
    def test_torch_linear_greedy_accepts_prefix_then_bonus(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        # Two rows, width 3.
        # row0 candidates [5, 7, 8], target predictions [7, 9, 1]:
        #   accept draft token 7, then append target bonus 9.
        # row1 candidates [2, 3, 4], target predictions [0, 4, 5]:
        #   reject immediately, append target bonus 0.
        logits = torch.full((6, 12), -10.0)
        for row, token_id in enumerate([7, 9, 1, 0, 4, 5]):
            logits[row, token_id] = 10.0
        draft_token = torch.tensor([5, 7, 8, 2, 3, 4], dtype=torch.int64)

        result = linear_greedy_verify(
            next_token_logits=logits,
            draft_token=draft_token,
            draft_token_num=3,
            backend="torch",
        )

        self.assertEqual(result.backend, "torch")
        self.assertEqual(result.accept_length.tolist(), [1, 0])
        self.assertEqual(result.accepted_indices.tolist(), [[0, 1, -1], [3, -1, -1]])
        self.assertEqual(result.predict[:6].tolist(), [7, 9, 1, 0, 4, 5])

    def test_torch_tree_greedy_accepts_matching_branch(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        # Tree row:
        #   0(root=5)
        #   ├── 1(token=7) -> 2(token=8)
        #   └── 3(token=9)
        # The target predicts 9 at root, so the verifier should reject the first
        # child branch, accept sibling node 3, then append node 3's target bonus.
        logits = torch.full((4, 12), -10.0)
        for row, token_id in enumerate([9, 8, 1, 4]):
            logits[row, token_id] = 10.0
        draft_token = torch.tensor([5, 7, 8, 9], dtype=torch.int64)
        next_token = torch.tensor([[1, 2, -1, -1]], dtype=torch.int64)
        next_sibling = torch.tensor([[-1, 3, -1, -1]], dtype=torch.int64)

        result = tree_greedy_verify(
            next_token_logits=logits,
            draft_token=draft_token,
            draft_token_num=4,
            retrieve_next_token=next_token,
            retrieve_next_sibling=next_sibling,
            backend="torch",
        )

        self.assertEqual(result.backend, "torch")
        self.assertEqual(result.accept_length.tolist(), [1])
        self.assertEqual(result.accepted_indices.tolist(), [[0, 3, -1, -1]])


if __name__ == "__main__":
    unittest.main()
