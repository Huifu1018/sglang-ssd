import unittest
from types import SimpleNamespace

from sglang_group.sglang.verify_input import (
    _rank_stable_verify_generator,
    _rank_stable_verify_seed,
)


class RankStableVerifySeedTests(unittest.TestCase):
    def _batch(self, *, output_ids=(4, 5)):
        sampling_params = SimpleNamespace(temperature=0.8, top_k=40, top_p=0.95)
        return SimpleNamespace(
            reqs=[
                SimpleNamespace(
                    rid="rid-1",
                    origin_input_ids=[1, 2, 3],
                    output_ids=list(output_ids),
                    sampling_params=sampling_params,
                )
            ]
        )

    def test_verify_seed_is_rank_stable_and_prefix_sensitive(self):
        seed = _rank_stable_verify_seed(batch=self._batch(), draft_token_num=5)

        self.assertEqual(
            seed,
            _rank_stable_verify_seed(batch=self._batch(), draft_token_num=5),
        )
        self.assertNotEqual(
            seed,
            _rank_stable_verify_seed(
                batch=self._batch(output_ids=(4, 6)),
                draft_token_num=5,
            ),
        )

    def test_verify_generator_replays_same_uniforms(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch is not installed")

        batch = self._batch()
        generator_a = _rank_stable_verify_generator(
            batch=batch,
            draft_token_num=5,
            device="cpu",
        )
        generator_b = _rank_stable_verify_generator(
            batch=batch,
            draft_token_num=5,
            device="cpu",
        )

        self.assertTrue(
            torch.equal(
                torch.rand((2, 3), generator=generator_a),
                torch.rand((2, 3), generator=generator_b),
            )
        )


if __name__ == "__main__":
    unittest.main()
