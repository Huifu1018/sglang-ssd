import unittest

from sglang_group.sglang.async_proposer import (
    AsyncProposalRequest,
    AsyncProposalService,
)
from sglang_group.sglang.proposer import BaseProposal, SamplingRequest


class FakeProposer:
    def __init__(self):
        self.calls = []
        self.evicted = []
        self.cleared = False

    def propose(
        self,
        rid,
        current_text,
        current_target_ids,
        *,
        max_target_tokens,
        method,
        sampling,
    ):
        self.calls.append((rid, tuple(current_target_ids), method, sampling.temperature))
        return BaseProposal(
            method=method,
            draft_token_ids=(11, 12),
            target_token_ids=tuple(range(100, 100 + max_target_tokens)),
            draft_prob_rows=None,
            cache_event="fake-rebuild",
            draft_context_tokens=len(current_target_ids),
            proposal_cache_event="miss",
        )

    def evict(self, rids):
        self.evicted.extend(str(rid) for rid in rids)

    def clear(self):
        self.cleared = True


def _request(rid="r1", target_ids=(1, 2), method="itl"):
    return AsyncProposalRequest(
        rid=rid,
        current_text="hello",
        current_target_ids=tuple(target_ids),
        max_target_tokens=2,
        method=method,
        sampling=SamplingRequest(temperature=0.0),
    )


class AsyncProposalServiceTests(unittest.TestCase):
    def test_prefetch_then_ready_hit(self):
        proposer = FakeProposer()
        service = AsyncProposalService(proposer, max_workers=1, max_entries=4)
        try:
            request = _request()

            self.assertIsNone(service.get_ready(request))
            self.assertTrue(service.submit(request))
            self.assertFalse(service.submit(request))
            service.drain(timeout_s=1.0)
            self.assertTrue(service.has_ready(request))

            proposal = service.get_ready(request)

            self.assertIsNotNone(proposal)
            self.assertEqual(proposal.proposal_cache_event, "async-hit")
            self.assertEqual(proposal.target_token_ids, (100, 101))
            stats = service.snapshot()
            self.assertEqual(stats["submitted"], 1)
            self.assertEqual(stats["completed"], 1)
            self.assertEqual(stats["submit_skips"], 1)
            self.assertEqual(stats["ready_hits"], 1)
        finally:
            service.shutdown()

    def test_cuda_event_pending_is_not_ready_until_complete(self):
        class FakeCudaOverlap:
            def __init__(self):
                self.ready = False
                self.waits = 0

            def run(self, fn):
                proposal = fn()
                return BaseProposal(
                    proposal.method,
                    proposal.draft_token_ids,
                    proposal.target_token_ids,
                    proposal.draft_prob_rows,
                    proposal.cache_event,
                    proposal.draft_context_tokens,
                    cuda_event=object(),
                    cuda_overlap=True,
                )

            def is_complete(self, proposal):
                return self.ready

            def wait(self, proposal):
                self.waits += 1

        cuda_overlap = FakeCudaOverlap()
        proposer = FakeProposer()
        service = AsyncProposalService(
            proposer,
            max_workers=1,
            max_entries=4,
            cuda_overlap=cuda_overlap,
        )
        try:
            request = _request()
            service.submit(request)
            service.drain(timeout_s=1.0)

            self.assertFalse(service.has_ready(request))
            self.assertIsNone(service.get_ready(request))
            cuda_overlap.ready = True

            proposal = service.get_ready(request)
            self.assertIsNotNone(proposal)
            self.assertTrue(proposal.cuda_overlap)
            self.assertEqual(service.snapshot()["cuda_pending"], 1)
        finally:
            service.shutdown()

    def test_sync_fallback_and_evict(self):
        proposer = FakeProposer()
        service = AsyncProposalService(proposer, max_workers=1, max_entries=4)
        try:
            request = _request(rid="r2", target_ids=(3, 4, 5))

            service.submit(request)
            proposal = service.propose_sync(request)
            service.evict(["r2"])

            self.assertEqual(proposal.target_token_ids, (100, 101))
            self.assertEqual(len(proposer.calls), 1)
            self.assertIn("r2", proposer.evicted)
            stats = service.snapshot()
            self.assertEqual(stats["sync_fallbacks"], 1)
        finally:
            service.shutdown()


if __name__ == "__main__":
    unittest.main()
