import unittest
from importlib.util import find_spec

if find_spec("torch") is None or find_spec("sglang") is None:
    raise unittest.SkipTest("torch or sglang is not installed")
from sglang_group.sglang.worker import _NativeForwardGate


class FakeEvent:
    def __init__(self, events):
        self.events = events
        self.synchronized = False

    def record(self, stream):
        self.events.append(("record", stream.name))

    def synchronize(self):
        self.synchronized = True
        self.events.append(("sync", None))


class FakeStream:
    def __init__(self, events):
        self.events = events
        self.name = "stream"

    def wait_event(self, event):
        self.events.append(("wait", event))


class FakeCuda:
    def __init__(self):
        self.events = []
        self.stream = FakeStream(self.events)

    def is_available(self):
        return True

    def current_stream(self):
        return self.stream

    def Event(self, *, enable_timing):
        self.events.append(("event", enable_timing))
        return FakeEvent(self.events)


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


class NativeForwardGateTests(unittest.TestCase):
    def test_orders_next_forward_after_previous_cuda_event(self):
        gate = _NativeForwardGate()
        fake_torch = FakeTorch()
        gate._load_torch = lambda: fake_torch

        with gate.draft_context():
            pass
        first_event = gate._last_cuda_event

        with gate.target_context():
            pass

        self.assertIsNotNone(first_event)
        self.assertIn(("wait", first_event), fake_torch.cuda.events)
        self.assertGreaterEqual(
            sum(1 for event in fake_torch.cuda.events if event[0] == "record"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
