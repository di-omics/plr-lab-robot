import unittest

from plr_lr import Workcell
from plr_lr.qc import arm_readiness


class QCTests(unittest.IsolatedAsyncioTestCase):
  async def test_ready_workcell_passes(self):
    wc = Workcell.sim()
    await wc.setup()
    report = await arm_readiness(wc, hand_eye_rms_mm=0.01)
    self.assertTrue(report.passed, report.summary())
    self.assertEqual(len(report.checks), 4)

  async def test_unhomed_fails_gate(self):
    wc = Workcell.sim()
    await wc.arm.setup()  # deliberately skip home
    report = await arm_readiness(wc)
    self.assertFalse(report.passed)
    homed = next(c for c in report.checks if c.name == "homed")
    self.assertFalse(homed.passed)

  async def test_bad_hand_eye_fails_gate(self):
    wc = Workcell.sim()
    await wc.setup()
    report = await arm_readiness(wc, hand_eye_rms_mm=2.0, rms_threshold_mm=0.5)
    self.assertFalse(report.passed)


if __name__ == "__main__":
  unittest.main()
