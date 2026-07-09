"""Readiness QC: a single gate that says whether the workcell is fit to run.

Structured like the wet-lab readiness checks used elsewhere in di-omics (a set of
named pass/fail checks with an overall gate), so the same reporting shape carries
from simulation into hardware bring-up. In simulation the checks cover setup,
homing, a gripper round trip, and (optionally) the hand-eye calibration residual.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Check:
  name: str
  passed: bool
  detail: str = ""


@dataclass
class ReadinessReport:
  checks: List[Check] = field(default_factory=list)

  @property
  def passed(self) -> bool:
    return all(c.passed for c in self.checks)

  def summary(self) -> str:
    lines = [f"readiness: {'PASS' if self.passed else 'FAIL'}"]
    for c in self.checks:
      mark = "ok " if c.passed else "XX "
      lines.append(f"  [{mark}] {c.name}: {c.detail}")
    return "\n".join(lines)


async def arm_readiness(
  workcell,
  hand_eye_rms_mm: Optional[float] = None,
  rms_threshold_mm: float = 0.5,
) -> ReadinessReport:
  """Run the readiness checks against a (simulated or real) workcell."""
  backend = workcell.backend
  checks: List[Check] = [
    Check("setup", backend.is_set_up, "backend connected"),
    Check("homed", backend.homed, "arm homed to reference"),
  ]

  await workcell.arm.open_gripper(90.0)
  opened = not await workcell.arm.is_gripper_closed()
  await workcell.arm.close_gripper(78.0)
  closed = await workcell.arm.is_gripper_closed()
  await workcell.arm.open_gripper(90.0)
  checks.append(Check("gripper", opened and closed, "open/close round trip"))

  if hand_eye_rms_mm is not None:
    ok = hand_eye_rms_mm <= rms_threshold_mm
    checks.append(
      Check("hand_eye", ok, f"rms {hand_eye_rms_mm:.4f} mm <= {rms_threshold_mm} mm")
    )

  return ReadinessReport(checks)
