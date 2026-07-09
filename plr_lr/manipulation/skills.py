"""A small skill framework: precondition, execute, verify.

Phase 3 goes beyond moving plates to dexterous end-effector actions (uncapping
threaded vessels, and later other manipulations). Every skill is gated: it checks
a precondition, runs, then verifies its own effect before reporting success. That
QC-around-every-action shape is deliberate, so a skill can be dropped into a
closed loop and trusted to say whether it actually worked.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class SkillResult:
  ok: bool
  skill: str
  message: str = ""
  data: dict = field(default_factory=dict)

  def __bool__(self) -> bool:
    return self.ok


class Skill(ABC):
  """Base class for a verifiable manipulation skill."""

  name: str = "skill"

  @abstractmethod
  async def precondition(self) -> Tuple[bool, str]:
    """Return (ok, reason) for whether the skill may run now."""

  @abstractmethod
  async def execute(self) -> dict:
    """Perform the motion. Return telemetry to attach to the result."""

  @abstractmethod
  async def verify(self) -> Tuple[bool, str]:
    """Return (ok, reason) for whether the intended effect was achieved."""

  async def run(self) -> SkillResult:
    ok, reason = await self.precondition()
    if not ok:
      return SkillResult(False, self.name, f"precondition failed: {reason}")
    data = await self.execute()
    ok, reason = await self.verify()
    return SkillResult(ok, self.name, reason or ("verified" if ok else "verification failed"), data)
