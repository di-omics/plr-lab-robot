"""Arm doubles for the learning suite, so none of it needs a vendor driver.

``plr_lr.learn`` imports no driver. Its tests must not import one either. A
learning package whose suite only collects when one particular vendor backend
happens to be installed has acquired a dependency the library itself refuses,
and the coupling is invisible until someone checks out a tree without that
driver and watches three files fail at import time rather than at assert time.

Four separate things were being borrowed from a driver, and one backend supplies
all four:

* something outside the ``SimulationArmBackend`` hierarchy, because
  :func:`plr_lr.learn.rollout.is_simulation` is an ``isinstance`` check and no
  subclass of the simulation backend can stand on the far side of it. Without a
  double here the hardware permission gate can only be tested by asserting that
  simulation is allowed, which is the half that was never in doubt;
* something that speaks PyLabRobot degrees at the arm boundary while the
  learning layer speaks radians, so the conversions in
  :func:`plr_lr.learn.record.observe` and
  :func:`plr_lr.learn.rollout.apply_action` are measured against a backend that
  disagrees with them about units, rather than against themselves;
* something that reports its gripper opening under a different attribute name
  than the simulation backend uses, so
  :func:`plr_lr.learn.record.gripper_width_of` is shown to duck-type across two
  names. Checked against one name it would pass with the seam hard-coded, and a
  renamed attribute would then silently become a constant column in every
  recorded episode;
* a call recorder, so a rollout the permission gate stopped can be told apart
  from one that reached the arm and commanded motion. "Raised the right error"
  and "raised it before touching anything" are different claims.

Two behaviours here are deliberately stricter than the simulation backend, and
both exist because simulation is too permissive to catch the bug:
:meth:`DegreeReportingBackend.freedrive_mode` refuses a partial axis list, and
it goes compliant in one step and enables in another so the second can be made
to fail. Simulation accepts any axis list and never fails halfway, so against it
alone a session that freed three axes of six, or that left an arm compliant
after a failure, reads exactly like a correct one.

None of this is a controller model and none of it is evidence about any arm. A
result produced against these doubles is a simulated result and must never be
reported as bench-validated.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Union

from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.backend import AccessPattern, SCARABackend
from pylabrobot.arms.precise_flex.coords import ElbowOrientation, PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

#: Where a double parks unless a test says otherwise. Well inside the default
#: workspace on every axis, so a test that fails is failing on the thing it
#: names and not on a start pose that happens to sit on a workspace face.
PARKED_POSE = (0.0, 0.0, 100.0, 0.0)

#: The two modes the freedrive double moves between. Named rather than spelled
#: inline so a test asserting the arm was handed back locked and not compliant
#: says which of the two that is.
MODE_MOTION = "motion"
MODE_COMPLIANT = "compliant"

Position = Union[PreciseFlexCartesianCoords, Dict[int, float]]


class ControllerRefused(RuntimeError):
  """Raised by a double that was asked for a partial freedrive, or armed to fail.

  A distinct type rather than a bare ``RuntimeError`` so a test can assert that
  the failure it arranged is the failure that propagated. Catching
  ``RuntimeError`` would also catch the "backend not set up" and "arm not homed"
  errors, and a test that passes because the wrong thing broke is worse than no
  test.
  """


class DegreeReportingBackend(SCARABackend):
  """A backend outside the simulation hierarchy that touches no hardware.

  Holds its pose in radians internally and exposes it only through
  ``get_cartesian_position``, which is a PyLabRobot ``Rotation`` and therefore
  degrees. ``move_to`` converts back. That makes the unit boundary a genuine
  round trip: a learning-layer radian that survives to the far side of this
  backend unchanged has actually been converted twice, and a test asserting it
  is no longer asserting a conversion against itself. Both representations are
  ordinary floats and a wrongly-converted small rotation still looks plausible,
  so nothing about the number itself would give the mistake away.

  ``_last_commanded_width_mm`` is None until the gripper has been commanded,
  which is the honest answer for a backend that only knows what it was told and
  never measured the jaws. Reporting 0.0 instead would put a constant column
  into every recorded episode, indistinguishable from a real width that never
  changed.

  Every call lands on :attr:`trace`, so a rollout stopped by the permission gate
  can be distinguished from one that got through and commanded motion.
  """

  def __init__(
    self,
    start: Sequence[float] = PARKED_POSE,
    roll_rad: float = 0.0,
    pitch_rad: float = 0.0,
    n_joints: int = 6,
    commanded_width_mm: Optional[float] = None,
  ) -> None:
    super().__init__()
    self._home_joints: Dict[int, float] = {i: 0.0 for i in range(1, int(n_joints) + 1)}
    self._joints: Dict[int, float] = dict(self._home_joints)
    self._x = float(start[0])
    self._y = float(start[1])
    self._z = float(start[2])
    self._yaw_rad = float(start[3])
    self._roll_rad = float(roll_rad)
    self._pitch_rad = float(pitch_rad)
    self._orientation = ElbowOrientation.RIGHT
    # Named unlike the simulation backend's ``_gripper_width`` on purpose: the
    # width seam is meant to be by attribute name across backends that disagree,
    # and two backends sharing one name would prove nothing about the second.
    self._last_commanded_width_mm = (
      None if commanded_width_mm is None else float(commanded_width_mm)
    )
    self._gripper_closed = False
    self._in_freedrive = False
    self.mode = MODE_MOTION
    self.trace: List[dict] = []
    self._armed_failures: Dict[str, int] = {}

  # -- what the double recorded ----------------------------------------------

  @property
  def calls(self) -> List[str]:
    """Just the call names, in order, for assertions about what was reached."""
    return [entry["action"] for entry in self.trace]

  def calls_to(self, action: str) -> List[dict]:
    """Every recorded call named ``action``, with the arguments it carried."""
    return [entry for entry in self.trace if entry["action"] == action]

  def fail(self, step: str, times: int = 1) -> None:
    """Arm :class:`ControllerRefused` on the next ``times`` calls to ``step``.

    Failures are armed rather than raised unconditionally so a test can drive
    the double to a known state first and then break exactly one step. The step
    that matters is the second half of entering freedrive: by then the arm is
    already compliant, and a caller that treats the raise as "freedrive was
    never entered" hands back a physically compliant arm while reporting that
    the capture failed.
    """
    self._armed_failures[step] = self._armed_failures.get(step, 0) + int(times)

  def _maybe_fail(self, step: str) -> None:
    remaining = self._armed_failures.get(step, 0)
    if remaining > 0:
      self._armed_failures[step] = remaining - 1
      raise ControllerRefused(f"{step} refused by an armed test double")

  def _log(self, action: str, **kw: Any) -> None:
    self.trace.append(dict({"action": action}, **kw))

  # -- lifecycle -------------------------------------------------------------

  async def setup(self) -> None:
    self._log("setup")

  async def stop(self) -> None:
    self._log("stop")

  async def home(self) -> None:
    self._joints = dict(self._home_joints)
    self._log("home", joints=dict(self._joints))

  async def move_to_safe(self) -> None:
    self._log("move_to_safe")

  async def halt(self) -> None:
    self._log("halt")

  # -- motion ----------------------------------------------------------------

  async def move_to(self, position: Position) -> None:
    if isinstance(position, dict):
      self._joints.update({int(k): float(v) for k, v in position.items()})
      self._log("move_to_joints", joints=dict(self._joints))
      return
    location = position.location
    self._x = float(location.x)
    self._y = float(location.y)
    self._z = float(location.z)
    # Inbound half of the unit boundary. PyLabRobot hands over degrees; this
    # backend keeps radians, so the value a test reads back off the trace is the
    # one that survived both conversions rather than the one that skipped them.
    self._roll_rad = math.radians(float(position.rotation.x))
    self._pitch_rad = math.radians(float(position.rotation.y))
    self._yaw_rad = math.radians(float(position.rotation.z))
    self._orientation = position.orientation
    self._log(
      "move_to",
      x=self._x,
      y=self._y,
      z=self._z,
      roll_rad=self._roll_rad,
      pitch_rad=self._pitch_rad,
      yaw_rad=self._yaw_rad,
    )

  async def approach(self, position: Position, access: Optional[AccessPattern] = None) -> None:
    self._log("approach")

  async def pick_up_resource(
    self,
    position: Position,
    plate_width: float,
    access: Optional[AccessPattern] = None,
  ) -> None:
    self._log("pick_up_resource", plate_width=float(plate_width))

  async def drop_resource(self, position: Position, access: Optional[AccessPattern] = None) -> None:
    self._log("drop_resource")

  # -- gripper ---------------------------------------------------------------

  async def open_gripper(self, gripper_width: float) -> None:
    self._last_commanded_width_mm = float(gripper_width)
    self._gripper_closed = False
    self._log("open_gripper", width=float(gripper_width))

  async def close_gripper(self, gripper_width: float) -> None:
    self._last_commanded_width_mm = float(gripper_width)
    self._gripper_closed = True
    self._log("close_gripper", width=float(gripper_width))

  async def is_gripper_closed(self) -> bool:
    self._log("is_gripper_closed")
    return self._gripper_closed

  # -- state readouts --------------------------------------------------------

  async def get_joint_position(self) -> Dict[int, float]:
    self._log("get_joint_position")
    return dict(self._joints)

  async def get_cartesian_position(self) -> PreciseFlexCartesianCoords:
    self._log("get_cartesian_position")
    # Outbound half of the unit boundary: a PyLabRobot Rotation is degrees.
    return PreciseFlexCartesianCoords(
      location=Coordinate(self._x, self._y, self._z),
      rotation=Rotation(
        math.degrees(self._roll_rad),
        math.degrees(self._pitch_rad),
        math.degrees(self._yaw_rad),
      ),
      orientation=self._orientation,
    )

  # -- freedrive -------------------------------------------------------------

  def _set_mode(self, mode: str) -> None:
    self.mode = mode
    self._log("set_mode", mode=mode)

  async def freedrive_mode(self, free_axes: List[int]) -> None:
    """Go compliant, then enable, refusing anything short of the whole arm.

    Both halves are here because the simulation backend has neither. It accepts
    any axis list, so a session that asked for three axes of six reads as
    correct against it; and it cannot fail between the two calls, so a caller
    that leaves an arm compliant after a half-entered freedrive also reads as
    correct. Freeing a subset is refused rather than partially honoured, because
    an arm that is compliant on some axes and locked on others is not the arm
    the caller was told it had hold of.
    """
    axes = [int(axis) for axis in free_axes]
    whole_arm = sorted(self._home_joints)
    if sorted(axes) != whole_arm:
      raise ControllerRefused(
        f"freedrive is all or nothing here: asked to free {axes}, which is not "
        f"the whole arm {whole_arm}. A partial list would leave some axes "
        "compliant and some locked while reporting that the request was honoured."
      )
    self._set_mode(MODE_COMPLIANT)
    self._maybe_fail("enable_freedrive")
    self._log("enable_freedrive")
    self._in_freedrive = True
    self._log("freedrive", axes=axes, whole_arm=True)

  async def end_freedrive_mode(self) -> None:
    self._set_mode(MODE_MOTION)
    self._in_freedrive = False
    self._log("end_freedrive")


async def homed_arm(backend: SCARABackend) -> ExperimentalSCARA:
  """A set-up and homed arm on ``backend``, with no hardware in the room.

  Deliberately not defaulted to a double: the caller names the backend, so a
  test that means to run against something outside the simulation hierarchy
  cannot end up quietly running against a simulated arm and passing for the
  wrong reason.
  """
  arm = ExperimentalSCARA(backend=backend)
  await arm.setup()
  await arm.home()
  return arm
