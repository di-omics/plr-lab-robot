"""A simulation backend for a plate-handling robotic arm.

PyLabRobot ships an arm front end (``ExperimentalSCARA``) and a hardware driver
(PreciseFlex) but no simulation backend, so you cannot exercise the arm API
without a physical robot. ``SimulationArmBackend`` fills that gap: it implements
the full ``SCARABackend`` interface, tracks joint and Cartesian state, gripper
state, and whether a plate is held, enforces a workspace envelope and a home
interlock, and records every command in a trace for validation and QC.

Drop it straight into the same code paths you would run on hardware::

    from pylabrobot.arms import ExperimentalSCARA
    from plr_lr.arm import SimulationArmBackend

    arm = ExperimentalSCARA(backend=SimulationArmBackend())
    await arm.setup()
    await arm.home()
    await arm.pick_up_resource(coords, plate_width=78.0)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from pylabrobot.arms.backend import (
  AccessPattern,
  HorizontalAccess,
  SCARABackend,
  VerticalAccess,
)
from pylabrobot.arms.precise_flex.coords import ElbowOrientation, PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation


class ArmNotHomedError(RuntimeError):
  """Raised when a motion is commanded before the arm has been homed."""


class GripperError(RuntimeError):
  """Raised on an invalid gripper or pick/drop sequence."""


class WorkspaceViolation(RuntimeError):
  """Raised when a commanded target falls outside the workspace envelope."""


@dataclass
class Workspace:
  """Axis-aligned motion envelope in the base frame, in mm."""

  x: Tuple[float, float] = (-500.0, 500.0)
  y: Tuple[float, float] = (-500.0, 500.0)
  z: Tuple[float, float] = (0.0, 400.0)

  def contains(self, c: Coordinate) -> bool:
    return (
      self.x[0] <= c.x <= self.x[1]
      and self.y[0] <= c.y <= self.y[1]
      and self.z[0] <= c.z <= self.z[1]
    )


Position = Union[PreciseFlexCartesianCoords, Dict[int, float]]


def _lift(position: PreciseFlexCartesianCoords, dz: float) -> PreciseFlexCartesianCoords:
  loc = position.location
  return PreciseFlexCartesianCoords(
    location=Coordinate(loc.x, loc.y, loc.z + dz),
    rotation=position.rotation,
    orientation=position.orientation,
  )


class SimulationArmBackend(SCARABackend):
  """Kinematic simulation backend implementing the full ``SCARABackend`` API."""

  N_JOINTS = 5

  def __init__(
    self,
    workspace: Optional[Workspace] = None,
    motion_latency: float = 0.0,
    home_joints: Optional[Dict[int, float]] = None,
  ) -> None:
    super().__init__()
    self.workspace = workspace or Workspace()
    self.motion_latency = motion_latency
    self._home_joints = home_joints or {i: 0.0 for i in range(1, self.N_JOINTS + 1)}
    self._joints: Dict[int, float] = dict(self._home_joints)
    z_top = self.workspace.z[1]
    self._cartesian = PreciseFlexCartesianCoords(
      location=Coordinate(0.0, 0.0, z_top),
      rotation=Rotation(0.0, 0.0, 0.0),
      orientation=ElbowOrientation.RIGHT,
    )
    self.safe_cartesian = PreciseFlexCartesianCoords(
      location=Coordinate(0.0, 0.0, z_top),
      rotation=Rotation(0.0, 0.0, 0.0),
      orientation=ElbowOrientation.RIGHT,
    )
    self._gripper_width = 90.0
    self._gripper_closed = False
    self._holding = False
    self._setup_done = False
    self._homed = False
    self._in_freedrive = False
    self.trace: List[dict] = []

  # -- introspection ---------------------------------------------------------

  @property
  def homed(self) -> bool:
    return self._homed

  @property
  def holding(self) -> bool:
    return self._holding

  @property
  def is_set_up(self) -> bool:
    return self._setup_done

  def _log(self, action: str, **kw) -> None:
    self.trace.append({"action": action, **kw})

  async def _dwell(self) -> None:
    if self.motion_latency:
      await asyncio.sleep(self.motion_latency)

  def _require_setup(self) -> None:
    if not self._setup_done:
      raise RuntimeError("Backend not set up; call setup() first")

  def _require_homed(self) -> None:
    self._require_setup()
    if not self._homed:
      raise ArmNotHomedError("Arm must be homed before commanding a target")

  def _check_workspace(self, c: Coordinate) -> None:
    if not self.workspace.contains(c):
      raise WorkspaceViolation(f"Target {c} is outside workspace {self.workspace}")

  # -- lifecycle -------------------------------------------------------------

  async def setup(self) -> None:
    self._setup_done = True
    self._log("setup")

  async def stop(self) -> None:
    self._setup_done = False
    self._homed = False
    self._log("stop")

  async def home(self) -> None:
    self._require_setup()
    self._joints = dict(self._home_joints)
    self._homed = True
    self._log("home", joints=dict(self._joints))

  async def move_to_safe(self) -> None:
    self._require_homed()
    await self._move_cartesian(self.safe_cartesian, tag="move_to_safe")

  # -- motion ----------------------------------------------------------------

  async def _move_cartesian(self, coords: PreciseFlexCartesianCoords, tag: str) -> None:
    self._check_workspace(coords.location)
    await self._dwell()
    self._cartesian = coords
    self._log(
      tag,
      x=coords.location.x,
      y=coords.location.y,
      z=coords.location.z,
      rz=coords.rotation.z,
    )

  async def move_to(self, position: Position) -> None:
    self._require_homed()
    if isinstance(position, dict):
      self._joints.update({int(k): float(v) for k, v in position.items()})
      await self._dwell()
      self._log("move_to_joints", joints=dict(self._joints))
    else:
      await self._move_cartesian(position, tag="move_to")

  @staticmethod
  def _clearance(access: AccessPattern) -> float:
    if isinstance(access, VerticalAccess):
      return access.clearance_mm
    return access.lift_height_mm

  @staticmethod
  def _approach_height(access: AccessPattern) -> float:
    if isinstance(access, VerticalAccess):
      return access.approach_height_mm
    return access.lift_height_mm

  async def approach(self, position: Position, access: Optional[AccessPattern] = None) -> None:
    self._require_homed()
    if isinstance(position, dict):
      await self.move_to(position)
      return
    access = access or VerticalAccess()
    await self._move_cartesian(_lift(position, self._approach_height(access)), tag="approach")

  async def pick_up_resource(
    self,
    position: Position,
    plate_width: float,
    access: Optional[AccessPattern] = None,
  ) -> None:
    self._require_homed()
    if self._holding:
      raise GripperError("Arm is already holding a resource")
    if isinstance(position, dict):
      raise GripperError("pick_up_resource requires Cartesian coordinates in simulation")
    access = access or VerticalAccess()
    await self.open_gripper(max(plate_width + 15.0, self._gripper_width))
    await self.approach(position, access=access)
    await self._move_cartesian(position, tag="descend_pick")
    await self.close_gripper(plate_width)
    self._holding = True
    await self._move_cartesian(_lift(position, self._clearance(access)), tag="retract_pick")
    self._log("pick_up_resource", plate_width=plate_width, held=True)

  async def drop_resource(
    self,
    position: Position,
    access: Optional[AccessPattern] = None,
  ) -> None:
    self._require_homed()
    if not self._holding:
      raise GripperError("Arm is not holding a resource")
    if isinstance(position, dict):
      raise GripperError("drop_resource requires Cartesian coordinates in simulation")
    access = access or VerticalAccess()
    await self.approach(position, access=access)
    await self._move_cartesian(position, tag="descend_drop")
    await self.open_gripper(self._gripper_width)
    self._holding = False
    await self._move_cartesian(_lift(position, self._clearance(access)), tag="retract_drop")
    self._log("drop_resource", held=False)

  # -- gripper ---------------------------------------------------------------

  async def open_gripper(self, gripper_width: float) -> None:
    self._gripper_width = gripper_width
    self._gripper_closed = False
    self._log("open_gripper", width=gripper_width)

  async def close_gripper(self, gripper_width: float) -> None:
    self._gripper_width = gripper_width
    self._gripper_closed = True
    self._log("close_gripper", width=gripper_width)

  async def is_gripper_closed(self) -> bool:
    return self._gripper_closed

  async def halt(self) -> None:
    self._log("halt")

  # -- state readouts --------------------------------------------------------

  async def get_joint_position(self) -> Dict[int, float]:
    return dict(self._joints)

  async def get_cartesian_position(self) -> PreciseFlexCartesianCoords:
    return self._cartesian

  # -- freedrive -------------------------------------------------------------

  async def freedrive_mode(self, free_axes: List[int]) -> None:
    self._in_freedrive = True
    self._log("freedrive", axes=list(free_axes))

  async def end_freedrive_mode(self) -> None:
    self._in_freedrive = False
    self._log("end_freedrive")
