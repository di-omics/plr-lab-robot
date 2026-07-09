"""Uncap and recap threaded vessels: rotate the wrist while tracking the thread.

This is the archetypal dexterous action beyond plate handling. To remove a screw
cap the end-effector grips it, then rotates counter-clockwise while lifting by the
thread pitch per turn; the reverse screws it back on. The motion stays entirely
within the ``ExperimentalSCARA`` API (Cartesian ``move_to`` with a rotating tool
plus gripper open/close), and updates the world's capped state so downstream steps
and QC can check it.
"""

from __future__ import annotations

import math
from typing import Tuple

from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.precise_flex.coords import PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from ..world import World
from .grasp import grasp_for_cap
from .skills import Skill


def _at(coords: PreciseFlexCartesianCoords, z: float, rz: float | None = None) -> PreciseFlexCartesianCoords:
  rot = coords.rotation if rz is None else Rotation(0.0, 0.0, rz)
  return PreciseFlexCartesianCoords(
    location=Coordinate(coords.location.x, coords.location.y, z),
    rotation=rot,
    orientation=coords.orientation,
  )


class DecapSkill(Skill):
  """Unscrew the cap off a vessel and park it."""

  name = "decap"

  def __init__(
    self,
    arm: ExperimentalSCARA,
    world: World,
    labware_name: str,
    coords: PreciseFlexCartesianCoords,
    cap_park: PreciseFlexCartesianCoords,
    degrees_per_step: float = 90.0,
  ) -> None:
    self.arm = arm
    self.world = world
    self.labware_name = labware_name
    self.coords = coords
    self.cap_park = cap_park
    self.degrees_per_step = degrees_per_step

  @property
  def lw(self):
    return self.world.labware[self.labware_name]

  async def precondition(self) -> Tuple[bool, str]:
    lw = self.lw
    if not lw.capped:
      return False, "vessel is already open"
    if lw.held:
      return False, "vessel is held by the arm"
    if lw.cap_turns <= 0 or lw.thread_pitch_mm <= 0 or lw.cap_diameter <= 0:
      return False, "cap thread geometry not specified"
    return True, ""

  async def execute(self) -> dict:
    lw = self.lw
    grasp = grasp_for_cap(lw.cap_diameter)
    top = self.coords.location.z + lw.grip_z

    # approach from above and grip the cap
    await self.arm.move_to(_at(self.coords, z=top + grasp.approach_mm))
    await self.arm.open_gripper(grasp.width_mm + 8.0)
    await self.arm.move_to(_at(self.coords, z=top))
    await self.arm.close_gripper(grasp.width_mm)

    # unscrew: rotate counter-clockwise, lifting by pitch per full turn
    total_deg = lw.cap_turns * 360.0
    applied = 0.0
    while applied < total_deg - 1e-9:
      step = min(self.degrees_per_step, total_deg - applied)
      applied += step
      z = top + (applied / 360.0) * lw.thread_pitch_mm
      await self.arm.move_to(_at(self.coords, z=z, rz=-math.radians(applied)))

    # clear the threads, carry the cap to the parking spot, release
    clear_z = top + lw.thread_pitch_mm * lw.cap_turns + 20.0
    await self.arm.move_to(_at(self.coords, z=clear_z, rz=-math.radians(applied)))
    await self.arm.move_to(_at(self.cap_park, z=self.cap_park.location.z + 40.0))
    await self.arm.move_to(self.cap_park)
    await self.arm.open_gripper(grasp.width_mm + 15.0)

    lw.capped = False
    self.world.parking[self.labware_name] = self.cap_park
    return {
      "turns": lw.cap_turns,
      "rotation_deg": applied,
      "lift_mm": lw.thread_pitch_mm * lw.cap_turns,
    }

  async def verify(self) -> Tuple[bool, str]:
    return (not self.lw.capped), ("cap removed" if not self.lw.capped else "still capped")


class RecapSkill(Skill):
  """Retrieve a parked cap and screw it back on."""

  name = "recap"

  def __init__(
    self,
    arm: ExperimentalSCARA,
    world: World,
    labware_name: str,
    coords: PreciseFlexCartesianCoords,
    degrees_per_step: float = 90.0,
  ) -> None:
    self.arm = arm
    self.world = world
    self.labware_name = labware_name
    self.coords = coords
    self.degrees_per_step = degrees_per_step

  @property
  def lw(self):
    return self.world.labware[self.labware_name]

  async def precondition(self) -> Tuple[bool, str]:
    lw = self.lw
    if lw.capped:
      return False, "vessel is already capped"
    if self.labware_name not in self.world.parking:
      return False, "no parked cap recorded for this vessel"
    return True, ""

  async def execute(self) -> dict:
    lw = self.lw
    grasp = grasp_for_cap(lw.cap_diameter)
    park: PreciseFlexCartesianCoords = self.world.parking[self.labware_name]  # type: ignore[assignment]
    top = self.coords.location.z + lw.grip_z

    # collect the cap from parking
    await self.arm.move_to(_at(park, z=park.location.z + 40.0))
    await self.arm.open_gripper(grasp.width_mm + 8.0)
    await self.arm.move_to(park)
    await self.arm.close_gripper(grasp.width_mm)
    await self.arm.move_to(_at(park, z=park.location.z + 40.0))

    # position above the vessel, engage threads and screw clockwise, lowering
    total_deg = lw.cap_turns * 360.0
    start_z = top + lw.thread_pitch_mm * lw.cap_turns
    await self.arm.move_to(_at(self.coords, z=start_z + 20.0, rz=math.radians(total_deg)))
    applied = total_deg
    await self.arm.move_to(_at(self.coords, z=start_z, rz=math.radians(applied)))
    while applied > 1e-9:
      step = min(self.degrees_per_step, applied)
      applied -= step
      z = top + (applied / 360.0) * lw.thread_pitch_mm
      await self.arm.move_to(_at(self.coords, z=z, rz=math.radians(applied)))
    await self.arm.open_gripper(grasp.width_mm + 15.0)
    await self.arm.move_to(_at(self.coords, z=top + 40.0, rz=0.0))

    lw.capped = True
    self.world.parking.pop(self.labware_name, None)
    return {"turns": lw.cap_turns}

  async def verify(self) -> Tuple[bool, str]:
    return (self.lw.capped), ("cap seated" if self.lw.capped else "not seated")
