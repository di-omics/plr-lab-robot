"""A workcell binds an arm to named, taught deck sites and moves plates between them.

The workcell is the unit you script against in phase 1. It owns an
``ExperimentalSCARA`` and a ``World``, remembers the Cartesian pose taught for
each site, and exposes ``move_plate`` on top of the arm's pick and drop.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Optional

from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.backend import AccessPattern
from pylabrobot.arms.precise_flex.coords import ElbowOrientation, PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from ..world import Labware, Pose2D, World
from .sim_backend import SimulationArmBackend


def site_coords(pose: Pose2D) -> PreciseFlexCartesianCoords:
  """Convert a deck pose to arm Cartesian coordinates (right-elbow by default)."""
  return PreciseFlexCartesianCoords(
    location=Coordinate(pose.x, pose.y, pose.z),
    rotation=Rotation(0.0, 0.0, pose.rz),
    orientation=ElbowOrientation.RIGHT,
  )


class Workcell:
  """An arm plus a taught deck it can move labware around."""

  def __init__(self, arm: ExperimentalSCARA, world: Optional[World] = None) -> None:
    self.arm = arm
    self.world = world or World()
    self.taught: Dict[str, PreciseFlexCartesianCoords] = {}

  @classmethod
  def sim(cls, **backend_kwargs) -> "Workcell":
    """Build a ready-to-run simulated workcell (no hardware)."""
    backend = SimulationArmBackend(**backend_kwargs)
    return cls(arm=ExperimentalSCARA(backend=backend))

  @property
  def backend(self) -> SimulationArmBackend:
    return self.arm.backend  # type: ignore[return-value]

  async def setup(self) -> None:
    await self.arm.setup()
    await self.arm.home()

  def add_site(
    self,
    name: str,
    x: float,
    y: float,
    z: float = 0.0,
    rz: float = 0.0,
    occupant: Optional[Labware] = None,
  ) -> str:
    pose = Pose2D(x, y, z, rz)
    self.world.add_site(name, pose, occupant=occupant)
    self.taught[name] = site_coords(pose)
    return name

  def coords_for(self, site: str) -> PreciseFlexCartesianCoords:
    return self.taught[site]

  async def move_plate(
    self,
    src: str,
    dst: str,
    access: Optional[AccessPattern] = None,
  ) -> str:
    """Move the labware occupying ``src`` to the empty site ``dst``."""
    lw = self.world.occupant_of(src)
    if lw is None:
      raise ValueError(f"No labware at site {src!r}")
    if self.world.sites[dst].occupant is not None:
      raise ValueError(f"Destination site {dst!r} is occupied")

    await self.arm.pick_up_resource(self.taught[src], plate_width=lw.plate_width, access=access)
    lw.held = True
    await self.arm.drop_resource(self.taught[dst], access=access)
    lw.held = False

    self.world.sites[src].occupant = None
    self.world.sites[dst].occupant = lw.name
    lw.pose = replace(self.world.sites[dst].pose)
    return lw.name
