"""Lightweight deck world model shared across the arm, vision, and manipulation phases.

The world holds named deck sites, labware instances (with pose, held, and capped
state), and a cap parking table. The arm simulation backend tracks kinematics; the
world tracks what is where. Keeping them separate lets the same world be driven by
the simulation backend today and a hardware backend later without changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional


@dataclass
class Pose2D:
  """A planar pose on the deck: position in mm, yaw in radians."""

  x: float = 0.0
  y: float = 0.0
  z: float = 0.0
  rz: float = 0.0


@dataclass
class Labware:
  """A physical item on the deck.

  Footprint defaults match an SBS microplate. For capped vessels set
  ``cap_diameter``, ``thread_pitch_mm``, and ``cap_turns`` so the decap skill
  knows how far to rotate and lift.
  """

  name: str
  kind: str = "plate"  # plate | tube | microtube
  size_x: float = 127.76
  size_y: float = 85.48
  plate_width: float = 78.0  # gripper width used to hold the plate, mm
  grip_z: float = 8.0  # height above the site origin where the gripper meets the item
  capped: bool = False
  cap_diameter: float = 0.0
  thread_pitch_mm: float = 0.0
  cap_turns: float = 0.0
  pose: Pose2D = field(default_factory=Pose2D)
  held: bool = False


@dataclass
class Site:
  """A named location on the deck that may hold one labware item."""

  name: str
  pose: Pose2D
  occupant: Optional[str] = None  # labware name, or None


class World:
  """Container for sites, labware, and parked caps."""

  def __init__(self) -> None:
    self.sites: Dict[str, Site] = {}
    self.labware: Dict[str, Labware] = {}
    self.parking: Dict[str, object] = {}  # labware name -> where its cap is parked

  def add_site(self, name: str, pose: Pose2D, occupant: Optional[Labware] = None) -> Site:
    site = Site(name=name, pose=pose, occupant=occupant.name if occupant else None)
    self.sites[name] = site
    if occupant is not None:
      occupant.pose = replace(pose)
      self.labware[occupant.name] = occupant
    return site

  def occupant_of(self, site_name: str) -> Optional[Labware]:
    occ = self.sites[site_name].occupant
    return self.labware[occ] if occ else None
