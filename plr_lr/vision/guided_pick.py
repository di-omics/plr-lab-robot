"""Close the loop: use a detection to correct a taught pick before gripping.

A taught site is only as good as the last calibration; labware placed by hand
drifts. Here the wrist camera looks at the nominal site, the detector reports the
labware in the image, we back-project that pixel onto the known deck plane to get
its true base-frame position, and we shift the pick target by the measured offset.
This turns an open-loop pick into a vision-guided one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.backend import AccessPattern
from pylabrobot.arms.precise_flex.coords import PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate

from .camera import CameraIntrinsics
from .detector import Detection
from .transforms import Transform


def pixel_to_plane(
  uv: Tuple[float, float],
  intr: CameraIntrinsics,
  base_from_cam: Transform,
  plane_z: float,
) -> np.ndarray:
  """Intersect the back-projected ray with the deck plane ``z = plane_z``."""
  d_base = base_from_cam.R @ intr.ray(uv)
  o_base = base_from_cam.t
  if abs(d_base[2]) < 1e-12:
    raise ValueError("view ray is parallel to the deck plane")
  s = (plane_z - o_base[2]) / d_base[2]
  return o_base + s * d_base


@dataclass
class PickCorrection:
  measured_xy: Tuple[float, float]
  taught_xy: Tuple[float, float]
  dx: float
  dy: float

  @property
  def magnitude(self) -> float:
    return float(np.hypot(self.dx, self.dy))


def estimate_offset(
  detection: Detection,
  intr: CameraIntrinsics,
  base_from_cam: Transform,
  taught: PreciseFlexCartesianCoords,
  plane_z: float,
) -> PickCorrection:
  """Measure how far the detected labware is from its taught position."""
  p = pixel_to_plane(detection.uv, intr, base_from_cam, plane_z)
  tx, ty = taught.location.x, taught.location.y
  return PickCorrection(
    measured_xy=(float(p[0]), float(p[1])),
    taught_xy=(float(tx), float(ty)),
    dx=float(p[0] - tx),
    dy=float(p[1] - ty),
  )


def apply_correction(
  taught: PreciseFlexCartesianCoords, corr: PickCorrection
) -> PreciseFlexCartesianCoords:
  loc = taught.location
  return PreciseFlexCartesianCoords(
    location=Coordinate(loc.x + corr.dx, loc.y + corr.dy, loc.z),
    rotation=taught.rotation,
    orientation=taught.orientation,
  )


async def vision_guided_pick(
  arm: ExperimentalSCARA,
  detection: Detection,
  intr: CameraIntrinsics,
  base_from_cam: Transform,
  taught: PreciseFlexCartesianCoords,
  plane_z: float,
  plate_width: float,
  access: Optional[AccessPattern] = None,
) -> PickCorrection:
  """Correct the taught pose from a detection, then pick. Returns the correction."""
  corr = estimate_offset(detection, intr, base_from_cam, taught, plane_z)
  target = apply_correction(taught, corr)
  await arm.pick_up_resource(target, plate_width=plate_width, access=access)
  return corr
