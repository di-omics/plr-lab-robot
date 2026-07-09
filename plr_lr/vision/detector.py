"""Object detection interface plus a ground-truth-backed simulation detector.

``Detector`` is the seam a real model plugs into. ``SimDetector`` projects known
labware positions through the eye-in-hand camera model and returns pixel
detections (with optional pixel noise), which lets the vision loop be validated
end to end on CPU with no weights. ``RFDetrDetector`` and ``SAM2Detector`` are
adapter stubs that name the real SOTA path (see the lab-cv and awesome-wetlab-cv
work); they raise until the optional ``vision`` extra and weights are provided.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from .camera import CameraIntrinsics
from .transforms import Transform


@dataclass
class Detection:
  """A single detected object in image space."""

  label: str
  uv: Tuple[float, float]
  confidence: float = 1.0
  truth_name: Optional[str] = None  # ground-truth association, sim only


@runtime_checkable
class Detector(Protocol):
  def detect(self, image=None, context=None) -> List[Detection]:
    ...


class SimDetector:
  """Projects known world points through the camera to synthesize detections."""

  def __init__(
    self,
    intrinsics: CameraIntrinsics,
    pixel_noise: float = 0.0,
    seed: Optional[int] = None,
  ) -> None:
    self.intr = intrinsics
    self.pixel_noise = pixel_noise
    self.rng = np.random.default_rng(seed)

  def observe(
    self,
    world_points: Dict[str, np.ndarray],
    base_from_cam: Transform,
    label: str = "labware",
  ) -> List[Detection]:
    """Return detections for the given base-frame points seen from ``base_from_cam``."""
    cam_from_base = base_from_cam.inv()
    dets: List[Detection] = []
    for name, p in world_points.items():
      p_cam = cam_from_base.apply(np.asarray(p, float))
      if p_cam[2] <= 0:
        continue  # behind the camera
      uv = self.intr.project(p_cam)
      if 0 <= uv[0] <= self.intr.width and 0 <= uv[1] <= self.intr.height:
        if self.pixel_noise:
          uv = uv + self.rng.normal(0.0, self.pixel_noise, size=2)
        dets.append(
          Detection(label=label, uv=(float(uv[0]), float(uv[1])), truth_name=name)
        )
    return dets


class _UnavailableDetector:
  extra = "vision"
  model = "model"

  def detect(self, image=None, context=None) -> List[Detection]:
    raise NotImplementedError(
      f"{self.model} adapter requires the optional extra and weights: "
      f"`pip install plr-lr[{self.extra}]` and supply a checkpoint."
    )


class RFDetrDetector(_UnavailableDetector):
  """Adapter seam for an RF-DETR labware detector (optional ``vision`` extra)."""

  model = "RF-DETR"


class SAM2Detector(_UnavailableDetector):
  """Adapter seam for a SAM2 instance segmenter (optional ``vision`` extra)."""

  model = "SAM2"
