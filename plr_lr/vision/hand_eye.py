"""Eye-in-hand hand-eye calibration by rigid point-set registration.

Given a camera mounted on the wrist and a calibration fixture with markers at
known base-frame positions, we observe those markers in the camera frame from one
or more known tool poses. For each observation we express the markers in the tool
frame (using the known tool pose) and pair them with their measured camera-frame
positions. Stacking all observations gives two corresponded point sets related by
the unknown hand-eye transform ``cam_from_tool``, which we recover with the
closed-form Kabsch/Umeyama solution and report an RMS residual for QC.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np

from .transforms import Transform

Observation = Tuple[Transform, Sequence, Sequence]
"""(base_from_tool, marker_base_positions, marker_camera_positions)."""


def kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[Transform, float]:
  """Best-fit rigid transform T with ``dst ~= T.apply(src)`` (no scale).

  Returns the transform and the RMS point residual.
  """
  src = np.asarray(src, float)
  dst = np.asarray(dst, float)
  if src.shape != dst.shape or src.shape[0] < 3:
    raise ValueError("need matching point sets with at least 3 points")
  cs = src.mean(axis=0)
  cd = dst.mean(axis=0)
  S = src - cs
  D = dst - cd
  H = S.T @ D
  U, _, Vt = np.linalg.svd(H)
  d = np.sign(np.linalg.det(Vt.T @ U.T))
  R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
  t = cd - R @ cs
  T = Transform.from_rt(R, t)
  resid = dst - T.apply(src)
  rms = float(np.sqrt((resid ** 2).sum(axis=1).mean()))
  return T, rms


@dataclass
class HandEyeResult:
  cam_from_tool: Transform
  rms_mm: float
  n_points: int

  @property
  def ok(self) -> bool:
    return self.rms_mm < 0.5


def calibrate_eye_in_hand(observations: List[Observation]) -> HandEyeResult:
  """Recover the wrist-to-camera transform from marker observations."""
  tool_pts = []
  cam_pts = []
  for base_from_tool, m_base, m_cam in observations:
    tool_from_base = base_from_tool.inv()
    tool_pts.append(np.atleast_2d(tool_from_base.apply(np.atleast_2d(m_base))))
    cam_pts.append(np.atleast_2d(np.asarray(m_cam, float)))
  tool_pts = np.vstack(tool_pts)
  cam_pts = np.vstack(cam_pts)
  cam_from_tool, rms = kabsch(tool_pts, cam_pts)
  return HandEyeResult(cam_from_tool=cam_from_tool, rms_mm=rms, n_points=len(tool_pts))
