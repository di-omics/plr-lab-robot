"""Pinhole camera model for an eye-in-hand (wrist-mounted) camera."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .transforms import Transform


@dataclass
class CameraIntrinsics:
  """Pinhole intrinsics in pixels."""

  fx: float
  fy: float
  cx: float
  cy: float
  width: int = 1280
  height: int = 1024

  def project(self, pts_cam) -> np.ndarray:
    """Project camera-frame 3D points to pixel coordinates."""
    P = np.atleast_2d(np.asarray(pts_cam, float))
    z = P[:, 2]
    u = self.fx * P[:, 0] / z + self.cx
    v = self.fy * P[:, 1] / z + self.cy
    uv = np.stack([u, v], axis=1)
    return uv[0] if np.ndim(pts_cam) == 1 else uv

  def ray(self, uv) -> np.ndarray:
    """Return the unit view ray in the camera frame for a pixel."""
    u, v = uv
    d = np.array([(u - self.cx) / self.fx, (v - self.cy) / self.fy, 1.0])
    return d / np.linalg.norm(d)

  def backproject(self, uv, depth: float) -> np.ndarray:
    """Back-project a pixel at a known depth to a camera-frame 3D point."""
    u, v = uv
    return np.array(
      [(u - self.cx) * depth / self.fx, (v - self.cy) * depth / self.fy, depth]
    )


def base_from_camera(base_from_tool: Transform, cam_from_tool: Transform) -> Transform:
  """Camera pose in the base frame for an eye-in-hand rig.

  ``base_from_camera = base_from_tool @ tool_from_camera`` where
  ``tool_from_camera = cam_from_tool.inv()``.
  """
  return base_from_tool @ cam_from_tool.inv()
