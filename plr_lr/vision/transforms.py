"""Rigid 3D transforms (4x4 homogeneous) used by the vision phase."""

from __future__ import annotations

import numpy as np


def rot_x(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def rot_y(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def rot_z(a: float) -> np.ndarray:
  c, s = np.cos(a), np.sin(a)
  return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


class Transform:
  """A rigid transform. ``A_from_B`` maps points in frame B into frame A."""

  def __init__(self, matrix: np.ndarray | None = None) -> None:
    self.M = np.eye(4) if matrix is None else np.asarray(matrix, float).reshape(4, 4)

  @classmethod
  def from_rt(cls, R: np.ndarray, t) -> "Transform":
    M = np.eye(4)
    M[:3, :3] = np.asarray(R, float)
    M[:3, 3] = np.asarray(t, float).ravel()
    return cls(M)

  @classmethod
  def from_euler(cls, rx: float, ry: float, rz: float, t=(0.0, 0.0, 0.0)) -> "Transform":
    return cls.from_rt(rot_z(rz) @ rot_y(ry) @ rot_x(rx), t)

  @property
  def R(self) -> np.ndarray:
    return self.M[:3, :3]

  @property
  def t(self) -> np.ndarray:
    return self.M[:3, 3]

  def inv(self) -> "Transform":
    Rt = self.R.T
    return Transform.from_rt(Rt, -Rt @ self.t)

  def apply(self, pts) -> np.ndarray:
    pts = np.asarray(pts, float)
    single = pts.ndim == 1
    P = np.atleast_2d(pts)
    out = (P @ self.R.T) + self.t
    return out[0] if single else out

  def __matmul__(self, other):
    if isinstance(other, Transform):
      return Transform(self.M @ other.M)
    return self.apply(other)

  def __repr__(self) -> str:
    t = self.t
    return f"Transform(t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}])"
