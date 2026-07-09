"""Grasp planning for caps and small vessels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Grasp:
  """A planned grasp: gripper width and how far above to start the approach."""

  width_mm: float
  approach_mm: float


def grasp_for_cap(cap_diameter_mm: float, clearance_mm: float = 1.5, approach_mm: float = 30.0) -> Grasp:
  """Close slightly inside the cap diameter so the pads bite without slipping."""
  if cap_diameter_mm <= 0:
    raise ValueError("cap_diameter_mm must be positive")
  return Grasp(width_mm=max(cap_diameter_mm - clearance_mm, 0.0), approach_mm=approach_mm)
