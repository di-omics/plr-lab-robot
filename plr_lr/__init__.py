"""plr-lr: bring a robotic arm under PyLabRobot control, then give it eyes and hands.

Three phases, built simulation-first:

  1. arm         stand up the arm (a simulation backend for ``ExperimentalSCARA``)
  2. vision      eye-in-hand calibration, detection, closed-loop guided pick
  3. manipulation dexterous skills beyond plate handling (uncap threaded vessels)
"""

from . import qc, vision
from .arm import (
  ArmNotHomedError,
  GripperError,
  SimulationArmBackend,
  Workcell,
  Workspace,
  WorkspaceViolation,
  site_coords,
)
from .world import Labware, Pose2D, Site, World

__version__ = "0.1.0"

__all__ = [
  "SimulationArmBackend",
  "Workspace",
  "Workcell",
  "site_coords",
  "ArmNotHomedError",
  "GripperError",
  "WorkspaceViolation",
  "World",
  "Site",
  "Labware",
  "Pose2D",
  "vision",
  "qc",
  "__version__",
]
