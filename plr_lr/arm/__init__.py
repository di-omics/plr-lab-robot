"""Phase 1: stand up the robotic arm under PyLabRobot control.

`sim_backend` exercises the arm API. `kinematics` is what lets you tell the arm where to
go: before it, `_joints` and `_cartesian` were independent variables that could disagree
without limit and nothing noticed. `stations` turns a taught-pose dictionary into places a
plate can legally be moved between.
"""

from .kinematics import (
  IKResult,
  JointLimits,
  KinematicModel,
  KinematicsRefusal,
  OutsideEnvelope,
  Pose,
  PoseOrigin,
  Reachability,
  forward,
  inverse,
  reachable,
  unverified_model,
  xarm6,
  xarm7,
)
from .sim_backend import (
  ArmNotHomedError,
  GripperError,
  SimulationArmBackend,
  Workspace,
  WorkspaceViolation,
)
from .stations import (
  CycleState,
  Legality,
  Move,
  MoveOutcome,
  PlacementTolerance,
  Station,
  sequence,
  station_map,
)
from .workcell import Workcell, site_coords

__all__ = [
  "ArmNotHomedError",
  "CycleState",
  "GripperError",
  "IKResult",
  "JointLimits",
  "KinematicModel",
  "KinematicsRefusal",
  "Legality",
  "Move",
  "MoveOutcome",
  "OutsideEnvelope",
  "PlacementTolerance",
  "Pose",
  "PoseOrigin",
  "Reachability",
  "SimulationArmBackend",
  "Station",
  "Workcell",
  "Workspace",
  "WorkspaceViolation",
  "forward",
  "inverse",
  "reachable",
  "sequence",
  "site_coords",
  "station_map",
  "unverified_model",
  "xarm6",
  "xarm7",
]
