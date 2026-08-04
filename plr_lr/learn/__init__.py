"""Phase 4: learn. Record what the arm did, model it, plan against the model, score it.

The three phases before this one produce an arm that can be commanded and a
camera that can measure. This one closes the loop into something trainable:

  record  ->  model  ->  plan  ->  evaluate

* ``record``   capture demonstrations, by hand in freedrive or from any driver,
               into a portable episode format (:mod:`plr_lr.learn.record`,
               :mod:`plr_lr.learn.dataset`)
* ``model``    fit what your arm actually does, and refuse to claim an
               improvement that does not hold up on held-out data
               (:mod:`plr_lr.learn.dynamics`)
* ``plan``     turn a model into a controller by searching over its predictions
               (:mod:`plr_lr.learn.planner`), behind one ``Policy`` seam that a
               learned model can be dropped into (:mod:`plr_lr.learn.policy`)
* ``evaluate`` score it in simulation with null controls that must fail
               (:mod:`plr_lr.learn.evaluate`), and only then let it near
               hardware (:mod:`plr_lr.learn.rollout`)

Everything in the built-in loop runs on a CPU with no weights. Learned-policy
classes accept an injected inference transport; they do not claim that a package
or checkpoint alone supplies the model-specific adapter.
"""

from .dataset import Dataset, DatasetLayoutError, episodes_from
from .dynamics import (
  DynamicsAdapterError,
  DynamicsFit,
  DynamicsModel,
  InsufficientData,
  KinematicDynamics,
  LearnedDynamics,
  ResidualDynamics,
  VideoWorldModel,
)
from .evaluate import (
  DynamicsScore,
  EvalReport,
  NullTestReport,
  ReachTask,
  TrialResult,
  evaluate,
  evaluate_dynamics,
  null_test,
  run_trials,
)
from .planner import Goal, SamplingPlanner
from .policy import (
  AdapterOutputError,
  GR00TPolicy,
  NoOpPolicy,
  OpenVLAPolicy,
  Pi0Policy,
  Policy,
  RandomPolicy,
  ReplayPolicy,
  VisionServoPolicy,
  VLAPolicy,
  reset_policy,
)
from .record import EpisodeRecorder, KinestheticSession, observe, record_scripted
from .rollout import (
  HardwarePermissionRequired,
  SafetyEnvelope,
  UnsafeAction,
  apply_action,
  is_simulation,
  rollout,
)
from .synthetic import MistrackingBackend, random_walk_actions, simulate_episode
from .types import (
  ACTION_FIELDS,
  ACTION_UNITS,
  BASE_STATE_FIELDS,
  BASE_STATE_UNITS,
  GRIPPER_UNCHANGED,
  POSE_FIELDS,
  Action,
  ActionMode,
  Episode,
  Observation,
  Step,
)

__all__ = [
  # types
  "Observation",
  "Action",
  "ActionMode",
  "Step",
  "Episode",
  "POSE_FIELDS",
  "BASE_STATE_FIELDS",
  "BASE_STATE_UNITS",
  "ACTION_FIELDS",
  "ACTION_UNITS",
  "GRIPPER_UNCHANGED",
  # record and dataset
  "observe",
  "EpisodeRecorder",
  "KinestheticSession",
  "record_scripted",
  "Dataset",
  "DatasetLayoutError",
  "episodes_from",
  # dynamics
  "DynamicsModel",
  "DynamicsAdapterError",
  "KinematicDynamics",
  "ResidualDynamics",
  "DynamicsFit",
  "InsufficientData",
  "LearnedDynamics",
  "VideoWorldModel",
  # synthetic ground truth
  "MistrackingBackend",
  "simulate_episode",
  "random_walk_actions",
  # policy and planning
  "AdapterOutputError",
  "Policy",
  "reset_policy",
  "NoOpPolicy",
  "RandomPolicy",
  "ReplayPolicy",
  "VisionServoPolicy",
  "VLAPolicy",
  "OpenVLAPolicy",
  "Pi0Policy",
  "GR00TPolicy",
  "Goal",
  "SamplingPlanner",
  # execution
  "SafetyEnvelope",
  "rollout",
  "apply_action",
  "is_simulation",
  "HardwarePermissionRequired",
  "UnsafeAction",
  # evaluation
  "ReachTask",
  "TrialResult",
  "EvalReport",
  "NullTestReport",
  "DynamicsScore",
  "run_trials",
  "null_test",
  "evaluate",
  "evaluate_dynamics",
]
