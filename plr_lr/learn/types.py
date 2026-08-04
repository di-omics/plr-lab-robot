"""Typed records for the learning phase: what the arm saw, and what it did next.

An ``Observation`` is everything a policy is allowed to condition on at one
instant: the arm's own proprioception (joints, tool pose, gripper) plus whatever
the eye-in-hand camera reported. An ``Action`` is what the policy asked for. A
``Step`` pairs them, and an ``Episode`` is an ordered run of steps with the task
string that describes what was being attempted.

Two vector layouts are defined here and nowhere else, because a state vector
whose field order is implicit is the fastest way to train a model on scrambled
data:

* ``Observation.state_vector()`` -> ``[x, y, z, rz, gripper_width,
  gripper_closed, holding, j1 .. jN]``
* ``Action.action_vector()`` -> ``[p0, p1, p2, p3, gripper]`` where ``p`` is a
  delta or an absolute pose depending on ``Action.mode``, and ``gripper`` is
  ``GRIPPER_UNCHANGED`` when the action leaves the gripper alone.

Both layouts are self-describing: ``Observation.state_fields()`` and
``ACTION_FIELDS`` name every element, while ``Observation.state_units()`` and
``ACTION_UNITS`` record the units. The dataset writer keeps both in
``meta.json`` so a training run never has to guess.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from ..vision.detector import Detection

POSE_FIELDS: Tuple[str, ...] = ("x", "y", "z", "rz")
BASE_STATE_FIELDS: Tuple[str, ...] = POSE_FIELDS + (
  "gripper_width",
  "gripper_closed",
  "holding",
)
ACTION_FIELDS: Tuple[str, ...] = ("p0", "p1", "p2", "p3", "gripper")
BASE_STATE_UNITS: Tuple[str, ...] = ("mm", "mm", "mm", "rad", "mm", "bool", "bool")
ACTION_UNITS: Tuple[str, ...] = ("mm", "mm", "mm", "rad", "mm")

#: Sentinel in the gripper slot of an action vector meaning "leave the gripper
#: as it is". A real width is always >= 0, so no valid command collides with it.
GRIPPER_UNCHANGED = -1.0


class ActionMode(str, Enum):
  """How the four pose numbers in an ``Action`` should be read."""

  DELTA = "delta"  # displacement from the current pose, mm and rad
  ABSOLUTE = "absolute"  # a pose in the arm base frame, mm and rad
  JOINTS = "joints"  # joint targets, degrees, taken from Action.joints


@dataclass(frozen=True)
class Observation:
  """One instant of what the arm could know about itself and the deck."""

  t: float = 0.0  # seconds since the start of the episode
  x: float = 0.0
  y: float = 0.0
  z: float = 0.0
  rz: float = 0.0
  #: Gripper opening in mm, or None when the backend does not know one yet. Not
  #: every backend tracks a width, and the xArm does not have one until the
  #: gripper has been commanded at least once.
  gripper_width: Optional[float] = None
  gripper_closed: bool = False
  holding: bool = False
  joints: Dict[int, float] = field(default_factory=dict)  # index -> degrees
  detections: Tuple[Detection, ...] = ()
  image_path: Optional[str] = None

  # -- vectors ---------------------------------------------------------------

  def pose_vector(self) -> np.ndarray:
    """The 4-vector the dynamics models and planner operate on."""
    return np.array([self.x, self.y, self.z, self.rz], float)

  def joint_indices(self) -> Tuple[int, ...]:
    return tuple(sorted(self.joints))

  def state_fields(self) -> Tuple[str, ...]:
    """Names for every element of ``state_vector()``, in order."""
    return BASE_STATE_FIELDS + tuple(f"j{i}" for i in self.joint_indices())

  def state_units(self) -> Tuple[str, ...]:
    """Units for every element of ``state_vector()``, in the same order."""
    return BASE_STATE_UNITS + tuple("deg" for _ in self.joint_indices())

  def state_vector(self) -> np.ndarray:
    """The flat observation a policy or training run consumes.

    An unknown gripper width becomes NaN, deliberately. Substituting 0.0 would
    put a plausible number in a column that means "we did not measure this", and
    a training run would fit it without ever noticing. A NaN fails loudly at the
    first batch, which is the outcome worth having.
    """
    head = [
      self.x,
      self.y,
      self.z,
      self.rz,
      np.nan if self.gripper_width is None else self.gripper_width,
      float(self.gripper_closed),
      float(self.holding),
    ]
    return np.array(head + [self.joints[i] for i in self.joint_indices()], float)

  # -- serialization ---------------------------------------------------------

  def to_dict(self) -> Dict[str, Any]:
    return {
      "t": self.t,
      "x": self.x,
      "y": self.y,
      "z": self.z,
      "rz": self.rz,
      "gripper_width": self.gripper_width,
      "gripper_closed": self.gripper_closed,
      "holding": self.holding,
      "joints": {str(k): v for k, v in self.joints.items()},
      "detections": [asdict(d) for d in self.detections],
      "image_path": self.image_path,
    }

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "Observation":
    dets = tuple(
      Detection(
        label=x["label"],
        uv=(float(x["uv"][0]), float(x["uv"][1])),
        confidence=float(x.get("confidence", 1.0)),
        truth_name=x.get("truth_name"),
      )
      for x in d.get("detections", [])
    )
    return cls(
      t=float(d.get("t", 0.0)),
      x=float(d.get("x", 0.0)),
      y=float(d.get("y", 0.0)),
      z=float(d.get("z", 0.0)),
      rz=float(d.get("rz", 0.0)),
      gripper_width=None if d.get("gripper_width") is None else float(d["gripper_width"]),
      gripper_closed=bool(d.get("gripper_closed", False)),
      holding=bool(d.get("holding", False)),
      joints={int(k): float(v) for k, v in d.get("joints", {}).items()},
      detections=dets,
      image_path=d.get("image_path"),
    )


@dataclass(frozen=True)
class Action:
  """A policy's request, in the arm's own units (mm and radians).

  Build one with :meth:`delta`, :meth:`absolute`, :meth:`joint_targets`, or
  :meth:`hold` rather than by hand, so ``mode`` and ``pose`` always agree.
  """

  mode: ActionMode = ActionMode.DELTA
  pose: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
  gripper: Optional[float] = None  # target width in mm, None leaves it alone
  joints: Optional[Dict[int, float]] = None  # degrees, JOINTS mode only
  terminate: bool = False  # the policy believes the task is finished

  # -- constructors ----------------------------------------------------------

  @classmethod
  def delta(
    cls,
    dx: float = 0.0,
    dy: float = 0.0,
    dz: float = 0.0,
    drz: float = 0.0,
    gripper: Optional[float] = None,
    terminate: bool = False,
  ) -> "Action":
    return cls(ActionMode.DELTA, (dx, dy, dz, drz), gripper, None, terminate)

  @classmethod
  def absolute(
    cls,
    x: float,
    y: float,
    z: float,
    rz: float = 0.0,
    gripper: Optional[float] = None,
    terminate: bool = False,
  ) -> "Action":
    return cls(ActionMode.ABSOLUTE, (x, y, z, rz), gripper, None, terminate)

  @classmethod
  def joint_targets(
    cls,
    joints: Dict[int, float],
    gripper: Optional[float] = None,
    terminate: bool = False,
  ) -> "Action":
    return cls(ActionMode.JOINTS, (0.0, 0.0, 0.0, 0.0), gripper, dict(joints), terminate)

  @classmethod
  def hold(cls, terminate: bool = False) -> "Action":
    """Command nothing. The null action, and the one a stuck policy emits."""
    return cls(ActionMode.DELTA, (0.0, 0.0, 0.0, 0.0), None, None, terminate)

  # -- vectors ---------------------------------------------------------------

  def action_vector(self) -> np.ndarray:
    g = GRIPPER_UNCHANGED if self.gripper is None else float(self.gripper)
    return np.array(list(self.pose) + [g], float)

  @property
  def translation(self) -> np.ndarray:
    """The (dx, dy, dz) part, whatever the mode calls it."""
    return np.array(self.pose[:3], float)

  def as_delta(self, current_pose: Sequence[float]) -> np.ndarray:
    """The displacement this action asks for, given where the arm is now.

    Delta actions are already displacements. Absolute actions become the
    difference from ``current_pose``. Joint actions have no Cartesian
    displacement that can be computed without kinematics, so they are refused
    here rather than silently treated as zero motion.
    """
    if self.mode is ActionMode.JOINTS:
      raise ValueError(
        "a JOINTS action has no Cartesian delta; run it through the arm's "
        "kinematics, or use DELTA or ABSOLUTE for Cartesian control"
      )
    p = np.array(self.pose, float)
    if self.mode is ActionMode.DELTA:
      return p
    delta = p - np.asarray(current_pose, float)
    # Orientations that differ by a full turn name the same wrist pose. Use the
    # shortest signed displacement so an absolute target just across +/-pi does
    # not become an almost-2pi command that the safety envelope then clips in
    # the wrong direction.
    delta[3] = np.arctan2(np.sin(delta[3]), np.cos(delta[3]))
    return delta

  def with_pose(self, pose: Sequence[float]) -> "Action":
    """Copy with a replaced pose, used by the safety envelope to clamp."""
    p = tuple(float(v) for v in pose)
    if len(p) != 4:
      raise ValueError(f"pose must have 4 elements, got {len(p)}")
    return Action(self.mode, p, self.gripper, self.joints, self.terminate)  # type: ignore[arg-type]

  # -- serialization ---------------------------------------------------------

  def to_dict(self) -> Dict[str, Any]:
    return {
      "mode": self.mode.value,
      "pose": list(self.pose),
      "gripper": self.gripper,
      "joints": None if self.joints is None else {str(k): v for k, v in self.joints.items()},
      "terminate": self.terminate,
    }

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "Action":
    joints = d.get("joints")
    p = tuple(float(v) for v in d.get("pose", (0.0, 0.0, 0.0, 0.0)))
    g = d.get("gripper")
    return cls(
      mode=ActionMode(d.get("mode", "delta")),
      pose=(p[0], p[1], p[2], p[3]),
      gripper=None if g is None else float(g),
      joints=None if joints is None else {int(k): float(v) for k, v in joints.items()},
      terminate=bool(d.get("terminate", False)),
    )


@dataclass
class Step:
  """What was observed, and what the policy did about it."""

  observation: Observation
  action: Action
  info: Dict[str, Any] = field(default_factory=dict)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "observation": self.observation.to_dict(),
      "action": self.action.to_dict(),
      "info": self.info,
    }

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "Step":
    return cls(
      observation=Observation.from_dict(d["observation"]),
      action=Action.from_dict(d["action"]),
      info=dict(d.get("info", {})),
    )


@dataclass
class Episode:
  """One attempt at one task, as an ordered list of steps.

  ``steps[i].observation`` is the state the arm was in when
  ``steps[i].action`` was chosen, so the transition the dynamics models learn is
  ``(steps[i].observation, steps[i].action) -> steps[i + 1].observation``. The
  final step's action has no recorded outcome unless ``final_observation`` is
  set, which the recorder always does.
  """

  task: str = ""
  steps: List[Step] = field(default_factory=list)
  final_observation: Optional[Observation] = None
  metadata: Dict[str, Any] = field(default_factory=dict)
  episode_index: int = 0

  def __len__(self) -> int:
    return len(self.steps)

  def __iter__(self) -> Iterator[Step]:
    return iter(self.steps)

  def append(self, observation: Observation, action: Action, **info: Any) -> Step:
    step = Step(observation=observation, action=action, info=dict(info))
    self.steps.append(step)
    return step

  @property
  def duration(self) -> float:
    """Seconds from the first observation to the last one recorded."""
    if not self.steps:
      return 0.0
    last = self.final_observation or self.steps[-1].observation
    return float(last.t - self.steps[0].observation.t)

  def observations(self) -> List[Observation]:
    """Every observation including the final one, in order."""
    obs = [s.observation for s in self.steps]
    if self.final_observation is not None:
      obs.append(self.final_observation)
    return obs

  def transitions(self) -> List[Tuple[Observation, Action, Observation]]:
    """The ``(observation, action, next observation)`` triples in this episode.

    Only complete transitions are returned. A step whose outcome was never
    observed is dropped rather than paired with a guess.
    """
    obs = self.observations()
    return [(obs[i], self.steps[i].action, obs[i + 1]) for i in range(len(obs) - 1)]

  # -- export ----------------------------------------------------------------

  def frames(self) -> List[Dict[str, Any]]:
    """Project-native interchange records, one dict per step.

    The JSONL export uses familiar robot-learning names such as
    ``observation.state`` and ``action``, but it is deliberately a plr-lr schema,
    not a native export for any external dataset library. In particular,
    ``action.mode`` preserves semantics that a bare action vector cannot carry.
    Adapt and validate these records against the target loader before training.

    ``action.mode`` travels with every row. Without it a delta of (5, 0, 0, 0)
    and an absolute target of (5, 0, 0, 0) are the same five numbers meaning
    entirely different things, and a training run has no way to tell which it
    was handed. A joint-space action has no Cartesian form at all, so this
    refuses rather than exporting the zeros that ``Action.as_delta`` already
    declines to invent.
    """
    out: List[Dict[str, Any]] = []
    for i, step in enumerate(self.steps):
      if step.action.mode is ActionMode.JOINTS:
        raise ValueError(
          f"step {i} of episode {self.episode_index} is a joint-space action, which has "
          "no Cartesian action vector. Exporting it would write zeros into the 'action' "
          "column alongside real Cartesian commands. Record Cartesian actions, or export "
          "joint-space episodes through a schema that has a place for them."
        )
      out.append(
        {
          "observation.state": step.observation.state_vector().tolist(),
          "action": step.action.action_vector().tolist(),
          "action.mode": step.action.mode.value,
          "timestamp": float(step.observation.t),
          "frame_index": i,
          "episode_index": self.episode_index,
          "task": self.task,
        }
      )
    return out

  # -- serialization ---------------------------------------------------------

  def to_dict(self) -> Dict[str, Any]:
    return {
      "task": self.task,
      "episode_index": self.episode_index,
      "metadata": self.metadata,
      "final_observation": (
        None if self.final_observation is None else self.final_observation.to_dict()
      ),
      "steps": [s.to_dict() for s in self.steps],
    }

  @classmethod
  def from_dict(cls, d: Dict[str, Any]) -> "Episode":
    fo = d.get("final_observation")
    return cls(
      task=d.get("task", ""),
      steps=[Step.from_dict(s) for s in d.get("steps", [])],
      final_observation=None if fo is None else Observation.from_dict(fo),
      metadata=dict(d.get("metadata", {})),
      episode_index=int(d.get("episode_index", 0)),
    )
