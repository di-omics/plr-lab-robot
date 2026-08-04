"""The seam a controller plugs into: observation in, action out.

``Policy`` is one method. Everything downstream of it, including the safety
envelope, the recorder, and the evaluation harness, only knows about that one
method, so a hand-written servo loop, a replayed human demonstration, and a
vision-language-action model trained on a cluster are interchangeable at the
point where they touch the arm.

Four policies here are real and run on a CPU with no weights:

* :class:`NoOpPolicy` and :class:`RandomPolicy` exist to fail. They are the null
  controls every evaluation runs, and a task that either one can pass is a task
  whose success criterion is broken.
* :class:`ReplayPolicy` plays back a recorded episode, which turns a kinesthetic
  demonstration into something the runner can execute and score.
* :class:`VisionServoPolicy` closes the loop on what the wrist camera sees,
  reusing the same back-projection the guided pick uses.

The remaining classes are thin adapters for learned models. This package does
not pretend that loading a checkpoint is enough: model-specific preprocessing,
inference, and unit conversion live in a caller-supplied synchronous transport.
The adapter validates what comes back before it becomes an arm action.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable, List, Optional, Protocol, Sequence, Union, runtime_checkable

import numpy as np

from ..vision.camera import CameraIntrinsics, base_from_camera
from ..vision.detector import Detection
from ..vision.guided_pick import pixel_to_plane
from ..vision.transforms import Transform
from .types import ACTION_FIELDS, GRIPPER_UNCHANGED, Action, ActionMode, Episode, Observation


@runtime_checkable
class Policy(Protocol):
  """Chooses what to do next from what is currently observable."""

  def act(self, observation: Observation) -> Action:
    ...


def reset_policy(policy: object) -> None:
  """Call ``policy.reset()`` if it has one. Runners use this between episodes."""
  reset = getattr(policy, "reset", None)
  if callable(reset):
    reset()


class NoOpPolicy:
  """Commands nothing, forever. A null control, and never a solution."""

  def act(self, observation: Observation) -> Action:
    return Action.hold()


class RandomPolicy:
  """Bounded random displacements. The second null control.

  Random motion should never satisfy a well-posed task. When it does, the
  success criterion is measuring something other than the task, which is exactly
  what :func:`plr_lr.learn.evaluate.null_test` is there to catch.
  """

  def __init__(
    self,
    step_mm: float = 20.0,
    step_rad: float = 0.1,
    seed: Optional[int] = None,
  ) -> None:
    self.step_mm = float(step_mm)
    self.step_rad = float(step_rad)
    self.seed = seed
    self.rng = np.random.default_rng(seed)

  def reset(self) -> None:
    self.rng = np.random.default_rng(self.seed)

  def act(self, observation: Observation) -> Action:
    d = self.rng.uniform(-self.step_mm, self.step_mm, size=3)
    drz = float(self.rng.uniform(-self.step_rad, self.step_rad))
    return Action.delta(float(d[0]), float(d[1]), float(d[2]), drz)


class ReplayPolicy:
  """Replays the actions of a recorded episode, in order.

  Terminates when the recording runs out rather than holding position forever,
  so a replay that was cut short is visible as a short rollout instead of a
  timeout.
  """

  def __init__(self, episode: Episode, loop: bool = False) -> None:
    self.actions: List[Action] = [s.action for s in episode.steps]
    self.loop = loop
    self.i = 0

  def reset(self) -> None:
    self.i = 0

  def act(self, observation: Observation) -> Action:
    if self.i >= len(self.actions):
      if not self.loop or not self.actions:
        return Action.hold(terminate=True)
      self.i = 0
    action = self.actions[self.i]
    self.i += 1
    return action


class VisionServoPolicy:
  """Drive the tool over the labware the wrist camera can see.

  Each step back-projects the detection onto the known deck plane, takes the
  in-plane error between that point and the tool, and commands a fraction of it.
  With no detection it holds rather than guessing, so an occluded or empty frame
  stalls the rollout instead of moving the arm on stale information.

  This is proportional control, not a learned policy. It is here because it
  gives the evaluation harness a controller that genuinely solves the task, so
  the null tests have something to be contrasted against.
  """

  def __init__(
    self,
    intrinsics: CameraIntrinsics,
    cam_from_tool: Transform,
    plane_z: float,
    label: Optional[str] = None,
    gain: float = 0.6,
    tolerance_mm: float = 1.0,
    max_step_mm: float = 25.0,
  ) -> None:
    if not 0.0 < gain <= 1.0:
      raise ValueError("gain must be in (0, 1]")
    self.intr = intrinsics
    self.cam_from_tool = cam_from_tool
    self.plane_z = float(plane_z)
    self.label = label
    self.gain = float(gain)
    self.tolerance_mm = float(tolerance_mm)
    self.max_step_mm = float(max_step_mm)

  def _select(self, detections: Sequence[Detection]) -> Optional[Detection]:
    candidates = [d for d in detections if self.label is None or d.label == self.label]
    if not candidates:
      return None
    return max(candidates, key=lambda d: d.confidence)

  def act(self, observation: Observation) -> Action:
    det = self._select(observation.detections)
    if det is None:
      return Action.hold()

    base_from_tool = Transform.from_euler(
      0.0, 0.0, observation.rz, t=(observation.x, observation.y, observation.z)
    )
    base_from_cam = base_from_camera(base_from_tool, self.cam_from_tool)
    target = pixel_to_plane(det.uv, self.intr, base_from_cam, self.plane_z)

    err = np.array([target[0] - observation.x, target[1] - observation.y], float)
    distance = float(np.hypot(err[0], err[1]))
    if distance <= self.tolerance_mm:
      return Action.hold(terminate=True)

    step = err * self.gain
    magnitude = float(np.hypot(step[0], step[1]))
    if magnitude > self.max_step_mm:
      step = step * (self.max_step_mm / magnitude)
    return Action.delta(dx=float(step[0]), dy=float(step[1]))


class AdapterOutputError(ValueError):
  """Raised when an injected model transport returns an unsafe or ambiguous action."""


AdapterTransport = Callable[[Observation, str], Union[Action, Sequence[float]]]


class _TransportPolicy:
  """Base for learned controllers connected through an injected callable."""

  extra = "policy"
  model = "policy"

  def __init__(
    self,
    checkpoint: Optional[str] = None,
    instruction: str = "",
    transport: Optional[AdapterTransport] = None,
    action_mode: Union[ActionMode, str] = ActionMode.DELTA,
  ) -> None:
    if transport is not None and not callable(transport):
      raise TypeError("transport must be a synchronous callable or None")
    try:
      mode = ActionMode(action_mode)
    except (TypeError, ValueError) as exc:
      raise ValueError("action_mode must be DELTA or ABSOLUTE") from exc
    if mode is ActionMode.JOINTS:
      raise ValueError(
        "action_mode JOINTS is unsupported: the documented five-number action vector "
        "contains a Cartesian pose, not joint targets"
      )
    self.checkpoint = checkpoint
    self.instruction = instruction
    self.transport = transport
    self.action_mode = mode

  def _validate_action(self, action: Action) -> Action:
    try:
      mode = ActionMode(action.mode)
    except (TypeError, ValueError) as exc:
      raise AdapterOutputError("transport returned an Action with an unknown mode") from exc
    if mode is ActionMode.JOINTS:
      raise AdapterOutputError(
        "transport returned a JOINTS Action, but no mapping from model joint indices "
        "to this arm was declared; return DELTA or ABSOLUTE Cartesian motion"
      )
    try:
      pose = np.asarray(action.pose, float)
    except (TypeError, ValueError, OverflowError) as exc:
      raise AdapterOutputError("transport Action pose must contain 4 finite numbers") from exc
    if pose.shape != (4,) or not np.all(np.isfinite(pose)):
      raise AdapterOutputError("transport Action pose must contain 4 finite numbers")
    if action.joints is not None:
      raise AdapterOutputError(
        "transport returned joint targets with a Cartesian Action mode; the output is ambiguous"
      )

    gripper = action.gripper
    if gripper is not None:
      try:
        gripper = float(gripper)
      except (TypeError, ValueError, OverflowError) as exc:
        raise AdapterOutputError("transport Action gripper width must be finite") from exc
      if not np.isfinite(gripper):
        raise AdapterOutputError("transport Action gripper width must be finite")
      if gripper < 0.0:
        raise AdapterOutputError("transport Action gripper width cannot be negative")

    return Action(
      mode=mode,
      pose=tuple(float(value) for value in pose),
      gripper=gripper,
      joints=None,
      terminate=bool(action.terminate),
    )

  def _from_vector(self, output: Any) -> Action:
    try:
      vector = np.asarray(output, float)
    except (TypeError, ValueError, OverflowError) as exc:
      raise AdapterOutputError(
        f"transport must return an Action or exactly {len(ACTION_FIELDS)} numbers "
        f"for {ACTION_FIELDS}"
      ) from exc
    if vector.shape != (len(ACTION_FIELDS),):
      raise AdapterOutputError(
        f"transport action vector must have shape ({len(ACTION_FIELDS)},), got {vector.shape}"
      )
    if not np.all(np.isfinite(vector)):
      raise AdapterOutputError("transport action vector contains a non-finite value")

    gripper_value = float(vector[4])
    if gripper_value == GRIPPER_UNCHANGED:
      gripper = None
    elif gripper_value < 0.0:
      raise AdapterOutputError(
        f"transport gripper width cannot be negative; use {GRIPPER_UNCHANGED} "
        "for GRIPPER_UNCHANGED"
      )
    else:
      gripper = gripper_value

    pose = tuple(float(value) for value in vector[:4])
    if self.action_mode is ActionMode.DELTA:
      return Action.delta(*pose, gripper=gripper)
    return Action.absolute(*pose, gripper=gripper)

  def act(self, observation: Observation) -> Action:
    if self.transport is None:
      raise NotImplementedError(
        f"the {self.model} policy has no inference transport. Installing a package or "
        "passing a checkpoint alone is not an adapter: model-specific preprocessing, "
        "inference, and action-unit mapping are still required. Pass a synchronous "
        "callable as transport(observation, instruction) that returns an Action or "
        f"the {len(ACTION_FIELDS)}-number ACTION_FIELDS vector."
      )
    output = self.transport(observation, self.instruction)
    if inspect.isawaitable(output):
      # Avoid an un-awaited-coroutine warning while still refusing an async seam:
      # Policy.act is synchronous, and rollout owns the async boundary around it.
      close = getattr(output, "close", None)
      if callable(close):
        close()
      raise AdapterOutputError(
        "transport returned an awaitable; VLA transports must be synchronous callables"
      )
    if isinstance(output, Action):
      return self._validate_action(output)
    return self._from_vector(output)


class VLAPolicy(_TransportPolicy):
  """Seam for a vision-language-action model conditioned on a task string.

  A VLA expects an image and an instruction. ``Observation.image_path`` carries
  the frame and ``instruction`` carries the task, so an injected transport has
  both without changing the ``Policy`` interface. No native checkpoint loader is
  claimed here.
  """

  model = "VLA"


class OpenVLAPolicy(VLAPolicy):
  model = "OpenVLA"


class Pi0Policy(VLAPolicy):
  model = "pi0"


class GR00TPolicy(VLAPolicy):
  model = "GR00T N1"
