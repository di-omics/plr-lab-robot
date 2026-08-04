"""Run a policy against an arm, with the envelope and the hardware gate in the way.

Every action a policy emits passes through a :class:`SafetyEnvelope` before it
reaches the arm. The envelope bounds how far one step may move, keeps the target
inside a workspace box, and caps how many steps a rollout gets. What it changed
is written into the episode, so a run that only succeeded because the envelope
kept dragging it back into bounds cannot be read as a clean success.

Hardware is fail-closed. :func:`rollout` refuses any backend that is not the
simulation backend unless the caller passes ``allow_hardware=True``. A learned
policy is an unreviewed program: the default is that it does not get to move a
real arm, and turning that off is a deliberate, greppable act at the call site.

    envelope = SafetyEnvelope(max_step_mm=20.0, workspace=Workspace(), max_steps=60)
    episode = await rollout(arm, policy, envelope=envelope, task="reach reader")

Two things this module now counts that it previously changed in silence.

**Clamping is counted, and it is counted per step.** ``n_clamped`` and
``clamp_magnitude_mm`` are recorded on every step and on every episode, whether
or not anything was clamped, so a zero is a measurement rather than an absence.
The reason is a measured one: :class:`~plr_lr.learn.policy.RandomPolicy` at
``step_mm=25.0`` against the default ``SafetyEnvelope`` has close to half of its
steps rewritten by the envelope, because it samples displacements from the cube
and the envelope bounds them to the inscribed ball. Nothing in the report said
so. A null control whose steps are mostly being rewritten is not the control
anybody thought they ran, and a correlation drawn across such a run is a
correlation with the envelope. See :data:`THROTTLED_CLAMP_FRACTION`.

**A clamped action is not the action that was screened.** When a ``screen`` is
supplied, an admitted verdict that is then clamped is downgraded to
``ABSTAIN(CLAMPED)`` and nothing is sent. The verdict was reached about a command
that the envelope has since rewritten, and a verdict does not transfer to a
different command. With ``screen=None`` this module behaves exactly as it did
before the argument existed.
"""

from __future__ import annotations

import asyncio
import inspect
import math
from dataclasses import dataclass, field, replace
from numbers import Integral, Real
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.precise_flex.coords import PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from ..arm.sim_backend import SimulationArmBackend, Workspace
from .policy import reset_policy
from .record import EpisodeRecorder, gripper_width_of
from .types import Action, ActionMode, Episode, Observation

#: A step sized exactly at the limit arrives as ``limit + 4e-15`` after the
#: floating-point round trip through a norm. Reporting that as a clamp would mark
#: every well-behaved step as clamped and bury the real ones.
_BOUNDARY_TOLERANCE = 1e-9

#: Above this share of clamped steps a null control is reported as
#: :data:`THROTTLED` and stops counting as evidence. An absolute constant, never
#: derived from the run being judged.
#:
#: The number matters because the measured rate is far above it. Scored through
#: the existing harness, ``RandomPolicy(step_mm=25.0)`` against
#: ``SafetyEnvelope(max_step_mm=25.0)`` has 47.0 percent of its steps rewritten at
#: the harness default of five trials, rising to 48.9 percent over five hundred,
#: and the reason is arithmetic rather than incidental: the policy draws
#: displacements uniformly from the cube of half-width ``step_mm`` while the
#: envelope bounds them to the inscribed ball, and the cube is ``1 - pi/6``,
#: 47.6 percent, outside that ball. A control that is being rewritten every other
#: step is measuring the envelope.
THROTTLED_CLAMP_FRACTION = 0.10

#: What a control above that share is labelled. A string rather than a bare
#: boolean because it prints beside the rate it invalidates.
THROTTLED = "THROTTLED"


class HardwarePermissionRequired(RuntimeError):
  """Raised when a policy would drive a non-simulated arm without explicit consent."""


class UnsafeAction(ValueError):
  """Raised when an action cannot be made safe rather than merely clamped."""


class StepLimitExceeded(UnsafeAction):
  """Raised when a command is larger than the backend will actually execute.

  A subclass of :class:`UnsafeAction` because that is what it is: a command that
  clamping cannot make safe, only smaller than the envelope's own bound and still
  too large for the controller underneath.

  This exists because the two limits are set independently and the defaults
  disagree. ``SafetyEnvelope.max_step_mm`` defaults to 25.0 mm, which is above the
  xArm's documented 10 mm ceiling on ``set_servo_cartesian``. A command clamped to
  25 mm therefore satisfies the envelope, is sent, is accepted by the controller,
  moves nothing, and returns success. Every layer above reads that as a completed
  step, the recorded episode contains an action that never happened, and a
  dynamics fit trained on it learns an arm with a very large tracking error in one
  direction.

  Declaring ``per_command_step_mm`` turns that into a refusal at the point of
  command. It is deliberately not a second clamp: silently shrinking to the
  controller's limit would hide that the envelope and the hardware disagree, and
  the disagreement is the thing worth knowing.
  """


@dataclass
class SafetyEnvelope:
  """Bounds every action before it reaches the arm.

  Args:
    max_step_mm: largest translation a single step may command. Larger requests
      are scaled down along their own direction, so the intent survives and the
      magnitude does not.
    max_step_rad: largest rotation a single step may command.
    workspace: box the resulting target must land in. Targets outside it are
      clipped per axis. A current pose already outside it is refused: returning
      to a trusted region is a recovery procedure, not a policy clamp.
    max_steps: rollout budget. A policy that never terminates stops here.
    allow_joint_actions: joint-space actions cannot be checked against a
      Cartesian workspace without forward kinematics, so they are refused by
      default rather than passed through unchecked.
    per_command_step_mm: the largest Cartesian step the backend underneath will
      actually execute, when it declares one. This is not a second clamp. A
      command above it raises :class:`StepLimitExceeded`, because the failure it
      guards against is a controller accepting an oversized command, moving
      nothing, and reporting success. ``None`` means no limit was declared, which
      is the honest state for the simulation backend and for any driver that has
      not been checked against its own documentation.
  """

  max_step_mm: float = 25.0
  max_step_rad: float = 0.2
  workspace: Optional[Workspace] = field(default_factory=Workspace)
  max_steps: int = 200
  allow_joint_actions: bool = False
  per_command_step_mm: Optional[float] = None

  def __post_init__(self) -> None:
    """Refuse an envelope whose declared limits cannot bound anything."""
    if self.per_command_step_mm is not None:
      # Strict about the type, unlike the two bounds below, which have accepted a
      # numeric string since before this field existed. This one is routinely
      # copied out of a vendor document or a config file, and a bool floats to
      # 1.0 while a string floats to whatever it happens to parse as; either one
      # would install a limit nobody chose and silence the refusal below.
      declared = self.per_command_step_mm
      if isinstance(declared, bool) or not isinstance(declared, Real):
        raise ValueError(
          f"per_command_step_mm must be a positive finite number of millimetres or "
          f"None, got {declared!r}"
        )
      declared = float(declared)
      if not math.isfinite(declared) or declared <= 0.0:
        raise ValueError(
          f"per_command_step_mm must be a positive finite number of millimetres or "
          f"None, got {self.per_command_step_mm!r}"
        )
      self.per_command_step_mm = declared

    for name in ("max_step_mm", "max_step_rad"):
      try:
        value = float(getattr(self, name))
      except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
      if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be a positive finite number, got {value!r}")
      setattr(self, name, value)

    if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, Integral):
      raise ValueError(f"max_steps must be a positive integer, got {self.max_steps!r}")
    if self.max_steps < 1:
      raise ValueError(f"max_steps must be a positive integer, got {self.max_steps!r}")
    self.max_steps = int(self.max_steps)

    if self.workspace is not None:
      for axis in ("x", "y", "z"):
        bounds = getattr(self.workspace, axis)
        try:
          lower, upper = (float(bounds[0]), float(bounds[1]))
        except (IndexError, TypeError, ValueError) as exc:
          raise ValueError(f"workspace.{axis} must contain two finite bounds") from exc
        if not math.isfinite(lower) or not math.isfinite(upper) or lower > upper:
          raise ValueError(
            f"workspace.{axis} must be finite and ordered low-to-high, got {bounds!r}"
          )

  @staticmethod
  def _require_finite(name: str, values: Sequence[float]) -> np.ndarray:
    try:
      array = np.asarray(values, float)
    except (TypeError, ValueError) as exc:
      raise UnsafeAction(f"{name} must contain only finite numbers") from exc
    if not np.all(np.isfinite(array)):
      raise UnsafeAction(f"{name} must contain only finite numbers, got {array.tolist()}")
    return array

  def _current_pose_is_inside_workspace(self, pose: np.ndarray) -> bool:
    if self.workspace is None:
      return True
    return self.workspace.contains(Coordinate(float(pose[0]), float(pose[1]), float(pose[2])))

  def _bound_translation(self, translation: np.ndarray) -> Tuple[np.ndarray, Optional[float]]:
    """Scale a displacement into the step limit. Returns it plus the size it was."""
    magnitude = float(np.linalg.norm(translation))
    if magnitude > self.max_step_mm * (1.0 + _BOUNDARY_TOLERANCE):
      return translation * (self.max_step_mm / magnitude), magnitude
    return translation, None

  def clamp(self, action: Action, pose: Sequence[float]) -> Tuple[Action, List[str]]:
    """Return a safe version of ``action`` plus notes on what was changed."""
    notes: List[str] = []

    if not isinstance(action, Action):
      raise UnsafeAction(f"expected an Action, got {type(action).__name__}")

    p = self._require_finite("current pose", pose)
    try:
      p = p.reshape(4)
    except ValueError as exc:
      raise UnsafeAction(f"current pose must have four elements, got shape {p.shape}") from exc
    if not self._current_pose_is_inside_workspace(p):
      raise UnsafeAction(
        f"current pose ({p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}) is outside the declared "
        f"workspace {self.workspace}. The envelope only bounds commands that start "
        "inside its trusted region; recover the arm under an explicit manual procedure."
      )

    if action.gripper is not None:
      try:
        gripper = float(action.gripper)
      except (TypeError, ValueError, OverflowError) as exc:
        raise UnsafeAction(
          f"gripper target must be a finite non-negative width in mm, got "
          f"{action.gripper!r}"
        ) from exc
      if not math.isfinite(gripper) or gripper < 0.0:
        raise UnsafeAction(
          f"gripper target must be a finite non-negative width in mm, got {action.gripper!r}"
        )

    if action.mode is ActionMode.JOINTS:
      if not self.allow_joint_actions:
        raise UnsafeAction(
          "joint-space action refused: the envelope cannot check joint targets "
          "against a Cartesian workspace. Pass allow_joint_actions=True to accept "
          "that this step is unchecked, or command Cartesian motion."
        )
      if action.joints is not None:
        self._require_finite("joint targets", tuple(action.joints.values()))
      return action, ["joint action passed through unchecked"]

    if action.mode not in (ActionMode.DELTA, ActionMode.ABSOLUTE):
      raise UnsafeAction(f"unsupported action mode {action.mode!r}")
    try:
      delta = self._require_finite("Cartesian action", action.as_delta(p)).reshape(4)
    except ValueError as exc:
      raise UnsafeAction("Cartesian action must have four pose elements") from exc

    translation, magnitude = self._bound_translation(delta[:3])
    if magnitude is not None:
      notes.append(f"translation {magnitude:.2f} mm scaled to {self.max_step_mm:.2f} mm")

    drz = float(delta[3])
    if abs(drz) > self.max_step_rad * (1.0 + _BOUNDARY_TOLERANCE):
      drz = float(np.sign(drz) * self.max_step_rad)
      notes.append(f"rotation {delta[3]:.4f} rad clamped to {drz:.4f} rad")

    target = np.array(
      [p[0] + translation[0], p[1] + translation[1], p[2] + translation[2], p[3] + drz]
    )
    if self.workspace is not None:
      clipped = np.array(
        [
          float(np.clip(target[0], *self.workspace.x)),
          float(np.clip(target[1], *self.workspace.y)),
          float(np.clip(target[2], *self.workspace.z)),
        ]
      )
      if not np.allclose(clipped, target[:3]):
        notes.append(
          f"target ({target[0]:.1f}, {target[1]:.1f}, {target[2]:.1f}) clipped to "
          f"workspace at ({clipped[0]:.1f}, {clipped[1]:.1f}, {clipped[2]:.1f})"
        )
        # Independently enforce the step limit on the rewritten displacement as
        # well. The current pose was proven inside the box above; recovery from an
        # already-outside pose is deliberately refused rather than synthesized.
        bounded, over = self._bound_translation(clipped - p[:3])
        if over is not None:
          notes.append(f"clipped step {over:.2f} mm scaled to {self.max_step_mm:.2f} mm")
        target[:3] = p[:3] + bounded

    if action.mode is ActionMode.ABSOLUTE:
      safe_pose = (float(target[0]), float(target[1]), float(target[2]), float(target[3]))
    else:
      safe_pose = (
        float(target[0] - p[0]),
        float(target[1] - p[1]),
        float(target[2] - p[2]),
        float(target[3] - p[3]),
      )
    safe = action.with_pose(safe_pose)
    self._check_step_limit(safe, p)
    return safe, notes

  def _check_step_limit(self, action: Action, pose: np.ndarray) -> None:
    """Refuse a command the backend underneath would accept and not execute.

    Checked on the clamped command rather than on the request, because the
    clamped one is what gets sent. An oversized request that the envelope brings
    down inside the controller's limit is fine; a request the envelope brings down
    to a value still above it is the case worth refusing, and it is the case the
    package defaults produce, ``max_step_mm=25.0`` against a documented 10 mm.
    """
    if self.per_command_step_mm is None:
      return
    delta = np.asarray(action.as_delta(pose), float).reshape(4)
    magnitude = float(np.linalg.norm(delta[:3]))
    if magnitude <= self.per_command_step_mm * (1.0 + _BOUNDARY_TOLERANCE):
      return
    raise StepLimitExceeded(
      f"a {magnitude:.2f} mm Cartesian step is above the {self.per_command_step_mm:.2f} mm "
      f"the backend declared it will execute per command, and this envelope clamps to "
      f"{self.max_step_mm:.2f} mm, which is already above that. The controller accepts a "
      "command this size, moves nothing, and returns success, so the step would be "
      "recorded as taken. Lower max_step_mm to the backend's limit, or issue the motion "
      "as several commands."
    )


def is_simulation(arm: ExperimentalSCARA) -> bool:
  """True when this arm is backed by the simulation backend."""
  return isinstance(arm.backend, SimulationArmBackend)


def per_command_step_mm_of(backend: object) -> Optional[float]:
  """The largest Cartesian step this backend says it executes, or None.

  Returning None rather than a generous default is the point. A default would be
  a number nobody measured being used to decide whether a command is going to do
  anything, and it would silence :class:`StepLimitExceeded` on exactly the drivers
  that have not been checked against their own documentation.
  """
  declared = getattr(backend, "per_command_step_mm", None)
  if declared is None:
    return None
  value = float(declared)
  if not math.isfinite(value) or value <= 0.0:
    raise ValueError(
      f"{type(backend).__name__} declares per_command_step_mm={declared!r}, which is "
      "not a positive finite number of millimetres. An unusable declaration is worse "
      "than none: it reads as a checked limit."
    )
  return value


def clamped_fraction(episode: Episode) -> float:
  """Share of an episode's steps the envelope rewrote. NaN for an empty episode.

  NaN and not 0.0. A run with no steps had nothing clamped in the same sense that
  it had nothing succeed, and a zero here averages into a control suite as
  evidence that the envelope was not interfering.
  """
  n = len(episode)
  if n <= 0:
    return float("nan")
  return float(int(episode.metadata.get("n_clamped", 0)) / n)


def is_throttled(fraction: float) -> bool:
  """Whether a clamped share invalidates a null control. See :data:`THROTTLED`.

  A NaN fraction is throttled: an unmeasured clamp rate is not a low one, and the
  control that reports it established nothing about the envelope's involvement.
  """
  value = float(fraction)
  if math.isnan(value):
    return True
  return value > THROTTLED_CLAMP_FRACTION


def _clamp_magnitude_mm(requested: Action, safe: Action, pose: np.ndarray) -> float:
  """How far the executed command sits from the one that was asked for, in mm.

  Zero for a joint-space action, which has no Cartesian displacement to compare;
  that is recorded as zero rather than NaN because the envelope genuinely did not
  move it, and the accompanying note already says the step went through unchecked.
  """
  if requested.mode is ActionMode.JOINTS or safe.mode is ActionMode.JOINTS:
    return 0.0
  try:
    asked = np.asarray(requested.as_delta(pose), float).reshape(4)
    sent = np.asarray(safe.as_delta(pose), float).reshape(4)
  except (TypeError, ValueError):
    return 0.0
  if not np.all(np.isfinite(asked)) or not np.all(np.isfinite(sent)):
    return 0.0
  return float(np.linalg.norm(asked[:3] - sent[:3]))


#: What ``rollout(screen=...)`` needs. Deliberately not the screen type itself:
#: importing it here would close a cycle through ``predict``, and the loop only
#: ever needs one method.
_SCREEN_REPORT_ATTRS = ("admitted", "reobserve", "to_dict", "clamped_downgrade")


def _reviewed(screen: Any, observation: Observation, action: Action) -> Any:
  """Ask the screen for its verdict on this step, and hold it to the seam.

  Structural rather than an isinstance check, because ``plr_lr.learn.screen``
  imports ``plr_lr.learn.predict``, which imports this module for the envelope.
  The attributes checked here are exactly the ones the loop below calls.
  """
  report = screen.review(observation, action)
  if inspect.isawaitable(report):
    close = getattr(report, "close", None)
    if callable(close):
      close()
    raise UnsafeAction(
      "screen.review() returned an awaitable. Screening is synchronous; this loop "
      "owns the async boundary, and a verdict awaited mid-step cannot be ordered "
      "against the command it decides."
    )
  missing = [attr for attr in _SCREEN_REPORT_ATTRS if not hasattr(report, attr)]
  if missing:
    raise UnsafeAction(
      f"screen.review() returned {type(report).__name__}, which is missing {missing}. "
      "Refusing before any motion: a verdict object this loop cannot read is one it "
      "would have to treat as an admission."
    )
  return report


def _has_motion(action: Action, pose: np.ndarray) -> bool:
  if action.mode is ActionMode.JOINTS:
    return bool(action.joints)
  return bool(np.any(np.abs(action.as_delta(pose)) > 1e-12))


async def apply_action(arm: ExperimentalSCARA, action: Action, pose: Sequence[float]) -> None:
  """Execute one action on the arm.

  The gripper slot of an action is a target width. A width below the current one
  is executed as a squeeze (``close_gripper``) and a width above it as a release
  (``open_gripper``), so a policy only has to emit a number. When the backend
  cannot report a current width the direction is genuinely unknown, and this
  refuses rather than picking one: on the xArm both calls issue the same physical
  move and differ only in whether the arm afterwards believes it is holding
  something, so guessing does not damage hardware, it corrupts the state every
  later decision is made from.

  Note that this drives pose and gripper only. It does not update the deck
  ``World``: which site holds what is bookkeeping the ``Workcell`` pick and drop
  calls maintain, and a policy moving the tool does not change it.
  """
  try:
    p = np.asarray(pose, float).reshape(4)
  except (TypeError, ValueError) as exc:
    raise UnsafeAction("current pose must contain four finite values") from exc
  if not np.all(np.isfinite(p)):
    raise UnsafeAction(f"current pose must contain four finite values, got {p.tolist()}")

  current_width: Optional[float] = None
  if action.gripper is not None:
    try:
      width = float(action.gripper)
    except (TypeError, ValueError, OverflowError) as exc:
      raise UnsafeAction(
        f"gripper target must be a finite non-negative width in mm, got {action.gripper!r}"
      ) from exc
    if not math.isfinite(width) or width < 0.0:
      raise UnsafeAction(
        f"gripper target must be a finite non-negative width in mm, got {action.gripper!r}"
      )
    current_width = gripper_width_of(arm.backend)
    if current_width is None:
      raise UnsafeAction(
        f"cannot tell whether commanding {width:.2f} mm opens or closes: "
        f"{type(arm.backend).__name__} reports no current gripper width. Command "
        "the gripper once through the arm API so the backend has a width to "
        "compare against, then run the policy."
      )
    if not math.isfinite(current_width):
      raise UnsafeAction(
        f"{type(arm.backend).__name__} reported a non-finite current gripper width; "
        "refusing before any pose or gripper motion"
      )

  delta: Optional[np.ndarray] = None
  if action.mode is ActionMode.JOINTS:
    if action.joints:
      try:
        joints = np.asarray(tuple(action.joints.values()), float)
      except (TypeError, ValueError) as exc:
        raise UnsafeAction("joint targets must contain only finite values") from exc
      if not np.all(np.isfinite(joints)):
        raise UnsafeAction("joint targets must contain only finite values")
  elif action.mode in (ActionMode.DELTA, ActionMode.ABSOLUTE):
    try:
      delta = np.asarray(action.as_delta(p), float).reshape(4)
    except (TypeError, ValueError) as exc:
      raise UnsafeAction("Cartesian action must contain four finite values") from exc
    if not np.all(np.isfinite(delta)):
      raise UnsafeAction(f"Cartesian action must contain only finite values, got {delta.tolist()}")
  else:
    raise UnsafeAction(f"unsupported action mode {action.mode!r}")

  if action.mode is ActionMode.JOINTS:
    if action.joints:
      await arm.move_to({int(k): float(v) for k, v in action.joints.items()})
  elif delta is not None and np.any(np.abs(delta) > 1e-12):
    target = p + delta
    current = await arm.get_cartesian_position()
    if not np.all(
      np.isfinite([current.rotation.x, current.rotation.y, current.rotation.z])
    ):
      raise UnsafeAction("arm reported a non-finite Cartesian rotation")
    await arm.move_to(
      PreciseFlexCartesianCoords(
        location=Coordinate(float(target[0]), float(target[1]), float(target[2])),
        # PyLabRobot's Rotation is degrees; the learning action and pose are
        # radians. This is the outbound half of the unit boundary paired with
        # record.observe().
        rotation=Rotation(current.rotation.x, current.rotation.y, math.degrees(float(target[3]))),
        orientation=current.orientation,
      )
    )

  if action.gripper is not None:
    width = float(action.gripper)
    assert current_width is not None  # checked before any pose motion above
    if width < current_width:
      await arm.close_gripper(width)
    else:
      await arm.open_gripper(width)


async def _choose(policy: Any, observation: Observation) -> Action:
  action = policy.act(observation)
  if inspect.isawaitable(action):
    action = await action
  if not isinstance(action, Action):
    raise UnsafeAction(
      f"policy.act() must return an Action, got {type(action).__name__}. "
      "Refusing before the value reaches the safety envelope."
    )
  return action


async def rollout(
  arm: ExperimentalSCARA,
  policy: Any,
  envelope: Optional[SafetyEnvelope] = None,
  task: str = "",
  max_steps: Optional[int] = None,
  recorder: Optional[EpisodeRecorder] = None,
  detection_hook: Optional[Any] = None,
  hz: Optional[float] = None,
  allow_hardware: bool = False,
  reset: bool = True,
  screen: Optional[Any] = None,
) -> Episode:
  """Step ``policy`` against ``arm`` and return the recorded episode.

  ``screen`` is optional and defaults to None, in which case this behaves exactly
  as it did before the argument existed: no verdict is requested, no ``screen``
  key appears on any step, and the recorded episode is identical to one produced
  without passing the argument at all. That equivalence is asserted by a test that
  runs the same seed both ways and compares the whole episode, because "the new
  argument changes nothing when it is None" is the kind of claim that is true when
  written and false three commits later.

  With a screen supplied, each step is decided before the envelope sees it and
  again after. A verdict that was admitted and is then clamped becomes
  ``ABSTAIN(CLAMPED)``: the envelope has rewritten the command, so the verdict is
  about something that is not being sent. Nothing is executed on a refusal, the
  refusal is recorded as a step, and the run stops unless the screen still has a
  bounded re-observation budgeted.

  Raises:
    HardwarePermissionRequired: if the arm is not simulated and
      ``allow_hardware`` was not explicitly set.
    StepLimitExceeded: if a clamped command is larger than the backend declares it
      will execute.
  """
  if screen is not None and not callable(getattr(screen, "review", None)):
    raise TypeError(
      f"screen must have review(observation, action) -> ScreenReport, and "
      f"{type(screen).__name__} does not. plr_lr.learn.screen.ScreenedPolicy is the "
      "thing that has one; a bare Screen decides a prediction, and this loop has "
      "never seen a prediction."
    )
  if hz is not None:
    if isinstance(hz, bool):
      raise ValueError(f"hz must be a positive finite number, got {hz!r}")
    try:
      hz = float(hz)
    except (TypeError, ValueError) as exc:
      raise ValueError(f"hz must be a positive finite number, got {hz!r}") from exc
    if not math.isfinite(hz) or hz <= 0.0:
      raise ValueError(f"hz must be a positive finite number, got {hz!r}")

  if not is_simulation(arm) and allow_hardware is not True:
    raise HardwarePermissionRequired(
      f"{type(arm.backend).__name__} is not the simulation backend. Running an "
      "unreviewed policy on a real arm needs allow_hardware=True at the call site. "
      "Evaluate the policy in simulation first: see plr_lr.learn.evaluate.evaluate, "
      "and confirm its null tests passed."
    )

  envelope = envelope or SafetyEnvelope()
  if max_steps is None:
    budget = envelope.max_steps
  else:
    if isinstance(max_steps, bool) or not isinstance(max_steps, Integral) or max_steps < 1:
      raise ValueError(f"max_steps must be a positive integer, got {max_steps!r}")
    # A call-site budget may make an envelope stricter, but cannot silently
    # enlarge the safety budget declared on the envelope itself.
    budget = min(envelope.max_steps, int(max_steps))
  rec = recorder or EpisodeRecorder(arm, task=task, detection_hook=detection_hook)
  if reset:
    reset_policy(policy)

  # The envelope's own declaration wins when it has one, so a caller who knows
  # the controller's limit can state it without touching the backend. Otherwise
  # the backend is asked, because it is the thing that knows.
  if envelope.per_command_step_mm is None:
    declared = per_command_step_mm_of(arm.backend)
    if declared is not None:
      envelope = replace(envelope, per_command_step_mm=declared)

  reason = "budget_exhausted"
  clamped_steps = 0
  clamp_total_mm = 0.0
  for _ in range(budget):
    obs = await rec.observe()
    pose = obs.pose_vector()
    requested = await _choose(policy, obs)

    # Screened before the envelope, because that is the command the screen is
    # being asked about. Screening the clamped one instead would make every
    # rewritten command look like the one the model proposed.
    report = _reviewed(screen, obs, requested) if screen is not None else None

    safe, notes = envelope.clamp(requested, pose)
    step_clamped = bool(notes)
    magnitude = _clamp_magnitude_mm(requested, safe, pose) if step_clamped else 0.0
    if step_clamped:
      clamped_steps += 1
      clamp_total_mm += magnitude

    info: dict = {"n_clamped": int(step_clamped), "clamp_magnitude_mm": magnitude}
    if notes:
      info["clamped"] = notes
      info["requested"] = requested.to_dict()

    declined = False
    if report is not None:
      if step_clamped and report.admitted:
        report = report.clamped_downgrade(magnitude)
      declined = not report.admitted
      if declined:
        info.setdefault("requested", requested.to_dict())
        # No fallback action. The refusal is a hold, and a hold is the absence of
        # a command rather than a gentler one.
        safe = Action.hold()
      info["screen"] = report.to_dict()

    rec.log(obs, safe, **info)

    if _has_motion(safe, pose) or safe.gripper is not None:
      await apply_action(arm, safe, pose)
    if declined and not bool(report.reobserve):
      reason = "screen_declined"
      break
    if safe.terminate:
      reason = "policy_terminated"
      break
    if hz:
      await asyncio.sleep(1.0 / hz)

  steps = len(rec.episode)
  return await rec.finish(
    stop_reason=reason,
    steps=steps,
    clamped_steps=clamped_steps,
    n_clamped=clamped_steps,
    clamp_magnitude_mm=clamp_total_mm,
    clamped_fraction=(float(clamped_steps / steps) if steps else float("nan")),
    screened=screen is not None,
    simulated=is_simulation(arm),
  )
