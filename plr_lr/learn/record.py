"""Capture what the arm did, so it can be learned from.

Two ways in, both producing the same ``Episode``:

* :class:`EpisodeRecorder` taps an arm while something else drives it. That
  something can be a scripted workcell routine, a hand-written skill, or a
  policy under :func:`plr_lr.learn.rollout.rollout`. Nothing about the driver is
  assumed, so the same recorder works against the simulation backend and against
  a real xArm.
* :class:`KinestheticSession` records a demonstration given by hand: put the arm
  in freedrive, drag it through the task, and sample the pose as it moves. This
  is the cheapest useful source of training data for a lab arm, because it needs
  no teleoperation rig and no teach pendant program.

One honest note about freedrive in simulation. On hardware the pose samples come
from the controller's encoders, so dragging the arm produces a real trajectory.
The simulation backend has no physics and no human hand: its reported pose is
whatever was last commanded, so it will not move on its own. Pass ``pose_source``
to supply the trajectory when you want to exercise the capture path without
hardware, and read the resulting episode as synthetic rather than demonstrated.

That last sentence used to be the only place the distinction lived. Every
episode carries ``metadata["pose_source"]`` now, set to :data:`POSE_SOURCE_ENCODER`
when the poses were read back off the arm and :data:`POSE_SOURCE_INJECTED` when a
caller supplied them, and an injected capture is never labelled ``kinesthetic``.
The failure this prevents is specific: a dataset whose every episode says a human
dragged the arm through the task, recorded on a machine with no arm attached,
where nothing on disk distinguishes it from one that was really demonstrated. A
model trained on it inherits the claim, and so does every paper that cites the
dataset.
"""

from __future__ import annotations

import asyncio
import math
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from pylabrobot.arms import ExperimentalSCARA

from ..vision.detector import Detection
from .types import Action, Episode, Observation

Clock = Callable[[], float]
DetectionHook = Callable[[], Sequence[Detection]]
PoseSource = Callable[[int, float], Optional[Tuple[float, float, float, float]]]

#: ``metadata["pose_source"]`` when every pose in the episode was read back from
#: the arm. It is the only value under which a trajectory may claim to be a
#: physical record of anything.
POSE_SOURCE_ENCODER = "encoder"

#: ``metadata["pose_source"]`` when a caller supplied the poses. The capture path
#: really ran, the freedrive calls really happened, and the trajectory in the
#: episode is still whatever the caller decided it should be.
POSE_SOURCE_INJECTED = "injected"

#: ``metadata["source"]`` for a demonstration a person actually gave by hand.
#: Reserved for :data:`POSE_SOURCE_ENCODER` captures; see
#: :meth:`KinestheticSession.capture`.
SOURCE_KINESTHETIC = "kinesthetic"

#: ``metadata["source"]`` for a capture whose poses were injected. A different
#: word, not a qualifier on the same one, so that filtering a mixed dataset down
#: to real demonstrations is an equality test rather than a careful read.
SOURCE_SYNTHETIC = "synthetic"


#: Backends disagree about where the gripper width lives, and none of them put it
#: on the ``ExperimentalSCARA`` front end. The simulation backend keeps
#: ``_gripper_width`` and initialises it. The xArm backend keeps
#: ``_last_commanded_width_mm`` and leaves it None until the gripper has actually
#: been commanded, because before that it has no idea how open the jaws are.
_WIDTH_ATTRS = ("_gripper_width", "_last_commanded_width_mm")


def gripper_width_of(backend: object) -> Optional[float]:
  """The gripper opening in mm, or None when the backend does not know one.

  Returning None rather than 0.0 is the point of this function. A width reported
  as 0.0 for an arm that simply does not track one becomes a constant column in
  every recorded episode, and a constant column is indistinguishable from a real
  measurement that never changed.
  """
  for attr in _WIDTH_ATTRS:
    value = getattr(backend, attr, None)
    if value is not None:
      return float(value)
  return None


async def observe(
  arm: ExperimentalSCARA,
  t: float = 0.0,
  detections: Sequence[Detection] = (),
  image_path: Optional[str] = None,
) -> Observation:
  """Read one observation out of an arm, simulated or real.

  Everything except the gripper width comes through the public arm API. The
  width is read from the backend by :func:`gripper_width_of`, and is None when
  the backend cannot say.
  """
  joints = await arm.get_joint_position()
  cart = await arm.get_cartesian_position()
  closed = await arm.is_gripper_closed()
  width = gripper_width_of(arm.backend)
  holding = bool(getattr(arm.backend, "holding", False))
  loc = cart.location
  return Observation(
    t=float(t),
    x=float(loc.x),
    y=float(loc.y),
    z=float(loc.z),
    # PyLabRobot's Rotation is expressed in degrees. The learning surface uses
    # radians everywhere (Action, Goal, dynamics, and the safety envelope), so
    # this read is the inbound unit boundary. Leaving the value in degrees is
    # especially dangerous because both representations are ordinary floats and
    # a small rotation still looks plausible.
    rz=math.radians(float(cart.rotation.z)),
    gripper_width=width,
    gripper_closed=bool(closed),
    holding=holding,
    joints={int(k): float(v) for k, v in joints.items()},
    detections=tuple(detections),
    image_path=image_path,
  )


class EpisodeRecorder:
  """Records observations and actions into an :class:`Episode`.

  Usage is two calls per step. Observe before acting, then log the action that
  was chosen for that observation::

      rec = EpisodeRecorder(arm, task="move plate to reader")
      obs = await rec.observe()
      rec.log(obs, action)
      ... execute the action ...
      episode = await rec.finish()

  ``finish`` records one last observation so the final action has an outcome,
  which is what makes the episode usable for fitting dynamics.
  """

  def __init__(
    self,
    arm: ExperimentalSCARA,
    task: str = "",
    clock: Optional[Clock] = None,
    detection_hook: Optional[DetectionHook] = None,
    metadata: Optional[Dict[str, Any]] = None,
  ) -> None:
    self.arm = arm
    self.task = task
    self.clock = clock or time.monotonic
    self.detection_hook = detection_hook
    self.episode = Episode(task=task, metadata=dict(metadata or {}))
    self._t0: Optional[float] = None

  def _elapsed(self) -> float:
    now = self.clock()
    if self._t0 is None:
      self._t0 = now
    return float(now - self._t0)

  async def observe(self, image_path: Optional[str] = None) -> Observation:
    """Read the arm's current state, stamped with time since the first read."""
    dets = tuple(self.detection_hook()) if self.detection_hook else ()
    return await observe(self.arm, t=self._elapsed(), detections=dets, image_path=image_path)

  def log(self, observation: Observation, action: Action, **info: Any) -> None:
    """Record that ``action`` was chosen in ``observation``."""
    self.episode.append(observation, action, **info)

  async def finish(self, **metadata: Any) -> Episode:
    """Take a closing observation and return the finished episode."""
    self.episode.final_observation = await self.observe()
    self.episode.metadata.update(metadata)
    return self.episode


class KinestheticSession:
  """Record a demonstration given by dragging the arm in freedrive.

  The session frees every axis, samples at ``hz`` until ``duration_s`` has
  elapsed or ``stop()`` is called, then returns freedrive and converts the pose
  samples into an episode whose actions are the deltas between consecutive
  samples. Those actions are exactly what a delta-action policy is trained to
  emit, so a captured demo can be replayed through the same runner that drives a
  learned policy.

  Freedrive is all-or-nothing on the xArm controller, and the hardware backend
  refuses a partial axis list rather than pretending a subset was honored, so
  this session always asks for every axis the arm reports.
  """

  def __init__(
    self,
    arm: ExperimentalSCARA,
    task: str = "",
    hz: float = 10.0,
    clock: Optional[Clock] = None,
    pose_source: Optional[PoseSource] = None,
    detection_hook: Optional[DetectionHook] = None,
  ) -> None:
    if hz <= 0:
      raise ValueError("hz must be positive")
    self.arm = arm
    self.task = task
    self.hz = hz
    self.clock = clock or time.monotonic
    self.pose_source = pose_source
    self.detection_hook = detection_hook
    self._stop = False

  def stop(self) -> None:
    """Ask a running capture to finish after the current sample."""
    self._stop = True

  @property
  def pose_provenance(self) -> str:
    """Where this session's poses will come from, decided before it runs.

    Read off the presence of ``pose_source`` rather than off what the capture
    observed, because the two can agree by accident: an injected trajectory that
    happens to replay the arm's own last-commanded pose is indistinguishable from
    an encoder read, and the one thing provenance must never do is depend on the
    values it is describing.
    """
    return POSE_SOURCE_ENCODER if self.pose_source is None else POSE_SOURCE_INJECTED

  async def _free_axes(self) -> List[int]:
    joints = await self.arm.get_joint_position()
    return sorted(int(k) for k in joints)

  async def capture(self, duration_s: float, max_samples: int = 10_000) -> Episode:
    """Drag-through capture. Returns an episode of delta actions.

    The episode is labelled ``source="kinesthetic"`` only when its poses came off
    the arm. With a ``pose_source`` injected the trajectory is whatever the caller
    generated, no hand touched anything, and the label is ``"synthetic"``. Both
    carry ``pose_source`` explicitly, so the distinction survives export to disk
    instead of living in whoever remembers how the file was made.
    """
    if duration_s <= 0:
      raise ValueError("duration_s must be positive")

    self._stop = False
    provenance = self.pose_provenance
    rec = EpisodeRecorder(
      self.arm,
      task=self.task,
      clock=self.clock,
      detection_hook=self.detection_hook,
      metadata={
        "source": (
          SOURCE_KINESTHETIC if provenance == POSE_SOURCE_ENCODER else SOURCE_SYNTHETIC
        ),
        "pose_source": provenance,
        "hz": self.hz,
      },
    )
    dt = 1.0 / self.hz

    # Entering freedrive is inside the try, not above it. On the xArm it is two
    # controller calls, set_mode(TEACH) then set_state(READY), and the first one
    # already makes the arm physically compliant. If the second raises and the
    # entry sat outside the try, the finally never runs and a compliant arm with
    # a payload on it is handed back to a caller who was told the capture failed.
    try:
      await self.arm.freedrive_mode(await self._free_axes())
      samples: List[Observation] = []
      i = 0
      t_start = self.clock()
      while not self._stop and i < max_samples:
        obs = await rec.observe()
        if self.pose_source is not None:
          injected = self.pose_source(i, obs.t)
          if injected is None:
            break
          x, y, z, rz = injected
          obs = Observation(
            t=obs.t,
            x=float(x),
            y=float(y),
            z=float(z),
            rz=float(rz),
            gripper_width=obs.gripper_width,
            gripper_closed=obs.gripper_closed,
            holding=obs.holding,
            joints=obs.joints,
            detections=obs.detections,
            image_path=obs.image_path,
          )
        samples.append(obs)
        i += 1
        if self.clock() - t_start >= duration_s:
          break
        await asyncio.sleep(dt)
    finally:
      await self.arm.end_freedrive_mode()

    return self._to_episode(samples, rec.episode.metadata)

  def _to_episode(self, samples: Sequence[Observation], metadata: Dict[str, Any]) -> Episode:
    """Turn a pose stream into delta actions between consecutive samples."""
    ep = Episode(task=self.task, metadata=dict(metadata))
    ep.metadata["n_samples"] = len(samples)
    for a, b in zip(samples, samples[1:]):
      raw_drz = b.rz - a.rz
      action = Action.delta(
        dx=b.x - a.x,
        dy=b.y - a.y,
        dz=b.z - a.z,
        # Controllers commonly wrap reported wrist angles at +/-pi. Crossing
        # that representation boundary is a small physical turn, not an almost
        # full revolution in the opposite direction.
        drz=math.atan2(math.sin(raw_drz), math.cos(raw_drz)),
        gripper=b.gripper_width if b.gripper_width != a.gripper_width else None,
      )
      ep.append(a, action)
    if samples:
      ep.final_observation = samples[-1]
    return ep


async def record_scripted(
  arm: ExperimentalSCARA,
  routine: Callable[[], Awaitable[None]],
  task: str = "",
  clock: Optional[Clock] = None,
) -> Episode:
  """Record the endpoints of a scripted routine as a single absolute action.

  A scripted skill drives the arm through calls this recorder cannot see from
  the outside, so what it captures is the before and after, not the path taken.
  That is enough to score whether the routine reached its goal and not enough to
  train a controller on. Use :class:`KinestheticSession` or
  :func:`plr_lr.learn.rollout.rollout` when the path matters.
  """
  rec = EpisodeRecorder(
    arm,
    task=task,
    clock=clock,
    # Both endpoints are read off the arm, so this one is an encoder record
    # whatever the routine did in between. Stamped rather than left absent: a
    # missing provenance field reads as "not known to be synthetic", and a
    # dataset mixing this with a kinesthetic capture would then have one episode
    # kind that says where its poses came from and one that does not.
    metadata={"source": "scripted", "pose_source": POSE_SOURCE_ENCODER},
  )
  before = await rec.observe()
  await routine()
  after = await observe(arm, t=rec._elapsed())
  rec.log(before, Action.absolute(after.x, after.y, after.z, after.rz, terminate=True))
  return await rec.finish()
