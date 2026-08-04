"""Tests for plr_lr.learn.record.

An episode is training data, so what is worth testing is not that the recorder
runs but that what it wrote is what happened. These cover the five ways a
recording can quietly lie: a timestamp that does not start at the start, a final
action whose outcome was never observed (which drops the last transition out of
every fit without saying so), a delta that is not the difference between the two
samples it sits between, a freedrive session that leaves the arm compliant on
the way out, and an episode that says a person demonstrated a trajectory that was
generated in software.

Nothing here reads the wall clock. Every recorder is handed a ``FakeClock``, so a
recorded timestamp is an input to the test rather than a measurement of how busy
the machine was.

The freedrive scope tests are run against ``DegreeReportingBackend`` as well as
the simulation one, and neither is a driver. Simulation accepts any axis list and
cannot fail partway through entering freedrive, so against it alone a session
that freed a subset, or that left an arm compliant after a failure, reads exactly
like a correct one. The double refuses a partial list and enters freedrive in two
steps so the second can be broken, which is the only reason it is here.
"""

from __future__ import annotations

import math
import unittest

from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.backend import VerticalAccess
from pylabrobot.arms.precise_flex.coords import ElbowOrientation, PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from plr_lr import Labware, SimulationArmBackend, Workcell
from plr_lr.learn import (
  Action,
  ActionMode,
  Episode,
  EpisodeRecorder,
  KinestheticSession,
  observe,
  record_scripted,
)
from plr_lr.learn.record import (
  POSE_SOURCE_ENCODER,
  POSE_SOURCE_INJECTED,
  SOURCE_KINESTHETIC,
  SOURCE_SYNTHETIC,
  gripper_width_of,
)
from plr_lr.vision import Detection
from tests._doubles import (
  MODE_COMPLIANT,
  MODE_MOTION,
  ControllerRefused,
  DegreeReportingBackend,
  homed_arm,
)

# A drag-through used by most of the capture tests: four (x, y, z, rz) samples
# that move on every axis, so a delta that dropped one would not survive.
TRACK = (
  (100.0, 0.0, 50.0, 0.0),
  (110.0, 5.0, 50.0, 0.1),
  (125.0, 5.0, 45.0, 0.1),
  (125.0, -5.0, 45.0, 0.25),
)


class DragInterrupted(RuntimeError):
  """Raised from inside a drag-through, to prove freedrive is left on the way out."""


class FakeClock:
  """A clock the test owns, so a recorded timestamp is an input not a measurement.

  Every call advances by a fixed step. Injected wherever a recorder would
  otherwise read ``time.monotonic``, because these tests assert on stamps
  exactly and against the real clock those assertions would be measuring load.
  """

  def __init__(self, step: float = 0.25, start: float = 0.0) -> None:
    self.step = step
    self.now = start

  def __call__(self) -> float:
    value = self.now
    self.now += self.step
    return value


class BinaryGripperBackend(SimulationArmBackend):
  """An arm whose gripper reports open or closed and no width at all.

  Binary grippers are ordinary lab hardware, and an observation of one has to
  say it does not know the width rather than report a number nobody measured.
  """

  def __init__(self, **kwargs) -> None:
    super().__init__(**kwargs)
    del self._gripper_width

  async def open_gripper(self, gripper_width: float) -> None:
    self._gripper_closed = False

  async def close_gripper(self, gripper_width: float) -> None:
    self._gripper_closed = True


def coords(x: float, y: float, z: float, rz: float = 0.0):
  """Build PyLabRobot coordinates from a learning-native pose in radians."""
  return PreciseFlexCartesianCoords(
    location=Coordinate(x, y, z),
    rotation=Rotation(0.0, 0.0, math.degrees(rz)),
    orientation=ElbowOrientation.RIGHT,
  )


async def sim_arm(backend=None) -> ExperimentalSCARA:
  arm = ExperimentalSCARA(backend=backend if backend is not None else SimulationArmBackend())
  await arm.setup()
  await arm.home()
  return arm


def replay(track):
  """A ``pose_source`` that plays a fixed trajectory and then ends the capture."""

  def source(i: int, t: float):
    return track[i] if i < len(track) else None

  return source


def dragged(arm: ExperimentalSCARA, track, task: str = "drag to the reader"):
  """A session whose drag-through is exactly ``track``.

  ``hz`` is high so the loop's pacing sleep is negligible, and the trajectory
  rather than the clock decides when the capture ends, so no assertion here
  depends on how long a sleep actually took.
  """
  return KinestheticSession(
    arm,
    task=task,
    hz=1000.0,
    clock=FakeClock(step=0.05),
    pose_source=replay(track),
  )


def freedrive_entries(backend):
  """Just the freedrive transitions from a backend trace, in the order they happened."""
  return [e["action"] for e in backend.trace if e["action"] in ("freedrive", "end_freedrive")]


class ObserveTests(unittest.IsolatedAsyncioTestCase):
  """One read of the arm, into the layout every model downstream consumes."""

  async def test_observe_reads_the_pose_gripper_and_joints_it_was_pointed_at(self):
    arm = await sim_arm()
    await arm.move_to(coords(120.0, -40.0, 25.0, 0.75))
    await arm.move_to({1: 12.5, 3: -7.25})
    await arm.close_gripper(64.0)

    raw = await arm.get_cartesian_position()
    self.assertAlmostEqual(raw.rotation.z, math.degrees(0.75))
    obs = await observe(arm, t=3.5)

    self.assertEqual(obs.t, 3.5)
    self.assertAlmostEqual(obs.x, 120.0)
    self.assertAlmostEqual(obs.y, -40.0)
    self.assertAlmostEqual(obs.z, 25.0)
    self.assertAlmostEqual(obs.rz, 0.75)
    self.assertAlmostEqual(obs.gripper_width, 64.0)
    self.assertTrue(obs.gripper_closed)
    self.assertFalse(obs.holding)
    self.assertEqual(obs.joints, {1: 12.5, 2: 0.0, 3: -7.25, 4: 0.0, 5: 0.0})
    self.assertTrue(all(isinstance(k, int) for k in obs.joints))
    self.assertEqual(obs.detections, ())
    self.assertIsNone(obs.image_path)

  async def test_observe_converts_backend_degrees_to_learning_radians(self):
    """A backend that keeps radians exposes PyLabRobot degrees at this boundary.

    The double stores the wrist angle in radians and only ever publishes it
    through a PyLabRobot ``Rotation``, which is degrees. So the first assertion
    establishes that the two representations really do differ here, and the
    second is then a measurement of the conversion rather than a restatement of
    whatever the test itself put in.
    """
    backend = DegreeReportingBackend(start=(200.0, -20.0, 300.0, 0.75), roll_rad=math.pi)
    arm = await homed_arm(backend)

    raw = await arm.get_cartesian_position()
    self.assertAlmostEqual(raw.rotation.z, math.degrees(0.75))

    obs = await observe(arm)
    self.assertAlmostEqual(obs.rz, 0.75)

  async def test_the_state_vector_carries_the_observation_it_was_read_from(self):
    """The field order is the contract every downstream fit is written against."""
    arm = await sim_arm()
    await arm.move_to(coords(120.0, -40.0, 25.0, 0.75))
    await arm.move_to({2: 30.0})
    await arm.close_gripper(64.0)

    obs = await observe(arm)
    vector = obs.state_vector()

    self.assertEqual(len(vector), len(obs.state_fields()))
    fields = dict(zip(obs.state_fields(), vector.tolist()))
    self.assertAlmostEqual(fields["x"], 120.0)
    self.assertAlmostEqual(fields["y"], -40.0)
    self.assertAlmostEqual(fields["z"], 25.0)
    self.assertAlmostEqual(fields["rz"], 0.75)
    self.assertAlmostEqual(fields["gripper_width"], 64.0)
    self.assertEqual(fields["gripper_closed"], 1.0)
    self.assertEqual(fields["holding"], 0.0)
    self.assertAlmostEqual(fields["j2"], 30.0)

  async def test_a_held_plate_is_observed_as_held_at_the_width_it_was_gripped(self):
    wc = Workcell.sim()
    await wc.setup()
    plate = Labware(name="sample_plate")
    wc.add_site("stage", x=150.0, y=0.0, z=10.0, occupant=plate)
    await wc.arm.pick_up_resource(wc.coords_for("stage"), plate_width=plate.plate_width)

    obs = await observe(wc.arm)

    self.assertTrue(obs.holding)
    self.assertTrue(obs.gripper_closed)
    self.assertAlmostEqual(obs.gripper_width, plate.plate_width)
    self.assertAlmostEqual(obs.x, 150.0)
    self.assertAlmostEqual(obs.z, 10.0 + VerticalAccess().clearance_mm)
    self.assertEqual(obs.t, 0.0)

  async def test_a_backend_that_tracks_no_width_reports_none_and_not_a_zero(self):
    """A width of 0.0 for an arm that measures none is a fabricated column."""
    arm = await sim_arm(BinaryGripperBackend())
    await arm.close_gripper(64.0)

    obs = await observe(arm)

    self.assertIsNone(obs.gripper_width)
    self.assertTrue(obs.gripper_closed)

  def test_the_width_seam_is_by_attribute_name_on_both_backends(self):
    """A rename on either backend has to fail here.

    The alternative is a dataset with a silently constant width column. The two
    backends keep the width under two different attribute names, and both names
    are read back with a value: a None answer alone would pass whether the name
    were honoured or simply absent, so it proves nothing about either seam.
    """
    uncommanded = DegreeReportingBackend()
    self.assertIsNone(gripper_width_of(uncommanded))
    self.assertAlmostEqual(gripper_width_of(DegreeReportingBackend(commanded_width_mm=64.0)), 64.0)
    self.assertAlmostEqual(gripper_width_of(SimulationArmBackend()), 90.0)

  async def test_detections_and_the_frame_they_came_from_ride_along(self):
    arm = await sim_arm()
    det = Detection(label="plate", uv=(640.0, 512.0), confidence=0.9, truth_name="plate")

    obs = await observe(arm, t=1.0, detections=[det], image_path="frame_000.png")

    self.assertEqual(obs.detections, (det,))
    self.assertEqual(obs.image_path, "frame_000.png")


class EpisodeRecorderTests(unittest.IsolatedAsyncioTestCase):
  """Timestamps that mean something, and a last action that has an outcome."""

  async def test_timestamps_start_at_zero_and_increase(self):
    arm = await sim_arm()
    # A nonzero start proves the absolute clock origin is subtracted out rather
    # than shipped into the episode.
    rec = EpisodeRecorder(arm, task="move plate", clock=FakeClock(step=0.25, start=17.0))

    stamps = [(await rec.observe()).t for _ in range(3)]

    self.assertEqual(stamps[0], 0.0)
    self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])), stamps)
    self.assertAlmostEqual(stamps[1], 0.25)
    self.assertAlmostEqual(stamps[2], 0.5)

  async def test_the_epoch_is_the_first_observation_not_the_constructor(self):
    arm = await sim_arm()
    clock = FakeClock(step=0.25)
    rec = EpisodeRecorder(arm, clock=clock)
    clock()
    clock()  # time passes between building the recorder and the first read

    self.assertEqual((await rec.observe()).t, 0.0)

  async def test_finish_gives_the_last_logged_action_an_outcome(self):
    arm = await sim_arm()
    rec = EpisodeRecorder(arm, task="reach", clock=FakeClock(step=0.5))

    first = await rec.observe()
    rec.log(first, Action.absolute(150.0, -20.0, 30.0, terminate=True), source="test")
    self.assertEqual(rec.episode.transitions(), [])  # nothing observed the outcome yet

    await arm.move_to(coords(150.0, -20.0, 30.0))
    ep = await rec.finish(operator="unit test")

    self.assertIsNotNone(ep.final_observation)
    self.assertAlmostEqual(ep.final_observation.x, 150.0)
    self.assertAlmostEqual(ep.final_observation.y, -20.0)
    self.assertAlmostEqual(ep.final_observation.z, 30.0)
    self.assertGreater(ep.final_observation.t, first.t)

    transitions = ep.transitions()
    self.assertEqual(len(transitions), 1)
    self.assertIs(transitions[0][0], first)
    self.assertIs(transitions[0][2], ep.final_observation)
    self.assertEqual(ep.task, "reach")
    self.assertEqual(ep.steps[0].info, {"source": "test"})
    self.assertEqual(ep.metadata["operator"], "unit test")

  async def test_the_finished_episode_reports_how_long_it_took(self):
    """Duration is what tells a reader whether a demo took 3 seconds or 300."""
    arm = await sim_arm()
    rec = EpisodeRecorder(arm, clock=FakeClock(step=0.5))

    rec.log(await rec.observe(), Action.hold())
    ep = await rec.finish()

    self.assertAlmostEqual(ep.duration, 0.5)

  async def test_the_detection_hook_is_read_into_every_observation(self):
    arm = await sim_arm()
    det = Detection(label="plate", uv=(640.0, 512.0), confidence=0.9)
    rec = EpisodeRecorder(arm, clock=FakeClock(), detection_hook=lambda: [det])

    obs = await rec.observe(image_path="frame_000.png")
    ep = await rec.finish()

    self.assertEqual(obs.detections, (det,))
    self.assertEqual(obs.image_path, "frame_000.png")
    self.assertEqual(ep.final_observation.detections, (det,))
    self.assertIsNone(ep.final_observation.image_path)


class KinestheticSessionTests(unittest.IsolatedAsyncioTestCase):
  """Drag-through capture: the deltas it produces, and the freedrive it must not leave on."""

  async def test_the_actions_are_the_deltas_between_consecutive_samples(self):
    arm = await sim_arm()
    track = TRACK

    ep = await dragged(arm, track).capture(duration_s=60.0)

    self.assertEqual(len(ep), len(track) - 1)
    self.assertEqual(ep.task, "drag to the reader")
    # These poses were injected, so this is not a demonstration. See
    # PoseProvenanceTests for what that label is worth.
    self.assertEqual(ep.metadata["source"], SOURCE_SYNTHETIC)
    self.assertEqual(ep.metadata["pose_source"], POSE_SOURCE_INJECTED)
    self.assertEqual(ep.metadata["hz"], 1000.0)
    self.assertEqual(ep.metadata["n_samples"], len(track))

    for i, step in enumerate(ep.steps):
      self.assertIs(step.action.mode, ActionMode.DELTA)
      # A demonstration does not get to claim the task is done: a replay ends
      # when the recording runs out, not when a captured flag says so.
      self.assertFalse(step.action.terminate)
      self.assertIsNone(step.action.gripper)
      want = tuple(b - a for a, b in zip(track[i], track[i + 1]))
      for got, expected in zip(step.action.pose, want):
        self.assertAlmostEqual(got, expected, places=9)
      # The delta has to be paired with the sample it was measured from, not the
      # one it lands on, or every transition in the episode is off by one.
      self.assertAlmostEqual(step.observation.x, track[i][0])
      self.assertAlmostEqual(step.observation.y, track[i][1])
      self.assertAlmostEqual(step.observation.z, track[i][2])
      self.assertAlmostEqual(step.observation.rz, track[i][3])

    self.assertIsNotNone(ep.final_observation)
    self.assertAlmostEqual(ep.final_observation.x, track[-1][0])
    self.assertAlmostEqual(ep.final_observation.y, track[-1][1])
    self.assertAlmostEqual(ep.final_observation.rz, track[-1][3])
    self.assertEqual(len(ep.transitions()), len(track) - 1)

    stamps = [o.t for o in ep.observations()]
    self.assertEqual(stamps[0], 0.0)
    self.assertTrue(all(b > a for a, b in zip(stamps, stamps[1:])), stamps)

  async def test_summing_the_deltas_replays_the_demonstrated_path(self):
    """The whole point of delta actions: they have to add back up to the demo."""
    arm = await sim_arm()

    ep = await dragged(arm, TRACK).capture(duration_s=60.0)

    here = list(TRACK[0])
    for step in ep.steps:
      here = [p + d for p, d in zip(here, step.action.pose)]
    for got, expected in zip(here, TRACK[-1]):
      self.assertAlmostEqual(got, expected, places=9)

  async def test_wrist_delta_takes_the_short_path_across_pi(self):
    arm = await sim_arm()
    track = ((0.0, 0.0, 100.0, 3.1), (0.0, 0.0, 100.0, -3.1))

    ep = await dragged(arm, track).capture(duration_s=60.0)

    self.assertEqual(len(ep), 1)
    self.assertGreater(ep.steps[0].action.pose[3], 0.0)
    self.assertAlmostEqual(ep.steps[0].action.pose[3], 2.0 * math.pi - 6.2, places=12)

  async def test_capture_enters_freedrive_once_and_leaves_it(self):
    backend = SimulationArmBackend()
    arm = await sim_arm(backend)

    await dragged(arm, TRACK).capture(duration_s=60.0)

    actions = freedrive_entries(backend)
    self.assertEqual(actions, ["freedrive", "end_freedrive"])
    self.assertEqual(backend.trace[-1]["action"], "end_freedrive")
    self.assertFalse(backend._in_freedrive)

  async def test_capture_leaves_freedrive_even_when_the_drag_raises(self):
    """An arm left compliant after a failed capture is a hazard, not a bug report."""
    backend = SimulationArmBackend()
    arm = await sim_arm(backend)

    def exploding_source(i: int, t: float):
      if i == 1:
        raise DragInterrupted("the operator pulled the arm into a fault")
      return (100.0, 0.0, 50.0, 0.0)

    session = KinestheticSession(
      arm, hz=1000.0, clock=FakeClock(step=0.001), pose_source=exploding_source
    )
    with self.assertRaises(DragInterrupted):
      await session.capture(duration_s=60.0)

    actions = freedrive_entries(backend)
    self.assertEqual(actions, ["freedrive", "end_freedrive"])
    self.assertEqual(backend.trace[-1]["action"], "end_freedrive")
    self.assertFalse(backend._in_freedrive)

  async def test_capture_frees_every_axis_the_arm_reports(self):
    """A partial list is refused by the xArm backend, so a subset is a real bug."""
    for n_joints in (5, 6):
      with self.subTest(n_joints=n_joints):
        backend = SimulationArmBackend(home_joints={i: 0.0 for i in range(1, n_joints + 1)})
        arm = await sim_arm(backend)

        await dragged(arm, TRACK).capture(duration_s=60.0)

        entry = [e for e in backend.trace if e["action"] == "freedrive"][-1]
        self.assertEqual(entry["axes"], list(range(1, n_joints + 1)))
        self.assertEqual(entry["axes"], sorted(await arm.get_joint_position()))

  async def test_a_strict_backend_accepts_the_axis_list_this_session_asks_for(self):
    """The refusal that makes a partial list a bug lives outside simulation.

    Simulation takes any list, so it cannot show that what the session asks for
    is what an all-or-nothing backend would honor. ``DegreeReportingBackend``
    raises on anything short of the whole arm, so this capture completing at all
    is the assertion, and the trace entry says which list got through.
    """
    backend = DegreeReportingBackend()
    arm = await homed_arm(backend)

    ep = await dragged(arm, TRACK).capture(duration_s=60.0)

    self.assertEqual(len(ep), len(TRACK) - 1)
    entry = [e for e in arm.backend.trace if e["action"] == "freedrive"][-1]
    self.assertEqual(entry["axes"], [1, 2, 3, 4, 5, 6])
    self.assertTrue(entry["whole_arm"])
    modes = [c["mode"] for c in backend.calls_to("set_mode")]
    self.assertIn(MODE_COMPLIANT, modes)
    self.assertEqual(modes[-1], MODE_MOTION)  # handed back locked, not compliant

  async def test_a_partial_axis_list_would_be_refused_by_that_backend(self):
    """The strictness the test above leans on, asserted rather than assumed.

    Without this, a double that quietly accepted any list would make the test
    above pass while proving nothing, which is the same failure as testing the
    session against simulation alone.
    """
    backend = DegreeReportingBackend()
    arm = await homed_arm(backend)

    with self.assertRaises(ControllerRefused):
      await arm.freedrive_mode([1, 2, 3])
    self.assertNotIn(MODE_COMPLIANT, [c["mode"] for c in backend.calls_to("set_mode")])

  async def test_a_half_entered_freedrive_is_still_handed_back(self):
    """Entering freedrive is two controller calls, and the second one can fail.

    Once the arm has gone compliant a failure after that point is exactly when
    the release matters most, and it is the case the simulation backend cannot
    produce: it enters freedrive in one step that never fails.
    """
    backend = DegreeReportingBackend()
    arm = await homed_arm(backend)
    backend.fail("enable_freedrive")

    session = KinestheticSession(
      arm, hz=1000.0, clock=FakeClock(step=0.001), pose_source=replay(TRACK)
    )
    with self.assertRaises(ControllerRefused):
      await session.capture(duration_s=60.0)

    modes = [c["mode"] for c in backend.calls_to("set_mode")]
    self.assertIn(MODE_COMPLIANT, modes)  # it really did get half way in
    self.assertEqual(modes[-1], MODE_MOTION)

  async def test_the_gripper_is_recorded_only_on_the_sample_where_it_moved(self):
    """A width on every delta would re-command the jaws at the sample rate."""
    backend = SimulationArmBackend()
    arm = await sim_arm(backend)
    track = [
      (0.0, 0.0, 100.0, 0.0),
      (10.0, 0.0, 100.0, 0.0),
      (20.0, 0.0, 100.0, 0.0),
      (30.0, 0.0, 100.0, 0.0),
    ]

    def source(i: int, t: float):
      if i == 2:
        # Simulation has no hand to squeeze the jaws with, so the width the next
        # sample reads back is moved directly.
        backend._gripper_width = 60.0
      return track[i] if i < len(track) else None

    session = KinestheticSession(
      arm, hz=1000.0, clock=FakeClock(step=0.001), pose_source=source
    )
    ep = await session.capture(duration_s=60.0)

    self.assertEqual([s.action.gripper for s in ep.steps], [None, None, 60.0])

  async def test_stop_ends_the_capture_after_the_current_sample(self):
    arm = await sim_arm()
    session = KinestheticSession(arm, hz=1000.0, clock=FakeClock(step=0.001))

    def source(i: int, t: float):
      if i == 2:
        session.stop()
      return (10.0 * i, 0.0, 100.0, 0.0)

    session.pose_source = source
    ep = await session.capture(duration_s=60.0)

    self.assertEqual(ep.metadata["n_samples"], 3)  # the sample stop() was called on is kept
    self.assertEqual(len(ep), 2)
    for step in ep.steps:
      self.assertAlmostEqual(step.action.pose[0], 10.0)

  async def test_a_capture_that_ends_before_its_first_sample_is_empty(self):
    arm = await sim_arm()

    ep = await dragged(arm, []).capture(duration_s=60.0)

    self.assertEqual(len(ep), 0)
    self.assertIsNone(ep.final_observation)
    self.assertEqual(ep.transitions(), [])
    self.assertEqual(ep.metadata["n_samples"], 0)

  async def test_a_session_that_could_never_sample_is_refused(self):
    arm = await sim_arm()
    with self.assertRaises(ValueError):
      KinestheticSession(arm, hz=0.0)
    with self.assertRaises(ValueError):
      KinestheticSession(arm, hz=-1.0)
    with self.assertRaises(ValueError):
      await KinestheticSession(arm, hz=10.0).capture(duration_s=0.0)


class PoseProvenanceTests(unittest.IsolatedAsyncioTestCase):
  """Regression: an injected trajectory must not claim a person dragged the arm.

  The confidently wrong claim this prevents, measured on this repository before
  the fix: ``KinestheticSession.capture`` stamped ``{"source": "kinesthetic"}``
  on every episode it produced, including one whose every pose came from a
  ``pose_source`` callable on a machine with no arm attached. Nothing written to
  disk distinguished the two, so a dataset assembled that way asserts a human
  demonstration that never happened, and every model and every citation
  downstream inherits the assertion.

  Provenance is read off how the session was configured and not off the poses it
  ended up with, so these tests pin the configuration, the label, and the fact
  that the recorded trajectory is not the arm's own.
  """

  async def test_an_injected_trajectory_is_labelled_synthetic_and_never_kinesthetic(self):
    arm = await sim_arm()
    session = dragged(arm, TRACK)
    self.assertEqual(session.pose_provenance, POSE_SOURCE_INJECTED)

    ep = await session.capture(duration_s=60.0)

    self.assertEqual(ep.metadata["pose_source"], POSE_SOURCE_INJECTED)
    self.assertNotEqual(ep.metadata["source"], SOURCE_KINESTHETIC)
    self.assertEqual(ep.metadata["source"], SOURCE_SYNTHETIC)

    # The load-bearing part: the recorded trajectory is not where the arm is, so
    # nothing physical produced it and the old label was a claim about the world.
    self.assertAlmostEqual(ep.final_observation.x, TRACK[-1][0])
    on_the_arm = await observe(arm)
    self.assertNotAlmostEqual(on_the_arm.x, ep.final_observation.x, places=3)

  async def test_a_capture_off_the_arms_own_poses_is_the_only_kinesthetic_one(self):
    arm = await sim_arm()
    session = KinestheticSession(arm, task="drag", hz=1000.0, clock=FakeClock(step=0.001))
    self.assertIsNone(session.pose_source)
    self.assertEqual(session.pose_provenance, POSE_SOURCE_ENCODER)

    ep = await session.capture(duration_s=60.0, max_samples=3)

    self.assertEqual(ep.metadata["source"], SOURCE_KINESTHETIC)
    self.assertEqual(ep.metadata["pose_source"], POSE_SOURCE_ENCODER)
    self.assertEqual(ep.metadata["n_samples"], 3)

  async def test_provenance_survives_the_round_trip_to_disk(self):
    """A field that does not serialise is a field the dataset does not have."""
    arm = await sim_arm()
    ep = await dragged(arm, TRACK).capture(duration_s=60.0)

    restored = Episode.from_dict(ep.to_dict())

    self.assertEqual(restored.metadata["pose_source"], POSE_SOURCE_INJECTED)
    self.assertEqual(restored.metadata["source"], SOURCE_SYNTHETIC)

  async def test_a_scripted_recording_declares_encoder_poses_too(self):
    """Absent is not the same as encoder, so both kinds say where they came from."""
    arm = await sim_arm()

    async def routine():
      await arm.move_to(coords(150.0, -20.0, 30.0))

    ep = await record_scripted(arm, routine, clock=FakeClock(step=0.5))

    self.assertEqual(ep.metadata["source"], "scripted")
    self.assertEqual(ep.metadata["pose_source"], POSE_SOURCE_ENCODER)


class RecordScriptedTests(unittest.IsolatedAsyncioTestCase):
  """The endpoints of a routine this recorder cannot see inside of."""

  async def test_a_scripted_routine_becomes_one_absolute_terminating_action(self):
    arm = await sim_arm()
    await arm.move_to(coords(0.0, 0.0, 200.0))

    async def routine():
      await arm.move_to(coords(60.0, 0.0, 120.0))
      await arm.move_to(coords(150.0, -20.0, 30.0, 0.4))

    ep = await record_scripted(arm, routine, task="stage the plate", clock=FakeClock(step=0.5))

    self.assertEqual(len(ep), 1)
    step = ep.steps[0]
    self.assertAlmostEqual(step.observation.x, 0.0)
    self.assertAlmostEqual(step.observation.z, 200.0)
    self.assertEqual(step.observation.t, 0.0)
    self.assertIs(step.action.mode, ActionMode.ABSOLUTE)
    self.assertTrue(step.action.terminate)
    for got, expected in zip(step.action.pose, (150.0, -20.0, 30.0, 0.4)):
      self.assertAlmostEqual(got, expected)

    self.assertEqual(ep.task, "stage the plate")
    self.assertEqual(ep.metadata["source"], "scripted")
    self.assertIsNotNone(ep.final_observation)
    self.assertAlmostEqual(ep.final_observation.x, 150.0)
    self.assertAlmostEqual(ep.final_observation.rz, 0.4)
    self.assertGreater(ep.final_observation.t, step.observation.t)
    self.assertEqual(len(ep.transitions()), 1)

  async def test_the_path_the_routine_took_is_not_in_the_record(self):
    """It captures the endpoints, and it must not imply it captured more."""
    arm = await sim_arm()
    await arm.move_to(coords(0.0, 0.0, 200.0))

    async def routine():
      await arm.move_to(coords(60.0, 0.0, 120.0))
      await arm.move_to(coords(150.0, -20.0, 30.0))

    ep = await record_scripted(arm, routine, clock=FakeClock(step=0.5))

    self.assertEqual(len(ep.observations()), 2)
    self.assertNotIn(60.0, [o.x for o in ep.observations()])

  async def test_a_routine_that_moves_nowhere_records_where_it_stayed(self):
    """A no-op routine must not read as a move, and must still be scorable."""
    arm = await sim_arm()
    await arm.move_to(coords(80.0, 10.0, 60.0, 0.2))

    async def routine():
      return None

    ep = await record_scripted(arm, routine, clock=FakeClock(step=0.5))

    for got, expected in zip(ep.steps[0].action.pose, (80.0, 10.0, 60.0, 0.2)):
      self.assertAlmostEqual(got, expected)
    obs, action, after = ep.transitions()[0]
    for got, expected in zip(action.as_delta(obs.pose_vector()), (0.0, 0.0, 0.0, 0.0)):
      self.assertAlmostEqual(got, expected)
    self.assertAlmostEqual(after.x, 80.0)


if __name__ == "__main__":
  unittest.main()
