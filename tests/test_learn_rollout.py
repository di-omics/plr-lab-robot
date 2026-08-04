"""Tests for plr_lr.learn.rollout, the layer between an unreviewed policy and an arm.

Organised around the guarantees this module makes rather than around its
functions, because the guarantees are what a refactor would quietly drop. They
are: that an oversized request is scaled rather than truncated, so the direction
a policy asked for survives; that a step sized exactly at the limit is not
reported as clamped, so the clamp count stays a signal instead of noise; that the
mode a policy chose comes back unchanged, so a delta is never re-read as an
absolute pose; that a joint-space action is refused rather than passed through a
Cartesian check that cannot see it; and that a non-simulated arm is refused
unless the call site said otherwise.

Nothing here needs an arm, and nothing here imports a driver. The hardware gate
is exercised with ``DegreeReportingBackend`` from ``tests._doubles``, which is a
``SCARABackend`` and deliberately not a ``SimulationArmBackend``, because the
gate is an isinstance check and no other kind of double can stand on the far
side of it. It is also the only backend in this file that disagrees with the
learning layer about units, which is what makes the radian round trip a
measurement rather than a restatement.
"""

from __future__ import annotations

import json
import math
import unittest
from typing import Optional, Sequence

import numpy as np
from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.precise_flex.coords import ElbowOrientation, PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from plr_lr import SimulationArmBackend, Workspace
from plr_lr.learn import (
  Action,
  ActionMode,
  Episode,
  EpisodeRecorder,
  HardwarePermissionRequired,
  MistrackingBackend,
  NoOpPolicy,
  Observation,
  RandomPolicy,
  SafetyEnvelope,
  UnsafeAction,
  apply_action,
  is_simulation,
  observe,
  rollout,
)
from plr_lr.learn.predict import FrameRef, RolloutPrediction, RolloutRequest
from plr_lr.learn.rollout import (
  THROTTLED_CLAMP_FRACTION,
  StepLimitExceeded,
  clamped_fraction,
  is_throttled,
  per_command_step_mm_of,
)
from plr_lr.learn.screen import (
  AbstainReason,
  CalibrationPoint,
  Screen,
  ScreenedPolicy,
)
from plr_lr.learn.types import ACTION_FIELDS
from tests._doubles import DegreeReportingBackend, homed_arm

# Well inside the default workspace on every axis, so a test that fails is
# failing on the thing it names and not on a start pose that happens to sit on a
# workspace face where a clip would fire for free.
START_POSE = (0.0, 0.0, 100.0, 0.0)


def coords_at(pose: Sequence[float]) -> PreciseFlexCartesianCoords:
  """Build PyLabRobot coordinates from a learning-native pose in radians."""
  return PreciseFlexCartesianCoords(
    location=Coordinate(float(pose[0]), float(pose[1]), float(pose[2])),
    rotation=Rotation(0.0, 0.0, math.degrees(float(pose[3]))),
    orientation=ElbowOrientation.RIGHT,
  )


async def sim_arm(
  workspace: Optional[Workspace] = None,
  start: Sequence[float] = START_POSE,
) -> ExperimentalSCARA:
  """A homed simulated arm parked at a known pose."""
  arm = ExperimentalSCARA(backend=SimulationArmBackend(workspace=workspace or Workspace()))
  await arm.setup()
  await arm.home()
  await arm.move_to(coords_at(start))
  return arm


class ScriptedPolicy:
  """Emits a fixed list of actions, then holds. No generator, so no seed needed."""

  def __init__(self, actions: Sequence[Action]) -> None:
    self.actions = list(actions)
    self.i = 0

  def reset(self) -> None:
    self.i = 0

  def act(self, observation: Observation) -> Action:
    if self.i >= len(self.actions):
      return Action.hold()
    action = self.actions[self.i]
    self.i += 1
    return action


class NonActionPolicy:
  """Returns the wrong type, to exercise the policy boundary refusal."""

  def act(self, observation: Observation):
    return {"dx": 1.0}


class TestEnvelopeConfiguration(unittest.TestCase):
  """A malformed bound is not a safety envelope."""

  def test_step_limits_must_be_positive_and_finite(self):
    for name in ("max_step_mm", "max_step_rad"):
      for value in (0.0, -1.0, float("inf"), float("-inf"), float("nan")):
        with self.subTest(name=name, value=value):
          with self.assertRaises(ValueError):
            SafetyEnvelope(**{name: value})

  def test_step_budget_must_be_a_positive_integer(self):
    for value in (0, -1, 1.5, True):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          SafetyEnvelope(max_steps=value)

  def test_workspace_bounds_must_be_finite_and_ordered(self):
    for bounds in ((1.0, -1.0), (0.0, float("inf")), (float("nan"), 1.0)):
      with self.subTest(bounds=bounds):
        with self.assertRaises(ValueError):
          SafetyEnvelope(workspace=Workspace(x=bounds))


class TestTranslationAndRotationLimits(unittest.TestCase):
  """An oversized request is scaled along its own direction, not truncated."""

  def setUp(self):
    self.env = SafetyEnvelope(max_step_mm=25.0, max_step_rad=0.2)

  def test_an_oversized_translation_keeps_its_direction(self):
    requested = Action.delta(30.0, 40.0, 0.0)
    safe, notes = self.env.clamp(requested, START_POSE)

    self.assertAlmostEqual(float(np.linalg.norm(safe.translation)), 25.0, places=12)
    # Same unit vector: scaling, not per-axis truncation, which would turn a
    # diagonal request into a differently aimed one.
    np.testing.assert_allclose(
      safe.translation / 25.0, requested.translation / 50.0, atol=1e-12
    )
    self.assertEqual(len(notes), 1)
    self.assertIn("50.00 mm scaled to 25.00 mm", notes[0])

  def test_a_translation_inside_the_limit_is_left_alone(self):
    requested = Action.delta(3.0, 4.0, 0.0)
    safe, notes = self.env.clamp(requested, START_POSE)
    self.assertEqual(notes, [])
    np.testing.assert_allclose(safe.pose, requested.pose, atol=1e-12)

  def test_rotation_is_clamped_and_keeps_its_sign(self):
    for drz, expected in ((1.0, 0.2), (-1.0, -0.2)):
      safe, notes = self.env.clamp(Action.delta(drz=drz), START_POSE)
      self.assertAlmostEqual(safe.pose[3], expected, places=12)
      self.assertEqual(len(notes), 1)
      self.assertIn("rad clamped to", notes[0])

  def test_rotation_inside_the_limit_is_left_alone(self):
    safe, notes = self.env.clamp(Action.delta(drz=0.1), START_POSE)
    self.assertEqual(notes, [])
    self.assertAlmostEqual(safe.pose[3], 0.1, places=12)

  def test_a_step_exactly_at_the_limit_is_not_reported_as_clamped(self):
    """The regression that would mark every well-behaved step as clamped.

    A step scaled to exactly the limit comes back out of ``norm`` fractionally
    above it. Without the boundary tolerance every step a well-behaved policy
    emits would be counted as a clamp, and the real clamps would be buried in
    the count. The first assertion is what makes the rest of this test mean
    something: the round trip really does overshoot.
    """
    direction = np.array([1.0, 1.0, 1.0])
    step = direction * (25.0 / float(np.linalg.norm(direction)))
    self.assertGreater(float(np.linalg.norm(step)), 25.0)

    safe, notes = self.env.clamp(Action.delta(*step), START_POSE)
    self.assertEqual(notes, [])
    np.testing.assert_allclose(safe.pose[:3], step, atol=1e-9)

    # Same boundary on the rotation axis.
    _, rot_notes = self.env.clamp(Action.delta(drz=0.2), START_POSE)
    self.assertEqual(rot_notes, [])

  def test_clamping_is_idempotent(self):
    """Re-clamping a clamped action must be silent, or the count keeps growing."""
    safe, notes = self.env.clamp(Action.delta(0.0, 0.0, 90.0), START_POSE)
    self.assertEqual(len(notes), 1)
    again, again_notes = self.env.clamp(safe, START_POSE)
    self.assertEqual(again_notes, [])
    np.testing.assert_allclose(again.pose, safe.pose, atol=1e-12)

  def test_non_finite_current_pose_or_cartesian_action_is_refused(self):
    with self.assertRaises(UnsafeAction):
      self.env.clamp(Action.hold(), (float("nan"), 0.0, 100.0, 0.0))
    for action in (
      Action.delta(float("nan"), 0.0, 0.0),
      Action.delta(float("inf"), 0.0, 0.0),
      Action.absolute(0.0, 0.0, 100.0, float("nan")),
    ):
      with self.subTest(action=action):
        with self.assertRaises(UnsafeAction):
          self.env.clamp(action, START_POSE)

  def test_non_finite_joint_or_gripper_targets_are_refused(self):
    joints = SafetyEnvelope(allow_joint_actions=True)
    with self.assertRaises(UnsafeAction):
      joints.clamp(Action.joint_targets({1: float("nan")}), START_POSE)
    for width in (float("nan"), float("inf"), -1.0):
      with self.subTest(width=width):
        with self.assertRaises(UnsafeAction):
          self.env.clamp(Action.delta(gripper=width), START_POSE)

  def test_non_numeric_gripper_and_non_action_inputs_are_refused(self):
    malformed = Action(mode=ActionMode.DELTA, gripper="wide")  # type: ignore[arg-type]
    with self.assertRaises(UnsafeAction):
      self.env.clamp(malformed, START_POSE)
    with self.assertRaises(UnsafeAction):
      self.env.clamp({"dx": 1.0}, START_POSE)  # type: ignore[arg-type]


class TestWorkspaceClipping(unittest.TestCase):
  """Targets outside the box are clipped per axis, and the clip is written down."""

  def setUp(self):
    # Step limits far above anything commanded here, so a note can only come
    # from the workspace and never from the step size.
    self.env = SafetyEnvelope(
      max_step_mm=1000.0,
      max_step_rad=10.0,
      workspace=Workspace(x=(-100.0, 100.0), y=(-100.0, 100.0), z=(0.0, 200.0)),
    )

  def test_only_the_offending_axis_is_clipped(self):
    safe, notes = self.env.clamp(Action.delta(50.0, 20.0, 0.0), (90.0, 0.0, 100.0, 0.0))
    self.assertEqual(len(notes), 1)
    self.assertIn("clipped to workspace", notes[0])
    self.assertIn("(140.0, 20.0, 100.0)", notes[0])
    self.assertIn("(100.0, 20.0, 100.0)", notes[0])
    # x rides the face at 100, y and z keep exactly what was asked for.
    np.testing.assert_allclose(safe.pose, (10.0, 20.0, 0.0, 0.0), atol=1e-12)

  def test_a_floor_is_clipped_as_well_as_a_ceiling(self):
    safe, notes = self.env.clamp(Action.delta(0.0, 0.0, -150.0), (0.0, 0.0, 100.0, 0.0))
    self.assertEqual(len(notes), 1)
    np.testing.assert_allclose(safe.pose, (0.0, 0.0, -100.0, 0.0), atol=1e-12)

  def test_a_target_inside_the_box_records_nothing(self):
    safe, notes = self.env.clamp(Action.delta(5.0, 5.0, 5.0), (0.0, 0.0, 100.0, 0.0))
    self.assertEqual(notes, [])
    np.testing.assert_allclose(safe.pose, (5.0, 5.0, 5.0, 0.0), atol=1e-12)

  def test_no_workspace_means_no_clipping(self):
    env = SafetyEnvelope(max_step_mm=1000.0, workspace=None)
    safe, notes = env.clamp(Action.delta(900.0, 0.0, 0.0), START_POSE)
    self.assertEqual(notes, [])
    np.testing.assert_allclose(safe.pose[:3], (900.0, 0.0, 0.0), atol=1e-12)


class TestWorkspaceRequiresTrustedStart(unittest.TestCase):
  """The envelope is a command bound, not an implicit recovery procedure."""

  def test_an_arm_outside_the_workspace_is_refused_even_for_a_hold(self):
    env = SafetyEnvelope(
      max_step_mm=25.0,
      workspace=Workspace(x=(-500.0, 500.0), y=(-500.0, 500.0), z=(0.0, 200.0)),
    )
    # 100 mm above the ceiling: reachable whenever the envelope box is tighter
    # than the arm's own, which is the reason to declare one.
    outside = (0.0, 0.0, 300.0, 0.0)
    with self.assertRaises(UnsafeAction) as ctx:
      env.clamp(Action.hold(), outside)
    self.assertIn("outside the declared workspace", str(ctx.exception))
    self.assertIn("manual procedure", str(ctx.exception))


class TestModeSurvivesTheClamp(unittest.TestCase):
  """A clamped absolute pose is still absolute, and a delta is still a delta.

  Returning the wrong mode is silent: the numbers are plausible either way and
  the arm goes somewhere else entirely.
  """

  def setUp(self):
    self.env = SafetyEnvelope(max_step_mm=25.0, max_step_rad=0.2)

  def test_absolute_comes_back_absolute(self):
    safe, notes = self.env.clamp(Action.absolute(100.0, 0.0, 100.0, 0.0), START_POSE)
    self.assertIs(safe.mode, ActionMode.ABSOLUTE)
    self.assertEqual(len(notes), 1)
    np.testing.assert_allclose(safe.pose, (25.0, 0.0, 100.0, 0.0), atol=1e-12)

  def test_delta_comes_back_delta(self):
    safe, notes = self.env.clamp(Action.delta(100.0, 0.0, 0.0), START_POSE)
    self.assertIs(safe.mode, ActionMode.DELTA)
    self.assertEqual(len(notes), 1)
    np.testing.assert_allclose(safe.pose, (25.0, 0.0, 0.0, 0.0), atol=1e-12)

  def test_the_two_modes_clamp_to_the_same_place(self):
    absolute, _ = self.env.clamp(Action.absolute(100.0, 60.0, 130.0, 1.0), START_POSE)
    delta, _ = self.env.clamp(Action.delta(100.0, 60.0, 30.0, 1.0), START_POSE)
    np.testing.assert_allclose(
      absolute.as_delta(START_POSE), delta.as_delta(START_POSE), atol=1e-12
    )

  def test_the_rest_of_the_action_is_carried_through(self):
    requested = Action.delta(100.0, 0.0, 0.0, gripper=70.0, terminate=True)
    safe, _ = self.env.clamp(requested, START_POSE)
    self.assertEqual(safe.gripper, 70.0)
    self.assertTrue(safe.terminate)


class TestJointActionsAreRefused(unittest.TestCase):
  """A joint target cannot be checked against a Cartesian box, so it is not."""

  def test_a_joint_action_is_refused_by_default(self):
    env = SafetyEnvelope()
    with self.assertRaises(UnsafeAction) as ctx:
      env.clamp(Action.joint_targets({1: 10.0}), START_POSE)
    self.assertIn("allow_joint_actions=True", str(ctx.exception))

  def test_opting_in_passes_it_through_and_says_it_was_unchecked(self):
    env = SafetyEnvelope(allow_joint_actions=True)
    requested = Action.joint_targets({1: 10.0, 2: -20.0})
    safe, notes = env.clamp(requested, START_POSE)
    self.assertIs(safe.mode, ActionMode.JOINTS)
    self.assertEqual(safe.joints, {1: 10.0, 2: -20.0})
    self.assertEqual(len(notes), 1)
    self.assertIn("unchecked", notes[0])


class TestHardwareGate(unittest.IsolatedAsyncioTestCase):
  """Driving a real arm is opt-in at the call site, and refused otherwise."""

  async def test_is_simulation_only_accepts_the_simulation_hierarchy(self):
    self.assertTrue(is_simulation(await sim_arm()))
    self.assertFalse(is_simulation(ExperimentalSCARA(backend=DegreeReportingBackend())))
    # The planted-defect arm is still a simulated arm, so the gate must not
    # block the very backend the evaluation harness scores against.
    self.assertTrue(is_simulation(ExperimentalSCARA(backend=MistrackingBackend(gain=0.85))))

  async def test_a_non_simulated_arm_is_refused(self):
    backend = DegreeReportingBackend(start=START_POSE)
    arm = ExperimentalSCARA(backend=backend)
    await arm.setup()

    with self.assertRaises(HardwarePermissionRequired) as ctx:
      await rollout(arm, NoOpPolicy(), envelope=SafetyEnvelope(max_steps=3))
    self.assertIn("allow_hardware=True", str(ctx.exception))
    # The refusal names the backend it refused, so the message identifies which
    # arm was about to be driven rather than only that some arm was.
    self.assertIn("DegreeReportingBackend", str(ctx.exception))
    # Refused before anything was read or commanded, not after.
    self.assertEqual(backend.calls, ["setup"])

  async def test_a_truthy_non_boolean_does_not_unlock_hardware(self):
    backend = DegreeReportingBackend(start=START_POSE)
    arm = ExperimentalSCARA(backend=backend)
    await arm.setup()

    with self.assertRaises(HardwarePermissionRequired):
      await rollout(
        arm,
        NoOpPolicy(),
        envelope=SafetyEnvelope(max_steps=1),
        allow_hardware="yes",  # type: ignore[arg-type]
      )
    self.assertEqual(backend.calls, ["setup"])

  async def test_the_gate_opens_when_the_call_site_opts_in(self):
    backend = DegreeReportingBackend(start=START_POSE)
    arm = ExperimentalSCARA(backend=backend)
    await arm.setup()

    episode = await rollout(
      arm,
      NoOpPolicy(),
      envelope=SafetyEnvelope(max_steps=2),
      task="reach reader",
      allow_hardware=True,
    )
    self.assertEqual(len(episode), 2)
    self.assertEqual(episode.task, "reach reader")
    self.assertIs(episode.metadata["simulated"], False)
    self.assertIn("get_cartesian_position", backend.calls)

  async def test_simulation_needs_no_permission_and_says_it_was_simulated(self):
    arm = await sim_arm()
    episode = await rollout(arm, NoOpPolicy(), envelope=SafetyEnvelope(max_steps=2))
    self.assertIs(episode.metadata["simulated"], True)


class TestApplyAction(unittest.IsolatedAsyncioTestCase):
  """One action, one arm, and what actually reaches the backend."""

  async def test_a_narrower_width_closes_the_gripper(self):
    arm = await sim_arm()
    self.assertEqual(arm.backend._gripper_width, 90.0)

    await apply_action(arm, Action.delta(gripper=70.0), START_POSE)
    entry = [e for e in arm.backend.trace if e["action"] == "close_gripper"][-1]
    self.assertAlmostEqual(entry["width"], 70.0)
    self.assertTrue(await arm.is_gripper_closed())
    # A gripper-only action commands the gripper and nothing else.
    self.assertEqual([e for e in arm.backend.trace if e["action"] == "open_gripper"], [])

  async def test_a_wider_width_opens_the_gripper(self):
    arm = await sim_arm()
    await apply_action(arm, Action.delta(gripper=70.0), START_POSE)
    await apply_action(arm, Action.delta(gripper=95.0), START_POSE)

    entry = [e for e in arm.backend.trace if e["action"] == "open_gripper"][-1]
    self.assertAlmostEqual(entry["width"], 95.0)
    self.assertFalse(await arm.is_gripper_closed())

  async def test_a_zero_motion_action_issues_no_move(self):
    arm = await sim_arm()
    before = list(arm.backend.trace)

    await apply_action(arm, Action.hold(), START_POSE)
    self.assertEqual(arm.backend.trace, before)
    cart = await arm.get_cartesian_position()
    self.assertAlmostEqual(cart.location.x, START_POSE[0])
    self.assertAlmostEqual(cart.location.z, START_POSE[2])

  async def test_a_joint_action_routes_to_move_to_with_a_joint_dict(self):
    arm = await sim_arm()
    moves_before = len([e for e in arm.backend.trace if e["action"] == "move_to"])

    await apply_action(arm, Action.joint_targets({1: 15.0, 2: -5.0}), START_POSE)
    entry = [e for e in arm.backend.trace if e["action"] == "move_to_joints"][-1]
    self.assertAlmostEqual(entry["joints"][1], 15.0)
    self.assertAlmostEqual(entry["joints"][2], -5.0)
    # A joint action has no Cartesian delta, so no Cartesian move may be issued.
    self.assertEqual(
      len([e for e in arm.backend.trace if e["action"] == "move_to"]), moves_before
    )

  async def test_an_empty_joint_action_issues_nothing(self):
    arm = await sim_arm()
    before = list(arm.backend.trace)
    await apply_action(arm, Action.joint_targets({}), START_POSE)
    self.assertEqual(arm.backend.trace, before)

  async def test_a_cartesian_delta_moves_the_tool_and_keeps_roll_and_pitch(self):
    arm = await sim_arm()
    # Park with a tool orientation that is not the identity. At roll and pitch
    # zero, preserving them and zeroing them look the same, so this test would
    # pass for the wrong reason from the default parked pose.
    await arm.move_to(
      PreciseFlexCartesianCoords(
        location=Coordinate(START_POSE[0], START_POSE[1], START_POSE[2]),
        rotation=Rotation(180.0, -3.0, START_POSE[3]),
        orientation=ElbowOrientation.LEFT,
      )
    )
    before = await arm.get_cartesian_position()

    await apply_action(arm, Action.delta(10.0, -5.0, 20.0, 0.3), START_POSE)
    after = await arm.get_cartesian_position()
    self.assertAlmostEqual(after.location.x, START_POSE[0] + 10.0)
    self.assertAlmostEqual(after.location.y, START_POSE[1] - 5.0)
    self.assertAlmostEqual(after.location.z, START_POSE[2] + 20.0)
    self.assertAlmostEqual(after.rotation.z, math.degrees(START_POSE[3] + 0.3))
    self.assertAlmostEqual((await observe(arm)).rz, START_POSE[3] + 0.3)
    # Only rz is under this module's control; the rest comes from the arm.
    self.assertAlmostEqual(after.rotation.x, 180.0)
    self.assertAlmostEqual(after.rotation.y, -3.0)
    self.assertAlmostEqual(after.rotation.x, before.rotation.x)
    self.assertAlmostEqual(after.rotation.y, before.rotation.y)
    self.assertIs(after.orientation, before.orientation)

  async def test_a_radian_action_reaches_a_degree_reporting_backend_in_radians(self):
    """Learning radians round-trip through PyLabRobot degrees and back to radians.

    ``apply_action`` converts radians out to degrees and the backend converts
    degrees back in, so the number recorded on the far side has been through
    both halves of the boundary. Asserted against the simulation backend this
    would prove nothing: it stores whatever ``Rotation`` it was handed, so a
    missing conversion and a correct one leave the same value behind.
    """
    backend = DegreeReportingBackend(start=(200.0, 0.0, 300.0, 0.1), roll_rad=math.pi)
    arm = await homed_arm(backend)

    await apply_action(arm, Action.delta(drz=0.2), (200.0, 0.0, 300.0, 0.1))

    command = backend.calls_to("move_to")[-1]
    self.assertAlmostEqual(command["yaw_rad"], 0.3)
    self.assertAlmostEqual((await observe(arm)).rz, 0.3)

  async def test_apply_action_refuses_non_finite_values_before_any_motion(self):
    arm = await sim_arm()
    trace_before = list(arm.backend.trace)
    with self.assertRaises(UnsafeAction):
      await apply_action(arm, Action.hold(), (float("nan"), 0.0, 100.0, 0.0))
    bad_actions = (
      Action.delta(float("nan"), 0.0, 0.0),
      Action.joint_targets({1: float("inf")}),
      Action.delta(1.0, 0.0, 0.0, gripper=float("nan")),
      Action(mode=ActionMode.DELTA, gripper="wide"),  # type: ignore[arg-type]
    )
    for action in bad_actions:
      with self.subTest(action=action):
        with self.assertRaises(UnsafeAction):
          await apply_action(arm, action, START_POSE)
    self.assertEqual(arm.backend.trace, trace_before)

  async def test_a_non_finite_current_gripper_width_is_refused_before_pose_motion(self):
    arm = await sim_arm()
    arm.backend._gripper_width = float("nan")
    trace_before = list(arm.backend.trace)

    with self.assertRaises(UnsafeAction):
      await apply_action(arm, Action.delta(dx=1.0, gripper=50.0), START_POSE)

    self.assertEqual(arm.backend.trace, trace_before)

  async def test_an_absolute_action_commands_the_pose_it_names(self):
    arm = await sim_arm()
    await apply_action(arm, Action.absolute(60.0, -20.0, 150.0, 0.0), START_POSE)
    after = await arm.get_cartesian_position()
    self.assertAlmostEqual(after.location.x, 60.0)
    self.assertAlmostEqual(after.location.y, -20.0)
    self.assertAlmostEqual(after.location.z, 150.0)


class TestRolloutLoop(unittest.IsolatedAsyncioTestCase):
  """What the loop records, when it stops, and what it admits to changing."""

  async def test_one_step_per_iteration_until_the_budget_runs_out(self):
    arm = await sim_arm()
    episode = await rollout(arm, NoOpPolicy(), envelope=SafetyEnvelope(max_steps=5))

    self.assertEqual(len(episode), 5)
    self.assertEqual(episode.metadata["steps"], 5)
    self.assertEqual(episode.metadata["stop_reason"], "budget_exhausted")
    self.assertIsNotNone(episode.final_observation)
    # The closing observation is what gives the last action an outcome.
    self.assertEqual(len(episode.transitions()), 5)

  async def test_max_steps_overrides_the_envelope_budget(self):
    arm = await sim_arm()
    episode = await rollout(arm, NoOpPolicy(), envelope=SafetyEnvelope(max_steps=50), max_steps=2)
    self.assertEqual(len(episode), 2)
    self.assertEqual(episode.metadata["stop_reason"], "budget_exhausted")

  async def test_max_steps_cannot_enlarge_the_envelope_budget(self):
    arm = await sim_arm()
    episode = await rollout(arm, NoOpPolicy(), envelope=SafetyEnvelope(max_steps=2), max_steps=50)
    self.assertEqual(len(episode), 2)

  async def test_an_invalid_call_site_budget_is_refused(self):
    arm = await sim_arm()
    for value in (0, -1, 1.5, True):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          await rollout(arm, NoOpPolicy(), envelope=SafetyEnvelope(), max_steps=value)

  async def test_hz_must_be_positive_and_finite_before_any_action(self):
    arm = await sim_arm()
    moves_before = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    for hz in (0.0, -1.0, float("inf"), float("nan"), True):
      with self.subTest(hz=hz):
        with self.assertRaises(ValueError):
          await rollout(
            arm,
            ScriptedPolicy([Action.delta(dx=1.0)]),
            envelope=SafetyEnvelope(max_steps=1),
            hz=hz,
          )
    moves_after = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    self.assertEqual(moves_after, moves_before)

  async def test_a_non_action_policy_result_is_refused_before_motion(self):
    arm = await sim_arm()
    moves_before = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    with self.assertRaises(UnsafeAction) as ctx:
      await rollout(arm, NonActionPolicy(), envelope=SafetyEnvelope(max_steps=1))
    self.assertIn("must return an Action", str(ctx.exception))
    moves_after = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    self.assertEqual(moves_after, moves_before)

  async def test_a_terminating_action_stops_the_rollout(self):
    arm = await sim_arm()
    policy = ScriptedPolicy(
      [Action.delta(5.0, 0.0, 0.0), Action.delta(5.0, 0.0, 0.0, terminate=True)]
    )
    episode = await rollout(arm, policy, envelope=SafetyEnvelope(max_steps=20))

    self.assertEqual(len(episode), 2)
    self.assertEqual(episode.metadata["stop_reason"], "policy_terminated")
    # The terminating action still ran: terminate is not an abort.
    self.assertAlmostEqual(episode.final_observation.x, START_POSE[0] + 10.0)

  async def test_a_clamp_is_counted_and_the_pre_clamp_request_is_kept(self):
    arm = await sim_arm()
    policy = ScriptedPolicy([Action.delta(100.0, 0.0, 0.0)])
    episode = await rollout(arm, policy, envelope=SafetyEnvelope(max_step_mm=10.0, max_steps=3))

    self.assertEqual(episode.metadata["clamped_steps"], 1)
    first = episode.steps[0]
    self.assertEqual(len(first.info["clamped"]), 1)
    self.assertIn("scaled to 10.00 mm", first.info["clamped"][0])
    # What was logged as the action is the safe one, and what the policy asked
    # for survives beside it. A run that only stayed in bounds because the
    # envelope dragged it back must not read as a clean success.
    np.testing.assert_allclose(first.action.pose, (10.0, 0.0, 0.0, 0.0), atol=1e-12)
    self.assertEqual(first.info["requested"]["pose"], [100.0, 0.0, 0.0, 0.0])
    self.assertEqual(first.info["requested"]["mode"], "delta")

    # Steps the envelope left alone carry the accounting and nothing else. A
    # recorded zero is a measurement; an absent key is indistinguishable from a
    # step that was never checked.
    for step in episode.steps[1:]:
      self.assertEqual(step.info, {"n_clamped": 0, "clamp_magnitude_mm": 0.0})
    self.assertAlmostEqual(episode.final_observation.x, START_POSE[0] + 10.0)

  async def test_an_unclamped_rollout_counts_no_clamps(self):
    arm = await sim_arm()
    policy = ScriptedPolicy([Action.delta(5.0, 0.0, 0.0), Action.delta(-5.0, 0.0, 0.0)])
    episode = await rollout(arm, policy, envelope=SafetyEnvelope(max_step_mm=10.0, max_steps=4))

    self.assertEqual(episode.metadata["clamped_steps"], 0)
    self.assertEqual(episode.metadata["n_clamped"], 0)
    self.assertEqual(episode.metadata["clamp_magnitude_mm"], 0.0)
    self.assertEqual(episode.metadata["clamped_fraction"], 0.0)
    self.assertTrue(
      all(step.info == {"n_clamped": 0, "clamp_magnitude_mm": 0.0} for step in episode.steps)
    )

  async def test_a_workspace_clip_is_counted_as_a_clamp(self):
    arm = await sim_arm(workspace=Workspace(x=(-50.0, 50.0), y=(-50.0, 50.0), z=(0.0, 200.0)))
    envelope = SafetyEnvelope(
      max_step_mm=25.0,
      workspace=Workspace(x=(-50.0, 50.0), y=(-50.0, 50.0), z=(0.0, 200.0)),
      max_steps=4,
    )
    policy = ScriptedPolicy([Action.delta(20.0, 0.0, 0.0)] * 4)
    episode = await rollout(arm, policy, envelope=envelope)

    self.assertGreaterEqual(episode.metadata["clamped_steps"], 1)
    clipped = [s for s in episode.steps if "clamped" in s.info]
    self.assertIn("clipped to workspace", clipped[-1].info["clamped"][0])
    self.assertLessEqual(episode.final_observation.x, 50.0)

  async def test_rollout_issues_no_motion_when_current_pose_is_outside_the_envelope(self):
    arm = await sim_arm(start=(0.0, 0.0, 300.0, 0.0))
    moves_before = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    envelope = SafetyEnvelope(
      workspace=Workspace(x=(-500.0, 500.0), y=(-500.0, 500.0), z=(0.0, 200.0)),
      max_steps=2,
    )

    with self.assertRaises(UnsafeAction):
      await rollout(arm, ScriptedPolicy([Action.delta(dz=-25.0)]), envelope=envelope)

    moves_after = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    self.assertEqual(moves_after, moves_before)

  async def test_a_joint_action_from_a_policy_is_refused_by_the_rollout(self):
    arm = await sim_arm()
    policy = ScriptedPolicy([Action.joint_targets({1: 10.0})])
    with self.assertRaises(UnsafeAction):
      await rollout(arm, policy, envelope=SafetyEnvelope(max_steps=3))

  async def test_the_policy_is_reset_before_the_rollout_by_default(self):
    arm = await sim_arm()
    policy = ScriptedPolicy([Action.delta(1.0, 0.0, 0.0), Action.hold(terminate=True)])
    policy.i = 2  # exhausted by a previous run

    episode = await rollout(arm, policy, envelope=SafetyEnvelope(max_steps=10))
    self.assertEqual(len(episode), 2)
    self.assertEqual(episode.metadata["stop_reason"], "policy_terminated")


class TickClock:
  """A clock that advances a fixed step per read, so an episode is reproducible.

  Needed for the byte-identity test below. ``time.monotonic`` puts a different
  number in every observation, so two otherwise identical runs differ in a field
  that has nothing to do with what is being compared.
  """

  def __init__(self, step: float = 0.01) -> None:
    self.step = float(step)
    self.t = 0.0

  def __call__(self) -> float:
    self.t += self.step
    return self.t


class StubPredictor:
  """Returns a scripted confidence per call, and nothing else."""

  name = "stub"
  state_dim = 4
  horizon = 1
  action_space = "ee_delta_mm"
  training_fingerprints = frozenset({"none:analytic-null-fit-on-no-data"})
  fitted_components = ()
  license_text = "MIT, this test."

  def __init__(self, confidences: Sequence[float]) -> None:
    self.confidences = list(confidences)
    self.calls = 0

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    value = self.confidences[min(self.calls, len(self.confidences) - 1)]
    self.calls += 1
    return RolloutPrediction(
      state=np.zeros((int(request.horizon), 4)),
      confidence=float(value),
      abstain=False,
      flag=False,
      readout="identity",
      n_forward_passes=1,
      wall_clock_s=0.0,
    )


def request_for(observation: Observation) -> RolloutRequest:
  """Build the request for one observation, with a frame stamped on its own clock."""
  return RolloutRequest(
    task_id="reach",
    episode_id="ep-%d" % round(observation.t * 1e6),
    state=observation.pose_vector(),
    state_fields=("x", "y", "z", "rz"),
    state_units=("mm", "mm", "mm", "rad"),
    frames=(FrameRef("f.png", "sha256:f", float(observation.t), "rendered"),),
    actions=np.zeros((1, len(ACTION_FIELDS))),
    horizon=1,
    tolerance=np.array([2.0, 2.0, 2.0, 0.05]),
  )


def calibrated_screen(trust_horizon: int = 4, max_frame_age_s: float = 10.0) -> Screen:
  points = [CalibrationPoint("cal%d" % i, 0.80 + 0.01 * i) for i in range(12)]
  return Screen.calibrate(
    points,
    source="dev seeds 0-99",
    trust_horizon=trust_horizon,
    max_frame_age_s=max_frame_age_s,
  )


def screened(policy, confidences, **kwargs) -> ScreenedPolicy:
  return ScreenedPolicy(
    policy, calibrated_screen(), StubPredictor(confidences), request_for, **kwargs
  )


class TestClampAccounting(unittest.IsolatedAsyncioTestCase):
  """What the envelope changed is counted, per step and per episode.

  The thing this guards is not arithmetic, it is visibility. Before these fields
  existed the only evidence of clamping was a free-text note, and a null control
  running with half its steps rewritten looked in the report exactly like one
  running clean.
  """

  async def test_every_step_carries_the_accounting_whether_or_not_it_clamped(self):
    arm = await sim_arm()
    policy = ScriptedPolicy([Action.delta(100.0, 0.0, 0.0), Action.delta(1.0, 0.0, 0.0)])
    episode = await rollout(arm, policy, envelope=SafetyEnvelope(max_step_mm=10.0, max_steps=2))

    for step in episode.steps:
      self.assertIn("n_clamped", step.info)
      self.assertIn("clamp_magnitude_mm", step.info)
    self.assertEqual(episode.steps[0].info["n_clamped"], 1)
    self.assertEqual(episode.steps[1].info["n_clamped"], 0)
    self.assertEqual(episode.steps[1].info["clamp_magnitude_mm"], 0.0)

  async def test_the_magnitude_is_how_far_the_sent_command_sits_from_the_asked_one(self):
    arm = await sim_arm()
    policy = ScriptedPolicy([Action.delta(100.0, 0.0, 0.0)])
    episode = await rollout(arm, policy, envelope=SafetyEnvelope(max_step_mm=10.0, max_steps=1))
    # Asked for 100 mm along x, sent 10 mm along x. The envelope removed 90 mm of
    # commanded motion, and until now nothing recorded that it had.
    self.assertAlmostEqual(episode.steps[0].info["clamp_magnitude_mm"], 90.0, places=9)

  async def test_the_episode_totals_what_the_steps_recorded(self):
    arm = await sim_arm()
    policy = ScriptedPolicy(
      [Action.delta(100.0, 0.0, 0.0), Action.delta(1.0, 0.0, 0.0), Action.delta(60.0, 0.0, 0.0)]
    )
    episode = await rollout(arm, policy, envelope=SafetyEnvelope(max_step_mm=10.0, max_steps=3))

    self.assertEqual(episode.metadata["n_clamped"], 2)
    self.assertEqual(episode.metadata["n_clamped"], episode.metadata["clamped_steps"])
    self.assertAlmostEqual(
      episode.metadata["clamp_magnitude_mm"],
      sum(s.info["clamp_magnitude_mm"] for s in episode.steps),
      places=9,
    )
    self.assertAlmostEqual(episode.metadata["clamped_fraction"], 2.0 / 3.0, places=12)
    self.assertAlmostEqual(clamped_fraction(episode), 2.0 / 3.0, places=12)

  def test_an_episode_with_no_steps_reports_nan_rather_than_zero(self):
    """A run that clamped nothing because it did nothing established nothing."""
    self.assertTrue(math.isnan(clamped_fraction(Episode())))
    self.assertTrue(is_throttled(float("nan")))

  def test_the_throttle_bound_is_an_absolute_constant(self):
    self.assertEqual(THROTTLED_CLAMP_FRACTION, 0.10)
    self.assertFalse(is_throttled(0.10))
    self.assertTrue(is_throttled(0.1001))

  async def test_the_random_null_control_is_throttled_at_the_package_defaults(self):
    """The measured rate that motivates all of the above.

    ``RandomPolicy(step_mm=25.0)`` draws displacements uniformly from the cube of
    half-width 25 mm; ``SafetyEnvelope(max_step_mm=25.0)`` bounds them to the
    inscribed ball. The cube is ``1 - pi/6`` outside that ball, 47.6 percent, and
    the workspace clip adds a little more. Measured here at five trials of sixty
    steps, which is the shape the evaluation harness runs by default.

    The assertion is a band rather than a point because the exact figure moves
    with the trial count. What must not move is which side of
    ``THROTTLED_CLAMP_FRACTION`` it lands on.
    """
    total = 0
    clamped = 0
    for trial in range(5):
      arm = await sim_arm()
      episode = await rollout(
        arm,
        RandomPolicy(step_mm=25.0, seed=trial),
        envelope=SafetyEnvelope(max_step_mm=25.0, max_steps=60),
      )
      total += len(episode)
      clamped += episode.metadata["n_clamped"]

    fraction = clamped / total
    self.assertEqual(total, 300)
    self.assertGreater(fraction, 0.40)
    self.assertLess(fraction, 0.55)
    self.assertAlmostEqual(fraction, 1.0 - math.pi / 6.0, delta=0.05)
    self.assertTrue(is_throttled(fraction))


class TestPerCommandStepLimit(unittest.IsolatedAsyncioTestCase):
  """A command the controller will accept and not execute is refused, not clamped."""

  def test_no_declared_limit_means_no_refusal(self):
    envelope = SafetyEnvelope(max_step_mm=25.0)
    self.assertIsNone(envelope.per_command_step_mm)
    safe, notes = envelope.clamp(Action.delta(100.0, 0.0, 0.0), START_POSE)
    self.assertEqual(len(notes), 1)
    self.assertAlmostEqual(float(np.linalg.norm(safe.translation)), 25.0, places=9)

  def test_a_clamped_command_still_above_the_declared_limit_is_refused(self):
    """The package defaults are exactly this case: clamp to 25, controller does 10."""
    envelope = SafetyEnvelope(max_step_mm=25.0, per_command_step_mm=10.0)
    with self.assertRaises(StepLimitExceeded) as ctx:
      envelope.clamp(Action.delta(100.0, 0.0, 0.0), START_POSE)
    message = str(ctx.exception)
    self.assertIn("25.00 mm", message)
    self.assertIn("10.00 mm", message)
    self.assertIn("moves nothing, and returns success", message)

  def test_a_command_inside_the_declared_limit_passes(self):
    envelope = SafetyEnvelope(max_step_mm=25.0, per_command_step_mm=10.0)
    safe, notes = envelope.clamp(Action.delta(5.0, 0.0, 0.0), START_POSE)
    self.assertEqual(notes, [])
    np.testing.assert_allclose(safe.pose[:3], (5.0, 0.0, 0.0), atol=1e-12)

  def test_a_command_exactly_at_the_declared_limit_passes(self):
    envelope = SafetyEnvelope(max_step_mm=10.0, per_command_step_mm=10.0)
    direction = np.array([1.0, 1.0, 1.0])
    step = direction * (10.0 / float(np.linalg.norm(direction)))
    safe, notes = envelope.clamp(Action.delta(*step), START_POSE)
    self.assertEqual(notes, [])
    np.testing.assert_allclose(safe.pose[:3], step, atol=1e-9)

  def test_it_is_an_unsafe_action_because_clamping_cannot_fix_it(self):
    self.assertTrue(issubclass(StepLimitExceeded, UnsafeAction))

  def test_a_malformed_declaration_is_refused_at_construction(self):
    for value in (0.0, -1.0, float("inf"), float("nan"), "10"):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          SafetyEnvelope(per_command_step_mm=value)

  def test_an_undeclaring_backend_reads_as_none_rather_than_as_generous(self):
    self.assertIsNone(per_command_step_mm_of(SimulationArmBackend()))

  def test_an_unusable_backend_declaration_is_refused(self):
    backend = SimulationArmBackend()
    backend.per_command_step_mm = -3.0
    with self.assertRaises(ValueError) as ctx:
      per_command_step_mm_of(backend)
    self.assertIn("reads as a checked limit", str(ctx.exception))

  async def test_rollout_picks_up_a_backend_declaration(self):
    arm = await sim_arm()
    arm.backend.per_command_step_mm = 10.0
    with self.assertRaises(StepLimitExceeded):
      await rollout(
        arm,
        ScriptedPolicy([Action.delta(100.0, 0.0, 0.0)]),
        envelope=SafetyEnvelope(max_step_mm=25.0, max_steps=2),
      )

  async def test_an_explicit_envelope_declaration_wins_over_the_backend(self):
    arm = await sim_arm()
    arm.backend.per_command_step_mm = 1.0
    episode = await rollout(
      arm,
      ScriptedPolicy([Action.delta(4.0, 0.0, 0.0)]),
      envelope=SafetyEnvelope(max_step_mm=5.0, per_command_step_mm=5.0, max_steps=1),
    )
    self.assertEqual(len(episode), 1)

  async def test_the_callers_envelope_is_not_mutated_by_the_backend_lookup(self):
    arm = await sim_arm()
    arm.backend.per_command_step_mm = 50.0
    envelope = SafetyEnvelope(max_step_mm=5.0, max_steps=1)
    await rollout(arm, NoOpPolicy(), envelope=envelope)
    self.assertIsNone(envelope.per_command_step_mm)


class TestScreenKwarg(unittest.IsolatedAsyncioTestCase):
  """The screen changes nothing when it is None, and everything when it is not."""

  async def _run(self, seed: int, **kwargs) -> str:
    arm = await sim_arm()
    recorder = EpisodeRecorder(arm, task="reach reader", clock=TickClock())
    episode = await rollout(
      arm,
      RandomPolicy(step_mm=25.0, seed=seed),
      envelope=SafetyEnvelope(max_step_mm=25.0, max_steps=8),
      recorder=recorder,
      **kwargs,
    )
    return json.dumps(episode.to_dict(), sort_keys=True)

  async def test_passing_screen_none_is_byte_identical_to_not_passing_it(self):
    """The whole episode, serialised, compared as text. Not a spot check.

    A deterministic clock is supplied so the comparison covers the observation
    timestamps too, which is where a difference would otherwise hide.
    """
    without = await self._run(7)
    with_none = await self._run(7, screen=None)
    self.assertEqual(without, with_none)
    # The run is a real one and not a degenerate two-step case: it clamped, so
    # the comparison covers the clamp bookkeeping as well as the clean steps.
    self.assertIn('"n_clamped"', without)
    self.assertNotIn('"screen"', without)

  async def test_an_admitted_verdict_is_recorded_and_the_action_runs(self):
    arm = await sim_arm()
    policy = screened(ScriptedPolicy([Action.delta(5.0, 0.0, 0.0)] * 2), [0.99])
    episode = await rollout(
      arm, policy, envelope=SafetyEnvelope(max_step_mm=25.0, max_steps=2), screen=policy
    )

    self.assertEqual(len(episode), 2)
    self.assertIs(episode.metadata["screened"], True)
    for step in episode.steps:
      self.assertEqual(step.info["screen"]["outcome"], "act")
      self.assertIsNone(step.info["screen"]["reason"])
      self.assertIn("split-conformal on successes", step.info["screen"]["calibrated_on"])
    self.assertAlmostEqual(episode.final_observation.x, START_POSE[0] + 10.0)

  async def test_a_clamped_admission_is_downgraded_and_nothing_is_sent(self):
    """The clamp-downgrade rule, end to end.

    The screen admitted a 100 mm step. The envelope turned it into a 10 mm step.
    The verdict was about the first one, so it does not carry over to the second,
    and the arm does not move at all.
    """
    arm = await sim_arm()
    policy = screened(ScriptedPolicy([Action.delta(100.0, 0.0, 0.0)]), [0.99])
    episode = await rollout(
      arm, policy, envelope=SafetyEnvelope(max_step_mm=10.0, max_steps=4), screen=policy
    )

    self.assertEqual(len(episode), 1)
    verdict = episode.steps[0].info["screen"]
    self.assertEqual(verdict["outcome"], "abstain")
    self.assertEqual(verdict["reason"], AbstainReason.CLAMPED.value)
    self.assertAlmostEqual(verdict["clamp_magnitude_mm"], 90.0, places=9)
    self.assertFalse(verdict["reobserve"])
    # The screen said act; the recorded action is a hold and the arm is where it
    # started. Both halves matter: one without the other is a bookkeeping change.
    self.assertEqual(episode.steps[0].action, Action.hold())
    self.assertEqual(episode.steps[0].info["requested"]["pose"], [100.0, 0.0, 0.0, 0.0])
    self.assertAlmostEqual(episode.final_observation.x, START_POSE[0])
    self.assertEqual(episode.metadata["stop_reason"], "screen_declined")

  async def test_a_refusal_stops_the_run_safe_with_no_fallback_action(self):
    arm = await sim_arm()
    policy = screened(
      ScriptedPolicy([Action.delta(5.0, 0.0, 0.0)] * 4), [0.10], max_reobservations=0
    )
    episode = await rollout(
      arm, policy, envelope=SafetyEnvelope(max_step_mm=25.0, max_steps=4), screen=policy
    )

    self.assertEqual(len(episode), 1)
    self.assertEqual(episode.metadata["stop_reason"], "screen_declined")
    self.assertEqual(episode.steps[0].info["screen"]["reason"], "low_confidence")
    self.assertEqual(episode.steps[0].action, Action.hold())
    self.assertAlmostEqual(episode.final_observation.x, START_POSE[0])

  async def test_a_bounded_re_observation_looks_again_and_then_the_run_continues(self):
    arm = await sim_arm()
    policy = screened(ScriptedPolicy([Action.delta(5.0, 0.0, 0.0)] * 4), [0.10, 0.10, 0.99])
    episode = await rollout(
      arm, policy, envelope=SafetyEnvelope(max_step_mm=25.0, max_steps=4), screen=policy
    )

    self.assertEqual(len(episode), 4)
    self.assertEqual(episode.metadata["stop_reason"], "budget_exhausted")
    outcomes = [s.info["screen"]["outcome"] for s in episode.steps]
    self.assertEqual(outcomes, ["abstain", "abstain", "act", "act"])
    self.assertTrue(episode.steps[0].info["screen"]["reobserve"])
    self.assertTrue(episode.steps[1].info["screen"]["reobserve"])
    # Two holds while it looked again, then two real 5 mm steps.
    self.assertAlmostEqual(episode.final_observation.x, START_POSE[0] + 10.0)

  async def test_the_third_refusal_is_terminal(self):
    arm = await sim_arm()
    policy = screened(ScriptedPolicy([Action.delta(5.0, 0.0, 0.0)] * 8), [0.10])
    episode = await rollout(
      arm, policy, envelope=SafetyEnvelope(max_step_mm=25.0, max_steps=8), screen=policy
    )

    self.assertEqual(len(episode), 3)
    self.assertEqual(episode.metadata["stop_reason"], "screen_declined")
    self.assertAlmostEqual(episode.final_observation.x, START_POSE[0])

  async def test_a_bare_screen_is_refused_at_the_call_site(self):
    arm = await sim_arm()
    with self.assertRaises(TypeError) as ctx:
      await rollout(arm, NoOpPolicy(), envelope=SafetyEnvelope(max_steps=1), screen=Screen(
        theta=0.5, trust_horizon=1, max_frame_age_s=1.0, calibrated_on="dev"
      ))
    self.assertIn("ScreenedPolicy", str(ctx.exception))

  async def test_a_review_returning_the_wrong_type_is_refused_before_motion(self):
    class WrongScreen:
      def review(self, observation, action):
        return {"outcome": "act"}

    arm = await sim_arm()
    moves_before = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    with self.assertRaises(UnsafeAction) as ctx:
      await rollout(
        arm,
        ScriptedPolicy([Action.delta(5.0, 0.0, 0.0)]),
        envelope=SafetyEnvelope(max_steps=1),
        screen=WrongScreen(),
      )
    self.assertIn("would have to treat as an admission", str(ctx.exception))
    moves_after = len([e for e in arm.backend.trace if e["action"] == "move_to"])
    self.assertEqual(moves_after, moves_before)

  async def test_an_awaitable_review_is_refused_and_the_coroutine_is_closed(self):
    class AsyncScreen:
      async def review(self, observation, action):
        return None

    arm = await sim_arm()
    with self.assertRaises(UnsafeAction) as ctx:
      await rollout(
        arm,
        NoOpPolicy(),
        envelope=SafetyEnvelope(max_steps=1),
        screen=AsyncScreen(),
      )
    self.assertIn("Screening is synchronous", str(ctx.exception))


if __name__ == "__main__":
  unittest.main()
