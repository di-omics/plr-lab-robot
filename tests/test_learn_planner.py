"""Tests for plr_lr.learn.policy and plr_lr.learn.planner.

Organised around the claims these two modules make, because the claims are what
would silently rot into something that still runs and still reports a number.

For the policies the claims are: that the two null controls really are null (one
never moves, the other only ever moves at random inside its declared bounds, and
both are reproducible enough to be worth reporting); that a replay is a replay,
in order, and stops rather than holding position forever; that the servo holds
still when it cannot see, stops when it has arrived, never asks for more than its
declared step, and actually closes the loop on the geometry; and that learned
adapters validate an injected transport rather than pretending a checkpoint is
already an integration.

For the planner the claims are: that goal error is measured the same way for one
pose and for a batch of them, and that rz error wraps rather than reporting the
long way round; that nothing the sampler proposes exceeds ``step_mm``, because a
safety envelope set to the same limit would silently rescale every step and the
planner would be scoring motions the arm never made; that the search width it
reports is the number of candidates that actually differ rather than the number
of rows it drew; and that ``_aim`` really inverts the dynamics model rather than
aiming geometrically, which is the one question a planner asks a model that a
mediocre model cannot fake.

The geometry for the servo tests is the overhead camera from
tests/test_guided_pick.py, re-expressed as an eye-in-hand rig so the policy
computes the camera pose itself from the observation.
"""

from __future__ import annotations

import unittest

import numpy as np

from plr_lr.learn import (
  ACTION_UNITS,
  BASE_STATE_UNITS,
  GRIPPER_UNCHANGED,
  Action,
  ActionMode,
  AdapterOutputError,
  Episode,
  Goal,
  GR00TPolicy,
  KinematicDynamics,
  NoOpPolicy,
  Observation,
  OpenVLAPolicy,
  Pi0Policy,
  Policy,
  RandomPolicy,
  ReplayPolicy,
  SafetyEnvelope,
  SamplingPlanner,
  VisionServoPolicy,
  VLAPolicy,
  reset_policy,
)
from plr_lr.learn.planner import n_distinct_sequences
from plr_lr.vision import CameraIntrinsics, Detection, SimDetector, Transform, base_from_camera

INTR = CameraIntrinsics(fx=1400.0, fy=1400.0, cx=640.0, cy=512.0)
PLANE_Z = 10.0

# The same downward-looking flip tests/test_guided_pick.py uses for its overhead
# camera, written as cam_from_tool for an eye-in-hand rig. The matrix is its own
# inverse, so this places the camera at the tool origin looking at the deck, and
# the policy's own base_from_camera() reproduces the overhead camera exactly.
DOWNWARD = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
CAM_FROM_TOOL = Transform.from_rt(DOWNWARD, [0.0, 0.0, 0.0])

# High enough above the deck that the targets used below stay inside the frame:
# at 240 mm of standoff this lens sees +-109 mm in u and +-88 mm in v, and
# SimDetector drops anything that falls outside the image.
TOOL_Z = 250.0


def wrist_view(x, y, z=TOOL_Z, rz=0.0):
  """The base_from_cam an eye-in-hand camera has when the tool is at (x, y, z)."""
  base_from_tool = Transform.from_euler(0.0, 0.0, rz, t=(x, y, z))
  return base_from_camera(base_from_tool, CAM_FROM_TOOL)


def servo_observation(x, y, target_xy, z=TOOL_Z, rz=0.0, label="labware"):
  """An observation whose detection is the projection of a known deck point."""
  dets = SimDetector(INTR).observe(
    {"plate": np.array([target_xy[0], target_xy[1], PLANE_Z])},
    wrist_view(x, y, z, rz),
    label=label,
  )
  return Observation(x=x, y=y, z=z, rz=rz, detections=tuple(dets))


def servo(**kwargs):
  kwargs.setdefault("plane_z", PLANE_Z)
  return VisionServoPolicy(INTR, CAM_FROM_TOOL, **kwargs)


def replayable(deltas, task="replay"):
  episode = Episode(task=task)
  for i, (dx, dy) in enumerate(deltas):
    episode.append(Observation(t=float(i)), Action.delta(dx=dx, dy=dy))
  return episode


def observation_at(pose):
  return Observation(x=float(pose[0]), y=float(pose[1]), z=float(pose[2]), rz=float(pose[3]))


class NullControlTests(unittest.TestCase):
  """The two policies that exist to fail."""

  def test_noop_commands_nothing_whatever_it_sees(self):
    policy = NoOpPolicy()
    for obs in (
      Observation(),
      Observation(x=180.0, y=-40.0, z=25.0, rz=1.2),
      servo_observation(0.0, 0.0, target_xy=(60.0, -40.0)),
    ):
      action = policy.act(obs)
      self.assertIs(action.mode, ActionMode.DELTA)
      np.testing.assert_array_equal(action.translation, np.zeros(3))
      self.assertEqual(action.pose[3], 0.0)
      self.assertIsNone(action.gripper)
      self.assertFalse(action.terminate)

  def test_noop_satisfies_the_policy_protocol(self):
    self.assertIsInstance(NoOpPolicy(), Policy)
    self.assertIsInstance(SamplingPlanner(Goal.at(0.0, 0.0, 0.0)), Policy)

  def test_random_is_reproducible_under_a_seed(self):
    a = RandomPolicy(seed=17)
    b = RandomPolicy(seed=17)
    obs = Observation()
    first = [a.act(obs).pose for _ in range(8)]
    second = [b.act(obs).pose for _ in range(8)]
    self.assertEqual(first, second)

  def test_random_differs_under_a_different_seed(self):
    a = [RandomPolicy(seed=1).act(Observation()).pose for _ in range(4)]
    b = [RandomPolicy(seed=2).act(Observation()).pose for _ in range(4)]
    self.assertNotEqual(a, b)

  def test_random_reset_restores_the_sequence(self):
    policy = RandomPolicy(seed=17)
    obs = Observation()
    first = [policy.act(obs).pose for _ in range(5)]
    self.assertNotEqual(first, [policy.act(obs).pose for _ in range(5)])
    policy.reset()
    self.assertEqual(first, [policy.act(obs).pose for _ in range(5)])

  def test_random_stays_inside_its_declared_step_bounds(self):
    policy = RandomPolicy(step_mm=12.0, step_rad=0.05, seed=3)
    obs = Observation()
    for _ in range(500):
      action = policy.act(obs)
      self.assertIs(action.mode, ActionMode.DELTA)
      self.assertLessEqual(float(np.abs(action.translation).max()), 12.0)
      self.assertLessEqual(abs(action.pose[3]), 0.05)
      self.assertFalse(action.terminate)

  def test_random_actually_moves(self):
    policy = RandomPolicy(step_mm=12.0, step_rad=0.05, seed=3)
    obs = Observation()
    magnitudes = [float(np.linalg.norm(policy.act(obs).translation)) for _ in range(50)]
    self.assertGreater(min(magnitudes), 0.0)
    self.assertGreater(max(magnitudes), 6.0)

  def test_reset_policy_is_a_no_op_on_a_policy_without_one(self):
    reset_policy(NoOpPolicy())
    policy = RandomPolicy(seed=4)
    first = policy.act(Observation()).pose
    reset_policy(policy)
    self.assertEqual(first, policy.act(Observation()).pose)


class ReplayPolicyTests(unittest.TestCase):
  def test_actions_come_back_in_order_then_terminate(self):
    deltas = [(1.0, 0.0), (0.0, 2.0), (-3.0, 3.0)]
    policy = ReplayPolicy(replayable(deltas))
    obs = Observation()
    for dx, dy in deltas:
      action = policy.act(obs)
      self.assertFalse(action.terminate)
      self.assertEqual(action.pose[:2], (dx, dy))
    ended = policy.act(obs)
    self.assertTrue(ended.terminate)
    np.testing.assert_array_equal(ended.translation, np.zeros(3))
    self.assertTrue(policy.act(obs).terminate)

  def test_reset_replays_from_the_start(self):
    policy = ReplayPolicy(replayable([(1.0, 0.0), (0.0, 2.0)]))
    obs = Observation()
    first = [policy.act(obs).pose for _ in range(2)]
    self.assertTrue(policy.act(obs).terminate)
    policy.reset()
    self.assertEqual(first, [policy.act(obs).pose for _ in range(2)])

  def test_loop_wraps_instead_of_terminating(self):
    deltas = [(1.0, 0.0), (0.0, 2.0)]
    policy = ReplayPolicy(replayable(deltas), loop=True)
    obs = Observation()
    seen = [policy.act(obs).pose[:2] for _ in range(6)]
    self.assertEqual(seen, [(1.0, 0.0), (0.0, 2.0)] * 3)
    self.assertFalse(any(policy.act(obs).terminate for _ in range(4)))

  def test_an_empty_recording_terminates_even_when_looping(self):
    for loop in (False, True):
      policy = ReplayPolicy(Episode(task="empty"), loop=loop)
      self.assertTrue(policy.act(Observation()).terminate)


class VisionServoTests(unittest.TestCase):
  def test_no_detection_holds_rather_than_guessing(self):
    action = servo().act(Observation(x=50.0, y=50.0, z=TOOL_Z))
    self.assertIs(action.mode, ActionMode.DELTA)
    np.testing.assert_array_equal(action.translation, np.zeros(3))
    self.assertFalse(action.terminate)

  def test_a_detection_of_another_label_is_not_a_detection(self):
    obs = servo_observation(0.0, 0.0, target_xy=(40.0, 20.0), label="labware")
    self.assertEqual(len(obs.detections), 1)
    action = servo(label="tube").act(obs)
    np.testing.assert_array_equal(action.translation, np.zeros(3))
    self.assertFalse(action.terminate)

  def test_arrival_terminates(self):
    policy = servo(tolerance_mm=1.0)
    obs = servo_observation(40.0, -20.0, target_xy=(40.0, -20.0))
    action = policy.act(obs)
    self.assertTrue(action.terminate)
    np.testing.assert_array_equal(action.translation, np.zeros(3))

    near = servo_observation(40.5, -20.0, target_xy=(40.0, -20.0))
    self.assertTrue(policy.act(near).terminate)

  def test_just_outside_tolerance_still_moves(self):
    policy = servo(tolerance_mm=1.0, gain=1.0)
    obs = servo_observation(0.0, 0.0, target_xy=(1.5, 0.0))
    action = policy.act(obs)
    self.assertFalse(action.terminate)
    self.assertAlmostEqual(float(action.pose[0]), 1.5, places=6)

  def test_max_step_caps_the_command_without_changing_its_direction(self):
    policy = servo(gain=1.0, max_step_mm=5.0)
    error = np.array([80.0, -60.0])
    obs = servo_observation(0.0, 0.0, target_xy=(error[0], error[1]))
    step = policy.act(obs).translation
    self.assertAlmostEqual(float(np.linalg.norm(step)), 5.0, places=9)
    self.assertAlmostEqual(float(step[2]), 0.0, places=12)
    unit = error / np.linalg.norm(error)
    np.testing.assert_allclose(step[:2] / 5.0, unit, atol=1e-9)

  def test_one_step_commands_the_gain_fraction_of_the_error(self):
    policy = servo(gain=0.6, max_step_mm=100.0)
    obs = servo_observation(10.0, 10.0, target_xy=(40.0, -10.0))
    step = policy.act(obs).translation
    np.testing.assert_allclose(step[:2], np.array([30.0, -20.0]) * 0.6, atol=1e-6)

  def test_the_most_confident_detection_wins(self):
    view = wrist_view(0.0, 0.0)
    projected = SimDetector(INTR).observe(
      {
        "left": np.array([-60.0, 0.0, PLANE_Z]),
        "right": np.array([60.0, 0.0, PLANE_Z]),
      },
      view,
    )
    uv = {d.truth_name: d.uv for d in projected}
    self.assertEqual(set(uv), {"left", "right"})
    obs = Observation(
      z=TOOL_Z,
      detections=(
        Detection(label="labware", uv=uv["left"], confidence=0.3),
        Detection(label="labware", uv=uv["right"], confidence=0.9),
      ),
    )
    step = servo(gain=1.0, max_step_mm=100.0).act(obs).translation
    self.assertAlmostEqual(float(step[0]), 60.0, places=5)

  def test_the_loop_drives_the_in_plane_error_to_zero(self):
    policy = servo(gain=0.6, tolerance_mm=1.0, max_step_mm=25.0)
    target = (60.0, -40.0)
    x, y = 0.0, 0.0
    errors = [float(np.hypot(target[0] - x, target[1] - y))]
    terminated = False
    for _ in range(40):
      action = policy.act(servo_observation(x, y, target_xy=target))
      if action.terminate:
        terminated = True
        break
      x += float(action.pose[0])
      y += float(action.pose[1])
      errors.append(float(np.hypot(target[0] - x, target[1] - y)))
    self.assertTrue(terminated)
    self.assertLessEqual(errors[-1], 1.0)
    for previous, current in zip(errors, errors[1:]):
      self.assertLess(current, previous)

  def test_the_loop_converges_with_the_tool_rotated(self):
    policy = servo(gain=0.6, tolerance_mm=1.0, max_step_mm=25.0)
    target = (35.0, 25.0)
    rz = 0.7
    x, y = 0.0, 0.0
    for _ in range(40):
      action = policy.act(servo_observation(x, y, target_xy=target, rz=rz))
      if action.terminate:
        break
      x += float(action.pose[0])
      y += float(action.pose[1])
    self.assertLessEqual(float(np.hypot(target[0] - x, target[1] - y)), 1.0)

  def test_a_gain_outside_zero_to_one_is_refused(self):
    for bad in (0.0, -0.5, 1.5):
      with self.assertRaises(ValueError):
        servo(gain=bad)


class VLAPolicyAdapterTests(unittest.TestCase):
  """The learned seams adapt an injected callable and validate every output."""

  def test_every_vla_seam_without_a_transport_refuses_honestly(self):
    for cls, name in (
      (VLAPolicy, "VLA"),
      (OpenVLAPolicy, "OpenVLA"),
      (Pi0Policy, "pi0"),
      (GR00TPolicy, "GR00T N1"),
    ):
      policy = cls(checkpoint="/nonexistent/weights.pt", instruction="pick the plate")
      with self.assertRaises(NotImplementedError) as ctx:
        policy.act(Observation())
      message = str(ctx.exception)
      self.assertIn(name, message)
      self.assertIn("checkpoint alone is not an adapter", message)
      self.assertIn("synchronous callable", message)

  def test_every_named_seam_maps_the_documented_delta_vector(self):
    observation = Observation(x=10.0, image_path="frames/000001.png")
    for cls in (VLAPolicy, OpenVLAPolicy, Pi0Policy, GR00TPolicy):
      calls = []

      def transport(seen, instruction):
        calls.append((seen, instruction))
        return [1.0, -2.0, 3.0, 0.25, GRIPPER_UNCHANGED]

      policy = cls(
        checkpoint="/models/example",
        instruction="pick the plate",
        transport=transport,
        action_mode=ActionMode.DELTA,
      )
      action = policy.act(observation)

      with self.subTest(policy=cls.__name__):
        self.assertEqual(calls, [(observation, "pick the plate")])
        self.assertEqual(policy.checkpoint, "/models/example")
        self.assertEqual(policy.instruction, "pick the plate")
        self.assertIs(policy.action_mode, ActionMode.DELTA)
        self.assertIs(action.mode, ActionMode.DELTA)
        self.assertEqual(action.pose, (1.0, -2.0, 3.0, 0.25))
        self.assertIsNone(action.gripper)

  def test_an_absolute_vector_and_real_gripper_width_are_mapped(self):
    policy = Pi0Policy(
      transport=lambda _obs, _instruction: np.array([100.0, -20.0, 30.0, 0.5, 42.0]),
      action_mode="absolute",
    )
    action = policy.act(Observation())
    self.assertIs(action.mode, ActionMode.ABSOLUTE)
    self.assertEqual(action.pose, (100.0, -20.0, 30.0, 0.5))
    self.assertEqual(action.gripper, 42.0)

  def test_a_direct_action_preserves_mode_gripper_and_termination(self):
    direct = Action.absolute(10.0, 20.0, 30.0, 0.5, gripper=25.0, terminate=True)
    policy = OpenVLAPolicy(
      checkpoint="weights.safetensors",
      instruction="place the tube",
      transport=lambda _obs, _instruction: direct,
      action_mode=ActionMode.DELTA,
    )
    self.assertEqual(policy.act(Observation()), direct)
    self.assertEqual(policy.checkpoint, "weights.safetensors")
    self.assertEqual(policy.instruction, "place the tube")

  def test_vector_output_must_have_exactly_the_documented_shape(self):
    for output in (
      1.0,
      [1.0, 2.0, 3.0, 4.0],
      [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
      np.zeros((1, 5)),
      {"action": [1.0, 2.0, 3.0, 4.0, 5.0]},
    ):
      policy = VLAPolicy(
        transport=lambda _obs, _instruction, value=output: value,
      )
      with self.subTest(output=output), self.assertRaises(AdapterOutputError) as ctx:
        policy.act(Observation())
      self.assertIn("transport", str(ctx.exception))

  def test_vector_output_must_be_finite(self):
    for value in (np.nan, np.inf, -np.inf):
      output = np.array([1.0, 2.0, 3.0, 0.5, 40.0])
      output[2] = value
      policy = VLAPolicy(
        transport=lambda _obs, _instruction, result=output: result,
      )
      with self.subTest(value=value), self.assertRaises(AdapterOutputError) as ctx:
        policy.act(Observation())
      self.assertIn("non-finite", str(ctx.exception))

  def test_direct_action_output_is_validated_too(self):
    for output in (
      Action.delta(dx=np.nan),
      Action(mode=ActionMode.DELTA, pose=(1.0, 2.0, 3.0)),
      Action.delta(gripper=np.inf),
      Action.delta(gripper=-0.5),
    ):
      policy = VLAPolicy(
        transport=lambda _obs, _instruction, result=output: result,
      )
      with self.subTest(output=output), self.assertRaises(AdapterOutputError):
        policy.act(Observation())

  def test_joint_modes_are_rejected_with_a_mapping_remedy(self):
    with self.assertRaises(ValueError) as ctx:
      VLAPolicy(transport=lambda _obs, _instruction: Action.hold(), action_mode="joints")
    self.assertIn("Cartesian", str(ctx.exception))

    policy = VLAPolicy(
      transport=lambda _obs, _instruction: Action.joint_targets({1: 10.0})
    )
    with self.assertRaises(AdapterOutputError) as ctx:
      policy.act(Observation())
    self.assertIn("JOINTS", str(ctx.exception))
    self.assertIn("mapping", str(ctx.exception))

  def test_negative_vector_gripper_is_only_valid_for_the_unchanged_sentinel(self):
    policy = VLAPolicy(
      transport=lambda _obs, _instruction: [0.0, 0.0, 0.0, 0.0, -0.5]
    )
    with self.assertRaises(AdapterOutputError) as ctx:
      policy.act(Observation())
    self.assertIn("GRIPPER_UNCHANGED", str(ctx.exception))

  def test_transport_must_be_callable_and_synchronous(self):
    with self.assertRaises(TypeError):
      VLAPolicy(transport="not callable")

    async def async_transport(_observation, _instruction):
      return Action.hold()

    policy = VLAPolicy(transport=async_transport)
    with self.assertRaises(AdapterOutputError) as ctx:
      policy.act(Observation())
    self.assertIn("synchronous", str(ctx.exception))

  def test_units_are_exported_next_to_the_public_vector_fields(self):
    self.assertEqual(BASE_STATE_UNITS, ("mm", "mm", "mm", "rad", "mm", "bool", "bool"))
    self.assertEqual(ACTION_UNITS, ("mm", "mm", "mm", "rad", "mm"))


class GoalTests(unittest.TestCase):
  def test_construction_canonicalizes_numeric_inputs(self):
    goal = Goal(
      pose=[1, np.float32(2), 3.0, np.int64(4)],
      tolerance_mm=np.float32(0.5),
      tolerance_rad=np.float32(0.25),
      rz_weight=np.int64(8),
    )
    self.assertEqual(goal.pose, (1.0, 2.0, 3.0, 4.0))
    self.assertIsInstance(goal.pose, tuple)
    self.assertTrue(all(isinstance(value, float) for value in goal.pose))
    self.assertIsInstance(goal.tolerance_mm, float)
    self.assertIsInstance(goal.tolerance_rad, float)
    self.assertIsInstance(goal.rz_weight, float)

  def test_a_goal_pose_must_be_exactly_four_finite_numbers(self):
    bad_poses = (
      (),
      (1.0, 2.0, 3.0),
      (1.0, 2.0, 3.0, 4.0, 5.0),
      (1.0, 2.0, 3.0, np.nan),
      (1.0, 2.0, np.inf, 4.0),
      (1.0, -np.inf, 3.0, 4.0),
      (1.0, 2.0, 3.0, "not an angle"),
    )
    for pose in bad_poses:
      with self.subTest(pose=pose), self.assertRaises(ValueError) as ctx:
        Goal(pose=pose)
      self.assertIn("4 finite", str(ctx.exception))

  def test_goal_bounds_are_finite_and_non_negative(self):
    for field, bad_values in (
      ("tolerance_mm", (-0.1, np.nan, np.inf)),
      ("rz_weight", (-0.1, np.nan, np.inf)),
      ("tolerance_rad", (-0.1, np.pi + 1e-12, np.nan, np.inf)),
    ):
      for value in bad_values:
        with self.subTest(field=field, value=value), self.assertRaises(ValueError):
          Goal(**{field: value})

  def test_zero_bounds_and_pi_rotation_tolerance_are_valid_endpoints(self):
    exact = Goal(tolerance_mm=0.0, tolerance_rad=0.0, rz_weight=0.0)
    unconstrained = Goal(tolerance_rad=np.pi)
    self.assertEqual(exact.tolerance_mm, 0.0)
    self.assertEqual(exact.tolerance_rad, 0.0)
    self.assertEqual(exact.rz_weight, 0.0)
    self.assertEqual(unconstrained.tolerance_rad, float(np.pi))

  def test_translation_error_for_one_pose_and_for_a_batch(self):
    goal = Goal.at(100.0, 0.0, 0.0)
    single = goal.translation_error([103.0, 4.0, 0.0, 0.0])
    self.assertEqual(np.ndim(single), 0)
    self.assertAlmostEqual(float(single), 5.0, places=9)

    batch = goal.translation_error(
      np.array([[100.0, 0.0, 0.0, 0.0], [103.0, 4.0, 0.0, 1.0], [100.0, 0.0, 12.0, 0.0]])
    )
    self.assertEqual(batch.shape, (3,))
    np.testing.assert_allclose(batch, [0.0, 5.0, 12.0], atol=1e-9)

  def test_translation_error_ignores_rz(self):
    goal = Goal.at(0.0, 0.0, 0.0, rz=0.0)
    self.assertAlmostEqual(float(goal.translation_error([0.0, 0.0, 0.0, 2.5])), 0.0, places=12)

  def test_rz_error_for_one_pose_and_for_a_batch(self):
    goal = Goal.at(0.0, 0.0, 0.0, rz=0.4)
    single = goal.rz_error([0.0, 0.0, 0.0, 0.1])
    self.assertEqual(np.ndim(single), 0)
    self.assertAlmostEqual(float(single), 0.3, places=9)

    batch = goal.rz_error(np.array([[0.0, 0.0, 0.0, 0.4], [0.0, 0.0, 0.0, 0.1]]))
    self.assertEqual(batch.shape, (2,))
    np.testing.assert_allclose(batch, [0.0, 0.3], atol=1e-9)

  def test_rz_error_wraps_across_pi(self):
    goal = Goal.at(0.0, 0.0, 0.0, rz=3.0)
    self.assertAlmostEqual(
      float(goal.rz_error([0.0, 0.0, 0.0, -3.0])), 2.0 * np.pi - 6.0, places=9
    )
    self.assertAlmostEqual(float(goal.rz_error([0.0, 0.0, 0.0, 3.0 + 2.0 * np.pi])), 0.0, places=9)
    spread = goal.rz_error(np.stack([np.zeros(64), np.zeros(64), np.zeros(64),
                                     np.linspace(-20.0, 20.0, 64)], axis=1))
    self.assertLessEqual(float(spread.max()), np.pi + 1e-12)
    self.assertGreaterEqual(float(spread.min()), 0.0)

  def test_cost_folds_rz_in_at_its_weight(self):
    unweighted = Goal.at(0.0, 0.0, 0.0, rz=0.0)
    weighted = Goal.at(0.0, 0.0, 0.0, rz=0.0, rz_weight=10.0)
    pose = [3.0, 4.0, 0.0, 0.5]
    self.assertAlmostEqual(float(unweighted.cost(pose)), 5.0, places=9)
    self.assertAlmostEqual(float(weighted.cost(pose)), 5.0 + 10.0 * 0.5, places=9)

  def test_reached_is_the_translation_tolerance(self):
    goal = Goal.at(100.0, 0.0, 0.0, tolerance_mm=2.0)
    self.assertTrue(goal.reached([100.0, 0.0, 2.0, 0.0]))
    self.assertFalse(goal.reached([100.0, 0.0, 2.001, 0.0]))

  def test_rotation_is_unconstrained_by_default_and_constrainable(self):
    loose = Goal.at(0.0, 0.0, 0.0)
    self.assertTrue(loose.reached([0.0, 0.0, 0.0, np.pi]))
    tight = Goal.at(0.0, 0.0, 0.0, rz=0.0, tolerance_rad=0.1)
    self.assertTrue(tight.reached([0.0, 0.0, 0.0, 0.05]))
    self.assertFalse(tight.reached([0.0, 0.0, 0.0, 0.2]))

  def test_reached_returns_a_bool_not_an_array(self):
    self.assertIsInstance(Goal.at(0.0, 0.0, 0.0).reached([0.0, 0.0, 0.0, 0.0]), bool)


class SamplingPlannerTests(unittest.TestCase):
  def test_construction_refuses_nonsense(self):
    goal = Goal.at(0.0, 0.0, 0.0)
    for kwargs in (
      {"horizon": 0},
      {"horizon": 1.5},
      {"n_samples": 0},
      {"n_samples": 2.5},
      {"step_mm": 0.0},
      {"step_mm": -1.0},
      {"step_mm": np.inf},
      {"step_mm": np.nan},
      {"step_rad": -0.1},
      {"step_rad": np.inf},
      {"step_rad": np.nan},
      {"goal_bias": 1.5},
      {"goal_bias": np.nan},
      {"refine_iters": -1},
      {"refine_iters": 1.5},
      {"action_cost": -0.1},
      {"action_cost": np.inf},
      {"action_cost": np.nan},
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        SamplingPlanner(goal, **kwargs)

  def test_a_rotation_sensitive_goal_requires_a_rotation_step(self):
    for goal in (
      Goal.at(0.0, 0.0, 0.0, rz=0.5, tolerance_rad=0.1),
      Goal.at(0.0, 0.0, 0.0, rz=0.5, rz_weight=10.0),
    ):
      with self.subTest(goal=goal), self.assertRaises(ValueError) as ctx:
        SamplingPlanner(goal, step_rad=0.0)
      self.assertIn("step_rad", str(ctx.exception))

  def test_the_same_seed_gives_the_same_actions(self):
    goal = Goal.at(200.0, -50.0, 40.0)
    poses = [(0.0, 0.0, 100.0, 0.0), (30.0, 10.0, 90.0, 0.2), (120.0, -20.0, 60.0, 0.0)]
    a = SamplingPlanner(goal, n_samples=64, seed=5)
    b = SamplingPlanner(goal, n_samples=64, seed=5)
    first = [a.act(observation_at(p)).pose for p in poses]
    self.assertEqual(first, [b.act(observation_at(p)).pose for p in poses])

    a.reset()
    self.assertEqual(first, [a.act(observation_at(p)).pose for p in poses])
    self.assertIsNone(SamplingPlanner(goal, seed=5).last_predicted_cost)

  def test_a_different_seed_explores_differently(self):
    # goal_bias is off here on purpose. With it on, the aimed candidate is the
    # same vector for every seed and usually wins, so two differently seeded
    # planners agree on the first step and the seed proves nothing.
    goal = Goal.at(200.0, -50.0, 40.0)
    pose = observation_at((0.0, 0.0, 100.0, 0.0))
    a = SamplingPlanner(goal, n_samples=64, goal_bias=0.0, seed=5).act(pose).pose
    b = SamplingPlanner(goal, n_samples=64, goal_bias=0.0, seed=6).act(pose).pose
    self.assertNotEqual(a, b)

  def test_no_sampled_translation_exceeds_step_mm(self):
    # The goal is far away and the model tracks at 40 percent, so every aimed
    # candidate wants a command several times step_mm and must come back clipped.
    planner = SamplingPlanner(
      Goal.at(900.0, 400.0, -200.0),
      dynamics=KinematicDynamics(gain=0.4),
      step_mm=25.0,
      n_samples=512,
      horizon=3,
      goal_bias=0.5,
      refine_iters=3,
      seed=7,
    )
    seqs = planner._sample(np.array([0.0, 0.0, 100.0, 0.0]))
    self.assertEqual(seqs.shape, (512, 3, 4))
    norms = np.linalg.norm(seqs[:, :, :3], axis=2)
    self.assertLessEqual(float(norms.max()), 25.0 * (1.0 + 1e-9))
    self.assertGreater(float(norms.max()), 0.0)

    n_biased = 256
    aimed = np.linalg.norm(seqs[:n_biased, 0, :3], axis=1)
    np.testing.assert_allclose(aimed, 25.0, atol=1e-9)

  def test_sampled_radii_span_orders_of_magnitude(self):
    planner = SamplingPlanner(
      Goal.at(50.0, 0.0, 0.0), step_mm=25.0, n_samples=512, horizon=2, goal_bias=0.0, seed=11
    )
    norms = np.linalg.norm(planner._sample(np.zeros(4))[:, :, :3], axis=2)
    self.assertLess(float(norms.min()), 0.1)
    self.assertGreater(float(norms.max()), 20.0)

  def test_goal_biased_candidates_aim_for_the_whole_horizon(self):
    dynamics = KinematicDynamics(gain=0.8, bias=(0.5, -0.25, 0.1, 0.0))
    goal = Goal.at(55.0, -20.0, 25.0, tolerance_mm=0.1)
    planner = SamplingPlanner(
      goal,
      dynamics=dynamics,
      step_mm=25.0,
      n_samples=20,
      horizon=4,
      goal_bias=0.5,
      refine_iters=30,
      seed=4,
    )
    pose = np.zeros(4)
    seqs = planner._sample(pose)

    # Every biased row is a full directed trajectory. If only the first step
    # were aimed, the remaining three random steps would carry the predicted
    # endpoint back away from the goal and terminal scoring would reject the
    # useful first command for motion the controller never intends to execute.
    for sequence in seqs[:10]:
      predicted = pose.copy()
      for action in sequence:
        predicted = np.asarray(dynamics.predict(predicted, action), float)
      self.assertLess(float(goal.translation_error(predicted)), 1e-6)

  def test_aim_inverts_a_model_whose_gain_is_not_one(self):
    dynamics = KinematicDynamics(gain=0.5)
    goal = Goal.at(10.0, 0.0, 0.0)
    planner = SamplingPlanner(goal, dynamics=dynamics, step_mm=50.0, refine_iters=30, seed=0)
    pose = np.zeros(4)

    # An arm that achieves half of what it is told has to be told twice as far.
    # Each pass removes half the remaining error, so 30 of them are exact to
    # well under a micron; the point is the 20, not the tolerance.
    delta = planner._aim(pose, goal.vector)
    self.assertAlmostEqual(float(delta[0]), 20.0, places=6)
    np.testing.assert_allclose(delta[1:], np.zeros(3), atol=1e-12)
    landed = np.asarray(dynamics.predict(pose, delta), float)
    self.assertLess(float(goal.translation_error(landed)), 1e-6)

  def test_aim_inverts_a_model_with_a_constant_offset(self):
    dynamics = KinematicDynamics(gain=1.0, bias=(3.0, -2.0, 1.0, 0.0))
    goal = Goal.at(40.0, 10.0, -5.0)
    planner = SamplingPlanner(goal, dynamics=dynamics, step_mm=100.0, refine_iters=1, seed=0)
    pose = np.zeros(4)

    landed = np.asarray(dynamics.predict(pose, planner._aim(pose, goal.vector)), float)
    self.assertLess(float(goal.translation_error(landed)), 1e-9)

  def test_refine_iters_zero_aims_geometrically_and_falls_short(self):
    dynamics = KinematicDynamics(gain=0.5)
    goal = Goal.at(10.0, 0.0, 0.0)
    planner = SamplingPlanner(goal, dynamics=dynamics, step_mm=50.0, refine_iters=0, seed=0)
    pose = np.zeros(4)

    delta = planner._aim(pose, goal.vector)
    self.assertAlmostEqual(float(delta[0]), 10.0, places=9)
    landed = np.asarray(dynamics.predict(pose, delta), float)
    self.assertAlmostEqual(float(landed[0]), 5.0, places=9)
    self.assertAlmostEqual(float(goal.translation_error(landed)), 5.0, places=9)

  def test_aim_never_exceeds_step_mm(self):
    planner = SamplingPlanner(
      Goal.at(1000.0, 0.0, 0.0),
      dynamics=KinematicDynamics(gain=0.3),
      step_mm=20.0,
      refine_iters=5,
      seed=0,
    )
    delta = planner._aim(np.zeros(4), planner.goal.vector)
    self.assertLessEqual(float(np.linalg.norm(delta[:3])), 20.0 * (1.0 + 1e-9))

  def test_aim_uses_the_short_rotation_across_pi_and_respects_step_rad(self):
    goal = Goal.at(
      0.0, 0.0, 0.0, rz=-3.0, tolerance_rad=0.01, rz_weight=10.0
    )
    planner = SamplingPlanner(goal, step_rad=0.2, refine_iters=0, seed=0)
    delta = planner._aim(np.array([0.0, 0.0, 0.0, 3.0]), goal.vector)

    # -3 - 3 is -6 radians numerically, but +0.283... is the short physical turn.
    self.assertGreater(float(delta[3]), 0.0)
    self.assertAlmostEqual(float(delta[3]), 0.2, places=12)

  def test_aim_inverts_rotation_gain_and_bias_in_radians(self):
    dynamics = KinematicDynamics(gain=(1.0, 1.0, 1.0, 0.5), bias=(0.0, 0.0, 0.0, 0.05))
    goal = Goal.at(
      0.0, 0.0, 0.0, rz=0.8, tolerance_rad=1e-6, rz_weight=10.0
    )
    planner = SamplingPlanner(
      goal, dynamics=dynamics, step_rad=2.0, refine_iters=40, seed=0
    )
    pose = np.zeros(4)
    landed = np.asarray(dynamics.predict(pose, planner._aim(pose, goal.vector)), float)
    self.assertLess(float(goal.rz_error(landed)), 1e-9)

  def test_planner_solves_a_rotation_only_goal_across_pi(self):
    dynamics = KinematicDynamics()
    goal = Goal.at(
      0.0, 0.0, 100.0, rz=-3.0, tolerance_mm=0.01,
      tolerance_rad=1e-6, rz_weight=10.0,
    )
    planner = SamplingPlanner(
      goal,
      dynamics=dynamics,
      horizon=4,
      n_samples=32,
      step_rad=0.1,
      goal_bias=1.0,
      seed=0,
    )
    pose = np.array([0.0, 0.0, 100.0, 3.0])
    turns = []
    for _ in range(10):
      action = planner.act(observation_at(pose))
      if action.terminate:
        break
      turns.append(float(action.pose[3]))
      pose = np.asarray(dynamics.predict(pose, np.array(action.pose, float)), float)

    self.assertTrue(goal.reached(pose))
    self.assertGreater(len(turns), 0)
    self.assertTrue(all(0.0 < turn <= 0.1 * (1.0 + 1e-12) for turn in turns))

  def test_a_planned_step_survives_an_envelope_set_to_the_same_limit(self):
    # The claim in _sample's docstring, stated against the envelope that would
    # do the rescaling. A single sampled step at step_mm * sqrt(3) would come
    # back with a clamp note here, and every score after it would describe a
    # motion the arm never made.
    dynamics = KinematicDynamics(gain=0.5)
    planner = SamplingPlanner(
      Goal.at(600.0, 200.0, -50.0),
      dynamics=dynamics,
      step_mm=25.0,
      n_samples=128,
      seed=13,
    )
    envelope = SafetyEnvelope(max_step_mm=25.0, workspace=None)
    pose = np.array([0.0, 0.0, 100.0, 0.0])
    for _ in range(20):
      action = planner.act(observation_at(pose))
      safe, notes = envelope.clamp(action, pose)
      self.assertEqual(notes, [])
      # The envelope rebuilds the delta as target minus pose, so a step it left
      # alone comes back changed in the last bits and no further.
      np.testing.assert_allclose(safe.translation, action.translation, atol=1e-12)
      pose = np.asarray(dynamics.predict(pose, np.array(action.pose, float)), float)

  def test_the_search_reports_distinct_candidates_and_not_the_draw_count(self):
    """Regression: the search width reported was the number of rows, not candidates.

    The confidently wrong number this prevents, measured on this repository at
    the shipped defaults: ``n_samples=256`` with ``goal_bias=0.25`` draws 256 rows
    and 64 of them are overwritten with one identical aimed trajectory, so 193
    distinct candidates are searched. The planner reported 256. An argmin does
    not care that a quarter of its input was one row repeated, so nothing
    downstream noticed, and "256 candidates per step" is the sort of figure that
    ends up in a table.
    """
    planner = SamplingPlanner(
      Goal.at(600.0, 200.0, -50.0),
      dynamics=KinematicDynamics(gain=0.5),
      step_mm=25.0,
      n_samples=256,
      horizon=4,
      goal_bias=0.25,
      seed=3,
    )
    pose = np.array([0.0, 0.0, 100.0, 0.0])

    seqs = planner._sample(pose)
    self.assertEqual(seqs.shape, (256, 4, 4))
    self.assertEqual(n_distinct_sequences(seqs), 193)
    self.assertEqual(planner.n_samples, 256)

    # And the number is recorded where a caller reads it, not just computable.
    self.assertIsNone(planner.last_n_distinct)
    planner.plan(pose)
    self.assertEqual(planner.last_n_distinct, 193)
    self.assertLess(planner.last_n_distinct, planner.n_samples)

  def test_the_duplicate_count_is_exactly_the_goal_biased_fraction(self):
    """Stated as arithmetic, so a change in sampling shows up as a changed count."""
    for n_samples, goal_bias in ((256, 0.25), (128, 0.5), (64, 1.0), (100, 0.0)):
      planner = SamplingPlanner(
        Goal.at(300.0, 0.0, 0.0),
        step_mm=25.0,
        n_samples=n_samples,
        horizon=2,
        goal_bias=goal_bias,
        seed=1,
      )
      seqs = planner._sample(np.array([0.0, 0.0, 100.0, 0.0]))
      n_biased = int(round(n_samples * goal_bias))
      expected = n_samples - max(n_biased - 1, 0)
      with self.subTest(n_samples=n_samples, goal_bias=goal_bias):
        self.assertEqual(n_distinct_sequences(seqs), expected)

  def test_reset_clears_the_reported_width_rather_than_leaving_a_stale_one(self):
    planner = SamplingPlanner(Goal.at(300.0, 0.0, 0.0), n_samples=32, seed=2)
    planner.plan(np.array([0.0, 0.0, 100.0, 0.0]))
    self.assertIsNotNone(planner.last_n_distinct)

    planner.reset()
    self.assertIsNone(planner.last_n_distinct)

    # A terminating step runs no search, so it must not leave the previous
    # step's width standing where it reads as this step's.
    goal = Goal.at(120.0, 30.0, 40.0, tolerance_mm=2.0)
    arrived = SamplingPlanner(goal, n_samples=32, seed=1)
    arrived.act(observation_at((0.0, 0.0, 100.0, 0.0)))
    self.assertIsNotNone(arrived.last_n_distinct)
    arrived.act(observation_at((120.5, 30.0, 40.0, 0.0)))
    self.assertIsNone(arrived.last_n_distinct)

  def test_act_terminates_once_the_goal_is_reached(self):
    goal = Goal.at(120.0, 30.0, 40.0, tolerance_mm=2.0)
    planner = SamplingPlanner(goal, n_samples=32, seed=1)
    action = planner.act(observation_at((120.5, 30.0, 40.0, 0.0)))
    self.assertTrue(action.terminate)
    self.assertIs(action.mode, ActionMode.DELTA)
    np.testing.assert_array_equal(action.translation, np.zeros(3))
    self.assertAlmostEqual(float(planner.last_predicted_cost), 0.5, places=9)

  def test_act_away_from_the_goal_moves_and_stays_inside_step_mm(self):
    goal = Goal.at(200.0, 0.0, 40.0, tolerance_mm=2.0)
    planner = SamplingPlanner(goal, step_mm=25.0, n_samples=128, seed=2)
    action = planner.act(observation_at((0.0, 0.0, 100.0, 0.0)))
    self.assertFalse(action.terminate)
    magnitude = float(np.linalg.norm(action.translation))
    self.assertGreater(magnitude, 0.0)
    self.assertLessEqual(magnitude, 25.0 * (1.0 + 1e-9))
    self.assertIsNotNone(planner.last_predicted_cost)

  def test_planning_against_a_mistracking_model_reaches_the_goal(self):
    # The model and the arm share the same gain here, so this scores the search
    # and the inversion, not model error.
    dynamics = KinematicDynamics(gain=0.6)
    goal = Goal.at(180.0, -60.0, 30.0, tolerance_mm=2.0)
    planner = SamplingPlanner(
      goal, dynamics=dynamics, step_mm=25.0, n_samples=256, seed=11
    )
    pose = np.array([0.0, 0.0, 100.0, 0.0])
    for _ in range(120):
      action = planner.act(observation_at(pose))
      if action.terminate:
        break
      pose = np.asarray(dynamics.predict(pose, np.array(action.pose, float)), float)
    self.assertTrue(goal.reached(pose))


if __name__ == "__main__":
  unittest.main()
