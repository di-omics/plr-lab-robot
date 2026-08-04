"""Tests for plr_lr.learn.types and plr_lr.learn.dataset.

These two modules are the only place the state and action vector layouts are
defined. Everything downstream (dynamics fitting, the planner's inversion of a
model, and any external training run reading the exported frames) receives a
bare list of numbers and trusts that the order is the documented one. A silently
reordered or renamed column produces a model that fits noise and reports a good
number while doing it, and nothing further down the pipeline can detect that. So
the ordering claims are asserted directly here: the names and the numbers are
taken from the same observation and compared element by element.

The dataset tests are mostly about refusals. A dataset that accepts an episode
recorded on a differently configured arm mixes two meanings into one
'observation.state' column, and once the files are on disk that cannot be
undone by reading them back.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from plr_lr.learn import (
  ACTION_FIELDS,
  BASE_STATE_FIELDS,
  GRIPPER_UNCHANGED,
  POSE_FIELDS,
  Action,
  ActionMode,
  Dataset,
  DatasetLayoutError,
  Episode,
  Observation,
  Step,
  episodes_from,
)
from plr_lr.learn.dataset import FORMAT_VERSION, FRAME_FIELDS as DATASET_FRAME_FIELDS
from plr_lr.learn.types import ACTION_UNITS, BASE_STATE_UNITS
from plr_lr.vision.detector import Detection

# A two joint layout is enough to prove the ordering and the layout refusal, and
# small enough that expected vectors can be written out in full.
JOINTS_2 = {1: 10.0, 2: -20.0}
JOINTS_3 = {1: 10.0, 2: -20.0, 3: 5.0}

FRAME_FIELDS = {
  "observation.state",
  "action",
  "action.mode",
  "timestamp",
  "frame_index",
  "episode_index",
  "task",
}


def make_observation(t=0.0, x=0.0, joints=None):
  return Observation(t=float(t), x=float(x), joints=dict(joints or {}))


def reject_json_constant(name):
  """Refuse the tokens Python emits but the JSON grammar has no room for.

  json.loads accepts NaN and Infinity by default, so a Python round trip cannot
  tell whether a file is readable by anything else. Passing this as
  ``parse_constant`` makes the reader as strict as JSON.parse or serde_json.
  """
  raise ValueError(f"{name} is not valid JSON")


def make_episode(task="move the tube to the reader", n_steps=3, joints=None, final=True):
  """An episode whose steps are distinguishable: obs.x and the action both count up."""
  ep = Episode(task=task)
  for i in range(n_steps):
    ep.append(
      make_observation(t=float(i), x=float(i), joints=joints),
      Action.delta(dx=float(i + 1)),
    )
  if final:
    ep.final_observation = make_observation(
      t=float(n_steps), x=float(n_steps), joints=joints
    )
  return ep


class StateVectorLayoutTests(unittest.TestCase):
  """state_fields() names exactly what state_vector() contains, in that order."""

  def test_state_vector_matches_the_documented_layout(self):
    obs = Observation(
      t=7.0,
      x=1.0,
      y=2.0,
      z=3.0,
      rz=0.25,
      gripper_width=40.0,
      gripper_closed=True,
      holding=True,
      joints=JOINTS_2,
    )
    np.testing.assert_allclose(
      obs.state_vector(),
      [1.0, 2.0, 3.0, 0.25, 40.0, 1.0, 1.0, 10.0, -20.0],
    )
    self.assertEqual(
      obs.state_fields(),
      ("x", "y", "z", "rz", "gripper_width", "gripper_closed", "holding", "j1", "j2"),
    )

  def test_time_is_not_part_of_the_state_vector(self):
    """t is metadata for the recorder, not something a policy conditions on."""
    a = Observation(t=0.0, x=5.0, joints=JOINTS_2)
    b = Observation(t=99.0, x=5.0, joints=JOINTS_2)
    self.assertNotIn("t", a.state_fields())
    np.testing.assert_allclose(a.state_vector(), b.state_vector())

  def test_names_and_numbers_stay_in_lockstep_for_every_joint_count(self):
    for joints in ({}, {0: 1.0}, JOINTS_2, {i: float(i) for i in range(1, 8)}):
      obs = Observation(x=1.0, joints=joints)
      fields = obs.state_fields()
      vector = obs.state_vector()
      self.assertEqual(len(fields), len(vector), f"layout desync for {joints}")
      self.assertEqual(len(fields), len(BASE_STATE_FIELDS) + len(joints))
      self.assertEqual(len(set(fields)), len(fields), "duplicate field name")
      named = dict(zip(fields, vector))
      for index, value in joints.items():
        self.assertAlmostEqual(named[f"j{index}"], value)

  def test_state_units_stay_aligned_with_every_state_layout(self):
    for joints in ({}, JOINTS_2, {i: float(i) for i in range(1, 8)}):
      obs = Observation(joints=joints)
      self.assertEqual(len(obs.state_units()), len(obs.state_fields()))
      self.assertEqual(obs.state_units()[: len(BASE_STATE_UNITS)], BASE_STATE_UNITS)
      self.assertEqual(obs.state_units()[len(BASE_STATE_UNITS) :], ("deg",) * len(joints))

  def test_joint_order_is_by_index_not_by_insertion(self):
    """Recorders build the dict in whatever order the driver reports joints."""
    scrambled = {}
    for index in (3, 1, 2):
      scrambled[index] = float(index * 10)
    obs = Observation(joints=scrambled)
    self.assertEqual(obs.joint_indices(), (1, 2, 3))
    self.assertEqual(obs.state_fields()[-3:], ("j1", "j2", "j3"))
    np.testing.assert_allclose(obs.state_vector()[-3:], [10.0, 20.0, 30.0])

  def test_the_head_of_the_layout_is_the_shared_constant(self):
    obs = Observation(joints=JOINTS_2)
    self.assertEqual(obs.state_fields()[: len(BASE_STATE_FIELDS)], BASE_STATE_FIELDS)
    self.assertEqual(BASE_STATE_FIELDS[: len(POSE_FIELDS)], POSE_FIELDS)

  def test_an_unmeasured_gripper_width_is_not_a_number(self):
    """0.0 in that column would be a plausible value for "we never measured it"."""
    unknown = Observation(gripper_width=None, joints=JOINTS_2)
    self.assertTrue(np.isnan(unknown.state_vector()[4]))
    known = Observation(gripper_width=0.0, joints=JOINTS_2)
    self.assertEqual(known.state_vector()[4], 0.0)
    self.assertFalse(np.isnan(known.state_vector()[4]))
    self.assertTrue(np.isfinite(np.delete(unknown.state_vector(), 4)).all())

  def test_booleans_are_carried_as_numbers(self):
    closed = Observation(gripper_closed=True, holding=False).state_vector()
    open_ = Observation(gripper_closed=False, holding=True).state_vector()
    self.assertEqual(closed[5], 1.0)
    self.assertEqual(closed[6], 0.0)
    self.assertEqual(open_[5], 0.0)
    self.assertEqual(open_[6], 1.0)
    self.assertEqual(closed.dtype, np.dtype(float))

  def test_pose_vector_is_the_first_four_state_elements(self):
    obs = Observation(x=1.5, y=-2.5, z=3.5, rz=0.75, joints=JOINTS_2)
    np.testing.assert_allclose(obs.pose_vector(), [1.5, -2.5, 3.5, 0.75])
    np.testing.assert_allclose(obs.pose_vector(), obs.state_vector()[: len(POSE_FIELDS)])


class ActionVectorTests(unittest.TestCase):
  """The action layout, and the sentinel that means 'do not touch the gripper'."""

  def test_action_vector_matches_action_fields(self):
    action = Action.absolute(10.0, 20.0, 30.0, 0.5, gripper=75.0)
    vector = action.action_vector()
    self.assertEqual(len(vector), len(ACTION_FIELDS))
    self.assertEqual(ACTION_FIELDS, ("p0", "p1", "p2", "p3", "gripper"))
    np.testing.assert_allclose(vector, [10.0, 20.0, 30.0, 0.5, 75.0])

  def test_an_absent_gripper_command_becomes_the_sentinel(self):
    np.testing.assert_allclose(
      Action.delta(1.0, 2.0, 3.0, 4.0).action_vector(),
      [1.0, 2.0, 3.0, 4.0, GRIPPER_UNCHANGED],
    )

  def test_a_zero_width_command_is_not_the_sentinel(self):
    """Fully closed is a real command, and the trap a falsy check would fall into."""
    self.assertLess(GRIPPER_UNCHANGED, 0.0)
    vector = Action.delta(gripper=0.0).action_vector()
    self.assertEqual(vector[4], 0.0)
    self.assertNotEqual(vector[4], GRIPPER_UNCHANGED)

  def test_the_same_slots_hold_the_pose_in_both_cartesian_modes(self):
    pose = (5.0, -6.0, 7.0, 0.125)
    delta = Action.delta(*pose).action_vector()[:4]
    absolute = Action.absolute(*pose).action_vector()[:4]
    np.testing.assert_allclose(delta, pose)
    np.testing.assert_allclose(absolute, pose)

  def test_hold_commands_nothing(self):
    action = Action.hold()
    self.assertIs(action.mode, ActionMode.DELTA)
    self.assertFalse(action.terminate)
    np.testing.assert_allclose(
      action.action_vector(), [0.0, 0.0, 0.0, 0.0, GRIPPER_UNCHANGED]
    )

  def test_translation_is_the_first_three_pose_numbers(self):
    np.testing.assert_allclose(
      Action.absolute(1.0, 2.0, 3.0, 9.0).translation, [1.0, 2.0, 3.0]
    )
    np.testing.assert_allclose(Action.delta(1.0, 2.0, 3.0, 9.0).translation, [1.0, 2.0, 3.0])

  def test_joint_targets_carry_no_pose(self):
    action = Action.joint_targets({1: 10.0, 2: 20.0}, gripper=30.0)
    self.assertIs(action.mode, ActionMode.JOINTS)
    np.testing.assert_allclose(action.action_vector(), [0.0, 0.0, 0.0, 0.0, 30.0])

  def test_joint_targets_copies_the_dict_it_was_given(self):
    source = {1: 10.0}
    action = Action.joint_targets(source)
    source[1] = 999.0
    self.assertEqual(action.joints, {1: 10.0})


class AsDeltaTests(unittest.TestCase):
  """A displacement is only meaningful once the mode says how to read the pose."""

  def test_a_delta_action_is_already_a_displacement(self):
    action = Action.delta(1.0, -2.0, 3.0, 0.5)
    for current in ([0.0, 0.0, 0.0, 0.0], [100.0, -50.0, 25.0, 1.5]):
      np.testing.assert_allclose(action.as_delta(current), [1.0, -2.0, 3.0, 0.5])

  def test_an_absolute_action_is_the_difference_from_here(self):
    action = Action.absolute(10.0, 20.0, 30.0, 0.5)
    np.testing.assert_allclose(action.as_delta([1.0, 2.0, 3.0, 0.25]), [9.0, 18.0, 27.0, 0.25])
    np.testing.assert_allclose(action.as_delta([10.0, 20.0, 30.0, 0.5]), [0.0, 0.0, 0.0, 0.0])

  def test_an_absolute_action_accepts_any_pose_sequence(self):
    action = Action.absolute(4.0, 4.0, 4.0, 4.0)
    from_tuple = action.as_delta((1.0, 1.0, 1.0, 1.0))
    from_array = action.as_delta(np.array([1.0, 1.0, 1.0, 1.0]))
    np.testing.assert_allclose(from_tuple, from_array)
    np.testing.assert_allclose(from_tuple, [3.0, 3.0, 3.0, 3.0])

  def test_an_absolute_rotation_uses_the_short_path_across_pi(self):
    action = Action.absolute(0.0, 0.0, 0.0, -3.0)
    delta = action.as_delta([0.0, 0.0, 0.0, 3.0])
    self.assertGreater(float(delta[3]), 0.0)
    self.assertAlmostEqual(float(delta[3]), 2.0 * np.pi - 6.0, places=12)

  def test_a_joints_action_refuses_instead_of_reporting_no_motion(self):
    action = Action.joint_targets({1: 45.0})
    with self.assertRaises(ValueError) as ctx:
      action.as_delta([0.0, 0.0, 0.0, 0.0])
    message = str(ctx.exception)
    self.assertIn("JOINTS", message)
    self.assertIn("kinematics", message)

  def test_as_delta_does_not_alias_the_stored_pose(self):
    action = Action.delta(1.0, 2.0, 3.0, 4.0)
    returned = action.as_delta([0.0, 0.0, 0.0, 0.0])
    returned[0] = 999.0
    self.assertEqual(action.pose[0], 1.0)


class WithPoseTests(unittest.TestCase):
  """The safety envelope clamps a pose and must not lose the rest of the action."""

  def test_only_the_pose_changes(self):
    original = Action(
      mode=ActionMode.ABSOLUTE,
      pose=(1.0, 2.0, 3.0, 4.0),
      gripper=12.5,
      joints={1: 45.0},
      terminate=True,
    )
    clamped = original.with_pose([9.0, 8.0, 7.0, 6.0])
    self.assertEqual(clamped.pose, (9.0, 8.0, 7.0, 6.0))
    self.assertIs(clamped.mode, ActionMode.ABSOLUTE)
    self.assertEqual(clamped.gripper, 12.5)
    self.assertEqual(clamped.joints, {1: 45.0})
    self.assertTrue(clamped.terminate)

  def test_the_original_is_left_alone(self):
    original = Action.delta(1.0, 2.0, 3.0, 4.0, gripper=None)
    clamped = original.with_pose([0.0, 0.0, 0.0, 0.0])
    self.assertEqual(original.pose, (1.0, 2.0, 3.0, 4.0))
    self.assertIsNot(clamped, original)
    self.assertIsNone(clamped.gripper)

  def test_the_pose_is_coerced_to_plain_floats(self):
    clamped = Action.delta().with_pose(np.array([1, 2, 3, 4]))
    self.assertEqual(clamped.pose, (1.0, 2.0, 3.0, 4.0))
    for value in clamped.pose:
      self.assertIsInstance(value, float)
      self.assertNotIsInstance(value, np.ndarray)

  def test_a_pose_that_is_not_four_long_is_refused(self):
    action = Action.delta()
    for pose in ([1.0, 2.0, 3.0], [1.0, 2.0, 3.0, 4.0, 5.0], []):
      with self.assertRaises(ValueError) as ctx:
        action.with_pose(pose)
      self.assertIn("4 elements", str(ctx.exception))


class TransitionTests(unittest.TestCase):
  """A transition needs an outcome. A step without one is dropped, not guessed."""

  def test_the_last_step_is_dropped_when_its_outcome_was_never_observed(self):
    ep = make_episode(n_steps=3, final=False)
    transitions = ep.transitions()
    self.assertEqual(len(transitions), 2)
    self.assertEqual(len(ep.observations()), 3)

  def test_a_final_observation_completes_the_last_transition(self):
    ep = make_episode(n_steps=3, final=True)
    self.assertEqual(len(ep.transitions()), 3)
    self.assertEqual(len(ep.observations()), 4)

  def test_each_action_is_paired_with_the_state_it_was_chosen_in(self):
    ep = make_episode(n_steps=3, final=True)
    for i, (obs, action, next_obs) in enumerate(ep.transitions()):
      self.assertEqual(obs.x, float(i))
      self.assertEqual(next_obs.x, float(i + 1))
      self.assertEqual(action.pose[0], float(i + 1))
      self.assertIs(action, ep.steps[i].action)

  def test_a_lone_step_with_no_outcome_yields_nothing(self):
    ep = make_episode(n_steps=1, final=False)
    self.assertEqual(len(ep), 1)
    self.assertEqual(ep.transitions(), [])

  def test_an_empty_episode_has_no_transitions(self):
    ep = Episode(task="nothing happened")
    self.assertEqual(len(ep), 0)
    self.assertEqual(ep.transitions(), [])
    self.assertEqual(ep.observations(), [])

  def test_iteration_and_length_walk_the_steps(self):
    ep = make_episode(n_steps=4, final=True)
    self.assertEqual(len(ep), 4)
    self.assertEqual([s.observation.x for s in ep], [0.0, 1.0, 2.0, 3.0])

  def test_append_returns_the_step_it_stored_with_its_info(self):
    ep = Episode()
    obs = make_observation(t=1.0, x=2.0)
    action = Action.hold()
    step = ep.append(obs, action, source="scripted", trial=3)
    self.assertIs(ep.steps[-1], step)
    self.assertIsInstance(step, Step)
    self.assertEqual(step.info, {"source": "scripted", "trial": 3})

  def test_duration_spans_the_first_observation_to_the_last(self):
    ep = Episode()
    ep.append(make_observation(t=2.0), Action.hold())
    ep.append(make_observation(t=3.0), Action.hold())
    ep.final_observation = make_observation(t=6.5)
    self.assertAlmostEqual(ep.duration, 4.5)


class FrameExportTests(unittest.TestCase):
  """The flat per-frame schema an external training loader reads."""

  def test_frames_have_exactly_the_documented_field_names(self):
    ep = make_episode(n_steps=2, joints=JOINTS_2)
    for frame in ep.frames():
      self.assertEqual(set(frame), FRAME_FIELDS)

  def test_there_is_one_frame_per_step_not_per_observation(self):
    ep = make_episode(n_steps=3, final=True)
    self.assertEqual(len(ep.observations()), 4)
    self.assertEqual(len(ep.frames()), 3)

  def test_frame_contents_come_from_the_vector_layouts(self):
    ep = make_episode(n_steps=2, joints=JOINTS_2)
    ep.episode_index = 7
    for i, frame in enumerate(ep.frames()):
      step = ep.steps[i]
      self.assertIsInstance(frame["observation.state"], list)
      self.assertIsInstance(frame["action"], list)
      np.testing.assert_allclose(frame["observation.state"], step.observation.state_vector())
      np.testing.assert_allclose(frame["action"], step.action.action_vector())
      self.assertEqual(frame["action.mode"], step.action.mode.value)
      self.assertEqual(frame["timestamp"], step.observation.t)
      self.assertEqual(frame["frame_index"], i)
      self.assertEqual(frame["episode_index"], 7)
      self.assertEqual(frame["task"], ep.task)

  def test_frames_survive_json(self):
    ep = Episode(task="pick")
    ep.append(
      Observation(t=0.5, x=1.0, gripper_width=42.5, joints=JOINTS_2),
      Action.absolute(1.0, 2.0, 3.0, 0.25, gripper=30.0),
    )
    self.assertEqual(json.loads(json.dumps(ep.frames())), ep.frames())


class RoundTripTests(unittest.TestCase):
  """Whatever a demonstration recorded must survive a year on disk unchanged."""

  def full_observation(self):
    return Observation(
      t=1.25,
      x=100.5,
      y=-20.25,
      z=15.0,
      rz=0.75,
      gripper_width=42.5,
      gripper_closed=True,
      holding=True,
      joints={2: -20.0, 1: 10.0},
      detections=(
        Detection(label="tube", uv=(640.5, 512.25), confidence=0.87, truth_name="tube_a1"),
        Detection(label="plate", uv=(10.0, 20.0)),
      ),
      image_path="frames/000123.png",
    )

  def test_observation_survives_json(self):
    obs = self.full_observation()
    back = Observation.from_dict(json.loads(json.dumps(obs.to_dict())))
    self.assertEqual(back, obs)

  def test_observation_detections_come_back_as_detections(self):
    obs = self.full_observation()
    back = Observation.from_dict(json.loads(json.dumps(obs.to_dict())))
    self.assertIsInstance(back.detections, tuple)
    self.assertEqual(len(back.detections), 2)
    first = back.detections[0]
    self.assertIsInstance(first, Detection)
    self.assertEqual(first.label, "tube")
    self.assertEqual(first.uv, (640.5, 512.25))
    self.assertAlmostEqual(first.confidence, 0.87)
    self.assertEqual(first.truth_name, "tube_a1")
    self.assertEqual(back.detections[1].confidence, 1.0)
    self.assertIsNone(back.detections[1].truth_name)

  def test_joint_keys_come_back_as_integers_not_strings(self):
    """JSON has no integer keys, so the layout would silently reorder as text."""
    obs = Observation(joints={2: -20.0, 10: 1.0, 1: 10.0})
    encoded = json.loads(json.dumps(obs.to_dict()))
    self.assertEqual(set(encoded["joints"]), {"1", "2", "10"})
    back = Observation.from_dict(encoded)
    self.assertEqual(sorted(back.joints), [1, 2, 10])
    self.assertEqual(back.joint_indices(), (1, 2, 10))
    self.assertEqual(back.state_fields()[-3:], ("j1", "j2", "j10"))

  def test_an_unmeasured_gripper_width_stays_unmeasured(self):
    """None and 0.0 mean different things, and JSON keeps them apart."""
    encoded = json.loads(json.dumps(Observation(gripper_width=None).to_dict()))
    self.assertIsNone(encoded["gripper_width"])
    self.assertIsNone(Observation.from_dict(encoded).gripper_width)

    encoded_zero = json.loads(json.dumps(Observation(gripper_width=0.0).to_dict()))
    self.assertEqual(encoded_zero["gripper_width"], 0.0)
    back = Observation.from_dict(encoded_zero)
    self.assertIsNotNone(back.gripper_width)
    self.assertEqual(back.gripper_width, 0.0)

  def test_an_empty_observation_survives(self):
    obs = Observation()
    back = Observation.from_dict(json.loads(json.dumps(obs.to_dict())))
    self.assertEqual(back, obs)
    self.assertEqual(back.detections, ())
    self.assertIsNone(back.image_path)

  def test_every_action_mode_survives_json(self):
    actions = [
      Action.delta(1.0, -2.0, 3.0, 0.5, gripper=None),
      Action.absolute(10.0, 20.0, 30.0, 0.25, gripper=60.0, terminate=True),
      Action.joint_targets({1: 45.0, 2: -10.0}, gripper=0.0),
      Action.hold(terminate=True),
    ]
    for action in actions:
      back = Action.from_dict(json.loads(json.dumps(action.to_dict())))
      self.assertEqual(back, action)
      self.assertIs(back.mode, action.mode)
      np.testing.assert_allclose(back.action_vector(), action.action_vector())

  def test_a_zero_gripper_width_does_not_come_back_as_unspecified(self):
    back = Action.from_dict(json.loads(json.dumps(Action.delta(gripper=0.0).to_dict())))
    self.assertEqual(back.gripper, 0.0)
    self.assertIsNotNone(back.gripper)
    self.assertEqual(back.action_vector()[4], 0.0)

  def test_an_unspecified_gripper_stays_none(self):
    back = Action.from_dict(json.loads(json.dumps(Action.delta().to_dict())))
    self.assertIsNone(back.gripper)
    self.assertEqual(back.action_vector()[4], GRIPPER_UNCHANGED)

  def test_joint_targets_keys_come_back_as_integers(self):
    action = Action.joint_targets({1: 45.0, 2: -10.0})
    back = Action.from_dict(json.loads(json.dumps(action.to_dict())))
    self.assertEqual(back.joints, {1: 45.0, 2: -10.0})

  def test_step_survives_json_with_its_info(self):
    step = Step(
      observation=self.full_observation(),
      action=Action.absolute(1.0, 2.0, 3.0, 0.5, gripper=30.0),
      info={"source": "kinesthetic", "clamped": True, "trial": 2},
    )
    back = Step.from_dict(json.loads(json.dumps(step.to_dict())))
    self.assertEqual(back, step)
    self.assertEqual(back.info["clamped"], True)

  def test_episode_survives_json(self):
    ep = make_episode(task="decap the tube", n_steps=3, joints=JOINTS_2, final=True)
    ep.metadata = {"arm": "xarm6", "simulated": True, "hz": 10}
    ep.episode_index = 4
    ep.steps[0].info["note"] = "first"
    back = Episode.from_dict(json.loads(json.dumps(ep.to_dict())))
    self.assertEqual(back, ep)
    self.assertEqual(back.task, "decap the tube")
    self.assertEqual(back.episode_index, 4)
    self.assertEqual(back.metadata, {"arm": "xarm6", "simulated": True, "hz": 10})
    self.assertEqual(len(back.transitions()), 3)

  def test_an_episode_without_a_final_observation_survives(self):
    ep = make_episode(n_steps=2, final=False)
    encoded = json.loads(json.dumps(ep.to_dict()))
    self.assertIsNone(encoded["final_observation"])
    back = Episode.from_dict(encoded)
    self.assertEqual(back, ep)
    self.assertIsNone(back.final_observation)
    self.assertEqual(len(back.transitions()), 1)

  def test_an_empty_episode_survives(self):
    back = Episode.from_dict(json.loads(json.dumps(Episode().to_dict())))
    self.assertEqual(back, Episode())


class DatasetTests(unittest.TestCase):
  """A directory of episodes, and the layout mismatch it must refuse."""

  def setUp(self):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.root = Path(tmp.name) / "demos"
    self.ds = Dataset(self.root)

  def episode_files(self):
    return sorted(p.name for p in self.ds.episode_dir.glob("episode_*.json"))

  def test_an_unwritten_dataset_reads_as_empty(self):
    self.assertEqual(self.ds.meta(), {})
    self.assertEqual(len(self.ds), 0)
    self.assertEqual(list(self.ds.episodes()), [])
    self.assertEqual(self.ds.next_index(), 0)
    self.assertFalse(self.root.exists())

  def test_add_assigns_dense_indices_matching_the_filenames(self):
    for expected in range(3):
      ep = make_episode(task=f"task {expected}", n_steps=2, joints=JOINTS_2)
      ep.episode_index = 99  # whatever the caller set is not trusted
      path = self.ds.add(ep)
      self.assertEqual(ep.episode_index, expected)
      self.assertEqual(path.name, f"episode_{expected:06d}.json")
      self.assertEqual(self.ds.next_index(), expected + 1)
    self.assertEqual(
      self.episode_files(),
      ["episode_000000.json", "episode_000001.json", "episode_000002.json"],
    )
    self.assertEqual(len(self.ds), 3)

  def test_the_written_index_is_the_one_on_disk(self):
    self.ds.add(make_episode(n_steps=1, joints=JOINTS_2))
    path = self.ds.add(make_episode(n_steps=1, joints=JOINTS_2))
    self.assertEqual(path.parent, self.root / "episodes")
    self.assertEqual(json.loads(path.read_text())["episode_index"], 1)

  def test_a_gap_never_reuses_or_overwrites_an_existing_episode(self):
    paths = [self.ds.add(make_episode(task=f"task {i}", joints=JOINTS_2)) for i in range(3)]
    preserved = paths[2].read_text()
    paths[1].unlink()

    added = self.ds.add(make_episode(task="after the gap", joints=JOINTS_2))

    self.assertEqual(added.name, "episode_000003.json")
    self.assertEqual(paths[2].read_text(), preserved)
    self.assertEqual(
      self.episode_files(),
      ["episode_000000.json", "episode_000002.json", "episode_000003.json"],
    )
    self.assertEqual(self.ds.meta()["n_episodes"], 3)
    self.assertEqual(len(self.ds), 3)

  def test_meta_records_the_layouts_and_the_counts(self):
    self.ds.add(make_episode(task="pick", n_steps=2, joints=JOINTS_2))
    self.ds.add(make_episode(task="place", n_steps=2, joints=JOINTS_2))
    self.ds.add(make_episode(task="pick", n_steps=2, joints=JOINTS_2))
    meta = self.ds.meta()
    self.assertEqual(meta["format_version"], FORMAT_VERSION)
    self.assertEqual(meta["action_fields"], list(ACTION_FIELDS))
    self.assertEqual(meta["action_units"], list(ACTION_UNITS))
    self.assertEqual(
      meta["state_fields"],
      ["x", "y", "z", "rz", "gripper_width", "gripper_closed", "holding", "j1", "j2"],
    )
    self.assertEqual(
      meta["state_units"], ["mm", "mm", "mm", "rad", "mm", "bool", "bool", "deg", "deg"]
    )
    self.assertEqual(meta["timestamp_unit"], "s")
    self.assertEqual(meta["schema"]["name"], "plr-lr-learning-frames")
    self.assertEqual(meta["schema"]["kind"], "project-native-interchange")
    self.assertEqual(meta["schema"]["encoding"], "jsonl")
    self.assertEqual(meta["schema"]["fields"], list(DATASET_FRAME_FIELDS))
    self.assertEqual(meta["schema"]["action_modes"]["delta"]["p3"], "rotation_delta")
    self.assertEqual(meta["schema"]["action_modes"]["absolute"]["p3"], "rotation")
    self.assertEqual(meta["n_episodes"], 3)
    self.assertEqual(meta["tasks"], ["pick", "place"])
    self.assertEqual(self.ds.meta_path, self.root / "meta.json")

  def test_meta_state_fields_name_the_exported_state_columns(self):
    ep = make_episode(n_steps=1, joints=JOINTS_2)
    self.ds.add(ep)
    frame = next(iter(self.ds.frames()))
    self.assertEqual(len(frame["observation.state"]), len(self.ds.meta()["state_fields"]))
    self.assertEqual(len(frame["action"]), len(self.ds.meta()["action_fields"]))

  def test_incompatible_format_or_unit_metadata_is_not_silently_reinterpreted(self):
    self.ds.add(make_episode(n_steps=1, joints=JOINTS_2))
    original = self.ds.meta()

    for key, bad_value in (("format_version", 999), ("action_units", ["mm"] * 5)):
      with self.subTest(key=key):
        tampered = dict(original)
        tampered[key] = bad_value
        self.ds.meta_path.write_text(json.dumps(tampered))
        with self.assertRaises(DatasetLayoutError) as ctx:
          self.ds.add(make_episode(n_steps=1, joints=JOINTS_2))
        self.assertIn(key, str(ctx.exception))
        self.assertEqual(self.episode_files(), ["episode_000000.json"])

  def test_episodes_and_load_round_trip_through_disk(self):
    written = [
      make_episode(task="pick", n_steps=2, joints=JOINTS_2),
      make_episode(task="place", n_steps=3, joints=JOINTS_2),
    ]
    for ep in written:
      self.ds.add(ep)
    read_back = list(self.ds.episodes())
    self.assertEqual(read_back, written)
    self.assertEqual([e.episode_index for e in read_back], [0, 1])
    self.assertEqual(self.ds.load(1), written[1])
    self.assertEqual(self.ds.load(1).task, "place")

  def test_loading_a_missing_index_says_so(self):
    self.ds.add(make_episode(n_steps=1, joints=JOINTS_2))
    with self.assertRaises(FileNotFoundError):
      self.ds.load(7)

  def test_transitions_span_every_episode(self):
    self.ds.add(make_episode(n_steps=2, joints=JOINTS_2, final=True))
    self.ds.add(make_episode(n_steps=3, joints=JOINTS_2, final=False))
    self.assertEqual(len(self.ds.transitions()), 2 + 2)

  def test_export_frames_writes_jsonl_with_a_global_index(self):
    self.ds.add(make_episode(task="pick", n_steps=2, joints=JOINTS_2))
    self.ds.add(make_episode(task="place", n_steps=3, joints=JOINTS_2))
    out = self.ds.export_frames(self.root / "frames.jsonl")
    lines = out.read_text().splitlines()
    self.assertEqual(len(lines), 5)
    records = [json.loads(line) for line in lines]
    self.assertEqual([r["index"] for r in records], [0, 1, 2, 3, 4])
    self.assertEqual([r["episode_index"] for r in records], [0, 0, 1, 1, 1])
    self.assertEqual([r["frame_index"] for r in records], [0, 1, 0, 1, 2])
    self.assertEqual([r["task"] for r in records], ["pick"] * 2 + ["place"] * 3)
    for record in records:
      self.assertEqual(set(record), FRAME_FIELDS | {"index"})
      self.assertEqual(record["action.mode"], "delta")

  def test_exported_frames_are_readable_by_a_strict_json_reader(self):
    # The gripper width was never measured, which is what an xArm reports before
    # its gripper has been commanded.
    self.ds.add(make_episode(n_steps=2, joints=JOINTS_2))
    out = self.ds.export_frames()
    for line in out.read_text().splitlines():
      frame = json.loads(line, parse_constant=reject_json_constant)
      self.assertIsNone(frame["observation.state"][4])
      self.assertEqual(frame["action.mode"], "delta")

  def test_export_frames_defaults_into_the_dataset_root(self):
    self.ds.add(make_episode(n_steps=1, joints=JOINTS_2))
    out = self.ds.export_frames()
    self.assertEqual(out, self.root / "frames.jsonl")
    self.assertEqual(len(out.read_text().splitlines()), 1)
    # The export must not be picked up as an episode on the next read.
    self.assertEqual(len(list(self.ds.episodes())), 1)

  def test_a_different_joint_count_is_refused(self):
    self.ds.add(make_episode(task="two joints", n_steps=2, joints=JOINTS_2))
    other_arm = make_episode(task="three joints", n_steps=2, joints=JOINTS_3)
    with self.assertRaises(DatasetLayoutError) as ctx:
      self.ds.add(other_arm)
    self.assertIn("j3", str(ctx.exception))

  def test_a_refused_episode_leaves_nothing_behind(self):
    self.ds.add(make_episode(task="two joints", n_steps=2, joints=JOINTS_2))
    before = self.episode_files()
    with self.assertRaises(DatasetLayoutError):
      self.ds.add(make_episode(task="three joints", n_steps=2, joints=JOINTS_3))
    self.assertEqual(self.episode_files(), before)
    self.assertEqual(len(self.ds), 1)
    self.assertEqual(self.ds.meta()["tasks"], ["two joints"])
    self.assertEqual(self.ds.next_index(), 1)

  def test_a_missing_joint_is_a_layout_change_too(self):
    self.ds.add(make_episode(n_steps=2, joints=JOINTS_2))
    with self.assertRaises(DatasetLayoutError):
      self.ds.add(make_episode(n_steps=2, joints={1: 10.0}))

  def test_a_joint_layout_change_inside_one_episode_is_refused(self):
    episode = make_episode(n_steps=2, joints=JOINTS_2)
    episode.steps[1] = Step(
      observation=make_observation(t=1.0, x=1.0, joints=JOINTS_3),
      action=episode.steps[1].action,
    )

    with self.assertRaises(DatasetLayoutError) as ctx:
      self.ds.add(episode)

    self.assertIn("observation 1 state layout", str(ctx.exception))
    self.assertIn("j3", str(ctx.exception))
    self.assertEqual(self.episode_files(), [])
    self.assertEqual(self.ds.meta(), {})

  def test_an_episode_with_no_observations_is_refused(self):
    with self.assertRaises(DatasetLayoutError):
      self.ds.add(Episode(task="nothing recorded"))
    self.assertEqual(self.ds.meta(), {})

  def test_a_matching_layout_is_accepted_whatever_the_joint_values(self):
    self.ds.add(make_episode(n_steps=2, joints={1: 0.0, 2: 0.0}))
    self.ds.add(make_episode(n_steps=2, joints={1: 90.0, 2: -45.0}))
    self.assertEqual(len(self.ds), 2)


class EpisodesFromTests(unittest.TestCase):
  """Fitting code takes a dataset or a list, and must not care which."""

  def setUp(self):
    tmp = tempfile.TemporaryDirectory()
    self.addCleanup(tmp.cleanup)
    self.root = Path(tmp.name) / "demos"

  def test_a_dataset_is_read_off_disk(self):
    ds = Dataset(self.root)
    ep = make_episode(task="pick", n_steps=2, joints=JOINTS_2)
    ds.add(ep)
    out = episodes_from(ds)
    self.assertIsInstance(out, list)
    self.assertEqual(out, [ep])

  def test_a_sequence_is_copied_not_aliased(self):
    episodes = [make_episode(n_steps=1), make_episode(n_steps=2)]
    out = episodes_from(episodes)
    self.assertEqual(out, episodes)
    self.assertIsNot(out, episodes)


if __name__ == "__main__":
  unittest.main()
