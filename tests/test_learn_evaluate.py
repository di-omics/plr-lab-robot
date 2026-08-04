"""Tests for plr_lr.learn.evaluate and plr_lr.learn.synthetic.

Organised around the claims the evaluation harness makes, because those claims
are the only reason to believe a number that comes out of it. The claims are:
that a trial starts where the task says it starts, and starts there again on a
rerun; that success is read off the final state and not off the best frame the
run happened to pass through; that a criterion a null control can satisfy is
reported as broken rather than scored; that n copies of one identical run are
refused rather than reported as n trials; that the aggregates say "no successes"
instead of averaging an empty set; and that a dynamics model is only credited
with an improvement it actually made.

:class:`MistrackingBackend` is tested against its own planted parameters, since
every recovery test downstream is scored against them, and against the phase-1
backend contract it inherits, since an arm that stopped refusing unhomed motion
would still look fine to every test that only checks where it ended up.
"""

from __future__ import annotations

import math
import unittest

import numpy as np
from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.precise_flex.coords import ElbowOrientation, PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from plr_lr import ArmNotHomedError, Pose2D, Workspace, WorkspaceViolation, site_coords
from plr_lr.learn import (
  Action,
  Episode,
  EvalReport,
  Goal,
  HardwarePermissionRequired,
  KinematicDynamics,
  MistrackingBackend,
  NoOpPolicy,
  Observation,
  ReachTask,
  ReplayPolicy,
  SafetyEnvelope,
  SamplingPlanner,
  TrialResult,
  evaluate,
  evaluate_dynamics,
  null_test,
  random_walk_actions,
  rollout,
  simulate_episode,
)
from plr_lr.learn.evaluate import (
  DEFAULT_START_JITTER_MM,
  MINIMUM_TRIALS,
  action_digest,
)
from tests._doubles import DegreeReportingBackend

START = (0.0, 0.0, 100.0, 0.0)


def cartesian(x: float, y: float, z: float, rz: float = 0.0) -> PreciseFlexCartesianCoords:
  """Build PyLabRobot coordinates from a learning-native rotation in radians."""
  return PreciseFlexCartesianCoords(
    location=Coordinate(x, y, z),
    rotation=Rotation(0.0, 0.0, math.degrees(rz)),
    orientation=ElbowOrientation.RIGHT,
  )


async def pose_of(arm: ExperimentalSCARA):
  """The tool pose as an Observation would read it, as (x, y, z, rz)."""
  c = await arm.get_cartesian_position()
  return (c.location.x, c.location.y, c.location.z, math.radians(c.rotation.z))


async def ready(backend: MistrackingBackend) -> ExperimentalSCARA:
  arm = ExperimentalSCARA(backend=backend)
  await arm.setup()
  await arm.home()
  return arm


def reach_task(**kwargs) -> ReachTask:
  """A reach far enough from the start that nothing accidental solves it.

  ``workspace`` is passed explicitly rather than left at the default. With
  ``ReachTask.workspace`` None the envelope built by ``run_trials`` does no
  clipping while the simulated arm still enforces its own default envelope, so a
  random walk can leave the box and abort the whole evaluation. That gap is the
  subject of ``EnvelopeDefaultTests`` below and is kept out of every other test.
  """
  kwargs.setdefault("goal", Goal.at(200.0, 0.0, 40.0, tolerance_mm=2.0))
  kwargs.setdefault("start", START)
  kwargs.setdefault("max_steps", 30)
  kwargs.setdefault("workspace", Workspace())
  return ReachTask(**kwargs)


def pass_through_demo() -> Episode:
  """A run that lands exactly on (20, 0, 100) and then walks 100 mm away from it.

  Every displacement is inside the default envelope limits, so the departure
  cannot be mistaken for a clamped step or a workspace clip.
  """
  actions = [Action.delta(dx=20.0)] + [Action.delta(dx=-25.0)] * 4
  return simulate_episode(KinematicDynamics(), actions, start=START)


class AlwaysSucceedsTask(ReachTask):
  """A deliberately broken criterion: everything passes, including doing nothing."""

  def succeeded(self, episode: Episode) -> bool:
    return True


class ReachTaskResetTests(unittest.IsolatedAsyncioTestCase):
  """A trial starts where the task says, and starts there again on a rerun."""

  async def test_reset_parks_the_arm_at_the_start_pose(self):
    task = reach_task(start=(30.0, -20.0, 120.0, 0.25), start_jitter_mm=0.0)
    arm = await task.reset()
    np.testing.assert_allclose(await pose_of(arm), (30.0, -20.0, 120.0, 0.25), atol=1e-4)
    self.assertTrue(arm.backend.homed)

  async def test_reset_without_jitter_repeats_the_same_pose_every_trial(self):
    """Jitter off is still available, and is still exactly repeatable.

    Asked for explicitly, because zero is no longer the default: see
    ``IdenticalTrialsTests`` for the report this default exists to stop.
    """
    task = reach_task(start_jitter_mm=0.0)
    first = await pose_of(await task.reset(trial=0, seed=0))
    later = await pose_of(await task.reset(trial=7, seed=3))
    np.testing.assert_allclose(first, START, atol=1e-4)
    np.testing.assert_allclose(later, START, atol=1e-4)

  async def test_jitter_is_seeded_and_reproducible(self):
    task = reach_task(start_jitter_mm=5.0)
    a = await pose_of(await task.reset(trial=1, seed=7))
    b = await pose_of(await task.reset(trial=1, seed=7))
    other_trial = await pose_of(await task.reset(trial=2, seed=7))
    other_seed = await pose_of(await task.reset(trial=1, seed=8))

    np.testing.assert_allclose(a, b, atol=1e-9)
    self.assertFalse(np.allclose(a, other_trial, atol=1e-6))
    self.assertFalse(np.allclose(a, other_seed, atol=1e-6))

  async def test_jitter_displaces_translation_only_and_stays_in_bounds(self):
    task = reach_task(start=(30.0, -20.0, 120.0, 0.25), start_jitter_mm=5.0)
    poses = [np.array(await pose_of(await task.reset(trial=i, seed=0))) for i in range(6)]
    for pose in poses:
      np.testing.assert_array_less(np.abs(pose[:3] - np.array(task.start[:3])), 5.0 + 1e-9)
      self.assertAlmostEqual(pose[3], 0.25, places=6)
    spread = np.max([np.abs(p[:3] - np.array(task.start[:3])).max() for p in poses])
    self.assertGreater(spread, 0.0)

  async def test_backend_factory_is_honored(self):
    task = reach_task(backend_factory=lambda: MistrackingBackend(gain=0.5), start_jitter_mm=0.0)
    arm = await task.reset()
    self.assertIsInstance(arm.backend, MistrackingBackend)
    # The arm is placed at the declared start, not driven there. Driving would run
    # the setup move through the planted defect, so a gain=0.5 backend would begin
    # at z = 250 while a gain=1.0 backend began at z = 100, and every comparison
    # between the two would be confounded by the thing being compared.
    np.testing.assert_allclose(await pose_of(arm), (0.0, 0.0, 100.0, 0.0), atol=1e-9)

    perfect = await reach_task(start_jitter_mm=0.0).reset()
    np.testing.assert_allclose(await pose_of(perfect), await pose_of(arm), atol=1e-9)

  async def test_a_hardware_backend_is_refused_before_setup_or_home(self):
    """Anything outside the simulation hierarchy is refused, and refused early.

    ``DegreeReportingBackend`` is a ``SCARABackend`` and not a
    ``SimulationArmBackend``, which is the whole of what the refusal tests, and
    it records every call it receives. The empty call list is the load-bearing
    assertion: refusing after setup and home would still raise the right error
    while having already energised and moved whatever is on the other end.
    """
    backend = DegreeReportingBackend()
    task = reach_task(backend_factory=lambda: backend)

    with self.assertRaises(HardwarePermissionRequired) as ctx:
      await task.reset()

    self.assertIn("simulation-only", str(ctx.exception))
    self.assertEqual(backend.calls, [])


class FinalStateScoringTests(unittest.IsolatedAsyncioTestCase):
  """Success is where the run ended, never the best frame along the way."""

  def test_an_episode_that_passed_through_the_goal_and_left_is_a_failure(self):
    goal = Goal.at(20.0, 0.0, 100.0, tolerance_mm=2.0)
    task = reach_task(goal=goal, start_jitter_mm=0.0)
    episode = pass_through_demo()

    errors = [float(goal.translation_error(o.pose_vector())) for o in episode.observations()]
    self.assertLessEqual(min(errors), goal.tolerance_mm)  # a max over frames would pass this
    self.assertFalse(task.succeeded(episode))
    self.assertAlmostEqual(task.final_error_mm(episode), 100.0, places=6)

  async def test_a_rollout_that_passed_through_the_goal_is_scored_as_a_failure(self):
    """The geometry needs an exactly known start, so jitter is switched off here.

    That makes the two replayed trials byte-identical, so the report is correctly
    marked invalid on that separate ground; what this test is about is the score,
    and the criterion behind it, so both are asserted directly. The identical
    trials themselves are the subject of ``IdenticalTrialsTests``.
    """
    goal = Goal.at(20.0, 0.0, 100.0, tolerance_mm=2.0)
    task = reach_task(goal=goal, start_jitter_mm=0.0)
    demo = pass_through_demo()

    arm = await task.reset()
    envelope = SafetyEnvelope(max_steps=task.max_steps, workspace=task.workspace)
    episode = await rollout(arm, ReplayPolicy(demo), envelope=envelope, task=task.name)

    errors = [float(goal.translation_error(o.pose_vector())) for o in episode.observations()]
    self.assertLessEqual(min(errors), goal.tolerance_mm)
    self.assertEqual(episode.metadata["clamped_steps"], 0)  # not a clamping artifact
    self.assertFalse(task.succeeded(episode))

    report = await evaluate(lambda: ReplayPolicy(demo), task, n_trials=2, seed=0)
    self.assertEqual(report.policy_name, "ReplayPolicy")
    self.assertEqual(report.success_rate, 0.0)
    self.assertAlmostEqual(report.mean_final_error_mm, 100.0, places=6)
    self.assertTrue(report.null.passed)  # the criterion discriminates
    self.assertEqual(report.n_distinct_action_sequences, 1)
    self.assertFalse(report.valid)

  def test_an_episode_with_no_final_observation_cannot_have_succeeded(self):
    task = reach_task(goal=Goal.at(0.0, 0.0, 0.0, tolerance_mm=1000.0))
    episode = Episode(task="reach")
    episode.append(Observation(x=0.0, y=0.0, z=0.0), Action.hold())
    self.assertIsNone(episode.final_observation)
    self.assertFalse(task.succeeded(episode))
    self.assertEqual(task.final_error_mm(episode), float("inf"))


class NullTestTests(unittest.IsolatedAsyncioTestCase):
  """A task a null control can pass is not a task, and must be reported as such."""

  async def test_a_well_posed_reach_passes_its_null_tests(self):
    report = await null_test(reach_task(), n_trials=3, seed=0)
    self.assertTrue(report.passed)
    self.assertEqual([r.name for r in report.results], ["NoOpPolicy", "RandomPolicy"])
    self.assertEqual([r.success_rate for r in report.results], [0.0, 0.0])
    # The trial count travels with the rate, so a rate of 0.00 over 0 trials
    # cannot be read as evidence.
    self.assertEqual([r.n_trials for r in report.results], [3, 3])
    self.assertIn("null tests: ok", report.summary())

  async def test_a_criterion_that_always_returns_true_is_caught(self):
    task = AlwaysSucceedsTask(
      goal=Goal.at(200.0, 0.0, 40.0, tolerance_mm=2.0),
      start=START,
      max_steps=10,
      workspace=Workspace(),
    )
    report = await null_test(task, n_trials=3, seed=0)
    self.assertFalse(report.passed)
    self.assertEqual([r.success_rate for r in report.results], [1.0, 1.0])
    self.assertIn("BROKEN CRITERION", report.summary())

  async def test_a_tolerance_wide_enough_to_pass_anything_is_caught(self):
    task = reach_task(goal=Goal.at(200.0, 0.0, 40.0, tolerance_mm=1e6), max_steps=10)
    report = await null_test(task, n_trials=3, seed=0)
    self.assertFalse(report.passed)
    self.assertEqual([r.success_rate for r in report.results], [1.0, 1.0])

  async def test_evaluate_marks_the_report_invalid_and_leads_its_summary_with_it(self):
    task = reach_task(goal=Goal.at(200.0, 0.0, 40.0, tolerance_mm=1e6), max_steps=10)
    report = await evaluate(NoOpPolicy, task, n_trials=3, seed=0)

    self.assertEqual(report.success_rate, 1.0)  # the confidently wrong number
    self.assertFalse(report.valid)
    self.assertIsNotNone(report.null)
    self.assertFalse(report.null.passed)
    summary = report.summary()
    self.assertTrue(summary.startswith("INVALID"), summary)
    self.assertIn("BROKEN CRITERION", summary)

  async def test_evaluate_names_a_seed_accepting_factory_without_an_unseeded_call(self):
    built_with = []

    def seeded_factory(trial_seed):
      built_with.append(trial_seed)
      return NoOpPolicy()

    report = await evaluate(
      seeded_factory, reach_task(max_steps=2), n_trials=2, null_check=False, seed=7
    )

    self.assertEqual(report.policy_name, "NoOpPolicy")
    self.assertEqual(built_with, [700_000, 700_001])

  async def test_null_check_false_leaves_the_report_unchecked_and_says_nothing(self):
    task = reach_task()
    report = await evaluate(NoOpPolicy, task, n_trials=2, null_check=False, seed=0)
    self.assertIsNone(report.null)
    self.assertFalse(report.null_checked)
    self.assertNotIn("null tests", report.summary())

    # valid is a claim that the null controls ran and failed. With the check
    # switched off none ran, so there is nothing to claim and valid must not say
    # otherwise: "nothing was checked" cannot report the same as "the check
    # passed", or the flag reads as trustworthy on exactly the runs where nothing
    # established that.
    self.assertFalse(report.valid)
    self.assertTrue(report.summary().startswith("UNVALIDATED"))

    broken = reach_task(goal=Goal.at(200.0, 0.0, 40.0, tolerance_mm=1e6), max_steps=10)
    unchecked = await evaluate(NoOpPolicy, broken, n_trials=2, null_check=False, seed=0)
    self.assertEqual(unchecked.success_rate, 1.0)
    self.assertFalse(unchecked.valid)


class IdenticalTrialsTests(unittest.IsolatedAsyncioTestCase):
  """Regression: n copies of one run must not be reported as n trials.

  The confidently wrong number this prevents, measured on this repository before
  the fix: ``ReachTask.start_jitter_mm`` defaulted to 0.0, so a seeded
  ``SamplingPlanner`` against a seeded simulation backend produced five trials
  that agreed on every recorded byte, and ``evaluate`` returned
  ``success_rate=1.00`` over "5 trials" with ``valid=True``. The identical
  trials were mentioned in a note under the number, and a note is not a term in
  anything a caller asserts on.

  Both halves are asserted here, because either alone can be undone silently: the
  default going back to zero, and the validity terms going back to being a note.
  """

  @staticmethod
  def planner_factory(task: ReachTask):
    return lambda: SamplingPlanner(task.goal, step_mm=25.0, n_samples=64, seed=0)

  async def test_five_identical_trials_are_refused_however_good_the_rate_looks(self):
    task = reach_task(start_jitter_mm=0.0)
    report = await evaluate(self.planner_factory(task), task, n_trials=5, seed=0)

    # The number itself is real, and on its own it reads as a strong result.
    self.assertEqual(report.success_rate, 1.0)
    self.assertEqual(report.n_trials, 5)
    self.assertTrue(report.null.passed)  # the criterion is not the problem
    # It is one trajectory. Both detectors have to see that, not just one.
    self.assertEqual(report.n_distinct_action_sequences, 1)
    self.assertTrue(report.trials_are_identical)
    self.assertFalse(report.trials_varied)
    self.assertFalse(report.valid)
    summary = report.summary()
    self.assertIn("same action sequence", summary)
    self.assertTrue(summary.startswith("INVALID"), summary)

  async def test_the_default_jitter_makes_five_trials_into_five_trials(self):
    self.assertEqual(ReachTask.start_jitter_mm, DEFAULT_START_JITTER_MM)
    self.assertGreater(DEFAULT_START_JITTER_MM, 0.0)

    task = reach_task()  # the default, deliberately not overridden
    self.assertEqual(task.start_jitter_mm, DEFAULT_START_JITTER_MM)
    report = await evaluate(self.planner_factory(task), task, n_trials=5, seed=0)

    self.assertEqual(report.n_distinct_action_sequences, 5)
    self.assertFalse(report.trials_are_identical)
    self.assertTrue(report.trials_varied)
    self.assertTrue(report.valid)
    self.assertIsNone(report.variation_note())
    self.assertIn("distinct actions: 5/5", report.summary())

  async def test_different_trajectories_that_converge_are_a_result_not_a_duplicate(self):
    """The false refusal the outcome test must not produce.

    A planner driven by a model that is wrong about the arm's gain reaches the
    same fixed point from every jittered start, so five different trajectories
    finish at the same error to the last bit. Reading "identical" off the outcome
    alone would call that invalid and print BROKEN CRITERION next to a criterion
    whose null controls passed, which would leave this harness unable to report a
    reproducible systematic failure at all. Identity is read off the actions.
    """
    task = reach_task(
      goal=Goal.at(200.0, 0.0, 40.0, tolerance_mm=0.5),
      max_steps=40,
      backend_factory=lambda: MistrackingBackend(
        gain=0.85, bias=(0.6, -0.4, 0.3, 0.0), workspace=Workspace()
      ),
    )
    report = await evaluate(
      lambda: SamplingPlanner(task.goal, dynamics=KinematicDynamics(), step_mm=25.0, seed=0),
      task,
      n_trials=5,
      seed=0,
    )

    errors = {round(t.final_error_mm, 12) for t in report.trials}
    self.assertEqual(len(errors), 1)  # a genuine fixed point
    self.assertTrue(report.outcomes_are_identical)
    self.assertEqual(report.success_rate, 0.0)

    self.assertEqual(report.n_distinct_action_sequences, 5)
    self.assertFalse(report.trials_are_identical)
    self.assertTrue(report.null.passed)
    self.assertTrue(report.valid)
    self.assertIsNone(report.variation_note())
    summary = report.summary()
    self.assertIn("fixed point", summary)
    # This is the run examples/05 prints. Every trial failed, so the note has to
    # say failure; it used to say "success" here, two lines under "success rate:
    # 0.00", in a file about to be published.
    self.assertIn("success rate:     0.00", summary)
    self.assertIn("converged to the same failure", summary)
    self.assertNotIn("converged to the same success", summary)

  async def test_one_trial_is_not_a_rate(self):
    task = reach_task()
    report = await evaluate(self.planner_factory(task), task, n_trials=1, seed=0)

    self.assertEqual(report.success_rate, 1.0)
    self.assertFalse(report.valid)
    self.assertIn(f"at least {MINIMUM_TRIALS}", report.variation_note())

  def test_the_digest_separates_two_action_sequences_that_agree_on_their_outcome(self):
    """Why the digest exists: the aggregate outcome cannot tell these apart.

    Two episodes that command different motions and finish with the same step
    count and the same final error are indistinguishable to
    ``trials_are_identical``. A report of those two is real evidence and must
    stay valid, which it only can if distinctness is read off the actions.
    """
    left = simulate_episode(KinematicDynamics(), [Action.delta(dx=10.0)], start=START)
    right = simulate_episode(KinematicDynamics(), [Action.delta(dx=-10.0)], start=START)

    self.assertNotEqual(action_digest(left), action_digest(right))
    self.assertEqual(action_digest(left), action_digest(left))  # and it is stable

    goal = Goal.at(0.0, 0.0, 100.0)
    self.assertAlmostEqual(
      float(goal.translation_error(left.final_observation.pose_vector())),
      float(goal.translation_error(right.final_observation.pose_vector())),
      places=12,
    )

  def test_hand_built_trials_carry_no_digest_and_are_not_evidence(self):
    """A TrialResult nobody ran must not be able to make a report valid."""
    trials = [
      TrialResult(success=True, steps=5, final_error_mm=float(i), clamped_steps=0,
                  stop_reason="policy_terminated")
      for i in range(3)
    ]
    report = EvalReport(task_name="reach", policy_name="Test", trials=trials)
    self.assertEqual(report.n_distinct_action_sequences, 1)
    self.assertFalse(report.trials_are_identical)  # the errors differ
    self.assertFalse(report.trials_varied)  # and still not evidence
    self.assertFalse(report.valid)


class ConvergedOutcomeNoteTests(unittest.TestCase):
  """The converged-outcome note has to name the outcome that was reached.

  The measured defect: the note was written as "converged to the same success"
  with no reference to ``TrialResult.success``, so a task on which every trial
  failed printed that sentence directly beneath ``success rate:     0.00``. Both
  directions are asserted, because a note hardcoded to "failure" would be exactly
  as wrong and would pass a test that only checked the failing case.

  Hand-built trials with distinct digests, so the only thing under test is the
  wording: identical outcomes make ``outcomes_are_identical`` true while distinct
  action digests keep ``trials_are_identical`` false, which is the one combination
  that prints the note at all.
  """

  @staticmethod
  def converged(success: bool) -> EvalReport:
    trials = [
      TrialResult(
        success=success,
        steps=7,
        final_error_mm=0.9189,
        clamped_steps=0,
        stop_reason="policy_terminated",
        action_digest=f"digest-{i}",
      )
      for i in range(3)
    ]
    return EvalReport(task_name="reach", policy_name="Test", trials=trials)

  def test_a_converged_failure_is_not_reported_as_a_success(self):
    report = self.converged(success=False)
    self.assertTrue(report.outcomes_are_identical)
    self.assertFalse(report.trials_are_identical)
    self.assertEqual(report.success_rate, 0.0)

    summary = report.summary()
    self.assertIn("converged to the same failure", summary)
    self.assertNotIn("converged to the same success", summary)
    self.assertIn("fixed point", summary)

  def test_a_converged_success_still_says_success(self):
    report = self.converged(success=True)
    self.assertTrue(report.outcomes_are_identical)
    self.assertFalse(report.trials_are_identical)
    self.assertEqual(report.success_rate, 1.0)

    summary = report.summary()
    self.assertIn("converged to the same success", summary)
    self.assertNotIn("converged to the same failure", summary)


class EnvelopeDefaultTests(unittest.IsolatedAsyncioTestCase):
  """What run_trials hands the rollout when the task did not name a workspace."""

  async def test_a_task_without_a_workspace_can_still_be_null_tested(self):
    task = ReachTask(goal=Goal.at(200.0, 0.0, 40.0, tolerance_mm=2.0), start=START, max_steps=60)
    self.assertIsNone(task.workspace)
    report = await null_test(task, n_trials=2, seed=6)
    self.assertTrue(report.passed)


class EvalReportTests(unittest.TestCase):
  """The aggregates, including what they say when there is nothing to average."""

  @staticmethod
  def trial(success: bool, steps: int, error: float, clamped: int = 0) -> TrialResult:
    return TrialResult(
      success=success,
      steps=steps,
      final_error_mm=error,
      clamped_steps=clamped,
      stop_reason="policy_terminated",
    )

  def report(self, trials) -> EvalReport:
    return EvalReport(task_name="reach", policy_name="Test", trials=list(trials))

  def test_success_rate_and_mean_error(self):
    report = self.report(
      [
        self.trial(True, steps=10, error=0.5),
        self.trial(False, steps=30, error=40.0),
        self.trial(False, steps=30, error=60.0),
        self.trial(False, steps=30, error=99.5),
      ]
    )
    self.assertEqual(report.n_trials, 4)
    self.assertEqual(report.success_rate, 0.25)
    self.assertAlmostEqual(report.mean_final_error_mm, 50.0, places=9)

  def test_mean_steps_counts_only_the_trials_that_succeeded(self):
    report = self.report(
      [
        self.trial(True, steps=10, error=0.5),
        self.trial(True, steps=20, error=0.5),
        self.trial(False, steps=300, error=90.0),
      ]
    )
    self.assertAlmostEqual(report.mean_steps_to_success, 15.0, places=9)

  def test_mean_steps_is_infinite_when_nothing_succeeded(self):
    report = self.report([self.trial(False, steps=30, error=90.0) for _ in range(3)])
    self.assertEqual(report.mean_steps_to_success, float("inf"))
    self.assertEqual(report.success_rate, 0.0)
    self.assertIn("steps to success: n/a", report.summary())

  def test_an_empty_report_averages_to_infinity_rather_than_zero(self):
    report = self.report([])
    self.assertEqual(report.n_trials, 0)
    self.assertEqual(report.success_rate, 0.0)
    self.assertEqual(report.mean_final_error_mm, float("inf"))
    self.assertEqual(report.mean_steps_to_success, float("inf"))
    # No null controls were attached, so there is no basis to call it valid.
    self.assertFalse(report.valid)

  def test_clamped_trials_counts_trials_not_steps(self):
    report = self.report(
      [
        self.trial(True, steps=10, error=0.5, clamped=7),
        self.trial(True, steps=10, error=0.5, clamped=1),
        self.trial(True, steps=10, error=0.5, clamped=0),
      ]
    )
    self.assertEqual(report.clamped_trials, 2)
    self.assertIn("trials clamped:   2/3", report.summary())


class EvaluateDynamicsTests(unittest.TestCase):
  """A model is credited with an improvement only when it made one."""

  @staticmethod
  def episodes(truth, n: int = 3):
    return [
      simulate_episode(truth, random_walk_actions(40, seed=s), start=START) for s in range(n)
    ]

  def test_a_model_matching_the_planted_truth_improves_on_the_baseline(self):
    truth = KinematicDynamics(gain=0.8, bias=[1.0, 0.0, -0.5, 0.0])
    score = evaluate_dynamics(truth, self.episodes(truth))
    self.assertEqual(score.n, 120)
    self.assertLess(score.rmse, 1e-9)
    self.assertGreater(score.baseline_rmse, 1.0)
    self.assertTrue(score.improved)
    self.assertIn("improved", score.summary())

  def test_a_mismatched_model_does_not_claim_an_improvement(self):
    truth = KinematicDynamics(gain=0.8, bias=[1.0, 0.0, -0.5, 0.0])
    score = evaluate_dynamics(KinematicDynamics(gain=1.6), self.episodes(truth))
    self.assertGreater(score.rmse, score.baseline_rmse)
    self.assertFalse(score.improved)
    self.assertIn("NO IMPROVEMENT", score.summary())

  def test_scoring_refuses_a_source_with_no_cartesian_transitions(self):
    joints_only = Episode(task="joints")
    joints_only.append(Observation(x=1.0, y=2.0, z=3.0), Action.joint_targets({1: 10.0}))
    joints_only.final_observation = Observation(x=4.0, y=5.0, z=6.0)

    with self.assertRaises(ValueError) as ctx:
      evaluate_dynamics(KinematicDynamics(), [joints_only])
    self.assertIn("no Cartesian transitions", str(ctx.exception))

    with self.assertRaises(ValueError):
      evaluate_dynamics(KinematicDynamics(), [])


class MistrackingBackendTests(unittest.IsolatedAsyncioTestCase):
  """The defect is planted, so it can be stated exactly and scored against."""

  async def test_gain_is_applied_to_the_commanded_displacement(self):
    backend = MistrackingBackend(gain=0.5)
    arm = await ready(backend)
    np.testing.assert_allclose(await pose_of(arm), (0.0, 0.0, 400.0, 0.0), atol=1e-9)

    await arm.move_to(cartesian(100.0, 0.0, 400.0))
    np.testing.assert_allclose(await pose_of(arm), (50.0, 0.0, 400.0, 0.0), atol=1e-4)

    # Gain compounds over moves: the second command closes half of what is left.
    await arm.move_to(cartesian(100.0, 0.0, 400.0))
    np.testing.assert_allclose(await pose_of(arm), (75.0, 0.0, 400.0, 0.0), atol=1e-4)

  async def test_per_axis_gain_and_bias_are_both_applied(self):
    backend = MistrackingBackend(gain=[1.0, 0.5, 1.0, 1.0], bias=[2.0, 0.0, -1.0, 0.01])
    arm = await ready(backend)
    await arm.move_to(cartesian(100.0, 80.0, 300.0, 0.2))
    np.testing.assert_allclose(await pose_of(arm), (102.0, 40.0, 299.0, 0.21), atol=1e-4)

  async def test_true_dynamics_is_the_arm_it_was_taken_from(self):
    backend = MistrackingBackend(gain=[0.9, 0.8, 0.7, 1.0], bias=[1.5, -2.0, 0.5, 0.02])
    np.testing.assert_allclose(backend.true_dynamics().gain, [0.9, 0.8, 0.7, 1.0])
    np.testing.assert_allclose(backend.true_dynamics().bias, [1.5, -2.0, 0.5, 0.02])

    arm = await ready(backend)
    model = backend.true_dynamics()
    for target in [(120.0, 60.0, 300.0, 0.3), (-40.0, -80.0, 150.0, -0.1)]:
      pose = np.array(await pose_of(arm), float)
      delta = np.array(target, float) - pose
      await arm.move_to(cartesian(*target))
      np.testing.assert_allclose(await pose_of(arm), model.predict(pose, delta), atol=1e-4)

  async def test_noise_is_seeded_and_reproducible(self):
    async def achieved(seed: int, noise_mm: float = 1.0):
      backend = MistrackingBackend(noise_mm=noise_mm, seed=seed)
      arm = await ready(backend)
      out = []
      for x in (50.0, 100.0, 150.0):
        await arm.move_to(cartesian(x, 0.0, 300.0, 0.3))
        out.append(await pose_of(arm))
      return np.array(out, float)

    same = await achieved(5)
    again = await achieved(5)
    other = await achieved(6)
    exact = await achieved(5, noise_mm=0.0)

    np.testing.assert_allclose(same, again, atol=1e-12)
    self.assertFalse(np.allclose(same, other, atol=1e-6))
    self.assertFalse(np.allclose(same[:, :3], exact[:, :3], atol=1e-6))
    # Noise is translation only, so a rotation command still lands where told.
    np.testing.assert_allclose(same[:, 3], 0.3, atol=1e-9)

  async def test_perfect_gain_reproduces_the_phase_one_backend(self):
    backend = MistrackingBackend(gain=1.0)
    arm = await ready(backend)
    await arm.move_to(cartesian(123.5, -45.25, 210.75, 0.4))
    np.testing.assert_allclose(await pose_of(arm), (123.5, -45.25, 210.75, 0.4), atol=1e-4)

  async def test_the_home_interlock_is_inherited(self):
    backend = MistrackingBackend(gain=0.9)
    arm = ExperimentalSCARA(backend=backend)
    await arm.setup()  # set up but not homed
    with self.assertRaises(ArmNotHomedError):
      await arm.move_to(cartesian(100.0, 0.0, 300.0))
    self.assertFalse(backend.homed)

  async def test_the_workspace_guard_is_inherited(self):
    backend = MistrackingBackend(gain=1.0, workspace=Workspace(x=(-200.0, 200.0)))
    arm = await ready(backend)
    with self.assertRaises(WorkspaceViolation):
      await arm.move_to(cartesian(999.0, 0.0, 300.0))

  async def test_the_workspace_guard_judges_where_the_arm_actually_lands(self):
    """A mistracked arm is checked on the pose it reaches, not the one commanded."""
    backend = MistrackingBackend(gain=0.5, workspace=Workspace(x=(-200.0, 200.0)))
    arm = await ready(backend)
    await arm.move_to(cartesian(390.0, 0.0, 400.0))  # lands at 195.0, inside
    np.testing.assert_allclose(await pose_of(arm), (195.0, 0.0, 400.0, 0.0), atol=1e-4)
    with self.assertRaises(WorkspaceViolation):
      await arm.move_to(cartesian(900.0, 0.0, 400.0))  # would land at 547.5

  async def test_the_gripper_and_pick_sequence_are_inherited(self):
    backend = MistrackingBackend(gain=0.9)
    arm = await ready(backend)
    self.assertFalse(backend.holding)

    await arm.pick_up_resource(site_coords(Pose2D(150.0, 0.0, 10.0)), plate_width=78.0)
    self.assertTrue(backend.holding)
    self.assertTrue(await arm.is_gripper_closed())
    self.assertAlmostEqual(backend._gripper_width, 78.0, places=6)

    await arm.drop_resource(site_coords(Pose2D(-150.0, 0.0, 10.0)))
    self.assertFalse(backend.holding)
    self.assertFalse(await arm.is_gripper_closed())

    actions = [e["action"] for e in backend.trace]
    for expected in ("home", "open_gripper", "descend_pick", "close_gripper", "descend_drop"):
      self.assertIn(expected, actions)


if __name__ == "__main__":
  unittest.main()
