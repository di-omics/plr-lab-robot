"""Tests for plr_lr.learn.dynamics.

A dynamics fit is the one part of this repository that can be wrong while
looking right. Least squares always reduces the error on the rows it was handed,
so a fit that reports its training error reports a number that is true and
useless, and a planner built on it will confidently drive the arm somewhere the
model invented. These tests are organised around the claims the module makes
about itself rather than around its functions, because the claims are what a
later refactor would quietly break.

The claims are: that the unlearned baseline does exactly what an open-loop
script assumes and does it identically whether called on one pose or a batch;
that an unfit residual is its base and nothing more; that a fit against a
PLANTED model recovers the planted numbers on inputs it never saw; that a fit
with nothing to learn says so; that a fit which learned only noise is caught by
the holdout even though its training error fell; that a fit whose holdout is not
independent of its training rows refuses to answer at all rather than answering
wrongly one time in twenty; that too little data is refused outright instead of
producing a perfect-looking fit; and that an injected learned-model adapter
validates every prediction before returning it.

Every generator here is seeded. The noise null in particular is sized on
purpose: 18 training transitions for 9 features is where least squares can
memorise noise, which is the regime the holdout exists to police.
"""

from __future__ import annotations

import unittest

import numpy as np

from plr_lr.learn import (
  Action,
  DynamicsModel,
  Episode,
  InsufficientData,
  KinematicDynamics,
  LearnedDynamics,
  Observation,
  ResidualDynamics,
  VideoWorldModel,
  random_walk_actions,
  simulate_episode,
)
from plr_lr.learn.dynamics import (
  IMPROVEMENT_MARGIN,
  SPLIT_UNSOUND,
  DynamicsAdapterError,
  N_FEATURES,
)

#: The tracking defect these tests plant and then demand back out of the fit.
PLANTED_GAIN = 0.85
PLANTED_BIAS = (0.5, -0.25, 0.1, 0.02)


def planted_dynamics() -> KinematicDynamics:
  """The model a correct fit has to recover, exactly."""
  return KinematicDynamics(gain=PLANTED_GAIN, bias=PLANTED_BIAS)


def excited_episodes(dynamics, n_episodes: int = 4, n_steps: int = 40, seed: int = 100):
  """Episodes that move every axis, rz included.

  The design matrix is ``[pose, delta, 1]``, so an axis that never moves leaves
  two of its nine columns identically zero and there is nothing to identify
  there. ``random_walk_actions`` defaults to ``step_rad=0``, which would make the
  recovered rz gain whatever the minimum-norm solution happened to pick rather
  than the planted one, and the fit would still look perfect on its own data.
  """
  episodes = []
  for k in range(n_episodes):
    actions = random_walk_actions(n_steps, step_mm=20.0, step_rad=0.25, seed=seed + k)
    start = (10.0 * k, -5.0 * k, 50.0, 0.1 * k)
    episodes.append(simulate_episode(dynamics, actions, start=start))
  return episodes


class PerfectPlusNoise:
  """A perfectly tracking arm with jitter no model could predict.

  This is the honest null for a residual fit: the base model is already right
  and the only thing left in the residual is noise. Making the next state pure
  noise instead would be a weaker test, because a fit that learns to ignore the
  action really would beat a baseline that still adds it, and reporting that
  improvement would be correct rather than a bug.
  """

  def __init__(self, sigma_mm: float, seed: int) -> None:
    self.base = KinematicDynamics()
    self.sigma_mm = float(sigma_mm)
    self.rng = np.random.default_rng(seed)

  def predict(self, pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    out = np.asarray(self.base.predict(pose, delta), float)
    return out + self.rng.normal(0.0, self.sigma_mm, size=out.shape)


def noise_episodes(seed: int = 11):
  """Twenty four transitions whose deviation from perfect tracking is pure noise.

  Sized so the default holdout leaves exactly the minimum training set the fit
  will accept, which is where overfitting is largest and the holdout has to do
  its job.
  """
  dynamics = PerfectPlusNoise(sigma_mm=5.0, seed=seed)
  episodes = []
  for k in range(4):
    actions = random_walk_actions(6, step_mm=20.0, step_rad=0.25, seed=400 + k)
    episodes.append(simulate_episode(dynamics, actions, start=(0.0, 0.0, 50.0, 0.0)))
  return episodes


#: The single-episode noise experiment used by SplitSoundnessTests, sized from the
#: measurement in that class's docstring. Twenty four transitions and a ten
#: percent holdout is the shape the false improvements were counted at.
NOISE_RUNS = 40
NOISE_STEPS = 24
NOISE_SIGMA_MM = 5.0
NOISE_HOLDOUT = 0.1


def single_noise_episode(run: int):
  """One episode of pure-noise residual, seeded by run index."""
  dynamics = PerfectPlusNoise(sigma_mm=NOISE_SIGMA_MM, seed=1000 + run)
  actions = random_walk_actions(NOISE_STEPS, step_mm=20.0, step_rad=0.25, seed=2000 + run)
  return simulate_episode(dynamics, actions, start=(0.0, 0.0, 50.0, 0.0))


def transition_arrays(episodes):
  """The (pose, delta, next) arrays a fit sees, extracted the way the fit does."""
  poses, deltas, nexts = [], [], []
  for ep in episodes:
    for obs, action, nxt in ep.transitions():
      pose = obs.pose_vector()
      poses.append(pose)
      deltas.append(action.as_delta(pose))
      nexts.append(nxt.pose_vector())
  return np.array(poses), np.array(deltas), np.array(nexts)


def rmse(pred: np.ndarray, truth: np.ndarray) -> float:
  """The same translation-only Euclidean rmse the fit reports, in mm."""
  return float(np.sqrt(np.mean(np.sum((np.atleast_2d(pred)[:, :3] - truth[:, :3]) ** 2, axis=1))))


class TestKinematicBaseline(unittest.TestCase):
  """The unlearned model: what every open-loop script already assumes."""

  def test_a_scalar_gain_scales_every_axis(self):
    model = KinematicDynamics(gain=0.9)
    out = model.predict(np.array([1.0, 2.0, 3.0, 0.1]), np.array([10.0, 20.0, 30.0, 1.0]))
    np.testing.assert_allclose(out, [10.0, 20.0, 30.0, 1.0], atol=1e-12)

  def test_a_per_axis_gain_and_bias_apply_axis_by_axis(self):
    model = KinematicDynamics(gain=[0.9, 0.8, 1.0, 0.5], bias=[1.0, -2.0, 0.5, 0.01])
    out = model.predict(np.array([1.0, 2.0, 3.0, 0.1]), np.array([10.0, 10.0, 10.0, 1.0]))
    np.testing.assert_allclose(out, [11.0, 8.0, 13.5, 0.61], atol=1e-12)

  def test_a_single_pose_returns_a_single_pose(self):
    """Shape is part of the contract: a planner indexes the result directly."""
    model = KinematicDynamics(gain=0.85)
    out = model.predict(np.zeros(4), np.ones(4))
    self.assertEqual(out.shape, (4,))

  def test_batched_prediction_matches_looping_one_at_a_time(self):
    """The batch path is the one the planner uses, so it has to agree."""
    model = KinematicDynamics(gain=[0.95, 0.8, 1.05, 0.5], bias=[0.3, -0.4, 0.2, -0.01])
    rng = np.random.default_rng(5)
    poses = rng.uniform(-250.0, 250.0, size=(16, 4))
    deltas = rng.uniform(-40.0, 40.0, size=(16, 4))

    batched = model.predict(poses, deltas)
    self.assertEqual(batched.shape, (16, 4))
    looped = np.array([model.predict(poses[i], deltas[i]) for i in range(16)])
    np.testing.assert_allclose(batched, looped, atol=0.0, rtol=0.0)

  def test_one_pose_broadcasts_across_a_batch_of_deltas(self):
    """Random shooting evaluates many candidate deltas from one current pose."""
    model = KinematicDynamics(gain=0.85, bias=PLANTED_BIAS)
    pose = np.array([100.0, -50.0, 30.0, 0.2])
    deltas = np.random.default_rng(6).uniform(-25.0, 25.0, size=(12, 4))

    out = model.predict(pose, deltas)
    self.assertEqual(out.shape, (12, 4))
    looped = np.array([model.predict(pose, deltas[i]) for i in range(12)])
    np.testing.assert_allclose(out, looped, atol=0.0, rtol=0.0)

  def test_a_vector_that_is_not_a_pose_is_refused(self):
    model = KinematicDynamics()
    with self.assertRaises(ValueError) as ctx:
      model.predict(np.zeros(3), np.zeros(3))
    self.assertIn("4-vectors", str(ctx.exception))

  def test_gain_and_bias_are_copied_from_the_caller(self):
    """A caller that reuses its array must not be able to mutate the model."""
    bias = np.array([1.0, 1.0, 1.0, 1.0])
    model = KinematicDynamics(gain=0.5, bias=bias)
    bias[0] = 99.0
    np.testing.assert_allclose(model.bias, [1.0, 1.0, 1.0, 1.0], atol=1e-12)

  def test_non_finite_or_wrong_shaped_parameters_are_refused_at_construction(self):
    for kwargs in (
      {"gain": np.nan},
      {"gain": [1.0, 1.0, 1.0, np.inf]},
      {"bias": [0.0, 0.0, 0.0, np.nan]},
      {"gain": [1.0, 2.0, 3.0]},
      {"bias": [1.0, 2.0, 3.0]},
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        KinematicDynamics(**kwargs)


class TestUnfitResidualIsItsBase(unittest.TestCase):
  """Before any data arrives, the residual model must add exactly nothing."""

  def test_no_weights_means_the_base_prediction_unchanged(self):
    base = KinematicDynamics(gain=[0.9, 0.85, 1.0, 0.7], bias=[0.5, -0.5, 0.25, 0.03])
    model = ResidualDynamics(base=base)
    self.assertIsNone(model.weights)

    rng = np.random.default_rng(7)
    poses = rng.uniform(-200.0, 200.0, size=(10, 4))
    deltas = rng.uniform(-30.0, 30.0, size=(10, 4))
    np.testing.assert_allclose(model.predict(poses, deltas), base.predict(poses, deltas), atol=0.0)
    np.testing.assert_allclose(
      model.predict(poses[0], deltas[0]), base.predict(poses[0], deltas[0]), atol=0.0
    )
    self.assertEqual(model.predict(poses[0], deltas[0]).shape, (4,))

  def test_the_default_base_is_perfect_tracking(self):
    model = ResidualDynamics()
    pose = np.array([10.0, 20.0, 30.0, 0.4])
    delta = np.array([1.0, -2.0, 3.0, 0.1])
    np.testing.assert_allclose(model.predict(pose, delta), pose + delta, atol=1e-12)


class TestPlantAndRecover(unittest.TestCase):
  """Fit against a known answer, then demand the known answer back.

  The data is generated by rolling actions through a KinematicDynamics of gain
  0.85 and a fixed bias, so the residual the fit is asked to learn really is a
  linear function of ``[pose, delta, 1]`` and an exact recovery is available. A
  fit that cannot recover an answer that is exactly in its hypothesis class has
  nothing to say about an arm.
  """

  def setUp(self):
    self.truth = planted_dynamics()
    self.episodes = excited_episodes(self.truth)
    self.model, self.fit = ResidualDynamics.fit(self.episodes, seed=7)

  def test_the_planted_gain_and_bias_come_back_out(self):
    zero = np.zeros(4)
    recovered_bias = self.model.predict(zero, zero)
    np.testing.assert_allclose(recovered_bias, PLANTED_BIAS, atol=1e-9)
    for axis in range(4):
      unit = np.zeros(4)
      unit[axis] = 1.0
      gain = self.model.predict(zero, unit)[axis] - recovered_bias[axis]
      self.assertAlmostEqual(gain, PLANTED_GAIN, places=9)

  def test_the_fit_matches_the_planted_model_on_fresh_probes(self):
    """Probes drawn independently of the training walk, so this is not recall."""
    rng = np.random.default_rng(999)
    poses = rng.uniform(-250.0, 250.0, size=(64, 4))
    deltas = rng.uniform(-40.0, 40.0, size=(64, 4))
    np.testing.assert_allclose(
      self.model.predict(poses, deltas), self.truth.predict(poses, deltas), atol=1e-8
    )

  def test_the_fit_beats_the_unlearned_baseline_on_held_out_data(self):
    # assertIs, not assertTrue: improved is three-valued, and assertTrue would
    # accept anything truthy while assertFalse elsewhere would accept the None
    # that means "not evaluated". Both directions have to be exact.
    self.assertIs(self.fit.improved, True)
    self.assertGreater(self.fit.baseline_holdout_rmse, 1.0)
    self.assertLess(self.fit.holdout_rmse, 1e-6)
    self.assertLess(self.fit.improvement_ratio, 1e-6)
    n_transitions = len(transition_arrays(self.episodes)[0])
    self.assertEqual(self.fit.n_train + self.fit.n_holdout, n_transitions)
    self.assertIn("improved", self.fit.summary())


class TestNullControls(unittest.TestCase):
  """The fits that must not claim anything, and the split that keeps them honest."""

  def test_a_perfect_arm_leaves_nothing_to_learn(self):
    """Nothing to learn is a result, and the fit has to report it as one."""
    episodes = excited_episodes(KinematicDynamics(), seed=200)
    model, fit = ResidualDynamics.fit(episodes, seed=3)

    self.assertIs(fit.improved, False)
    self.assertEqual(fit.holdout_rmse, 0.0)
    self.assertEqual(fit.baseline_holdout_rmse, 0.0)
    self.assertEqual(fit.improvement_ratio, 1.0)
    np.testing.assert_allclose(model.weights, np.zeros((N_FEATURES, 4)), atol=1e-12)
    self.assertIn("NO IMPROVEMENT", fit.summary())

  def test_a_residual_that_is_only_noise_does_not_survive_the_holdout(self):
    """Train error falls, holdout error rises, and only the holdout is believed."""
    episodes = noise_episodes(seed=11)
    model, fit = ResidualDynamics.fit(episodes, seed=5)

    self.assertEqual((fit.n_train, fit.n_holdout), (18, 6))
    self.assertEqual(fit.split_by, "episode")
    # The fit did what least squares always does on its own rows.
    self.assertLess(fit.train_rmse, fit.holdout_rmse)
    self.assertLess(fit.train_rmse, fit.baseline_holdout_rmse)
    # And none of that survives contact with transitions it never saw.
    self.assertIs(fit.improved, False)
    self.assertGreater(fit.holdout_rmse, fit.baseline_holdout_rmse)
    self.assertGreater(fit.improvement_ratio, 1.0)
    self.assertIn("NO IMPROVEMENT", fit.summary())
    # The model still comes back, so the caller decides rather than the fit.
    self.assertIsNotNone(model.weights)

  def test_train_and_holdout_are_distinct_numbers_from_distinct_rows(self):
    """A fit graded on its own homework looks better, so both numbers are kept."""
    episodes = noise_episodes(seed=11)
    model, fit = ResidualDynamics.fit(episodes, seed=5)
    self.assertNotEqual(fit.train_rmse, fit.holdout_rmse)

    # fit()'s seed is a documented argument, so the episode-level split is
    # reproducible. Replaying it is the only way to check that the number
    # labelled holdout was scored on whole episodes the fit never touched.
    poses, deltas, nexts, groups = ResidualDynamics._transitions(episodes)
    train, holdout, split_by = ResidualDynamics._split(groups, 0.25, seed=5)
    self.assertEqual(split_by, "episode")
    self.assertEqual(len(set(train.tolist()) & set(holdout.tolist())), 0)
    self.assertEqual(set(groups[train].tolist()) & set(groups[holdout].tolist()), set())

    self.assertAlmostEqual(
      fit.train_rmse, rmse(model.predict(poses[train], deltas[train]), nexts[train]), places=12
    )
    self.assertAlmostEqual(
      fit.holdout_rmse, rmse(model.predict(poses[holdout], deltas[holdout]), nexts[holdout]),
      places=12,
    )
    self.assertAlmostEqual(
      fit.baseline_holdout_rmse,
      rmse(KinematicDynamics().predict(poses[holdout], deltas[holdout]), nexts[holdout]),
      places=12,
    )

  def test_the_improvement_ratio_is_the_holdout_ratio_and_not_the_train_one(self):
    episodes = noise_episodes(seed=11)
    _, fit = ResidualDynamics.fit(episodes, seed=5)
    self.assertAlmostEqual(
      fit.improvement_ratio, fit.holdout_rmse / fit.baseline_holdout_rmse, places=12
    )
    self.assertNotAlmostEqual(
      fit.improvement_ratio, fit.train_rmse / fit.baseline_holdout_rmse, places=3
    )


class TestSplitSoundness(unittest.TestCase):
  """Regression: a single-episode fit is not allowed to claim an improvement.

  The confidently wrong number this prevents, measured on this repository. Over
  40 seeded runs, each one episode of 24 transitions whose residual is pure
  Gaussian noise at 5 mm and nothing else, split at a 0.1 holdout fraction, the
  old two-valued ``improved`` returned True on 2 of them. A 5 percent rate of
  claiming a learned correction for encoder noise, printed as the word
  "improved", on data with nothing in it to learn.

  The cause is the split, not the margin. One episode cannot be split by episode,
  so the fallback picks rows at random out of one continuous motion: frame 41
  trains, frame 42 is held out, and the two are the same movement a fortieth of a
  second apart. The holdout exists to be independent of the training rows and
  under that split it is not, so no verdict computed from it means anything.

  Three things are asserted, because any one of them alone can be reverted
  quietly: that the underlying numbers still produce the false claim, so the test
  keeps measuring something real; that ``improved`` returns None rather than True
  on every one of those runs; and that None is not False, so the documented
  ``if not fit.improved`` guard still keeps the baseline.
  """

  @staticmethod
  def two_valued(fit) -> bool:
    """The rule this module applied before the split was consulted.

    Kept verbatim so the regression is measured rather than remembered. If the
    module's own numbers stop producing false improvements the first test below
    fails, and the rest of this class stops proving anything, which is the
    correct outcome: at that point the defect is gone for a different reason and
    somebody has to look.
    """
    if fit.baseline_holdout_rmse <= 1e-12:
      return False
    if fit.holdout_rmse > fit.baseline_holdout_rmse * (1.0 - IMPROVEMENT_MARGIN):
      return False
    return fit.holdout_rotation_rmse <= fit.baseline_holdout_rotation_rmse + 1e-12

  def fits(self):
    for run in range(NOISE_RUNS):
      _, fit = ResidualDynamics.fit(
        [single_noise_episode(run)], holdout_fraction=NOISE_HOLDOUT, seed=run
      )
      yield fit

  def test_one_episode_cannot_be_split_by_episode(self):
    fit = next(self.fits())
    self.assertEqual(fit.split_by, "transition")
    self.assertFalse(fit.split_is_sound)

  def test_the_underlying_numbers_still_claim_improvements_on_pure_noise(self):
    """The defect is in the data, not in the property. This measures it."""
    claimed = [self.two_valued(fit) for fit in self.fits()]
    self.assertGreater(
      sum(claimed), 0, "the noise experiment stopped producing false improvements"
    )

  def test_improved_is_none_on_every_run_including_the_false_ones(self):
    verdicts = [fit.improved for fit in self.fits()]
    self.assertEqual(len(verdicts), NOISE_RUNS)
    self.assertEqual(verdicts.count(True), 0)
    self.assertEqual(verdicts.count(False), 0)
    self.assertTrue(all(v is None for v in verdicts))

  def test_none_is_not_false_and_the_documented_guard_still_holds_the_baseline(self):
    """``if not fit.improved`` has to keep the baseline for all three values."""
    fit = next(self.fits())
    self.assertIsNone(fit.improved)
    self.assertIsNot(fit.improved, False)
    self.assertTrue(not fit.improved)  # the guard in the module docstring

  def test_the_summary_prints_the_marker_in_place_of_a_verdict(self):
    summary = next(self.fits()).summary()
    self.assertIn(SPLIT_UNSOUND, summary)
    self.assertNotIn("NO IMPROVEMENT", summary)
    self.assertNotIn("dynamics fit: improved", summary)

  def test_the_refusal_is_about_the_split_and_not_about_noise(self):
    """A sound split on the same noise still answers, and answers False.

    Without this the fix could be "never claim anything", which would also make
    the false-positive rate zero and would destroy the only claim the module
    exists to make.
    """
    _, fit = ResidualDynamics.fit(noise_episodes(seed=11), seed=5)
    self.assertEqual(fit.split_by, "episode")
    self.assertTrue(fit.split_is_sound)
    self.assertIs(fit.improved, False)
    self.assertIn("NO IMPROVEMENT", fit.summary())

    _, planted = ResidualDynamics.fit(excited_episodes(planted_dynamics()), seed=7)
    self.assertIs(planted.improved, True)


class TestRefusals(unittest.TestCase):
  """Every refusal fires instead of returning a number that looks fine."""

  def test_too_few_training_transitions_are_refused(self):
    """Nine rows and nine features fit perfectly and predict nothing."""
    episode = simulate_episode(
      planted_dynamics(), random_walk_actions(12, step_mm=20.0, step_rad=0.25, seed=1)
    )
    self.assertEqual(len(episode.transitions()), 12)
    with self.assertRaises(InsufficientData) as ctx:
      ResidualDynamics.fit([episode])
    message = str(ctx.exception)
    self.assertIn(f"9 training transitions for {N_FEATURES} features", message)
    self.assertIn(f"at least {N_FEATURES * 2 + 3}", message)

  def test_the_threshold_is_min_transitions_per_feature_times_n_features(self):
    """The same data is accepted or refused purely on the declared threshold."""
    actions = random_walk_actions(12, step_mm=20.0, step_rad=0.25, seed=1)
    episode = simulate_episode(planted_dynamics(), actions)

    _, fit = ResidualDynamics.fit([episode], min_transitions_per_feature=1)
    self.assertEqual(fit.n_train, N_FEATURES)

    # One row below the same threshold, and it is refused.
    shorter = simulate_episode(
      planted_dynamics(), random_walk_actions(11, step_mm=20.0, step_rad=0.25, seed=1)
    )
    with self.assertRaises(InsufficientData):
      ResidualDynamics.fit([shorter], min_transitions_per_feature=1)

  def test_joint_only_episodes_have_no_cartesian_transitions_to_fit(self):
    """Joint targets are dropped rather than read as zero Cartesian motion."""
    episode = Episode(task="joints only")
    for i in range(40):
      episode.append(
        Observation(t=float(i), x=float(i), y=-float(i), z=50.0, rz=0.01 * i),
        Action.joint_targets({1: 10.0 * i, 2: -5.0 * i}),
      )
    episode.final_observation = Observation(t=40.0, x=40.0, y=-40.0, z=50.0, rz=0.4)
    self.assertEqual(len(episode.transitions()), 40)

    with self.assertRaises(InsufficientData) as ctx:
      ResidualDynamics.fit([episode])
    self.assertIn("no Cartesian transitions", str(ctx.exception))

  def test_a_holdout_fraction_outside_the_open_unit_interval_is_refused(self):
    """Zero holdout would let a fit report its training error as its score."""
    episodes = excited_episodes(planted_dynamics(), n_episodes=1, seed=300)
    for fraction in (0.0, 1.0, -0.25, 1.5):
      with self.assertRaises(ValueError) as ctx:
        ResidualDynamics.fit(episodes, holdout_fraction=fraction)
      self.assertIn("holdout_fraction", str(ctx.exception))

  def test_non_finite_weights_raise_instead_of_returning_a_number(self):
    """A diverging residual has to stop the planner, not feed it infinities."""
    diverged = ResidualDynamics(
      base=KinematicDynamics(), weights=np.full((N_FEATURES, 4), np.inf)
    )
    with self.assertRaises(FloatingPointError) as ctx:
      diverged.predict(np.array([1.0, 2.0, 3.0, 0.1]), np.array([5.0, 5.0, 5.0, 0.0]))
    self.assertIn("not finite", str(ctx.exception))

    with self.assertRaises(FloatingPointError):
      diverged.predict(np.zeros((3, 4)), np.ones((3, 4)))

    # The guard is not simply always on: finite weights predict finite poses.
    finite = ResidualDynamics(weights=np.zeros((N_FEATURES, 4)))
    np.testing.assert_allclose(finite.predict(np.zeros(4), np.ones(4)), np.ones(4), atol=1e-12)


class TestLearnedModelAdapters(unittest.TestCase):
  """Injected predictors are useful only while their boundary stays explicit."""

  ADAPTERS = (LearnedDynamics, VideoWorldModel)

  def test_a_single_prediction_is_batched_for_the_callable_then_unbatched_for_the_caller(self):
    pose = np.array([10.0, -20.0, 30.0, 0.2])
    delta = np.array([5.0, 4.0, -3.0, 0.1])

    for adapter in self.ADAPTERS:
      seen = []

      def predictor(pose_batch, delta_batch):
        seen.append((pose_batch.shape, delta_batch.shape, pose_batch.dtype, delta_batch.dtype))
        return pose_batch + 0.5 * delta_batch

      checkpoint = {"name": "caller-owned-test-checkpoint", "revision": 3}
      model = adapter(predictor=predictor, checkpoint=checkpoint)
      out = model.predict(pose, delta)

      with self.subTest(adapter=adapter.__name__):
        self.assertEqual(out.shape, (4,))
        np.testing.assert_allclose(out, pose + 0.5 * delta, atol=1e-12)
        self.assertEqual(seen, [((1, 4), (1, 4), np.dtype(float), np.dtype(float))])
        self.assertIs(model.checkpoint, checkpoint)

  def test_batch_prediction_and_pair_broadcasting_are_preserved(self):
    poses = np.array(
      [[0.0, 10.0, 20.0, 0.0], [30.0, 40.0, 50.0, 0.1], [60.0, 70.0, 80.0, 0.2]]
    )
    one_delta = np.array([1.0, -2.0, 3.0, 0.05])

    for adapter in self.ADAPTERS:
      calls = []

      def predictor(pose_batch, delta_batch):
        calls.append((pose_batch.copy(), delta_batch.copy()))
        return pose_batch + delta_batch

      out = adapter(predictor).predict(poses, one_delta)
      with self.subTest(adapter=adapter.__name__):
        self.assertEqual(out.shape, (3, 4))
        np.testing.assert_allclose(out, poses + one_delta, atol=1e-12)
        self.assertEqual(calls[0][0].shape, (3, 4))
        self.assertEqual(calls[0][1].shape, (3, 4))
        np.testing.assert_allclose(calls[0][1], np.repeat(one_delta[None, :], 3, axis=0))

  def test_batched_output_must_have_exactly_one_four_vector_per_input(self):
    wrong_outputs = (
      np.zeros(4),
      np.zeros((1, 3)),
      np.zeros((2, 4)),
      np.zeros((1, 4, 1)),
    )
    for wrong in wrong_outputs:
      model = LearnedDynamics(lambda pose_batch, delta_batch, value=wrong: value)
      with self.subTest(shape=wrong.shape), self.assertRaises(DynamicsAdapterError) as ctx:
        model.predict(np.zeros(4), np.zeros(4))
      self.assertIn(f"shape {wrong.shape}", str(ctx.exception))
      self.assertIn("expected exactly (1, 4)", str(ctx.exception))

  def test_non_numeric_output_is_an_adapter_error(self):
    model = LearnedDynamics(lambda pose_batch, delta_batch: [["not-a-number"] * 4])
    with self.assertRaises(DynamicsAdapterError) as ctx:
      model.predict(np.zeros(4), np.zeros(4))
    self.assertIn("numeric", str(ctx.exception))
    self.assertIn("(1, 4)", str(ctx.exception))

  def test_non_finite_predictions_are_refused_for_single_and_batch_calls(self):
    def non_finite(pose_batch, delta_batch):
      out = pose_batch + delta_batch
      out[0, 0] = np.nan
      return out

    for adapter in self.ADAPTERS:
      model = adapter(non_finite)
      for pose, delta in ((np.zeros(4), np.zeros(4)), (np.zeros((3, 4)), np.zeros((3, 4)))):
        with self.subTest(adapter=adapter.__name__, shape=pose.shape):
          with self.assertRaises(FloatingPointError) as ctx:
            model.predict(pose, delta)
          self.assertIn("not finite", str(ctx.exception))

  def test_finite_but_implausible_cartesian_predictions_are_refused(self):
    def implausible(pose_batch, delta_batch):
      out = pose_batch + delta_batch
      out[:, 2] = ResidualDynamics.PLAUSIBLE_MM + 1.0
      return out

    for adapter in self.ADAPTERS:
      with self.subTest(adapter=adapter.__name__):
        with self.assertRaises(FloatingPointError) as ctx:
          adapter(implausible).predict(np.zeros(4), np.zeros(4))
        self.assertIn("plausible range", str(ctx.exception))
        self.assertIn(f"{ResidualDynamics.PLAUSIBLE_MM:.0f} mm", str(ctx.exception))

  def test_the_plausible_range_boundary_itself_is_allowed(self):
    def boundary(pose_batch, delta_batch):
      out = np.zeros_like(pose_batch)
      out[:, :3] = ResidualDynamics.PLAUSIBLE_MM
      return out

    out = LearnedDynamics(boundary).predict(np.zeros(4), np.zeros(4))
    np.testing.assert_allclose(out[:3], ResidualDynamics.PLAUSIBLE_MM, atol=0.0)

  def test_an_async_predictor_is_refused_without_leaking_a_coroutine(self):
    async def asynchronous(pose_batch, delta_batch):
      return pose_batch + delta_batch

    with self.assertRaises(DynamicsAdapterError) as ctx:
      LearnedDynamics(asynchronous).predict(np.zeros(4), np.zeros(4))
    self.assertIn("synchronous", str(ctx.exception))

  def test_a_non_callable_predictor_is_refused_at_construction(self):
    with self.assertRaises(TypeError) as ctx:
      LearnedDynamics(predictor=np.zeros((1, 4)))
    self.assertIn("callable", str(ctx.exception))

  def test_checkpoint_metadata_without_a_callable_is_not_an_adapter(self):
    for adapter in self.ADAPTERS:
      model = adapter(checkpoint={"path": "weights.bin"})
      with self.subTest(adapter=adapter.__name__):
        with self.assertRaises(NotImplementedError) as ctx:
          model.predict(np.zeros(4), np.zeros(4))
        message = str(ctx.exception)
        self.assertIn(model.model, message)
        self.assertIn("Checkpoint metadata was recorded", message)
        self.assertIn("installed package alone is not an adapter", message)
        self.assertIn("predictor(pose_batch, delta_batch)", message)

  def test_the_adapters_satisfy_the_dynamics_model_protocol_before_use(self):
    for model in (VideoWorldModel(), LearnedDynamics(), KinematicDynamics(), ResidualDynamics()):
      self.assertIsInstance(model, DynamicsModel)


if __name__ == "__main__":
  unittest.main()
