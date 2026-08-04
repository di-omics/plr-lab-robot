"""Tests for plr_lr.learn.bench.report: all sixteen refusals, each on a real input.

Every rule R1 through R16 is fired here by constructing the situation it exists for.
A refusal that no call can reach is documentation rather than a rule, and the first
time a documented-only refusal would have mattered is the first time somebody finds
out it does not work.

Two of the sixteen are enforced by absence rather than by a raise, and both are
tested by walking the source:

* **R9** forbids a percentile, a quantile or a max-over-frames selection anywhere in
  the benchmark package, in :mod:`plr_lr.learn.selective` and in the published
  single-file copy ``scripts/score_standalone.py``. That is the portfolio's own
  documented way of manufacturing a confident wrong number, and the two files outside
  the package are the ones that actually hold AURC, the risk-coverage curve and the
  frozen threshold, so policing only the package would leave the arithmetic itself
  unguarded.
* **The command line has no hardware flag on any verb.** This package has exactly
  one hardware permission gate, on the rollout call, and it needs three independent
  conditions that all default to off. A second gate here would be the one somebody
  finds first.

Two more tests here are about the trust horizon ``k*`` rather than about a printed
cell. ``k*`` is what :meth:`plr_lr.learn.screen.Screen.check` refuses
``HORIZON_EXCEEDED`` above, so it decides how many actions the arm runs open-loop
before it looks again. One test holds a predictor fixed and turns its abstention rate
up, and requires ``k*`` not to grow. The other requires the command line to refuse
outright when no horizon was trustworthy, rather than building a screen on a default.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import unittest
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from plr_lr.learn.bench import commands, controls, metrics, packio, report, runner
from plr_lr.learn.bench.controls import evaluate_rows
from plr_lr.learn.bench.packio import (
  FixtureMismatch,
  advance,
  builtin_demo_pack,
  require_fixture_match,
)
from plr_lr.learn.bench.report import (
  CONFOUNDED_BY_CLAMP,
  FIXTURE_MISMATCH,
  NOT_ACTION_CONDITIONED,
  NULL_CELL,
  REFUSAL_RULES,
  RULE_IDS,
  THRESHOLD_SENSITIVE,
  ArmCorrelation,
  BenchReport,
  BenchRow,
  Refusal,
  SimulatedBasis,
  TableNotComparable,
  UnstampedTiming,
  abstention_cell,
  basis_label,
  compare,
  latency_claim,
  require_comparable,
  require_hardware_basis,
  require_interval,
  require_paired_ci,
  risk_cell,
  underpowered_refusal,
)
from plr_lr.learn.bench.runner import SeedPlan, risk_at_full_coverage, run_bench
from plr_lr.learn.bench.tasks import TASKS, TASKS_BY_ID
from plr_lr.learn.predict import (
  NO_TRAINING_DATA,
  FittedComponent,
  RolloutPrediction,
)
from plr_lr.learn.screen import NotCalibrated, Screen
from plr_lr.learn.selective import (
  EpisodeOutcome,
  FrozenThreshold,
  MetricRefused,
  Rate,
  RiskCoverageCurve,
  paired_bootstrap_ci,
  score,
)

BENCH_DIR = Path(report.__file__).parent
LEARN_DIR = BENCH_DIR.parent
REPO_ROOT = LEARN_DIR.parent.parent

#: The two files R9 has to cover that are not in the bench package. AURC, the
#: risk-coverage curve and the frozen threshold live in ``selective.py``, and
#: ``score_standalone.py`` is the published single-file copy of the same scorer, so a
#: percentile appearing in either would be a percentile in the arithmetic itself
#: rather than in a module that imports it.
R9_EXTRA_SOURCES: Tuple[Path, ...] = (
  LEARN_DIR / "selective.py",
  REPO_ROOT / "scripts" / "score_standalone.py",
)

GATE_N = 40
THRESHOLD = FrozenThreshold(value=0.5, calibrated_on="tests", n_calibration=30)
SCREEN = Screen(
  theta=0.5,
  trust_horizon=1,
  max_frame_age_s=1.0,
  calibrated_on="tests: a hand-written threshold, declared as one",
  n_calibration=30,
)
SEEDS = SeedPlan(epoch="test-epoch")


def _row(
  task_id: str = "S3",
  horizon: int = 1,
  outcomes: Optional[Sequence[EpisodeOutcome]] = None,
  n: int = 40,
  action_conditioned: bool = True,
  refusals: Tuple[Refusal, ...] = (),
) -> BenchRow:
  """One row, assembled from outcomes rather than from a run, so it can be crafted."""
  if outcomes is None:
    outcomes = [
      EpisodeOutcome(episode_id=f"e{i}", hit=i % 5 != 0, confidence=0.9, unknowable=i % 5 == 0)
      for i in range(n)
    ]
  selective = score(outcomes, THRESHOLD, task_id=task_id, horizon=horizon, n_permutations=50)
  errors = [0.5 for _ in outcomes]
  hits = [int(o.hit) for o in outcomes]
  return BenchRow(
    task_id=task_id,
    horizon=horizon,
    seed_epoch="test-epoch",
    pack_id="test-pack",
    split="dev",
    basis="simulated",
    tolerance=tuple(TASKS_BY_ID[task_id].tolerance),
    n=len(outcomes),
    selective=selective,
    tpe=metrics.TolerancePredictionError.of(errors),
    gap=metrics.success_gap(task_id, horizon, hits, hits, seed=0, n_resamples=100),
    compute=metrics.ComputeDeclaration.of([1] * len(outcomes), [0.0] * len(outcomes)),
    pixel=metrics.PixelColumn(),
    overlap=metrics.training_overlap([o.episode_id for o in outcomes], set()),
    action_conditioned=action_conditioned,
    refusals=refusals,
  )


class _TunedPredictor:
  """One fixed accuracy profile, and a separate dial for how much of it is withheld.

  Built for the trust-horizon tests, where the whole question is what changes when a
  model abstains more. Its answer for a given episode at a given horizon never
  depends on ``abstain_share``: it drives the true transition under whatever actions
  it was handed, so it clears the action-conditioning gate, and it spoils a fixed,
  deterministic share of episodes by a wide margin so the miss rate at each horizon
  is the number ``miss_share`` names. ``abstain_share`` then decides only whether an
  episode it was about to get wrong is reported as a miss or withheld as a refusal.

  It never abstains at the shortest horizon. That keeps the gate, which runs there,
  identical across every setting of the dial, so a change in the trust horizon cannot
  be blamed on a different gate outcome. It is also what a real model does: the
  confidence decays with the horizon, so the refusals appear at the far end.

  Its confidence rises with the episode's rank and the spoiled episodes are the lowest
  ranks, so it is confident when it is right, which is what makes withholding its
  least confident answers the flattering move it is.
  """

  name = "tuned"
  license_text = "MIT, tests"

  def __init__(
    self,
    task: Any,
    pack: Any,
    miss_share: Dict[int, float],
    abstain_share: float,
  ) -> None:
    self.task = task
    self.state_dim = int(task.state_dim)
    self.horizon = int(task.max_horizon)
    self.action_space = task.action_space
    self.training_fingerprints = frozenset({NO_TRAINING_DATA})
    self.fitted_components: Tuple[Any, ...] = ()
    self.miss_share = {int(k): float(v) for k, v in miss_share.items()}
    self.abstain_share = float(abstain_share)
    episodes = pack.for_task(task.task_id)
    self._by_id = {e.episode_id: e for e in episodes}
    self._rank = {e.episode_id: i for i, e in enumerate(episodes)}
    self._n = len(episodes)

  def predict(self, request: Any) -> RolloutPrediction:
    horizon = int(request.horizon)
    episode = self._by_id[request.episode_id]
    position = (self._rank[request.episode_id] * 100) // self._n
    miss_cut = int(100 * self.miss_share[horizon])
    miss = position < miss_cut
    share = 0.0 if horizon == int(self.task.horizons[0]) else self.abstain_share
    if miss and position < int(miss_cut * share):
      return RolloutPrediction.abstained(horizon, self.state_dim)
    terminal = np.asarray(
      advance(
        self.task.task_id,
        episode.state0,
        np.asarray(request.actions, dtype=float),
        episode.coefficients or {},
      ),
      dtype=float,
    )
    if miss:
      terminal = terminal + 1000.0
    return RolloutPrediction(
      state=np.tile(terminal, (horizon, 1)),
      confidence=0.5 + 0.004 * position,
      abstain=False,
      flag=False,
      readout="tuned",
      n_forward_passes=1,
      wall_clock_s=0.0,
    )


class _NormPredictor:
  """A predictor that genuinely reads its action argument, minus one declaration.

  Its terminal state moves with the Euclidean norm of the commanded translation, so
  it clears the action-conditioning gate from every direction rather than from most
  of them. That matters here: these fixtures exist to fire the provenance refusals,
  and a predictor that tripped the gate first would fire the wrong one.
  """

  name = "norm predictor"
  license_text = "MIT, tests"

  def __init__(
    self,
    state_dim: int,
    horizon: int,
    fingerprints: Optional[FrozenSet[str]] = None,
    components: Tuple[Any, ...] = (),
  ) -> None:
    self.state_dim = int(state_dim)
    self.horizon = int(horizon)
    self.action_space = "task_scalar"
    self.training_fingerprints = fingerprints
    self.fitted_components = components

  def predict(self, request: Any) -> RolloutPrediction:
    actions = np.asarray(request.actions, dtype=float)
    magnitude = float(np.sum(np.sqrt(np.sum(actions[:, :3] ** 2, axis=1))))
    terminal = np.asarray(request.state, dtype=float) + magnitude
    return RolloutPrediction(
      state=np.tile(terminal, (int(request.horizon), 1)),
      confidence=0.8,
      abstain=False,
      flag=False,
      readout="norm",
      n_forward_passes=1,
      wall_clock_s=0.0,
    )


class RuleTableTest(unittest.TestCase):
  def test_there_are_exactly_sixteen_rules_and_they_are_r1_to_r16(self) -> None:
    self.assertEqual(len(REFUSAL_RULES), 16)
    self.assertEqual(RULE_IDS, tuple(f"R{i}" for i in range(1, 17)))

  def test_every_rule_carries_a_printed_token_and_a_reason(self) -> None:
    for rule, label, why in REFUSAL_RULES:
      self.assertTrue(label, rule)
      self.assertTrue(why, rule)

  def test_an_invented_rule_identifier_is_refused(self) -> None:
    with self.assertRaises(ValueError):
      Refusal("R99", "MADE-UP", "not in the published table")


class R1ZeroCoverageTest(unittest.TestCase):
  def test_risk_at_zero_coverage_prints_null_and_never_zero(self) -> None:
    curve = RiskCoverageCurve.from_outcomes((1, 0, 1, 1), (0.9, 0.8, 0.7, 0.6))
    self.assertEqual(risk_cell(curve, 0), NULL_CELL)
    self.assertNotEqual(risk_cell(curve, 0), "0.0000")

  def test_a_missing_curve_also_prints_null(self) -> None:
    self.assertEqual(risk_cell(None, 3), NULL_CELL)

  def test_a_real_coverage_prints_a_number(self) -> None:
    curve = RiskCoverageCurve.from_outcomes((1, 0, 1, 1), (0.9, 0.8, 0.7, 0.6))
    self.assertEqual(risk_cell(curve, 4), "0.2500")


class R2UnderpoweredTest(unittest.TestCase):
  def test_below_thirty_nothing_prints(self) -> None:
    refusal = underpowered_refusal(10, "S1@1")
    self.assertIsNotNone(refusal)
    self.assertEqual(refusal.rule, "R2")
    self.assertEqual(refusal.label, "UNDERPOWERED")
    self.assertTrue(refusal.blocking)

  def test_between_thirty_and_two_hundred_it_prints_stamped_and_unrankable(self) -> None:
    refusal = underpowered_refusal(60, "S1@1")
    self.assertEqual(refusal.label, "PROVISIONAL")
    self.assertFalse(refusal.blocking)

  def test_at_two_hundred_there_is_no_refusal(self) -> None:
    self.assertIsNone(underpowered_refusal(200, "S1@1"))

  def test_a_row_below_the_rankable_floor_is_not_rankable(self) -> None:
    self.assertFalse(_row(n=40).rankable)


class R3IntervalTest(unittest.TestCase):
  def test_a_bare_float_cannot_enter_a_report_as_a_rate(self) -> None:
    with self.assertRaises(MetricRefused):
      require_interval(0.93, "TolAcc")

  def test_a_rate_carrying_its_counts_passes(self) -> None:
    rate = require_interval(Rate(9, 30, name="TolAcc"), "TolAcc")
    self.assertIsNotNone(rate.interval)

  def test_a_delta_without_a_paired_interval_is_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      require_paired_ci(0.12, "SuccessGap")

  def test_a_paired_interval_passes(self) -> None:
    ci = paired_bootstrap_ci([1.0] * 30, [0.0] * 30, n_resamples=100, seed=0)
    self.assertIs(require_paired_ci(ci, "SuccessGap"), ci)


class R4InterventionTest(unittest.TestCase):
  """A model that ignores its actions gets a refusal, not a low score."""

  @classmethod
  def setUpClass(cls) -> None:
    from plr_lr.learn.predict import BlindPredictor

    pack = builtin_demo_pack(n_episodes=GATE_N)
    task = TASKS_BY_ID["S3"]
    blind = BlindPredictor(state_dim=task.state_dim, horizon=task.max_horizon)
    cls.report = run_bench(
      blind, (task,), pack, SCREEN, SEEDS, n_permutations=20, n_resamples=50
    )

  def test_no_row_is_emitted_at_all(self) -> None:
    self.assertEqual(self.report.rows, ())

  def test_the_refusal_is_r4(self) -> None:
    self.assertEqual([r.rule for r in self.report.refusals], ["R4"])
    self.assertIn("INTERVENTION-REFUSED", self.report.refusals[0].label)

  def test_the_report_is_not_valid(self) -> None:
    self.assertFalse(self.report.valid)

  def test_the_summary_says_a_refusal_and_not_a_low_score(self) -> None:
    text = self.report.summary()
    self.assertIn("refusal", text)
    self.assertIn("no rows", text)


class R5AbstentionPairTest(unittest.TestCase):
  def test_the_credit_rate_is_unreachable_without_the_false_alarm_rate(self) -> None:
    row = _row()
    self.assertFalse(hasattr(row.selective.credit, "acr"))

  def test_the_printed_cell_carries_both(self) -> None:
    cell, refusal = abstention_cell(_row().selective)
    self.assertIsNone(refusal)
    self.assertIn("ACR", cell)
    self.assertIn("FAR", cell)
    self.assertIn("RefusalRecall", cell)


class R6BandTest(unittest.TestCase):
  def test_abstention_metrics_outside_the_band_are_refused(self) -> None:
    outcomes = [
      EpisodeOutcome(episode_id=f"e{i}", hit=True, confidence=0.9, unknowable=False)
      for i in range(40)
    ]
    row = _row(outcomes=outcomes)
    cell, refusal = abstention_cell(row.selective)
    self.assertIsNotNone(refusal)
    self.assertEqual(refusal.rule, "R6")
    self.assertIn("REFUSED", cell)

  def test_inside_the_band_they_print(self) -> None:
    _cell, refusal = abstention_cell(_row().selective)
    self.assertIsNone(refusal)


class R7AndR8ProvenanceTest(unittest.TestCase):
  """A checkpoint with no declared training set, and an undeclared fitted probe."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.pack = builtin_demo_pack(n_episodes=GATE_N)
    cls.task = TASKS_BY_ID["S3"]

  def _run(self, predictor: Any) -> BenchReport:
    return run_bench(
      predictor, (self.task,), self.pack, SCREEN, SEEDS, n_permutations=20, n_resamples=50
    )

  def test_an_undeclared_training_set_withholds_every_row(self) -> None:
    predictor = _NormPredictor(self.task.state_dim, self.task.max_horizon, fingerprints=None)
    out = self._run(predictor)
    self.assertEqual(out.rows, ())
    self.assertEqual([r.rule for r in out.refusals], ["R7"])
    self.assertIn("PROVENANCE: UNDECLARED", out.refusals[0].label)

  def test_an_undeclared_fitted_component_withholds_every_row(self) -> None:
    predictor = _NormPredictor(
      self.task.state_dim,
      self.task.max_horizon,
      fingerprints=frozenset({NO_TRAINING_DATA}),
      components=(
        FittedComponent(
          name="latent_to_uL_linear_probe",
          fit_on_split="UNDECLARED",
          n_examples=100,
          description="a probe whose split is not named",
        ),
      ),
    )
    out = self._run(predictor)
    self.assertEqual(out.rows, ())
    self.assertEqual([r.rule for r in out.refusals], ["R8"])

  def test_a_declared_probe_does_not_withhold(self) -> None:
    predictor = _NormPredictor(
      self.task.state_dim,
      self.task.max_horizon,
      fingerprints=frozenset({NO_TRAINING_DATA}),
      components=(
        FittedComponent(
          name="latent_to_uL_linear_probe",
          fit_on_split="dev",
          n_examples=100,
          description="a probe whose split is named",
        ),
      ),
    )
    out = self._run(predictor)
    self.assertEqual([r.rule for r in out.refusals], [])


def r9_guarded_sources() -> List[Path]:
  """Every file R9's AST half is walked over: the package plus the two that own it.

  The measured gap this closes is a scope gap rather than a live percentile. The
  guard used to glob ``plr_lr/learn/bench/*.py`` only, which is every module that
  *imports* the risk-coverage arithmetic and not the one that *defines* it. Both
  files below hold clean today; the point is that nothing mechanically stopped a
  fitted cut point from appearing in either tomorrow.
  """
  return sorted(BENCH_DIR.glob("*.py")) + list(R9_EXTRA_SOURCES)


class R9MeanAndSelectionTest(unittest.TestCase):
  def test_the_mean_of_the_tolerance_error_is_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      _ = metrics.TolerancePredictionError.of([1.0, 2.0, 1000.0]).mean

  def test_the_guard_covers_the_two_files_outside_the_package(self) -> None:
    """A guard whose file list has gone stale is a guard over nothing."""
    covered = r9_guarded_sources()
    for path in R9_EXTRA_SOURCES:
      self.assertIn(path, covered)
      self.assertTrue(path.is_file(), f"{path} is missing, so R9 covers nothing there")
    self.assertGreater(len(covered), len(R9_EXTRA_SOURCES))

  def test_no_scoring_source_reads_a_percentile_or_a_selection(self) -> None:
    banned = ("percentile", "quantile", "argmax", "argmin")
    for path in r9_guarded_sources():
      tree = ast.parse(path.read_text(encoding="ascii"))
      for node in ast.walk(tree):
        name = None
        if isinstance(node, ast.Attribute):
          name = node.attr
        elif isinstance(node, ast.Name):
          name = node.id
        if name is None:
          continue
        for token in banned:
          self.assertNotIn(token, name.lower(), f"{path.name}: {name}")


class R10NoAggregateTest(unittest.TestCase):
  def test_there_is_no_cross_task_aggregate_scalar(self) -> None:
    out = BenchReport(
      model="m", pack_id="p", seed_epoch="e", split="dev", basis="simulated"
    )
    with self.assertRaises(MetricRefused):
      _ = out.aggregate_score

  def test_the_dict_says_the_aggregate_was_refused(self) -> None:
    out = BenchReport(
      model="m", pack_id="p", seed_epoch="e", split="dev", basis="simulated"
    )
    self.assertIn("REFUSED", out.to_dict()["aggregate_score"])

  def test_the_summary_says_so_in_words(self) -> None:
    out = BenchReport(
      model="m", pack_id="p", seed_epoch="e", split="dev", basis="simulated"
    )
    self.assertIn("no cross-task aggregate", out.summary())


class R11ComparabilityTest(unittest.TestCase):
  def test_rows_at_different_horizons_are_not_one_table(self) -> None:
    with self.assertRaises(TableNotComparable) as caught:
      require_comparable((_row(horizon=1), _row(horizon=2)))
    self.assertIn("horizon", str(caught.exception))

  def test_the_mismatched_column_is_named(self) -> None:
    left = _row()
    right = _row(task_id="S2")
    with self.assertRaises(TableNotComparable) as caught:
      require_comparable((left, right))
    self.assertIn("tolerance", str(caught.exception))

  def test_two_matching_rows_are_comparable(self) -> None:
    require_comparable((_row(), _row()))

  def test_one_row_is_always_comparable_with_itself(self) -> None:
    require_comparable((_row(),))


class R12LatencyTest(unittest.TestCase):
  def test_a_host_arrival_clock_cannot_support_a_latency_claim(self) -> None:
    with self.assertRaises(UnstampedTiming) as caught:
      latency_claim("host_arrival", 10.0, 20.0, 30.0)
    self.assertIn("UNSTAMPED-TIMING", str(caught.exception))

  def test_a_rendered_clock_cannot_either(self) -> None:
    with self.assertRaises(UnstampedTiming):
      latency_claim("rendered", 10.0, 20.0, 30.0)

  def test_the_device_stream_can_and_prints_no_mean(self) -> None:
    text = latency_claim("device_30000", 10.0, 20.0, 30.0, n_stalls=3)
    self.assertIn("p99", text)
    self.assertIn("3 stalls", text)
    self.assertIn("No mean", text)


class R13BasisTest(unittest.TestCase):
  def test_a_pack_with_no_device_serial_reads_simulated(self) -> None:
    pack = builtin_demo_pack(n_episodes=GATE_N)
    self.assertEqual(basis_label(pack), "simulated")

  def test_asking_a_simulated_run_for_a_hardware_label_is_refused(self) -> None:
    pack = builtin_demo_pack(n_episodes=GATE_N)
    with self.assertRaises(SimulatedBasis) as caught:
      require_hardware_basis(pack)
    self.assertIn("bench-validated", str(caught.exception))

  def test_every_row_prints_its_basis(self) -> None:
    self.assertIn("basis: simulated", _row().format())


class R14ThresholdSensitiveTest(unittest.TestCase):
  """When the frozen-threshold order and the threshold-free order disagree."""

  @classmethod
  def setUpClass(cls) -> None:
    # A: every confidence above theta, four in five correct. High ADA, and a
    # confidence channel that carries nothing, so a poor excess AURC.
    a_outcomes = [
      EpisodeOutcome(episode_id=f"e{i}", hit=i % 5 != 0, confidence=0.9, unknowable=i % 5 == 0)
      for i in range(40)
    ]
    # B: half correct, ordered perfectly, but every confidence below theta. A
    # perfect excess AURC and an ADA that only counts the deferrals it got right.
    b_outcomes = [
      EpisodeOutcome(
        episode_id=f"e{i}",
        hit=i < 20,
        confidence=0.45 - i * 0.005,
        unknowable=i % 5 == 0,
      )
      for i in range(40)
    ]
    cls.a = _row(outcomes=a_outcomes)
    cls.b = _row(outcomes=b_outcomes)

  def test_the_two_orderings_really_do_disagree(self) -> None:
    self.assertLess(self.b.e_aurc, self.a.e_aurc)
    self.assertGreater(self.a.ada_value, self.b.ada_value)

  def test_the_ranking_prints_threshold_sensitive(self) -> None:
    comparison = compare(self.a, self.b)
    self.assertTrue(comparison.threshold_sensitive)
    self.assertIn(THRESHOLD_SENSITIVE, comparison.format())

  def test_two_rows_that_agree_are_ranked(self) -> None:
    comparison = compare(self.a, self.a)
    self.assertFalse(comparison.threshold_sensitive)


class R15ClampConfoundTest(unittest.TestCase):
  def test_a_disagreement_beyond_the_interval_withholds_the_headline(self) -> None:
    delta = paired_bootstrap_ci([1.0] * 30, [0.0] * 30, n_resamples=100, seed=0)
    correlation = ArmCorrelation(
      rho_all=0.62, rho_unclamped=0.03, delta=delta, n_all=30, n_unclamped=18
    )
    self.assertTrue(correlation.confounded)
    self.assertIn(CONFOUNDED_BY_CLAMP, correlation.headline())

  def test_agreement_inside_the_interval_prints_the_correlation(self) -> None:
    delta = paired_bootstrap_ci([1.0] * 30, [1.0] * 30, n_resamples=100, seed=0)
    correlation = ArmCorrelation(
      rho_all=0.62, rho_unclamped=0.60, delta=delta, n_all=30, n_unclamped=28
    )
    self.assertFalse(correlation.confounded)
    self.assertNotIn(CONFOUNDED_BY_CLAMP, correlation.headline())

  def test_a_missing_subset_correlation_is_treated_as_confounded(self) -> None:
    delta = paired_bootstrap_ci([1.0] * 30, [1.0] * 30, n_resamples=100, seed=0)
    correlation = ArmCorrelation(
      rho_all=0.62, rho_unclamped=None, delta=delta, n_all=30, n_unclamped=0
    )
    self.assertTrue(correlation.confounded)


class R16FixtureMismatchTest(unittest.TestCase):
  def test_a_submission_on_other_data_scores_nothing(self) -> None:
    pack = builtin_demo_pack(n_episodes=GATE_N)
    with self.assertRaises(FixtureMismatch) as caught:
      require_fixture_match(pack, ["0" * 64])
    self.assertIn(FIXTURE_MISMATCH, str(caught.exception))

  def test_the_token_is_the_one_the_rule_table_declares(self) -> None:
    labels = {rule: label for rule, label, _why in REFUSAL_RULES}
    self.assertEqual(labels["R16"], FIXTURE_MISMATCH)


class HeadlineWithholdingTest(unittest.TestCase):
  """The number is not printed beside the control that refutes it."""

  def test_a_model_that_is_not_action_conditioned_gets_no_headline(self) -> None:
    row = _row(action_conditioned=False)
    self.assertTrue(row.headline_withheld)
    self.assertIn(NOT_ACTION_CONDITIONED, row.headline())
    # The refutation prints and the number does not. Printing both is how the
    # number gets quoted and the refutation gets dropped.
    self.assertNotIn("lower is better", row.headline())
    self.assertNotIn("{0:.4f}".format(row.e_aurc), row.headline())

  def test_an_action_conditioned_model_gets_one(self) -> None:
    row = _row(action_conditioned=True)
    self.assertFalse(row.headline_withheld)
    self.assertIn("lower is better", row.headline())
    self.assertIn("{0:.4f}".format(row.e_aurc), row.headline())

  def test_a_blocking_refusal_also_withholds_it(self) -> None:
    row = _row(refusals=(underpowered_refusal(10, "S3@1"),))
    self.assertTrue(row.headline_withheld)
    self.assertNotIn("lower is better", row.headline())

  def test_the_row_prints_declared_compute_and_the_pixel_column(self) -> None:
    text = _row().format()
    self.assertIn("declared compute (NOT RANKED)", text)
    self.assertIn("pixel (NOT RANKED)", text)


class ReportValidityTest(unittest.TestCase):
  def test_a_gate_that_never_ran_is_not_a_pass(self) -> None:
    out = BenchReport(
      model="m", pack_id="p", seed_epoch="e", split="dev", basis="simulated",
      rows=(_row(),),
    )
    self.assertFalse(out.gate_passed)
    self.assertFalse(out.valid)

  def test_controls_that_never_ran_are_not_a_pass(self) -> None:
    out = BenchReport(
      model="m", pack_id="p", seed_epoch="e", split="dev", basis="simulated",
      rows=(_row(),),
    )
    self.assertFalse(out.controls_passed)
    self.assertIn("NOT RUN", out.summary())

  def test_the_json_form_round_trips(self) -> None:
    out = BenchReport(
      model="m", pack_id="p", seed_epoch="e", split="dev", basis="simulated",
      rows=(_row(),),
    )
    raw = json.loads(out.to_json())
    self.assertEqual(raw["benchmark"], "DeferLab v0.1")
    self.assertEqual(len(raw["rows"]), 1)
    self.assertEqual(raw["rows"][0]["risk_at_zero_coverage"], NULL_CELL)


class NoHardwareFlagTest(unittest.TestCase):
  """The benchmark runner has no hardware flag on any verb, AST-asserted."""

  def setUp(self) -> None:
    self.source = Path(commands.__file__).read_text(encoding="ascii")
    self.tree = ast.parse(self.source)

  def test_no_option_string_names_hardware_an_arm_or_an_address(self) -> None:
    banned = ("hardware", "arm", "i-am-at", "robot", "host", "ip", "port", "serial")
    for node in ast.walk(self.tree):
      if not isinstance(node, ast.Call):
        continue
      func = node.func
      if not isinstance(func, ast.Attribute) or func.attr != "add_argument":
        continue
      for argument in node.args:
        self.assertIsInstance(argument, ast.Constant)
        option = str(argument.value).lower()
        for token in banned:
          self.assertNotIn(token, option, f"{option} would be a second hardware gate")

  def test_no_permission_token_appears_anywhere_in_the_package(self) -> None:
    """Neither of the other two conditions on the one hardware gate is reachable here.

    The third condition is the command-line flag, and it is checked as an argparse
    option above rather than as raw text, because the module docstring names it in
    order to explain why it is absent.
    """
    for path in sorted(BENCH_DIR.glob("*.py")):
      text = path.read_text(encoding="ascii")
      self.assertNotIn("allow_hardware", text, path.name)
      self.assertNotIn("PLR_LR_ARM_HOST", text, path.name)

  def test_every_declared_verb_exists_and_none_of_them_is_a_hardware_verb(self) -> None:
    self.assertEqual(
      sorted(commands.VERBS),
      ["controls", "probe", "receipt", "run", "score", "screen-fit"],
    )

  def test_the_parser_rejects_an_invented_hardware_flag(self) -> None:
    parser = commands.build_parser()
    with self.assertRaises(SystemExit):
      parser.parse_args(["run", "--i-am-at-the-arm"])


#: The miss rate the trust-horizon fixture is built to. TRUST_RISK is 0.25, so the
#: honest trust horizon for this profile is 1: acceptable at one step, and shown to be
#: unacceptable at two and at three.
TRUST_TASK = TASKS_BY_ID["S3"]
TRUST_MISS = {1: 0.10, 2: 0.40, 3: 0.60}
TRUST_N = 200


def _ranked_only_risk(rows: Any) -> Optional[float]:
  """The pre-fix denominator, kept verbatim so a test can show what it would report.

  Risk over the episodes the model agreed to answer. That is selective risk at
  whatever coverage the model chose, and it is the arithmetic
  :func:`plr_lr.learn.bench.runner.risk_at_full_coverage` replaced.
  """
  hits, _confidences = rows.ranked()
  return None if not hits else 1.0 - sum(hits) / len(hits)


class TrustHorizonAbstentionTest(unittest.TestCase):
  """Abstaining must not buy a longer trust horizon.

  ``k*`` is not a display number. :meth:`plr_lr.learn.screen.Screen.check` refuses
  ``HORIZON_EXCEEDED`` above it, so it is the quantity that decides how many actions
  the arm executes open-loop before it re-observes. If the risk it is measured
  against is taken over only the episodes the model chose to answer, then refusing
  more shortens the denominator faster than it shortens the numerator and the model
  is handed permission to run longer chunks for having answered less.
  """

  @classmethod
  def setUpClass(cls) -> None:
    cls.pack = builtin_demo_pack(tasks=(TRUST_TASK,), n_episodes=TRUST_N)

  def _rows(self, share: float, horizon: int) -> Any:
    predictor = _TunedPredictor(TRUST_TASK, self.pack, TRUST_MISS, share)
    return evaluate_rows(predictor, TRUST_TASK, self.pack, int(horizon))

  def _k_star(self, share: float, risk_of: Any) -> Optional[int]:
    risk = {
      int(horizon): risk_of(self._rows(share, int(horizon)))
      for horizon in TRUST_TASK.horizons
    }
    return metrics.trust_horizon(risk)

  def test_the_fixture_really_does_abstain(self) -> None:
    """Otherwise every assertion below would hold for the emptiest of reasons."""
    answered = self._rows(0.0, 2)
    withheld = self._rows(0.75, 2)
    self.assertEqual(answered.n_ranked, answered.n)
    self.assertLess(withheld.n_ranked, withheld.n)
    self.assertEqual(withheld.n, answered.n)

  def test_an_abstention_stays_in_the_denominator(self) -> None:
    answered = self._rows(0.0, 2)
    withheld = self._rows(0.75, 2)
    self.assertAlmostEqual(
      risk_at_full_coverage(withheld), risk_at_full_coverage(answered), places=12
    )
    self.assertAlmostEqual(risk_at_full_coverage(answered), 1.0 - answered.tol_acc, 12)

  def test_a_model_that_refuses_everything_is_at_risk_one_and_not_at_no_risk(self) -> None:
    """The all-abstain null must reach the bound from above, not be waved past it.

    Under the pre-fix denominator this row had no ranked episodes at all, so its risk
    was ``None`` and the horizon walk stopped on a missing measurement. It now stops
    on a measured risk of 1.0, which is the same refusal for the honest reason.
    """
    rows = evaluate_rows(
      controls.AlwaysAbstainPredictor(TRUST_TASK, 1), TRUST_TASK, self.pack, 1
    )
    self.assertEqual(rows.n_ranked, 0)
    self.assertEqual(risk_at_full_coverage(rows), 1.0)
    self.assertIsNone(metrics.trust_horizon({1: risk_at_full_coverage(rows)}))

  def test_a_task_with_no_episodes_has_no_risk_rather_than_a_flattering_one(self) -> None:
    empty = controls.EpisodeRows(
      task_id=TRUST_TASK.task_id,
      horizon=1,
      episode_ids=(),
      hits=(),
      loose_hits=(),
      errors=(),
      confidences=(),
      abstained=(),
      unknowable=(),
      flagged=(),
      n_forward_passes=(),
      wall_clock_s=(),
    )
    self.assertIsNone(risk_at_full_coverage(empty))
    self.assertIsNone(metrics.trust_horizon({1: risk_at_full_coverage(empty)}))

  def test_abstaining_more_does_not_lengthen_the_trust_horizon(self) -> None:
    baseline = self._k_star(0.0, risk_at_full_coverage)
    self.assertEqual(baseline, 1)
    for share in (0.25, 0.5, 0.75, 1.0):
      self.assertEqual(
        self._k_star(share, risk_at_full_coverage),
        baseline,
        f"abstention share {share} moved the trust horizon",
      )

  def test_the_pre_fix_denominator_would_have_paid_for_the_abstention(self) -> None:
    """The fixture is known to expose the defect, so the test above is not vacuous.

    A guard built on a fixture that could not have failed is a guard that establishes
    nothing, and this is the assertion that says this one could have.
    """
    self.assertEqual(self._k_star(0.0, _ranked_only_risk), 1)
    self.assertEqual(self._k_star(0.75, _ranked_only_risk), 2)

  def test_the_whole_run_gives_the_abstainer_no_extra_horizon(self) -> None:
    """End to end through run_bench, gate and controls included."""
    measured = {}
    for share in (0.0, 0.75):
      predictor = _TunedPredictor(TRUST_TASK, self.pack, TRUST_MISS, share)
      out = run_bench(
        predictor,
        (TRUST_TASK,),
        self.pack,
        SCREEN,
        SEEDS,
        n_permutations=50,
        n_resamples=200,
      )
      self.assertEqual([r.rule for r in out.refusals], [], f"share {share}")
      self.assertTrue(out.controls.passed)
      measured[share] = out.trust_horizons[TRUST_TASK.task_id]
    self.assertEqual(measured[0.0], 1)
    self.assertEqual(measured[0.75], measured[0.0])


class ScreenRefusesWithoutATrustHorizonTest(unittest.TestCase):
  """``None`` from ``metrics.trust_horizon`` means no screen, not a screen of one.

  ``Screen.__post_init__`` already says a trust horizon below one is a reason not to
  build a screen rather than a screen that admits nothing. Defaulting to one is the
  same mistake wearing a legal value: it is a live, calibrated-looking screen that
  admits every single-step prediction on a task nothing was trustworthy at, and it
  prints the calibration string a measured screen would.
  """

  #: Above TRUST_RISK at every horizon, so no horizon is trustworthy, while sixty
  #: percent of episodes still land inside tolerance and supply calibration successes.
  #: That separates this refusal from the too-few-successes one.
  MISS = 0.40

  def _predictors(self, pack: Any) -> Dict[str, Any]:
    return {
      task.task_id: _TunedPredictor(
        task, pack, {int(h): self.MISS for h in task.horizons}, 0.0
      )
      for task in TASKS
    }

  def test_no_trustworthy_horizon_refuses_the_screen(self) -> None:
    pack = builtin_demo_pack(n_episodes=60)
    with self.assertRaises(NotCalibrated) as caught:
      commands.fit_screen(pack, self._predictors(pack))
    self.assertIn("no trustworthy horizon", str(caught.exception))

  def test_the_refusal_names_the_horizon_and_not_the_sample_size(self) -> None:
    """The other NotCalibrated is about too few successes. This one is not that."""
    pack = builtin_demo_pack(n_episodes=60)
    with self.assertRaises(NotCalibrated) as caught:
      commands.fit_screen(pack, self._predictors(pack))
    self.assertNotIn("successful held-out episodes", str(caught.exception))

  def test_a_measured_horizon_still_builds_a_screen(self) -> None:
    """The refusal is about the measurement, not about abstention or about tasks."""
    pack = builtin_demo_pack(n_episodes=60)
    screen = commands.fit_screen(pack, runner.floor_predictors(TASKS))
    self.assertTrue(screen.is_calibrated)
    self.assertGreaterEqual(screen.trust_horizon, 1)

  def test_the_cli_refuses_rather_than_building_a_permissive_screen(self) -> None:
    argv = ["run", "--episodes", "60"]
    args = commands.build_parser().parse_args(argv)
    predictors = self._predictors(commands._calibration_pack(args))
    original = commands.floor_predictors
    commands.floor_predictors = lambda tasks=TASKS: predictors
    try:
      with self.assertRaises(SystemExit) as caught:
        commands.main(argv)
    finally:
      commands.floor_predictors = original
    message = str(caught.exception)
    self.assertIn("no screen could be fitted", message)
    self.assertIn("no trustworthy horizon", message)

  def test_the_screen_fit_verb_reports_the_refusal_and_exits_non_zero(self) -> None:
    argv = ["screen-fit", "--episodes", "60"]
    args = commands.build_parser().parse_args(argv)
    predictors = self._predictors(commands._calibration_pack(args))
    original = commands.floor_predictors
    commands.floor_predictors = lambda tasks=TASKS: predictors
    buffer = io.StringIO()
    try:
      with contextlib.redirect_stdout(buffer):
        code = commands.main(argv)
    finally:
      commands.floor_predictors = original
    self.assertEqual(code, 1)
    printed = buffer.getvalue()
    self.assertIn("NOT CALIBRATED", printed)
    self.assertNotIn("trust horizon 1", printed)


class PackageBoundaryTest(unittest.TestCase):
  def test_the_bench_package_imports_no_optional_dependency(self) -> None:
    banned = ("torch", "cv2", "requests", "urllib", "socket", "scipy", "pandas")
    for path in sorted(BENCH_DIR.glob("*.py")):
      tree = ast.parse(path.read_text(encoding="ascii"))
      for node in ast.walk(tree):
        names: List[str] = []
        if isinstance(node, ast.Import):
          names = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
          names = [(node.module or "").split(".")[0]]
        for name in names:
          self.assertNotIn(name, banned, f"{path.name} imports {name}")

  def test_the_modules_in_the_manifest_all_exist(self) -> None:
    wanted = {
      "__init__.py", "__main__.py", "commands.py", "controls.py", "metrics.py",
      "packio.py", "report.py", "runner.py", "tasks.py",
    }
    self.assertTrue(wanted.issubset({p.name for p in BENCH_DIR.glob("*.py")}))

  def test_the_modules_are_the_ones_the_slice_owns(self) -> None:
    self.assertTrue(hasattr(runner, "run_bench"))
    self.assertTrue(hasattr(controls, "run_control_suite"))
    self.assertTrue(hasattr(packio, "SCENES_FORMAT_VERSION"))


if __name__ == "__main__":
  unittest.main()
