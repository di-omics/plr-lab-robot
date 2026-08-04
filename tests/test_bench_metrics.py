"""Tests for plr_lr.learn.bench.metrics.

Every expected number here is either hand-computed and written out as the
arithmetic that produced it, or it is a structural claim about a refusal. A metrics
test that computes its expectation with the code under test proves only that the
code is deterministic.

The load-bearing structural claim is that this module composes
:mod:`plr_lr.learn.selective` rather than re-deriving it. This repository has
already paid once for two modules growing their own copy of the same quantity and
then disagreeing, so there is a test asserting that no risk-coverage or excess-AURC
arithmetic is defined here at all.
"""

from __future__ import annotations

import ast
import math
import unittest
from pathlib import Path

from plr_lr.learn.bench import metrics, runner
from plr_lr.learn.bench.metrics import (
  BENCH_CORPUS,
  LOOSE_TOLERANCE,
  NOT_RANKED,
  NO_TRAINING_DATA,
  TRUST_RISK,
  ComputeDeclaration,
  LooseCriterionMissing,
  OverlapCount,
  PixelColumn,
  ScoringScope,
  TolerancePredictionError,
  auroc_not_ranked,
  ece_not_ranked,
  in_tolerance,
  loose_hit,
  loose_tolerance_for,
  success_gap,
  tolerance_error,
  training_overlap,
  trust_horizon,
)
from plr_lr.learn.bench.packio import builtin_demo_pack
from plr_lr.learn.bench.report import REFUSAL_RULES
from plr_lr.learn.bench.runner import SeedPlan, floor_predictor_for, run_bench
from plr_lr.learn.bench.tasks import TASKS
from plr_lr.learn.screen import Screen
from plr_lr.learn.selective import MetricRefused, Rate, RiskCoverageCurve

METRICS_SOURCE = Path(metrics.__file__)
RUNNER_SOURCE = Path(runner.__file__)

# The six-episode fixture, sorted by descending confidence. Losses in rank order
# are [0, 0, 1, 0, 1, 1], so N = 6 and e = 3.
#
#   risk(1) = 0/1 = 0
#   risk(2) = 0/2 = 0
#   risk(3) = 1/3
#   risk(4) = 1/4
#   risk(5) = 2/5
#   risk(6) = 3/6
#   AURC        = (0 + 0 + 1/3 + 1/4 + 2/5 + 1/2) / 6 = 1.48333.../6 = 0.2472222...
#   AURC_oracle = (0 + 0 + 0 + 1/4 + 2/5 + 3/6) / 6  = 1.15/6      = 0.1916666...
#   AURC_random = 3/6                                              = 0.5
#   E_AURC      = 0.2472222... - 0.1916666...        = 1/18        = 0.0555555...
SIX_CONFIDENCES = (0.9, 0.8, 0.7, 0.6, 0.5, 0.4)
SIX_HITS = (1, 1, 0, 1, 0, 0)
HAND_AURC = (0.0 + 0.0 + 1.0 / 3.0 + 1.0 / 4.0 + 2.0 / 5.0 + 1.0 / 2.0) / 6.0
HAND_AURC_ORACLE = (1.0 / 4.0 + 2.0 / 5.0 + 3.0 / 6.0) / 6.0
HAND_E_AURC = 1.0 / 18.0


class HandComputedCurveTest(unittest.TestCase):
  """The arithmetic above, against the curve the benchmark actually scores with."""

  def setUp(self) -> None:
    self.curve = RiskCoverageCurve.from_outcomes(SIX_HITS, SIX_CONFIDENCES, seed=0)

  def test_aurc_matches_the_hand_computation(self) -> None:
    self.assertAlmostEqual(self.curve.aurc, HAND_AURC, places=12)
    self.assertAlmostEqual(self.curve.aurc, 0.2472222222222222, places=12)

  def test_oracle_and_random_bands_match_the_hand_computation(self) -> None:
    self.assertAlmostEqual(self.curve.aurc_oracle, HAND_AURC_ORACLE, places=12)
    self.assertAlmostEqual(self.curve.aurc_random, 0.5, places=12)

  def test_excess_aurc_is_one_eighteenth(self) -> None:
    self.assertAlmostEqual(self.curve.e_aurc, HAND_E_AURC, places=12)

  def test_the_no_information_band_is_not_zero(self) -> None:
    """A constant-confidence model lands here, not at zero.

    Reading the band as zero is the specific error that makes a base-rate artifact
    look like a nearly perfect model, so it is asserted rather than assumed.
    """
    band = self.curve.aurc_random - self.curve.aurc_oracle
    self.assertAlmostEqual(band, 0.5 - HAND_AURC_ORACLE, places=12)
    self.assertGreater(band, 0.3)

  def test_a_perfect_ordering_scores_exactly_zero_excess(self) -> None:
    curve = RiskCoverageCurve.from_outcomes((1, 1, 1, 0, 0, 0), SIX_CONFIDENCES, seed=0)
    self.assertLess(abs(curve.e_aurc), 1e-12)

  def test_risk_at_zero_coverage_is_none_and_never_zero(self) -> None:
    self.assertIsNone(self.curve.risk(0))

  def test_coverage_at_risk_reads_the_hand_computed_cut(self) -> None:
    # cumulative losses [0,0,1,1,2,3] against 0.2*m = [.2,.4,.6,.8,1.0,1.2]:
    # only m = 1 and m = 2 stay at or below, so the largest is 2.
    rate = self.curve.coverage_at_risk(0.20)
    self.assertIsNotNone(rate)
    self.assertEqual(rate.numerator, 2)
    self.assertEqual(rate.denominator, 6)


class NoParallelVocabularyTest(unittest.TestCase):
  """metrics.py must compose the selective core, not grow a second copy of it."""

  def setUp(self) -> None:
    self.tree = ast.parse(METRICS_SOURCE.read_text(encoding="ascii"))

  def test_no_aurc_arithmetic_is_defined_here(self) -> None:
    defined = {
      node.name.lower()
      for node in ast.walk(self.tree)
      if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for banned in ("aurc", "e_aurc", "risk_coverage", "wilson", "bootstrap"):
      for name in defined:
        self.assertNotIn(
          banned,
          name,
          f"{name} looks like a second implementation of {banned}; that quantity "
          "lives in plr_lr.learn.selective and must be imported, not re-derived.",
        )

  def test_no_percentile_or_argmax_selection(self) -> None:
    """Rule R9's AST half, applied to the module that is allowed quartiles at all.

    Quartiles here are order statistics of the halves of a sorted sample and are
    printed as a spread. A percentile call is the shape a threshold gets read off,
    and max-over-frames selection is this portfolio's own documented way of
    manufacturing a confident wrong number.
    """
    banned = ("percentile", "quantile", "argmax", "argmin", "nanargmax")
    for node in ast.walk(self.tree):
      name = None
      if isinstance(node, ast.Attribute):
        name = node.attr
      elif isinstance(node, ast.Name):
        name = node.id
      if name is None:
        continue
      for token in banned:
        self.assertNotIn(
          token,
          name.lower(),
          f"{name} reads a selection or a fitted quantile off the data being scored",
        )


class ToleranceCriterionTest(unittest.TestCase):
  def test_error_is_the_worst_field_in_units_of_its_own_tolerance(self) -> None:
    # |0.4 - 0| / 0.5 = 0.8 and |1.0 - 0| / 2.0 = 0.5, so the max is 0.8.
    self.assertAlmostEqual(
      tolerance_error((0.4, 1.0), (0.0, 0.0), (0.5, 2.0)), 0.8, places=12
    )

  def test_a_joint_criterion_needs_no_weight(self) -> None:
    """S4's half a millimetre and half a degree, as one number.

    Both fields are divided by their own tolerance before the max, so there is no
    weight to choose and nowhere for one fitted after seeing the errors to enter.
    """
    self.assertTrue(in_tolerance((0.49, 0.49), (0.0, 0.0), (0.5, 0.5)))
    self.assertFalse(in_tolerance((0.49, 0.51), (0.0, 0.0), (0.5, 0.5)))

  def test_a_non_finite_prediction_is_a_miss_and_not_a_crash(self) -> None:
    self.assertEqual(tolerance_error((float("nan"),), (0.0,), (1.0,)), float("inf"))
    self.assertFalse(in_tolerance((float("inf"),), (0.0,), (1.0,)))

  def test_a_zero_tolerance_is_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      tolerance_error((1.0,), (0.0,), (0.0,))

  def test_mismatched_widths_are_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      tolerance_error((1.0, 2.0), (0.0,), (1.0,))

  def test_every_loose_criterion_is_wider_than_the_protocol_tolerance(self) -> None:
    """So SuccessGap is non-negative by construction rather than by special casing."""
    from plr_lr.learn.bench.tasks import TASKS

    for task in TASKS:
      loose = loose_tolerance_for(task.task_id)
      self.assertEqual(len(loose), task.state_dim)
      for wide, tight in zip(loose, task.tolerance):
        self.assertGreaterEqual(wide, tight)

  def test_an_undeclared_loose_criterion_is_refused_not_defaulted(self) -> None:
    with self.assertRaises(LooseCriterionMissing):
      loose_tolerance_for("S99")

  def test_loose_hit_uses_the_policy_layer_criterion(self) -> None:
    self.assertTrue(loose_hit("S2", (0.9,), (0.0,)))
    self.assertFalse(in_tolerance((0.9,), (0.0,), (0.30,)))


class TolerancePredictionErrorTest(unittest.TestCase):
  def test_median_and_quartiles_are_order_statistics(self) -> None:
    # sorted: [1, 2, 3, 4, 5, 6, 7]; median 4; lower half [1,2,3] -> 2;
    # upper half [5,6,7] -> 6.
    tpe = TolerancePredictionError.of([4.0, 1.0, 7.0, 2.0, 6.0, 3.0, 5.0])
    self.assertEqual(tpe.median, 4.0)
    self.assertEqual(tpe.q1, 2.0)
    self.assertEqual(tpe.q3, 6.0)

  def test_the_mean_is_refused_rather_than_returned(self) -> None:
    tpe = TolerancePredictionError.of([1.0, 1.0, 1.0, 1000.0])
    with self.assertRaises(MetricRefused):
      _ = tpe.mean

  def test_a_single_diverged_episode_does_not_move_the_median(self) -> None:
    """The reason the mean is refused, stated as a measurement."""
    tpe = TolerancePredictionError.of([0.1, 0.2, 0.3, 1e6])
    self.assertLess(tpe.median, 1.0)

  def test_non_finite_errors_are_counted_and_excluded_from_the_summary(self) -> None:
    tpe = TolerancePredictionError.of([0.5, float("inf"), 1.5])
    self.assertEqual(tpe.n, 3)
    self.assertEqual(tpe.n_infinite, 1)
    self.assertEqual(tpe.median, 1.0)

  def test_an_all_infinite_column_is_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      TolerancePredictionError.of([float("inf"), float("inf")])

  def test_the_dict_says_the_mean_was_refused(self) -> None:
    tpe = TolerancePredictionError.of([1.0, 2.0])
    self.assertIn("REFUSED", tpe.to_dict()["mean"])


class SuccessGapTest(unittest.TestCase):
  def test_the_gap_carries_both_rates_and_a_paired_interval(self) -> None:
    loose = [1] * 40
    strict = [1] * 30 + [0] * 10
    gap = success_gap("S2", 1, loose, strict, seed=0, n_resamples=200)
    self.assertAlmostEqual(gap.gap, 1.0 - 0.75, places=12)
    self.assertIsInstance(gap.loose, Rate)
    self.assertIsInstance(gap.strict, Rate)
    self.assertFalse(gap.delta.includes_zero)
    self.assertIn("95% Wilson", gap.format())
    self.assertIn("paired", gap.format())

  def test_mismatched_columns_are_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      success_gap("S2", 1, [1, 1], [1], seed=0, n_resamples=10)


class TrustHorizonTest(unittest.TestCase):
  def test_it_stops_at_the_first_horizon_that_is_not_trustworthy(self) -> None:
    self.assertEqual(trust_horizon({1: 0.1, 2: 0.2, 4: 0.5}), 2)

  def test_a_hole_does_not_licence_the_far_horizon(self) -> None:
    """Acceptable at 1, unacceptable at 2, acceptable again at 4 is not a horizon."""
    self.assertEqual(trust_horizon({1: 0.1, 2: 0.9, 4: 0.05}), 1)

  def test_none_when_no_horizon_was_trustworthy(self) -> None:
    self.assertIsNone(trust_horizon({1: 0.9, 2: 0.95}))
    self.assertIsNone(trust_horizon({}))

  def test_an_unranked_horizon_stops_the_walk(self) -> None:
    self.assertIsNone(trust_horizon({1: None, 2: 0.0}))

  def test_the_threshold_is_an_absolute_constant(self) -> None:
    self.assertEqual(TRUST_RISK, 0.25)


#: The scope every overlap test below compares against, written out once. Nothing in
#: it is defaulted: a scope with an axis left blank reports a declaration that names
#: that axis as uncheckable, and a fixture that leaned on that would be testing the
#: withholding path while claiming to test the comparison.
SCOPE = ScoringScope(
  corpus=BENCH_CORPUS,
  split="scoring",
  seed_epoch="demo-2026-08-03",
  pack_id="builtin-demo-20260803",
  task_id="S3",
  seed_range=(100, 199),
)


class ScoringScopeTest(unittest.TestCase):
  """The object that says what was compared, and refuses to imply a comparison."""

  def test_a_scope_that_will_not_name_its_corpus_is_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      ScoringScope(corpus="", split="scoring")

  def test_a_scope_that_will_not_name_its_split_is_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      ScoringScope(corpus=BENCH_CORPUS, split="")

  def test_an_empty_seed_range_is_refused(self) -> None:
    """An empty range meets nothing, so every declared range would read as clean."""
    with self.assertRaises(MetricRefused):
      ScoringScope(corpus=BENCH_CORPUS, split="scoring", seed_range=(200, 100))

  def test_the_range_is_normalised_to_a_pair_of_integers(self) -> None:
    scope = ScoringScope(corpus=BENCH_CORPUS, split="scoring", seed_range=(100.0, 199.0))
    self.assertEqual(scope.seed_range, (100, 199))

  def test_describe_names_every_axis_that_was_compared(self) -> None:
    text = SCOPE.describe()
    for part in (
      "corpus=deferlab",
      "split=scoring",
      "epoch=demo-2026-08-03",
      "pack=builtin-demo-20260803",
      "task=S3",
      "seeds=100-199",
    ):
      self.assertIn(part, text)

  def test_an_undeclared_axis_is_left_out_rather_than_printed_blank(self) -> None:
    """A blank axis in the printed scope reads as an axis that was compared."""
    text = ScoringScope(corpus=BENCH_CORPUS, split="dev").describe()
    self.assertEqual(text, "corpus=deferlab split=dev")


class TrainingOverlapLeakTest(unittest.TestCase):
  """Declarations that reach the rows being scored. The check has to fire."""

  def test_duplicates_are_preserved_in_the_count(self) -> None:
    """The measured defect this replaces used a frozenset size and under-reported.

    Forty repeats of one leaked row are forty leaked rows. Counting the distinct
    set reports one, and a summary built on that carries no leak line at all.
    """
    rows = ["a"] * 40 + ["b", "c"]
    overlap = training_overlap(rows, {"deferlab:episode=a"}, SCOPE)
    self.assertEqual(overlap.n_rows_seen, 40)
    self.assertEqual(overlap.n_distinct_seen, 1)
    self.assertEqual(overlap.n_rows, 42)
    self.assertEqual(overlap.n_distinct_rows, 3)
    self.assertTrue(overlap.leaked)
    self.assertTrue(overlap.established)
    self.assertEqual(overlap.verdict, "LEAK")
    self.assertIn("LEAK", overlap.format())

  def test_the_duplicate_preserving_count_is_printed_first(self) -> None:
    """Both counts print, and the one a reader lands on is the larger true one."""
    rows = ["a"] * 40 + ["b", "c"]
    text = training_overlap(rows, {"deferlab:episode=a"}, SCOPE).format()
    self.assertLess(text.index("40 of 42"), text.index("1 of 3"))

  def test_a_bare_corpus_declaration_covers_every_scored_row(self) -> None:
    """Fit on this benchmark's corpus, narrowed by nothing, is a leak on every row."""
    overlap = training_overlap(["a", "b", "c"], {BENCH_CORPUS}, SCOPE)
    self.assertEqual(overlap.verdict, "LEAK")
    self.assertEqual(overlap.n_rows_seen, 3)
    self.assertAlmostEqual(overlap.fraction_rows_seen, 1.0, places=12)

  def test_the_split_being_scored_is_a_leak(self) -> None:
    """The case the previous version printed a clean line for."""
    overlap = training_overlap(["a", "b"], {"deferlab:split=scoring"}, SCOPE)
    self.assertEqual(overlap.verdict, "LEAK")
    self.assertEqual(overlap.matched, ("deferlab:split=scoring",))

  def test_a_declared_seed_range_that_meets_the_scored_one_is_a_leak(self) -> None:
    overlap = training_overlap(["a", "b"], {"deferlab:seeds=150-400"}, SCOPE)
    self.assertEqual(overlap.verdict, "LEAK")
    self.assertEqual(overlap.n_rows_seen, 2)

  def test_an_episode_list_leaks_exactly_the_episodes_it_names(self) -> None:
    overlap = training_overlap(
      ["a", "b", "c"], {"deferlab:episode=a:episode=c"}, SCOPE
    )
    self.assertEqual(overlap.verdict, "LEAK")
    self.assertEqual(overlap.n_rows_seen, 2)

  def test_separate_declarations_are_a_union_over_rows(self) -> None:
    overlap = training_overlap(
      ["a", "b", "c"], {"deferlab:episode=a", "deferlab:episode=b"}, SCOPE
    )
    self.assertEqual(overlap.n_rows_seen, 2)
    self.assertEqual(len(overlap.matched), 2)

  def test_the_scope_that_was_compared_is_recorded_on_the_answer(self) -> None:
    """A verdict that will not say what it compared against cannot be checked."""
    overlap = training_overlap(["a"], {"deferlab:split=scoring"}, SCOPE)
    self.assertEqual(overlap.scope, SCOPE.describe())
    self.assertEqual(overlap.to_dict()["verdict"], "LEAK")
    self.assertTrue(overlap.to_dict()["established"])


class TrainingOverlapCleanTest(unittest.TestCase):
  """Declarations compared in full and found not to reach these rows.

  Every one of these has to be scored rather than withheld. A leak check that
  refuses a model trained on an outside corpus, or on the development split, refuses
  every honest submission, and a benchmark nobody can enter establishes exactly as
  little as one that clears everybody.
  """

  def test_a_clean_run_says_so(self) -> None:
    overlap = training_overlap(["a", "b"], {"deferlab:episode=z"}, SCOPE)
    self.assertFalse(overlap.leaked)
    self.assertEqual(overlap.n_rows_seen, 0)
    self.assertEqual(overlap.verdict, "CLEAN")
    self.assertNotIn("LEAK", overlap.format())

  def test_another_corpus_is_clean_and_not_withheld(self) -> None:
    overlap = training_overlap(["a", "b"], {"droid:split=train"}, SCOPE)
    self.assertEqual(overlap.verdict, "CLEAN")
    self.assertFalse(overlap.leaked)
    self.assertEqual(overlap.unresolved, ())

  def test_the_development_split_is_clean_while_the_scoring_split_is_scored(self) -> None:
    overlap = training_overlap(["a", "b"], {"deferlab:split=dev"}, SCOPE)
    self.assertEqual(overlap.verdict, "CLEAN")
    self.assertFalse(overlap.leaked)

  def test_another_task_in_the_same_corpus_is_clean(self) -> None:
    overlap = training_overlap(["a", "b"], {"deferlab:task=S1"}, SCOPE)
    self.assertEqual(overlap.verdict, "CLEAN")

  def test_another_seed_epoch_is_clean(self) -> None:
    overlap = training_overlap(["a"], {"deferlab:epoch=demo-2026-01-01"}, SCOPE)
    self.assertEqual(overlap.verdict, "CLEAN")

  def test_a_seed_range_that_does_not_meet_the_scored_one_is_clean(self) -> None:
    overlap = training_overlap(["a"], {"deferlab:seeds=0-99"}, SCOPE)
    self.assertEqual(overlap.verdict, "CLEAN")

  def test_the_terms_within_one_declaration_are_conjunctive(self) -> None:
    """Right split, wrong task, is not a leak on this task's rows."""
    overlap = training_overlap(["a"], {"deferlab:split=scoring:task=S1"}, SCOPE)
    self.assertEqual(overlap.verdict, "CLEAN")

  def test_the_analytic_null_declaration_is_clean_with_and_without_a_scope(self) -> None:
    """A model fit on nothing has made a complete declaration either way."""
    self.assertEqual(training_overlap(["a"], {NO_TRAINING_DATA}, SCOPE).verdict, "CLEAN")
    self.assertEqual(training_overlap(["a"], {NO_TRAINING_DATA}).verdict, "CLEAN")

  def test_no_declaration_at_all_is_not_this_functions_refusal(self) -> None:
    """An empty fingerprint set is R7's business upstream, and reaches here as clean."""
    self.assertEqual(training_overlap(["a", "b"], set(), SCOPE).verdict, "CLEAN")

  def test_the_analytic_null_literal_matches_the_predict_module(self) -> None:
    """Restated in metrics.py to keep it stdlib-only, so the two are pinned together."""
    from plr_lr.learn.predict import NO_TRAINING_DATA as PREDICT_LITERAL

    self.assertEqual(NO_TRAINING_DATA, PREDICT_LITERAL)


class TrainingOverlapUncheckableTest(unittest.TestCase):
  """Declarations this harness could not read. Withheld, never cleared.

  Skipping the part that could not be read is the defect this replaces: the check
  intersected scored episode identifiers against declared corpus and split names,
  matched nothing, and printed that as proof of no leak.
  """

  def test_a_key_this_module_does_not_understand_withholds_the_row(self) -> None:
    overlap = training_overlap(["a"], {"deferlab:checkpoint=v3"}, SCOPE)
    self.assertEqual(overlap.verdict, "LEAK-UNCHECKABLE")
    self.assertEqual(overlap.unresolved, ("deferlab:checkpoint=v3",))
    self.assertIn("LEAK-UNCHECKABLE", overlap.format())

  def test_a_term_that_is_not_a_key_value_pair_withholds_the_row(self) -> None:
    overlap = training_overlap(["a"], {"deferlab:everything"}, SCOPE)
    self.assertEqual(overlap.verdict, "LEAK-UNCHECKABLE")

  def test_a_term_with_an_empty_value_withholds_the_row(self) -> None:
    overlap = training_overlap(["a"], {"deferlab:split="}, SCOPE)
    self.assertEqual(overlap.verdict, "LEAK-UNCHECKABLE")

  def test_a_seed_range_with_nothing_to_compare_against_withholds_the_row(self) -> None:
    scope = ScoringScope(corpus=BENCH_CORPUS, split="scoring")
    overlap = training_overlap(["a"], {"deferlab:seeds=0-99"}, scope)
    self.assertEqual(overlap.verdict, "LEAK-UNCHECKABLE")

  def test_a_seed_term_that_is_not_a_range_withholds_the_row(self) -> None:
    overlap = training_overlap(["a"], {"deferlab:seeds=lots"}, SCOPE)
    self.assertEqual(overlap.verdict, "LEAK-UNCHECKABLE")

  def test_without_a_scope_every_real_declaration_is_uncheckable(self) -> None:
    """The state the runner was calling this in, kept as a test rather than a bug."""
    for entry in ("droid", "droid:split=train", "deferlab:split=dev"):
      overlap = training_overlap(["a", "b"], {entry})
      self.assertEqual(overlap.verdict, "LEAK-UNCHECKABLE", entry)
      self.assertTrue(overlap.leaked, entry)

  def test_uncheckable_is_not_established_and_still_fails_safe(self) -> None:
    """The one boolean the callers read is the conservative half of the distinction."""
    overlap = training_overlap(["a"], {"deferlab:checkpoint=v3"}, SCOPE)
    self.assertTrue(overlap.leaked)
    self.assertFalse(overlap.established)
    self.assertEqual(overlap.n_rows_seen, 0)

  def test_one_unreadable_declaration_withholds_a_row_the_others_cleared(self) -> None:
    overlap = training_overlap(
      ["a"], {"droid:split=train", "deferlab:checkpoint=v3"}, SCOPE
    )
    self.assertEqual(overlap.verdict, "LEAK-UNCHECKABLE")
    self.assertTrue(overlap.leaked)

  def test_an_established_leak_outranks_an_unreadable_declaration(self) -> None:
    """Both lines print, and the verdict is the stronger of the two answers."""
    overlap = training_overlap(
      ["a"], {"deferlab:split=scoring", "deferlab:checkpoint=v3"}, SCOPE
    )
    self.assertEqual(overlap.verdict, "LEAK")
    self.assertIn("LEAK:", overlap.format())
    self.assertIn("LEAK-UNCHECKABLE:", overlap.format())


class TrainingOverlapCountTest(unittest.TestCase):
  def test_the_fraction_uses_the_duplicate_preserving_numerator(self) -> None:
    overlap = OverlapCount(n_rows=10, n_rows_seen=4, n_distinct_rows=2, n_distinct_seen=1)
    self.assertAlmostEqual(overlap.fraction_rows_seen, 0.4, places=12)

  def test_one_declaration_passed_as_a_bare_string_is_refused(self) -> None:
    """Iterated, it is one corpus per character, and the answer comes back CLEAN."""
    with self.assertRaises(MetricRefused):
      training_overlap(["a"], "deferlab:split=scoring", SCOPE)
    with self.assertRaises(MetricRefused):
      training_overlap("abc", {"deferlab"}, SCOPE)

  def test_the_three_verdicts_are_three_different_answers(self) -> None:
    verdicts = {
      training_overlap(["a"], {"deferlab:split=scoring"}, SCOPE).verdict,
      training_overlap(["a"], {"droid"}, SCOPE).verdict,
      training_overlap(["a"], {"deferlab:checkpoint=v3"}, SCOPE).verdict,
    }
    self.assertEqual(verdicts, {"LEAK", "CLEAN", "LEAK-UNCHECKABLE"})


class DeclaredComputeTest(unittest.TestCase):
  def test_compute_is_declared_and_never_scored(self) -> None:
    compute = ComputeDeclaration.of([1, 3, 5], [0.1, 0.3, 0.5], hardware="one 4090")
    self.assertEqual(compute.median_forward_passes, 3.0)
    self.assertAlmostEqual(compute.median_wall_clock_s, 0.3, places=12)
    self.assertFalse(compute.scored)
    self.assertIn(NOT_RANKED, compute.format())
    self.assertIn("one 4090", compute.format())

  def test_an_undeclared_hardware_string_is_a_word_and_not_a_blank(self) -> None:
    compute = ComputeDeclaration.of([1], [0.0], hardware="")
    self.assertEqual(compute.hardware, "UNDECLARED")

  def test_mismatched_columns_are_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      ComputeDeclaration.of([1, 2], [0.1])


class PixelColumnTest(unittest.TestCase):
  def test_the_column_can_never_be_ranked(self) -> None:
    column = PixelColumn(psnr=99.0, ssim=1.0, declared_by="N5:PixelPerfect")
    self.assertFalse(column.rankable)
    self.assertIn(NOT_RANKED, column.format())

  def test_an_undeclared_column_says_so_rather_than_printing_zero(self) -> None:
    column = PixelColumn()
    self.assertFalse(column.declared)
    self.assertIn("not declared", column.format())


class BelowTheFoldTest(unittest.TestCase):
  def test_ece_is_zero_for_a_perfectly_calibrated_column(self) -> None:
    # Ten episodes at confidence 1.0 that are all hits: accuracy 1.0, mean
    # confidence 1.0, so the weighted absolute gap is exactly zero.
    self.assertAlmostEqual(ece_not_ranked([1.0] * 10, [1] * 10), 0.0, places=12)

  def test_ece_reads_the_gap_when_confidence_overstates_accuracy(self) -> None:
    # All ten at confidence 1.0, five hits: |0.5 - 1.0| weighted by 1.0 = 0.5.
    self.assertAlmostEqual(
      ece_not_ranked([1.0] * 10, [1] * 5 + [0] * 5), 0.5, places=12
    )

  def test_auroc_is_one_for_a_perfect_separation(self) -> None:
    self.assertAlmostEqual(
      auroc_not_ranked([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]), 1.0, places=12
    )

  def test_auroc_is_none_and_not_a_half_when_one_class_is_empty(self) -> None:
    """0.5 there reads as a measurement of a model that was never tested."""
    self.assertIsNone(auroc_not_ranked([0.9, 0.8], [1, 1]))

  def test_median_of_an_empty_sample_is_refused(self) -> None:
    with self.assertRaises(MetricRefused):
      metrics.median_of([])


class LooseToleranceTableTest(unittest.TestCase):
  def test_every_entry_is_finite_and_positive(self) -> None:
    for task_id, tolerance in LOOSE_TOLERANCE.items():
      for value in tolerance:
        self.assertTrue(math.isfinite(value), task_id)
        self.assertGreater(value, 0.0, task_id)


# -- the leak check, end to end ------------------------------------------------

POWER_SCREEN = Screen(
  theta=0.5,
  trust_horizon=1,
  max_frame_age_s=1.0,
  calibrated_on="tests: a hand-written threshold, declared as one",
  n_calibration=30,
)
POWER_SEEDS = SeedPlan(epoch="test-epoch")

#: Enough episodes for every task to clear the action-conditioning gate, so the
#: difference between the two submissions below is the leak check and not the gate.
POWER_N = 200


class _Declaring:
  """The declared floor row, with a submitter's training declaration bolted on.

  The floor predictor rather than a new double on purpose: it clears the gate and
  the control suite on every task, so a row that is withheld here was withheld by
  the leak check and by nothing else.
  """

  def __init__(self, inner, fingerprints) -> None:
    self._inner = inner
    self.name = "floor declaring {0}".format(sorted(fingerprints))
    self.license_text = inner.license_text
    self.hardware = inner.hardware
    self.state_dim = inner.state_dim
    self.horizon = inner.horizon
    self.action_space = inner.action_space
    self.training_fingerprints = frozenset(fingerprints)
    self.fitted_components = ()

  def predict(self, request):
    return self._inner.predict(request)


class LeakCheckPowerTest(unittest.TestCase):
  """Both directions, through run_bench, on a pack of the scoring split.

  This lives beside the unit tests for :func:`training_overlap` because it is the
  same claim measured where it matters: the unit tests say the three verdicts are
  computed correctly, and this says the harness reaches them with a scope in hand.
  Between those two the check was structurally incapable of firing once, and then
  refused every submission that declared any training corpus at all. Both states
  pass a unit test of the arithmetic alone.
  """

  @classmethod
  def setUpClass(cls) -> None:
    cls.pack = builtin_demo_pack(n_episodes=POWER_N, split="scoring")

  def _run(self, fingerprints):
    predictors = {
      task.task_id: _Declaring(floor_predictor_for(task), fingerprints) for task in TASKS
    }
    return run_bench(
      predictors,
      TASKS,
      self.pack,
      POWER_SCREEN,
      POWER_SEEDS,
      n_permutations=20,
      n_resamples=50,
    )

  def test_a_model_fit_on_the_scoring_split_is_refused_on_every_row(self) -> None:
    report = self._run(("deferlab:split=scoring",))
    self.assertEqual(len(report.rows), 16)
    for row in report.rows:
      rules = [r.rule for r in row.refusals]
      self.assertIn("R7", rules, f"{row.task_id}@{row.horizon}")
      self.assertTrue(row.blocking, f"{row.task_id}@{row.horizon}")
      self.assertIn("headline withheld", row.headline())
      self.assertNotIn("E-AURC", row.headline())
      self.assertIn("LEAK:", row.overlap.format())
      self.assertEqual(row.overlap.verdict, "LEAK")
      self.assertEqual(row.overlap.n_rows_seen, POWER_N)

  def test_a_model_fit_on_the_development_split_is_scored(self) -> None:
    """The half a check that refuses everybody would also pass. It has to be here."""
    report = self._run(("deferlab:split=dev",))
    self.assertEqual(len(report.rows), 16)
    for row in report.rows:
      self.assertEqual([r.rule for r in row.refusals], [], f"{row.task_id}@{row.horizon}")
      self.assertEqual(row.overlap.verdict, "CLEAN")
      self.assertEqual(row.blocking, ())
      self.assertIn("E-AURC", row.headline())
      self.assertIn("training overlap: none of", row.overlap.format())

  def test_a_model_fit_on_an_outside_corpus_is_scored(self) -> None:
    report = self._run(("droid:split=train",))
    self.assertEqual(len(report.rows), 16)
    for row in report.rows:
      self.assertEqual([r.rule for r in row.refusals], [])
      self.assertEqual(row.overlap.verdict, "CLEAN")

  def test_a_declaration_the_harness_cannot_read_withholds_every_row(self) -> None:
    report = self._run(("deferlab:checkpoint=v3",))
    self.assertEqual(len(report.rows), 16)
    for row in report.rows:
      self.assertIn("R7", [r.rule for r in row.refusals])
      self.assertEqual(row.overlap.verdict, "LEAK-UNCHECKABLE")

  def test_the_row_records_the_scope_it_was_compared_against(self) -> None:
    """A clean line that will not say what it compared is the line this replaces."""
    report = self._run(("deferlab:split=dev",))
    row = report.rows[0]
    self.assertEqual(
      row.overlap.scope,
      POWER_SEEDS.scope_for(self.pack, TASKS[0]).describe(),
    )
    self.assertIn("corpus=deferlab", row.overlap.scope)
    self.assertIn("split=scoring", row.overlap.scope)

  def test_the_scope_carries_the_split_being_scored_and_not_the_declared_one(self) -> None:
    scope = POWER_SEEDS.scope_for(self.pack, TASKS[0])
    self.assertEqual(scope.corpus, BENCH_CORPUS)
    self.assertEqual(scope.split, self.pack.split)
    self.assertEqual(scope.seed_epoch, self.pack.seed_epoch)
    self.assertEqual(scope.pack_id, self.pack.pack_id)
    self.assertEqual(scope.task_id, TASKS[0].task_id)

  def test_a_run_that_was_not_told_its_seed_range_says_so_rather_than_clearing(
    self,
  ) -> None:
    """No seed range in the plan, so a seeds= declaration is withheld, not waved on."""
    self.assertIsNone(POWER_SEEDS.seed_range)
    report = self._run(("deferlab:seeds=0-99",))
    for row in report.rows:
      self.assertEqual(row.overlap.verdict, "LEAK-UNCHECKABLE")

  def test_a_declared_seed_range_is_compared_once_the_plan_names_one(self) -> None:
    scope = SeedPlan(epoch="test-epoch", seed_range=(0, 99)).scope_for(
      self.pack, TASKS[0]
    )
    self.assertEqual(training_overlap(["a"], {"deferlab:seeds=50-70"}, scope).verdict, "LEAK")
    self.assertEqual(
      training_overlap(["a"], {"deferlab:seeds=100-199"}, scope).verdict, "CLEAN"
    )


class LeakRemovesTheRankableRowTest(unittest.TestCase):
  """The consequence, on the one row of this pack that is rankable at all.

  Separate from :class:`LeakCheckPowerTest` because it needs the full four hundred
  episodes per task: below that the floor predictor's confidence ordering does not
  beat its own shuffle anywhere, so ``valid`` is false on every row for a reason
  that has nothing to do with a leak, and an assertion about it would pass while
  measuring nothing. Two runs of one pack, and the pack is the expensive part.
  """

  @classmethod
  def setUpClass(cls) -> None:
    pack = builtin_demo_pack(n_episodes=400, split="scoring")

    def run(fingerprints):
      predictors = {
        t.task_id: _Declaring(floor_predictor_for(t), fingerprints) for t in TASKS
      }
      return run_bench(
        predictors, TASKS, pack, POWER_SCREEN, POWER_SEEDS, n_permutations=20, n_resamples=50
      )

    cls.clean = run(("deferlab:split=dev",))
    cls.leaked = run(("deferlab:split=scoring",))

  def test_declaring_the_development_split_leaves_a_rankable_row(self) -> None:
    rankable = [(r.task_id, r.horizon) for r in self.clean.rows if r.rankable]
    self.assertTrue(rankable, "no row was rankable, so the next test measures nothing")

  def test_declaring_the_scoring_split_leaves_none(self) -> None:
    self.assertEqual([(r.task_id, r.horizon) for r in self.leaked.rows if r.rankable], [])
    self.assertEqual([(r.task_id, r.horizon) for r in self.leaked.rows if r.valid], [])


class RefusalLabelTest(unittest.TestCase):
  """Every refusal this runner prints carries the label its rule publishes.

  The runner emitted ``R7 LEAK`` while the published sixteen-rule table prints
  ``PROVENANCE: UNDECLARED`` beside R7, and nothing caught it because
  :class:`plr_lr.learn.bench.report.Refusal` validates the rule identifier and not
  the label. A reader cross-referencing that line against the table found a rule
  about an adapter with no declared training set.
  """

  def test_the_label_map_is_the_published_table(self) -> None:
    self.assertEqual(
      runner.RULE_LABELS, {rule: label for rule, label, _why in REFUSAL_RULES}
    )

  def test_the_label_is_derived_and_not_passed(self) -> None:
    self.assertEqual(runner._refusal("R7", "why").label, "PROVENANCE: UNDECLARED")
    self.assertEqual(runner._refusal("R4", "why").label, "INTERVENTION-REFUSED")

  def test_a_rule_outside_the_published_table_raises(self) -> None:
    with self.assertRaises(ValueError):
      runner._refusal("R99", "not in the published table")

  def test_the_runner_builds_no_refusal_around_the_label_map(self) -> None:
    """AST, not a grep: a second construction site is how the mismatch got in."""
    tree = ast.parse(RUNNER_SOURCE.read_text(encoding="ascii"))
    inside_helper = {
      node
      for function in ast.walk(tree)
      if isinstance(function, ast.FunctionDef) and function.name == "_refusal"
      for node in ast.walk(function)
    }
    built = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.Call)
      and isinstance(node.func, ast.Name)
      and node.func.id == "Refusal"
    ]
    self.assertTrue(built)
    for node in built:
      self.assertIn(node, inside_helper, f"Refusal built directly at line {node.lineno}")


if __name__ == "__main__":
  unittest.main()
