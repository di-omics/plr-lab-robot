"""Tests for plr_lr.learn.selective and scripts/score_standalone.py.

Every expected number here is either hand-computed and written out as the
arithmetic that produced it, or it is a structural claim about a refusal. That
rule exists because a metrics test that computes the expected value with the
code under test proves only that the code is deterministic, and this module is
the one place in the build where a wrong number would look most like a right
one: an inverted null band, a risk of zero at zero coverage, or an interval-free
rate all read perfectly plausibly on a scorecard.

The claims under test are that an oracle ordering scores exactly zero excess
AURC; that a constant confidence lands on ``AURC_random - AURC_oracle`` and not
on zero; that a confidence carrying no information cannot be told apart from
always acting; that nothing prints below thirty episodes; and that the two
one-sided abstention numbers cannot be reached without their partners.
"""

from __future__ import annotations

import ast
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from plr_lr.learn.selective import (
  BOOTSTRAP_RESAMPLES,
  HEADLINE_W_FAIL,
  MIN_N_RANKABLE,
  MIN_N_REPORTABLE,
  PERMUTATION_RESAMPLES,
  TIER_PROVISIONAL,
  TIER_RANKABLE,
  TIER_UNDERPOWERED,
  W_FAIL_WEIGHTS,
  AbstentionBandRefused,
  AbstentionCredit,
  ActDeferTable,
  EpisodeOutcome,
  FrozenThreshold,
  Rate,
  RiskCoverageCurve,
  ThresholdNotFrozen,
  confidence_shuffle_p,
  decision_utility,
  ks_two_sample,
  paired_bootstrap_ci,
  score,
  spearman_rho,
  tier_of,
  wilson_interval,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
STANDALONE = REPO_ROOT / "scripts" / "score_standalone.py"

THRESHOLD = FrozenThreshold(value=0.6, calibrated_on="dev seeds 0-99", n_calibration=100)


def separated(n: int, n_errors: int):
  """Hits and confidences where every hit outranks every error: the oracle order."""
  hits = [1] * (n - n_errors) + [0] * n_errors
  confidences = [1.0 - i / (2.0 * n) for i in range(n)]
  return hits, confidences


def informative_outcomes(
  n: int,
  seed: int = 1,
  unknowable_every: int = 7,
  abstain_on_unknowable: bool = True,
):
  """Episodes whose confidence really does order the errors, with refusals mixed in.

  The unknowable episodes are interleaved by index rather than grouped, matching
  the fixture rule that a model must not be able to find them by position, and
  the resulting fraction stays inside the band the abstention metrics require.
  With ``abstain_on_unknowable`` false the model predicts on them instead and is
  wrong, since the deciding latent was never visible; that variant keeps the
  ranked count equal to the episode count, which is what the tier boundaries are
  stated in.
  """
  rng = random.Random(seed)
  outcomes = []
  for i in range(n):
    if i % unknowable_every == 0:
      if abstain_on_unknowable:
        outcomes.append(
          EpisodeOutcome(f"e{i}", hit=False, confidence=None, abstain=True, unknowable=True)
        )
      else:
        outcomes.append(
          EpisodeOutcome(f"e{i}", hit=False, confidence=rng.random() * 0.3, unknowable=True)
        )
      continue
    confidence = rng.random()
    outcomes.append(EpisodeOutcome(f"e{i}", hit=confidence > 0.3, confidence=confidence))
  return outcomes


def uninformative_outcomes(n: int, seed: int = 7, unknowable_every: int = 7):
  """Same shape, but the confidence is drawn independently of whether it was a hit."""
  rng = random.Random(seed)
  outcomes = []
  for i in range(n):
    if i % unknowable_every == 0:
      outcomes.append(
        EpisodeOutcome(f"e{i}", hit=False, confidence=None, abstain=True, unknowable=True)
      )
      continue
    confidence = rng.random()
    hit = rng.random() > 0.3
    outcomes.append(EpisodeOutcome(f"e{i}", hit=hit, confidence=confidence))
  return outcomes


class WilsonTest(unittest.TestCase):
  """Hand-computed intervals at the three counts a benchmark actually hits."""

  # z = 1.96 so z^2 = 3.8416, and with n = 30 the denominator is n + z^2 = 33.8416.
  # centre = (k + z^2/2) / 33.8416 = (k + 1.9208) / 33.8416
  # half   = (1.96 / 33.8416) * sqrt(k*(30-k)/30 + 0.9604)

  def test_zero_of_thirty(self):
    # k = 0: sqrt(0 + 0.9604) = 0.98, so half = (1.96*0.98)/33.8416 = 1.9208/33.8416,
    # which is the centre exactly. The interval is therefore (0, 2*centre) =
    # (0, 3.8416/33.8416). Seeing no failures in thirty tries is not a rate of zero.
    low, high = wilson_interval(0, 30)
    self.assertEqual(low, 0.0)
    self.assertAlmostEqual(high, 3.8416 / 33.8416, places=15)
    self.assertAlmostEqual(high, 0.11351709139047798, places=15)

  def test_fifteen_of_thirty(self):
    # k = 15: centre = 16.9208/33.8416 = 0.5 exactly, since 33.8416/2 = 16.9208.
    # half = (1.96/33.8416) * sqrt(15*15/30 + 0.9604) = (1.96/33.8416)*sqrt(8.4604)
    #      = 0.1684614877282624
    low, high = wilson_interval(15, 30)
    half = (1.96 / 33.8416) * math.sqrt(8.4604)
    self.assertAlmostEqual((low + high) / 2.0, 0.5, places=15)
    self.assertAlmostEqual(low, 0.5 - half, places=15)
    self.assertAlmostEqual(low, 0.3315385122717376, places=15)
    self.assertAlmostEqual(high, 0.6684614877282624, places=15)

  def test_thirty_of_thirty(self):
    # k = 30: half = 1.9208/33.8416 again, centre = 31.9208/33.8416, so
    # low  = (31.9208 - 1.9208)/33.8416 = 30/33.8416 = 0.8864829086095221
    # high = (31.9208 + 1.9208)/33.8416 = 33.8416/33.8416 = 1.0 exactly.
    low, high = wilson_interval(30, 30)
    self.assertAlmostEqual(low, 30.0 / 33.8416, places=15)
    self.assertAlmostEqual(low, 0.8864829086095221, places=15)
    self.assertEqual(high, 1.0)

  def test_thirty_of_thirty_is_not_a_point(self):
    """Thirty out of thirty is not proof of 1.0, which the normal interval claims."""
    low, _ = wilson_interval(30, 30)
    self.assertLess(low, 0.9)

  def test_refuses_an_impossible_proportion(self):
    with self.assertRaises(ValueError):
      wilson_interval(31, 30)
    with self.assertRaises(ValueError):
      wilson_interval(0, 0)


class RateTest(unittest.TestCase):
  """A rate keeps its counts, so it can never be printed without an interval."""

  def test_a_bare_float_rate_is_not_constructible(self):
    with self.assertRaises(TypeError):
      Rate(0.5, 30)  # type: ignore[arg-type]
    with self.assertRaises(TypeError):
      Rate(15, 30.0)  # type: ignore[arg-type]

  def test_counts_must_be_a_proportion(self):
    with self.assertRaises(ValueError):
      Rate(31, 30)
    with self.assertRaises(ValueError):
      Rate(0, 0)

  def test_interval_travels_with_every_rate(self):
    rate = Rate(15, 30, name="ADA")
    self.assertEqual(rate.value, 0.5)
    self.assertIsNotNone(rate.interval)
    self.assertIn("95% Wilson", rate.format())

  def test_underpowered_rate_has_no_number(self):
    rate = Rate(10, 29, name="ADA")
    self.assertIsNone(rate.value)
    self.assertIsNone(rate.interval)
    self.assertEqual(rate.format(), "ADA: UNDERPOWERED n=29")


class TierTest(unittest.TestCase):
  def test_three_tiers_at_the_documented_cut_points(self):
    self.assertEqual(tier_of(MIN_N_REPORTABLE - 1), TIER_UNDERPOWERED)
    self.assertEqual(tier_of(MIN_N_REPORTABLE), TIER_PROVISIONAL)
    self.assertEqual(tier_of(MIN_N_RANKABLE - 1), TIER_PROVISIONAL)
    self.assertEqual(tier_of(MIN_N_RANKABLE), TIER_RANKABLE)


class RiskCoverageTest(unittest.TestCase):
  """The four curve numbers, hand-computed on a six-episode fixture."""

  def curve(self):
    # confidences descend, so the ranked losses are [0, 0, 1, 0, 1, 0]: N = 6, e = 2.
    hits = [1, 1, 0, 1, 0, 1]
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4]
    return RiskCoverageCurve.from_outcomes(hits, confidences, seed=0)

  def test_risk_at_each_coverage(self):
    curve = self.curve()
    self.assertEqual(curve.risk(1), 0.0)
    self.assertEqual(curve.risk(2), 0.0)
    self.assertAlmostEqual(curve.risk(3), 1.0 / 3.0, places=15)
    self.assertAlmostEqual(curve.risk(4), 1.0 / 4.0, places=15)
    self.assertAlmostEqual(curve.risk(5), 2.0 / 5.0, places=15)
    self.assertAlmostEqual(curve.risk(6), 2.0 / 6.0, places=15)

  def test_zero_coverage_is_none_and_never_zero(self):
    """Refusal rule R1: zero risk at zero coverage is the most flattering lie."""
    curve = self.curve()
    self.assertIsNone(curve.risk(0))
    self.assertIsNotNone(curve.risk(1))
    self.assertEqual(curve.coverage(0), 0.0)
    self.assertEqual(curve.coverage(6), 1.0)

  def test_hand_computed_aurc_and_bands(self):
    curve = self.curve()
    # AURC = (0 + 0 + 1/3 + 1/4 + 2/5 + 1/3) / 6 = (79/60)/6 = 79/360
    self.assertAlmostEqual(curve.aurc, 79.0 / 360.0, places=15)
    # oracle: 4 correct, so only m = 5 and m = 6 carry an error:
    # (1/5 + 2/6)/6 = (8/15)/6 = 4/45
    self.assertAlmostEqual(curve.aurc_oracle, 4.0 / 45.0, places=15)
    # random: e/N = 2/6
    self.assertAlmostEqual(curve.aurc_random, 2.0 / 6.0, places=15)
    # E-AURC = 79/360 - 32/360 = 47/360
    self.assertAlmostEqual(curve.e_aurc, 47.0 / 360.0, places=15)

  def test_coverage_at_risk_and_its_none(self):
    curve = self.curve()
    # cumulative losses are 0,0,1,1,2,2; the largest m with cumulative <= 0.2*m is 2.
    rate = curve.coverage_at_risk(0.20)
    self.assertIsNotNone(rate)
    self.assertEqual((rate.numerator, rate.denominator), (2, 6))
    all_wrong = RiskCoverageCurve.from_outcomes([0] * 6, [0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    self.assertIsNone(all_wrong.coverage_at_risk(0.20))

  def test_oracle_ranking_has_zero_excess_aurc(self):
    hits, confidences = separated(200, 40)
    curve = RiskCoverageCurve.from_outcomes(hits, confidences, seed=0)
    self.assertLess(abs(curve.e_aurc), 1e-12)
    self.assertAlmostEqual(curve.aurc, curve.aurc_oracle, places=15)
    self.assertGreater(curve.aurc_random, curve.aurc_oracle)

  def test_constant_confidence_lands_on_the_random_minus_oracle_band(self):
    """The null band is AURC_random - AURC_oracle, not zero. An inverted draft of

    this formula reads a constant-confidence model as near perfect, which is the
    single most damaging error this module could ship.
    """
    n, n_errors = 400, 100
    hits = [1] * (n - n_errors) + [0] * n_errors
    reference = RiskCoverageCurve.from_outcomes(hits, [1.0] * n, seed=0)
    band = reference.aurc_random - reference.aurc_oracle
    self.assertGreater(band, 0.15)
    values = []
    for seed in range(40):
      curve = RiskCoverageCurve.from_outcomes(hits, [0.5] * n, seed=seed)
      values.append(curve.e_aurc)
      # Each single run scatters, since a constant confidence leaves the whole
      # ordering to the recorded tie break. None of them is anywhere near zero.
      self.assertLess(abs(curve.e_aurc - band), 0.10)
      self.assertGreater(curve.e_aurc, 0.5 * band)
    self.assertLess(abs(sum(values) / len(values) - band), 0.01)

  def test_a_curve_needs_finite_confidences(self):
    with self.assertRaises(ValueError):
      RiskCoverageCurve.from_outcomes([1, 0], [float("nan"), 0.5])
    with self.assertRaises(ValueError):
      RiskCoverageCurve.from_outcomes([1, 0], [1.5, 0.5])
    with self.assertRaises(ValueError):
      RiskCoverageCurve.from_outcomes([], [])


class PermutationTest(unittest.TestCase):
  def test_oracle_ordering_beats_its_own_shuffle(self):
    hits, confidences = separated(200, 40)
    result = confidence_shuffle_p(hits, confidences, seed=0, n_permutations=PERMUTATION_RESAMPLES)
    self.assertEqual(result.n_permutations, 1000)
    self.assertAlmostEqual(result.p_value, 1.0 / 1001.0, places=12)
    self.assertFalse(result.no_information)

  def test_null_median_sits_on_the_no_information_band(self):
    n, n_errors = 400, 100
    hits = [1] * (n - n_errors) + [0] * n_errors
    curve = RiskCoverageCurve.from_outcomes(hits, [1.0] * n, seed=0)
    band = curve.aurc_random - curve.aurc_oracle
    result = confidence_shuffle_p(hits, [0.5] * n, seed=0, n_permutations=PERMUTATION_RESAMPLES)
    self.assertLess(abs(result.null_median - band), 0.02)
    # Permuting a constant changes nothing, so every permutation ties the
    # observation and the p-value is exactly 1: the strongest possible statement
    # that this confidence channel carries no information.
    self.assertEqual(result.p_value, 1.0)
    self.assertTrue(result.no_information)

  def test_zero_information_confidence_is_indistinguishable_from_always_acting(self):
    """Three ways of asking the same question, all of which must say "no evidence".

    Always acting is the constant-confidence control: it makes the same
    predictions and orders them at random. A zero-information confidence has to
    be indistinguishable from it, because the only thing it does differently is
    order the same errors in a different arbitrary way. If any of these three
    reported a number, the number would be the base error rate wearing a
    calibration label.
    """
    outcomes = uninformative_outcomes(400)
    ranked = [o for o in outcomes if not o.abstain]
    hits = [1 if o.hit else 0 for o in ranked]
    confidences = [float(o.confidence) for o in ranked]
    always_act = [1.0] * len(hits)

    zero_information = confidence_shuffle_p(
      hits, confidences, seed=0, n_permutations=PERMUTATION_RESAMPLES
    )
    constant = confidence_shuffle_p(
      hits, always_act, seed=0, n_permutations=PERMUTATION_RESAMPLES
    )
    self.assertEqual(zero_information.n_permutations, 1000)
    self.assertGreater(zero_information.p_value, 0.05)
    self.assertTrue(zero_information.no_information)
    self.assertEqual(constant.p_value, 1.0)
    self.assertTrue(constant.no_information)

    # Both sit in the same band, which is AURC_random - AURC_oracle and not zero.
    curve = RiskCoverageCurve.from_outcomes(hits, always_act, seed=0)
    band = curve.aurc_random - curve.aurc_oracle
    self.assertLess(abs(zero_information.observed - band), 0.10)
    self.assertLess(abs(constant.observed - band), 0.10)
    self.assertGreater(band, 0.2)

    # And screening on it buys nothing over screening on the same confidences
    # shuffled: same coverage, same base rate, no pairing.
    theta = float(THRESHOLD.value)
    shuffled = list(confidences)
    random.Random(99).shuffle(shuffled)
    screened = []
    screened_shuffled = []
    position = 0
    for outcome in outcomes:
      if outcome.abstain:
        screened.append(0.0)
        screened_shuffled.append(0.0)
        continue
      payoff = 1.0 if outcome.hit else HEADLINE_W_FAIL
      screened.append(payoff if float(outcome.confidence) >= theta else 0.0)
      screened_shuffled.append(payoff if shuffled[position] >= theta else 0.0)
      position += 1
    interval = paired_bootstrap_ci(screened, screened_shuffled, seed=0)
    self.assertEqual(interval.n_resamples, BOOTSTRAP_RESAMPLES)
    self.assertTrue(interval.includes_zero, interval.format())

  def test_report_marks_a_no_information_row_invalid(self):
    outcomes = uninformative_outcomes(400)
    report = score(outcomes, THRESHOLD, task_id="S3", horizon=1, seed=0)
    self.assertTrue(report.no_information)
    self.assertFalse(report.valid)
    self.assertFalse(report.rankable)
    self.assertIn("NO INFORMATION", report.format())


class ThresholdTest(unittest.TestCase):
  def test_an_undeclared_threshold_is_refused(self):
    with self.assertRaises(ThresholdNotFrozen):
      FrozenThreshold(value=0.5, calibrated_on="UNDECLARED", n_calibration=100)
    with self.assertRaises(ThresholdNotFrozen):
      FrozenThreshold(value=0.5, calibrated_on="   ", n_calibration=100)
    with self.assertRaises(ThresholdNotFrozen):
      FrozenThreshold(value=0.5, calibrated_on="dev", n_calibration=0)
    with self.assertRaises(ThresholdNotFrozen):
      FrozenThreshold(value=1.5, calibrated_on="dev", n_calibration=10)

  def test_score_will_not_take_a_bare_number(self):
    outcomes = informative_outcomes(60)
    with self.assertRaises(ThresholdNotFrozen):
      score(outcomes, 0.6, n_permutations=0)  # type: ignore[arg-type]

  def test_the_threshold_prints_verbatim(self):
    report = score(informative_outcomes(60), THRESHOLD, n_permutations=0)
    self.assertIn("dev seeds 0-99", report.format())


class ActDeferTest(unittest.TestCase):
  def table(self):
    return ActDeferTable(
      n_act_hit=60, n_act_miss=10, n_defer_hit=20, n_defer_miss=10, threshold=THRESHOLD
    )

  def test_hand_computed_two_by_two(self):
    table = self.table()
    self.assertEqual(table.n, 100)
    # ADA = (act & hit + defer & miss)/N = (60 + 10)/100
    self.assertAlmostEqual(table.ada.value, 0.70, places=15)
    self.assertAlmostEqual(table.unsafe_act_rate.value, 0.10, places=15)
    self.assertAlmostEqual(table.wasted_defer_rate.value, 0.20, places=15)

  def test_ada_never_prints_alone(self):
    text = self.table().format()
    self.assertIn("ADA:", text)
    self.assertIn("unsafe_act_rate:", text)
    self.assertIn("wasted_defer_rate:", text)
    self.assertIn("act   & hit", text)
    self.assertIn("defer & miss", text)

  def test_hand_computed_decision_utility_at_all_three_weights(self):
    utilities = decision_utility(self.table())
    self.assertEqual(tuple(u.w_fail for u in utilities), W_FAIL_WEIGHTS)
    # 60 correct acts, 10 incorrect acts, 100 episodes:
    #   w = -1  -> (60 - 10)/100  = +0.50
    #   w = -4  -> (60 - 40)/100  = +0.20
    #   w = -10 -> (60 - 100)/100 = -0.40
    self.assertAlmostEqual(utilities[0].u_over_n, 0.50, places=15)
    self.assertAlmostEqual(utilities[1].u_over_n, 0.20, places=15)
    self.assertAlmostEqual(utilities[2].u_over_n, -0.40, places=15)
    self.assertEqual(utilities[1].w_fail, HEADLINE_W_FAIL)
    self.assertIn("stipulated preference", utilities[1].format())
    self.assertNotIn("stipulated preference", utilities[0].format())

  def test_a_zero_threshold_acts_on_everything(self):
    outcomes = [EpisodeOutcome(f"e{i}", hit=(i % 4 != 0), confidence=0.5) for i in range(40)]
    always = FrozenThreshold(value=0.0, calibrated_on="always act, by construction",
                             n_calibration=1)
    report = score(outcomes, always, n_permutations=0)
    self.assertEqual(report.table.n_defer_hit, 0)
    self.assertEqual(report.table.n_defer_miss, 0)
    # every episode acted, so ADA collapses onto the base accuracy 30/40
    self.assertAlmostEqual(report.table.ada.value, 0.75, places=15)


class AbstentionCreditTest(unittest.TestCase):
  def credit(self):
    return AbstentionCredit(
      n=400,
      n_unknowable=60,
      n_abstained=48,
      n_unknowable_abstained=40,
      n_knowable_abstained=8,
      n_credited=240,
    )

  def test_acr_cannot_be_formatted_without_far(self):
    """Refusal rule R5, enforced by the type: there is no attribute to print alone."""
    credit = self.credit()
    self.assertFalse(hasattr(credit, "acr"))
    self.assertFalse(hasattr(credit, "refusal_recall"))
    acr, far = credit.credit_and_false_alarm()
    self.assertEqual(acr.name, "ACR")
    self.assertEqual(far.name, "FAR")
    # ACR = credited / N = 240/400
    self.assertAlmostEqual(acr.value, 0.60, places=15)
    # FAR denominator is the knowable episodes only: 8 / (400 - 60)
    self.assertAlmostEqual(far.value, 8.0 / 340.0, places=15)
    text = credit.format()
    self.assertIn("ACR:", text)
    self.assertIn("FAR:", text)

  def test_recall_and_precision_travel_together(self):
    credit = self.credit()
    recall, precision = credit.recall_and_precision()
    self.assertAlmostEqual(recall.value, 40.0 / 60.0, places=15)
    self.assertAlmostEqual(precision.value, 40.0 / 48.0, places=15)

  def test_precision_is_none_not_zero_when_nothing_was_refused(self):
    credit = AbstentionCredit(
      n=400, n_unknowable=60, n_abstained=0,
      n_unknowable_abstained=0, n_knowable_abstained=0, n_credited=280,
    )
    recall, precision = credit.recall_and_precision()
    self.assertEqual(recall.numerator, 0)
    self.assertIsNone(precision)
    self.assertIn("RefusalPrecision: undefined", credit.format())

  def test_outside_the_band_the_numbers_are_refused(self):
    """Refusal rule R6: below 5% unknowable these measure the base rate."""
    credit = AbstentionCredit(
      n=400, n_unknowable=4, n_abstained=4,
      n_unknowable_abstained=4, n_knowable_abstained=0, n_credited=280,
    )
    self.assertFalse(credit.in_band)
    with self.assertRaises(AbstentionBandRefused):
      credit.credit_and_false_alarm()
    with self.assertRaises(AbstentionBandRefused):
      credit.recall_and_precision()
    self.assertIn("REFUSED", credit.format())
    self.assertNotIn("ACR:", credit.format())

  def test_always_abstain_scores_the_base_rate_and_a_full_false_alarm(self):
    """The N7 control: abstaining on everything is not a free lunch."""
    outcomes = []
    for i in range(200):
      unknowable = i % 5 == 0
      outcomes.append(
        EpisodeOutcome(f"e{i}", hit=False, confidence=None, abstain=True, unknowable=unknowable)
      )
    report = score(outcomes, THRESHOLD, n_permutations=0)
    acr, far = report.credit.credit_and_false_alarm()
    self.assertAlmostEqual(acr.value, 0.20, places=15)  # exactly the unknowable base rate
    self.assertAlmostEqual(far.value, 1.0, places=15)
    self.assertIsNone(report.curve)
    self.assertIn("no ranked episodes", report.format())

  def test_a_report_that_prints_acr_always_prints_far(self):
    report = score(informative_outcomes(200), THRESHOLD, n_permutations=0)
    text = report.format()
    self.assertIn("ACR:", text)
    self.assertIn("FAR:", text)
    payload = report.to_dict()
    self.assertIn("ACR", payload["abstention"])
    self.assertIn("FAR", payload["abstention"])


class EpisodeOutcomeTest(unittest.TestCase):
  def test_abstaining_while_banking_a_prediction_is_refused(self):
    with self.assertRaises(ValueError):
      EpisodeOutcome("e0", hit=True, confidence=0.9, abstain=True)
    with self.assertRaises(ValueError):
      EpisodeOutcome("e0", hit=True, confidence=None, abstain=False)
    with self.assertRaises(ValueError):
      EpisodeOutcome("e0", hit=True, confidence=1.5)


class MinimumNTest(unittest.TestCase):
  """Section 3.3: two tiers, and no parameter anywhere that lifts either one."""

  def test_twenty_nine_episodes_refuse_to_produce_a_number(self):
    outcomes = informative_outcomes(29, abstain_on_unknowable=False)
    report = score(outcomes, THRESHOLD, task_id="S1", horizon=1, n_permutations=0)
    self.assertEqual((report.n, report.n_ranked), (29, 29))
    text = report.format()
    self.assertIn("UNDERPOWERED n=29", text)
    for forbidden in ("E-AURC", "AURC", "ADA", "ACR", "FAR", "Wilson", "U/N"):
      self.assertNotIn(forbidden, text)
    payload = report.to_dict()
    for key in ("e_aurc", "aurc", "aurc_oracle", "aurc_random", "act_defer", "abstention",
                "utility", "permutation", "spearman_rho", "coverage_at_risk"):
      self.assertEqual(payload[key], "UNDERPOWERED n=29")
    self.assertFalse(report.valid)
    self.assertFalse(report.rankable)

  def test_thirty_episodes_print_provisionally_and_are_not_rankable(self):
    outcomes = informative_outcomes(30, abstain_on_unknowable=False)
    report = score(outcomes, THRESHOLD, task_id="S1", horizon=1, n_permutations=200)
    text = report.format()
    self.assertEqual((report.n, report.n_ranked), (30, 30))
    self.assertEqual(report.tier, TIER_PROVISIONAL)
    self.assertIn(TIER_PROVISIONAL, text)
    self.assertIn("E-AURC", text)
    self.assertIn("Wilson", text)
    self.assertTrue(report.valid)
    self.assertFalse(report.rankable)

  def test_two_hundred_episodes_are_rankable(self):
    outcomes = informative_outcomes(200, abstain_on_unknowable=False)
    report = score(outcomes, THRESHOLD, task_id="S1", horizon=1, n_permutations=200)
    self.assertEqual((report.n, report.n_ranked), (200, 200))
    self.assertEqual(report.tier, TIER_RANKABLE)
    self.assertEqual(report.curve_tier, TIER_RANKABLE)
    self.assertTrue(report.rankable)
    self.assertIn("rankable: True", report.format())
    self.assertNotIn(TIER_PROVISIONAL, report.headline())

  def test_a_shuffle_control_that_did_not_run_is_not_a_pass(self):
    """Skipping the control is no information, not a clean bill of health."""
    outcomes = informative_outcomes(200, abstain_on_unknowable=False)
    report = score(outcomes, THRESHOLD, task_id="S1", horizon=1, n_permutations=0)
    self.assertFalse(report.shuffle_control_ran)
    self.assertTrue(report.no_information)
    self.assertFalse(report.valid)
    self.assertFalse(report.rankable)
    self.assertIn("was not run", report.format())

  def test_a_rankable_episode_count_with_too_few_ranked_is_not_rankable(self):
    """Abstaining hard buys the rates their power and costs the curve its own."""
    outcomes = informative_outcomes(240, unknowable_every=3)
    report = score(outcomes, THRESHOLD, task_id="S1", horizon=1, n_permutations=0)
    self.assertEqual(report.tier, TIER_RANKABLE)
    self.assertEqual(report.curve_tier, TIER_PROVISIONAL)
    self.assertFalse(report.rankable)

  def test_there_is_no_escape_hatch(self):
    """No parameter named allow_underpowered exists on anything in this module."""
    import inspect

    from plr_lr.learn import selective

    for name, obj in vars(selective).items():
      if name.startswith("_"):
        continue
      if inspect.isfunction(obj):
        self.assertNotIn("allow_underpowered", inspect.signature(obj).parameters)
    source = (REPO_ROOT / "plr_lr" / "learn" / "selective.py").read_text()
    self.assertNotIn("allow_underpowered", source)


class HeadlineTest(unittest.TestCase):
  def test_both_bands_print_in_the_headline_row(self):
    report = score(informative_outcomes(200), THRESHOLD, n_permutations=200)
    headline = report.headline()
    self.assertIn("E-AURC", headline)
    self.assertIn("AURC", headline)
    self.assertIn("oracle", headline)
    self.assertIn("random", headline)
    self.assertIn("no-information band", headline)


class SpearmanTest(unittest.TestCase):
  def test_perfect_inverse_ranking(self):
    self.assertAlmostEqual(spearman_rho([1, 2, 3, 4, 5], [5, 4, 3, 2, 1]), -1.0, places=12)
    self.assertAlmostEqual(spearman_rho([1, 2, 3, 4, 5], [1, 2, 3, 4, 5]), 1.0, places=12)

  def test_a_constant_column_has_no_correlation_to_report(self):
    self.assertIsNone(spearman_rho([1, 1, 1, 1], [1, 2, 3, 4]))
    self.assertIsNone(spearman_rho([1.0], [2.0]))

  def test_a_weak_correlation_prints_not_informative(self):
    outcomes = uninformative_outcomes(400, seed=3)
    report = score(outcomes, THRESHOLD, n_permutations=0)
    self.assertLess(abs(report.rho), 0.1)
    self.assertIn("NOT-INFORMATIVE", report.rho_cell())

  def test_confidence_that_orders_errors_reports_a_number(self):
    report = score(informative_outcomes(400), THRESHOLD, n_permutations=0)
    self.assertLess(report.rho, -0.1)  # high confidence, low error
    self.assertNotIn("NOT-INFORMATIVE", report.rho_cell())


class KolmogorovSmirnovTest(unittest.TestCase):
  def test_exact_p_on_a_hand_countable_case(self):
    # Two fully separated samples of three. D = 1.0. Of the C(6,3) = 20 lattice
    # paths, exactly two ever reach |i - j| = 3: all of one sample before any of
    # the other, in either order. So P(D >= 1) = 2/20 = 0.1.
    result = ks_two_sample([1.0, 2.0, 3.0], [4.0, 5.0, 6.0])
    self.assertTrue(result.exact)
    self.assertAlmostEqual(result.statistic, 1.0, places=15)
    self.assertAlmostEqual(result.p_value, 0.1, places=12)

  def test_identical_samples(self):
    result = ks_two_sample([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    self.assertEqual(result.statistic, 0.0)
    self.assertEqual(result.p_value, 1.0)

  def test_exactness_switches_at_thirty_five(self):
    small = ks_two_sample(list(range(34)), [v + 0.5 for v in range(34)])
    self.assertTrue(small.exact)
    large = ks_two_sample(list(range(35)), [v + 0.5 for v in range(35)])
    self.assertFalse(large.exact)
    self.assertGreaterEqual(large.p_value, 0.0)
    self.assertLessEqual(large.p_value, 1.0)

  def test_empty_sample_is_refused(self):
    with self.assertRaises(ValueError):
      ks_two_sample([], [1.0])


class PairedBootstrapTest(unittest.TestCase):
  def test_identical_columns_give_an_interval_on_zero(self):
    values = [float(i % 3) for i in range(50)]
    interval = paired_bootstrap_ci(values, list(values), n_resamples=1000, seed=0)
    self.assertEqual(interval.point, 0.0)
    self.assertEqual((interval.low, interval.high), (0.0, 0.0))
    self.assertTrue(interval.includes_zero)

  def test_a_constant_offset_excludes_zero(self):
    a = [float(i % 5) for i in range(50)]
    b = [v - 1.0 for v in a]
    interval = paired_bootstrap_ci(a, b, n_resamples=1000, seed=0)
    self.assertAlmostEqual(interval.point, 1.0, places=12)
    self.assertFalse(interval.includes_zero)

  def test_the_seed_is_recorded_and_reproduces(self):
    a = [float((i * 7) % 11) for i in range(60)]
    b = [float((i * 5) % 13) for i in range(60)]
    first = paired_bootstrap_ci(a, b, n_resamples=500, seed=11)
    second = paired_bootstrap_ci(a, b, n_resamples=500, seed=11)
    self.assertEqual(first, second)
    self.assertEqual(first.seed, 11)
    self.assertIn("seed 11", first.format())

  def test_mismatched_lengths_are_refused(self):
    with self.assertRaises(ValueError):
      paired_bootstrap_ci([1.0, 2.0], [1.0])


class StdlibOnlyTest(unittest.TestCase):
  """The scoring core stays importable with nothing installed, and stays honest."""

  # Assembled from halves so this guard does not trip over its own source text.
  BANNED = ("percent" + "ile", "quant" + "ile", "arg" + "max")

  #: Everything the scoring core is allowed to import. numpy and pylabrobot are
  #: absent on purpose: the standalone copy has to run where neither is installed.
  ALLOWED = {"__future__", "dataclasses", "json", "math", "random", "sys", "typing"}

  def imported_modules(self, path: Path):
    tree = ast.parse(path.read_text())
    modules = set()
    for node in ast.walk(tree):
      if isinstance(node, ast.Import):
        for alias in node.names:
          modules.add(alias.name.split(".")[0])
      elif isinstance(node, ast.ImportFrom):
        if node.level:
          modules.add(f"<relative import of {node.module}>")
        elif node.module:
          modules.add(node.module.split(".")[0])
    return modules

  def test_selective_imports_nothing_beyond_the_standard_library(self):
    modules = self.imported_modules(REPO_ROOT / "plr_lr" / "learn" / "selective.py")
    self.assertEqual(modules - self.ALLOWED, set())

  def test_standalone_imports_nothing_beyond_the_standard_library(self):
    modules = self.imported_modules(STANDALONE)
    self.assertEqual(modules - self.ALLOWED, set())
    self.assertNotIn("plr_lr", modules)

  def test_no_data_derived_cut_points_anywhere(self):
    """Refusal rule R9: the portfolio's own documented way of manufacturing numbers."""
    for path in (REPO_ROOT / "plr_lr" / "learn" / "selective.py", STANDALONE):
      source = path.read_text()
      for token in self.BANNED:
        self.assertNotIn(token, source, f"{token} in {path.name}")

  def test_both_files_are_ascii(self):
    for path in (REPO_ROOT / "plr_lr" / "learn" / "selective.py", STANDALONE,
                 Path(__file__).resolve()):
      path.read_bytes().decode("ascii")


class StandaloneAgreementTest(unittest.TestCase):
  """The duplicated scorer must agree with the package, or the copy has drifted."""

  def payload(self):
    outcomes = informative_outcomes(200, seed=1)
    episodes = []
    for outcome in outcomes:
      episodes.append({
        "episode_id": outcome.episode_id,
        "hit": bool(outcome.hit),
        "confidence": None if outcome.confidence is None else float(outcome.confidence),
        "abstain": bool(outcome.abstain),
        "unknowable": bool(outcome.unknowable),
      })
    return {
      "task_id": "S2",
      "horizon": 2,
      "seed": 0,
      "threshold": {
        "value": 0.6,
        "calibrated_on": "dev seeds 0-99",
        "n_calibration": 100,
      },
      "episodes": episodes,
    }, outcomes

  def run_standalone(self, path, *arguments, stdin_text=None):
    command = [sys.executable, str(STANDALONE)]
    if path is not None:
      command.append(str(path))
    command.extend(arguments)
    result = subprocess.run(
      command,
      input=stdin_text,
      capture_output=True,
      text=True,
      check=False,
    )
    self.assertEqual(result.returncode, 0, result.stderr)
    return result.stdout

  def test_text_and_json_agree_with_the_package(self):
    payload, outcomes = self.payload()
    expected = score(outcomes, THRESHOLD, task_id="S2", horizon=2, seed=0,
                     n_permutations=PERMUTATION_RESAMPLES)
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "results.json"
      path.write_text(json.dumps(payload))
      text = self.run_standalone(path)
      as_json = self.run_standalone(path, "--json")
      from_stdin = self.run_standalone(None, stdin_text=json.dumps(payload))
    self.assertEqual(text.strip(), expected.format())
    self.assertEqual(from_stdin.strip(), expected.format())
    self.assertEqual(
      json.loads(as_json),
      json.loads(json.dumps(expected.to_dict(), sort_keys=True)),
    )

  def test_the_standalone_refuses_a_file_with_no_threshold(self):
    payload, _ = self.payload()
    del payload["threshold"]
    with tempfile.TemporaryDirectory() as directory:
      path = Path(directory) / "results.json"
      path.write_text(json.dumps(payload))
      result = subprocess.run(
        [sys.executable, str(STANDALONE), str(path)],
        capture_output=True,
        text=True,
        check=False,
      )
    self.assertNotEqual(result.returncode, 0)
    self.assertIn("REFUSED", result.stderr)
    self.assertIn("threshold", result.stderr)
    self.assertNotIn("Traceback", result.stderr)
    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
  unittest.main()
