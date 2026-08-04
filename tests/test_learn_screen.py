"""Tests for plr_lr.learn.screen, the gate between a prediction and a motion.

Organised around the guarantees rather than the functions, because the
guarantees are what a refactor drops. They are: that admission needs all five
conditions and no four of them will do; that every one of the seven abstention
reasons is reachable by a real call; that the threshold is frozen, is printed
with its provenance, and refuses to decide the episodes it was fitted on; that a
clamped command loses its verdict rather than inheriting it; that a deferral ends
the run instead of producing a quieter action; and that all six null screens fail.

The conformal arithmetic is checked against numbers worked out by hand rather
than against the implementation's own output, because a calibration test that
asserts what the code did is a test that will keep passing after the code is
wrong.
"""

from __future__ import annotations

import inspect
import math
import unittest

import numpy as np

from plr_lr.learn.predict import FrameRef, RolloutPrediction, RolloutRequest
from plr_lr.learn.readout import ReadoutResult
from plr_lr.learn.screen import (
  ALPHA,
  MAX_REOBSERVATIONS,
  MIN_CALIBRATION,
  REASON_PRECEDENCE,
  REFUSALS,
  UNCALIBRATED,
  AbstainReason,
  AlwaysAbstainScreen,
  AlwaysAdmitScreen,
  CalibrationPoint,
  CeilingRefusal,
  ClampOnlyScreen,
  ConstantConfidenceScreen,
  NotCalibrated,
  PredictionScreen,
  ReasonBlindScreen,
  Screen,
  ScreenDeclined,
  ScreenOutcome,
  ScreenReport,
  ScreenedPolicy,
  StaleBlindScreen,
  StepScreen,
  default_null_screens,
  screen_case,
  screen_null_check,
)
from plr_lr.learn.types import ACTION_FIELDS, Action, ActionMode, Observation

TRUST_HORIZON = 4
MAX_FRAME_AGE_S = 0.5
FAILED_READOUT = ReadoutResult.refused("marker", "underdetermined", 1)
GOOD_READOUT = ReadoutResult(
  name="marker", ok=True, state=np.zeros(4), reason="measured", n_markers=3, residual_mm=0.0
)


def calibrated(theta_source=None, **kwargs) -> Screen:
  """A screen whose theta lands at 0.80, built through the real calibration path."""
  points = [CalibrationPoint("cal%d" % i, 0.80 + 0.01 * i) for i in range(12)]
  return Screen.calibrate(
    theta_source or points,
    source=kwargs.pop("source", "dev seeds 0-99"),
    trust_horizon=kwargs.pop("trust_horizon", TRUST_HORIZON),
    max_frame_age_s=kwargs.pop("max_frame_age_s", MAX_FRAME_AGE_S),
    **kwargs,
  )


def verdict(screen: Screen, case) -> str:
  report = screen.check(
    case.prediction,
    case.request,
    now_s=case.now_s,
    readout=case.readout,
    clamped=case.clamped,
    clamp_magnitude_mm=case.clamp_magnitude_mm,
    ceiling=case.ceiling,
  )
  return "ACT" if report.admitted else "ABSTAIN(%s)" % report.reason.value


def suite_cases():
  """One case per admission condition, plus one clean pass to contrast them with."""
  return [
    screen_case("clean", 0.95),
    screen_case("low_confidence", 0.10),
    screen_case("stale", 0.95, now_s=5.0, t_capture_s=1.0),
    screen_case("horizon", 0.95, horizon=TRUST_HORIZON + 1),
    screen_case("readout", 0.95, readout=FAILED_READOUT),
    screen_case("clamped", 0.95, clamped=True, ceiling=False),
    screen_case("ceiling", 0.95, ceiling=True),
    screen_case("adapter_abstained", 0.0, abstain=True),
  ]


class StubPredictor:
  """Returns a scripted confidence per call. Enough to drive the screen honestly."""

  name = "stub"
  state_dim = 4
  horizon = 1
  action_space = "ee_delta_mm"
  training_fingerprints = frozenset({"none:analytic-null-fit-on-no-data"})
  fitted_components = ()
  license_text = "MIT, this test."

  def __init__(self, confidences):
    self.confidences = list(confidences)
    self.calls = 0

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    i = min(self.calls, len(self.confidences) - 1)
    self.calls += 1
    value = self.confidences[i]
    if value is None:
      return RolloutPrediction.abstained(int(request.horizon), 4)
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


class MovingPolicy:
  """Always asks for the same 5 mm step, so a fallback would be visible as motion."""

  def act(self, observation: Observation) -> Action:
    return Action.delta(5.0, 0.0, 0.0)


class TestTheEnumsAreClosed(unittest.TestCase):
  """Two outcomes, seven reasons, and every reason ranked."""

  def test_there_are_exactly_two_outcomes(self):
    self.assertEqual({o.value for o in ScreenOutcome}, {"act", "abstain"})

  def test_the_seven_reasons_are_the_ones_the_design_names(self):
    self.assertEqual(
      {r.name for r in AbstainReason},
      {
        "LOW_CONFIDENCE",
        "READOUT_FAILED",
        "HORIZON_EXCEEDED",
        "CLAMPED",
        "STALE",
        "NOT_CALIBRATED",
        "CEILING",
      },
    )

  def test_every_reason_has_a_place_in_the_precedence_order(self):
    """A reason missing from the order would silently never be reported first."""
    self.assertEqual(set(REASON_PRECEDENCE), set(AbstainReason))
    self.assertEqual(len(REASON_PRECEDENCE), len(set(REASON_PRECEDENCE)))


class TestCalibration(unittest.TestCase):
  """Split conformal on successes, with the arithmetic checked against hand values."""

  def test_the_threshold_matches_a_hand_computed_conformal_quantile(self):
    """n=19 at alpha=0.10 puts the rank at 18, so theta is the second smallest.

    ceil((19 + 1) * 0.90) = 18. The nonconformity scores are 1 - c sorted
    ascending, so the 18th of 19 is the second largest score, which is the second
    smallest confidence. With confidences 0.01 through 0.19 that is 0.02.
    """
    points = [CalibrationPoint("e%d" % i, i / 100.0) for i in range(1, 20)]
    screen = Screen.calibrate(
      points, source="dev", trust_horizon=TRUST_HORIZON, max_frame_age_s=MAX_FRAME_AGE_S
    )
    self.assertEqual(screen.n_calibration, 19)
    self.assertAlmostEqual(screen.theta, 0.02, places=12)

  def test_at_the_floor_the_threshold_is_the_least_confident_success(self):
    """n=10 at alpha=0.10 gives rank 10, the whole sample, so theta is the minimum.

    This is the honest statement of what ten episodes buy: not a quantile, just
    "at least as confident as the worst success we saw".
    """
    points = [CalibrationPoint("e%d" % i, 0.50 + i / 100.0) for i in range(10)]
    screen = Screen.calibrate(
      points, source="dev", trust_horizon=TRUST_HORIZON, max_frame_age_s=MAX_FRAME_AGE_S
    )
    self.assertAlmostEqual(screen.theta, 0.50, places=12)

  def test_the_threshold_admits_all_but_alpha_of_the_successes_it_was_fitted_on(self):
    """The coverage claim, checked on the calibration set itself.

    Also a regression test for a measured floating-point defect. Deriving theta as
    ``1 - sorted(1 - c)[rank - 1]`` lands it an ulp above the calibration point it
    came from at c = 0.02, so the screen refused one of its own successes and the
    realised miscoverage came out at 2/19 against a claimed 1/10. Theta is taken
    as an order statistic of the confidences instead.
    """
    points = [CalibrationPoint("e%d" % i, i / 100.0) for i in range(1, 20)]
    screen = Screen.calibrate(
      points, source="dev", trust_horizon=TRUST_HORIZON, max_frame_age_s=MAX_FRAME_AGE_S
    )
    refused = sum(1 for p in points if not screen.check_theta(p.confidence))
    self.assertEqual(refused, 1)
    self.assertLessEqual(refused / len(points), ALPHA)
    self.assertIn(screen.theta, [p.confidence for p in points])

  def test_alpha_and_min_calibration_are_absolute_constants_in_the_signature(self):
    """They are arguments only so a receipt can record them, never to be tuned."""
    params = inspect.signature(Screen.calibrate).parameters
    self.assertEqual(params["alpha"].default, ALPHA)
    self.assertEqual(params["min_calibration"].default, MIN_CALIBRATION)
    self.assertEqual(ALPHA, 0.10)
    self.assertEqual(MIN_CALIBRATION, 10)

  def test_too_few_successes_refuses_rather_than_returning_a_permissive_default(self):
    points = [CalibrationPoint("e%d" % i, 0.9) for i in range(MIN_CALIBRATION - 1)]
    with self.assertRaises(NotCalibrated) as ctx:
      Screen.calibrate(
        points, source="dev", trust_horizon=TRUST_HORIZON, max_frame_age_s=MAX_FRAME_AGE_S
      )
    self.assertIn("admits everything", str(ctx.exception))

  def test_an_alpha_too_small_for_the_sample_refuses(self):
    """The conformal rank falls outside the sample, so no finite theta carries it."""
    points = [CalibrationPoint("e%d" % i, 0.9) for i in range(10)]
    with self.assertRaises(NotCalibrated) as ctx:
      Screen.calibrate(
        points,
        source="dev",
        trust_horizon=TRUST_HORIZON,
        max_frame_age_s=MAX_FRAME_AGE_S,
        alpha=0.01,
      )
    self.assertIn("outside the sample", str(ctx.exception))

  def test_a_duplicated_episode_is_refused(self):
    points = [CalibrationPoint("same", 0.9) for _ in range(12)]
    with self.assertRaises(ValueError) as ctx:
      Screen.calibrate(
        points, source="dev", trust_horizon=TRUST_HORIZON, max_frame_age_s=MAX_FRAME_AGE_S
      )
    self.assertIn("more than once", str(ctx.exception))

  def test_bare_floats_are_refused_because_they_carry_no_episode_identity(self):
    with self.assertRaises(ValueError) as ctx:
      Screen.calibrate(
        [0.9] * 12, source="dev", trust_horizon=TRUST_HORIZON, max_frame_age_s=MAX_FRAME_AGE_S
      )
    self.assertIn("no episode identity", str(ctx.exception))

  def test_an_abstained_episode_is_not_a_calibration_point(self):
    with self.assertRaises(ValueError):
      CalibrationPoint("e", float("nan"))
    with self.assertRaises(ValueError):
      CalibrationPoint("e", 1.5)

  def test_the_source_must_be_named_and_is_printed_verbatim(self):
    for bad in ("", UNCALIBRATED):
      with self.subTest(source=bad):
        with self.assertRaises(ValueError):
          Screen.calibrate(
            [CalibrationPoint("e%d" % i, 0.9) for i in range(12)],
            source=bad,
            trust_horizon=TRUST_HORIZON,
            max_frame_age_s=MAX_FRAME_AGE_S,
          )
    screen = calibrated(source="dev seeds 0-99, xArm7 bench, 2026-08-03")
    self.assertIn("dev seeds 0-99, xArm7 bench, 2026-08-03", screen.calibrated_on)
    self.assertIn("split-conformal on successes", screen.calibrated_on)
    self.assertIn("alpha=0.1000", screen.calibrated_on)


class TestScreenConfiguration(unittest.TestCase):
  """A screen that cannot state its threshold's provenance is not a screen."""

  def test_a_numeric_theta_with_no_provenance_is_refused(self):
    with self.assertRaises(ValueError) as ctx:
      Screen(
        theta=0.0,
        trust_horizon=1,
        max_frame_age_s=1.0,
        calibrated_on=UNCALIBRATED,
      )
    self.assertIn("quoted as a calibration", str(ctx.exception))

  def test_theta_outside_the_unit_interval_is_refused(self):
    for theta in (-0.1, 1.1, float("inf")):
      with self.subTest(theta=theta):
        with self.assertRaises(ValueError):
          Screen(theta=theta, trust_horizon=1, max_frame_age_s=1.0, calibrated_on="dev")

  def test_a_trust_horizon_below_one_is_refused(self):
    with self.assertRaises(ValueError) as ctx:
      Screen(theta=0.5, trust_horizon=0, max_frame_age_s=1.0, calibrated_on="dev")
    self.assertIn("reason not to build a screen", str(ctx.exception))

  def test_the_staleness_bound_has_no_default_and_must_be_positive(self):
    params = inspect.signature(Screen.calibrate).parameters
    self.assertIs(params["max_frame_age_s"].default, inspect.Parameter.empty)
    self.assertIs(params["trust_horizon"].default, inspect.Parameter.empty)
    for bound in (0.0, -1.0, float("nan")):
      with self.subTest(bound=bound):
        with self.assertRaises(ValueError):
          Screen(theta=0.5, trust_horizon=1, max_frame_age_s=bound, calibrated_on="dev")

  def test_an_uncalibrated_screen_refuses_everything_it_is_shown(self):
    screen = Screen.uncalibrated()
    self.assertFalse(screen.is_calibrated)
    self.assertTrue(math.isnan(screen.theta))
    for case in suite_cases():
      with self.subTest(case=case.label):
        report = screen.check(case.prediction, case.request, now_s=case.now_s, readout=None)
        self.assertFalse(report.admitted)
        self.assertIs(report.reason, AbstainReason.NOT_CALIBRATED)

  def test_check_theta_refuses_on_a_nan_rather_than_admitting_on_one(self):
    """The negated comparison would admit both of these. This one refuses both."""
    self.assertFalse(Screen.uncalibrated().check_theta(1.0))
    self.assertFalse(calibrated().check_theta(float("nan")))
    self.assertTrue(calibrated().check_theta(0.80))


class TestAdmissionNeedsAllFiveConditions(unittest.TestCase):
  """Each condition, alone, is enough to refuse. Together they are the gate."""

  def setUp(self):
    self.screen = calibrated()

  def test_a_clean_case_is_admitted(self):
    case = screen_case("clean", 0.95)
    report = self.screen.check(case.prediction, case.request, now_s=1.0, readout=None)
    self.assertTrue(report.admitted)
    self.assertEqual(report.reasons, ())
    self.assertIsNone(report.reason)

  def test_confidence_below_theta_refuses(self):
    case = screen_case("low", 0.79)
    report = self.screen.check(case.prediction, case.request, now_s=1.0, readout=None)
    self.assertIs(report.reason, AbstainReason.LOW_CONFIDENCE)
    self.assertIn("below the frozen threshold", " ".join(report.notes))

  def test_confidence_exactly_at_theta_is_admitted(self):
    case = screen_case("edge", 0.80)
    self.assertTrue(
      self.screen.check(case.prediction, case.request, now_s=1.0, readout=None).admitted
    )

  def test_an_adapter_that_abstained_is_not_overridden(self):
    case = screen_case("refused", 0.0, abstain=True)
    report = self.screen.check(case.prediction, case.request, now_s=1.0, readout=None)
    self.assertIs(report.reason, AbstainReason.LOW_CONFIDENCE)
    self.assertTrue(math.isnan(report.confidence))
    self.assertIn("a refusal upstream is not overridden here", " ".join(report.notes))

  def test_a_horizon_past_the_measured_trust_horizon_refuses(self):
    case = screen_case("far", 0.95, horizon=TRUST_HORIZON + 1)
    report = self.screen.check(case.prediction, case.request, now_s=1.0, readout=None)
    self.assertIs(report.reason, AbstainReason.HORIZON_EXCEEDED)
    case_ok = screen_case("near", 0.95, horizon=TRUST_HORIZON)
    self.assertTrue(
      self.screen.check(case_ok.prediction, case_ok.request, now_s=1.0, readout=None).admitted
    )

  def test_a_failed_readout_refuses_and_names_the_readout(self):
    case = screen_case("blind", 0.95)
    report = self.screen.check(
      case.prediction, case.request, now_s=1.0, readout=FAILED_READOUT
    )
    self.assertIs(report.reason, AbstainReason.READOUT_FAILED)
    self.assertFalse(report.readout_ok)
    self.assertEqual(report.readout, "marker")

  def test_a_successful_readout_is_recorded_and_admits(self):
    case = screen_case("seen", 0.95)
    report = self.screen.check(case.prediction, case.request, now_s=1.0, readout=GOOD_READOUT)
    self.assertTrue(report.admitted)
    self.assertTrue(report.readout_ok)

  def test_a_frame_older_than_the_declared_bound_refuses(self):
    case = screen_case("old", 0.95, t_capture_s=1.0)
    report = self.screen.check(
      case.prediction, case.request, now_s=1.0 + MAX_FRAME_AGE_S + 1e-6, readout=None
    )
    self.assertIs(report.reason, AbstainReason.STALE)
    self.assertAlmostEqual(report.frame_age_s, MAX_FRAME_AGE_S + 1e-6, places=9)

  def test_a_frame_exactly_at_the_bound_is_admitted(self):
    case = screen_case("edge", 0.95, t_capture_s=1.0)
    report = self.screen.check(
      case.prediction, case.request, now_s=1.0 + MAX_FRAME_AGE_S, readout=None
    )
    self.assertTrue(report.admitted)

  def test_a_frame_stamped_in_the_future_refuses_as_a_clock_mismatch(self):
    case = screen_case("future", 0.95, t_capture_s=10.0)
    report = self.screen.check(case.prediction, case.request, now_s=1.0, readout=None)
    self.assertIs(report.reason, AbstainReason.STALE)
    self.assertIn("not the same clock", " ".join(report.notes))

  def test_a_request_with_no_frame_is_refused_for_being_undatable(self):
    request = RolloutRequest(
      task_id="t",
      episode_id="e",
      state=np.zeros(4),
      state_fields=("x", "y", "z", "rz"),
      state_units=("mm", "mm", "mm", "rad"),
      frames=(),
      actions=np.zeros((1, len(ACTION_FIELDS))),
      horizon=1,
      tolerance=np.array([2.0, 2.0, 2.0, 0.05]),
    )
    case = screen_case("dated", 0.95)
    report = self.screen.check(case.prediction, request, now_s=1.0, readout=None)
    self.assertIs(report.reason, AbstainReason.STALE)
    self.assertIn("undatable", " ".join(report.notes))
    self.assertTrue(math.isnan(report.frame_age_s))

  def test_the_newest_frame_decides_staleness_not_the_first_one(self):
    request = RolloutRequest(
      task_id="t",
      episode_id="e",
      state=np.zeros(4),
      state_fields=("x", "y", "z", "rz"),
      state_units=("mm", "mm", "mm", "rad"),
      frames=(
        FrameRef("old.png", "sha256:old", 0.0, "rendered"),
        FrameRef("new.png", "sha256:new", 1.0, "rendered"),
      ),
      actions=np.zeros((1, len(ACTION_FIELDS))),
      horizon=1,
      tolerance=np.array([2.0, 2.0, 2.0, 0.05]),
    )
    case = screen_case("multi", 0.95)
    report = self.screen.check(case.prediction, request, now_s=1.2, readout=None)
    self.assertTrue(report.admitted)
    self.assertAlmostEqual(report.frame_age_s, 0.2, places=9)

  def test_the_ceiling_refuses_separately_from_a_low_confidence(self):
    case = screen_case("invisible", 0.99, ceiling=True)
    report = self.screen.check(
      case.prediction, case.request, now_s=1.0, readout=None, ceiling=True
    )
    self.assertIs(report.reason, AbstainReason.CEILING)
    self.assertIn("not visible to the camera", " ".join(report.notes))

  def test_a_clamped_command_refuses(self):
    case = screen_case("clamped", 0.99)
    report = self.screen.check(
      case.prediction, case.request, now_s=1.0, readout=None, clamped=True,
      clamp_magnitude_mm=7.5,
    )
    self.assertIs(report.reason, AbstainReason.CLAMPED)
    self.assertAlmostEqual(report.clamp_magnitude_mm, 7.5, places=12)

  def test_the_readout_argument_has_no_default(self):
    """Passing None is a decision. Forgetting it must not be the permissive branch."""
    params = inspect.signature(Screen.check).parameters
    self.assertIs(params["readout"].default, inspect.Parameter.empty)
    self.assertIs(params["now_s"].default, inspect.Parameter.empty)


class TestABrokenRowInvariantNeverAdmits(unittest.TestCase):
  """A prediction that breaks the abstain/NaN contract is refused, not trusted.

  ``plr_lr.learn.predict.validate`` is where the row invariant is enforced, and
  the screen is not a second copy of it. What the screen guarantees is weaker and
  sufficient: whatever shape a malformed prediction arrives in, it is not
  admitted. The screen sits in front of an arm, so the one outcome it may never
  produce from a contradictory input is permission to move.
  """

  def setUp(self):
    self.screen = calibrated()
    self.request = screen_case("row", 0.99).request

  def broken(self, confidence, abstain):
    return RolloutPrediction(
      state=np.zeros((1, 4)),
      confidence=confidence,
      abstain=abstain,
      flag=False,
      readout="identity",
      n_forward_passes=1,
      wall_clock_s=0.0,
    )

  def test_abstain_while_carrying_a_high_confidence_still_refuses(self):
    report = self.screen.check(
      self.broken(0.99, True), self.request, now_s=1.0, readout=None
    )
    self.assertFalse(report.admitted)
    self.assertIs(report.reason, AbstainReason.LOW_CONFIDENCE)
    self.assertIn("breaks the row invariant", " ".join(report.notes))

  def test_a_nan_confidence_without_abstaining_is_not_treated_as_a_high_one(self):
    report = self.screen.check(
      self.broken(float("nan"), False), self.request, now_s=1.0, readout=None
    )
    self.assertFalse(report.admitted)
    self.assertIs(report.reason, AbstainReason.LOW_CONFIDENCE)
    self.assertIn("never treated as a high one", " ".join(report.notes))

  def test_a_non_numeric_confidence_refuses_rather_than_raising_mid_run(self):
    report = self.screen.check(
      self.broken("very", False), self.request, now_s=1.0, readout=None
    )
    self.assertFalse(report.admitted)
    self.assertTrue(math.isnan(report.confidence))


class TestTheThresholdNeverSeesTheDataItJudges(unittest.TestCase):
  """A screen refuses to decide the episodes it was fitted on."""

  def test_a_calibration_episode_is_refused_as_not_calibrated(self):
    screen = calibrated()
    case = screen_case("cal0", 0.99, episode_id="cal0")
    report = screen.check(case.prediction, case.request, now_s=1.0, readout=None)
    self.assertFalse(report.admitted)
    self.assertIs(report.reason, AbstainReason.NOT_CALIBRATED)
    self.assertIn("already seen its answer", " ".join(report.notes))

  def test_an_episode_outside_the_calibration_set_is_decided_normally(self):
    screen = calibrated()
    case = screen_case("scoring-7", 0.99, episode_id="scoring-7")
    self.assertTrue(
      screen.check(case.prediction, case.request, now_s=1.0, readout=None).admitted
    )

  def test_the_calibration_ids_are_recorded_on_the_screen(self):
    screen = calibrated()
    self.assertEqual(screen.n_calibration, 12)
    self.assertIn("cal0", screen.calibration_ids)
    self.assertIsInstance(screen.calibration_ids, frozenset)


class TestReportInvariants(unittest.TestCase):
  """A verdict cannot be admitted and carry a reason, or refuse and carry none."""

  def test_act_with_a_reason_is_refused_at_construction(self):
    with self.assertRaises(ValueError):
      ScreenReport(
        outcome=ScreenOutcome.ACT,
        reasons=(AbstainReason.STALE,),
        confidence=0.9,
        theta=0.8,
        calibrated_on="dev",
        horizon=1,
        trust_horizon=2,
        frame_age_s=0.0,
        max_frame_age_s=1.0,
        readout_ok=True,
      )

  def test_abstain_with_no_reason_is_refused_at_construction(self):
    with self.assertRaises(ValueError) as ctx:
      ScreenReport(
        outcome=ScreenOutcome.ABSTAIN,
        reasons=(),
        confidence=0.9,
        theta=0.8,
        calibrated_on="dev",
        horizon=1,
        trust_horizon=2,
        frame_age_s=0.0,
        max_frame_age_s=1.0,
        readout_ok=True,
      )
    self.assertIn("indistinguishable from a crash", str(ctx.exception))

  def test_every_reason_that_applied_is_recorded_not_only_the_first(self):
    screen = calibrated()
    case = screen_case("everything", 0.10, horizon=TRUST_HORIZON + 1, now_s=99.0)
    report = screen.check(
      case.prediction,
      case.request,
      now_s=99.0,
      readout=FAILED_READOUT,
      clamped=True,
      clamp_magnitude_mm=1.0,
      ceiling=True,
    )
    self.assertEqual(
      set(report.reasons),
      {
        AbstainReason.CEILING,
        AbstainReason.CLAMPED,
        AbstainReason.STALE,
        AbstainReason.READOUT_FAILED,
        AbstainReason.HORIZON_EXCEEDED,
        AbstainReason.LOW_CONFIDENCE,
      },
    )
    self.assertIs(report.reason, AbstainReason.CEILING)

  def test_reasons_are_reported_in_precedence_order(self):
    screen = calibrated()
    case = screen_case("two", 0.10, now_s=99.0)
    report = screen.check(case.prediction, case.request, now_s=99.0, readout=None)
    self.assertEqual(
      report.reasons, (AbstainReason.STALE, AbstainReason.LOW_CONFIDENCE)
    )

  def test_the_summary_prints_the_provenance_verbatim_every_time(self):
    screen = calibrated(source="dev seeds 0-99")
    case = screen_case("clean", 0.95)
    text = screen.check(case.prediction, case.request, now_s=1.0, readout=None).summary()
    self.assertIn(screen.calibrated_on, text)
    self.assertIn("dev seeds 0-99", text)

  def test_to_dict_is_json_safe_and_turns_nan_into_null(self):
    screen = Screen.uncalibrated()
    case = screen_case("clean", 0.95)
    row = screen.check(case.prediction, case.request, now_s=1.0, readout=None).to_dict()
    self.assertIsNone(row["theta"])
    self.assertEqual(row["outcome"], "abstain")
    self.assertEqual(row["reason"], "not_calibrated")
    self.assertIsInstance(row["reasons"], list)


class TestClampDowngrade(unittest.TestCase):
  """A clamped action is not the action that was screened."""

  def setUp(self):
    self.screen = calibrated()
    case = screen_case("clean", 0.99)
    self.admitted = self.screen.check(
      case.prediction, case.request, now_s=1.0, readout=None
    )

  def test_an_admitted_verdict_becomes_abstain_clamped(self):
    self.assertTrue(self.admitted.admitted)
    downgraded = self.admitted.clamped_downgrade(12.5)
    self.assertFalse(downgraded.admitted)
    self.assertIs(downgraded.reason, AbstainReason.CLAMPED)
    self.assertAlmostEqual(downgraded.clamp_magnitude_mm, 12.5, places=12)
    self.assertIn("not the one being sent", " ".join(downgraded.notes))

  def test_the_downgrade_is_terminal_and_schedules_no_re_observation(self):
    """Looking again does not un-clamp an oversized command."""
    self.assertFalse(self.admitted.clamped_downgrade(1.0).reobserve)
    self.assertFalse(
      self.admitted.with_reobservation(False).clamped_downgrade(1.0).reobserve
    )

  def test_downgrading_an_already_refused_verdict_keeps_its_other_reasons(self):
    case = screen_case("stale", 0.99, now_s=99.0)
    stale = self.screen.check(case.prediction, case.request, now_s=99.0, readout=None)
    downgraded = stale.clamped_downgrade(3.0)
    self.assertEqual(
      set(downgraded.reasons), {AbstainReason.CLAMPED, AbstainReason.STALE}
    )
    self.assertIs(downgraded.reason, AbstainReason.CLAMPED)

  def test_a_negative_or_non_finite_magnitude_is_refused(self):
    for value in (-1.0, float("nan"), float("inf")):
      with self.subTest(value=value):
        with self.assertRaises(ValueError):
          self.admitted.clamped_downgrade(value)

  def test_an_admitted_verdict_cannot_be_given_a_re_observation(self):
    with self.assertRaises(ValueError):
      self.admitted.with_reobservation(True)


class TestRefusalsAreReachable(unittest.TestCase):
  """Every exception this module declares is produced by some call."""

  def test_screen_declined_carries_its_reason(self):
    screen = calibrated()
    case = screen_case("low", 0.10)
    report = screen.check(case.prediction, case.request, now_s=1.0, readout=None)
    with self.assertRaises(ScreenDeclined) as ctx:
      report.raise_for_refusal()
    self.assertIs(ctx.exception.reason, AbstainReason.LOW_CONFIDENCE)

  def test_a_ceiling_refusal_is_its_own_type(self):
    screen = calibrated()
    case = screen_case("invisible", 0.99)
    report = screen.check(
      case.prediction, case.request, now_s=1.0, readout=None, ceiling=True
    )
    with self.assertRaises(CeilingRefusal) as ctx:
      report.raise_for_refusal()
    self.assertIs(ctx.exception.reason, AbstainReason.CEILING)
    self.assertIsInstance(ctx.exception, ScreenDeclined)

  def test_an_admitted_verdict_raises_nothing(self):
    screen = calibrated()
    case = screen_case("clean", 0.99)
    screen.check(case.prediction, case.request, now_s=1.0, readout=None).raise_for_refusal()

  def test_every_declared_refusal_was_exercised(self):
    self.assertEqual(set(REFUSALS), {ScreenDeclined, CeilingRefusal, NotCalibrated})


class TestNullScreens(unittest.TestCase):
  """Six things that satisfy the Protocol, and all six must fail."""

  def setUp(self):
    self.screen = calibrated()
    self.cases = suite_cases()

  def test_all_six_nulls_fail(self):
    suite = screen_null_check(self.screen, self.cases)
    self.assertEqual(len(suite.results), 6)
    self.assertTrue(suite.passed, suite.summary())
    for result in suite.results:
      with self.subTest(null=result.name):
        self.assertTrue(result.failed)
        self.assertGreater(result.n_disagreed, 0)

  def test_each_null_satisfies_the_protocol_which_is_why_that_proves_nothing(self):
    for null in default_null_screens(self.screen):
      with self.subTest(null=type(null).__name__):
        self.assertIsInstance(null, PredictionScreen)
    self.assertIsInstance(self.screen, PredictionScreen)

  def test_always_admit_admits_the_stale_frame_the_screen_refuses(self):
    stale = self.cases[2]
    self.assertEqual(verdict(self.screen, stale), "ABSTAIN(stale)")
    self.assertEqual(verdict(AlwaysAdmitScreen(), stale), "ACT")

  def test_always_abstain_refuses_the_clean_case_the_screen_admits(self):
    clean = self.cases[0]
    self.assertEqual(verdict(self.screen, clean), "ACT")
    self.assertEqual(verdict(AlwaysAbstainScreen(), clean), "ABSTAIN(low_confidence)")

  def test_constant_confidence_admits_a_prediction_the_model_was_unsure_of(self):
    low = self.cases[1]
    self.assertEqual(verdict(self.screen, low), "ABSTAIN(low_confidence)")
    self.assertEqual(verdict(ConstantConfidenceScreen(self.screen, 1.0), low), "ACT")

  def test_reason_blind_gets_every_decision_right_and_every_reason_wrong(self):
    """The one that a boolean-only comparison would let through."""
    null = ReasonBlindScreen(self.screen)
    admitted_same = 0
    reasons_wrong = 0
    for case in self.cases:
      want, got = verdict(self.screen, case), verdict(null, case)
      if want.startswith("ACT") == got.startswith("ACT"):
        admitted_same += 1
      if want != got:
        reasons_wrong += 1
    self.assertEqual(admitted_same, len(self.cases))
    self.assertGreater(reasons_wrong, 0)

  def test_clamp_only_ignores_the_model_entirely(self):
    low = self.cases[1]
    clamped = self.cases[5]
    self.assertEqual(verdict(ClampOnlyScreen(), low), "ACT")
    self.assertEqual(verdict(ClampOnlyScreen(), clamped), "ABSTAIN(clamped)")

  def test_stale_blind_admits_a_frame_from_four_seconds_ago(self):
    stale = self.cases[2]
    self.assertEqual(verdict(StaleBlindScreen(self.screen), stale), "ACT")

  def test_nothing_checked_is_not_the_same_answer_as_the_check_passed(self):
    empty = screen_null_check(self.screen, [])
    self.assertFalse(empty.passed)
    self.assertEqual(empty.n_cases, 0)
    self.assertIn("NOT RUN", empty.summary())

  def test_a_null_that_matched_everywhere_fails_the_suite(self):
    """The suite's own failure mode: a control that is not a control.

    Handing it the reference screen as a candidate null must not pass, or the
    suite would be reporting that a screen failed to be a screen.
    """
    suite = screen_null_check(self.screen, self.cases, nulls=[self.screen])
    self.assertFalse(suite.passed)
    self.assertIn("BROKEN SCREEN", suite.summary())


class TestScreenedPolicy(unittest.TestCase):
  """Deferral is terminal, bounded re-observation is not a fallback action."""

  def build(self, confidences, **kwargs):
    return ScreenedPolicy(
      MovingPolicy(), calibrated(), StubPredictor(confidences), request_for, **kwargs
    )

  def test_an_admitted_verdict_lets_the_inner_policy_act(self):
    policy = self.build([0.99])
    action = policy.act(Observation(t=1.0))
    self.assertIs(action.mode, ActionMode.DELTA)
    np.testing.assert_allclose(action.pose, (5.0, 0.0, 0.0, 0.0), atol=1e-12)
    self.assertTrue(policy.reports[-1].admitted)

  def test_a_refusal_produces_a_hold_and_never_a_smaller_version_of_the_action(self):
    policy = self.build([0.10])
    action = policy.act(Observation(t=1.0))
    self.assertEqual(action, Action.hold())
    np.testing.assert_allclose(action.pose, (0.0, 0.0, 0.0, 0.0), atol=1e-12)
    self.assertIsNone(action.gripper)
    self.assertFalse(action.terminate)

  def test_the_budget_is_two_re_observations_and_then_the_run_stops(self):
    policy = self.build([0.10] * 6)
    actions = [policy.act(Observation(t=1.0 + i)) for i in range(4)]
    self.assertEqual(MAX_REOBSERVATIONS, 2)
    self.assertFalse(actions[0].terminate)
    self.assertFalse(actions[1].terminate)
    self.assertTrue(actions[2].terminate)
    self.assertTrue(actions[3].terminate)
    self.assertTrue(policy.stopped_safe)
    self.assertEqual([r.reobserve for r in policy.reports], [True, True, False, False])
    # Every one of them is the null action. Not one is a reduced-speed variant.
    for action in actions:
      self.assertEqual(action.pose, (0.0, 0.0, 0.0, 0.0))

  def test_the_budget_is_spent_across_refusals_not_reset_by_an_admission(self):
    """Two refusals, one success, one refusal: the run is over.

    Resetting the counter on success turns a bounded budget into an unbounded
    one, because a policy that intermittently clears the screen never exhausts it.
    """
    policy = self.build([0.10, 0.10, 0.99, 0.10])
    for i in range(3):
      policy.act(Observation(t=1.0 + i))
    last = policy.act(Observation(t=4.0))
    self.assertTrue(last.terminate)
    self.assertTrue(policy.stopped_safe)

  def test_reset_clears_the_budget_and_the_inner_policy(self):
    class Counting(MovingPolicy):
      def __init__(self):
        self.resets = 0

      def reset(self):
        self.resets += 1

    inner = Counting()
    policy = ScreenedPolicy(inner, calibrated(), StubPredictor([0.10] * 6), request_for)
    for i in range(3):
      policy.act(Observation(t=1.0 + i))
    self.assertTrue(policy.stopped_safe)
    policy.reset()
    self.assertFalse(policy.stopped_safe)
    self.assertEqual(policy.reports, [])
    self.assertEqual(inner.resets, 1)

  def test_a_zero_budget_stops_on_the_first_refusal(self):
    policy = self.build([0.10], max_reobservations=0)
    self.assertTrue(policy.act(Observation(t=1.0)).terminate)

  def test_review_returns_the_verdict_act_produced_without_predicting_again(self):
    predictor = StubPredictor([0.99])
    policy = ScreenedPolicy(MovingPolicy(), calibrated(), predictor, request_for)
    observation = Observation(t=1.0)
    action = policy.act(observation)
    self.assertEqual(predictor.calls, 1)
    report = policy.review(observation, action)
    self.assertEqual(predictor.calls, 1)
    self.assertIs(report, policy.reports[-1])

  def test_review_before_act_refuses_rather_than_inventing_an_admission(self):
    policy = self.build([0.99])
    with self.assertRaises(ScreenDeclined) as ctx:
      policy.review(Observation(t=1.0), Action.hold())
    self.assertIn("out of order", str(ctx.exception))

  def test_it_satisfies_the_rollout_seam_and_a_bare_screen_does_not(self):
    self.assertIsInstance(self.build([0.99]), StepScreen)
    self.assertNotIsInstance(calibrated(), StepScreen)

  def test_the_readout_travels_through_to_the_screen(self):
    class Refusing:
      name = "refusing"

      def read(self, observation):
        return FAILED_READOUT

    policy = self.build([0.99], readout=Refusing())
    action = policy.act(Observation(t=1.0))
    self.assertEqual(action, Action.hold())
    self.assertIs(policy.reports[-1].reason, AbstainReason.READOUT_FAILED)

  def test_the_ceiling_predicate_refuses_a_confident_prediction(self):
    policy = self.build([0.99], ceiling_for=lambda request: True)
    policy.act(Observation(t=1.0))
    self.assertIs(policy.reports[-1].reason, AbstainReason.CEILING)

  def test_a_request_builder_returning_the_wrong_type_is_refused(self):
    policy = ScreenedPolicy(
      MovingPolicy(), calibrated(), StubPredictor([0.99]), lambda obs: {"state": 1}
    )
    with self.assertRaises(TypeError) as ctx:
      policy.act(Observation(t=1.0))
    self.assertIn("no frames", str(ctx.exception))

  def test_construction_refuses_objects_that_cannot_play_their_part(self):
    with self.assertRaises(TypeError):
      ScreenedPolicy(object(), calibrated(), StubPredictor([0.9]), request_for)
    with self.assertRaises(TypeError):
      ScreenedPolicy(MovingPolicy(), calibrated(), object(), request_for)
    with self.assertRaises(TypeError):
      ScreenedPolicy(MovingPolicy(), calibrated(), StubPredictor([0.9]), "not callable")
    with self.assertRaises(ValueError):
      ScreenedPolicy(
        MovingPolicy(),
        calibrated(),
        StubPredictor([0.9]),
        request_for,
        max_reobservations=-1,
      )


if __name__ == "__main__":
  unittest.main()
