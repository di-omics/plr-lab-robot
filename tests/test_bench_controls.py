"""Tests for plr_lr.learn.bench.controls.

Nine controls must fail and one must pass. This module asserts each of those
outcomes individually rather than asserting that the suite passed, because a suite
that passed is exactly what a suite which ran nothing also reports unless somebody
checks.

Two claims here are structural rather than numeric.

``ControlSuiteReport.passed`` is four conditions, and each of the four is tested by
constructing a report that satisfies the other three. An empty result list, a zero
episode count, a control that did not run and a throttled null all have to produce
``False`` on their own, because each of them is a way of arriving at "the check
passed" from "nothing was checked".

``N6 OracleLeak`` failing raises rather than scoring. The oracle is handed the true
terminal state; if it does not come back perfect then the tolerance lookup, the
truth lookup or the comparison is wrong, and every number the harness would print
about a real model rests on the same arithmetic. A low score there would be read as
the model's fault.
"""

from __future__ import annotations

import hashlib
import unittest
from typing import Any, Dict
from unittest import mock

import numpy as np

from plr_lr.learn.bench import controls
from plr_lr.learn.bench.controls import (
  BANDS,
  MUST_FAIL,
  MUST_PASS,
  N1_TASKS,
  N4_ORDERINGS,
  THROTTLED_CLAMPED_FRACTION,
  AlwaysAbstainPredictor,
  AlwaysConfidentPredictor,
  ControlResult,
  ControlSuiteReport,
  HarnessBroken,
  OracleLeakPredictor,
  PixelPerfectPredictor,
  SeedLeakPredictor,
  ShuffledActionPredictor,
  build_request,
  constant_confidence_e_aurc,
  evaluate_rows,
  observable_signature,
  resolve_predictor,
  run_control_suite,
)
from plr_lr.learn.bench.packio import (
  ScenePack,
  SceneEpisode,
  SceneFrame,
  builtin_demo_pack,
)
from plr_lr.learn.bench.runner import floor_predictors
from plr_lr.learn.bench.tasks import TASKS, TASKS_BY_ID
from plr_lr.learn.predict import (
  BlindPredictor,
  NoisePredictor,
  RolloutPrediction,
  interventional_check,
)
from plr_lr.learn.selective import RiskCoverageCurve

#: Enough episodes for the thirty-observation reporting floor to be cleared on both
#: the knowable and the whole-task denominators, and small enough to stay fast.
N = 60

PACK = builtin_demo_pack(n_episodes=N)
FLOOR = floor_predictors(TASKS)


def _rows(predictor: Any, task_id: str, horizon: int = None):
  task = TASKS_BY_ID[task_id]
  return evaluate_rows(
    predictor, task, PACK, int(task.horizons[0] if horizon is None else horizon)
  )


class SharedEvaluationTest(unittest.TestCase):
  """Controls and submissions go through one function, on purpose."""

  def test_a_control_and_a_submission_are_scored_by_the_same_code(self) -> None:
    task = TASKS_BY_ID["S3"]
    control = evaluate_rows(OracleLeakPredictor(task, 1, PACK), task, PACK, 1)
    submission = evaluate_rows(FLOOR["S3"], task, PACK, 1)
    self.assertEqual(control.episode_ids, submission.episode_ids)
    self.assertEqual(control.unknowable, submission.unknowable)

  def test_the_request_carries_no_knowability_and_no_invisible_latent(self) -> None:
    task = TASKS_BY_ID["S3"]
    episode = next(e for e in PACK.for_task("S3") if e.unknowable)
    request = build_request(task, episode, 1)
    for field in ("knowability", "hidden", "unknowable", "hidden_keys"):
      self.assertFalse(hasattr(request, field), field)

  def test_abstentions_stay_in_the_denominator(self) -> None:
    task = TASKS_BY_ID["S3"]
    rows = evaluate_rows(AlwaysAbstainPredictor(task, 1), task, PACK, 1)
    self.assertEqual(rows.n, N)
    self.assertEqual(rows.n_ranked, 0)
    self.assertEqual(rows.tol_acc, 0.0)

  def test_a_mapping_of_predictors_is_resolved_per_task(self) -> None:
    self.assertIs(resolve_predictor(FLOOR, TASKS_BY_ID["S1"]), FLOOR["S1"])
    self.assertIs(resolve_predictor(FLOOR["S1"], TASKS_BY_ID["S1"]), FLOOR["S1"])

  def test_a_missing_task_in_a_mapping_is_refused_not_skipped(self) -> None:
    with self.assertRaises(ValueError):
      resolve_predictor({"S1": FLOOR["S1"]}, TASKS_BY_ID["S3"])


class N1ConstantPredictorTest(unittest.TestCase):
  def test_it_fails_tolerance_on_every_task_it_is_required_to(self) -> None:
    for task_id in N1_TASKS:
      task = TASKS_BY_ID[task_id]
      rows = _rows(BlindPredictor(state_dim=task.state_dim, horizon=1), task_id)
      self.assertLessEqual(rows.tol_acc, BANDS["N1_MAX_TOL_ACC"], task_id)

  def test_no_task_is_exempt(self) -> None:
    """S2 and S3 were exempt on a justification the shipped fixture contradicts.

    The stated reason was that a do-nothing predictor on a single-scalar latent
    lands inside tolerance too often for the requirement to be fair. Measured at
    horizon 1 on this pack it lands at roughly a quarter to a third, well under the
    0.50 ceiling, so the exemption was protecting two tasks from a bound they
    already met.
    """
    self.assertEqual(N1_TASKS, tuple(t.task_id for t in TASKS))

  def test_the_two_formerly_exempt_tasks_clear_the_ceiling_with_room(self) -> None:
    for task_id in ("S2", "S3"):
      task = TASKS_BY_ID[task_id]
      rows = _rows(BlindPredictor(state_dim=task.state_dim, horizon=1), task_id)
      self.assertLess(rows.tol_acc, BANDS["N1_MAX_TOL_ACC"] - 0.05, task_id)

  def test_the_suite_runs_it_on_every_task(self) -> None:
    seen = {r.task_id for r in run_control_suite(
      FLOOR, PACK, tasks=TASKS, seed=0, n_permutations=10, n_resamples=50
    ).results if r.control_id == "N1"}
    self.assertEqual(seen, {t.task_id for t in TASKS})


class N2ShuffledActionTest(unittest.TestCase):
  def test_permuting_actions_across_the_batch_destroys_the_score(self) -> None:
    for task in TASKS:
      shuffled = _rows(ShuffledActionPredictor(task, task.horizons[0], PACK, seed=0),
                       task.task_id)
      truth = _rows(OracleLeakPredictor(task, task.horizons[0], PACK), task.task_id)
      self.assertEqual(truth.tol_acc, 1.0, task.task_id)
      self.assertLess(shuffled.tol_acc, truth.tol_acc - BANDS["N2_MIN_GAP"], task.task_id)

  def test_no_episode_is_paired_with_its_own_actions(self) -> None:
    task = TASKS_BY_ID["S3"]
    predictor = ShuffledActionPredictor(task, 1, PACK, seed=0)
    for episode_id, partner in predictor._partner.items():
      self.assertNotEqual(episode_id, partner.episode_id)

  def test_it_recomputes_the_truth_when_the_pack_publishes_coefficients(self) -> None:
    task = TASKS_BY_ID["S3"]
    predictor = ShuffledActionPredictor(task, 1, PACK, seed=0)
    self.assertTrue(predictor.exact)

  def test_it_falls_back_to_the_partner_terminal_on_a_scoring_pack(self) -> None:
    scoring = builtin_demo_pack(n_episodes=N, split="scoring")
    task = TASKS_BY_ID["S3"]
    predictor = ShuffledActionPredictor(task, 1, scoring, seed=0)
    self.assertFalse(predictor.exact)
    rows = evaluate_rows(predictor, task, scoring, 1)
    self.assertLess(rows.tol_acc, 1.0)


class N4AlwaysConfidentTest(unittest.TestCase):
  def test_a_constant_confidence_lands_on_the_no_information_band(self) -> None:
    """And not on zero. Reading the band as zero makes a base rate look perfect."""
    for task in TASKS:
      rows = _rows(AlwaysConfidentPredictor(task, task.horizons[0], PACK), task.task_id)
      hits, confidences = rows.ranked()
      curve = RiskCoverageCurve.from_outcomes(hits, confidences)
      band = curve.aurc_random - curve.aurc_oracle
      observed = constant_confidence_e_aurc(rows)
      self.assertLessEqual(abs(observed - band), BANDS["N4_E_AURC_TOL"], task.task_id)

  def test_the_band_is_only_zero_when_there_are_no_errors_to_rank(self) -> None:
    """Otherwise a constant confidence landing near zero would read as near perfect."""
    task = TASKS_BY_ID["S4"]
    rows = _rows(AlwaysConfidentPredictor(task, 1, PACK), "S4")
    hits, confidences = rows.ranked()
    curve = RiskCoverageCurve.from_outcomes(hits, confidences)
    self.assertGreater(curve.n_errors, 0)
    self.assertGreater(curve.aurc_random - curve.aurc_oracle, 0.0)

  def test_one_ordering_is_a_draw_and_the_median_is_the_estimate(self) -> None:
    """Every confidence is equal, so the sort is decided by the tie-break alone."""
    task = TASKS_BY_ID["S4"]
    rows = _rows(AlwaysConfidentPredictor(task, 1, PACK), "S4")
    hits, confidences = rows.ranked()
    draws = {
      round(RiskCoverageCurve.from_outcomes(hits, confidences, seed=s).e_aurc, 9)
      for s in range(N4_ORDERINGS)
    }
    self.assertGreater(len(draws), 1)

  def test_every_confidence_really_is_one(self) -> None:
    task = TASKS_BY_ID["S2"]
    rows = _rows(AlwaysConfidentPredictor(task, 1, PACK), "S2")
    self.assertEqual(set(rows.confidences), {1.0})


class N5PixelPerfectTest(unittest.TestCase):
  def test_it_tops_the_pixel_column(self) -> None:
    task = TASKS_BY_ID["S4"]
    psnr, ssim = PixelPerfectPredictor(task, 1, PACK).declared_pixel_metrics()
    self.assertGreaterEqual(psnr, BANDS["N5_MIN_PSNR"])
    self.assertEqual(ssim, 1.0)

  def test_it_still_cannot_be_ranked(self) -> None:
    """Its confidence is constant, so it cannot beat its own confidence shuffle."""
    from plr_lr.learn.selective import confidence_shuffle_p

    task = TASKS_BY_ID["S4"]
    rows = _rows(PixelPerfectPredictor(task, 1, PACK), "S4")
    hits, confidences = rows.ranked()
    permutation = confidence_shuffle_p(hits, confidences, seed=0, n_permutations=100)
    self.assertTrue(permutation.no_information)


class N6OracleLeakTest(unittest.TestCase):
  def test_the_oracle_scores_exactly_one_on_every_task_and_horizon(self) -> None:
    for task in TASKS:
      for horizon in task.horizons:
        rows = evaluate_rows(
          OracleLeakPredictor(task, horizon, PACK), task, PACK, horizon
        )
        self.assertEqual(rows.tol_acc, BANDS["N6_REQUIRED_TOL_ACC"],
                         f"{task.task_id}@{horizon}")

  def test_a_failing_oracle_raises_rather_than_scoring(self) -> None:
    """A hard AssertionError, because the harness is what is broken, not the model."""

    class BrokenOracle(OracleLeakPredictor):
      def predict(self, request):
        episode = self._episode(request)
        wrong = tuple(v + 1e6 for v in episode.terminal(int(request.horizon)))
        return self._answer(request, wrong, 1.0, "broken-oracle")

    task = TASKS_BY_ID["S2"]
    with mock.patch.object(controls, "OracleLeakPredictor", BrokenOracle):
      with self.assertRaises(HarnessBroken):
        controls._n6(task, PACK, 1)

  def test_harness_broken_is_an_assertion_error(self) -> None:
    self.assertTrue(issubclass(HarnessBroken, AssertionError))


class N7AlwaysAbstainTest(unittest.TestCase):
  def test_it_ranks_nothing_and_its_credit_is_the_base_rate(self) -> None:
    task = TASKS_BY_ID["S2"]
    result = controls._n7(task, PACK, 1)
    self.assertTrue(result.ran)
    self.assertTrue(result.met)
    self.assertIn("FAR 1.0000", result.observed)

  def test_abstaining_on_a_knowable_episode_earns_nothing(self) -> None:
    task = TASKS_BY_ID["S2"]
    rows = evaluate_rows(AlwaysAbstainPredictor(task, 1), task, PACK, 1)
    credited = sum(
      1 for unknowable, abstained in zip(rows.unknowable, rows.abstained)
      if unknowable and abstained
    )
    self.assertEqual(credited, sum(rows.unknowable))
    self.assertLess(credited, rows.n)

  def test_an_abstention_carries_no_prediction(self) -> None:
    task = TASKS_BY_ID["S2"]
    prediction = AlwaysAbstainPredictor(task, 1).predict(
      build_request(task, PACK.for_task("S2")[0], 1)
    )
    self.assertIsInstance(prediction, RolloutPrediction)
    self.assertTrue(prediction.abstain)
    self.assertTrue(bool(np.isnan(prediction.state).all()))
    self.assertTrue(bool(np.isnan(prediction.confidence)))


def _republished(pack: ScenePack, suffix: str = "-redraw") -> ScenePack:
  """The same pack with every episode published a second time under a new name.

  The observables are copied exactly, which is what a scoring seed range overlapping
  the published one produces: the generator draws are continuous, so a re-draw of
  the same seed reproduces ``state0`` and the action block bit for bit. Only the
  episode identifier and the frame hashes differ, and they have to differ, because
  :class:`ScenePack` refuses duplicate identifiers and duplicate frame sequences.
  Those two refusals are exactly why this leak gets past the format layer and has to
  be caught by a control.
  """
  out = []
  for episode in pack.episodes:
    out.append(episode)
    twin_id = episode.episode_id + suffix
    out.append(
      SceneEpisode(
        episode_id=twin_id,
        task_id=episode.task_id,
        state0=episode.state0,
        actions=episode.actions,
        terminal_by_horizon=dict(episode.terminal_by_horizon),
        knowability=episode.knowability,
        frames=tuple(
          SceneFrame(
            path=f"frames/{episode.task_id}/{twin_id}-{index:02d}.png",
            sha256=hashlib.sha256((suffix + frame.sha256).encode("ascii")).hexdigest(),
            t_capture_s=frame.t_capture_s,
            clock_source=frame.clock_source,
          )
          for index, frame in enumerate(episode.frames)
        ),
        coefficients=(
          None if episode.coefficients is None else dict(episode.coefficients)
        ),
        hidden=None if episode.hidden is None else dict(episode.hidden),
        hidden_keys=episode.hidden_keys,
      )
    )
  return ScenePack(
    pack_id=pack.pack_id + "-republished",
    seed_epoch=pack.seed_epoch,
    split=pack.split,
    source="a deliberately leaky fixture, built here for the N8 negative control",
    episodes=tuple(out),
  )


def _contradictory(pack: ScenePack) -> ScenePack:
  """The same pack with every episode published twice under two different answers.

  Same observables, terminal states moved a hundred tolerances away. A probe keyed
  on observables cannot recall both, so this is what an ambiguous pack does to N8's
  positive control, and the control has to refuse rather than report the withheld
  half against a probe that demonstrably does not work.
  """
  out = []
  for episode in _republished(pack, suffix="-contradiction").episodes:
    if not episode.episode_id.endswith("-contradiction"):
      out.append(episode)
      continue
    tolerance = TASKS_BY_ID[episode.task_id].tolerance
    out.append(
      SceneEpisode(
        episode_id=episode.episode_id,
        task_id=episode.task_id,
        state0=episode.state0,
        actions=episode.actions,
        terminal_by_horizon={
          horizon: tuple(v + 100.0 * t for v, t in zip(row, tolerance))
          for horizon, row in episode.terminal_by_horizon.items()
        },
        knowability=episode.knowability,
        frames=episode.frames,
        coefficients=(
          None if episode.coefficients is None else dict(episode.coefficients)
        ),
        hidden=None if episode.hidden is None else dict(episode.hidden),
        hidden_keys=episode.hidden_keys,
      )
    )
  return ScenePack(
    pack_id=pack.pack_id + "-contradictory",
    seed_epoch=pack.seed_epoch,
    split=pack.split,
    source="an ambiguous fixture, built here for the N8 positive control",
    episodes=tuple(out),
  )


#: Small, because N8 is quadratic in nothing but the tests below run it per task.
LEAKY = _republished(builtin_demo_pack(n_episodes=40))


class N8SeedLeakTest(unittest.TestCase):
  """A must-fail control that has never been observed to fail is not a control.

  The previous version of N8 could not fail on any input. Off the published half it
  answered ``state0``, and ``state0`` is what ``BlindPredictor`` answers too, so the
  gap it closed against that floor was exactly +0.0000 on every task and every pack
  by construction rather than by measurement. ``test_it_fails_when_the_partition_
  does_not_bite`` is the test that could not have been written against it.
  """

  def test_a_model_fit_on_the_published_half_is_perfect_on_it(self) -> None:
    task = TASKS_BY_ID["S2"]
    episodes = PACK.for_task("S2")
    dev = episodes[: len(episodes) // 2]
    leak = SeedLeakPredictor(task, 1, PACK, [e.episode_id for e in dev])
    rows = evaluate_rows(leak, task, PACK, 1, episodes=dev)
    self.assertEqual(rows.tol_acc, BANDS["N8_MIN_DEV_RECALL"])

  def test_the_recall_is_a_lift_over_the_floor_and_not_a_free_one(self) -> None:
    """The positive control. The probe has to beat the do-nothing floor where it
    has a memory to answer from, or its silence on the withheld half means nothing.
    """
    for task in TASKS:
      episodes = PACK.for_task(task.task_id)
      dev = episodes[: len(episodes) // 2]
      leak = SeedLeakPredictor(task, 1, PACK, [e.episode_id for e in dev])
      recalled = evaluate_rows(leak, task, PACK, 1, episodes=dev).tol_acc
      floor = evaluate_rows(
        BlindPredictor(state_dim=task.state_dim, horizon=1), task, PACK, 1, episodes=dev
      ).tol_acc
      self.assertEqual(recalled, 1.0, task.task_id)
      self.assertGreater(recalled - floor, 0.5, task.task_id)

  def test_it_answers_the_do_nothing_state_when_nothing_is_republished(self) -> None:
    task = TASKS_BY_ID["S2"]
    episodes = PACK.for_task("S2")
    dev, withheld = episodes[: len(episodes) // 2], episodes[len(episodes) // 2:]
    leak = SeedLeakPredictor(task, 1, PACK, [e.episode_id for e in dev])
    request = build_request(task, withheld[0], 1)
    self.assertIsNone(leak.republication(request))
    prediction = leak.predict(request)
    self.assertEqual(list(prediction.state[-1]), list(withheld[0].state0))

  def test_a_republished_episode_is_answered_out_of_memory(self) -> None:
    task = TASKS_BY_ID["S2"]
    published = [e for e in LEAKY.for_task("S2") if not e.episode_id.endswith("-redraw")]
    twin = next(e for e in LEAKY.for_task("S2") if e.episode_id.endswith("-redraw"))
    leak = SeedLeakPredictor(task, 1, LEAKY, [e.episode_id for e in published])
    request = build_request(task, twin, 1)
    match = leak.republication(request)
    self.assertIsNotNone(match)
    self.assertEqual(match.episode_id + "-redraw", twin.episode_id)
    prediction = leak.predict(request)
    self.assertEqual(list(prediction.state[-1]), list(twin.terminal(1)))

  def test_it_does_not_close_the_gap_on_the_withheld_half(self) -> None:
    for task in TASKS:
      result = controls._n8(task, PACK, int(task.horizons[0]), seed=0)
      self.assertTrue(result.ran, task.task_id)
      self.assertTrue(result.met, f"{task.task_id}: {result.observed}")
      self.assertIn(f"0 of {N - N // 2} withheld episodes republish", result.observed)
      self.assertIn("probe recall 1.0000", result.observed)

  def test_it_fails_when_the_partition_does_not_bite(self) -> None:
    """The negative control. Every withheld episode is a republished one here."""
    failed = []
    for task in TASKS:
      result = controls._n8(task, LEAKY, int(task.horizons[0]), seed=0)
      self.assertTrue(result.ran, task.task_id)
      if not result.met:
        failed.append(task.task_id)
      self.assertIn("republish a published one", result.observed)
    self.assertEqual(failed, [t.task_id for t in TASKS])

  def test_the_failure_names_the_republished_episodes(self) -> None:
    result = controls._n8(TASKS_BY_ID["S2"], LEAKY, 1, seed=0)
    self.assertFalse(result.met)
    joined = " ".join(result.notes)
    self.assertIn("are the same episode as a published one", joined)
    self.assertIn("S2-e", joined)

  def test_a_leaky_pack_fails_the_whole_suite(self) -> None:
    report = run_control_suite(
      FLOOR, LEAKY, tasks=TASKS, seed=0, n_permutations=50, n_resamples=200
    )
    self.assertFalse(report.passed)
    self.assertEqual(
      {r.control_id for r in report.failures()}, {"N8"}, report.summary()
    )

  def test_an_ambiguous_pack_breaks_the_recall_the_control_rests_on(self) -> None:
    """One signature, two published terminals. The probe cannot recall both."""
    ambiguous = _contradictory(builtin_demo_pack(n_episodes=8))
    task = TASKS_BY_ID["S2"]
    published = ambiguous.for_task("S2")
    leak = SeedLeakPredictor(task, 1, ambiguous, [e.episode_id for e in published])
    rows = evaluate_rows(leak, task, ambiguous, 1, episodes=published)
    self.assertLess(rows.tol_acc, BANDS["N8_MIN_DEV_RECALL"])

  def test_a_probe_that_cannot_recall_its_memory_establishes_nothing(self) -> None:
    """Every recall forced to miss, which is what an ambiguous pack does to it."""
    task = TASKS_BY_ID["S2"]
    with mock.patch.object(
      controls.SeedLeakPredictor,
      "republication",
      lambda self, request: None,
    ):
      result = controls._n8(task, PACK, 1, seed=0)
    self.assertTrue(result.ran)
    self.assertFalse(result.met)
    self.assertIn("recalled only", " ".join(result.notes))

  def test_the_control_carries_evidence_rather_than_being_marked_exempt(self) -> None:
    result = controls._n8(TASKS_BY_ID["S2"], PACK, 1, seed=0)
    self.assertTrue(result.evidential)

  def test_the_signature_is_the_observables_and_only_those(self) -> None:
    task = TASKS_BY_ID["S3"]
    episode = PACK.for_task("S3")[0]
    request = build_request(task, episode, 1)
    self.assertEqual(
      observable_signature(request.state, request.actions),
      observable_signature(episode.state0, episode.actions[:1]),
    )
    other = PACK.for_task("S3")[1]
    self.assertNotEqual(
      observable_signature(episode.state0, episode.actions[:1]),
      observable_signature(other.state0, other.actions[:1]),
    )


class N9ClampOnlyTest(unittest.TestCase):
  def test_it_cannot_reproduce_a_correlation_a_pack_has_no_column_for(self) -> None:
    task = TASKS_BY_ID["S2"]
    submitted = _rows(FLOOR["S2"], "S2")
    result = controls._n9(task, submitted)
    self.assertTrue(result.met)
    self.assertFalse(result.evidential)
    self.assertIn("no clamp counts", result.observed)

  def test_a_non_evidential_pass_is_reported_as_one(self) -> None:
    task = TASKS_BY_ID["S2"]
    result = controls._n9(task, _rows(FLOOR["S2"], "S2"))
    report = ControlSuiteReport(results=(result,), n=N)
    self.assertEqual(report.n_non_evidential, 1)
    self.assertIn("trivially", report.summary())


class N10InterventionalGateTest(unittest.TestCase):
  def test_both_blind_nulls_are_refused_by_the_gate(self) -> None:
    for task in TASKS:
      episodes = PACK.for_task(task.task_id)
      requests = [build_request(task, e, task.horizons[0]) for e in episodes]
      for predictor in (
        BlindPredictor(state_dim=task.state_dim, horizon=task.horizons[0]),
        NoisePredictor(state_dim=task.state_dim, horizon=task.horizons[0], sigma=1.0),
      ):
        result = interventional_check(predictor, requests, seed=0)
        self.assertFalse(result.passed, f"{task.task_id} {predictor.name}")

  def test_the_declared_floor_clears_the_gate_on_every_task(self) -> None:
    """The floor genuinely reads its action argument, so the gate must admit it."""
    for task in TASKS:
      episodes = PACK.for_task(task.task_id)
      requests = [build_request(task, e, task.horizons[0]) for e in episodes]
      result = interventional_check(FLOOR[task.task_id], requests, seed=0)
      self.assertTrue(result.passed, f"{task.task_id}: {result.summary()}")


class SuitePassedTest(unittest.TestCase):
  """Four conditions, each tested by satisfying the other three."""

  def _ok(self, **overrides: Any) -> ControlResult:
    base: Dict[str, Any] = dict(
      control_id="N1",
      name="control",
      requirement=MUST_FAIL,
      task_id="S1",
      ran=True,
      met=True,
      observed="fine",
    )
    base.update(overrides)
    return ControlResult(**base)

  def test_an_empty_result_list_does_not_pass(self) -> None:
    self.assertFalse(ControlSuiteReport(results=(), n=10).passed)

  def test_a_zero_episode_count_does_not_pass(self) -> None:
    self.assertFalse(ControlSuiteReport(results=(self._ok(),), n=0).passed)

  def test_a_control_that_did_not_run_does_not_pass(self) -> None:
    report = ControlSuiteReport(results=(self._ok(), self._ok(ran=False, met=False)), n=10)
    self.assertFalse(report.passed)

  def test_a_control_that_did_not_meet_its_requirement_does_not_pass(self) -> None:
    report = ControlSuiteReport(results=(self._ok(), self._ok(met=False)), n=10)
    self.assertFalse(report.passed)

  def test_a_throttled_null_invalidates_the_suite(self) -> None:
    throttled = self._ok(clamped_fraction=THROTTLED_CLAMPED_FRACTION + 0.01)
    self.assertTrue(throttled.throttled)
    self.assertIn("THROTTLED NULL", throttled.line())
    self.assertFalse(ControlSuiteReport(results=(throttled,), n=10).passed)

  def test_a_clamp_fraction_at_the_bound_is_not_throttled(self) -> None:
    self.assertFalse(self._ok(clamped_fraction=THROTTLED_CLAMPED_FRACTION).throttled)

  def test_all_four_conditions_together_pass(self) -> None:
    self.assertTrue(ControlSuiteReport(results=(self._ok(),), n=10).passed)

  def test_a_suite_that_was_never_run_says_so_rather_than_printing_a_verdict(self) -> None:
    self.assertIn("NOT RUN", ControlSuiteReport().summary())


class ClampRateTest(unittest.TestCase):
  """An unmeasured clamp rate is not a clamp rate of zero.

  Nothing in a pack-only run goes through a safety envelope, so no producer sets
  this. The default was ``0.0``, which printed and serialised as a measured rate of
  nothing, and the one thing the throttle exists to catch is a null whose actions
  were quietly rewritten before they reached the backend.
  """

  def _result(self, **overrides: Any) -> ControlResult:
    base: Dict[str, Any] = dict(
      control_id="N1",
      name="control",
      requirement=MUST_FAIL,
      task_id="S1",
      ran=True,
      met=True,
      observed="fine",
    )
    base.update(overrides)
    return ControlResult(**base)

  def test_the_default_is_unmeasured_and_not_zero(self) -> None:
    result = self._result()
    self.assertIsNone(result.clamped_fraction)
    self.assertFalse(result.clamp_rate_measured)

  def test_a_measured_zero_is_a_different_state(self) -> None:
    result = self._result(clamped_fraction=0.0)
    self.assertTrue(result.clamp_rate_measured)
    self.assertEqual(result.clamped_fraction, 0.0)
    self.assertNotEqual(result.to_dict(), self._result().to_dict())

  def test_the_two_states_serialise_differently(self) -> None:
    unmeasured = self._result().to_dict()
    measured = self._result(clamped_fraction=0.0).to_dict()
    self.assertIsNone(unmeasured["clamped_fraction"])
    self.assertEqual(measured["clamped_fraction"], 0.0)
    self.assertFalse(unmeasured["clamp_rate_measured"])
    self.assertTrue(measured["clamp_rate_measured"])

  def test_only_a_measured_rate_prints_on_the_line(self) -> None:
    self.assertNotIn("clamped", self._result().line())
    self.assertIn("[clamped 0.000]", self._result(clamped_fraction=0.0).line())

  def test_unmeasured_is_not_throttled_but_the_suite_says_so_once(self) -> None:
    report = ControlSuiteReport(results=(self._result(), self._result()), n=10)
    self.assertTrue(report.passed)
    self.assertEqual(report.n_clamp_unmeasured, 2)
    self.assertIn("no measured clamp rate", report.summary())
    self.assertIn("not a measurement of zero", report.summary())

  def test_a_fully_measured_suite_says_nothing_about_it(self) -> None:
    report = ControlSuiteReport(results=(self._result(clamped_fraction=0.0),), n=10)
    self.assertEqual(report.n_clamp_unmeasured, 0)
    self.assertNotIn("no measured clamp rate", report.summary())

  def test_the_pack_only_suite_measures_no_clamp_rate_at_all(self) -> None:
    report = run_control_suite(
      FLOOR, PACK, tasks=TASKS, seed=0, n_permutations=10, n_resamples=50
    )
    self.assertEqual(report.n_clamp_unmeasured, len(report.results))
    self.assertEqual(report.to_dict()["n_clamp_unmeasured"], len(report.results))


class FullSuiteTest(unittest.TestCase):
  """The whole suite against the declared floor, which is what the runner gates on."""

  @classmethod
  def setUpClass(cls) -> None:
    cls.report = run_control_suite(
      FLOOR, PACK, tasks=TASKS, seed=0, n_permutations=50, n_resamples=200
    )

  def test_the_suite_passes(self) -> None:
    self.assertTrue(self.report.passed, self.report.summary())

  def test_every_control_ran(self) -> None:
    self.assertTrue(all(r.ran for r in self.report.results))

  def test_all_ten_controls_are_represented(self) -> None:
    seen = {r.control_id for r in self.report.results}
    self.assertEqual(seen, {f"N{i}" for i in range(1, 11)})

  def test_exactly_one_control_is_a_must_pass(self) -> None:
    must_pass = {r.control_id for r in self.report.results if r.requirement == MUST_PASS}
    self.assertEqual(must_pass, {"N6"})

  def test_the_action_conditioned_verdict_is_recorded_per_task(self) -> None:
    self.assertEqual(set(self.report.action_conditioned), {t.task_id for t in TASKS})

  def test_the_dict_form_is_complete(self) -> None:
    raw = self.report.to_dict()
    self.assertIn("passed", raw)
    self.assertEqual(len(raw["results"]), len(self.report.results))


class BandsTest(unittest.TestCase):
  def test_every_band_is_an_absolute_constant(self) -> None:
    for key, value in BANDS.items():
      self.assertIsInstance(value, float, key)

  def test_the_oracle_band_demands_perfection(self) -> None:
    self.assertEqual(BANDS["N6_REQUIRED_TOL_ACC"], 1.0)

  def test_the_throttle_bound_is_a_tenth(self) -> None:
    self.assertEqual(THROTTLED_CLAMPED_FRACTION, 0.10)


if __name__ == "__main__":
  unittest.main()
