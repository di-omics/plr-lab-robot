"""Running the benchmark, in the one order the refusals require.

The order is the design. The action-conditioning gate runs first, before any
episode is scored, and its failure produces a refusal rather than a low score. The
null-control suite runs second, and no scorecard is emitted unless it passes. Only
then is a single row computed. Reversing any two of those three lets a number exist
before the thing that would have withheld it, and once a number exists somebody
quotes it.

The declared floor predictor lives here and is a first-class part of the design
rather than a test fixture. It is the classical model this repository already
carries -- :class:`plr_lr.learn.dynamics.ResidualDynamics` over a kinematic base for
the pose task, and the pack's own declared transition at its published nominal
coefficients for the rest -- wired to the rollout-prediction contract. A world model
that does not beat it on a task has produced no evidence of value on that task, and
saying so in the same table is cheaper than arguing about it afterwards.

Its confidence is a self-consistency estimate: predict under the commanded actions,
predict again under a small seeded jitter of them, and read the spread. That is
ensemble-free on purpose, because the ensemble path is unaffordable on one consumer
card and a build that only supports the unaffordable estimator ships zero usable
ones.

Its abstention channel is the same spread against an absolute bound. It is
deliberately **not** wired to anything that could see the knowability label: this
predictor has no way to tell an independence-constructed episode from any other, its
refusal recall on those episodes will be close to zero, and that is the honest first
number. A floor row that scored well on abstention would mean the fixture had a tell
in it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..dynamics import KinematicDynamics, ResidualDynamics
from ..predict import (
  INTERVENTION_K,
  NO_TRAINING_DATA,
  InterventionResult,
  ProvenanceUndeclared,
  RolloutPrediction,
  RolloutRequest,
  UndeclaredComponent,
  interventional_check,
)
from ..screen import Screen
from ..selective import (
  PERMUTATION_RESAMPLES,
  FrozenThreshold,
  ThresholdNotFrozen,
  score,
)
from . import metrics
from .controls import (
  EpisodeRows,
  build_request,
  evaluate_rows,
  resolve_predictor,
  run_control_suite,
)
from .packio import NOMINAL, ScenePack, advance
from .report import (
  REFUSAL_RULES,
  BenchReport,
  BenchRow,
  Refusal,
  basis_label,
  underpowered_refusal,
)
from .tasks import TASKS, BenchTask

#: How many jittered replays the floor predictor averages its confidence over. An
#: absolute constant: a self-consistency estimate whose sample count was tuned
#: against the episodes being scored would be a fitted component, and an undeclared
#: one at that.
FLOOR_SELF_CONSISTENCY_K = 4

#: The jitter applied to the three continuous action columns when measuring
#: self-consistency, in the columns' own units. Absolute, and deliberately not scaled
#: to the commanded magnitude: a jitter proportional to the command would grow with
#: any perturbation the harness applied, so the predictor would refuse hardest
#: exactly where the action-conditioning gate is asking it to answer.
FLOOR_JITTER = 0.10

#: Spread above which the floor predictor refuses, in tolerance units. Absolute, and
#: set above the width one boolean field contributes when it flips, because a boolean
#: flip is a different kind of uncertainty and folding it into a continuous spread
#: makes the bound mean nothing. A flipping boolean raises the flag instead.
FLOOR_ABSTAIN_SPREAD = 1.5

#: The declared unit of a state field that is a yes/no, not a measurement.
BOOL_UNIT = "bool"

#: Every published rule identifier, mapped to the token the published table prints
#: beside it. Built from :data:`plr_lr.learn.bench.report.REFUSAL_RULES` rather than
#: restated, so the two cannot drift.
RULE_LABELS: Dict[str, str] = {rule: label for rule, label, _why in REFUSAL_RULES}


def _refusal(rule: str, detail: str, blocking: bool = True) -> Refusal:
  """A refusal carrying the label its rule publishes, which is not a free field.

  :class:`plr_lr.learn.bench.report.Refusal` validates the rule identifier and not
  the label, and this runner used the gap: it emitted ``R7 LEAK`` for a training
  overlap while the published sixteen-rule table prints ``PROVENANCE: UNDECLARED``
  beside R7. A reader cross-referencing that line against the table finds a rule
  about an adapter with no declared training set, which is a different refusal from
  the one that fired. Deriving the label here rather than passing one means a
  mismatch cannot be written in the first place, and an identifier outside the
  published table raises rather than printing a token nothing can be checked
  against.
  """
  try:
    label = RULE_LABELS[rule]
  except KeyError:
    raise ValueError(
      f"refusal rule {rule!r} is not one of {sorted(RULE_LABELS)}. A rule identifier "
      "this runner invented cannot be cross-referenced against the published table."
    ) from None
  return Refusal(rule, label, detail, blocking)


@dataclass(frozen=True)
class SeedPlan:
  """Every seed the run depends on, recorded so a row can be regenerated.

  The epoch prints on every row and cross-epoch tables are refused. Seed secrecy is
  a speed bump and this build says so: the generator is public, so anybody can
  enumerate seeds. The real defences are the gate, the seed-leak control and the
  declared training fingerprints, and calling the epoch a guarantee would be the
  dishonest part.
  """

  epoch: str
  tie_break_seed: int = 0
  gate_seed: int = 0
  control_seed: int = 0
  bootstrap_seed: int = 0
  seed_range: Optional[Tuple[int, int]] = None

  def to_dict(self) -> Dict[str, Any]:
    return {
      "epoch": self.epoch,
      "tie_break_seed": self.tie_break_seed,
      "gate_seed": self.gate_seed,
      "control_seed": self.control_seed,
      "bootstrap_seed": self.bootstrap_seed,
      "seed_range": list(self.seed_range) if self.seed_range is not None else None,
    }

  def scope_for(self, pack: ScenePack, task: BenchTask) -> metrics.ScoringScope:
    """What one row is being scored on, in the vocabulary a declaration is written in.

    Every field here is something the run already knows and the leak check
    previously was not told: without it the check received scored episode
    identifiers and declared corpus, split and seed-range names, two vocabularies
    that are disjoint by construction, and answered the same way for every
    submission it ever saw.

    ``seed_range`` is the one field that can still be absent, because a run may
    genuinely not have been told which scoring seeds drew this pack. Absent, a
    declaration narrowing itself on ``seeds=`` comes back uncheckable and withholds
    the row, which is the honest answer to a comparison that did not happen.
    """
    return metrics.ScoringScope(
      corpus=metrics.BENCH_CORPUS,
      split=pack.split,
      seed_epoch=pack.seed_epoch,
      pack_id=pack.pack_id,
      task_id=task.task_id,
      seed_range=self.seed_range,
    )


# -- the declared floor --------------------------------------------------------


def _stable_seed(*parts: str) -> int:
  """A seed from strings alone, reproducible across processes.

  Not ``hash()``: string hashing is salted per interpreter, so a jitter seeded from
  it produces different confidences on every run and a row that is supposed to be
  reproducible is reproducible only within one process.
  """
  total = 0
  for part in parts:
    for character in part:
      total = (total * 131 + ord(character)) % (2 ** 61 - 1)
  return total


class ResidualFloorPredictor:
  """The classical floor row: no weights, no GPU, no checkpoint, and it is declared.

  For the pose task this is :class:`plr_lr.learn.dynamics.ResidualDynamics` over a
  :class:`plr_lr.learn.dynamics.KinematicDynamics` base, iterated one commanded
  delta at a time, which is the model every open-loop script in this repository
  already implies. For the other tasks it is the pack's own declared transition
  evaluated at its published nominal coefficients, which is the honest classical
  answer for a task whose latent is a scalar.

  It knows nothing that is not published. In particular it holds the nominal value
  of every latent an independence-constructed episode redraws, so on those episodes
  it is wrong at the rate the construction guarantees, and it has no channel that
  could notice.
  """

  name = "ResidualDynamics floor"
  license_text = "MIT, this package. No external checkpoint is involved."
  hardware = "CPU only, no accelerator"

  def __init__(self, task: BenchTask, horizon: Optional[int] = None) -> None:
    self.task = task
    self.state_dim = int(task.state_dim)
    self.horizon = int(horizon if horizon is not None else task.max_horizon)
    self.action_space = task.action_space
    self.training_fingerprints = frozenset({NO_TRAINING_DATA})
    self.fitted_components: Tuple[Any, ...] = ()
    self._pose_model = ResidualDynamics(
      base=KinematicDynamics(gain=NOMINAL["S1"]["gain_lo"])
    )

  def _terminal(
    self,
    state: Sequence[float],
    actions: Sequence[Sequence[float]],
  ) -> Tuple[float, ...]:
    if self.task.task_id == "S1":
      pose = np.asarray(state, dtype=float)
      for row in actions:
        delta = np.asarray(row[:4], dtype=float)
        pose = np.asarray(self._pose_model.predict(pose, delta), dtype=float)
      return tuple(float(v) for v in pose)
    return advance(self.task.task_id, state, actions, {})

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    state = [float(v) for v in np.asarray(request.state, dtype=float)]
    actions = [[float(v) for v in row] for row in np.asarray(request.actions, dtype=float)]
    tolerance = [float(v) for v in np.asarray(request.tolerance, dtype=float)]
    terminal = self._terminal(state, actions)

    rng = np.random.default_rng(_stable_seed(request.task_id, request.episode_id))
    spread = 0.0
    flipped = False
    replays: List[Tuple[float, ...]] = []
    for _ in range(FLOOR_SELF_CONSISTENCY_K):
      jittered = []
      for row in actions:
        noise = rng.normal(0.0, FLOOR_JITTER, size=3)
        jittered.append([row[0] + noise[0], row[1] + noise[1], row[2] + noise[2],
                         row[3], row[4]])
      replays.append(self._terminal(state, jittered))
    for index, tau in enumerate(tolerance):
      values = [replay[index] for replay in replays]
      if not all(math.isfinite(v) for v in values):
        spread = float("inf")
        break
      if request.state_units[index] == BOOL_UNIT:
        # A yes/no that flips under jitter is a warning, not a width. Dividing its
        # flip by a tolerance produces a number in the same units as a continuous
        # spread and it is not the same quantity; adding the two would make the
        # abstention bound mean whichever of the two happened to dominate.
        flipped = flipped or (max(values) != min(values))
        continue
      width = (max(values) - min(values)) / tau
      if width > spread:
        spread = width

    horizon = int(request.horizon)
    if not all(math.isfinite(v) for v in terminal) or spread > FLOOR_ABSTAIN_SPREAD:
      return RolloutPrediction.abstained(
        horizon, self.state_dim, readout="self-consistency spread above bound"
      )
    confidence = float(math.exp(-spread))
    if confidence < 0.0:
      confidence = 0.0
    if confidence > 1.0:
      confidence = 1.0
    return RolloutPrediction(
      state=np.tile(np.asarray(terminal, dtype=float), (horizon, 1)),
      confidence=confidence,
      abstain=False,
      flag=bool(flipped),
      readout="analytic-nominal",
      n_forward_passes=1 + FLOOR_SELF_CONSISTENCY_K,
      wall_clock_s=0.0,
    )


def floor_predictor_for(task: BenchTask, horizon: Optional[int] = None) -> Any:
  """The declared floor row for one task. Named so a caller cannot get a different one."""
  return ResidualFloorPredictor(task, horizon)


def floor_predictors(tasks: Sequence[BenchTask] = TASKS) -> Dict[str, Any]:
  """One declared floor predictor per task, keyed by task id."""
  return {task.task_id: floor_predictor_for(task) for task in tasks}


# -- the run -------------------------------------------------------------------


_predictor_for = resolve_predictor


def _threshold_from(screen: Screen) -> FrozenThreshold:
  """The frozen act/defer threshold, carrying its calibration source verbatim.

  An uncalibrated screen is refused rather than defaulted. A default threshold of
  zero admits every prediction ever made and looks, in every log and every table,
  exactly like a calibrated screen that found the model confident.
  """
  if not screen.is_calibrated or math.isnan(float(screen.theta)):
    raise ThresholdNotFrozen(
      "run_bench needs a calibrated screen. An uncalibrated one carries theta NaN, "
      "and substituting any number for it here would produce an act/defer table "
      "decided by a threshold nobody chose. Fit one with the screen-fit verb."
    )
  return FrozenThreshold(
    value=float(screen.theta),
    calibrated_on=screen.calibrated_on,
    n_calibration=max(1, int(screen.n_calibration)),
  )


def _declared(predictor: Any, screen: Screen, seeds: SeedPlan) -> Dict[str, Any]:
  fingerprints = sorted(getattr(predictor, "training_fingerprints", ()) or ())
  components = getattr(predictor, "fitted_components", ())
  return {
    "training_fingerprints": fingerprints or ["UNDECLARED"],
    "fitted_components": [
      f"{c.name} fit on {c.fit_on_split} over {c.n_examples} examples"
      for c in components
    ] or ["none declared"],
    "license_text": getattr(predictor, "license_text", "UNDECLARED"),
    "hardware": getattr(predictor, "hardware", metrics.UNDECLARED_HARDWARE),
    "threshold": screen.calibrated_on,
    "seeds": seeds.to_dict(),
  }


def _gate(
  predictor: Any,
  tasks: Sequence[BenchTask],
  pack: ScenePack,
  seeds: SeedPlan,
) -> Dict[str, InterventionResult]:
  out: Dict[str, InterventionResult] = {}
  for task in tasks:
    episodes = pack.for_task(task.task_id)
    if len(episodes) < INTERVENTION_K:
      raise ValueError(
        f"task {task.task_id} carries {len(episodes)} episodes and the gate needs at "
        f"least {INTERVENTION_K}. Running it on fewer would make the gate outcome "
        "attributable to the harness rather than to the model."
      )
    horizon = int(task.horizons[0])
    requests = [build_request(task, episode, horizon) for episode in episodes]
    out[task.task_id] = interventional_check(
      _predictor_for(predictor, task), requests, seed=seeds.gate_seed
    )
  return out


def risk_at_full_coverage(rows: EpisodeRows) -> Optional[float]:
  """The risk the trust horizon is measured against: every episode in the denominator.

  ``metrics.trust_horizon`` and :attr:`plr_lr.learn.screen.Screen.trust_horizon` both
  name this quantity "risk at full coverage", and full coverage means the whole task
  and not the part of it the model agreed to answer. An abstention is therefore a row
  in the denominator and a miss in the numerator, exactly as
  :attr:`EpisodeRows.tol_acc` already counts it.

  The measured defect this replaces took the risk over ``rows.ranked()``, which drops
  the abstained rows from both halves of the fraction. That is selective risk at
  whatever coverage the model happened to choose, and it is a different point on the
  risk-coverage curve for every abstention rate, so it is not comparable across
  horizons or across models. Its consequence is worse than incomparability:
  ``k*`` is what :meth:`plr_lr.learn.screen.Screen.check` refuses ``HORIZON_EXCEEDED``
  above, so it decides how many actions the arm runs open-loop before it looks again,
  and a denominator chosen by the model's own refusals means abstaining more buys a
  longer open-loop chunk. Measured on the S3 demonstration pack at 200 episodes: one
  predictor, 40 percent miss rate at horizon 2, ``k*`` of 1 when it answers everything
  and 2 once it withholds half of the episodes it was about to get wrong.

  Counting an abstention as a miss here is not a second penalty for a refusal that was
  already priced elsewhere. The screen refuses an abstained prediction on its own path,
  by reason ``ABSTAIN``, whatever ``k*`` is. ``k*`` is a claim about the horizon rather
  than about one episode, and a model that will not answer at horizon ``k`` has not
  shown that horizon ``k`` is safe.

  ``None`` when there are no episodes at all, which is a task with nothing in it rather
  than a task the model refused: a zero-length denominator has no risk, and returning
  a number for one would let an empty task print as trustworthy.
  """
  if not rows.n:
    return None
  return 1.0 - rows.tol_acc


def _row(
  task: BenchTask,
  horizon: int,
  rows: EpisodeRows,
  pack: ScenePack,
  seeds: SeedPlan,
  threshold: FrozenThreshold,
  predictor: Any,
  action_conditioned: bool,
  n_permutations: int,
  n_resamples: Optional[int],
) -> BenchRow:
  selective = score(
    rows.outcomes(),
    threshold,
    task_id=task.task_id,
    horizon=horizon,
    seed=seeds.tie_break_seed,
    n_permutations=int(n_permutations),
  )
  predicted_errors = [
    error for error, abstained in zip(rows.errors, rows.abstained) if not abstained
  ]
  tpe = metrics.TolerancePredictionError.of(predicted_errors or [float("inf")])
  flagged_errors = [
    error
    for error, abstained, flag in zip(rows.errors, rows.abstained, rows.flagged)
    if not abstained and flag
  ]
  flagged_tpe = (
    metrics.TolerancePredictionError.of(flagged_errors) if flagged_errors else None
  )
  gap = metrics.success_gap(
    task.task_id,
    horizon,
    rows.loose_hits,
    rows.hits,
    seed=seeds.bootstrap_seed,
    n_resamples=n_resamples,
  )
  compute = metrics.ComputeDeclaration.of(
    rows.n_forward_passes,
    rows.wall_clock_s,
    hardware=getattr(predictor, "hardware", metrics.UNDECLARED_HARDWARE),
  )
  declared_pixels = getattr(predictor, "declared_pixel_metrics", None)
  if callable(declared_pixels):
    psnr, ssim = declared_pixels()
    pixel = metrics.PixelColumn(
      psnr=float(psnr), ssim=float(ssim), declared_by=getattr(predictor, "name", "adapter")
    )
  else:
    pixel = metrics.PixelColumn()
  overlap = metrics.training_overlap(
    rows.episode_ids,
    getattr(predictor, "training_fingerprints", ()) or (),
    seeds.scope_for(pack, task),
  )
  refusals: List[Refusal] = []
  under = underpowered_refusal(rows.n, f"{task.task_id}@{horizon}")
  if under is not None:
    refusals.append(under)
  if overlap.leaked:
    refusals.append(
      _refusal(
        "R7",
        f"{task.task_id}@{horizon}: the declared training set does not clear these "
        f"rows, so the provenance this score would stand on is not established. "
        f"{overlap.format()}",
      )
    )
  return BenchRow(
    task_id=task.task_id,
    horizon=int(horizon),
    seed_epoch=pack.seed_epoch,
    pack_id=pack.pack_id,
    split=pack.split,
    basis=basis_label(pack),
    tolerance=tuple(float(t) for t in task.tolerance),
    n=rows.n,
    selective=selective,
    tpe=tpe,
    gap=gap,
    compute=compute,
    pixel=pixel,
    overlap=overlap,
    action_conditioned=bool(action_conditioned),
    flagged_tpe=flagged_tpe,
    refusals=tuple(refusals),
  )


def run_bench(
  predictor: Any,
  tasks: Sequence[BenchTask],
  pack: ScenePack,
  screen: Screen,
  seeds: SeedPlan,
  n_permutations: int = PERMUTATION_RESAMPLES,
  n_resamples: Optional[int] = None,
  model_name: Optional[str] = None,
) -> BenchReport:
  """Gate, then controls, then rows. Never any other order.

  Returns a report in every case, including the cases where it contains no rows at
  all. A refusal is a result and is published like one: a run that produced no score
  because its model ignores its action argument reads differently from a run that
  produced a low score, and collapsing the two is the thing this benchmark exists to
  stop.
  """
  tasks = tuple(tasks)
  name = model_name or getattr(predictor, "name", type(predictor).__name__)
  if isinstance(predictor, dict) and model_name is None:
    first = next(iter(predictor.values()))
    name = getattr(first, "name", "predictor set")
  declared = _declared(_predictor_for(predictor, tasks[0]), screen, seeds)
  base = {
    "model": name,
    "pack_id": pack.pack_id,
    "seed_epoch": pack.seed_epoch,
    "split": pack.split,
    "basis": basis_label(pack),
    "declared": declared,
  }

  pack.check_interleaved()

  try:
    threshold = _threshold_from(screen)
  except ThresholdNotFrozen as exc:
    return BenchReport(
      refusals=(_refusal("R3", str(exc)),),
      **base
    )

  try:
    intervention = _gate(predictor, tasks, pack, seeds)
  except ProvenanceUndeclared as exc:
    return BenchReport(refusals=(_refusal("R7", str(exc)),), **base)
  except UndeclaredComponent as exc:
    return BenchReport(
      refusals=(_refusal("R8", str(exc)),), **base
    )

  failed = sorted(t for t, result in intervention.items() if not result.passed)
  if failed:
    return BenchReport(
      intervention=intervention,
      refusals=(
        _refusal(
          "R4",
          f"tasks {failed} did not clear the action-conditioning gate, so no score is "
          "emitted for this model at all. This is a refusal and not a low score: a low "
          "score still gets ranked, and a ranked non-world-model is the exact "
          "claim-versus-evidence mismatch this benchmark indicts.",
        ),
      ),
      **base
    )

  subject_rows: Dict[Tuple[str, int], EpisodeRows] = {}
  try:
    for task in tasks:
      for horizon in task.horizons:
        subject_rows[(task.task_id, int(horizon))] = evaluate_rows(
          _predictor_for(predictor, task), task, pack, int(horizon)
        )
  except ProvenanceUndeclared as exc:
    return BenchReport(
      intervention=intervention,
      refusals=(_refusal("R7", str(exc)),),
      **base
    )
  except UndeclaredComponent as exc:
    return BenchReport(
      intervention=intervention,
      refusals=(_refusal("R8", str(exc)),),
      **base
    )

  controls = run_control_suite(
    predictor,
    pack,
    tasks=tasks,
    seed=seeds.control_seed,
    n_permutations=n_permutations,
    n_resamples=n_resamples,
    subject_rows=subject_rows,
  )
  if not controls.passed:
    return BenchReport(intervention=intervention, controls=controls, **base)

  rows: List[BenchRow] = []
  trust: Dict[str, Optional[int]] = {}
  for task in tasks:
    risk_by_horizon: Dict[int, Optional[float]] = {}
    for horizon in task.horizons:
      evaluated = subject_rows[(task.task_id, int(horizon))]
      rows.append(
        _row(
          task,
          int(horizon),
          evaluated,
          pack,
          seeds,
          threshold,
          _predictor_for(predictor, task),
          controls.action_conditioned.get(task.task_id, False),
          n_permutations,
          n_resamples,
        )
      )
      risk_by_horizon[int(horizon)] = risk_at_full_coverage(evaluated)
    trust[task.task_id] = metrics.trust_horizon(risk_by_horizon)

  return BenchReport(
    rows=tuple(rows),
    controls=controls,
    intervention=intervention,
    trust_horizons=trust,
    **base
  )


__all__ = [
  "FLOOR_ABSTAIN_SPREAD",
  "FLOOR_JITTER",
  "FLOOR_SELF_CONSISTENCY_K",
  "ResidualFloorPredictor",
  "SeedPlan",
  "floor_predictor_for",
  "floor_predictors",
  "risk_at_full_coverage",
  "run_bench",
]
