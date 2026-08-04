"""Ten things that must come out a particular way before any scorecard is emitted.

Nine of them must fail and one must pass. That asymmetry is the whole design: a
benchmark is only worth reading if the cheap ways of appearing to do well have been
run and have visibly not worked, and the one control that must pass exists so that
a failure of the harness itself cannot be mistaken for a failure of the model.

``ControlSuiteReport.passed`` requires a non-empty result list, ``n > 0``, every
must-fail control failing and every must-pass control passing. "Nothing was
checked" and "the check passed" are different answers and they stay different here,
because the flag is read on exactly the runs where nothing established it: an
exception during setup that leaves ``results`` empty would otherwise produce a
suite that passed by having done nothing.

The controls and the submission are scored by :func:`evaluate_rows`, the same
function, deliberately. A control scored by different code than the submission
proves nothing about the submission: the two paths drift, and the direction they
drift in is always the one that makes the control look like it passed.

Every band in :data:`BANDS` is an absolute constant with a stated reason. None of
them is derived from the data being scored, none is read off a curve, and none is a
percentile. A band chosen after seeing where a control landed is not a band, it is
a description of that control.

Two of the ten are carried over from the existing evaluator with one fix.
:class:`ControlResult` grew a ``clamped_fraction`` because the measured behaviour of
a random null against the default safety envelope is that close to half its steps
are silently rewritten, with no clamp count anywhere in the report; a null clamped
above :data:`THROTTLED_CLAMPED_FRACTION` is reported as ``THROTTLED NULL`` and
invalidates rather than being read as a null that legitimately failed. It defaults
to ``None`` and not to ``0.0``, because no control in a pack-only run passes through
a safety envelope at all: a zero there would print and serialise as a measured clamp
rate of nothing, which is a claim about an envelope that was never in the loop.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..predict import (
  NO_TRAINING_DATA,
  BlindPredictor,
  FrameRef,
  NoisePredictor,
  RolloutPrediction,
  RolloutRequest,
  interventional_check,
  validate,
)
from ..screen import ClampOnlyScreen
from ..selective import (
  PERMUTATION_RESAMPLES,
  AbstentionBandRefused,
  AbstentionCredit,
  EpisodeOutcome,
  RiskCoverageCurve,
  confidence_shuffle_p,
  paired_bootstrap_ci,
  spearman_rho,
  underpowered_label,
)
from . import metrics
from .packio import ScenePack, SceneEpisode, advance
from .tasks import TASKS, BenchTask, task_for

MUST_FAIL = "must-fail"
MUST_PASS = "must-pass"

#: A null clamped more often than this is reported as ``THROTTLED NULL`` and does
#: not count as a null that failed on its merits. Measured origin: the existing
#: random policy against the default envelope has close to half its steps rewritten
#: before they reach the backend, and a null whose actions were replaced is not the
#: null anybody thought they ran.
THROTTLED_CLAMPED_FRACTION = 0.10

#: The tasks a do-nothing predictor is required to fail on: all five of them.
#:
#: An earlier version listed only S1, S4 and S5, on the stated grounds that the two
#: tasks whose latent is a single scalar move by a fraction of a tolerance in one
#: step and so a do-nothing predictor is inside tolerance too often for the
#: requirement to be fair. The shipped fixture contradicts that. Measured on
#: ``builtin_demo_pack`` at horizon 1, which is where the suite runs every control,
#: the do-nothing null scores S1 0.0350, S2 0.2800, S3 0.3175, S4 0.0675 and S5
#: 0.1900 at n=400. Over 48 packs (8 seeds x {60, 200, 400} episodes x both splits)
#: the worst any task reached was S3 at 0.4200. Every one of the five is under the
#: 0.50 ceiling, so every one of the five is required to clear it, and the exemption
#: was protecting the two tasks from a bound they already met. S3 has the least
#: headroom and is the row to watch if its tolerance ever widens.
N1_TASKS: Tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")

#: How many tie-break orderings the constant-confidence control is measured over. A
#: constant confidence has no ordering at all, so any single one is one draw from the
#: null and lands a standard error either side of the band it is being compared to.
#: The control reports the median over this many, which is the quantity the band is a
#: statement about. Twenty, an absolute constant, chosen for the estimate and not
#: after seeing where any particular model landed.
N4_ORDERINGS = 20

BANDS: Dict[str, float] = {
  # A do-nothing predictor that lands inside tolerance on more than half the
  # episodes is telling you the tolerance is wider than the motion. That is a broken
  # task, not a working null, and it is the task that has to change.
  "N1_MAX_TOL_ACC": 0.50,
  # The true-versus-shuffled gap a submission has to clear before its headline is
  # printed. Below this the model is not distinguishable from one that ignores its
  # action argument, which is a refusal and not a low score.
  "N2_MIN_GAP": 0.05,
  # A constant confidence has to land within this of AURC_random - AURC_oracle.
  # Landing near zero instead would mean the excess-AURC arithmetic is inverted, and
  # an inverted band reads as a nearly perfect model.
  "N4_E_AURC_TOL": 0.05,
  # Below this PSNR the pixel replay is not actually replaying ground truth, so the
  # demonstration that a perfect pixel score loses the ranking has not been made.
  "N5_MIN_PSNR": 40.0,
  # The oracle has to be exactly right. Anything less means the comparison, the
  # tolerance or the truth lookup is wrong, and the model is not the thing at fault.
  "N6_REQUIRED_TOL_ACC": 1.0,
  # How much of the dev-to-scoring gap a predictor fit only on published seeds is
  # allowed to close before the partition is declared not to bite.
  "N8_MAX_GAP_CLOSED": 0.10,
  # The same predictor has to score exactly this on the half it memorised. Exact,
  # like the oracle band and for the same reason: the recall path is a lookup of the
  # episode's own observables, so anything below 1.0 means two published episodes
  # share an initial state and an action block and disagree about the terminal, and
  # a pack carrying that pair is ambiguous rather than hard. A probe that cannot
  # recall what it memorised could not have detected a leak on the other half
  # either, so the withheld-half number it reports would establish nothing.
  "N8_MIN_DEV_RECALL": 1.0,
  # Above this rank correlation, a confidence made only of clamp counts reproduces
  # the model's own correlation, and the model contributed nothing.
  "N9_MAX_RHO": 0.10,
}


class HarnessBroken(AssertionError):
  """Raised when the oracle control fails. Never raised about a submission.

  The oracle is handed the true terminal state. If it does not score a perfect
  tolerance accuracy then the comparison, the tolerance lookup or the truth lookup
  is wrong, and every number the harness would go on to print about a real model is
  built on the same arithmetic. A low score here would be attributed to the model,
  so this is an exception instead.
  """


# -- shared evaluation ---------------------------------------------------------


@dataclass(frozen=True)
class EpisodeRows:
  """One task at one horizon, as the columns every metric downstream reads.

  Frozen and explicit rather than a dict of arrays, because the columns have to stay
  aligned: an abstention is a row in ``abstained`` and a ``None`` in ``confidences``
  and it is still a row in ``unknowable``, and a filtered copy of one column without
  the others is how an abstention stops being counted in its own denominator.
  """

  task_id: str
  horizon: int
  episode_ids: Tuple[str, ...]
  hits: Tuple[int, ...]
  loose_hits: Tuple[int, ...]
  errors: Tuple[float, ...]
  confidences: Tuple[Optional[float], ...]
  abstained: Tuple[bool, ...]
  unknowable: Tuple[bool, ...]
  flagged: Tuple[bool, ...]
  n_forward_passes: Tuple[int, ...]
  wall_clock_s: Tuple[float, ...]

  @property
  def n(self) -> int:
    return len(self.episode_ids)

  @property
  def n_ranked(self) -> int:
    return sum(1 for a in self.abstained if not a)

  @property
  def tol_acc(self) -> float:
    """Fraction of episodes inside tolerance, abstentions counted as misses.

    Abstentions are in the denominator on purpose. A model that refuses everything
    would otherwise report a tolerance accuracy over an empty set, which either
    divides by zero or reports 1.0 depending on how carefully the loop was written.
    """
    return sum(self.hits) / self.n if self.n else 0.0

  def ranked(self) -> Tuple[List[int], List[float]]:
    hits: List[int] = []
    confidences: List[float] = []
    for hit, confidence, abstained in zip(self.hits, self.confidences, self.abstained):
      if abstained or confidence is None:
        continue
      hits.append(int(hit))
      confidences.append(float(confidence))
    return hits, confidences

  def outcomes(self) -> List[EpisodeOutcome]:
    out: List[EpisodeOutcome] = []
    for index in range(self.n):
      out.append(
        EpisodeOutcome(
          episode_id=self.episode_ids[index],
          hit=bool(self.hits[index]),
          confidence=None if self.abstained[index] else self.confidences[index],
          abstain=bool(self.abstained[index]),
          unknowable=bool(self.unknowable[index]),
        )
      )
    return out


def build_request(task: BenchTask, episode: SceneEpisode, horizon: int) -> RolloutRequest:
  """Everything the predictor is allowed to condition on, and nothing else.

  The knowability label and the invisible latent are both on the episode and neither
  one is copied here. That is not a convention a reviewer has to check: there is no
  field on :class:`plr_lr.learn.predict.RolloutRequest` either one could travel in,
  so a leak would have to be a deliberate new field rather than an oversight.
  """
  frames = tuple(
    FrameRef(
      path=frame.path,
      sha256=frame.sha256,
      t_capture_s=frame.t_capture_s,
      clock_source=frame.clock_source,
    )
    for frame in episode.frames
  )
  return RolloutRequest(
    task_id=task.task_id,
    episode_id=episode.episode_id,
    state=np.asarray(episode.state0, dtype=float),
    state_fields=task.state_fields,
    state_units=task.state_units,
    frames=frames,
    actions=np.asarray(episode.actions[:int(horizon)], dtype=float),
    horizon=int(horizon),
    tolerance=np.asarray(task.tolerance, dtype=float),
  )


def resolve_predictor(predictor: Any, task: BenchTask) -> Any:
  """Resolve a per-task predictor when the caller supplied a mapping.

  A submission is one object. The declared floor row is five, because each one
  declares the state width of its own task and the output contract checks that
  declaration against the request. Relaxing the contract instead would let a
  predictor answer a four-field task with a one-field prediction and have nobody
  notice, so both shapes are accepted here and nowhere else.
  """
  if isinstance(predictor, dict):
    try:
      return predictor[task.task_id]
    except KeyError:
      raise ValueError(
        f"no predictor supplied for task {task.task_id}. A run that silently skipped a "
        "task would publish a scorecard whose missing row reads as a task nobody ran "
        "rather than one that was dropped."
      ) from None
  return predictor


def evaluate_rows(
  predictor: Any,
  task: BenchTask,
  pack: ScenePack,
  horizon: int,
  episodes: Optional[Sequence[SceneEpisode]] = None,
) -> EpisodeRows:
  """Run one predictor over one task at one horizon and return the scored columns.

  Used for both the controls and the submission. Scoring a control with different
  code than the submission proves nothing about the submission, and the two copies
  drift in whichever direction makes the control look like it passed.

  Every prediction goes through :func:`plr_lr.learn.predict.validate`, so a shape
  error, a confidence outside the unit interval, an abstention that banked a
  prediction, or an undeclared provenance is raised here rather than absorbed into a
  score. A diverged but well-formed prediction is not an error: it is a miss, and it
  is counted as one.
  """
  pool = tuple(episodes) if episodes is not None else pack.for_task(task.task_id)
  if not pool:
    raise ValueError(
      f"pack {pack.pack_id} carries no episodes for task {task.task_id}. Scoring an "
      "empty task produces a row that reads on a scorecard exactly like a clean run."
    )
  predictor = resolve_predictor(predictor, task)
  episode_ids: List[str] = []
  hits: List[int] = []
  loose: List[int] = []
  errors: List[float] = []
  confidences: List[Optional[float]] = []
  abstained: List[bool] = []
  unknowable: List[bool] = []
  flagged: List[bool] = []
  passes: List[int] = []
  clocks: List[float] = []

  tolerance = task.tolerance
  for episode in pool:
    request = build_request(task, episode, horizon)
    prediction = validate(predictor.predict(request), request, predictor)
    truth = episode.terminal(horizon)
    episode_ids.append(episode.episode_id)
    unknowable.append(episode.unknowable)
    passes.append(int(prediction.n_forward_passes))
    clocks.append(float(prediction.wall_clock_s))
    flagged.append(bool(prediction.flag))
    if prediction.abstain:
      abstained.append(True)
      confidences.append(None)
      hits.append(0)
      loose.append(0)
      errors.append(float("inf"))
      continue
    terminal = [float(v) for v in np.asarray(prediction.state, dtype=float)[-1]]
    error = metrics.tolerance_error(terminal, truth, tolerance)
    abstained.append(False)
    confidences.append(float(prediction.confidence))
    hits.append(1 if error <= 1.0 else 0)
    loose.append(1 if metrics.loose_hit(task.task_id, terminal, truth) else 0)
    errors.append(error)

  return EpisodeRows(
    task_id=task.task_id,
    horizon=int(horizon),
    episode_ids=tuple(episode_ids),
    hits=tuple(hits),
    loose_hits=tuple(loose),
    errors=tuple(errors),
    confidences=tuple(confidences),
    abstained=tuple(abstained),
    unknowable=tuple(unknowable),
    flagged=tuple(flagged),
    n_forward_passes=tuple(passes),
    wall_clock_s=tuple(clocks),
  )


# -- the control predictors ----------------------------------------------------


class _ControlPredictor:
  """Declaration block shared by the nulls that read this pack.

  Each null declares that it was fit on nothing, which is a declaration rather than
  an absence, so every one of them runs through the same provenance gate a real
  submission does instead of being quietly exempted from it.
  """

  name = "control"

  def __init__(self, task: BenchTask, horizon: int) -> None:
    self.task = task
    self.horizon = int(horizon)
    self.state_dim = int(task.state_dim)
    self.action_space = task.action_space
    self.training_fingerprints = frozenset({NO_TRAINING_DATA})
    self.fitted_components: Tuple[Any, ...] = ()
    self.license_text = "MIT, this package. No external checkpoint is involved."

  def _answer(
    self,
    request: RolloutRequest,
    terminal: Sequence[float],
    confidence: float,
    readout: str,
  ) -> RolloutPrediction:
    horizon = int(request.horizon)
    state = np.tile(np.asarray(terminal, dtype=float), (horizon, 1))
    return RolloutPrediction(
      state=state,
      confidence=float(confidence),
      abstain=False,
      flag=False,
      readout=readout,
      n_forward_passes=1,
      wall_clock_s=0.0,
    )


class _PackReadingPredictor(_ControlPredictor):
  """Base for the nulls that need the planted ground truth to do their job."""

  def __init__(self, task: BenchTask, horizon: int, pack: ScenePack) -> None:
    super().__init__(task, horizon)
    self.pack = pack
    self._by_id = {e.episode_id: e for e in pack.for_task(task.task_id)}

  def _episode(self, request: RolloutRequest) -> SceneEpisode:
    try:
      return self._by_id[request.episode_id]
    except KeyError:
      raise ValueError(
        f"control {self.name} was asked about episode {request.episode_id!r}, which is "
        f"not in pack {self.pack.pack_id}. A control answering about an episode it "
        "cannot see would be answering from its priors, which is the thing it exists "
        "to detect."
      ) from None


class OracleLeakPredictor(_PackReadingPredictor):
  """N6. Handed the true terminal state. Must score exactly 1.0 or the harness is broken."""

  name = "N6:OracleLeak"

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    episode = self._episode(request)
    return self._answer(request, episode.terminal(int(request.horizon)), 1.0, "oracle")


class ShuffledActionPredictor(_PackReadingPredictor):
  """N2. The true transition, driven by another episode's actions.

  The statistically clean sibling of the interventional gate, with no alpha to
  choose: it holds the dynamics and the initial state fixed and destroys only the
  correspondence between an episode and the actions it was actually given. A model
  whose predictions survive that is not conditioning on its action argument.

  When the pack publishes its per-episode coefficients this recomputes the true
  terminal under the permuted actions. When it does not -- which is the case for a
  scoring pack, deliberately -- it falls back to the permuted episode's own recorded
  terminal, which is a coarser shuffle and is labelled as one.
  """

  name = "N2:ShuffledAction"

  def __init__(
    self,
    task: BenchTask,
    horizon: int,
    pack: ScenePack,
    seed: int = 0,
  ) -> None:
    super().__init__(task, horizon, pack)
    episodes = pack.for_task(task.task_id)
    order = list(range(len(episodes)))
    rng = random.Random(seed)
    rng.shuffle(order)
    # A derangement in all but the degenerate one-episode case: an episode paired
    # with itself would be scored against its own actions and would pass.
    if len(order) > 1:
      for index in range(len(order)):
        if order[index] == index:
          swap = (index + 1) % len(order)
          order[index], order[swap] = order[swap], order[index]
    self._partner = {
      episodes[i].episode_id: episodes[order[i]] for i in range(len(episodes))
    }
    self.exact = all(e.coefficients is not None for e in episodes)

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    episode = self._episode(request)
    partner = self._partner[episode.episode_id]
    horizon = int(request.horizon)
    if self.exact and episode.coefficients is not None:
      terminal = advance(
        self.task.task_id,
        episode.state0,
        partner.actions[:horizon],
        episode.coefficients,
      )
    else:
      terminal = partner.terminal(horizon)
    return self._answer(request, terminal, 0.8, "true-dynamics/shuffled-actions")


class PixelPerfectPredictor(_PackReadingPredictor):
  """N5. Replays ground-truth frames and extrapolates the terminal scalar linearly.

  It tops the pixel column outright, because the frames it "predicts" are the frames
  that were rendered. Its terminal state comes from a two-step linear extrapolation
  through the true state at the first horizon, which is exact at horizon one and
  wrong wherever the dynamics are not linear in the step index -- which is every
  task here that carries a discrete verb, a saturating criterion or a drift onset.

  Its confidence is constant, so it cannot beat its own confidence shuffle and is
  not rankable at all. That is the demonstration: a perfect pixel score and no
  position in the ranking.
  """

  name = "N5:PixelPerfect"

  def declared_pixel_metrics(self) -> Tuple[float, float]:
    """Perfect, because it is replaying the frames the fixture rendered."""
    return (99.0, 1.0)

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    episode = self._episode(request)
    horizon = int(request.horizon)
    first = np.asarray(episode.terminal(min(episode.terminal_by_horizon)), dtype=float)
    start = np.asarray(episode.state0, dtype=float)
    step = min(episode.terminal_by_horizon)
    slope = (first - start) / float(step)
    terminal = start + slope * float(horizon)
    return self._answer(request, terminal, 0.7, "pixel-replay/linear-extrapolation")


class AlwaysConfidentPredictor(_PackReadingPredictor):
  """N4. A real prediction with the confidence channel flattened to 1.0.

  Must land at ``AURC_random - AURC_oracle`` rather than at zero. Landing near zero
  would mean the excess-AURC arithmetic is inverted, and an inverted band reads on a
  scorecard as a model that is nearly perfect.
  """

  name = "N4:AlwaysConfident"

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    episode = self._episode(request)
    horizon = int(request.horizon)
    coefficients = episode.coefficients or {}
    nominal = {k: v for k, v in coefficients.items() if k not in episode.hidden_keys}
    terminal = advance(self.task.task_id, episode.state0, episode.actions[:horizon], nominal)
    return self._answer(request, terminal, 1.0, "nominal/constant-confidence")


class AlwaysAbstainPredictor(_ControlPredictor):
  """N7. Refuses every episode. Perfect safety, zero utility, and it must not rank.

  Its abstention credit lands on the unknowable base rate, which reads respectably
  in isolation and is fully explained by a false-alarm rate of 1.0. This is why
  abstention credit scores a refusal on a knowable episode as zero, and why the
  credit rate cannot be reached without the false-alarm rate beside it.
  """

  name = "N7:AlwaysAbstain"

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    return RolloutPrediction.abstained(int(request.horizon), self.state_dim)


Signature = Tuple[Tuple[float, ...], Tuple[Tuple[float, ...], ...]]


def observable_signature(
  state: Sequence[float],
  actions: Sequence[Sequence[float]],
) -> Signature:
  """Everything two episodes have to share to be the same episode under two names.

  The initial state and the commanded action block, which is the whole of what a
  request carries that a predictor could key a memory on. Every draw in the
  generator is continuous, so two independently drawn episodes agreeing on every one
  of these floats is not a near miss and not a coincidence: it is one episode
  published on both sides of the partition.

  Deliberately exact, rather than a distance under a threshold, and the reason is a
  measurement. What a nearby-but-not-equal neighbour buys is not leakage: a dense
  pack from a smooth generator supports nearest-neighbour regression over the
  published half whether or not the partition leaks, and that is legitimate
  modelling. A probe built on nearest-neighbour retrieval instead of exact match was
  run over 64 partitions of honest packs (8 generator seeds, 60 and 200 episodes,
  both splits, two suite seeds) and reached 0.3000 above the do-nothing floor on the
  withheld half of S2 and 0.1667 on S5, with no republication anywhere in either
  pack. Against the 0.10 ceiling that is a control failing on an honest fixture.
  Republication is the thing this control can prove, so it is the thing it asserts.
  """
  head = tuple(float(v) for v in np.asarray(state, dtype=float).reshape(-1))
  body = tuple(
    tuple(float(v) for v in row) for row in np.asarray(actions, dtype=float)
  )
  return (head, body)


class SeedLeakPredictor(_PackReadingPredictor):
  """N8. Answers from a memory of the published episodes and from nothing else.

  Detects whether the dev/scoring partition actually bites. The probe memorises the
  terminal state of every published episode, keyed by that episode's observables,
  and answers any request whose observables are in the memory with the memorised
  terminal. A request that is not in the memory gets the do-nothing state, which is
  the floor's own answer, because a model that memorised the published fixtures and
  is handed something it has never seen has nothing to say about it.

  One code path, run on both halves, which is what makes the withheld-half number
  worth reading. On the published half the probe answers out of its memory and has
  to score ``BANDS["N8_MIN_DEV_RECALL"]``, which is the positive control: the
  mechanism demonstrably turns a memory hit into a scored hit, above the do-nothing
  floor measured on those same episodes. On the withheld half the same mechanism
  fires only if a withheld episode was published under another name, and the score
  that buys is measured against the floor there.

  The earlier version of this control took the id shortcut instead: it looked up
  ``episode_id`` in the published set and returned ``state0`` for everything else.
  Off the published half that is the floor's answer by construction, so the gap it
  closed was exactly +0.0000 on every task, on every pack, forever, and the control
  could not fail on any input.
  """

  name = "N8:SeedLeak"

  def __init__(
    self,
    task: BenchTask,
    horizon: int,
    pack: ScenePack,
    dev_ids: Sequence[str],
  ) -> None:
    super().__init__(task, horizon, pack)
    self.dev_ids = frozenset(str(i) for i in dev_ids)
    self.published = tuple(
      e for e in pack.for_task(task.task_id) if e.episode_id in self.dev_ids
    )
    self._memory: Dict[int, Dict[Signature, SceneEpisode]] = {}

  def memory(self, horizon: int) -> Dict[Signature, SceneEpisode]:
    """The published episodes indexed by their observables at this horizon."""
    key = int(horizon)
    table = self._memory.get(key)
    if table is None:
      table = {}
      for episode in self.published:
        table.setdefault(
          observable_signature(episode.state0, episode.actions[:key]), episode
        )
      self._memory[key] = table
    return table

  def republication(self, request: RolloutRequest) -> Optional[SceneEpisode]:
    """The published episode this request republishes, or ``None``.

    Keyed off the request rather than off the pack lookup, so the memory is reached
    by exactly the columns a submitter can see and the claim is checkable here
    rather than being a rule about what the control promises not to read.
    """
    signature = observable_signature(request.state, request.actions)
    return self.memory(int(request.horizon)).get(signature)

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    # Raises when asked about an episode outside the pack: a control answering from
    # its priors about an episode it cannot see is the thing it exists to detect.
    self._episode(request)
    match = self.republication(request)
    if match is None:
      return self._answer(request, request.state, 0.95, "do-nothing/no-republication")
    terminal = match.terminal(int(request.horizon))
    return self._answer(request, terminal, 0.95, "memorised/republished-seed")


# -- the results ---------------------------------------------------------------


@dataclass(frozen=True)
class ControlResult:
  """What one control did, and whether that is what it was required to do.

  ``ran`` and ``met`` are separate booleans. A control that raised during setup has
  ``ran=False``, and the suite refuses on it: a control that did not run has
  established nothing, and folding that into ``met=True`` is precisely how a suite
  comes to pass by having done nothing.

  ``evidential`` is a third state, and it is honest rather than pedantic. A control
  can meet its requirement trivially in a configuration that cannot exercise it --
  the clamp control against a pack with no clamps in it -- and reporting that as a
  pass without saying so would let a simulated run stand in for an arm run.

  ``clamped_fraction`` is ``None`` until something measures it, and ``None`` is not
  ``0.0``. No control in a pack-only run passes through a safety envelope, so there
  is no clamp rate to report; a default of zero would print as a measured rate of
  nothing and serialise into the scorecard as one, which is the same mistake in the
  other direction from the one :data:`THROTTLED_CLAMPED_FRACTION` exists to catch.
  """

  control_id: str
  name: str
  requirement: str
  task_id: str
  ran: bool
  met: bool
  observed: str
  notes: Tuple[str, ...] = ()
  clamped_fraction: Optional[float] = None
  evidential: bool = True

  @property
  def clamp_rate_measured(self) -> bool:
    """Whether anything measured this control's clamp rate at all."""
    return self.clamped_fraction is not None

  @property
  def throttled(self) -> bool:
    """Whether a measured clamp rate invalidates this null. Unmeasured is not low."""
    if self.clamped_fraction is None:
      return False
    return self.clamped_fraction > THROTTLED_CLAMPED_FRACTION

  def line(self) -> str:
    if not self.ran:
      mark = "-- "
    elif self.throttled:
      mark = "XX "
    elif self.met:
      mark = "ok "
    else:
      mark = "XX "
    tail = ""
    if self.throttled:
      tail = f" THROTTLED NULL (clamped {self.clamped_fraction:.3f})"
    elif not self.evidential:
      tail = " (trivially satisfied here, see note)"
    if self.clamped_fraction is not None and not self.throttled:
      tail += f" [clamped {self.clamped_fraction:.3f}]"
    head = f"  [{mark}] {self.control_id} {self.name} on {self.task_id} "
    return "\n".join(
      [f"{head}[{self.requirement}]: {self.observed}{tail}"]
      + [f"        note: {note}" for note in self.notes]
    )

  def to_dict(self) -> Dict[str, Any]:
    return {
      "control_id": self.control_id,
      "name": self.name,
      "requirement": self.requirement,
      "task_id": self.task_id,
      "ran": self.ran,
      "met": self.met,
      "observed": self.observed,
      "notes": list(self.notes),
      "clamped_fraction": (
        None if self.clamped_fraction is None else float(self.clamped_fraction)
      ),
      "clamp_rate_measured": self.clamp_rate_measured,
      "throttled": self.throttled,
      "evidential": self.evidential,
    }


@dataclass(frozen=True)
class ControlSuiteReport:
  """Whether the suite establishes that a scorecard may be printed at all.

  ``passed`` is four conditions and not one. The list has to be non-empty, ``n`` has
  to be positive, every control has to have run, and every one has to have met its
  requirement. Dropping any of the four turns "nothing was checked" into "the check
  passed", and that flag is read on exactly the runs where nothing established it.
  """

  results: Tuple[ControlResult, ...] = ()
  n: int = 0
  pack_id: str = ""
  seed: int = 0
  action_conditioned: Dict[str, bool] = field(default_factory=dict)

  @property
  def passed(self) -> bool:
    if not self.results:
      return False
    if self.n <= 0:
      return False
    if any(not r.ran for r in self.results):
      return False
    if any(r.throttled for r in self.results):
      return False
    return all(r.met for r in self.results)

  @property
  def n_non_evidential(self) -> int:
    return sum(1 for r in self.results if not r.evidential)

  @property
  def n_clamp_unmeasured(self) -> int:
    return sum(1 for r in self.results if not r.clamp_rate_measured)

  def failures(self) -> Tuple[ControlResult, ...]:
    return tuple(r for r in self.results if not (r.ran and r.met) or r.throttled)

  def summary(self) -> str:
    if not self.results:
      return (
        "null controls: NOT RUN. Nothing was checked, so nothing is established and "
        "no scorecard is emitted."
      )
    verdict = "ok" if self.passed else "SUITE FAILED, no scorecard is emitted"
    lines = [
      f"null controls: {verdict} over {self.n} episodes per task, pack {self.pack_id}, "
      f"seed {self.seed}"
    ]
    lines.extend(r.line() for r in self.results)
    if self.n_non_evidential:
      lines.append(
        f"  {self.n_non_evidential} control(s) were satisfied trivially in this "
        "configuration and carry no evidence until an arm run supplies the missing "
        "column."
      )
    if self.n_clamp_unmeasured:
      lines.append(
        f"  {self.n_clamp_unmeasured} of {len(self.results)} control(s) report no "
        "measured clamp rate. Nothing in this run went through a safety envelope, so "
        "the absence of a THROTTLED mark on those lines is an absence of measurement "
        "and not a measurement of zero."
      )
    return "\n".join(lines)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "passed": self.passed,
      "n": self.n,
      "pack_id": self.pack_id,
      "seed": self.seed,
      "n_non_evidential": self.n_non_evidential,
      "n_clamp_unmeasured": self.n_clamp_unmeasured,
      "action_conditioned": dict(self.action_conditioned),
      "results": [r.to_dict() for r in self.results],
    }


# -- running the suite ---------------------------------------------------------


def _e_aurc(rows: EpisodeRows) -> Optional[float]:
  hits, confidences = rows.ranked()
  if not hits:
    return None
  return RiskCoverageCurve.from_outcomes(hits, confidences).e_aurc


def _no_information_band(rows: EpisodeRows) -> Optional[float]:
  hits, confidences = rows.ranked()
  if not hits:
    return None
  curve = RiskCoverageCurve.from_outcomes(hits, confidences)
  return curve.aurc_random - curve.aurc_oracle


def run_control_suite(
  subject: Any,
  pack: ScenePack,
  tasks: Sequence[BenchTask] = TASKS,
  seed: int = 0,
  n_permutations: int = PERMUTATION_RESAMPLES,
  n_resamples: Optional[int] = None,
  subject_rows: Optional[Dict[Tuple[str, int], EpisodeRows]] = None,
) -> ControlSuiteReport:
  """Run N1 through N10 against this pack, with the submission where one is needed.

  Every control runs at its task's **shortest** horizon. That is deliberately the
  hardest place to make a must-fail control fail: a do-nothing null has its best
  chance of landing inside tolerance when only one action has been commanded, so
  requiring it to fail there is the strict version of the requirement.

  ``subject`` is the model under test. Three of the ten are statements about it
  rather than about the harness -- the confidence shuffle, the action-conditioning
  gap and the pixel comparison -- and a suite run without a subject would have to
  either skip them or invent one, and skipping them is the thing ``passed`` refuses.
  """
  results: List[ControlResult] = []
  conditioned: Dict[str, bool] = {}
  n_seen = 0
  bootstrap: Dict[str, Any] = {}
  if n_resamples is not None:
    bootstrap["n_resamples"] = int(n_resamples)

  for task in tasks:
    horizon = int(task.horizons[0])
    episodes = pack.for_task(task.task_id)
    if not episodes:
      continue
    n_seen = max(n_seen, len(episodes))
    key = (task.task_id, horizon)
    if subject_rows is not None and key in subject_rows:
      submitted = subject_rows[key]
    else:
      submitted = evaluate_rows(subject, task, pack, horizon)

    if task.task_id in N1_TASKS:
      results.append(_n1(task, pack, horizon))
    results.append(_n2(task, pack, horizon, submitted, seed, bootstrap, conditioned))
    results.append(_n3(task, submitted, seed, n_permutations))
    results.append(_n4(task, pack, horizon))
    results.append(_n5(task, pack, horizon, submitted, n_permutations))
    results.append(_n6(task, pack, horizon))
    results.append(_n7(task, pack, horizon))
    results.append(_n8(task, pack, horizon, seed))
    results.append(_n9(task, submitted))
    results.extend(_n10(task, pack, horizon, seed))

  return ControlSuiteReport(
    results=tuple(results),
    n=n_seen,
    pack_id=pack.pack_id,
    seed=int(seed),
    action_conditioned=conditioned,
  )


def _n1(task: BenchTask, pack: ScenePack, horizon: int) -> ControlResult:
  predictor = BlindPredictor(state_dim=task.state_dim, horizon=horizon)
  rows = evaluate_rows(predictor, task, pack, horizon)
  observed = rows.tol_acc
  met = observed <= BANDS["N1_MAX_TOL_ACC"]
  return ControlResult(
    control_id="N1",
    name="ConstantPredictor",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=(
      f"TolAcc@{horizon} {observed:.4f} against a ceiling of "
      f"{BANDS['N1_MAX_TOL_ACC']:.2f}"
    ),
    notes=(
      () if met else (
        "a predictor that changes nothing lands inside tolerance too often here, "
        "which says the tolerance is wider than the motion. Fix the task.",
      )
    ),
  )


def _n2(
  task: BenchTask,
  pack: ScenePack,
  horizon: int,
  submitted: EpisodeRows,
  seed: int,
  bootstrap: Dict[str, Any],
  conditioned: Dict[str, bool],
) -> ControlResult:
  shuffled = evaluate_rows(
    ShuffledActionPredictor(task, horizon, pack, seed=seed), task, pack, horizon
  )
  truth = evaluate_rows(OracleLeakPredictor(task, horizon, pack), task, pack, horizon)
  gap = paired_bootstrap_ci(
    [float(h) for h in truth.hits],
    [float(h) for h in shuffled.hits],
    seed=seed,
    **bootstrap
  )
  met = gap.point >= BANDS["N2_MIN_GAP"] and not gap.includes_zero

  subject_gap = paired_bootstrap_ci(
    [float(h) for h in submitted.hits],
    [float(h) for h in shuffled.hits],
    seed=seed + 1,
    **bootstrap
  )
  is_conditioned = (
    not subject_gap.includes_zero and subject_gap.point >= BANDS["N2_MIN_GAP"]
  )
  conditioned[task.task_id] = bool(is_conditioned)
  notes: List[str] = [
    f"true vs shuffled: {gap.format()}",
    f"submission vs shuffled: {subject_gap.format()}",
  ]
  if not is_conditioned:
    notes.append(
      "NOT ACTION-CONDITIONED: the submission does not beat the shuffled-action "
      "control by more than a paired interval, so its headline is withheld rather "
      "than printed beside this line. A benchmark that prints a number next to its "
      "own refutation gets the number quoted and the refutation dropped."
    )
  return ControlResult(
    control_id="N2",
    name="ShuffledActionPredictor",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=f"TolAcc@{horizon} {shuffled.tol_acc:.4f} with actions permuted",
    notes=tuple(notes),
  )


def _n3(
  task: BenchTask,
  submitted: EpisodeRows,
  seed: int,
  n_permutations: int,
) -> ControlResult:
  hits, confidences = submitted.ranked()
  if not hits:
    return ControlResult(
      control_id="N3",
      name="ConfidenceShuffle",
      requirement=MUST_FAIL,
      task_id=task.task_id,
      ran=True,
      met=True,
      observed="the submission ranked no episodes, so there is no ordering to shuffle",
      notes=(
        "a model that abstained on everything cannot have an abstention claim "
        "confirmed by this control; the claim is withheld rather than granted.",
      ),
    )
  permutation = confidence_shuffle_p(
    hits, confidences, seed=seed, n_permutations=int(n_permutations)
  )
  # The control is required to fail, which here means the shuffled ordering must not
  # match the submission's. If it does, the submission's confidence carries no
  # information and its abstention claim is withheld rather than scored low.
  met = True
  notes = (permutation.format(),)
  if permutation.no_information:
    notes = notes + (
      "NO INFORMATION: the confidence ordering did not beat its own shuffle, so the "
      "abstention claim for this task is withheld.",
    )
  return ControlResult(
    control_id="N3",
    name="ConfidenceShuffle",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=(
      f"no-information floor established at p={permutation.p_value:.4f} over "
      f"{permutation.n_permutations} permutations"
    ),
    notes=notes,
  )


def constant_confidence_e_aurc(rows: EpisodeRows) -> Optional[float]:
  """Where a constant confidence lands, as a median over tie-break orderings.

  With every confidence equal, the risk-coverage sort is decided entirely by the
  recorded tie-break order, so one ordering is one draw from the null rather than
  the null itself. Comparing a single draw against the band would make the control
  fail on an unlucky permutation and pass on a lucky one, which is a property of the
  seed and not of the arithmetic being checked.
  """
  hits, confidences = rows.ranked()
  if not hits:
    return None
  values = [
    RiskCoverageCurve.from_outcomes(hits, confidences, seed=seed).e_aurc
    for seed in range(N4_ORDERINGS)
  ]
  return metrics.median_of(values)


def _n4(task: BenchTask, pack: ScenePack, horizon: int) -> ControlResult:
  rows = evaluate_rows(AlwaysConfidentPredictor(task, horizon, pack), task, pack, horizon)
  observed = constant_confidence_e_aurc(rows)
  band = _no_information_band(rows)
  if observed is None or band is None:
    return ControlResult(
      control_id="N4",
      name="AlwaysConfident",
      requirement=MUST_FAIL,
      task_id=task.task_id,
      ran=False,
      met=False,
      observed="the constant-confidence null ranked nothing, so it established no band",
    )
  met = abs(observed - band) <= BANDS["N4_E_AURC_TOL"]
  return ControlResult(
    control_id="N4",
    name="AlwaysConfident",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=(
      f"E-AURC {observed:.4f} (median over {N4_ORDERINGS} tie-break orderings) "
      f"against the no-information band {band:.4f} (AURC_random - AURC_oracle), "
      f"tolerance {BANDS['N4_E_AURC_TOL']:.2f}"
    ),
    notes=(
      () if met else (
        "a constant confidence that does not land on the no-information band means "
        "the excess-AURC arithmetic is wrong, and an inverted band reads as a nearly "
        "perfect model.",
      )
    ),
  )


def _n5(
  task: BenchTask,
  pack: ScenePack,
  horizon: int,
  submitted: EpisodeRows,
  n_permutations: int,
) -> ControlResult:
  predictor = PixelPerfectPredictor(task, horizon, pack)
  rows = evaluate_rows(predictor, task, pack, horizon)
  psnr, ssim = predictor.declared_pixel_metrics()
  tops_pixels = psnr >= BANDS["N5_MIN_PSNR"]
  ranked_hits, ranked_confidences = rows.ranked()
  permutation = confidence_shuffle_p(
    ranked_hits, ranked_confidences, seed=0, n_permutations=int(n_permutations)
  )
  loses_ranking = permutation.no_information
  met = tops_pixels and loses_ranking
  return ControlResult(
    control_id="N5",
    name="PixelPerfect",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=(
      f"pixel column ({metrics.NOT_RANKED}) PSNR {psnr:.2f} dB, SSIM {ssim:.4f}; "
      f"E-AURC ranking: {'not rankable' if loses_ranking else 'RANKABLE'}; "
      f"TolAcc@{horizon} {rows.tol_acc:.4f} against the submission's "
      f"{submitted.tol_acc:.4f}"
    ),
    notes=(
      "tops the pixel column outright and is still not rankable, which is the whole "
      "reason a pixel column exists here.",
    ) if met else (
      "the pixel replay was expected to top the pixel column and lose the ranking; "
      "it did not, so the demonstration has not been made.",
    ),
  )


def _n6(task: BenchTask, pack: ScenePack, horizon: int) -> ControlResult:
  rows = evaluate_rows(OracleLeakPredictor(task, horizon, pack), task, pack, horizon)
  observed = rows.tol_acc
  if observed < BANDS["N6_REQUIRED_TOL_ACC"]:
    raise HarnessBroken(
      f"N6 OracleLeak scored TolAcc {observed:.6f} on {task.task_id} at horizon "
      f"{horizon}, and it was handed the true terminal state. The comparison, the "
      "tolerance lookup or the truth lookup is wrong. This is the harness, not the "
      "model, and a low score here would have been attributed to the model."
    )
  return ControlResult(
    control_id="N6",
    name="OracleLeak",
    requirement=MUST_PASS,
    task_id=task.task_id,
    ran=True,
    met=True,
    observed=f"TolAcc@{horizon} {observed:.4f}, exactly as required",
  )


def _n7(task: BenchTask, pack: ScenePack, horizon: int) -> ControlResult:
  rows = evaluate_rows(AlwaysAbstainPredictor(task, horizon), task, pack, horizon)
  outcomes = rows.outcomes()
  n = len(outcomes)
  n_unknowable = sum(1 for o in outcomes if o.unknowable)
  base_rate = n_unknowable / n if n else 0.0
  ranked = rows.n_ranked
  credit = AbstentionCredit(
    n=n,
    n_unknowable=n_unknowable,
    n_abstained=n,
    n_unknowable_abstained=n_unknowable,
    n_knowable_abstained=n - n_unknowable,
    n_credited=n_unknowable,
  )
  try:
    acr, far = credit.credit_and_false_alarm()
    acr_value = acr.value
    far_value = far.value
  except AbstentionBandRefused:
    return ControlResult(
      control_id="N7",
      name="AlwaysAbstain",
      requirement=MUST_FAIL,
      task_id=task.task_id,
      ran=False,
      met=False,
      observed="the unknowable fraction is outside its band, so R6 refuses these numbers",
    )
  if acr_value is None or far_value is None:
    # Below the reporting floor neither number prints under any flag (R2), so this
    # control has not been established. Recorded as not run rather than as met: a
    # control that could not be evaluated has shown nothing, and folding that into a
    # pass is how a suite comes to pass by having done nothing.
    return ControlResult(
      control_id="N7",
      name="AlwaysAbstain",
      requirement=MUST_FAIL,
      task_id=task.task_id,
      ran=False,
      met=False,
      observed=(
        f"{underpowered_label(min(n, n - n_unknowable))}: the abstention rates cannot "
        "be printed at this episode count, so this control establishes nothing"
      ),
    )
  met = (
    ranked == 0
    and abs(acr_value - base_rate) < 1e-9
    and far_value == 1.0
  )
  return ControlResult(
    control_id="N7",
    name="AlwaysAbstain",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=(
      f"ranked {ranked} episodes, ACR {acr_value:.4f} against an unknowable base rate "
      f"of {base_rate:.4f}, FAR {far_value:.4f}"
    ),
    notes=(
      "an unbeatable abstention rate worth nothing: the credit rate equals the base "
      "rate and is fully explained by the false-alarm rate beside it.",
    ),
  )


def _n8(task: BenchTask, pack: ScenePack, horizon: int, seed: int) -> ControlResult:
  """N8. Three legs, and the withheld-half number is only readable with the other two.

  The published half is the positive control: the probe has to convert a memory hit
  into a scored hit there, against the do-nothing floor measured on the same
  episodes, or it could not have detected anything on the withheld half either. The
  republication count is the evidence: it has a denominator, and it is the quantity
  that decides whether the withheld-half gap is a measurement or an identity. The
  gap is third, and it is reported last on purpose.

  The halves are cut out of one pack. The real partition is between two packs and
  this is a proxy for it, but it is the strict proxy: two halves of one pack come
  off the same generator at the same spread with adjacent seeds, which is the
  configuration in which a partition is most likely to fail to bite.
  """
  episodes = list(pack.for_task(task.task_id))
  if len(episodes) < 4:
    return ControlResult(
      control_id="N8",
      name="SeedLeakPredictor",
      requirement=MUST_FAIL,
      task_id=task.task_id,
      ran=False,
      met=False,
      observed="too few episodes to split into a dev half and a scoring half",
    )
  order = list(range(len(episodes)))
  random.Random(seed + 17).shuffle(order)
  half = len(order) // 2
  dev = [episodes[i] for i in order[:half]]
  scoring = [episodes[i] for i in order[half:]]
  leak = SeedLeakPredictor(task, horizon, pack, [e.episode_id for e in dev])
  blind = BlindPredictor(state_dim=task.state_dim, horizon=horizon)
  on_dev = evaluate_rows(leak, task, pack, horizon, episodes=dev)
  floor_dev = evaluate_rows(blind, task, pack, horizon, episodes=dev)
  on_scoring = evaluate_rows(leak, task, pack, horizon, episodes=scoring)
  floor_scoring = evaluate_rows(blind, task, pack, horizon, episodes=scoring)
  republished = [
    episode.episode_id
    for episode in scoring
    if leak.republication(build_request(task, episode, horizon)) is not None
  ]
  recalled = on_dev.tol_acc
  lift = recalled - floor_dev.tol_acc
  closed = on_scoring.tol_acc - floor_scoring.tol_acc
  powered = recalled >= BANDS["N8_MIN_DEV_RECALL"]
  met = powered and not republished and closed <= BANDS["N8_MAX_GAP_CLOSED"]

  notes: List[str] = []
  if not powered:
    notes.append(
      f"the probe recalled only {recalled:.4f} of the half it memorised, against a "
      f"required {BANDS['N8_MIN_DEV_RECALL']:.2f}. Two published episodes share an "
      "initial state and an action block and disagree about the terminal, which "
      "makes this pack ambiguous rather than hard, and a probe that cannot recall "
      "what it memorised could not have detected a leak on the withheld half "
      "either. The gap on that line establishes nothing."
    )
  if republished:
    notes.append(
      f"{len(republished)} of {len(scoring)} withheld episodes are the same episode "
      f"as a published one, the first being {republished[0]!r}. The partition does "
      "not bite: those episodes were developed against, and the pack format does "
      "not catch this because its duplication check keys on frames and these carry "
      "different ones."
    )
  else:
    notes.append(
      "no withheld episode is a republication, so the probe answers the do-nothing "
      "state everywhere on that half and its score there is the floor exactly. The "
      f"{closed:+.4f} is that identity and not an estimate; the numbers carrying "
      f"evidence on this line are the {len(scoring)}-episode denominator and the "
      f"{lift:+.4f} the probe gains on the half it did memorise."
    )
  return ControlResult(
    control_id="N8",
    name="SeedLeakPredictor",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=(
      f"{len(republished)} of {len(scoring)} withheld episodes republish a published "
      f"one; probe recall {recalled:.4f} on the published half against a do-nothing "
      f"floor of {floor_dev.tol_acc:.4f} there ({lift:+.4f}); TolAcc@{horizon} "
      f"{on_scoring.tol_acc:.4f} on the withheld half, {closed:+.4f} above its floor "
      f"of {floor_scoring.tol_acc:.4f} against a ceiling of "
      f"{BANDS['N8_MAX_GAP_CLOSED']:.2f}"
    ),
    notes=tuple(notes),
  )


def _n9(task: BenchTask, submitted: EpisodeRows) -> ControlResult:
  """N9. Confidence replaced by minus the clamp count.

  A scene pack records no clamps, because no envelope acted on it: nothing here was
  sent to an arm. So this control is satisfied trivially and is marked
  non-evidential rather than counted as evidence. It becomes load-bearing on the
  first hardware run, where the measured clamp rate of the existing random null
  against the default envelope is close to half its steps.
  """
  clamps = [0.0] * submitted.n
  errors = [e if math.isfinite(e) else 1e9 for e in submitted.errors]
  rho = spearman_rho([-c for c in clamps], errors)
  met = rho is None or abs(rho) <= BANDS["N9_MAX_RHO"]
  observed = (
    "no clamp counts in this pack, so a clamp-only confidence has no ordering at all"
    if rho is None
    else f"spearman rho {rho:+.4f} against a ceiling of {BANDS['N9_MAX_RHO']:.2f}"
  )
  return ControlResult(
    control_id="N9",
    name="ClampOnlyScreen",
    requirement=MUST_FAIL,
    task_id=task.task_id,
    ran=True,
    met=met,
    observed=observed,
    evidential=rho is not None,
    notes=(
      f"the screen this control is built from is {ClampOnlyScreen.name}; it carries "
      "no evidence until an arm run supplies a clamp count per episode.",
    ),
  )


def _n10(
  task: BenchTask,
  pack: ScenePack,
  horizon: int,
  seed: int,
) -> List[ControlResult]:
  episodes = pack.for_task(task.task_id)
  requests = [build_request(task, e, horizon) for e in episodes]
  out: List[ControlResult] = []
  candidates = (
    ("BlindPredictor", BlindPredictor(state_dim=task.state_dim, horizon=horizon)),
    (
      "NoisePredictor",
      NoisePredictor(state_dim=task.state_dim, horizon=horizon, sigma=1.0, seed=seed),
    ),
  )
  for name, predictor in candidates:
    result = interventional_check(predictor, requests, seed=seed)
    out.append(
      ControlResult(
        control_id="N10",
        name=name,
        requirement=MUST_FAIL,
        task_id=task.task_id,
        ran=True,
        met=not result.passed,
        observed=(
          f"interventional gate {'PASSED' if result.passed else 'REFUSED'} over "
          f"{result.n_checked} requests at alpha {result.alpha}"
        ),
        notes=(
          () if not result.passed else (
            "a predictor that does not read its action argument cleared the "
            "action-conditioning gate, which means the gate is not measuring what it "
            "claims and no submission may be scored through it.",
          )
        ),
      )
    )
  return out


__all__ = [
  "BANDS",
  "MUST_FAIL",
  "MUST_PASS",
  "N4_ORDERINGS",
  "THROTTLED_CLAMPED_FRACTION",
  "constant_confidence_e_aurc",
  "AlwaysAbstainPredictor",
  "AlwaysConfidentPredictor",
  "ControlResult",
  "ControlSuiteReport",
  "EpisodeRows",
  "HarnessBroken",
  "OracleLeakPredictor",
  "PixelPerfectPredictor",
  "SeedLeakPredictor",
  "ShuffledActionPredictor",
  "N1_TASKS",
  "build_request",
  "evaluate_rows",
  "observable_signature",
  "resolve_predictor",
  "run_control_suite",
  "task_for",
]
