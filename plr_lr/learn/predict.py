"""What an outside world model has to satisfy before any number is printed about it.

:mod:`plr_lr.learn.dynamics` already carries a ``DynamicsModel`` seam and it stays
exactly as it is: a planner asks it for the pose that follows a displacement and it
answers in poses. That signature has no slot for a frame. Bridging a video model
through it would mean the frame arrives by some other route -- a closure, a global,
whatever the adapter author had to hand -- and once the frame is off the call
signature there is no observation identity and no capture time, so the question that
decides whether a bench prediction is worth anything cannot even be asked: is this
about the plate in front of the arm now, or about one from thirty seconds ago.
:class:`RolloutRequest` carries the frames, their hashes and their capture times
explicitly, which is what makes staleness checkable rather than assumed.

Four refusals live here, and each one exists because the permissive version of it
has already been observed to produce a confident wrong number:

* **A checkpoint is not an adapter.** :class:`InjectedPredictor` raises
  :class:`NotImplementedError` until a callable is injected. Naming a checkpoint or
  pip-installing a package establishes no state, action or unit semantics, and an
  adapter that quietly returns zeros looks exactly like one that works.
* **Abstaining is not free, and neither is abstaining while banking a prediction.**
  ``abstain`` is True if and only if the predicted state is all-NaN *and* the
  confidence is NaN, checked row for row in :func:`validate`. Refusing and flagging
  stay separate acts, because collapsing them is what turns a refusal into missing
  data; but a row that abstains while still carrying a usable prediction can be
  scored as a refusal when it was right and as a prediction when it was right, which
  is a free lunch and is closed here rather than argued about later.
* **A model with no declared training set cannot be scored.** Absent
  ``training_fingerprints`` withholds the row with ``PROVENANCE: UNDECLARED``, and a
  fitted readout probe with an undeclared split withholds it too. A probe fit on the
  scoring episodes is part of the model whether or not its author thinks of it that
  way, and the leak check that only inspects the predictor cannot see it.
* **A model that ignores its action argument is refused, not ranked.**
  :func:`interventional_check` pushes the commanded actions a full envelope step
  along each perturbable column in turn and requires the terminal prediction to move
  along at least one of them, on a majority of the requests it checks. A low score
  still gets quoted; a refusal does not. This is the whole difference between a world
  model and a video prior that extrapolates the scene whatever you asked for.

  Both quantifiers in that sentence were once wrong at the same time, and the gate
  decided submissions by CLI default because of it. It drew one isotropic direction
  per request and demanded that every request clear the bar. A task responds on the
  axes it responds on: S5 is a delta-z task, so a direction drawn near the lateral
  plane moved its terminal state by 0.0106 tolerances against a 0.2500 bar, and one
  such draw in sixteen refused the declared floor predictor. Gate seeds 0, 1, 3 and 4
  passed it and seeds 2 and 5 refused it, and since a refusal withholds the whole
  scorecard under rule R4, the same model on the same pack scored every row or no
  rows depending on a default nobody set deliberately. The repair is in what the gate
  measures, not in the bar: :data:`INTERVENTION_ALPHA` and :data:`INTERVENTION_K` are
  the numbers they always were.

``isinstance(adapter, RolloutPredictor)`` is not evidence that the adapter can
predict. The Protocol is structural, so a bare :class:`InjectedPredictor` with no
callable injected satisfies it and then raises on the first call. That is stated
here, asserted in the tests, and is the reason no code path in this package treats
an isinstance check as a capability check.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, replace
from numbers import Integral, Real
from typing import Any, FrozenSet, List, Optional, Protocol, Sequence, Tuple
from typing import runtime_checkable

import numpy as np

from .dynamics import ResidualDynamics
from .policy import AdapterOutputError
from .rollout import SafetyEnvelope
from .types import ACTION_FIELDS

#: Terminal-state movement, in tolerance units, demanded per envelope step of
#: commanded difference between two action sequences. An absolute constant: it is
#: never fitted, never read from a submission, and never derived from the data being
#: screened. A threshold tuned on the models being tested measures the tuner.
#:
#: It has not moved while the gate around it was repaired, and it must not: the way
#: to make a seed-unstable gate stable is to fix what it measures, never to lower the
#: bar until the seed that refused stops refusing. That second move fits a published
#: constant to the one submission it was about to reject.
INTERVENTION_ALPHA = 0.25

#: How many separate requests the interventional gate has to clear. Also absolute.
#: One request can respond by accident; a majority of sixteen cannot.
INTERVENTION_K = 16

#: The commanded interventions the gate pushes along, one per perturbable action
#: column, and the reason this gate is not decided by a random draw.
#:
#: The measured defect these replace: the gate drew a single **isotropic** 3-D
#: direction per request and demanded a response along it. A task responds on the
#: axes it physically responds on, not on the sphere. Measured on the built-in
#: 200-episode pack with each task's own declared floor predictor, pushing a full
#: envelope step along each column separately and reading terminal divergence in
#: tolerance units, worst episode of 60:
#:
#: ===== ======== ======== ======== ================
#: task  p0       p1       p2       max over the three
#: ===== ======== ======== ======== ================
#: S1    11.7500  11.7500  11.7500  11.7500
#: S2     2.5902   2.5817   2.5843   2.5902
#: S3    36.0803  12.4916  12.0756  36.0803
#: S4    14.3500  14.3500   0.3500  14.3500
#: S5     0.1536   0.1536   2.8333   2.8333
#: ===== ======== ======== ======== ================
#:
#: The bar at horizon 1 is 0.2500. S5 is a delta-z task: it answers on ``p2`` at
#: eleven times the bar and on the two lateral columns at six tenths of it, so an
#: isotropic direction that happened to land near the lateral plane produced a
#: divergence of 0.0106 through no fault of the model and refused the whole
#: submission. S4 is the mirror image and was one unlucky draw from the same
#: outcome. Probing a complete basis of the perturbable columns and asking whether
#: the prediction moved along **any** of them asks the question the gate is actually
#: for, which is whether the adapter reads its action argument at all, and it asks it
#: without a fixture author having to declare per task which axis is the live one:
#: a declared axis list would put the bar a model must clear under the control of
#: whoever writes the fixtures.
INTERVENTION_AXES: Tuple[Tuple[float, float, float], ...] = (
  (1.0, 0.0, 0.0),
  (0.0, 1.0, 0.0),
  (0.0, 0.0, 1.0),
)

#: The fraction of the k requests that has to respond. Absolute, and it is one half
#: rather than one because the verdict must not be decided by any single request in
#: either direction: a strict majority is nine of sixteen, so a model that reads its
#: actions can lose seven episodes to saturation and still be scored, and a model
#: that does not read them is nine short rather than one.
#:
#: The alternative aggregations were weighed. Averaging the divergences over the k
#: draws is refused for the same reason the mean of TPE is refused: the quantity is
#: heavy-tailed, and one enormous response on one episode carries a model that is
#: inert on the rest. A p-value over the draws is refused because it would put a
#: continuous figure of merit on :class:`InterventionResult`, somebody would sort a
#: table by it, and a model that ignores its actions would then appear in that table
#: at all. Counting decisions rather than magnitudes keeps the gate a floor. The
#: statistically clean sibling with no constant to choose is the paired
#: shuffled-action control, which is meant to run alongside this and not instead
#: of it.
INTERVENTION_QUORUM = 0.5

#: What the one seeded probe is called on a printed row. It exists to catch a
#: response that lives on a diagonal and cancels on each column separately; because
#: the verdict is the largest response over all probes, this one can only ever help a
#: model pass, so no seed can turn a pass into a refusal through it.
RANDOM_PROBE = "random"

#: The commanded difference the gate uses, read from the package default envelope
#: rather than from a caller-supplied one. A caller who could pass a smaller
#: envelope could shrink the bar its own model has to clear.
ENVELOPE_STEP_MM = SafetyEnvelope().max_step_mm

#: Shared with :class:`plr_lr.learn.dynamics.ResidualDynamics` on purpose, so the
#: two adapters cannot drift into disagreeing about what counts as absurd.
PLAUSIBLE_MM = ResidualDynamics.PLAUSIBLE_MM

#: Where a frame's timestamp came from. Only ``device_30000`` carries a clock the
#: arm stamped itself; the other two are the host noticing a frame, which is a
#: different measurement wearing the same units.
CLOCK_SOURCES: Tuple[str, ...] = ("device_30000", "host_arrival", "rendered")

#: How the five action columns are meant to be read.
ACTION_SPACES: Tuple[str, ...] = ("ee_delta_mm", "joint_rad", "task_scalar")

#: Splits a fitted component may honestly name. Anything else, including the
#: scoring split, is treated as undeclared.
FIT_SPLITS: Tuple[str, ...] = ("dev", "calibration")

#: The literal a component uses to say it will not name its split.
UNDECLARED = "UNDECLARED"

#: What an analytic null declares instead of a training set. Declaring that nothing
#: was fit is a declaration; having no ``training_fingerprints`` attribute at all is
#: an absence, and the two are not the same answer.
NO_TRAINING_DATA = "none:analytic-null-fit-on-no-data"

#: Quoted back in every output-contract refusal so the adapter author is told the
#: shape that was wanted rather than only the shape that arrived.
PREDICT_SIGNATURE = (
  "predict(request: RolloutRequest) -> RolloutPrediction with state of shape "
  "(request.horizon, len(request.state)) in the units of request.state_units, "
  "confidence in [0, 1], and abstain True exactly when state is all-NaN and "
  "confidence is NaN"
)


class WithheldRow(RuntimeError):
  """Base for the refusals that withhold a score rather than break a call.

  A withheld row is not a zero and not a failure of the model to predict well. It
  says the harness was not given enough to be allowed to publish anything about
  this adapter, which is a different sentence and prints differently.
  """


class ProvenanceUndeclared(WithheldRow):
  """Raised when an adapter offers no training fingerprints. Rule R7.

  The permissive alternative is the measured one: a leak check written as
  ``getattr(model, "trained_on", None)`` returns ``None`` for every adapter that
  does not happen to use this package's own fitting code, so an injected model
  scored on the exact episodes it was fit to reports no leak at all and the summary
  carries no warning line. Making the declaration mandatory turns a silent absence
  into a refusal.
  """


class UndeclaredComponent(WithheldRow):
  """Raised when a fitted component will not name the split it was fit on. Rule R8.

  A latent-to-scalar probe is part of the model. The predictor's own fingerprints do
  not cover it, so a probe fit on the scoring episodes rides in behind a clean
  provenance declaration unless it is declared separately.
  """


class InterventionFailed(RuntimeError):
  """Raised when a predictor's output does not depend on the actions it was given.

  Deliberately an exception rather than a score. A low score still gets ranked, and
  a ranked non-world-model is exactly the claim-versus-evidence mismatch this
  contract exists to prevent: the failure mode in the literature is a model that
  ignores its conditioning entirely while reporting a headline above ninety percent.
  """


class NotAWorldModel(TypeError):
  """Raised on construction of an adapter for a model that emits actions.

  A taxonomy written in prose gets read once. A constructor error gets hit every
  time somebody tries to enter a policy in a world-model leaderboard, which is the
  only moment at which the distinction is load-bearing.
  """


@dataclass(frozen=True)
class FrameRef:
  """One observed frame, identified well enough to argue about later.

  ``sha256`` is what makes "did you run on our fixtures" mechanically checkable
  without hosting the fixtures, and ``t_capture_s`` with ``clock_source`` is what
  makes a staleness bound and a latency claim checkable separately. A frame with a
  host-arrival stamp cannot support a per-action millisecond claim, and recording
  the clock source is what lets that be refused instead of assumed.
  """

  path: str
  sha256: str
  t_capture_s: float
  clock_source: str

  def __post_init__(self) -> None:
    if not isinstance(self.path, str) or not self.path:
      raise ValueError("FrameRef.path must be a non-empty string")
    if not isinstance(self.sha256, str) or not self.sha256:
      raise ValueError(
        "FrameRef.sha256 must be a non-empty string. An unhashed frame cannot be "
        "matched against a fixture manifest, so a submission carrying one can never "
        "be shown to have run on the data it claims."
      )
    if isinstance(self.t_capture_s, bool) or not isinstance(self.t_capture_s, Real):
      raise ValueError("FrameRef.t_capture_s must be a number of seconds")
    if not np.isfinite(float(self.t_capture_s)):
      raise ValueError("FrameRef.t_capture_s must be finite")
    if self.clock_source not in CLOCK_SOURCES:
      raise ValueError(
        f"FrameRef.clock_source must be one of {list(CLOCK_SOURCES)}, got "
        f"{self.clock_source!r}. An unnamed clock reads like a device clock in a "
        "table and is usually a host arrival time, which is a different measurement."
      )


@dataclass(frozen=True)
class FittedComponent:
  """A part of an adapter that was fit to data, declared by the submitter.

  Readout probes are the usual case. They are small, they are easy to forget, and a
  probe fit on the episodes being scored is a leak that the predictor's own
  fingerprints say nothing about.
  """

  name: str
  fit_on_split: str
  n_examples: int
  description: str


@dataclass(frozen=True)
class RolloutRequest:
  """Everything a predictor is allowed to condition on, with nothing left implicit.

  The frames are on the call signature rather than captured in a closure, so the
  prediction can be tied to the observation it was made from. ``tolerance`` travels
  with the request because every downstream comparison is per-field error divided by
  that field's tolerance: without it a millimetre and a microlitre end up added
  together and the resulting number is labelled in whichever unit came first.
  """

  task_id: str
  episode_id: str
  state: np.ndarray
  state_fields: Tuple[str, ...]
  state_units: Tuple[str, ...]
  frames: Tuple[FrameRef, ...]
  actions: np.ndarray
  horizon: int
  tolerance: np.ndarray
  instruction: Optional[str] = None

  def __post_init__(self) -> None:
    """Refuse a request that cannot be scored, at the point it is built.

    Every check here fails loudly now instead of producing a number later. A zero
    tolerance in particular divides every error by zero and reports either a perfect
    score or a NaN depending on the rounding, and both read as a result.
    """
    for name in ("task_id", "episode_id"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value:
        raise ValueError(f"RolloutRequest.{name} must be a non-empty string")

    state = _as_float_array("RolloutRequest.state", self.state)
    if state.ndim != 1 or state.size < 1:
      raise ValueError(
        f"RolloutRequest.state must be a 1-D vector of at least one field, got shape "
        f"{state.shape}"
      )
    if not np.all(np.isfinite(state)):
      raise ValueError("RolloutRequest.state must be finite")
    object.__setattr__(self, "state", state)

    dim = int(state.size)
    for name in ("state_fields", "state_units"):
      names = tuple(getattr(self, name))
      if len(names) != dim:
        raise ValueError(
          f"RolloutRequest.{name} has {len(names)} entries for a {dim}-field state. "
          "A state vector whose field order is implicit is the fastest way to score "
          "a model against scrambled columns."
        )
      object.__setattr__(self, name, names)

    frames = tuple(self.frames)
    for frame in frames:
      if not isinstance(frame, FrameRef):
        raise ValueError("RolloutRequest.frames must contain FrameRef entries only")
    object.__setattr__(self, "frames", frames)

    actions = _as_float_array("RolloutRequest.actions", self.actions)
    width = len(ACTION_FIELDS)
    if actions.ndim != 2 or actions.shape[1] != width:
      raise ValueError(
        f"RolloutRequest.actions must have shape (K, {width}) in the "
        f"{list(ACTION_FIELDS)} layout, got shape {actions.shape}"
      )
    if not np.all(np.isfinite(actions)):
      raise ValueError("RolloutRequest.actions must be finite")
    object.__setattr__(self, "actions", actions)

    if isinstance(self.horizon, bool) or not isinstance(self.horizon, Integral):
      raise ValueError(f"RolloutRequest.horizon must be an integer, got {self.horizon!r}")
    if int(self.horizon) != actions.shape[0] or int(self.horizon) < 1:
      raise ValueError(
        f"RolloutRequest.horizon is {self.horizon} for {actions.shape[0]} commanded "
        "actions. The horizon is the number of actions, not a separate claim about it."
      )
    object.__setattr__(self, "horizon", int(self.horizon))

    tolerance = _as_float_array("RolloutRequest.tolerance", self.tolerance)
    if tolerance.shape != (dim,):
      raise ValueError(
        f"RolloutRequest.tolerance must have one entry per state field, expected "
        f"({dim},) and got {tolerance.shape}"
      )
    if not np.all(np.isfinite(tolerance)) or np.any(tolerance <= 0.0):
      raise ValueError(
        "RolloutRequest.tolerance must be finite and strictly positive. Every metric "
        "downstream divides by it, so a zero entry reports a perfect score or a NaN "
        "depending on rounding, and both of those read as a result."
      )
    object.__setattr__(self, "tolerance", tolerance)

    if self.instruction is not None and not isinstance(self.instruction, str):
      raise ValueError("RolloutRequest.instruction must be a string or None")

  def newest_frame(self) -> Optional[FrameRef]:
    """The most recently captured frame, or None when the request carries no frames.

    Frames arrive oldest first, but the staleness bound is about the newest one, so
    the ordering convention is resolved here rather than at each call site.
    """
    if not self.frames:
      return None
    return max(self.frames, key=lambda f: float(f.t_capture_s))


@dataclass(frozen=True)
class RolloutPrediction:
  """What an adapter returns, before :func:`validate` has looked at it.

  Deliberately holds no validation of its own. An adapter has to be able to build a
  wrong prediction, or the tests that prove the wrong ones are caught would be
  testing the constructor instead of the contract.

  ``abstain`` and ``flag`` are separate booleans because refusing and warning are
  separate acts. ``flag`` with a finite prediction is legal and is scored as an act.
  ``abstain`` with a finite prediction is not legal, and :func:`validate` is where
  that is enforced.
  """

  state: np.ndarray
  confidence: float
  abstain: bool
  flag: bool
  readout: str
  n_forward_passes: int
  wall_clock_s: float

  @classmethod
  def abstained(
    cls,
    horizon: int,
    state_dim: int,
    readout: str = "abstained",
    n_forward_passes: int = 0,
    wall_clock_s: float = 0.0,
  ) -> "RolloutPrediction":
    """Build the one abstention shape the row invariant accepts.

    Hand-rolling an abstention is where the invariant gets broken: an all-zero state
    with ``abstain=True`` is the common mistake and it is indistinguishable from a
    confident prediction of the origin. Building it here means an adapter that
    wants to refuse cannot accidentally bank a prediction while refusing.
    """
    if int(horizon) < 1 or int(state_dim) < 1:
      raise ValueError("horizon and state_dim must both be at least 1")
    state = np.full((int(horizon), int(state_dim)), np.nan, dtype=float)
    return cls(
      state=state,
      confidence=float("nan"),
      abstain=True,
      flag=False,
      readout=readout,
      n_forward_passes=int(n_forward_passes),
      wall_clock_s=float(wall_clock_s),
    )


@runtime_checkable
class RolloutPredictor(Protocol):
  """A model that predicts task state forward from frames and commanded actions.

  Satisfying this Protocol is a structural fact about attribute names and says
  nothing about whether the adapter can predict: a bare
  :class:`InjectedPredictor` with no callable injected passes ``isinstance`` and
  then raises :class:`NotImplementedError` on the first call. Nothing in this
  package treats an isinstance check as evidence of capability, and the test suite
  asserts both halves of that sentence.

  ``license_text`` is read from the downloaded artifact, not from an announcement.
  At least one model in the current inventory ships a model card and an announcement
  that name different licenses, and the receipt has to carry whichever one actually
  travelled with the weights.
  """

  name: str
  state_dim: int
  horizon: int
  action_space: str
  training_fingerprints: FrozenSet[str]
  fitted_components: Tuple[FittedComponent, ...]
  license_text: str

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    ...


def _as_float_array(label: str, value: Any) -> np.ndarray:
  try:
    return np.asarray(value, dtype=float)
  except (TypeError, ValueError) as exc:
    raise ValueError(f"{label} must be numeric") from exc


def _model_name(predictor: Optional[Any]) -> str:
  """Name the thing being refused, so an error in a batch of adapters is placeable."""
  if predictor is None:
    return "rollout predictor"
  declared = getattr(predictor, "name", None)
  if isinstance(declared, str) and declared:
    return declared
  return type(predictor).__name__


def _fail(model: str, detail: str) -> AdapterOutputError:
  return AdapterOutputError(f"{model} adapter: {detail} Expected {PREDICT_SIGNATURE}.")


def _required_responses(k: int) -> int:
  """How many of ``k`` requests have to respond: a strict majority, as a count.

  Written as a floor plus one rather than a rounded half so that the answer is a
  strict majority at every ``k`` including the odd ones, and so that the smallest
  legal gate, ``k=1``, still demands its single request.
  """
  return int(INTERVENTION_QUORUM * int(k)) + 1


def _unit_direction(rng: Any) -> np.ndarray:
  """One drawn direction of unit length, with the degenerate draw handled.

  A zero draw is astronomically unlikely and silently catastrophic: normalising it
  produces NaN, every divergence computed against it is NaN, and the request reads as
  an abstention by a model that did not abstain.
  """
  drawn = rng.normal(size=3)
  norm = float(np.linalg.norm(drawn))
  if not norm > 0.0:
    return np.array([1.0, 0.0, 0.0])
  return drawn / norm


def _mm_columns(request: RolloutRequest) -> Tuple[int, ...]:
  """State columns measured in millimetres, by declared unit or by field name."""
  columns: List[int] = []
  for index, (name, unit) in enumerate(zip(request.state_fields, request.state_units)):
    if unit == "mm" or name in ("x", "y", "z"):
      columns.append(index)
  return tuple(columns)


def _check_awaitable(raw: Any, model: str) -> None:
  """Refuse an async predictor, and close the coroutine before doing so.

  Closing matters. An un-awaited coroutine that is simply dropped emits a
  ``RuntimeWarning`` from somewhere unrelated later in the run, which reads as a
  flaky harness rather than as the adapter contract being broken here.
  """
  if not inspect.isawaitable(raw):
    return
  close = getattr(raw, "close", None)
  if callable(close):
    close()
  raise AdapterOutputError(
    f"{model} adapter: predict returned an awaitable. Prediction is synchronous "
    "here; whatever owns the event loop owns the async boundary, because a "
    "predictor awaited inside a scoring loop cannot be timed or ordered. Expected "
    f"{PREDICT_SIGNATURE}."
  )


def _validate_output(
  pred: Any,
  request: RolloutRequest,
  predictor: Optional[Any] = None,
) -> RolloutPrediction:
  """The output half of the contract: shape, finiteness, and the row invariant.

  Split out from :func:`validate` because the interventional gate has to call a
  predictor twice before any score exists, and a provenance refusal at that point
  would be reported as a failure to condition on actions. Two different refusals
  should not be able to wear each other's message.
  """
  model = _model_name(predictor)
  _check_awaitable(pred, model)

  if not isinstance(pred, RolloutPrediction):
    raise _fail(
      model,
      f"predict returned {type(pred).__name__}, not a RolloutPrediction. A tuple or "
      "dict is read positionally by whoever unpacks it next, and a swapped "
      "confidence and abstain flag is not detectable after that point.",
    )

  for name in ("abstain", "flag"):
    value = getattr(pred, name)
    if not isinstance(value, (bool, np.bool_)):
      raise _fail(
        model,
        f"{name} is {value!r}, which is not a bool. A truthy non-boolean reads as "
        "True in one place and compares unequal to True in another.",
      )
  abstain = bool(pred.abstain)
  flagged = bool(pred.flag)

  if abstain and flagged:
    raise _fail(
      model,
      "abstain and flag are both True. Refusing to answer and answering with a "
      "warning are different acts; a row that does both is scored as whichever one "
      "turns out to have been right.",
    )

  if not isinstance(pred.readout, str) or not pred.readout:
    raise _fail(
      model,
      f"readout is {pred.readout!r}. Every row records which readout produced it, "
      "because two submitters with the same weights and different readouts are not "
      "comparable and an empty string hides that.",
    )

  if isinstance(pred.n_forward_passes, bool) or not isinstance(pred.n_forward_passes, Integral):
    raise _fail(model, f"n_forward_passes is {pred.n_forward_passes!r}, not an integer.")
  if int(pred.n_forward_passes) < 0:
    raise _fail(model, f"n_forward_passes is {pred.n_forward_passes!r}, which is negative.")

  if isinstance(pred.wall_clock_s, bool) or not isinstance(pred.wall_clock_s, Real):
    raise _fail(model, f"wall_clock_s is {pred.wall_clock_s!r}, not a number of seconds.")
  wall_clock = float(pred.wall_clock_s)
  if not np.isfinite(wall_clock) or wall_clock < 0.0:
    raise _fail(
      model,
      f"wall_clock_s is {pred.wall_clock_s!r}. Declared compute prints on every row "
      "and is never scored, but a negative or non-finite one is not a measurement.",
    )

  try:
    state = np.asarray(pred.state, dtype=float)
  except (TypeError, ValueError) as exc:
    raise _fail(model, "state is not a numeric array.") from exc

  horizon = int(request.horizon)
  dim = int(request.state.size)
  if state.shape != (horizon, dim):
    raise _fail(
      model,
      f"state has shape {state.shape}; this request wants exactly {(horizon, dim)}.",
    )
  if predictor is not None:
    declared_dim = getattr(predictor, "state_dim", None)
    if not isinstance(declared_dim, Integral) or isinstance(declared_dim, bool):
      raise _fail(model, f"state_dim is {declared_dim!r}, not an integer.")
    if int(declared_dim) != dim:
      raise _fail(
        model,
        f"declares state_dim={int(declared_dim)} but this request carries {dim} "
        "state fields. A predictor whose declared width disagrees with the task is "
        "reading some other task's columns.",
      )

  if isinstance(pred.confidence, (bool, np.bool_)) or not isinstance(pred.confidence, Real):
    raise _fail(model, f"confidence is {pred.confidence!r}, not a number in [0, 1].")
  confidence = float(pred.confidence)

  nan_rows = np.isnan(state).all(axis=1)
  finite_rows = np.isfinite(state).all(axis=1)
  if abstain:
    if not bool(nan_rows.all()):
      offenders = np.where(~nan_rows)[0].tolist()
      raise _fail(
        model,
        f"abstain is True but rows {offenders} of state are not all-NaN. Abstaining "
        "while banking a usable prediction is scored as a refusal when the refusal "
        "was right and as a prediction when the prediction was right, which is the "
        "one free lunch this contract exists to close.",
      )
    if not np.isnan(confidence):
      raise _fail(
        model,
        f"abstain is True but confidence is {confidence!r}. An abstention carries no "
        "confidence; a numeric one is ranked against acting rows in the "
        "risk-coverage sort.",
      )
  else:
    if not bool(finite_rows.all()):
      offenders = np.where(~finite_rows)[0].tolist()
      raise _fail(
        model,
        f"abstain is False but rows {offenders} of state are not finite. A partly "
        "NaN prediction is neither a refusal nor an answer, and every metric here "
        "silently drops it.",
      )
    if np.isnan(confidence):
      raise _fail(
        model,
        "abstain is False but confidence is NaN. The row invariant is that abstain "
        "is True exactly when the state is all-NaN and the confidence is NaN.",
      )
    if not (0.0 <= confidence <= 1.0):
      raise _fail(
        model,
        f"confidence is {confidence!r}, outside [0, 1]. Confidences are sorted "
        "against each other across models, so an unbounded score changes another "
        "model's ranking rather than its own.",
      )
    columns = _mm_columns(request)
    if columns and np.any(np.abs(state[:, list(columns)]) > PLAUSIBLE_MM):
      raise FloatingPointError(
        f"{model} adapter: predicted a position more than {PLAUSIBLE_MM:.0f} mm from "
        "the origin. A diverging model passes through large finite nonsense long "
        "before it reaches inf, and every consumer of this row would act on each "
        f"step of the way. Expected {PREDICT_SIGNATURE}."
      )

  return replace(pred, state=state, confidence=confidence, wall_clock_s=wall_clock)


def _check_provenance(request: RolloutRequest, predictor: Optional[Any]) -> None:
  """Withhold the row unless the adapter has declared what it was fit on.

  Rules R7 and R8. Passing no predictor at all is itself an undeclared provenance:
  the caller has produced a prediction with nothing attached that could be checked
  for overlap with the scoring episodes, and the conservative answer to that is the
  same as for an adapter that declares nothing.
  """
  model = _model_name(predictor)
  has_predictor = predictor is not None
  fingerprints = getattr(predictor, "training_fingerprints", None) if has_predictor else None
  if fingerprints is None or not isinstance(fingerprints, (frozenset, set)) or not fingerprints:
    raise ProvenanceUndeclared(
      f"PROVENANCE: UNDECLARED for {model} on {request.task_id}/{request.episode_id}. "
      "Score withheld (R7). Declare training_fingerprints as a non-empty frozenset "
      "of strings identifying what this model was fit on; an analytic model that was "
      f"fit on nothing declares {NO_TRAINING_DATA!r}, which is a declaration and not "
      "an absence. A leak check that reads an attribute nothing sets reports no leak "
      "for every model that did not use this package's own fitting code."
    )
  for entry in fingerprints:
    if not isinstance(entry, str) or not entry or entry == UNDECLARED:
      raise ProvenanceUndeclared(
        f"PROVENANCE: UNDECLARED for {model}: training_fingerprints contains "
        f"{entry!r}. Score withheld (R7). Every entry names a corpus, a split or a "
        "seed range; a blank or a literal UNDECLARED is an absence wearing the shape "
        "of a declaration."
      )

  components = getattr(predictor, "fitted_components", ()) if has_predictor else ()
  try:
    components = tuple(components)
  except TypeError as exc:
    raise UndeclaredComponent(
      f"{model}: fitted_components is not iterable. Score withheld (R8)."
    ) from exc
  for component in components:
    if not isinstance(component, FittedComponent):
      raise UndeclaredComponent(
        f"{model}: fitted_components contains {type(component).__name__}, which "
        "cannot name a split. Score withheld (R8)."
      )
    if component.fit_on_split not in FIT_SPLITS:
      raise UndeclaredComponent(
        f"{model}: fitted component {component.name!r} declares fit_on_split="
        f"{component.fit_on_split!r}. Score withheld (R8). Allowed values are "
        f"{list(FIT_SPLITS)}. A readout probe is part of the model, the predictor's "
        "own fingerprints say nothing about it, and a probe fit on the scoring "
        "episodes rides in behind a clean provenance declaration otherwise."
      )
    if isinstance(component.n_examples, bool) or not isinstance(component.n_examples, Integral):
      raise UndeclaredComponent(
        f"{model}: fitted component {component.name!r} has n_examples="
        f"{component.n_examples!r}, which is not an integer. Score withheld (R8)."
      )
    if int(component.n_examples) < 1:
      raise UndeclaredComponent(
        f"{model}: fitted component {component.name!r} declares n_examples="
        f"{int(component.n_examples)}. Score withheld (R8). A component fit on no "
        "examples was not fit; if it is a fixed transform, it is not a fitted "
        "component and does not belong in this tuple."
      )


def validate(
  pred: Any,
  request: RolloutRequest,
  predictor: Optional[Any] = None,
) -> RolloutPrediction:
  """Check one prediction against the contract and return it in canonical form.

  ``pred`` is whatever the adapter returned, including an awaitable or a dict.
  ``predictor`` is the adapter that returned it, and it is optional only so that the
  refusal for omitting it can be the same one as for an adapter that declares no
  provenance: without it there is nothing to check the scoring episodes against, so
  the row is withheld rather than scored.

  The output-contract guards run first and the provenance guards last, so an adapter
  with a broken shape hears about the shape rather than about its paperwork.

  A predictor's declared ``horizon`` is deliberately not checked here. Asking a
  model to predict further than it declares is a screening decision with its own
  reason code, and duplicating it as an output-contract error would let the two
  drift apart.
  """
  checked = _validate_output(pred, request, predictor)
  _check_provenance(request, predictor)
  return checked


@dataclass(frozen=True)
class InterventionResult:
  """What the action-conditioning gate observed. Not a score, and not rankable.

  Carries no aggregate figure of merit on purpose. The moment this produces a number
  between zero and one, somebody sorts a table by it and a model that ignores its
  actions appears in that table at all, which is the outcome the gate exists to make
  impossible. ``passed`` is the only thing to read; everything else is diagnosis.
  """

  model: str
  passed: bool
  n_checked: int
  n_abstained: int
  alpha: float
  k: int
  seed: int
  step_bound_mm: float
  divergence: Tuple[float, ...] = ()
  required: Tuple[float, ...] = ()
  failed: Tuple[int, ...] = ()
  notes: Tuple[str, ...] = ()
  n_responded: int = 0
  n_required: int = 0
  probes: Tuple[str, ...] = ()
  probe_divergence: Tuple[Tuple[float, ...], ...] = ()

  def summary(self) -> str:
    verdict = "PASSED" if self.passed else "REFUSED"
    finite = [d for d in self.divergence if np.isfinite(d)]
    smallest = f"{min(finite):.4f}" if finite else "none finite"
    bar = f"{max(self.required):.4f}" if self.required else "nothing measured"
    lines = [
      f"interventional gate: {verdict} ({self.model})",
      f"  {self.n_checked} requests checked at alpha {self.alpha}, "
      f"step bound {self.step_bound_mm:.2f} mm, seed {self.seed}",
      f"  probes per request: {list(self.probes)}, response is the largest of them",
      f"  terminal divergence in tolerance units: smallest {smallest}, "
      f"required more than {bar}",
      f"  {self.n_responded} of {self.n_checked} requests responded, "
      f"{self.n_required} required; {len(self.failed)} did not, "
      f"{self.n_abstained} abstained",
    ]
    weakest = self.weakest_probe()
    if weakest is not None:
      name, value = weakest
      lines.append(
        f"  weakest single probe over every request: {name} at {value:.4f}, which is "
        "what a single-direction gate would have been decided by"
      )
    lines.extend("  note: " + note for note in self.notes)
    return "\n".join(lines)

  def weakest_probe(self) -> Optional[Tuple[str, float]]:
    """The smallest response any one probe drew, named. Diagnosis, never a score.

    This is the number whose absence let the seed-instability ship: a gate that reads
    one direction per request is decided by exactly this quantity, and printing it
    beside the verdict is what makes visible that a task answers on one axis and not
    on the sphere. It is deliberately not part of ``passed``.
    """
    best: Optional[Tuple[str, float]] = None
    for row in self.probe_divergence:
      for name, value in zip(self.probes, row):
        if not np.isfinite(value):
          continue
        if best is None or float(value) < best[1]:
          best = (name, float(value))
    return best

  def raise_for_refusal(self) -> None:
    """Turn a failed gate into the refusal it is. Raises :class:`InterventionFailed`."""
    if self.passed:
      return
    raise InterventionFailed(
      "action-conditioning gate refused this adapter, so no score is emitted for it "
      "(R4). A model whose terminal prediction does not move when the commanded "
      "actions move is not predicting the effect of an action; scoring it anyway "
      "publishes a world-model number for something that is not one.\n"
      + self.summary()
    )


def interventional_check(
  p: Any,
  requests: Sequence[RolloutRequest],
  alpha: float = INTERVENTION_ALPHA,
  k: int = INTERVENTION_K,
  seed: int = 0,
) -> InterventionResult:
  """Does this predictor's output actually depend on the actions it was handed?

  For each of ``k`` requests, predict the commanded action sequence, then predict it
  again once per probe with that probe's direction added at a full envelope step on
  every step. A request **responds** when the terminal state moved by more than
  ``alpha`` tolerances per envelope step of commanded difference along **at least
  one** probe. The gate passes when a strict majority of the ``k`` requests
  responded.

  Both sides of that comparison are made dimensionless before they meet. The state
  divergence is divided by the request's own per-field tolerance and the commanded
  divergence by the envelope step bound, because a state field may be microlitres
  while the action columns are millimetres, and a raw comparison of the two is a
  unit error that produces a plausible-looking pass.

  Two quantifiers, and getting either one wrong is what made this gate decide
  submissions by CLI default.

  * **Over directions the quantifier is "some", not "every".** The claim being
    tested is that the adapter reads its action argument, and that claim is
    existential: a model reads its actions if there is a commanded change that moves
    its prediction. Demanding a response along every direction tested instead asks
    whether the *task* responds isotropically, which no bench task does. S5 answers
    on delta z and barely on the two lateral columns; the earlier gate drew one
    isotropic direction per request, and on gate seeds 2 and 5 it drew one close
    enough to the lateral plane to read 0.0106 and 0.0379 against a 0.2500 bar and
    refuse the declared floor predictor outright. Under rule R4 a refusal withholds
    the entire scorecard, so ``--gate-seed 2`` scored zero rows and ``--gate-seed 3``
    scored all of them, for the same model on the same pack.
  * **Over requests the quantifier is a majority, not "every".** A single request is
    allowed to be uninformative. Some episode of some task will sit against a
    saturation of the model's own readout, and one such episode vetoing a submission
    is the same single-draw defect one layer up. The nulls make the margin plain
    rather than marginal: they respond on none of the sixteen and need nine.

  :data:`INTERVENTION_AXES` is a complete basis of the three perturbable columns, so
  no linear dependence on the commanded action can hide from it, and one seeded
  random direction is appended per request to cover a response that lives on a
  diagonal and cancels on each axis alone. Adding probes can only raise a response,
  never lower one, so the seed can no longer turn a pass into a refusal: it chooses
  which requests are drawn and one extra probe that is incapable of causing a
  refusal on its own.

  The bar is a floor, not a fidelity measure, and it is set low on purpose: it
  separates a model that reads its action argument from one that does not. The
  paired shuffled-action control is the statistically clean sibling with no alpha to
  choose, and both are meant to run.

  Runs before any score is emitted. Failure is a refusal, not a low score.
  """
  if not isinstance(alpha, Real) or isinstance(alpha, bool) or not 0.0 < float(alpha) < 1.0:
    raise ValueError(f"alpha must be a fraction strictly between 0 and 1, got {alpha!r}")
  if isinstance(k, bool) or not isinstance(k, Integral) or int(k) < 1:
    raise ValueError(f"k must be a positive integer, got {k!r}")

  pool = tuple(requests)
  k = int(k)
  if len(pool) < k:
    raise ValueError(
      f"interventional_check needs at least k={k} requests and was given "
      f"{len(pool)}. Running it on fewer and reporting the result would let a gate "
      "outcome be attributable to the harness rather than to the model, and 'nothing "
      "was checked' must never be able to print as 'the check passed'."
    )

  model = _model_name(p)
  rng = np.random.default_rng(seed)
  order = rng.permutation(len(pool))[:k]
  probes = tuple(ACTION_FIELDS[column] for column in range(len(INTERVENTION_AXES)))
  probes = probes + (RANDOM_PROBE,)
  divergence: List[float] = []
  required: List[float] = []
  failed: List[int] = []
  notes: List[str] = []
  probe_divergence: List[Tuple[float, ...]] = []
  n_abstained = 0
  n_partial = 0

  for index in order:
    request = pool[int(index)]
    directions = [np.asarray(axis, dtype=float) for axis in INTERVENTION_AXES]
    directions.append(_unit_direction(rng))

    first = _validate_output(p.predict(request), request, p)

    row: List[float] = []
    bars: List[float] = []
    for direction in directions:
      other_actions = request.actions.copy()
      other_actions[:, :3] = other_actions[:, :3] + direction * ENVELOPE_STEP_MM
      other = replace(request, actions=other_actions)

      commanded_mm = float(
        np.linalg.norm(np.sum(other_actions[:, :3] - request.actions[:, :3], axis=0))
      )
      bars.append(float(alpha) * commanded_mm / ENVELOPE_STEP_MM)

      if first.abstain:
        row.append(float("nan"))
        continue
      second = _validate_output(p.predict(other), other, p)
      if second.abstain:
        row.append(float("nan"))
        continue
      moved = np.abs(second.state[-1] - first.state[-1]) / request.tolerance
      row.append(float(np.max(moved)))

    probe_divergence.append(tuple(row))
    bar = max(bars)
    required.append(bar)

    usable = [value for value in row if np.isfinite(value)]
    if not usable:
      n_abstained += 1
      divergence.append(float("nan"))
      failed.append(int(index))
      continue
    if len(usable) < len(row):
      n_partial += 1

    observed = max(usable)
    divergence.append(observed)
    if not observed > bar:
      failed.append(int(index))

  n_responded = len(divergence) - len(failed)
  n_required = _required_responses(k)

  if n_abstained:
    notes.append(
      f"{n_abstained} of {k} gate requests abstained on every probe. An adapter that "
      "refuses on the gate has produced no evidence that it reads its action "
      "argument, and the gate is a precondition for a score rather than a score "
      "itself."
    )
  if n_partial:
    notes.append(
      f"{n_partial} of {k} gate requests abstained on some probes and answered on "
      "others. Those requests were decided on the probes they answered: an "
      "abstention is an absence of evidence about action conditioning, not evidence "
      "against it."
    )
  passed = len(divergence) == k and n_responded >= n_required

  return InterventionResult(
    model=model,
    passed=passed,
    n_checked=k,
    n_abstained=n_abstained,
    alpha=float(alpha),
    k=k,
    seed=int(seed),
    step_bound_mm=float(ENVELOPE_STEP_MM),
    divergence=tuple(divergence),
    required=tuple(required),
    failed=tuple(failed),
    notes=tuple(notes),
    n_responded=int(n_responded),
    n_required=int(n_required),
    probes=probes,
    probe_divergence=tuple(probe_divergence),
  )


def require_action_conditioned(
  p: Any,
  requests: Sequence[RolloutRequest],
  alpha: float = INTERVENTION_ALPHA,
  k: int = INTERVENTION_K,
  seed: int = 0,
) -> InterventionResult:
  """Run the gate and raise unless it passed. The form a scoring path should call.

  :func:`interventional_check` returns its result so the harness can record it, which
  means a caller can also ignore ``passed``. This wrapper cannot be ignored: the
  refusal happens on the way out.
  """
  result = interventional_check(p, requests, alpha=alpha, k=k, seed=seed)
  result.raise_for_refusal()
  return result


class InjectedPredictor:
  """The seam an outside world model plugs into. Empty until something is injected.

  Mirrors :class:`plr_lr.learn.dynamics._InjectedDynamics`: this package loads no
  checkpoints and claims no state, action or unit mapping for any of them. The
  caller owns loading whatever it trusts and supplies a synchronous callable taking
  a :class:`RolloutRequest` and returning a :class:`RolloutPrediction`.

  A bare instance satisfies ``isinstance(instance, RolloutPredictor)`` because the
  Protocol is structural and the declaration attributes are all present. It still
  cannot predict, and calling :meth:`predict` raises :class:`NotImplementedError`.
  That gap is the point: an isinstance check tells you an object has the right
  attribute names, never that a model is behind them, and a registry that admitted
  adapters on an isinstance check would admit every empty one.
  """

  def __init__(
    self,
    name: str = "injected world model",
    state_dim: int = 4,
    horizon: int = 1,
    action_space: str = "ee_delta_mm",
    predictor: Optional[Any] = None,
    training_fingerprints: Optional[FrozenSet[str]] = None,
    fitted_components: Tuple[FittedComponent, ...] = (),
    license_text: str = UNDECLARED,
    checkpoint: Optional[Any] = None,
  ) -> None:
    if predictor is not None and not callable(predictor):
      raise TypeError("predictor must be a synchronous callable or None")
    if action_space not in ACTION_SPACES:
      raise ValueError(
        f"action_space must be one of {list(ACTION_SPACES)}, got {action_space!r}. "
        "The five action columns are read in different units under each one, and an "
        "unnamed action space is read as whichever the reader assumed."
      )
    if int(state_dim) < 1 or int(horizon) < 1:
      raise ValueError("state_dim and horizon must both be at least 1")
    self.name = str(name)
    self.state_dim = int(state_dim)
    self.horizon = int(horizon)
    self.action_space = str(action_space)
    self.predictor = predictor
    self.training_fingerprints: FrozenSet[str] = frozenset(training_fingerprints or ())
    self.fitted_components = tuple(fitted_components)
    self.license_text = str(license_text)
    self.checkpoint = checkpoint

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    """Call the injected callable and hold its output to the contract.

    Only the output half of the contract is enforced here. Provenance withholds a
    score rather than breaking a call, so it is checked by :func:`validate` at the
    point a number would be published, not at the point a prediction is made.
    """
    if self.predictor is None:
      note = " Checkpoint metadata was recorded." if self.checkpoint is not None else ""
      raise NotImplementedError(
        f"the {self.name} adapter has no predictor callable.{note} A checkpoint or an "
        "installed package alone is not an adapter: it establishes no state, action "
        "or unit mapping. Inject a synchronous "
        f"{PREDICT_SIGNATURE}, after mapping the checkpoint's own conventions onto "
        "this request's state_fields, state_units and ACTION_FIELDS layout."
      )
    return _validate_output(self.predictor(request), request, self)

  def __repr__(self) -> str:
    state = "empty" if self.predictor is None else "injected"
    return f"InjectedPredictor({self.name!r}, {state}, state_dim={self.state_dim})"


class ActionEmittingAdapter:
  """Refuses construction. A model that emits actions is a policy, not a world model.

  The distinction decides what a number means. A world model is asked what the bench
  will look like after a commanded action and is scored on whether it was right; a
  policy is asked what to do and is scored on whether the task succeeded. Ranking
  the second on a board built for the first publishes an accuracy figure for a
  question the model was never asked.

  Written as a constructor error rather than a paragraph in a README because a
  taxonomy stated in prose is read once, while this fires every time somebody tries.
  """

  model = "action-emitting model"

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    raise NotAWorldModel(
      f"{self.model} emits actions, so it is a policy and cannot be registered as a "
      "RolloutPredictor. GR00T, pi0 and Cosmos Policy all belong at the "
      "plr_lr.learn.policy.VLAPolicy seam, which is conditioned on an observation "
      "and an instruction and returns an Action. A world model answers a different "
      "question: given this state, these frames and these commanded actions, what "
      "will the state be. Scoring a policy on that board reports an accuracy for a "
      "question it was never asked."
    )


class GR00TAdapter(ActionEmittingAdapter):
  model = "GR00T N1.7"


class Pi0Adapter(ActionEmittingAdapter):
  model = "pi0"


class CosmosPolicyAdapter(ActionEmittingAdapter):
  model = "Cosmos Policy"


def _stable_seed(seed: int, *parts: str) -> int:
  """A seed that depends only on the strings given, and reproduces across processes.

  Deliberately not ``hash()``: string hashing is salted per interpreter, so a null
  seeded from it produces different numbers on every run and the control that is
  supposed to fail every time fails only most of the time.
  """
  digest = hashlib.sha256("|".join((str(seed),) + parts).encode("ascii")).digest()
  return int.from_bytes(digest[:8], "big")


class _NullPredictor:
  """Shared declaration block for the reference nulls below.

  The nulls declare a training fingerprint saying they were fit on nothing. That is
  a declaration, and it is what lets them run through the same provenance gate as a
  real submission instead of being exempted from it.
  """

  name = "null"
  action_space = "ee_delta_mm"
  license_text = "MIT, this package. No external checkpoint is involved."

  def __init__(self, state_dim: int = 4, horizon: int = 2, confidence: float = 0.9) -> None:
    if int(state_dim) < 1 or int(horizon) < 1:
      raise ValueError("state_dim and horizon must both be at least 1")
    if not 0.0 <= float(confidence) <= 1.0:
      raise ValueError("confidence must be in [0, 1]")
    self.state_dim = int(state_dim)
    self.horizon = int(horizon)
    self.confidence = float(confidence)
    self.training_fingerprints: FrozenSet[str] = frozenset({NO_TRAINING_DATA})
    self.fitted_components: Tuple[FittedComponent, ...] = ()


class BlindPredictor(_NullPredictor):
  """Predicts that nothing changes, whatever it was asked to do. Must fail the gate.

  The cheapest thing that satisfies the Protocol and scores respectably on smooth
  tasks, where most of the state most of the time is where it already was. It reads
  the state and never looks at the actions, so its two gate predictions are
  identical and its divergence is exactly zero rather than merely small.
  """

  name = "BlindPredictor"

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    state = np.tile(np.asarray(request.state, dtype=float), (int(request.horizon), 1))
    return RolloutPrediction(
      state=state,
      confidence=self.confidence,
      abstain=False,
      flag=False,
      readout="identity",
      n_forward_passes=0,
      wall_clock_s=0.0,
    )


class NoisePredictor(_NullPredictor):
  """The blind null with noise on it, seeded per episode and never per action.

  The noise is drawn from the task and episode identifiers alone. That is the honest
  version of this null: a model whose randomness depended on the action argument
  would be reading the action argument, which is precisely the property the gate
  looks for, and a null that could pass the gate by accident is not a null.
  """

  name = "NoisePredictor"

  def __init__(
    self,
    state_dim: int = 4,
    horizon: int = 2,
    confidence: float = 0.5,
    sigma: float = 1.0,
    seed: int = 0,
  ) -> None:
    super().__init__(state_dim=state_dim, horizon=horizon, confidence=confidence)
    if not float(sigma) > 0.0:
      raise ValueError("sigma must be positive")
    self.sigma = float(sigma)
    self.seed = int(seed)

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    rng = np.random.default_rng(_stable_seed(self.seed, request.task_id, request.episode_id))
    shape = (int(request.horizon), int(request.state.size))
    noise = rng.normal(0.0, self.sigma, size=shape)
    state = np.asarray(request.state, dtype=float)[None, :] + noise
    return RolloutPrediction(
      state=state,
      confidence=self.confidence,
      abstain=False,
      flag=False,
      readout="identity+noise",
      n_forward_passes=0,
      wall_clock_s=0.0,
    )


#: Every refusal this module raises. The reachability test walks this tuple and
#: demands a call that produces each one, because a refusal that no call can reach
#: is documentation rather than a rule.
REFUSALS: Tuple[type, ...] = (
  NotImplementedError,
  AdapterOutputError,
  FloatingPointError,
  ProvenanceUndeclared,
  UndeclaredComponent,
  InterventionFailed,
  NotAWorldModel,
)
