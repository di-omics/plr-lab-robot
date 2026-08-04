"""The gate between a prediction and a motion, and the reasons it stays shut.

Everything upstream of this module produces numbers. :mod:`plr_lr.learn.predict`
says what a prediction has to look like before anybody is allowed to talk about
it, and :mod:`plr_lr.learn.readout` says how pixels or latents become a pose.
Neither one decides anything. This module is the only place in the package where
a prediction is converted into permission to move, and it is written so that
every path through it that is not a clean pass ends in a refusal that is recorded
with a reason from a closed set.

Admission requires **all** of five conditions, and there is no weighting, no
score, and no way to trade one against another:

* the adapter did not abstain;
* the confidence is at least ``theta``, the threshold frozen at calibration time
  and reprinted verbatim as ``calibrated_on`` on every report it decides;
* the requested horizon is inside ``trust_horizon``, the measured horizon at
  which risk at full coverage was still acceptable;
* the readout says it measured something;
* the newest frame is younger than a declared staleness bound, checked against
  ``FrameRef.t_capture_s`` rather than against when the prediction happened to be
  computed.

Three things about this design are load-bearing and are easy to undo by accident.

**Deferral is a terminal state, not a retry.** There is no fallback action here,
no reduced-speed variant, no "try the safe version instead". The documented
failure mode of learned controllers is that they convert uncertainty into
unwarranted actionability, and a fallback action is precisely that conversion
wearing a cautious name: the run continues, something moves, and the abstention
appears in the log as though it had been honoured. A bounded re-observation is
allowed, at most :data:`MAX_REOBSERVATIONS` of them, because looking again is not
acting. After that the run stops safe and stays stopped.

**A clamped action is not the action that was screened.** The safety envelope
rewrites a command after the screen has passed on it, so the ACT verdict belongs
to a command that is not the one about to be sent. Every clamped step is recorded
as ``ABSTAIN(CLAMPED)``. This is not fastidiousness: the measured clamp rate of
the existing random null against the default envelope is close to half its steps,
and without this rule an arm study would be reporting "abstention predicts
failure" when what it measured was "clamping predicts failure".

**The threshold is never derived from the data being screened.** ``alpha`` and
``min_calibration`` are absolute constants in the signature of
:meth:`Screen.calibrate`, and a screen refuses to decide an episode whose
identifier appears in its own calibration set. A threshold fitted on the
episodes it then judges is a threshold that has already seen the answer.

Calibration follows the split-conformal protocol on **held-out successful
episodes only**. No failure data is required or wanted: the question is what a
confident success looks like, and the alarm fires when the current episode stops
looking like one. Roughly ten successful trajectories is the floor, and it is a
floor for an arithmetic reason rather than a stylistic one, spelled out in
:meth:`Screen.calibrate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from enum import Enum
from numbers import Integral, Real
from typing import Any, Callable, Dict, FrozenSet, List, Optional, Protocol, Sequence, Tuple
from typing import runtime_checkable

import numpy as np

from .predict import FrameRef, RolloutPrediction, RolloutRequest
from .readout import ReadoutResult, read_or_refuse
from .types import ACTION_FIELDS, Action, Observation

#: Target miscoverage for split conformal calibration. An absolute constant: it
#: is never read from a submission, never tuned against the models being screened,
#: and never recomputed from the episodes being judged. A threshold tuned on the
#: data it will decide measures the tuner.
ALPHA = 0.10

#: Fewest successful held-out episodes that can produce a threshold. Also
#: absolute. Below this the conformal quantile index falls off the end of the
#: sample and the only threshold consistent with the data is one that admits
#: everything, which is not a calibration, it is the absence of one.
MIN_CALIBRATION = 10

#: How many times a screened run may look again before it stops. Two, and then
#: the run is over. Re-observation is not action, so a bounded number of them
#: does not convert uncertainty into motion; an unbounded number turns a terminal
#: refusal back into a retry loop that eventually finds a frame it likes.
MAX_REOBSERVATIONS = 2

#: What ``calibrated_on`` reads when no calibration has happened. A sentinel and
#: not an empty string, because an empty provenance field prints as whitespace in
#: a table and reads as though it were simply not shown.
UNCALIBRATED = "UNCALIBRATED"


class ScreenOutcome(str, Enum):
  """The only two answers. There is deliberately no third, softer one."""

  ACT = "act"
  ABSTAIN = "abstain"


class AbstainReason(str, Enum):
  """Why a prediction was not acted on. A closed set, recorded per step.

  Collapsing these into a single boolean is the specific mistake that turned a
  model's refusals into missing data in the closest published work: the samples
  where the model declined were dropped from the analysis rather than scored, so
  a correct refusal and a crashed run were the same row.
  """

  LOW_CONFIDENCE = "low_confidence"
  READOUT_FAILED = "readout_failed"
  HORIZON_EXCEEDED = "horizon_exceeded"
  CLAMPED = "clamped"
  STALE = "stale"
  NOT_CALIBRATED = "not_calibrated"
  CEILING = "ceiling"


#: Order in which simultaneous reasons are reported. The reasons that invalidate
#: the question come before the ones that grade the answer: there is no point
#: reporting that a confidence was low when the threshold it was compared against
#: was never calibrated, or when the deciding latent is not visible to the camera
#: at all. Every applicable reason is still recorded in ``ScreenReport.reasons``;
#: this only fixes which one is named first.
REASON_PRECEDENCE: Tuple[AbstainReason, ...] = (
  AbstainReason.NOT_CALIBRATED,
  AbstainReason.CEILING,
  AbstainReason.CLAMPED,
  AbstainReason.STALE,
  AbstainReason.READOUT_FAILED,
  AbstainReason.HORIZON_EXCEEDED,
  AbstainReason.LOW_CONFIDENCE,
)


class ScreenDeclined(RuntimeError):
  """Raised when a caller asks a refused verdict to proceed anyway.

  Carries the reason so a handler can branch on why rather than on the message.
  Every refusal is already recorded as a step before this is ever raised; the
  exception exists for call sites that want the refusal to be loud rather than
  merely written down.
  """

  def __init__(self, reason: AbstainReason, message: str) -> None:
    super().__init__(message)
    self.reason = reason


class CeilingRefusal(ScreenDeclined):
  """Raised when the gate is asked to act on a prediction it cannot resolve.

  The deciding latent is invisible to the camera: reagent identity in a correctly
  filled well, whether a tip is sealed rather than merely seated, whether the
  contents match the label. No amount of model scale changes this, so the refusal
  is separate from a low confidence, which more data or a better model could fix.
  """

  def __init__(self, message: str) -> None:
    super().__init__(AbstainReason.CEILING, message)


class NotCalibrated(ValueError):
  """Raised when a threshold is asked for and the data cannot support one.

  Returning a permissive default here is the failure this prevents. A screen
  built with ``theta = 0.0`` admits everything and looks, in every log and every
  table, exactly like a screen that was calibrated and found the model confident.
  """


@dataclass(frozen=True)
class CalibrationPoint:
  """One held-out **successful** episode and the confidence it was produced with.

  The parameter this is passed as is named ``successes`` and means it. Split
  conformal on successes answers "how confident does this model look when it is
  right", and mixing failures in drags the quantile down until the threshold
  admits the failures too. No failure data is needed, which is the property that
  makes this affordable on a real bench: ten good trajectories, not a fault
  injection campaign.
  """

  episode_id: str
  confidence: float

  def __post_init__(self) -> None:
    if not isinstance(self.episode_id, str) or not self.episode_id:
      raise ValueError("CalibrationPoint.episode_id must be a non-empty string")
    if isinstance(self.confidence, bool) or not isinstance(self.confidence, Real):
      raise ValueError(f"CalibrationPoint.confidence must be a number, got {self.confidence!r}")
    value = float(self.confidence)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
      raise ValueError(
        f"CalibrationPoint.confidence must be a finite number in [0, 1], got {value!r}. "
        "An abstained episode carries a NaN confidence and is not a calibration "
        "point: the threshold is fitted to what acting on a success looks like."
      )
    object.__setattr__(self, "confidence", value)


@dataclass(frozen=True)
class ScreenReport:
  """One verdict, with everything it was decided from written down beside it.

  Frozen, and the invariant that ACT carries no reasons is enforced rather than
  assumed. A report that says ACT while carrying a reason is the shape a partial
  refactor leaves behind, and downstream code reads the boolean.
  """

  outcome: ScreenOutcome
  reasons: Tuple[AbstainReason, ...]
  confidence: float
  theta: float
  calibrated_on: str
  horizon: int
  trust_horizon: int
  frame_age_s: float
  max_frame_age_s: float
  readout_ok: bool
  readout: str = "none"
  clamp_magnitude_mm: float = 0.0
  reobserve: bool = False
  notes: Tuple[str, ...] = ()

  def __post_init__(self) -> None:
    object.__setattr__(self, "reasons", tuple(self.reasons))
    object.__setattr__(self, "notes", tuple(self.notes))
    if self.outcome is ScreenOutcome.ACT and self.reasons:
      raise ValueError(
        f"a report cannot be ACT and carry reasons {[r.value for r in self.reasons]}. "
        "Whichever of the two is right, the other one is what somebody downstream "
        "will read."
      )
    if self.outcome is ScreenOutcome.ABSTAIN and not self.reasons:
      raise ValueError(
        "an abstention with no reason is indistinguishable from a crash in every "
        "aggregate that counts refusals. Name the reason."
      )
    if self.outcome is ScreenOutcome.ACT and self.reobserve:
      raise ValueError("an admitted verdict does not schedule a re-observation")

  @property
  def admitted(self) -> bool:
    """True only for ACT. The one boolean anything downstream is allowed to read."""
    return self.outcome is ScreenOutcome.ACT

  @property
  def reason(self) -> Optional[AbstainReason]:
    """The reason named first, by :data:`REASON_PRECEDENCE`. None when admitted."""
    return self.reasons[0] if self.reasons else None

  def clamped_downgrade(self, clamp_magnitude_mm: float) -> "ScreenReport":
    """Rewrite an admitted verdict as ``ABSTAIN(CLAMPED)`` after the envelope acted.

    The single most important transition in this module. The verdict was reached
    about a command that no longer exists: the envelope scaled it, clipped it, or
    both, and what is about to be sent differs from what was screened by
    ``clamp_magnitude_mm``. An ACT verdict does not transfer to a different
    command, so it is not carried over, it is replaced.

    Re-observation is deliberately not offered. Looking again does not un-clamp a
    command that was too large for the envelope; the policy would propose the same
    oversized motion and the loop would spin until the budget ran out, which reads
    in the record as a run that simply took a while.
    """
    magnitude = float(clamp_magnitude_mm)
    if not math.isfinite(magnitude) or magnitude < 0.0:
      raise ValueError(f"clamp_magnitude_mm must be finite and non-negative, got {magnitude!r}")
    reasons = _ordered(set(self.reasons) | {AbstainReason.CLAMPED})
    return replace(
      self,
      outcome=ScreenOutcome.ABSTAIN,
      reasons=reasons,
      clamp_magnitude_mm=magnitude,
      reobserve=False,
      notes=self.notes
      + (
        f"the safety envelope changed this command by {magnitude:.3f} mm after it was "
        "screened, so the verdict was about a command that is not the one being sent",
      ),
    )

  def with_reobservation(self, reobserve: bool) -> "ScreenReport":
    """Record whether a bounded look-again is still budgeted for this refusal."""
    if self.admitted and reobserve:
      raise ValueError("an admitted verdict does not schedule a re-observation")
    return replace(self, reobserve=bool(reobserve))

  def raise_for_refusal(self) -> None:
    """Turn a refusal into the exception it is. No-op when admitted."""
    if self.admitted:
      return
    reason = self.reason
    assert reason is not None  # guaranteed by the invariant in __post_init__
    message = self.summary()
    if reason is AbstainReason.CEILING:
      raise CeilingRefusal(message)
    raise ScreenDeclined(reason, message)

  def to_dict(self) -> Dict[str, Any]:
    """A JSON-safe record of the verdict, written into every screened step."""
    return {
      "outcome": self.outcome.value,
      "reason": None if self.reason is None else self.reason.value,
      "reasons": [r.value for r in self.reasons],
      "confidence": None if math.isnan(self.confidence) else self.confidence,
      "theta": None if math.isnan(self.theta) else self.theta,
      "calibrated_on": self.calibrated_on,
      "horizon": self.horizon,
      "trust_horizon": self.trust_horizon,
      "frame_age_s": None if math.isnan(self.frame_age_s) else self.frame_age_s,
      "max_frame_age_s": self.max_frame_age_s,
      "readout_ok": self.readout_ok,
      "readout": self.readout,
      "clamp_magnitude_mm": self.clamp_magnitude_mm,
      "reobserve": self.reobserve,
      "notes": list(self.notes),
    }

  def summary(self) -> str:
    head = "ACT" if self.admitted else f"ABSTAIN({self.reason.value})"  # type: ignore[union-attr]
    lines = [
      f"screen: {head}",
      f"  confidence {_fmt(self.confidence)} against theta {_fmt(self.theta)}",
      # Printed verbatim, every time. A frozen threshold whose provenance is a
      # second lookup is a frozen threshold nobody checks.
      f"  calibrated_on: {self.calibrated_on}",
      f"  horizon {self.horizon} against trust horizon {self.trust_horizon}",
      f"  frame age {_fmt(self.frame_age_s)} s against bound "
      f"{self.max_frame_age_s:.3f} s",
      f"  readout {self.readout}: {'ok' if self.readout_ok else 'FAILED'}",
    ]
    if len(self.reasons) > 1:
      others = ", ".join(r.value for r in self.reasons[1:])
      lines.append(f"  also: {others}")
    if self.clamp_magnitude_mm:
      lines.append(f"  clamp magnitude {self.clamp_magnitude_mm:.3f} mm")
    if not self.admitted:
      lines.append(
        "  re-observe and look again" if self.reobserve else "  terminal: the run stops safe"
      )
    lines.extend(f"  note: {note}" for note in self.notes)
    return "\n".join(lines)


def _fmt(value: float) -> str:
  return "nan" if math.isnan(value) else format(float(value), ".4f")


def _ordered(reasons: Any) -> Tuple[AbstainReason, ...]:
  """Reasons in :data:`REASON_PRECEDENCE` order, deduplicated."""
  present = set(reasons)
  return tuple(r for r in REASON_PRECEDENCE if r in present)


@runtime_checkable
class PredictionScreen(Protocol):
  """Decides one prediction. Satisfied by :class:`Screen` and by every null below.

  Structural, so satisfying it establishes nothing about whether the decision is
  any good. That is what :func:`screen_null_check` is for: six things satisfy this
  Protocol and all six must fail.
  """

  name: str

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    ...


@runtime_checkable
class StepScreen(Protocol):
  """What :func:`plr_lr.learn.rollout.rollout` needs: a verdict on one action.

  Narrow on purpose. The rollout loop knows nothing about predictions, requests,
  frames or thresholds; it asks for the verdict that applies to the action it is
  holding, applies the clamp downgrade if the envelope then changes that action,
  and records the result. :class:`ScreenedPolicy` is the thing that satisfies
  this. A bare :class:`Screen` does not, because a screen decides a prediction and
  the rollout loop has never seen one.
  """

  def review(self, observation: Observation, action: Action) -> ScreenReport:
    ...


@dataclass(frozen=True)
class Screen:
  """A frozen threshold plus the four other conditions admission requires.

  Build one with :meth:`calibrate`. Constructing it directly is allowed because a
  receipt from a previous calibration has to be reconstructible, and every field
  needed to reconstruct it is here, but the ``calibrated_on`` string is mandatory
  and is reprinted verbatim on every verdict so a hand-written threshold cannot
  quietly pass for a fitted one.

  Args:
    theta: the frozen confidence threshold. NaN only for an uncalibrated screen.
    trust_horizon: ``k*``, the measured horizon at which risk at full coverage was
      still acceptable. A measurement carried in from the benchmark, not a
      guarantee derived here.
    max_frame_age_s: the declared staleness bound, checked against
      ``FrameRef.t_capture_s``. There is no default. A default staleness bound is
      a number nobody chose being used to decide whether a prediction is about the
      plate in front of the arm now or one from thirty seconds ago.
    calibrated_on: provenance of ``theta``, printed verbatim on every report.
    alpha: the miscoverage the threshold was fitted at, recorded for the receipt.
    n_calibration: how many successful episodes it was fitted on.
    calibration_ids: which ones. A screen refuses to decide any of them.
  """

  theta: float
  trust_horizon: int
  max_frame_age_s: float
  calibrated_on: str
  alpha: float = ALPHA
  n_calibration: int = 0
  calibration_ids: FrozenSet[str] = field(default_factory=frozenset)
  name: str = "screen"

  def __post_init__(self) -> None:
    if not isinstance(self.calibrated_on, str) or not self.calibrated_on:
      raise ValueError("Screen.calibrated_on must be a non-empty string")
    if isinstance(self.theta, bool) or not isinstance(self.theta, Real):
      raise ValueError(f"Screen.theta must be a number, got {self.theta!r}")
    theta = float(self.theta)
    if self.calibrated_on == UNCALIBRATED:
      if not math.isnan(theta):
        raise ValueError(
          f"an uncalibrated screen carries theta NaN, not {theta!r}. A numeric "
          "threshold with no provenance is the shape of a guess that later gets "
          "quoted as a calibration."
        )
    elif not math.isfinite(theta) or not 0.0 <= theta <= 1.0:
      raise ValueError(f"Screen.theta must be finite and in [0, 1], got {theta!r}")
    object.__setattr__(self, "theta", theta)

    if isinstance(self.trust_horizon, bool) or not isinstance(self.trust_horizon, Integral):
      raise ValueError(f"Screen.trust_horizon must be an integer, got {self.trust_horizon!r}")
    if int(self.trust_horizon) < 1:
      raise ValueError(
        f"Screen.trust_horizon must be at least 1, got {self.trust_horizon!r}. A "
        "measured trust horizon of zero means no horizon was trustworthy, which is a "
        "reason not to build a screen rather than a screen that admits nothing."
      )
    object.__setattr__(self, "trust_horizon", int(self.trust_horizon))

    bound = _positive_finite("Screen.max_frame_age_s", self.max_frame_age_s)
    object.__setattr__(self, "max_frame_age_s", bound)

    if isinstance(self.alpha, bool) or not isinstance(self.alpha, Real):
      raise ValueError(f"Screen.alpha must be a number, got {self.alpha!r}")
    alpha = float(self.alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
      raise ValueError(f"Screen.alpha must be strictly between 0 and 1, got {alpha!r}")
    object.__setattr__(self, "alpha", alpha)

    if isinstance(self.n_calibration, bool) or not isinstance(self.n_calibration, Integral):
      raise ValueError("Screen.n_calibration must be an integer")
    object.__setattr__(self, "n_calibration", int(self.n_calibration))
    object.__setattr__(self, "calibration_ids", frozenset(self.calibration_ids))

  @property
  def is_calibrated(self) -> bool:
    return self.calibrated_on != UNCALIBRATED

  # -- construction ----------------------------------------------------------

  @classmethod
  def uncalibrated(
    cls,
    trust_horizon: int = 1,
    max_frame_age_s: float = 1.0,
    name: str = "screen",
  ) -> "Screen":
    """A screen that refuses everything with ``NOT_CALIBRATED``.

    The honest state of a gate before any calibration data exists, and the right
    thing to hand to a run that is being wired up. The alternative that this
    exists to avoid is a placeholder ``theta`` of 0.0, which is not a disabled
    screen, it is a screen that admits every prediction ever made.
    """
    return cls(
      theta=float("nan"),
      trust_horizon=trust_horizon,
      max_frame_age_s=max_frame_age_s,
      calibrated_on=UNCALIBRATED,
      name=name,
    )

  @classmethod
  def calibrate(
    cls,
    successes: Sequence[CalibrationPoint],
    source: str,
    trust_horizon: int,
    max_frame_age_s: float,
    alpha: float = ALPHA,
    min_calibration: int = MIN_CALIBRATION,
    name: str = "screen",
  ) -> "Screen":
    """Split conformal on held-out successful episodes. Nothing else is needed.

    ``successes`` are episodes the arm actually completed inside tolerance, drawn
    from a split that will not be screened. Failure data is not required, which is
    the property that makes this affordable on a bench: roughly ten good
    trajectories, not a fault-injection campaign. The nonconformity score is
    ``1 - confidence``, so the threshold is the confidence that all but an
    ``alpha`` share of known-good episodes exceeded, and a future episode that
    looks less confident than a success usually does is refused.

    ``alpha`` and ``min_calibration`` are absolute constants in this signature and
    are never derived from the episodes being screened. They are arguments only so
    that a receipt can record which values produced a stored threshold; passing a
    value computed from the scoring data reproduces exactly the failure the split
    exists to prevent, and nothing here can detect that, so it is stated instead.

    ``min_calibration`` has an arithmetic floor as well as a stylistic one. The
    conformal rank is ``ceil((n + 1) * (1 - alpha))`` and it has to land inside
    the sample: at ``alpha = 0.10`` that needs at least nine points, and below
    that the only threshold consistent with the data admits everything. Ten is
    the smallest round number above the floor.

    Raises:
      NotCalibrated: too few successes, or an ``alpha`` too small for this ``n``.
      ValueError: a duplicated episode, which silently inflates ``n`` and drags
        the conformal rank down onto a lower quantile than the data supports.
    """
    if isinstance(min_calibration, bool) or not isinstance(min_calibration, Integral):
      raise ValueError(f"min_calibration must be an integer, got {min_calibration!r}")
    if int(min_calibration) < 1:
      raise ValueError("min_calibration must be at least 1")
    if isinstance(alpha, bool) or not isinstance(alpha, Real):
      raise ValueError(f"alpha must be a number, got {alpha!r}")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0.0 < alpha < 1.0:
      raise ValueError(f"alpha must be strictly between 0 and 1, got {alpha!r}")
    if not isinstance(source, str) or not source:
      raise ValueError(
        "source must name the split theta was fitted on. It is printed verbatim on "
        "every verdict this screen decides, and an unnamed split cannot be checked "
        "for overlap with the episodes being screened."
      )
    if source == UNCALIBRATED:
      raise ValueError(f"source must not be the literal {UNCALIBRATED!r}")

    points = tuple(successes)
    for point in points:
      if not isinstance(point, CalibrationPoint):
        raise ValueError(
          f"calibration takes CalibrationPoint entries, got {type(point).__name__}. "
          "A bare float carries no episode identity, so the screen cannot later "
          "refuse to decide the episodes it was fitted on."
        )
    ids = [p.episode_id for p in points]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
      raise ValueError(
        f"calibration episodes {duplicates} appear more than once. A repeated episode "
        "inflates n without adding information, which moves the conformal rank onto a "
        "lower quantile and produces a threshold the data does not support."
      )

    n = len(points)
    if n < int(min_calibration):
      raise NotCalibrated(
        f"{n} successful held-out episodes, and {int(min_calibration)} is the floor. "
        "Below it the conformal rank falls off the end of the sample and the only "
        "threshold consistent with the data is one that admits everything, which "
        "looks in every log exactly like a calibrated screen finding the model "
        "confident. Record more successful episodes, or run with "
        "Screen.uncalibrated()."
      )

    rank = int(math.ceil((n + 1) * (1.0 - alpha)))
    if rank > n:
      raise NotCalibrated(
        f"alpha={alpha} needs at least {int(math.ceil(1.0 / alpha)) - 1} calibration "
        f"episodes and there are {n}. The conformal quantile sits outside the sample, "
        "so no finite threshold carries the coverage this alpha claims."
      )

    # Taken as an order statistic of the confidences directly, not as
    # ``1 - sorted(1 - c)[rank - 1]``. The two are the same quantile and are not
    # the same float: the round trip through the nonconformity score can land
    # theta an ulp above the calibration point it came from, and the screen then
    # refuses one of its own successes. Measured, not hypothetical, at
    # confidence 0.02, where 1 - (1 - 0.02) is 0.020000000000000018.
    theta = float(sorted((p.confidence for p in points), reverse=True)[rank - 1])
    calibrated_on = (
      f"{source} | split-conformal on successes | n={n} | alpha={alpha:.4f} | "
      f"rank={rank}/{n} | theta={theta:.6f}"
    )
    return cls(
      theta=theta,
      trust_horizon=trust_horizon,
      max_frame_age_s=max_frame_age_s,
      calibrated_on=calibrated_on,
      alpha=alpha,
      n_calibration=n,
      calibration_ids=frozenset(ids),
      name=name,
    )

  # -- the decision ----------------------------------------------------------

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    """All five conditions, evaluated together, with every failure recorded.

    Every condition is evaluated even after one has already failed, so the report
    carries the full set rather than the first one hit. A refusal that names only
    the first tripped wire hides the fact that a prediction was stale *and*
    beyond the trust horizon, and the reader repairs one and re-runs.

    ``readout`` is a required argument with no default, including when there is no
    readout stage; pass ``None`` deliberately. A default of "no readout, therefore
    fine" is the permissive branch, and it is reached by forgetting rather than by
    deciding.

    ``now_s`` and ``FrameRef.t_capture_s`` must be on the same clock. Which clock
    that is travels on ``FrameRef.clock_source``; the screen only measures the
    difference and cannot tell a device stamp from a host arrival time.
    """
    reasons: List[AbstainReason] = []
    notes: List[str] = []

    if not self.is_calibrated:
      reasons.append(AbstainReason.NOT_CALIBRATED)
      notes.append("this screen has no calibrated threshold, so it admits nothing")
    elif request.episode_id in self.calibration_ids:
      reasons.append(AbstainReason.NOT_CALIBRATED)
      notes.append(
        f"episode {request.episode_id} was one of the {self.n_calibration} episodes "
        "theta was fitted on, so this screen has already seen its answer"
      )

    if ceiling:
      reasons.append(AbstainReason.CEILING)
      notes.append(
        "the deciding latent for this episode is not visible to the camera, so no "
        "prediction from frames can resolve it at any model scale"
      )

    if clamped:
      reasons.append(AbstainReason.CLAMPED)

    age = self._frame_age(request, now_s, reasons, notes)

    readout_ok = True
    readout_name = "none"
    if readout is not None:
      readout_name = readout.name
      readout_ok = bool(readout.ok)
      if not readout_ok:
        reasons.append(AbstainReason.READOUT_FAILED)
        notes.append(f"readout {readout.name} refused: {readout.reason}")

    if int(request.horizon) > self.trust_horizon:
      reasons.append(AbstainReason.HORIZON_EXCEEDED)
      notes.append(
        f"horizon {int(request.horizon)} is past the measured trust horizon "
        f"{self.trust_horizon}, where risk at full coverage was last acceptable"
      )

    confidence = self._confidence(prediction, reasons, notes)


    magnitude = float(clamp_magnitude_mm)
    if not math.isfinite(magnitude) or magnitude < 0.0:
      raise ValueError(f"clamp_magnitude_mm must be finite and non-negative, got {magnitude!r}")

    ordered = _ordered(reasons)
    return ScreenReport(
      outcome=ScreenOutcome.ABSTAIN if ordered else ScreenOutcome.ACT,
      reasons=ordered,
      confidence=confidence,
      theta=self.theta,
      calibrated_on=self.calibrated_on,
      horizon=int(request.horizon),
      trust_horizon=self.trust_horizon,
      frame_age_s=age,
      max_frame_age_s=self.max_frame_age_s,
      readout_ok=readout_ok,
      readout=readout_name,
      clamp_magnitude_mm=magnitude,
      notes=tuple(notes),
    )

  def _frame_age(
    self,
    request: RolloutRequest,
    now_s: float,
    reasons: List[AbstainReason],
    notes: List[str],
  ) -> float:
    """Age of the newest frame, and the staleness verdict that follows from it."""
    if isinstance(now_s, bool) or not isinstance(now_s, Real) or not math.isfinite(float(now_s)):
      raise ValueError(f"now_s must be a finite number of seconds, got {now_s!r}")
    frame: Optional[FrameRef] = request.newest_frame()
    if frame is None:
      reasons.append(AbstainReason.STALE)
      notes.append(
        "this request carries no frame, so the staleness bound cannot be checked. A "
        "prediction whose observation cannot be dated is not refused for being old, "
        "it is refused for being undatable."
      )
      return float("nan")
    age = float(now_s) - float(frame.t_capture_s)
    if age < 0.0:
      reasons.append(AbstainReason.STALE)
      notes.append(
        f"the newest frame is stamped {abs(age):.3f} s in the future against the clock "
        f"the screen was given ({frame.clock_source}), so the two are not the same clock"
      )
    elif age > self.max_frame_age_s:
      reasons.append(AbstainReason.STALE)
      notes.append(
        f"the newest frame is {age:.3f} s old against a declared bound of "
        f"{self.max_frame_age_s:.3f} s"
      )
    return age

  def _confidence(
    self,
    prediction: RolloutPrediction,
    reasons: List[AbstainReason],
    notes: List[str],
  ) -> float:
    """The confidence to report, and whether it clears the frozen threshold."""
    abstained = bool(prediction.abstain)
    raw = prediction.confidence
    confidence = float(raw) if isinstance(raw, Real) and not isinstance(raw, bool) else float("nan")
    if abstained:
      reasons.append(AbstainReason.LOW_CONFIDENCE)
      notes.append("the adapter abstained; a refusal upstream is not overridden here")
      if not math.isnan(confidence):
        notes.append(
          "the adapter abstained while still reporting a confidence, which breaks the "
          "row invariant plr_lr.learn.predict.validate enforces"
        )
      return float("nan")
    if math.isnan(confidence):
      reasons.append(AbstainReason.LOW_CONFIDENCE)
      notes.append(
        "the adapter did not abstain but reported a NaN confidence, which breaks the "
        "row invariant; an unmeasured confidence is never treated as a high one"
      )
      return confidence
    if not self.check_theta(confidence):
      reasons.append(AbstainReason.LOW_CONFIDENCE)
      notes.append(
        f"confidence {confidence:.4f} is below the frozen threshold {_fmt(self.theta)}"
      )
    return confidence

  def check_theta(self, confidence: float) -> bool:
    """Does one confidence clear the frozen threshold? The whole comparison.

    Written as ``theta <= confidence`` rather than as the negation of the opposite
    test. With a NaN on either side every ordered comparison is False, so this
    returns False for an unmeasured confidence and False for an uncalibrated
    threshold, and both of those are refusals. The negated form,
    ``not (confidence < theta)``, returns True in exactly those two cases, which
    is an admission granted by the absence of a number.
    """
    return bool(self.theta <= float(confidence))

  def summary(self) -> str:
    return "\n".join(
      [
        f"{self.name}: theta {_fmt(self.theta)}",
        f"  calibrated_on: {self.calibrated_on}",
        f"  trust horizon {self.trust_horizon}, staleness bound "
        f"{self.max_frame_age_s:.3f} s",
        f"  alpha {self.alpha:.4f} over {self.n_calibration} successful episodes",
      ]
    )


def _positive_finite(label: str, value: Any) -> float:
  if isinstance(value, bool) or not isinstance(value, Real):
    raise ValueError(f"{label} must be a positive finite number, got {value!r}")
  out = float(value)
  if not math.isfinite(out) or out <= 0.0:
    raise ValueError(f"{label} must be a positive finite number, got {value!r}")
  return out


class ScreenedPolicy:
  """A policy whose every action has to clear the screen before it is proposed.

  Wraps an inner policy and stands in front of it. On each observation it builds
  the request, asks the predictor, runs the readout, and screens the result. Only
  an admitted verdict lets the inner policy speak.

  What happens on a refusal is the part worth reading. There is no fallback
  action, no slower variant, no reduced-amplitude version of what the policy
  wanted. The refusal produces :meth:`plr_lr.learn.types.Action.hold`, and after
  ``max_reobservations`` refusals the hold carries ``terminate`` and the run is
  over. A fallback action would convert the abstention back into motion, which is
  the exact failure this whole path exists to prevent, and it would do it while
  leaving a log full of correctly recorded abstentions.

  The re-observation budget is per run and is reset with the policy. It is not a
  retry of the same decision: the arm has not moved, the world may have, and a
  fresh frame is a new observation rather than a second attempt at the old one.

  Args:
    policy: the inner controller. Only reached on an admitted verdict.
    screen: the calibrated gate.
    predictor: anything with ``predict(request) -> RolloutPrediction``.
    request_for: builds the request for one observation. This is where frames,
      hashes and capture times are attached, and it is a caller argument because
      nothing in this package knows how the caller's frames are stored.
    readout: optional. Passed straight through to the screen, refusals and all.
    ceiling_for: optional predicate naming the episodes whose deciding latent the
      camera cannot see. Returning True refuses at the ceiling rather than
      reporting a low confidence, because the two have different remedies.
    now_for: what clock the staleness bound is measured on. Defaults to the
      observation's own timestamp, which is the same clock a recorder stamps
      frames with when it is the thing capturing them.
    max_reobservations: how many refusals precede the terminal one.
  """

  def __init__(
    self,
    policy: Any,
    screen: Screen,
    predictor: Any,
    request_for: Callable[[Observation], RolloutRequest],
    readout: Optional[Any] = None,
    ceiling_for: Optional[Callable[[RolloutRequest], bool]] = None,
    now_for: Optional[Callable[[Observation], float]] = None,
    max_reobservations: int = MAX_REOBSERVATIONS,
  ) -> None:
    if not callable(getattr(policy, "act", None)):
      raise TypeError("policy must have an act(observation) -> Action method")
    if not callable(getattr(predictor, "predict", None)):
      raise TypeError("predictor must have a predict(request) -> RolloutPrediction method")
    if not callable(request_for):
      raise TypeError("request_for must be callable")
    if ceiling_for is not None and not callable(ceiling_for):
      raise TypeError("ceiling_for must be callable or None")
    if now_for is not None and not callable(now_for):
      raise TypeError("now_for must be callable or None")
    if isinstance(max_reobservations, bool) or not isinstance(max_reobservations, Integral):
      raise ValueError("max_reobservations must be an integer")
    if int(max_reobservations) < 0:
      raise ValueError(
        "max_reobservations must be zero or more. A negative budget reads as "
        "'unbounded' to exactly one reader and as 'none' to the next."
      )
    self.policy = policy
    self.screen = screen
    self.predictor = predictor
    self.request_for = request_for
    self.readout = readout
    self.ceiling_for = ceiling_for
    self.now_for = now_for
    self.max_reobservations = int(max_reobservations)
    self.reports: List[ScreenReport] = []
    self.n_deferred = 0
    self._pending: Optional[ScreenReport] = None

  def reset(self) -> None:
    """Clear the re-observation budget and the recorded verdicts, and reset inner."""
    self.reports = []
    self.n_deferred = 0
    self._pending = None
    reset = getattr(self.policy, "reset", None)
    if callable(reset):
      reset()

  def act(self, observation: Observation) -> Action:
    request = self.request_for(observation)
    if not isinstance(request, RolloutRequest):
      raise TypeError(
        f"request_for returned {type(request).__name__}, not a RolloutRequest. Without "
        "the request there are no frames, no capture times and no tolerance, so the "
        "staleness and horizon conditions cannot be checked at all."
      )
    prediction = self.predictor.predict(request)
    result = None if self.readout is None else read_or_refuse(self.readout, observation)
    ceiling = bool(self.ceiling_for(request)) if self.ceiling_for is not None else False
    now_s = float(self.now_for(observation)) if self.now_for is not None else float(observation.t)

    report = self.screen.check(
      prediction,
      request,
      now_s=now_s,
      readout=result,
      clamped=False,
      ceiling=ceiling,
    )
    if report.admitted:
      self._record(report)
      return self.policy.act(observation)

    self.n_deferred += 1
    reobserve = self.n_deferred <= self.max_reobservations
    report = report.with_reobservation(reobserve)
    self._record(report)
    # Hold, never a fallback. terminate goes on the hold that exhausts the budget,
    # so the run stops safe rather than looping until the step budget runs out and
    # records itself as a rollout that simply took a while.
    return Action.hold(terminate=not reobserve)

  def review(self, observation: Observation, action: Action) -> ScreenReport:
    """The verdict for the action just produced. The rollout-facing seam.

    Refuses rather than recomputing. Screening again here would call the predictor
    a second time on the same observation, and a stochastic adapter would then be
    judged on one prediction and act on another.
    """
    if self._pending is None:
      raise ScreenDeclined(
        AbstainReason.NOT_CALIBRATED,
        "review() was asked for a verdict before act() produced one. The rollout loop "
        "calls act() first; a screen asked out of order has nothing to report and "
        "must not invent an admission.",
      )
    return self._pending

  def _record(self, report: ScreenReport) -> None:
    self._pending = report
    self.reports.append(report)

  @property
  def stopped_safe(self) -> bool:
    """True once the re-observation budget is spent. Terminal, never cleared."""
    return self.n_deferred > self.max_reobservations

  def __repr__(self) -> str:
    return (
      f"ScreenedPolicy({type(self.policy).__name__}, theta={_fmt(self.screen.theta)}, "
      f"deferred={self.n_deferred}/{self.max_reobservations})"
    )


# -- null screens -------------------------------------------------------------
#
# Six things that satisfy PredictionScreen and must all fail. They are here rather
# than in the tests because a control that lives only in a test file is deleted by
# the first refactor that finds it inconvenient, and because the harness has to be
# able to run them against a real submission's cases, not only against a fixture.


class AlwaysAdmitScreen:
  """Admits everything. The screen you get by deleting the screen.

  Must fail. It agrees with a real screen exactly on the episodes a real screen
  would admit, which is why a suite that only exercises the happy path cannot
  tell the two apart.
  """

  name = "null:always-admit"

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    return _null_report(
      self.name,
      request,
      prediction,
      reasons=(),
      clamp_magnitude_mm=clamp_magnitude_mm,
    )


class AlwaysAbstainScreen:
  """Refuses everything. Perfect safety, zero utility, and it must fail too.

  Its abstention rate is unbeatable and it is worth nothing, which is the reason
  abstention credit scores a refusal on a knowable episode as zero. A suite that
  rewards refusing would rank this first.
  """

  name = "null:always-abstain"

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    return _null_report(
      self.name,
      request,
      prediction,
      reasons=(AbstainReason.LOW_CONFIDENCE,),
      clamp_magnitude_mm=clamp_magnitude_mm,
    )


class ConstantConfidenceScreen:
  """A real screen fed one fixed confidence. The no-information control.

  Every other condition is checked properly, so this passes any test that only
  exercises staleness or horizons. What it destroys is the one channel the screen
  is supposed to be reading: with the confidence held constant, the model's own
  uncertainty contributes nothing to any verdict, and the screen decides purely on
  the mechanical conditions. A screen that cannot be distinguished from this one
  is not using the model.
  """

  def __init__(self, inner: Screen, confidence: float = 1.0) -> None:
    if not 0.0 <= float(confidence) <= 1.0:
      raise ValueError("confidence must be in [0, 1]")
    self.inner = inner
    self.confidence = float(confidence)
    self.name = f"null:constant-confidence({self.confidence:.2f})"

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    flattened = replace(prediction, confidence=self.confidence, abstain=False, flag=False)
    return self.inner.check(
      flattened,
      request,
      now_s=now_s,
      readout=readout,
      clamped=clamped,
      clamp_magnitude_mm=clamp_magnitude_mm,
      ceiling=ceiling,
    )


class ReasonBlindScreen:
  """Reads the boolean and throws the reason away. The most tempting null.

  It gets every accept/defer decision right, so every aggregate that counts
  admissions matches a real screen exactly. What it loses is the ability to tell a
  stale frame from a clamped command from a model that was simply unsure, and
  those have different remedies: one is a camera problem, one is an envelope
  problem, and one is a model problem. A run screened by this is a run where every
  refusal reads as the model's fault.

  This is not hypothetical. Collapsing refusal into one flag is what caused the
  closest published work to record a model's refusals as missing data.
  """

  name = "null:reason-blind"

  def __init__(self, inner: Screen) -> None:
    self.inner = inner

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    report = self.inner.check(
      prediction,
      request,
      now_s=now_s,
      readout=readout,
      clamped=clamped,
      clamp_magnitude_mm=clamp_magnitude_mm,
      ceiling=ceiling,
    )
    if report.admitted:
      return report
    return replace(report, reasons=(AbstainReason.LOW_CONFIDENCE,))


class ClampOnlyScreen:
  """Confidence replaced by minus the clamp count. The model contributes nothing.

  Admits exactly when the envelope left the command alone. If this reproduces a
  screen's correlation with arm outcomes, then the correlation was never about the
  model's uncertainty: it was about which commands were too large for the
  envelope, and the report has to say so in that sentence rather than in a
  footnote.
  """

  name = "null:clamp-only"

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    reasons = (AbstainReason.CLAMPED,) if clamped else ()
    return _null_report(
      self.name,
      request,
      prediction,
      reasons=reasons,
      clamp_magnitude_mm=clamp_magnitude_mm,
    )


class StaleBlindScreen:
  """A real screen with the frame-age condition removed. Must fail.

  The condition it drops is the one with no visible symptom. A stale prediction is
  well formed, confident, inside the trust horizon and about a bench that has
  moved on, and every other check passes it. This is why ``t_capture_s`` is on the
  request rather than left to a closure: without it this null and a real screen
  are the same code.
  """

  name = "null:stale-blind"

  def __init__(self, inner: Screen) -> None:
    self.inner = inner

  def check(
    self,
    prediction: RolloutPrediction,
    request: RolloutRequest,
    now_s: float,
    readout: Optional[ReadoutResult],
    clamped: bool = False,
    clamp_magnitude_mm: float = 0.0,
    ceiling: bool = False,
  ) -> ScreenReport:
    frame = request.newest_frame()
    fresh = now_s if frame is None else float(frame.t_capture_s)
    report = self.inner.check(
      prediction,
      request,
      now_s=fresh,
      readout=readout,
      clamped=clamped,
      clamp_magnitude_mm=clamp_magnitude_mm,
      ceiling=ceiling,
    )
    if frame is None and not report.admitted:
      # A request with no frame trips STALE in the real screen for being undatable.
      # This null drops that too, or it would not be blind to staleness.
      reasons = _ordered(set(report.reasons) - {AbstainReason.STALE})
      outcome = ScreenOutcome.ABSTAIN if reasons else ScreenOutcome.ACT
      return replace(report, outcome=outcome, reasons=reasons, reobserve=False)
    return report


def _null_report(
  name: str,
  request: RolloutRequest,
  prediction: RolloutPrediction,
  reasons: Tuple[AbstainReason, ...],
  clamp_magnitude_mm: float,
) -> ScreenReport:
  """A verdict shaped like a real one, decided by something that is not a screen."""
  raw = prediction.confidence
  confidence = float(raw) if isinstance(raw, Real) and not isinstance(raw, bool) else float("nan")
  return ScreenReport(
    outcome=ScreenOutcome.ABSTAIN if reasons else ScreenOutcome.ACT,
    reasons=reasons,
    confidence=confidence,
    theta=float("nan"),
    calibrated_on=UNCALIBRATED,
    horizon=int(request.horizon),
    trust_horizon=int(request.horizon),
    frame_age_s=float("nan"),
    max_frame_age_s=float("inf"),
    readout_ok=True,
    readout=name,
    clamp_magnitude_mm=float(clamp_magnitude_mm),
    notes=(f"decided by {name}, which is a null control and not a screen",),
  )


@dataclass(frozen=True)
class ScreenCase:
  """One decision the reference screen and every null are asked to make."""

  label: str
  prediction: RolloutPrediction
  request: RolloutRequest
  now_s: float
  readout: Optional[ReadoutResult] = None
  clamped: bool = False
  clamp_magnitude_mm: float = 0.0
  ceiling: bool = False


@dataclass(frozen=True)
class NullScreenOutcome:
  """What one null did. ``failed`` True is the required result."""

  name: str
  failed: bool
  n_cases: int
  n_disagreed: int
  first_disagreement: str

  def line(self) -> str:
    mark = "ok " if self.failed else "XX "
    detail = (
      f"disagreed on {self.n_disagreed}/{self.n_cases}, first at {self.first_disagreement}"
      if self.failed
      else f"matched the screen on all {self.n_cases} cases"
    )
    return f"  [{mark}] {self.name}: {detail}"


@dataclass(frozen=True)
class NullScreenSuite:
  """Whether every null screen failed, which is the only acceptable outcome.

  ``passed`` requires a non-empty result list, a non-zero case count, and every
  null failing. "Nothing was checked" and "the check passed" stay different
  answers here for the same reason they do in the policy null tests: the flag is
  read on exactly the runs where nothing established it.
  """

  results: Tuple[NullScreenOutcome, ...] = ()
  n_cases: int = 0

  @property
  def passed(self) -> bool:
    if not self.results or self.n_cases <= 0:
      return False
    return all(r.failed for r in self.results)

  def summary(self) -> str:
    if not self.results:
      return "null screens: NOT RUN (nothing was checked, so nothing is established)"
    verdict = "ok" if self.passed else "BROKEN SCREEN"
    lines = [f"null screens: {verdict} over {self.n_cases} cases"]
    lines.extend(r.line() for r in self.results)
    return "\n".join(lines)


def screen_null_check(
  reference: Screen,
  cases: Sequence[ScreenCase],
  nulls: Optional[Sequence[Any]] = None,
) -> NullScreenSuite:
  """Run every null screen against the same cases and require each one to fail.

  A null fails by disagreeing with the reference screen on the pair
  ``(outcome, reason)`` for at least one case. The reason is part of the
  comparison on purpose: a null that gets every accept and defer right while
  reporting the wrong reason has still lost the information the whole reason enum
  exists to carry, and comparing booleans alone would pass it.
  """
  pool = tuple(cases)
  if not pool:
    return NullScreenSuite(results=(), n_cases=0)
  candidates = tuple(nulls) if nulls is not None else default_null_screens(reference)
  results: List[NullScreenOutcome] = []
  for null in candidates:
    disagreed: List[str] = []
    for case in pool:
      want = _verdict(reference, case)
      got = _verdict(null, case)
      if want != got:
        disagreed.append(f"{case.label} (screen {want}, null {got})")
    results.append(
      NullScreenOutcome(
        name=str(getattr(null, "name", type(null).__name__)),
        failed=bool(disagreed),
        n_cases=len(pool),
        n_disagreed=len(disagreed),
        first_disagreement=disagreed[0] if disagreed else "",
      )
    )
  return NullScreenSuite(results=tuple(results), n_cases=len(pool))


def default_null_screens(reference: Screen) -> Tuple[Any, ...]:
  """The six nulls, three of which need the reference screen to sit in front of."""
  return (
    AlwaysAdmitScreen(),
    AlwaysAbstainScreen(),
    ConstantConfidenceScreen(reference, confidence=1.0),
    ReasonBlindScreen(reference),
    ClampOnlyScreen(),
    StaleBlindScreen(reference),
  )


def _verdict(screen: Any, case: ScreenCase) -> str:
  report = screen.check(
    case.prediction,
    case.request,
    now_s=case.now_s,
    readout=case.readout,
    clamped=case.clamped,
    clamp_magnitude_mm=case.clamp_magnitude_mm,
    ceiling=case.ceiling,
  )
  if report.admitted:
    return "ACT"
  reason = report.reason
  return f"ABSTAIN({reason.value})" if reason is not None else "ABSTAIN(?)"


def screen_case(
  label: str,
  confidence: float,
  now_s: float = 1.0,
  t_capture_s: float = 1.0,
  horizon: int = 1,
  abstain: bool = False,
  readout: Optional[ReadoutResult] = None,
  clamped: bool = False,
  ceiling: bool = False,
  episode_id: Optional[str] = None,
) -> ScreenCase:
  """Build one four-field pose case, so a suite can be written in a few lines.

  Deliberately concrete rather than general. The nulls have to be run against real
  :class:`RolloutRequest` and :class:`RolloutPrediction` objects, because half of
  what distinguishes a screen from a null lives on those types: the capture time,
  the horizon, the row invariant tying abstention to NaN.
  """
  frames = (
    FrameRef(
      path=f"{label}.png",
      sha256=f"sha256:{label}",
      t_capture_s=float(t_capture_s),
      clock_source="rendered",
    ),
  )
  request = RolloutRequest(
    task_id="screen-null",
    episode_id=episode_id or label,
    state=np.zeros(4, dtype=float),
    state_fields=("x", "y", "z", "rz"),
    state_units=("mm", "mm", "mm", "rad"),
    frames=frames,
    actions=np.zeros((int(horizon), len(ACTION_FIELDS)), dtype=float),
    horizon=int(horizon),
    tolerance=np.array([2.0, 2.0, 2.0, 0.05], dtype=float),
  )
  if abstain:
    prediction = RolloutPrediction.abstained(int(horizon), 4, readout="abstained")
  else:
    prediction = RolloutPrediction(
      state=np.zeros((int(horizon), 4), dtype=float),
      confidence=float(confidence),
      abstain=False,
      flag=False,
      readout="identity",
      n_forward_passes=1,
      wall_clock_s=0.0,
    )
  return ScreenCase(
    label=label,
    prediction=prediction,
    request=request,
    now_s=float(now_s),
    readout=readout,
    clamped=clamped,
    ceiling=ceiling,
  )


#: Every refusal this module raises. Walked by the reachability test, because a
#: refusal no call can reach is documentation rather than a rule.
REFUSALS: Tuple[type, ...] = (
  ScreenDeclined,
  CeilingRefusal,
  NotCalibrated,
)
