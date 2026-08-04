"""Score what a model earns for the predictions it declined to make.

A benchmark that only scores predictions pays a model to guess. Guessing is
cheap on a bench right up to the moment it is not, so the number that matters
here is not accuracy: it is whether the confidence attached to a prediction
orders the errors, and whether a refusal is credited the same as a correct
answer when the deciding quantity was never visible in the first place.

Four things are refusals of the structural kind, enforced by the types rather
than by a comment somebody can delete:

* :class:`Rate` cannot be built from a float. Every rate in this module carries
  the counts it came from, so its Wilson interval is always available and a
  naked ``0.93`` can never enter a report. A rate without an interval is the
  single most common way a benchmark overstates itself, and thirteen surveyed
  real-robot papers report none at all.
* Below :data:`MIN_N_REPORTABLE` observations nothing prints. Not behind a flag,
  not with an asterisk: :meth:`SelectiveReport.format` emits ``UNDERPOWERED
  n=<k>`` and no number at all. There is deliberately no parameter that turns
  this off, because a parameter that turns it off is a parameter that gets used.
* :class:`AbstentionCredit` has no attribute named ``acr``. The abstention
  credit rate is only reachable through :meth:`AbstentionCredit.credit_and_false_alarm`,
  which hands back the false-alarm rate in the same tuple. A model that abstains
  on everything scores a respectable ACR, and quoting that alone is unfalsifiable.
* :class:`FrozenThreshold` refuses an undeclared calibration source. A threshold
  read off the data being scored measures the answer key, not the model.

``E_AURC = AURC - AURC_oracle`` is the only ranking number, and the oracle and
random bands print in the same row so the band a score sits in is never a second
lookup. A constant-confidence model does not land at zero; it lands at
``AURC_random - AURC_oracle``, and reading that gap as "close to perfect" is the
exact mistake this layout exists to prevent.

Nothing here imports numpy, and that is not an accident of style.
``scripts/score_standalone.py`` is a copy of this scoring core that has to run on
a bare ``python:3.9-slim`` with zero installs, so an outside submitter can check
a result file without taking on a dependency tree to do it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

#: Two-sided normal deviate for a 95% interval. An absolute constant: a
#: confidence level chosen after seeing the interval is not a confidence level.
WILSON_Z = 1.96

#: Below this many observations the harness prints no number under any flag.
MIN_N_REPORTABLE = 30

#: Below this many observations a number prints, stamped PROVISIONAL, and may
#: not enter a leaderboard row or a delta.
MIN_N_RANKABLE = 200

TIER_UNDERPOWERED = "UNDERPOWERED"
TIER_PROVISIONAL = "PROVISIONAL"
TIER_RANKABLE = "RANKABLE"

#: Risk levels at which coverage is reported. Absolute constants, not read off
#: the curve being scored.
RISK_LEVELS = (0.05, 0.10, 0.20)

#: Cost of an incorrect act, in units of one correct act. All three print. The
#: middle one is the headline and is a stipulated preference, not a measurement:
#: it encodes "a failed motion costs a reset and can damage labware, an
#: abstention costs one cycle". Someone with different labware should say so and
#: read a different column rather than argue with this one.
W_FAIL_WEIGHTS = (-1.0, -4.0, -10.0)
HEADLINE_W_FAIL = -4.0

BOOTSTRAP_RESAMPLES = 10000
PERMUTATION_RESAMPLES = 1000

#: Two-sample KS falls back to the asymptotic p-value at or above this size.
#: Below it the exact lattice-path count is cheap and the asymptotic form is
#: badly wrong, which matters because small n is where this benchmark lives.
KS_EXACT_MAX_N = 35

#: Abstention metrics are refused outside this band because outside it they are
#: a statement about the base rate rather than about the model.
UNKNOWABLE_BAND = (0.05, 0.40)

#: Below this absolute rank correlation the cell prints NOT-INFORMATIVE rather
#: than a number, because a rho of 0.03 quoted to two decimals reads as evidence.
INFORMATIVE_RHO = 0.1

#: A confidence ordering must beat its own shuffle at this level to claim it
#: carries information. Failing to is not a low score, it is no score.
NO_INFORMATION_P = 0.05

_EPS = 1e-12


class MetricRefused(ValueError):
  """Raised when a number exists arithmetically but would not be a measurement."""


class ThresholdNotFrozen(ValueError):
  """Raised when an act/defer threshold has no declared calibration source."""


class AbstentionBandRefused(MetricRefused):
  """Raised when the unknowable fraction puts abstention metrics outside their band."""


# -- tiers ---------------------------------------------------------------------


def tier_of(n: int) -> str:
  """Which of the three reporting tiers ``n`` observations buy.

  Two cut points rather than one. A single floor at 30 lets a 31-episode row sit
  in a league table next to a 400-episode row; a single floor at 200 throws away
  everything a small pilot legitimately shows. Below 30 nothing is a
  measurement, between 30 and 200 something is visible but not comparable.
  """
  if n < MIN_N_REPORTABLE:
    return TIER_UNDERPOWERED
  if n < MIN_N_RANKABLE:
    return TIER_PROVISIONAL
  return TIER_RANKABLE


def underpowered_label(n: int) -> str:
  """The only thing an underpowered quantity is allowed to say about itself."""
  return f"UNDERPOWERED n={n}"


# -- rates ---------------------------------------------------------------------


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z) -> Tuple[float, float]:
  """Wilson score interval for a binomial proportion, clamped to [0, 1].

  Wilson rather than the textbook normal interval because the normal interval
  gives ``[0.0, 0.0]`` at zero successes and ``[1.0, 1.0]`` at n successes.
  Those are the two results a benchmark is most likely to hit and the two it is
  least entitled to state without width. Wilson stays inside the unit interval
  and keeps its width there: 0/30 reads ``[0.000, 0.114]``, which is the honest
  version of "we saw none".
  """
  if trials <= 0:
    raise ValueError("wilson_interval needs at least one trial")
  if not 0 <= successes <= trials:
    raise ValueError(f"{successes} successes out of {trials} trials is not a proportion")
  n = float(trials)
  k = float(successes)
  z2 = z * z
  denominator = n + z2
  center = (k + z2 / 2.0) / denominator
  half = (z / denominator) * math.sqrt(k * (n - k) / n + z2 / 4.0)
  low = center - half
  high = center + half
  return (0.0 if low < 0.0 else low, 1.0 if high > 1.0 else high)


@dataclass(frozen=True)
class Rate:
  """A count over a denominator. Never a float, so never an interval-free number.

  Constructing this from a proportion is refused on purpose. Once a rate is a
  bare float the counts behind it are gone, the interval cannot be recovered,
  and the number travels into a table with nothing attached to it. Keeping the
  numerator and denominator means the interval and the reporting tier are always
  derivable at the point of printing.
  """

  numerator: int
  denominator: int
  name: str = "rate"

  def __post_init__(self) -> None:
    for label, value in (("numerator", self.numerator), ("denominator", self.denominator)):
      if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(
          f"Rate.{label} must be an integer count, got {type(value).__name__}. A rate "
          "built from a proportion has already lost the counts its interval needs."
        )
    if self.denominator <= 0:
      raise ValueError("Rate needs a positive denominator; an empty set has no rate")
    if not 0 <= self.numerator <= self.denominator:
      raise ValueError(f"Rate numerator {self.numerator} is not within 0..{self.denominator}")

  @property
  def n(self) -> int:
    return self.denominator

  @property
  def tier(self) -> str:
    return tier_of(self.denominator)

  @property
  def rankable(self) -> bool:
    return self.tier == TIER_RANKABLE

  @property
  def value(self) -> Optional[float]:
    """The proportion, or ``None`` below :data:`MIN_N_REPORTABLE`.

    ``None`` rather than the number, because the caller that wanted the number
    anyway is exactly the caller this tier exists to stop.
    """
    if self.tier == TIER_UNDERPOWERED:
      return None
    return self.numerator / self.denominator

  @property
  def interval(self) -> Optional[Tuple[float, float]]:
    if self.tier == TIER_UNDERPOWERED:
      return None
    return wilson_interval(self.numerator, self.denominator)

  def format(self) -> str:
    if self.tier == TIER_UNDERPOWERED:
      return f"{self.name}: {underpowered_label(self.denominator)}"
    value = self.numerator / self.denominator
    low, high = wilson_interval(self.numerator, self.denominator)
    stamp = f" {TIER_PROVISIONAL}" if self.tier == TIER_PROVISIONAL else ""
    return (
      f"{self.name}: {value:.4f} [{low:.4f}, {high:.4f}] 95% Wilson "
      f"({self.numerator}/{self.denominator}){stamp}"
    )

  def to_dict(self) -> Dict[str, Any]:
    interval = self.interval
    return {
      "name": self.name,
      "numerator": self.numerator,
      "denominator": self.denominator,
      "value": self.value,
      "low": None if interval is None else interval[0],
      "high": None if interval is None else interval[1],
      "tier": self.tier,
    }


# -- risk and coverage ---------------------------------------------------------


def _tie_break_order(n: int, seed: int) -> List[int]:
  """A recorded order for episodes that share a confidence.

  Ties are common and their resolution is not neutral: sorting ties by episode
  id sorts them by whatever the generator happened to emit first, which on these
  fixtures correlates with the planted parameters. A seeded shuffle is recorded
  on the curve so the ordering is reproducible and is nobody's choice.
  """
  order = list(range(n))
  random.Random(seed).shuffle(order)
  return order


@dataclass(frozen=True)
class RiskCoverageCurve:
  """Losses ordered by descending confidence, and the four numbers read off them.

  This is the mathematical object, not the publishable claim. It answers at any
  n because the arithmetic is defined at any n; whether the answer may be
  printed is :class:`SelectiveReport`'s decision, and it applies the tier.
  """

  losses_by_rank: Tuple[int, ...]
  order: Tuple[int, ...]
  seed: int

  @classmethod
  def from_outcomes(
    cls,
    hits: Sequence[int],
    confidences: Sequence[float],
    seed: int = 0,
  ) -> "RiskCoverageCurve":
    if len(hits) != len(confidences):
      raise ValueError(f"{len(hits)} hits against {len(confidences)} confidences")
    n = len(hits)
    if n == 0:
      raise ValueError("a risk-coverage curve over zero episodes is not a curve")
    losses: List[int] = []
    for hit in hits:
      value = int(bool(hit))
      losses.append(1 - value)
    scores: List[float] = []
    for c in confidences:
      score = float(c)
      if not math.isfinite(score):
        raise ValueError("confidence must be finite to be ranked; an abstention has none")
      if not 0.0 <= score <= 1.0:
        raise ValueError(f"confidence {score} is outside [0, 1]")
      scores.append(score)
    tie_rank = [0] * n
    for position, index in enumerate(_tie_break_order(n, seed)):
      tie_rank[index] = position
    order = sorted(range(n), key=lambda i: (-scores[i], tie_rank[i]))
    return cls(
      losses_by_rank=tuple(losses[i] for i in order),
      order=tuple(order),
      seed=seed,
    )

  @property
  def n(self) -> int:
    return len(self.losses_by_rank)

  @property
  def n_errors(self) -> int:
    return sum(self.losses_by_rank)

  def coverage(self, m: int) -> float:
    """Fraction of episodes accepted at rank cut ``m``."""
    if not 0 <= m <= self.n:
      raise ValueError(f"coverage needs 0 <= m <= {self.n}, got {m}")
    return m / self.n

  def risk(self, m: int) -> Optional[float]:
    """Error rate among the ``m`` most confident episodes, ``None`` at m = 0.

    ``None`` and never ``0.0``. Zero risk at zero coverage is arithmetically
    true and is the most flattering lie this curve can tell: a model that
    answers nothing has not achieved a perfect selective risk, it has produced
    no evidence. Refusal rule R1.
    """
    if not 0 <= m <= self.n:
      raise ValueError(f"risk needs 0 <= m <= {self.n}, got {m}")
    if m == 0:
      return None
    return sum(self.losses_by_rank[:m]) / m

  @property
  def aurc(self) -> float:
    """Mean selective risk over every coverage level."""
    total = 0.0
    cumulative = 0
    for m in range(1, self.n + 1):
      cumulative += self.losses_by_rank[m - 1]
      total += cumulative / m
    return total / self.n

  @property
  def aurc_oracle(self) -> float:
    """AURC of the best ordering that exists for this error count.

    Not zero, and this is the whole point of reporting it. An accurate model
    gets a low AURC because it makes few errors, not because its confidence
    ranks them, and subtracting the oracle is what separates the two.
    """
    correct = self.n - self.n_errors
    total = 0.0
    for m in range(1, self.n + 1):
      excess = m - correct
      if excess > 0:
        total += excess / m
    return total / self.n

  @property
  def aurc_random(self) -> float:
    """AURC expected from a confidence that carries no information at all."""
    return self.n_errors / self.n

  @property
  def e_aurc(self) -> float:
    """The headline. Lower is better; the no-information band is above it.

    A constant-confidence model lands near ``aurc_random - aurc_oracle``, not
    near zero, so a small E-AURC is only meaningful against that band. Both
    bands print in the same row for exactly this reason.
    """
    return self.aurc - self.aurc_oracle

  def coverage_at_risk(self, r: float) -> Optional[Rate]:
    """Largest coverage whose selective risk stays at or below ``r``.

    This reads a cut point off the curve, which is legitimate here and is not
    legitimate for the act/defer threshold. The difference is what the cut point
    is used for: this one summarises a curve that is already published, while
    a decision threshold fitted to the scored data would let the model be judged
    against a rule chosen after seeing the answers. The decision threshold is a
    :class:`FrozenThreshold` and comes from a declared calibration split.

    ``None`` when no coverage achieves the level, which is a real answer and is
    not the same as a coverage of zero.
    """
    if not 0.0 <= r <= 1.0:
      raise ValueError(f"risk level {r} is outside [0, 1]")
    best = 0
    cumulative = 0
    for m in range(1, self.n + 1):
      cumulative += self.losses_by_rank[m - 1]
      if cumulative <= r * m + _EPS:
        best = m
    if best == 0:
      return None
    return Rate(best, self.n, name=f"coverage@risk<={r:.2f}")


# -- act and defer -------------------------------------------------------------


@dataclass(frozen=True)
class FrozenThreshold:
  """An act/defer threshold plus the split it was fitted on, printed verbatim.

  The calibration source is required rather than optional because a threshold
  with no stated origin is indistinguishable from one tuned against the scored
  episodes, and tuning it there turns the act/defer table into a statement about
  the answer key. Published work finds threshold selection matters more than the
  choice of uncertainty method, so the threshold is part of the result and gets
  reported like one.
  """

  value: float
  calibrated_on: str
  n_calibration: int

  def __post_init__(self) -> None:
    if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
      raise ThresholdNotFrozen("threshold value must be a number in [0, 1]")
    if not math.isfinite(float(self.value)) or not 0.0 <= float(self.value) <= 1.0:
      raise ThresholdNotFrozen(f"threshold {self.value} is outside [0, 1]")
    source = str(self.calibrated_on).strip()
    if not source or source.upper() == "UNDECLARED":
      raise ThresholdNotFrozen(
        "a threshold needs a declared calibration source. An undeclared threshold "
        "cannot be distinguished from one fitted to the episodes being scored, and "
        "that fit measures the answer key rather than the model."
      )
    if isinstance(self.n_calibration, bool) or not isinstance(self.n_calibration, int):
      raise ThresholdNotFrozen("n_calibration must be an integer count")
    if self.n_calibration < 1:
      raise ThresholdNotFrozen("a threshold fitted on nothing is not calibrated")

  def describe(self) -> str:
    return f"theta={float(self.value):.4f} calibrated_on={self.calibrated_on!r} " \
           f"(n={self.n_calibration})"


@dataclass(frozen=True)
class ActDeferTable:
  """The full act/defer 2x2, which is the only form these counts are released in.

  ADA is a property here and never a return value on its own. Accepted-decision
  accuracy is high both for a model that acts well and for a model that defers
  on everything hard, and those are different systems. The two off-diagonal
  rates separate them, so all four cells travel together.
  """

  n_act_hit: int
  n_act_miss: int
  n_defer_hit: int
  n_defer_miss: int
  threshold: FrozenThreshold

  @property
  def n(self) -> int:
    return self.n_act_hit + self.n_act_miss + self.n_defer_hit + self.n_defer_miss

  @property
  def ada(self) -> Rate:
    return Rate(self.n_act_hit + self.n_defer_miss, self.n, name="ADA")

  @property
  def unsafe_act_rate(self) -> Rate:
    return Rate(self.n_act_miss, self.n, name="unsafe_act_rate")

  @property
  def wasted_defer_rate(self) -> Rate:
    return Rate(self.n_defer_hit, self.n, name="wasted_defer_rate")

  def format(self) -> str:
    lines = [
      f"act/defer at {self.threshold.describe()}",
      f"  act   & hit {self.n_act_hit:5d} | act   & miss {self.n_act_miss:5d}",
      f"  defer & hit {self.n_defer_hit:5d} | defer & miss {self.n_defer_miss:5d}",
      f"  {self.ada.format()}",
      f"  {self.unsafe_act_rate.format()}",
      f"  {self.wasted_defer_rate.format()}",
    ]
    return "\n".join(lines)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "n_act_hit": self.n_act_hit,
      "n_act_miss": self.n_act_miss,
      "n_defer_hit": self.n_defer_hit,
      "n_defer_miss": self.n_defer_miss,
      "threshold": {
        "value": float(self.threshold.value),
        "calibrated_on": self.threshold.calibrated_on,
        "n_calibration": self.threshold.n_calibration,
      },
      "ADA": self.ada.to_dict(),
      "unsafe_act_rate": self.unsafe_act_rate.to_dict(),
      "wasted_defer_rate": self.wasted_defer_rate.to_dict(),
    }


# -- abstention credit ---------------------------------------------------------


@dataclass(frozen=True)
class AbstentionCredit:
  """Counts behind the abstention metrics. There is no attribute named ``acr``.

  Refusal rule R5 says ACR never prints without FAR, and a comment saying so is
  not a rule. So the credit rate is unreachable except through
  :meth:`credit_and_false_alarm`, which returns both, and refusal recall is
  unreachable except through :meth:`recall_and_precision`. A model that abstains
  on everything scores an ACR equal to the unknowable base rate, which looks
  respectable in isolation and is fully explained by an FAR of 1.0.
  """

  n: int
  n_unknowable: int
  n_abstained: int
  n_unknowable_abstained: int
  n_knowable_abstained: int
  n_credited: int

  @property
  def unknowable_fraction(self) -> float:
    return self.n_unknowable / self.n

  @property
  def in_band(self) -> bool:
    low, high = UNKNOWABLE_BAND
    return low <= self.unknowable_fraction <= high

  def _require_band(self) -> None:
    if not self.in_band:
      low, high = UNKNOWABLE_BAND
      raise AbstentionBandRefused(
        f"unknowable fraction {self.unknowable_fraction:.3f} is outside "
        f"[{low:.2f}, {high:.2f}], so these numbers are dominated by the base rate "
        "rather than by the model and are refused. Refusal rule R6."
      )

  def credit_and_false_alarm(self) -> Tuple[Rate, Rate]:
    """ACR and FAR, together, because ACR alone is unfalsifiable."""
    self._require_band()
    n_knowable = self.n - self.n_unknowable
    acr = Rate(self.n_credited, self.n, name="ACR")
    far = Rate(self.n_knowable_abstained, n_knowable, name="FAR")
    return (acr, far)

  def recall_and_precision(self) -> Tuple[Rate, Optional[Rate]]:
    """RefusalRecall and RefusalPrecision, together, for the same reason.

    Precision is ``None`` when the model never abstained, which is a real state
    and not a zero: there is no set of refusals to be right about. It still
    travels in the tuple so recall is never handed over unpaired.
    """
    self._require_band()
    recall = Rate(self.n_unknowable_abstained, self.n_unknowable, name="RefusalRecall")
    if self.n_abstained == 0:
      return (recall, None)
    precision = Rate(
      self.n_unknowable_abstained, self.n_abstained, name="RefusalPrecision"
    )
    return (recall, precision)

  def format(self) -> str:
    header = (
      f"abstention credit (unknowable fraction {self.unknowable_fraction:.3f}, "
      f"band [{UNKNOWABLE_BAND[0]:.2f}, {UNKNOWABLE_BAND[1]:.2f}])"
    )
    if not self.in_band:
      return header + "\n  REFUSED: outside the band these measure the base rate (R6)"
    acr, far = self.credit_and_false_alarm()
    recall, precision = self.recall_and_precision()
    lines = [header, f"  {acr.format()}", f"  {far.format()}", f"  {recall.format()}"]
    if precision is None:
      lines.append("  RefusalPrecision: undefined, the model never abstained")
    else:
      lines.append(f"  {precision.format()}")
    return "\n".join(lines)

  def to_dict(self) -> Dict[str, Any]:
    out: Dict[str, Any] = {
      "n": self.n,
      "n_unknowable": self.n_unknowable,
      "n_abstained": self.n_abstained,
      "unknowable_fraction": self.unknowable_fraction,
      "in_band": self.in_band,
    }
    if not self.in_band:
      out["refused"] = "R6: unknowable fraction outside band"
      return out
    acr, far = self.credit_and_false_alarm()
    recall, precision = self.recall_and_precision()
    out["ACR"] = acr.to_dict()
    out["FAR"] = far.to_dict()
    out["RefusalRecall"] = recall.to_dict()
    out["RefusalPrecision"] = None if precision is None else precision.to_dict()
    return out


# -- decision utility ----------------------------------------------------------


@dataclass(frozen=True)
class UtilityAtWeight:
  """Expected utility per episode at one stipulated failure cost."""

  w_fail: float
  n: int
  n_correct_act: int
  n_incorrect_act: int
  n_defer: int

  @property
  def total(self) -> float:
    return self.n_correct_act * 1.0 + self.n_incorrect_act * self.w_fail

  @property
  def u_over_n(self) -> float:
    return self.total / self.n

  def format(self) -> str:
    tag = " (headline, a stipulated preference, not a measurement)"
    suffix = tag if self.w_fail == HEADLINE_W_FAIL else ""
    return f"U/N at w_fail={self.w_fail:+.0f}: {self.u_over_n:+.4f}{suffix}"

  def to_dict(self) -> Dict[str, Any]:
    return {
      "w_fail": self.w_fail,
      "u_over_n": self.u_over_n,
      "n_correct_act": self.n_correct_act,
      "n_incorrect_act": self.n_incorrect_act,
      "n_defer": self.n_defer,
      "headline": self.w_fail == HEADLINE_W_FAIL,
    }


def decision_utility(table: ActDeferTable) -> Tuple[UtilityAtWeight, ...]:
  """Utility per episode at every weight in :data:`W_FAIL_WEIGHTS`.

  All three print. Reporting only the headline weight hides that the ordering
  between two models can flip with the cost of a failure, and the flip is the
  interesting part: it is the difference between a model that is accurate and a
  model that knows when it is not.
  """
  n_defer = table.n_defer_hit + table.n_defer_miss
  return tuple(
    UtilityAtWeight(
      w_fail=w,
      n=table.n,
      n_correct_act=table.n_act_hit,
      n_incorrect_act=table.n_act_miss,
      n_defer=n_defer,
    )
    for w in W_FAIL_WEIGHTS
  )


# -- statistics ----------------------------------------------------------------


def _mean(values: Sequence[float]) -> float:
  return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
  """Median, kept because the mean of a heavy-tailed score is not a summary."""
  ordered = sorted(values)
  n = len(ordered)
  middle = n // 2
  if n % 2 == 1:
    return ordered[middle]
  return (ordered[middle - 1] + ordered[middle]) / 2.0


def _mean_difference(a: Sequence[float], b: Sequence[float]) -> float:
  return _mean(a) - _mean(b)


@dataclass(frozen=True)
class BootstrapCI:
  """A paired bootstrap interval, with the seed that produced it."""

  point: float
  low: float
  high: float
  n_resamples: int
  seed: int

  @property
  def includes_zero(self) -> bool:
    return self.low <= 0.0 <= self.high

  def format(self) -> str:
    verdict = " INCLUDES ZERO" if self.includes_zero else ""
    return (
      f"delta {self.point:+.4f} [{self.low:+.4f}, {self.high:+.4f}] 95% paired "
      f"bootstrap, {self.n_resamples} resamples, seed {self.seed}{verdict}"
    )

  def to_dict(self) -> Dict[str, Any]:
    return {
      "point": self.point,
      "low": self.low,
      "high": self.high,
      "n_resamples": self.n_resamples,
      "seed": self.seed,
      "includes_zero": self.includes_zero,
    }


def paired_bootstrap_ci(
  a: Sequence[float],
  b: Sequence[float],
  n_resamples: int = BOOTSTRAP_RESAMPLES,
  seed: int = 0,
  statistic: Optional[Callable[[Sequence[float], Sequence[float]], float]] = None,
) -> BootstrapCI:
  """Resample a shared episode index and report the interval on the difference.

  Shared, not independent: the two models saw the same episodes, and resampling
  them separately throws away the pairing and widens the interval by exactly the
  between-episode variance the comparison is trying to remove. The seed is
  recorded on the result so a reported interval can be regenerated.

  The interval is the 2.5% and 97.5% points of the resample distribution. That
  is a summary of a distribution this function generated, not a cut point read
  off the data being scored, which is the operation refused elsewhere in this
  module.
  """
  if len(a) != len(b):
    raise ValueError(f"paired bootstrap needs equal lengths, got {len(a)} and {len(b)}")
  n = len(a)
  if n == 0:
    raise ValueError("paired bootstrap over zero episodes is not a comparison")
  if n_resamples < 1:
    raise ValueError("paired bootstrap needs at least one resample")
  stat = statistic if statistic is not None else _mean_difference
  values_a = [float(x) for x in a]
  values_b = [float(x) for x in b]
  point = float(stat(values_a, values_b))
  rng = random.Random(seed)
  population = range(n)
  draws: List[float] = []
  for _ in range(n_resamples):
    picks = rng.choices(population, k=n)
    draws.append(float(stat([values_a[i] for i in picks], [values_b[i] for i in picks])))
  draws.sort()
  low_index = int(math.floor(0.025 * n_resamples))
  high_index = int(math.ceil(0.975 * n_resamples)) - 1
  if high_index < low_index:
    high_index = low_index
  if high_index > n_resamples - 1:
    high_index = n_resamples - 1
  return BootstrapCI(
    point=point,
    low=draws[low_index],
    high=draws[high_index],
    n_resamples=n_resamples,
    seed=seed,
  )


@dataclass(frozen=True)
class PermutationResult:
  """The confidence-shuffle control: does the pairing carry anything at all."""

  observed: float
  p_value: float
  n_permutations: int
  seed: int
  null_median: float

  @property
  def no_information(self) -> bool:
    return self.p_value >= NO_INFORMATION_P

  def format(self) -> str:
    verdict = " NO INFORMATION" if self.no_information else ""
    return (
      f"confidence shuffle: p={self.p_value:.4f} over {self.n_permutations} "
      f"permutations, null median E-AURC {self.null_median:.4f}, seed {self.seed}"
      f"{verdict}"
    )

  def to_dict(self) -> Dict[str, Any]:
    return {
      "observed_e_aurc": self.observed,
      "p_value": self.p_value,
      "n_permutations": self.n_permutations,
      "seed": self.seed,
      "null_median_e_aurc": self.null_median,
      "no_information": self.no_information,
    }


def confidence_shuffle_p(
  hits: Sequence[int],
  confidences: Sequence[float],
  seed: int = 0,
  n_permutations: int = PERMUTATION_RESAMPLES,
) -> PermutationResult:
  """Permute the confidences, keep the errors, and ask how often that does better.

  The control holds the score distribution and the error count exactly fixed and
  destroys only the pairing between them, which isolates the one claim the
  confidence channel makes. Nothing that is an artifact of a low base error rate
  survives it, because the shuffled runs have the same base error rate.

  The p-value counts the permutation itself, ``(1 + better) / (1 + n)``, so a
  perfect ordering reports ``1/1001`` rather than a zero it did not earn.
  """
  observed_curve = RiskCoverageCurve.from_outcomes(hits, confidences, seed=seed)
  observed = observed_curve.e_aurc
  rng = random.Random(seed + 1)
  shuffled = [float(c) for c in confidences]
  at_least_as_good = 0
  null_values: List[float] = []
  for _ in range(n_permutations):
    rng.shuffle(shuffled)
    value = RiskCoverageCurve.from_outcomes(hits, shuffled, seed=seed).e_aurc
    null_values.append(value)
    if value <= observed + _EPS:
      at_least_as_good += 1
  return PermutationResult(
    observed=observed,
    p_value=(1 + at_least_as_good) / (1 + n_permutations),
    n_permutations=n_permutations,
    seed=seed,
    null_median=_median(null_values),
  )


def _average_ranks(values: Sequence[float]) -> List[float]:
  """Ranks with ties averaged, which binary error columns are made of."""
  n = len(values)
  order = sorted(range(n), key=lambda i: values[i])
  ranks = [0.0] * n
  start = 0
  while start < n:
    stop = start + 1
    while stop < n and values[order[stop]] == values[order[start]]:
      stop += 1
    shared = (start + stop - 1) / 2.0
    for position in range(start, stop):
      ranks[order[position]] = shared
    start = stop
  return ranks


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
  """Rank correlation, or ``None`` when one column never varies.

  ``None`` rather than zero: a constant confidence has no rank ordering to
  correlate, and reporting 0.0 there implies a measurement was made. Error is a
  binary column here, so ties are the normal case and ranks are averaged.
  """
  if len(x) != len(y):
    raise ValueError(f"spearman needs equal lengths, got {len(x)} and {len(y)}")
  n = len(x)
  if n < 2:
    return None
  rx = _average_ranks([float(v) for v in x])
  ry = _average_ranks([float(v) for v in y])
  mx = _mean(rx)
  my = _mean(ry)
  sxy = 0.0
  sxx = 0.0
  syy = 0.0
  for i in range(n):
    dx = rx[i] - mx
    dy = ry[i] - my
    sxy += dx * dy
    sxx += dx * dx
    syy += dy * dy
  if sxx <= _EPS or syy <= _EPS:
    return None
  return sxy / math.sqrt(sxx * syy)


@dataclass(frozen=True)
class KSResult:
  """Two-sample Kolmogorov-Smirnov statistic and how its p-value was obtained."""

  statistic: float
  p_value: float
  n_a: int
  n_b: int
  exact: bool

  def format(self) -> str:
    kind = "exact" if self.exact else "asymptotic"
    return (
      f"KS D={self.statistic:.4f} p={self.p_value:.4f} ({kind}, "
      f"n={self.n_a} vs {self.n_b})"
    )

  def to_dict(self) -> Dict[str, Any]:
    return {
      "statistic": self.statistic,
      "p_value": self.p_value,
      "n_a": self.n_a,
      "n_b": self.n_b,
      "exact": self.exact,
    }


def _ks_paths_within(n: int, m: int, threshold: int) -> int:
  """Lattice paths from (0,0) to (n,m) that never reach the KS band edge.

  Integer arithmetic throughout: the deviation at a lattice point is
  ``|i*m - j*n| / (n*m)``, so the comparison is exact and the path count is an
  exact Python integer. Floating point here would silently misclassify the
  boundary paths, which are the only ones the p-value depends on.
  """
  if threshold <= 0:
    return 0
  counts = [[0] * (m + 1) for _ in range(n + 1)]
  counts[0][0] = 1
  for i in range(n + 1):
    for j in range(m + 1):
      if i == 0 and j == 0:
        continue
      if abs(i * m - j * n) >= threshold:
        continue
      total = 0
      if i > 0:
        total += counts[i - 1][j]
      if j > 0:
        total += counts[i][j - 1]
      counts[i][j] = total
  return counts[n][m]


def _ks_asymptotic_p(statistic: float, n: int, m: int) -> float:
  effective = n * m / (n + m)
  root = math.sqrt(effective)
  scaled = (root + 0.12 + 0.11 / root) * statistic
  if scaled <= 0.0:
    return 1.0
  total = 0.0
  for k in range(1, 101):
    term = 2.0 * ((-1.0) ** (k - 1)) * math.exp(-2.0 * k * k * scaled * scaled)
    total += term
  if total < 0.0:
    return 0.0
  return 1.0 if total > 1.0 else total


def ks_two_sample(a: Sequence[float], b: Sequence[float]) -> KSResult:
  """Two-sample KS with an exact p-value below :data:`KS_EXACT_MAX_N`.

  The asymptotic form is what every library reaches for and it is wrong at the
  sizes this benchmark actually runs at: the modal per-condition trial count in
  the surveyed real-robot literature is ten to twenty. At those sizes the exact
  count costs microseconds, so there is no reason to accept the approximation.
  """
  n = len(a)
  m = len(b)
  if n == 0 or m == 0:
    raise ValueError("two-sample KS needs both samples non-empty")
  sorted_a = sorted(float(v) for v in a)
  sorted_b = sorted(float(v) for v in b)
  cut_points = sorted(set(sorted_a) | set(sorted_b))
  statistic = 0.0
  i = 0
  j = 0
  for value in cut_points:
    while i < n and sorted_a[i] <= value:
      i += 1
    while j < m and sorted_b[j] <= value:
      j += 1
    gap = abs(i / n - j / m)
    if gap > statistic:
      statistic = gap
  exact = n < KS_EXACT_MAX_N and m < KS_EXACT_MAX_N
  if exact:
    threshold = int(round(statistic * n * m))
    within = _ks_paths_within(n, m, threshold)
    total_paths = math.comb(n + m, n)
    p_value = 1.0 - within / total_paths
    if p_value < 0.0:
      p_value = 0.0
    if p_value > 1.0:
      p_value = 1.0
  else:
    p_value = _ks_asymptotic_p(statistic, n, m)
  return KSResult(statistic=statistic, p_value=p_value, n_a=n, n_b=m, exact=exact)


# -- the scorecard -------------------------------------------------------------


@dataclass(frozen=True)
class EpisodeOutcome:
  """One scored episode. Abstaining and predicting are mutually exclusive here.

  ``abstain`` is true if and only if ``confidence`` is ``None``, checked row for
  row. Abstaining while banking a prediction is an open gaming channel: the
  submission collects the refusal credit and the prediction is still there to be
  scored if it happens to be right. Flagging a low-trust prediction is a
  different act and keeps its confidence.
  """

  episode_id: str
  hit: bool
  confidence: Optional[float] = None
  abstain: bool = False
  unknowable: bool = False

  def __post_init__(self) -> None:
    if self.abstain and self.confidence is not None:
      raise ValueError(
        f"episode {self.episode_id!r} abstained but carries a confidence. An "
        "abstention with a banked prediction collects the refusal credit and keeps "
        "the prediction, which is two payouts for one act."
      )
    if not self.abstain:
      if self.confidence is None:
        raise ValueError(f"episode {self.episode_id!r} predicted but has no confidence")
      value = float(self.confidence)
      if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"episode {self.episode_id!r} confidence {value} is outside [0, 1]")


@dataclass(frozen=True)
class SelectiveReport:
  """Everything the harness is entitled to say about one task at one horizon.

  This is the layer that applies the reporting tier. The curve underneath
  answers at any n because arithmetic does; below :data:`MIN_N_REPORTABLE` this
  object prints ``UNDERPOWERED n=<k>`` and nothing else, and there is no
  argument to :func:`score` that changes that.
  """

  task_id: str
  horizon: int
  seed: int
  n: int
  n_ranked: int
  n_abstained: int
  curve: Optional[RiskCoverageCurve]
  coverage_at_risk: Tuple[Tuple[float, Optional[Rate]], ...]
  table: ActDeferTable
  credit: AbstentionCredit
  utilities: Tuple[UtilityAtWeight, ...]
  permutation: Optional[PermutationResult]
  rho: Optional[float]

  @property
  def tier(self) -> str:
    """The tier the scored episode count buys, which governs every rate here."""
    return tier_of(self.n)

  @property
  def curve_tier(self) -> str:
    """The tier the ranked episode count buys, which governs the curve numbers.

    Kept separate from :attr:`tier` because a model that abstains often has
    plenty of episodes for its abstention rates and too few ranked ones for a
    risk-coverage curve. Collapsing the two would either suppress rates that are
    well powered or publish a curve that is not.
    """
    return tier_of(self.n_ranked)

  @property
  def rankable(self) -> bool:
    return (
      self.tier == TIER_RANKABLE
      and self.curve_tier == TIER_RANKABLE
      and not self.no_information
    )

  @property
  def shuffle_control_ran(self) -> bool:
    return self.permutation is not None

  @property
  def no_information(self) -> bool:
    """True unless the confidence ordering was shown to carry information.

    Four states collapse into one refusal on purpose: nothing was ranked, too
    little was ranked for the control to mean anything, the control was not run,
    or it ran and the ordering did not beat its own shuffle. Every one of them is
    the same claim about what has been demonstrated, which is nothing. Treating
    "the control did not run" as a pass is how an unfalsifiable row gets ranked.
    """
    if self.curve is None or self.curve_tier == TIER_UNDERPOWERED:
      return True
    if self.permutation is None:
      return True
    return self.permutation.no_information

  @property
  def valid(self) -> bool:
    return self.tier != TIER_UNDERPOWERED and not self.no_information

  def information_cell(self) -> Optional[str]:
    """The reason no information claim is being made, or ``None`` if one is."""
    if self.curve is None:
      return "NO INFORMATION: nothing was ranked, so no ordering was demonstrated"
    if self.curve_tier == TIER_UNDERPOWERED:
      return f"NO INFORMATION: only {self.n_ranked} episodes were ranked"
    if self.permutation is None:
      return "NO INFORMATION: the confidence shuffle control was not run"
    if self.permutation.no_information:
      return "NO INFORMATION: the confidence ordering did not beat its own shuffle"
    return None

  def rho_cell(self) -> str:
    """Rank correlation, or the reason there is no number to print."""
    if self.rho is None:
      return "spearman rho (confidence vs error): undefined, confidence never varied"
    if abs(self.rho) < INFORMATIVE_RHO:
      return "spearman rho (confidence vs error): NOT-INFORMATIVE"
    return f"spearman rho (confidence vs error): {self.rho:+.4f}"

  def headline(self) -> str:
    """E-AURC with both bands in the same row, always."""
    if self.curve is None:
      return "E-AURC: no ranked episodes, the risk-coverage curve is undefined"
    if self.curve_tier == TIER_UNDERPOWERED:
      return f"E-AURC: {underpowered_label(self.n_ranked)}"
    stamp = f" {TIER_PROVISIONAL}" if self.curve_tier == TIER_PROVISIONAL else ""
    return (
      f"E-AURC {self.curve.e_aurc:.4f} (lower is better) | AURC {self.curve.aurc:.4f} "
      f"| oracle {self.curve.aurc_oracle:.4f} | random {self.curve.aurc_random:.4f} "
      f"| no-information band {self.curve.aurc_random - self.curve.aurc_oracle:.4f}"
      f"{stamp}"
    )

  def format(self) -> str:
    head = f"DeferLab selective scorecard -- task {self.task_id} horizon {self.horizon}"
    if self.tier == TIER_UNDERPOWERED:
      return "\n".join([
        head,
        f"  {underpowered_label(self.n)}",
        "  Below thirty episodes nothing here is a measurement, so no number is "
        "printed and no flag prints one.",
      ])
    lines = [
      head,
      f"  episodes: {self.n} scored, {self.n_ranked} ranked, "
      f"{self.n_abstained} abstained, seed {self.seed} [{self.tier}]",
      f"  {self.headline()}",
    ]
    if self.curve_tier == TIER_UNDERPOWERED:
      lines.append(
        f"  risk-coverage, shuffle control and rank correlation: "
        f"{underpowered_label(self.n_ranked)}"
      )
    else:
      if self.permutation is not None:
        lines.append(f"  {self.permutation.format()}")
      lines.append(f"  {self.rho_cell()}")
      for level, rate in self.coverage_at_risk:
        if rate is None:
          lines.append(f"  coverage@risk<={level:.2f}: none, no coverage reaches that risk")
        else:
          lines.append(f"  {rate.format()}")
    for line in self.table.format().split("\n"):
      lines.append(f"  {line}")
    for line in self.credit.format().split("\n"):
      lines.append(f"  {line}")
    for utility in self.utilities:
      lines.append(f"  {utility.format()}")
    note = self.information_cell()
    if note is not None:
      lines.append(f"  {note}")
    lines.append(f"  rankable: {self.rankable}")
    return "\n".join(lines)

  def to_dict(self) -> Dict[str, Any]:
    out: Dict[str, Any] = {
      "task_id": self.task_id,
      "horizon": self.horizon,
      "seed": self.seed,
      "n": self.n,
      "n_ranked": self.n_ranked,
      "n_abstained": self.n_abstained,
      "tier": self.tier,
      "curve_tier": self.curve_tier,
    }
    if self.tier == TIER_UNDERPOWERED:
      label = underpowered_label(self.n)
      for key in (
        "e_aurc",
        "aurc",
        "aurc_oracle",
        "aurc_random",
        "coverage_at_risk",
        "act_defer",
        "abstention",
        "utility",
        "permutation",
        "spearman_rho",
      ):
        out[key] = label
      out["no_information"] = True
      out["information_note"] = label
      out["valid"] = False
      out["rankable"] = False
      return out
    curve_label = underpowered_label(self.n_ranked)
    if self.curve is None:
      out["e_aurc"] = None
      out["aurc"] = None
      out["aurc_oracle"] = None
      out["aurc_random"] = None
      out["coverage_at_risk"] = None
      out["permutation"] = None
      out["spearman_rho"] = None
      out["spearman_cell"] = "spearman rho (confidence vs error): no ranked episodes"
    elif self.curve_tier == TIER_UNDERPOWERED:
      for key in ("e_aurc", "aurc", "aurc_oracle", "aurc_random", "coverage_at_risk",
                  "permutation", "spearman_rho"):
        out[key] = curve_label
      out["spearman_cell"] = curve_label
    else:
      out["e_aurc"] = self.curve.e_aurc
      out["aurc"] = self.curve.aurc
      out["aurc_oracle"] = self.curve.aurc_oracle
      out["aurc_random"] = self.curve.aurc_random
      out["coverage_at_risk"] = [
        {"risk": level, "coverage": None if rate is None else rate.to_dict()}
        for level, rate in self.coverage_at_risk
      ]
      out["permutation"] = None if self.permutation is None else self.permutation.to_dict()
      out["spearman_rho"] = self.rho
      out["spearman_cell"] = self.rho_cell()
    out["act_defer"] = self.table.to_dict()
    out["abstention"] = self.credit.to_dict()
    out["utility"] = [u.to_dict() for u in self.utilities]
    out["no_information"] = self.no_information
    out["information_note"] = self.information_cell()
    out["valid"] = self.valid
    out["rankable"] = self.rankable
    return out


def score(
  outcomes: Sequence[EpisodeOutcome],
  threshold: FrozenThreshold,
  task_id: str = "",
  horizon: int = 0,
  seed: int = 0,
  n_permutations: int = PERMUTATION_RESAMPLES,
) -> SelectiveReport:
  """Build the scorecard for one task at one horizon.

  Abstained episodes are counted in every rate and are absent from the
  risk-coverage curve, because ranking them would mean inventing a confidence
  for a refusal, and an invented confidence is the thing the whole exercise is
  supposed to detect. The two denominators are reported separately and each
  quantity is tiered against its own.
  """
  if len(outcomes) == 0:
    raise ValueError("nothing to score: an empty episode list is not a result")
  if not isinstance(threshold, FrozenThreshold):
    raise ThresholdNotFrozen(
      "score() takes a FrozenThreshold, not a bare number, so the calibration "
      "source travels with the threshold into the report."
    )

  n = len(outcomes)
  theta = float(threshold.value)
  n_act_hit = 0
  n_act_miss = 0
  n_defer_hit = 0
  n_defer_miss = 0
  n_unknowable = 0
  n_abstained = 0
  n_unknowable_abstained = 0
  n_knowable_abstained = 0
  n_credited = 0
  ranked_hits: List[int] = []
  ranked_confidences: List[float] = []

  for outcome in outcomes:
    hit = bool(outcome.hit)
    if outcome.unknowable:
      n_unknowable += 1
    if outcome.abstain:
      n_abstained += 1
      if outcome.unknowable:
        n_unknowable_abstained += 1
        n_credited += 1
      else:
        n_knowable_abstained += 1
      acted = False
    else:
      confidence = float(outcome.confidence)
      ranked_hits.append(1 if hit else 0)
      ranked_confidences.append(confidence)
      acted = confidence >= theta
      if not outcome.unknowable and hit:
        n_credited += 1
    if acted:
      if hit:
        n_act_hit += 1
      else:
        n_act_miss += 1
    else:
      if hit:
        n_defer_hit += 1
      else:
        n_defer_miss += 1

  table = ActDeferTable(
    n_act_hit=n_act_hit,
    n_act_miss=n_act_miss,
    n_defer_hit=n_defer_hit,
    n_defer_miss=n_defer_miss,
    threshold=threshold,
  )
  credit = AbstentionCredit(
    n=n,
    n_unknowable=n_unknowable,
    n_abstained=n_abstained,
    n_unknowable_abstained=n_unknowable_abstained,
    n_knowable_abstained=n_knowable_abstained,
    n_credited=n_credited,
  )

  curve: Optional[RiskCoverageCurve] = None
  permutation: Optional[PermutationResult] = None
  rho: Optional[float] = None
  levels: List[Tuple[float, Optional[Rate]]] = []
  if ranked_hits:
    curve = RiskCoverageCurve.from_outcomes(ranked_hits, ranked_confidences, seed=seed)
    levels = [(level, curve.coverage_at_risk(level)) for level in RISK_LEVELS]
    if n_permutations > 0:
      permutation = confidence_shuffle_p(
        ranked_hits, ranked_confidences, seed=seed, n_permutations=n_permutations
      )
    rho = spearman_rho(ranked_confidences, [1 - h for h in ranked_hits])

  return SelectiveReport(
    task_id=task_id,
    horizon=horizon,
    seed=seed,
    n=n,
    n_ranked=len(ranked_hits),
    n_abstained=n_abstained,
    curve=curve,
    coverage_at_risk=tuple(levels),
    table=table,
    credit=credit,
    utilities=decision_utility(table),
    permutation=permutation,
    rho=rho,
  )
