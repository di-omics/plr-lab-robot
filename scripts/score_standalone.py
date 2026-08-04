"""Score a DeferLab result file with nothing installed. Reads JSON, prints the card.

    python scripts/score_standalone.py results.json
    cat results.json | python scripts/score_standalone.py
    python scripts/score_standalone.py results.json --json

This file duplicates the scoring core of ``plr_lr/learn/selective.py`` on
purpose, and the duplication is the feature rather than an oversight waiting to
be refactored away. A submitter who wants to check a number before believing it
should not have to install this package, resolve pylabrobot and numpy, and
thereby run the scorer inside the same environment that produced the result. One
file on a bare ``python:3.9-slim`` with zero pip installs is a much smaller thing
to trust, and it is the only version of this scorer that a sceptic can audit in
a single sitting.

The cost of that choice is real: two copies can drift. So
``tests/test_learn_selective.py`` runs this script against a fixture and asserts
that both the printed card and the JSON agree with the package implementation
character for character. If the copies drift, that test fails, which is the
whole reason it exists.

Carried over: the reporting tiers, Wilson intervals, the risk-coverage curve and
its oracle and random bands, the act/defer 2x2, abstention credit, decision
utility, the confidence-shuffle permutation control, and the rank correlation.
Left behind: the paired bootstrap and the two-sample KS test, which compare two
submissions rather than score one, and so belong to the harness rather than to a
result file.

Input schema::

    {
      "task_id": "S2", "horizon": 2, "seed": 0,
      "threshold": {"value": 0.6, "calibrated_on": "dev seeds 0-99",
                    "n_calibration": 100},
      "episodes": [
        {"episode_id": "S2-0001", "hit": true, "confidence": 0.91,
         "abstain": false, "unknowable": false},
        {"episode_id": "S2-0002", "hit": false, "confidence": null,
         "abstain": true, "unknowable": true}
      ]
    }

An abstaining episode carries ``"confidence": null`` and nothing else. An
abstention that also banks a prediction is rejected rather than scored.
"""

from __future__ import annotations

import json
import math
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

WILSON_Z = 1.96
MIN_N_REPORTABLE = 30
MIN_N_RANKABLE = 200

TIER_UNDERPOWERED = "UNDERPOWERED"
TIER_PROVISIONAL = "PROVISIONAL"
TIER_RANKABLE = "RANKABLE"

RISK_LEVELS = (0.05, 0.10, 0.20)
W_FAIL_WEIGHTS = (-1.0, -4.0, -10.0)
HEADLINE_W_FAIL = -4.0
PERMUTATION_RESAMPLES = 1000
UNKNOWABLE_BAND = (0.05, 0.40)
INFORMATIVE_RHO = 0.1
NO_INFORMATION_P = 0.05
_EPS = 1e-12

USAGE = "usage: score_standalone.py [results.json] [--json]"


class MetricRefused(ValueError):
  """Raised when a number exists arithmetically but would not be a measurement."""


class ThresholdNotFrozen(ValueError):
  """Raised when an act/defer threshold has no declared calibration source."""


class AbstentionBandRefused(MetricRefused):
  """Raised when the unknowable fraction puts abstention metrics outside their band."""


def tier_of(n: int) -> str:
  if n < MIN_N_REPORTABLE:
    return TIER_UNDERPOWERED
  if n < MIN_N_RANKABLE:
    return TIER_PROVISIONAL
  return TIER_RANKABLE


def underpowered_label(n: int) -> str:
  return f"UNDERPOWERED n={n}"


def wilson_interval(successes: int, trials: int, z: float = WILSON_Z) -> Tuple[float, float]:
  """Wilson score interval, clamped to [0, 1]. See selective.py for why not normal."""
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
  """A count over a denominator, so the interval is always recoverable."""

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


def _tie_break_order(n: int, seed: int) -> List[int]:
  order = list(range(n))
  random.Random(seed).shuffle(order)
  return order


@dataclass(frozen=True)
class RiskCoverageCurve:
  """Losses ordered by descending confidence. The mathematics, not the claim."""

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
      losses.append(1 - int(bool(hit)))
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
    if not 0 <= m <= self.n:
      raise ValueError(f"coverage needs 0 <= m <= {self.n}, got {m}")
    return m / self.n

  def risk(self, m: int) -> Optional[float]:
    """None at zero coverage, never 0.0. Refusal rule R1."""
    if not 0 <= m <= self.n:
      raise ValueError(f"risk needs 0 <= m <= {self.n}, got {m}")
    if m == 0:
      return None
    return sum(self.losses_by_rank[:m]) / m

  @property
  def aurc(self) -> float:
    total = 0.0
    cumulative = 0
    for m in range(1, self.n + 1):
      cumulative += self.losses_by_rank[m - 1]
      total += cumulative / m
    return total / self.n

  @property
  def aurc_oracle(self) -> float:
    correct = self.n - self.n_errors
    total = 0.0
    for m in range(1, self.n + 1):
      excess = m - correct
      if excess > 0:
        total += excess / m
    return total / self.n

  @property
  def aurc_random(self) -> float:
    return self.n_errors / self.n

  @property
  def e_aurc(self) -> float:
    """The headline. The no-information band is aurc_random - aurc_oracle, not zero."""
    return self.aurc - self.aurc_oracle

  def coverage_at_risk(self, r: float) -> Optional[Rate]:
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


@dataclass(frozen=True)
class FrozenThreshold:
  """A threshold plus the split it was fitted on, printed verbatim."""

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
  """The full 2x2. ADA is a property here and never a return value on its own."""

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


@dataclass(frozen=True)
class AbstentionCredit:
  """Counts behind the abstention metrics. There is no attribute named ``acr``."""

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
    """RefusalRecall and RefusalPrecision, together, for the same reason."""
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


def _mean(values: Sequence[float]) -> float:
  return sum(values) / len(values)


def _median(values: Sequence[float]) -> float:
  ordered = sorted(values)
  n = len(ordered)
  middle = n // 2
  if n % 2 == 1:
    return ordered[middle]
  return (ordered[middle - 1] + ordered[middle]) / 2.0


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
  """Permute the confidences, keep the errors, count how often that does better."""
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
  """Rank correlation, or None when one column never varies."""
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
class EpisodeOutcome:
  """One scored episode. Abstaining and predicting are mutually exclusive."""

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
  """Everything the harness is entitled to say about one task at one horizon."""

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
    return tier_of(self.n)

  @property
  def curve_tier(self) -> str:
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

    "The control did not run" counts as no information, not as a pass.
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
    """The reason no information claim is being made, or None if one is."""
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
    if self.rho is None:
      return "spearman rho (confidence vs error): undefined, confidence never varied"
    if abs(self.rho) < INFORMATIVE_RHO:
      return "spearman rho (confidence vs error): NOT-INFORMATIVE"
    return f"spearman rho (confidence vs error): {self.rho:+.4f}"

  def headline(self) -> str:
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
  """Build the scorecard. Abstained episodes count in the rates, never in the curve."""
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


# -- the file interface --------------------------------------------------------


def report_from_payload(payload: Dict[str, Any]) -> SelectiveReport:
  """Turn a parsed result file into a scorecard, refusing a malformed one loudly."""
  if not isinstance(payload, dict):
    raise ValueError("the result file must be a JSON object")
  raw_threshold = payload.get("threshold")
  if not isinstance(raw_threshold, dict):
    raise ThresholdNotFrozen(
      "the result file has no 'threshold' object. A score without the threshold it "
      "was produced under is not reproducible and is not accepted."
    )
  threshold = FrozenThreshold(
    value=float(raw_threshold.get("value")),
    calibrated_on=str(raw_threshold.get("calibrated_on", "")),
    n_calibration=int(raw_threshold.get("n_calibration", 0)),
  )
  raw_episodes = payload.get("episodes")
  if not isinstance(raw_episodes, list) or not raw_episodes:
    raise ValueError("the result file has no 'episodes' list to score")
  outcomes: List[EpisodeOutcome] = []
  for index, row in enumerate(raw_episodes):
    if not isinstance(row, dict):
      raise ValueError(f"episode {index} is not a JSON object")
    confidence = row.get("confidence", None)
    outcomes.append(
      EpisodeOutcome(
        episode_id=str(row.get("episode_id", index)),
        hit=bool(row["hit"]),
        confidence=None if confidence is None else float(confidence),
        abstain=bool(row.get("abstain", False)),
        unknowable=bool(row.get("unknowable", False)),
      )
    )
  return score(
    outcomes,
    threshold,
    task_id=str(payload.get("task_id", "")),
    horizon=int(payload.get("horizon", 0)),
    seed=int(payload.get("seed", 0)),
  )


def main(argv: Sequence[str]) -> int:
  as_json = False
  path: Optional[str] = None
  for argument in argv:
    if argument == "--json":
      as_json = True
    elif argument in ("-h", "--help"):
      print(USAGE)
      return 0
    elif argument.startswith("-"):
      print(f"unknown option {argument}\n{USAGE}", file=sys.stderr)
      return 2
    elif path is None:
      path = argument
    else:
      print(f"more than one input file given\n{USAGE}", file=sys.stderr)
      return 2
  if path is None:
    text = sys.stdin.read()
  else:
    with open(path, "r") as handle:
      text = handle.read()
  if not text.strip():
    print("no input. " + USAGE, file=sys.stderr)
    return 2
  try:
    report = report_from_payload(json.loads(text))
  except (ValueError, TypeError) as refusal:
    # A refusal printed as a sentence, not as a traceback. The submitter reading
    # this is checking a number, not debugging the scorer, and a stack trace
    # invites the conclusion that the file is fine and the tool is broken.
    print(f"REFUSED: {refusal}", file=sys.stderr)
    return 1
  if as_json:
    print(json.dumps(report.to_dict(), sort_keys=True, indent=2))
  else:
    print(report.format())
  return 0


if __name__ == "__main__":
  sys.exit(main(sys.argv[1:]))
