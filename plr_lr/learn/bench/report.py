"""What the harness is allowed to print, and the sixteen things it declines to.

Every refusal below is a code path with a test behind it. A refusal that exists only
as a sentence in a README is a refusal that has never executed, and the first time
it would have mattered is the first time somebody notices it does not work.

The two that matter most are the two that withhold rather than downgrade.

**A model that fails the action-conditioning gate gets a refusal, not a low score.**
A low score is still a score: it is ranked, it appears in a table, and the table is
what gets quoted. The documented failure mode in the current literature is a model
that ignores its conditioning entirely while reporting a headline above ninety
percent, and a benchmark that ranked such a model would be reproducing exactly the
claim-versus-evidence mismatch it exists to indict. Rule R4.

**When the shuffled-action control cannot be beaten, the headline is withheld and
``NOT ACTION-CONDITIONED`` is printed in its place.** Not the number *and* the
warning: a benchmark that prints a number next to its own refutation gets the number
quoted and the refutation dropped. Rule R2 of the same family, enforced here.

There is no cross-task aggregate anywhere in this module, and
:attr:`BenchReport.aggregate_score` raises rather than returning ``None``. A
composite index lets a strong S1 hide a broken S3, and a ``None`` return would be
``or 0.0``-ed by the next caller. Rule R10.

Every row reads ``basis: simulated`` until a hardware run record with a device
serial hash exists. That rule is not negotiable by a flag, and its origin is
concrete: a driver in this portfolio whose hardware paths have never executed
because the instrument would not power on, and which must therefore never be
described as bench-validated. The same rule binds here.

Rule R9 is enforced by absence rather than by a raise, so the surface it is asserted
over is part of the rule. That surface is this package **plus**
:mod:`plr_lr.learn.selective` and ``scripts/score_standalone.py``, because AURC, the
risk-coverage curve and the frozen threshold live in those two and not here. A rule
policed only over the modules that import the arithmetic is a rule that stops holding
the first time a cut point is read off the data in the module that owns it. The AST
walk is in ``tests/test_bench_report.py``.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

from ..predict import InterventionResult
from ..selective import (
  MIN_N_RANKABLE,
  MIN_N_REPORTABLE,
  TIER_UNDERPOWERED,
  AbstentionBandRefused,
  BootstrapCI,
  MetricRefused,
  Rate,
  RiskCoverageCurve,
  SelectiveReport,
  tier_of,
  underpowered_label,
)
from .controls import ControlSuiteReport
from .metrics import (
  ComputeDeclaration,
  OverlapCount,
  PixelColumn,
  SuccessGap,
  TolerancePredictionError,
)
from .packio import BASIS_SIMULATED, ScenePack

#: The one clock source that can support a per-action millisecond claim. Every other
#: source is the host noticing a frame, which is a different measurement wearing the
#: same units. Rule R12.
DEVICE_CLOCK = "device_30000"

#: The printed tokens. Named so a test can assert the exact string reached the page,
#: rather than asserting that some refusal happened somewhere.
NOT_ACTION_CONDITIONED = "NOT ACTION-CONDITIONED"
PROVENANCE_UNDECLARED = "PROVENANCE: UNDECLARED"
THRESHOLD_SENSITIVE = "THRESHOLD-SENSITIVE"
CONFOUNDED_BY_CLAMP = "CONFOUNDED-BY-CLAMP"
FIXTURE_MISMATCH = "FIXTURE-MISMATCH"
NULL_CELL = "null"

#: Every rule, its printed token, and one line saying what permissive behaviour it
#: prevents. Walked by the test that requires each one to fire on a constructed
#: input, because a rule nothing can reach is documentation rather than a rule.
REFUSAL_RULES: Tuple[Tuple[str, str, str], ...] = (
  ("R1", NULL_CELL, "selective risk at zero coverage: the most flattering possible lie"),
  ("R2", "UNDERPOWERED", "any rate below thirty; any rankable row below two hundred"),
  ("R3", "NO-INTERVAL", "any rate without a Wilson interval, any delta without a CI"),
  ("R4", "INTERVENTION-REFUSED", "any headline for a model that ignores its actions"),
  ("R5", "ACR-WITHOUT-FAR", "one-sided abstention numbers are unfalsifiable"),
  ("R6", "BAND-REFUSED", "abstention metrics outside the unknowable band"),
  ("R7", PROVENANCE_UNDECLARED, "any score for an adapter with no declared training set"),
  ("R8", "COMPONENT-UNDECLARED", "any score for an undeclared fitted component"),
  ("R9", "MEAN-REFUSED", "the mean of TPE, any percentile threshold, max-over-frames"),
  ("R10", "NO-AGGREGATE", "any cross-task aggregate scalar"),
  ("R11", "NOT-COMPARABLE", "any table mixing epochs, horizons, tolerances or counts"),
  ("R12", "UNSTAMPED-TIMING", "any per-action latency claim off a host arrival clock"),
  ("R13", "SIMULATED", "any hardware label for a run with no device serial hash"),
  ("R14", THRESHOLD_SENSITIVE, "a ranking whose ADA and E-AURC orders disagree"),
  ("R15", CONFOUNDED_BY_CLAMP, "an arm correlation the clamp subset does not reproduce"),
  ("R16", FIXTURE_MISMATCH, "a submission whose frame hashes are not the fixture's"),
)

RULE_IDS: Tuple[str, ...] = tuple(rule for rule, _label, _why in REFUSAL_RULES)


class TableNotComparable(MetricRefused):
  """Raised when rows that are not the same measurement are put in one table. R11."""


class UnstampedTiming(MetricRefused):
  """Raised when a latency claim is attempted off a clock the arm did not stamp. R12."""


class SimulatedBasis(MetricRefused):
  """Raised when a simulated run is asked to carry a hardware label. R13."""


@dataclass(frozen=True)
class Refusal:
  """One thing the harness declined to report, and why, in the row that wanted it."""

  rule: str
  label: str
  detail: str
  blocking: bool = True

  def __post_init__(self) -> None:
    if self.rule not in RULE_IDS:
      raise ValueError(
        f"refusal rule {self.rule!r} is not one of {list(RULE_IDS)}. An ad-hoc rule "
        "identifier cannot be cross-referenced against the published table, so a "
        "reader cannot check that the refusal is one this benchmark declared."
      )

  def line(self) -> str:
    return f"{self.rule} {self.label}: {self.detail}"

  def to_dict(self) -> Dict[str, Any]:
    return {
      "rule": self.rule,
      "label": self.label,
      "detail": self.detail,
      "blocking": self.blocking,
    }


# -- the individual rules ------------------------------------------------------


def risk_cell(curve: Optional[RiskCoverageCurve], m: int) -> str:
  """Selective risk at rank cut ``m``, printing ``null`` at zero coverage. Rule R1.

  Zero risk at zero coverage is arithmetically true and is the single most
  flattering thing this curve can say: a model that answered nothing has not
  achieved a perfect selective risk, it has produced no evidence. The literal
  ``null`` is what prints, and it is what lands in the JSON.
  """
  if curve is None:
    return NULL_CELL
  value = curve.risk(int(m))
  if value is None:
    return NULL_CELL
  return f"{value:.4f}"


def underpowered_refusal(n: int, what: str) -> Optional[Refusal]:
  """Rule R2, as a record rather than an exception, since the row still prints."""
  if n < MIN_N_REPORTABLE:
    return Refusal(
      "R2",
      "UNDERPOWERED",
      f"{what}: {underpowered_label(n)}. Below thirty observations nothing here is a "
      "measurement, and there is no flag that prints one anyway.",
    )
  if n < MIN_N_RANKABLE:
    return Refusal(
      "R2",
      "PROVISIONAL",
      f"{what}: n={n} is visible but not comparable, so this row may not enter a "
      "leaderboard or a delta.",
      blocking=False,
    )
  return None


def require_interval(value: Any, what: str) -> Rate:
  """Rule R3. A rate arrives as counts or it does not arrive.

  A bare float has already lost the counts its interval needs, and the interval
  cannot be recovered afterwards. Thirteen surveyed real-robot papers report no
  interval at all, which is the practice this refusal exists to not join.
  """
  if isinstance(value, Rate):
    return value
  raise MetricRefused(
    f"{what} arrived as {type(value).__name__} rather than a Rate. A rate built from "
    "a proportion cannot produce its own Wilson interval, and a number without one "
    "travels into a table with nothing attached to it (R3)."
  )


def require_paired_ci(value: Any, what: str) -> BootstrapCI:
  """Rule R3 for deltas. A difference without a paired interval is not a comparison."""
  if isinstance(value, BootstrapCI):
    return value
  raise MetricRefused(
    f"{what} arrived as {type(value).__name__} rather than a paired bootstrap "
    "interval. Two models saw the same episodes; a difference reported without the "
    "paired interval throws away the pairing and the reader cannot tell whether the "
    "difference is anything (R3)."
  )


def abstention_cell(report: SelectiveReport) -> Tuple[str, Optional[Refusal]]:
  """ACR and FAR together, or the reason there is no pair to print. Rules R5 and R6.

  The credit rate is unreachable in :mod:`plr_lr.learn.selective` except through a
  call that hands back the false-alarm rate in the same tuple, so R5 is enforced by
  the type rather than by this function remembering to print both. What this adds is
  the band refusal: outside a five-to-forty percent unknowable fraction the two
  numbers are about the base rate rather than about the model.
  """
  credit = report.credit
  try:
    acr, far = credit.credit_and_false_alarm()
  except AbstentionBandRefused as exc:
    return (
      f"abstention: REFUSED, {exc}",
      Refusal("R6", "BAND-REFUSED", str(exc)),
    )
  recall, precision = credit.recall_and_precision()
  lines = [acr.format(), far.format(), recall.format()]
  if precision is None:
    lines.append("RefusalPrecision: undefined, the model never abstained")
  else:
    lines.append(precision.format())
  return (" | ".join(lines), None)


def require_comparable(rows: Sequence["BenchRow"]) -> None:
  """Rule R11. Refuse a table whose rows are not the same measurement.

  Prints the mismatch rather than a generic complaint, because the usual cause is a
  seed epoch rotating between two runs that were then pasted into one table, and the
  reader has to be told which column moved.
  """
  if len(rows) < 2:
    return
  keys = ("seed_epoch", "horizon", "tolerance", "n", "split")
  first = rows[0]
  for row in rows[1:]:
    for key in keys:
      left = getattr(first, key)
      right = getattr(row, key)
      if left != right:
        raise TableNotComparable(
          f"NOT-COMPARABLE (R11): {first.task_id}@{first.horizon} and "
          f"{row.task_id}@{row.horizon} disagree on {key}: {left!r} against {right!r}. "
          "These are two measurements, not two entries."
        )


def latency_claim(
  clock_source: str,
  p50_ms: float,
  p95_ms: float,
  p99_ms: float,
  n_stalls: int = 0,
) -> str:
  """Rule R12. A per-action latency claim, or a refusal naming the clock.

  Only one stream on the arm carries a timestamp the controller stamped itself.
  Everything else is the host noticing that a packet arrived, which includes the
  scheduling delay of the process doing the noticing. A mean is refused separately
  and unconditionally: the community-reported failure mode on this controller is
  roughly one two-hundred-millisecond stall every ten seconds, and a mean hides it
  entirely while the p99 shows it.
  """
  if clock_source != DEVICE_CLOCK:
    raise UnstampedTiming(
      f"UNSTAMPED-TIMING (R12): a per-action latency claim was attempted with "
      f"clock_source={clock_source!r}. Only {DEVICE_CLOCK!r} carries a device "
      "timestamp; everything else measures when this host noticed a packet."
    )
  return (
    f"latency p50 {p50_ms:.2f} ms | p95 {p95_ms:.2f} ms | p99 {p99_ms:.2f} ms | "
    f"{n_stalls} stalls. No mean is printed: the reported stall pattern on this "
    "controller is invisible in a mean and visible in the p99."
  )


def basis_label(pack: ScenePack) -> str:
  """Rule R13. ``simulated`` unless a device serial hash says otherwise."""
  return pack.basis


def require_hardware_basis(pack: ScenePack) -> None:
  """Rule R13, as a refusal, for anything that wants to say ``bench-validated``."""
  if pack.basis == BASIS_SIMULATED:
    raise SimulatedBasis(
      f"SIMULATED (R13): pack {pack.pack_id} carries no device serial hash, so every "
      "row built from it reads basis: simulated and none of it may be called "
      "bench-validated. A simulation-only result described as a hardware result is "
      "the specific claim this rule exists to make impossible."
    )


@dataclass(frozen=True)
class Comparison:
  """Two rows ranked against each other, or the reason they are not. Rule R14."""

  left: str
  right: str
  e_aurc_order: str
  ada_order: str
  threshold_sensitive: bool
  detail: str

  def format(self) -> str:
    if self.threshold_sensitive:
      return f"{THRESHOLD_SENSITIVE}: {self.detail}"
    return f"ranking {self.left} vs {self.right}: {self.detail}"


def _order(left: Optional[float], right: Optional[float], lower_is_better: bool) -> str:
  if left is None or right is None:
    return "undefined"
  if left == right:
    return "tied"
  better_left = left < right if lower_is_better else left > right
  return "left" if better_left else "right"


def compare(left: "BenchRow", right: "BenchRow") -> Comparison:
  """Rank two rows, and refuse the ranking when the two orderings disagree. Rule R14.

  The frozen-threshold act/defer accuracy and the threshold-free excess AURC are
  answering different questions, and published work finds that threshold selection
  matters more than the choice of uncertainty method. When the two disagree about
  which model is better, the honest report is that the ordering depends on the
  threshold, not a coin flip dressed as a result.
  """
  require_comparable((left, right))
  left_e = left.e_aurc
  right_e = right.e_aurc
  left_ada = left.ada_value
  right_ada = right.ada_value
  e_order = _order(left_e, right_e, lower_is_better=True)
  ada_order = _order(left_ada, right_ada, lower_is_better=False)
  disagree = (
    e_order in ("left", "right")
    and ada_order in ("left", "right")
    and e_order != ada_order
  )
  detail = (
    f"E-AURC prefers {e_order}, frozen-threshold ADA prefers {ada_order}"
    if disagree
    else f"E-AURC prefers {e_order}, ADA prefers {ada_order}"
  )
  return Comparison(
    left=f"{left.task_id}@{left.horizon}",
    right=f"{right.task_id}@{right.horizon}",
    e_aurc_order=e_order,
    ada_order=ada_order,
    threshold_sensitive=disagree,
    detail=detail,
  )


@dataclass(frozen=True)
class ArmCorrelation:
  """The abstention-versus-arm-outcome correlation, computed twice. Rule R15."""

  rho_all: Optional[float]
  rho_unclamped: Optional[float]
  delta: BootstrapCI
  n_all: int
  n_unclamped: int

  @property
  def confounded(self) -> bool:
    if self.rho_all is None or self.rho_unclamped is None:
      return True
    return not self.delta.includes_zero

  def headline(self) -> str:
    if self.confounded:
      return (
        f"{CONFOUNDED_BY_CLAMP}: the all-episodes correlation "
        f"({self.rho_all}) and the unclamped-subset correlation "
        f"({self.rho_unclamped}) disagree beyond the paired interval "
        f"{self.delta.format()}. The headline is withheld: what was measured may be "
        "that clamping predicts failure rather than that abstention does."
      )
    return (
      f"abstention correlates with arm outcome at rho {self.rho_all:.4f} over "
      f"{self.n_all} episodes, and at {self.rho_unclamped:.4f} over the "
      f"{self.n_unclamped} episodes the envelope left alone. {self.delta.format()}"
    )


# -- the row -------------------------------------------------------------------


@dataclass(frozen=True)
class BenchRow:
  """One task at one horizon. There is no row that is more than one of either.

  A row carries its own seed epoch, split, basis and tolerance, so two rows can be
  checked for comparability without going back to whatever produced them. That is
  what makes R11 mechanical rather than a matter of the reader remembering.
  """

  task_id: str
  horizon: int
  seed_epoch: str
  pack_id: str
  split: str
  basis: str
  tolerance: Tuple[float, ...]
  n: int
  selective: SelectiveReport
  tpe: TolerancePredictionError
  gap: SuccessGap
  compute: ComputeDeclaration
  pixel: PixelColumn
  overlap: OverlapCount
  action_conditioned: bool = True
  flagged_tpe: Optional[TolerancePredictionError] = None
  refusals: Tuple[Refusal, ...] = ()

  @property
  def blocking(self) -> Tuple[Refusal, ...]:
    return tuple(r for r in self.refusals if r.blocking)

  @property
  def e_aurc(self) -> Optional[float]:
    if self.selective.curve is None or self.selective.curve_tier == TIER_UNDERPOWERED:
      return None
    return self.selective.curve.e_aurc

  @property
  def ada_value(self) -> Optional[float]:
    return self.selective.table.ada.value

  @property
  def headline_withheld(self) -> bool:
    """Is the E-AURC headline withheld rather than printed?

    Withheld when the model did not beat the shuffled-action control, and withheld
    when any blocking refusal applies. Printing the number beside its own refutation
    is how the number gets quoted and the refutation gets dropped, so the number
    does not print at all.
    """
    return (not self.action_conditioned) or bool(self.blocking)

  @property
  def valid(self) -> bool:
    """Three conditions, all of which have to hold at once.

    The selective report has to be valid on its own terms, the model has to have
    beaten the shuffled-action control, and no blocking refusal may apply. The basis
    is deliberately not one of them: a simulated row is a perfectly valid row, it
    just may never be described as a hardware result.
    """
    return self.selective.valid and self.action_conditioned and not self.blocking

  @property
  def rankable(self) -> bool:
    return self.valid and self.selective.rankable and self.n >= MIN_N_RANKABLE

  def headline(self) -> str:
    if not self.action_conditioned:
      return (
        f"{NOT_ACTION_CONDITIONED}: this model did not beat the shuffled-action "
        "control by more than a paired interval, so the E-AURC headline is withheld "
        "rather than printed beside this line."
      )
    if self.blocking:
      reasons = "; ".join(r.line() for r in self.blocking)
      return f"headline withheld: {reasons}"
    return self.selective.headline()

  def format(self) -> str:
    lines = [
      f"-- {self.task_id} horizon {self.horizon} "
      f"[epoch {self.seed_epoch}, split {self.split}, basis: {self.basis}]",
      f"  tolerance: {list(self.tolerance)}",
      f"  {self.headline()}",
    ]
    if self.action_conditioned and not self.blocking:
      lines.append(f"  risk at zero coverage: {risk_cell(self.selective.curve, 0)}")
    for line in self.selective.table.format().split("\n"):
      lines.append(f"  {line}")
    cell, _refusal = abstention_cell(self.selective)
    lines.append(f"  {cell}")
    lines.append(f"  {self.tpe.format()}")
    if self.flagged_tpe is not None:
      lines.append(f"  flagged rows only: {self.flagged_tpe.format()}")
    for line in self.gap.format().split("\n"):
      lines.append(f"  {line}")
    for utility in self.selective.utilities:
      lines.append(f"  {utility.format()}")
    lines.append(f"  {self.compute.format()}")
    lines.append(f"  {self.pixel.format()}")
    lines.append(f"  {self.overlap.format()}")
    note = self.selective.information_cell()
    if note is not None:
      lines.append(f"  {note}")
    for refusal in self.refusals:
      lines.append(f"  {refusal.line()}")
    lines.append(f"  rankable: {self.rankable}")
    return "\n".join(lines)

  def to_dict(self) -> Dict[str, Any]:
    cell, _refusal = abstention_cell(self.selective)
    return {
      "task_id": self.task_id,
      "horizon": self.horizon,
      "seed_epoch": self.seed_epoch,
      "pack_id": self.pack_id,
      "split": self.split,
      "basis": self.basis,
      "tolerance": [float(t) for t in self.tolerance],
      "n": self.n,
      "headline": self.headline(),
      "headline_withheld": self.headline_withheld,
      "action_conditioned": self.action_conditioned,
      "risk_at_zero_coverage": risk_cell(self.selective.curve, 0),
      "selective": self.selective.to_dict(),
      "abstention_cell": cell,
      "tpe": self.tpe.to_dict(),
      "tpe_flagged": None if self.flagged_tpe is None else self.flagged_tpe.to_dict(),
      "success_gap": self.gap.to_dict(),
      "compute": self.compute.to_dict(),
      "pixel": self.pixel.to_dict(),
      "training_overlap": self.overlap.to_dict(),
      "refusals": [r.to_dict() for r in self.refusals],
      "valid": self.valid,
      "rankable": self.rankable,
    }


# -- the report ----------------------------------------------------------------


@dataclass(frozen=True)
class BenchReport:
  """Every row this run is entitled to publish, and every reason some are missing.

  There is no attribute here that reduces the run to one number.
  :attr:`aggregate_score` raises, and it raises rather than returning ``None``
  because a ``None`` gets defaulted by the next caller and a composite index lets a
  strong task hide a broken one.
  """

  model: str
  pack_id: str
  seed_epoch: str
  split: str
  basis: str
  rows: Tuple[BenchRow, ...] = ()
  controls: Optional[ControlSuiteReport] = None
  intervention: Dict[str, InterventionResult] = field(default_factory=dict)
  trust_horizons: Dict[str, Optional[int]] = field(default_factory=dict)
  refusals: Tuple[Refusal, ...] = ()
  declared: Dict[str, Any] = field(default_factory=dict)

  @property
  def aggregate_score(self) -> float:
    raise MetricRefused(
      "NO-AGGREGATE (R10): there is no cross-task aggregate scalar and there will not "
      "be one. A composite index lets a strong S1 hide a broken S3, and the per-task "
      "rows are the result. Read rows."
    )

  @property
  def blocking(self) -> Tuple[Refusal, ...]:
    return tuple(r for r in self.refusals if r.blocking)

  @property
  def controls_passed(self) -> bool:
    return self.controls is not None and self.controls.passed

  @property
  def gate_passed(self) -> bool:
    """Did every task clear the action-conditioning gate?

    ``False`` when the gate did not run at all. A gate that was not run has
    established nothing, and treating an absent gate as a pass is how a non-world
    model gets a world-model row.
    """
    if not self.intervention:
      return False
    return all(result.passed for result in self.intervention.values())

  @property
  def valid(self) -> bool:
    if not self.gate_passed:
      return False
    if not self.controls_passed:
      return False
    if self.blocking:
      return False
    if not self.rows:
      return False
    return all(row.valid for row in self.rows)

  def rows_for(self, task_id: str) -> Tuple[BenchRow, ...]:
    return tuple(r for r in self.rows if r.task_id == task_id)

  def summary(self) -> str:
    lines = [
      f"DeferLab v0.1 scorecard -- {self.model}",
      f"  pack {self.pack_id} | epoch {self.seed_epoch} | split {self.split} | "
      f"basis: {self.basis}",
    ]
    for key in sorted(self.declared):
      lines.append(f"  declared {key}: {self.declared[key]}")
    if self.intervention:
      for task_id in sorted(self.intervention):
        for index, line in enumerate(self.intervention[task_id].summary().split("\n")):
          lines.append(f"  {task_id} {line}" if index == 0 else f"  {line}")
    else:
      lines.append(
        "  interventional gate: NOT RUN. Nothing was checked, so nothing is "
        "established and no headline is printed."
      )
    if self.controls is not None:
      for line in self.controls.summary().split("\n"):
        lines.append(f"  {line}")
    else:
      lines.append("  null controls: NOT RUN, so no scorecard is emitted.")
    for refusal in self.refusals:
      lines.append(f"  {refusal.line()}")
    if not self.rows:
      lines.append(
        "  no rows. This run produced a refusal rather than a score, which is a "
        "different outcome from a low one."
      )
    for task_id in sorted(self.trust_horizons):
      horizon = self.trust_horizons[task_id]
      lines.append(
        f"  trust horizon {task_id}: "
        + ("none, no horizon was trustworthy" if horizon is None else str(horizon))
      )
    for row in self.rows:
      for line in row.format().split("\n"):
        lines.append(f"  {line}")
    lines.append(f"  valid: {self.valid}")
    lines.append(
      "  no cross-task aggregate is printed (R10): a composite index lets a strong "
      "task hide a broken one."
    )
    return "\n".join(lines)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "benchmark": "DeferLab v0.1",
      "model": self.model,
      "pack_id": self.pack_id,
      "seed_epoch": self.seed_epoch,
      "split": self.split,
      "basis": self.basis,
      "declared": dict(self.declared),
      "gate_passed": self.gate_passed,
      "intervention": {
        task_id: {
          "passed": result.passed,
          "n_checked": result.n_checked,
          "n_abstained": result.n_abstained,
          "alpha": result.alpha,
          "seed": result.seed,
          "summary": result.summary(),
        }
        for task_id, result in self.intervention.items()
      },
      "controls": None if self.controls is None else self.controls.to_dict(),
      "trust_horizons": dict(self.trust_horizons),
      "refusals": [r.to_dict() for r in self.refusals],
      "rows": [row.to_dict() for row in self.rows],
      "valid": self.valid,
      "aggregate_score": "REFUSED (R10): no cross-task aggregate scalar exists",
    }

  def to_json(self, indent: int = 2) -> str:
    return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def tier_note(n: int) -> str:
  """The reporting tier ``n`` buys, spelled out. Rule R2 in one line."""
  return f"n={n} [{tier_of(n)}]"


def finite_or_none(value: float) -> Optional[float]:
  """``None`` for a non-finite number, so a NaN never prints as a measurement."""
  number = float(value)
  return number if math.isfinite(number) else None


__all__ = [
  "CONFOUNDED_BY_CLAMP",
  "DEVICE_CLOCK",
  "FIXTURE_MISMATCH",
  "NOT_ACTION_CONDITIONED",
  "NULL_CELL",
  "PROVENANCE_UNDECLARED",
  "REFUSAL_RULES",
  "RULE_IDS",
  "THRESHOLD_SENSITIVE",
  "ArmCorrelation",
  "BenchReport",
  "BenchRow",
  "Comparison",
  "Refusal",
  "SimulatedBasis",
  "TableNotComparable",
  "UnstampedTiming",
  "abstention_cell",
  "basis_label",
  "compare",
  "finite_or_none",
  "latency_claim",
  "require_comparable",
  "require_hardware_basis",
  "require_interval",
  "require_paired_ci",
  "risk_cell",
  "tier_note",
  "underpowered_refusal",
]
