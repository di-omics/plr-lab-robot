"""The rollout-level numbers, composed on top of the selective-prediction core.

Nothing here re-derives AURC, E-AURC, the oracle and random bands, the Wilson
interval, the paired bootstrap, the confidence shuffle, the act/defer table, the
abstention credit or the decision utility. All of those live in
:mod:`plr_lr.learn.selective` and are imported. This repository has already paid
for a parallel measurement vocabulary once: two modules that each grew their own
version of the same quantity, drifted, and then disagreed in whichever direction
made a control look like it had passed. A second AURC here would be exactly that,
so the rule is that a metric with a home does not get a second one.

What is genuinely new at this layer is the comparison that turns a predicted
terminal state into a hit:

    err_i = max_d ( |s_hat_i[d] - s_i[d]| / tau[d] )
    hit_i = 1 if err_i <= 1 else 0

Per-field division by that field's own tolerance, then a max over fields. S4's
joint half-a-millimetre-and-half-a-degree criterion is therefore one number with no
weighting choice in it, and there is nowhere for a weight fitted after seeing the
errors to enter.

Five refusals live here.

* **The mean of the tolerance prediction error is unreachable.**
  :class:`TolerancePredictionError` has no ``mean`` attribute that returns a
  number; asking for one raises. The error distribution is heavy-tailed, a single
  diverged episode moves the mean by more than the other three hundred and
  ninety-nine, and the resulting figure reads as a typical error. Refusal rule R9.
* **Success is read off the final observation only.** Never a minimum over steps,
  never a maximum over frames, never a percentile threshold. Max-over-frames
  selection is this portfolio's own documented way of manufacturing a confident
  wrong number, and the AST check that enforces it is in the test module.
* **The quartiles are order statistics, not a fitted quantile.** They come from the
  halves of a sorted sample, computed here, and no threshold anywhere in the
  benchmark is read off a percentile of the data being scored.
* **Declared compute is printed and never scored.** A leaderboard that hides
  compute lets a sixty-four-card entry look like a single-card entry, against a
  published three-to-four-times latency tax. :class:`ComputeDeclaration` carries
  ``scored = False`` as a field so a sort key cannot reach it by accident.
* **A training declaration this module cannot read withholds the row.** It is never
  skipped and never counted as no overlap. Skipping the part you cannot read is how
  the measured defect this replaces came about: the leak check intersected scored
  *episode identifiers* against declared *corpus, split and seed-range names*, two
  vocabularies that are disjoint by construction, so the count was structurally zero
  and every model got an affirmative no-leak line, including one that had declared it
  was fit on the very split being scored. :func:`training_overlap` now compares a
  declaration against the scope being scored, in that scope's own vocabulary, and
  reports ``LEAK``, ``LEAK-UNCHECKABLE`` and ``CLEAN`` as three different answers.

The pixel column exists for one reason and says so: so that the ``PixelPerfect``
control can visibly top it and still lose the ranking. It is printed under a header
literally reading ``NOT RANKED``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..selective import (
  MetricRefused,
  Rate,
  BootstrapCI,
  paired_bootstrap_ci,
)
# The one median in the build. Imported rather than rewritten so that "median, not
# mean" cannot come to mean two slightly different things in two modules.
from ..selective import _median as _median_of

#: Risk at full coverage above which a horizon is no longer trusted. An absolute
#: constant. A trust horizon read off the curve it is describing would be a
#: threshold fitted to the data it then judges.
TRUST_RISK = 0.25

#: The header the pixel column prints under, verbatim, every time.
NOT_RANKED = "NOT RANKED"

#: What a submission that declared no hardware string gets printed as. A blank
#: reads in a table as though the column were simply not shown.
UNDECLARED_HARDWARE = "UNDECLARED"

#: The literal an analytic model declares in place of a training set. Restated from
#: :data:`plr_lr.learn.predict.NO_TRAINING_DATA` rather than imported, so this module
#: stays stdlib-only and travels into the standalone scorer, and pinned by a test so
#: the two cannot drift. Declaring that nothing was fit is a declaration; having no
#: declaration is an absence, and the two are not the same answer.
NO_TRAINING_DATA = "none:analytic-null-fit-on-no-data"

#: The scope keys a training fingerprint may narrow itself with, after its corpus.
#: The set is closed. A key this module does not understand makes the whole
#: declaration uncheckable and is recorded as such rather than skipped: skipping the
#: part that could not be read is exactly how a leak check ends up comparing two
#: vocabularies that never meet.
#:
#: ``split``, ``epoch``, ``pack`` and ``task`` compare verbatim against the matching
#: field of the :class:`ScoringScope`. ``seeds`` is ``lo-hi`` inclusive and matches
#: when the declared range meets the range being scored. ``episode`` names one
#: episode identifier and may repeat, in which case the identifiers are read as a
#: list; every other key is conjunctive with the rest of the entry.
FINGERPRINT_KEYS: Tuple[str, ...] = (
  "split",
  "epoch",
  "pack",
  "task",
  "episode",
  "seeds",
)

#: What separates a fingerprint's corpus from its narrowing terms:
#: ``deferlab:split=scoring:task=S3``.
FINGERPRINT_SEPARATOR = ":"

#: The corpus token this benchmark's own fixtures are declared under, and the one
#: word a submitter has to write for any of the narrowing terms above to be compared
#: against anything. A declaration naming some other corpus -- ``droid``, ``oxe``, a
#: lab's own recordings -- is a statement that the model was not fit on these
#: episodes, and it is scored as clean rather than withheld: a leak check that
#: refuses every submission naming an outside corpus refuses everybody, which
#: establishes exactly as little as one that refuses nobody.
#:
#: A constant rather than a per-pack field because it is the published half of the
#: contract: the submitter writes it into the declaration before ever seeing the
#: pack, and a corpus name that varied by fixture could not be written down in
#: advance.
BENCH_CORPUS = "deferlab"

#: The binary criterion a policy layer would report, per task, as a tolerance
#: vector in the task's own units. Every entry is wider than the protocol tolerance
#: in :mod:`plr_lr.learn.bench.tasks`, so ``LooseSuccess >= TolAcc`` by construction
#: and the gap is non-negative without any special handling. A positive gap restates
#: the published finding that policies complete lab tasks while violating the
#: quantitative tolerances the protocol requires, one layer down at the world model.
LOOSE_TOLERANCE: Dict[str, Tuple[float, ...]] = {
  "S1": (10.0, 10.0, 10.0, 0.50),
  "S2": (1.00,),
  "S3": (2.00,),
  "S4": (2.00, 2.00, 2.00, 0.50),
  "S5": (2.00, 0.50),
}


class LooseCriterionMissing(MetricRefused):
  """Raised when a task has no declared policy-layer criterion to difference against.

  Defaulting to the protocol tolerance is the permissive alternative and it reports
  a success gap of exactly zero for every task that forgot to declare one, which is
  the most interesting possible result arrived at by an oversight.
  """


# -- the tolerance criterion ---------------------------------------------------


def tolerance_error(
  predicted: Sequence[float],
  truth: Sequence[float],
  tolerance: Sequence[float],
) -> float:
  """Worst per-field error in units of that field's tolerance.

  A value at or below 1.0 is a hit. Returns ``inf`` for a non-finite prediction
  rather than raising, because a diverged prediction is a legitimate scored outcome
  and the row that carries it is a miss, not an error in the harness. An abstention
  never reaches here: it has no predicted state to compare.
  """
  values = [float(v) for v in predicted]
  targets = [float(v) for v in truth]
  taus = [float(v) for v in tolerance]
  if not (len(values) == len(targets) == len(taus)):
    raise MetricRefused(
      f"tolerance_error was given {len(values)} predicted fields, {len(targets)} true "
      f"fields and {len(taus)} tolerances. A comparison across mismatched widths "
      "silently compares whichever columns happen to line up."
    )
  if not values:
    raise MetricRefused("tolerance_error over zero fields is not a comparison")
  worst = 0.0
  for value, target, tau in zip(values, targets, taus):
    if tau <= 0.0 or not math.isfinite(tau):
      raise MetricRefused(
        f"tolerance {tau!r} is not positive and finite. Every error here is divided "
        "by it, so a zero reports a perfect score or a NaN depending on rounding."
      )
    if not math.isfinite(value) or not math.isfinite(target):
      return float("inf")
    scaled = abs(value - target) / tau
    if scaled > worst:
      worst = scaled
  return worst


def in_tolerance(
  predicted: Sequence[float],
  truth: Sequence[float],
  tolerance: Sequence[float],
) -> bool:
  """The hit indicator. Evaluated on the final state only, never a min over steps."""
  return tolerance_error(predicted, truth, tolerance) <= 1.0


def loose_tolerance_for(task_id: str) -> Tuple[float, ...]:
  try:
    return LOOSE_TOLERANCE[task_id]
  except KeyError:
    raise LooseCriterionMissing(
      f"task {task_id!r} declares no policy-layer criterion, so SuccessGap has "
      f"nothing to difference against. Known tasks are {sorted(LOOSE_TOLERANCE)}."
    ) from None


def loose_hit(
  task_id: str,
  predicted: Sequence[float],
  truth: Sequence[float],
) -> bool:
  """Did the prediction satisfy the binary criterion a policy layer would report?"""
  return in_tolerance(predicted, truth, loose_tolerance_for(task_id))


# -- tolerance prediction error ------------------------------------------------


def _tukey_quartiles(values: Sequence[float]) -> Tuple[float, float]:
  """Q1 and Q3 as order statistics of the sorted halves.

  Deliberately not a fitted quantile and deliberately not read off a percentile
  function. These two numbers describe a spread that is being printed; they are
  never used as a threshold, and the whole reason they are computed by hand here is
  so that no percentile call exists anywhere a threshold could later be taken from.
  """
  ordered = sorted(float(v) for v in values)
  n = len(ordered)
  if n == 0:
    raise MetricRefused("no quartiles over an empty sample")
  if n == 1:
    return (ordered[0], ordered[0])
  half = n // 2
  lower = ordered[:half]
  upper = ordered[half + 1:] if n % 2 == 1 else ordered[half:]
  return (_median_of(lower), _median_of(upper))


@dataclass(frozen=True)
class TolerancePredictionError:
  """Median tolerance error with its interquartile range. The mean is refused.

  ``TPE`` is a heavy-tailed quantity: one diverged episode out of four hundred can
  move a mean past every other episode combined, and the resulting figure prints as
  a typical error. There is deliberately no attribute that hands back a mean, and
  :attr:`mean` raises rather than returning ``None``, because a ``None`` gets
  ``or 0.0``-ed by the next caller.
  """

  n: int
  median: float
  q1: float
  q3: float
  n_infinite: int = 0

  @classmethod
  def of(cls, errors: Sequence[float]) -> "TolerancePredictionError":
    finite = [float(e) for e in errors if math.isfinite(float(e))]
    n_infinite = len(errors) - len(finite)
    if not finite:
      raise MetricRefused(
        "every scored episode produced a non-finite error, so there is no error "
        "distribution to summarise. A diverged model is a miss on every row, which "
        "is a result; a median of nothing is not."
      )
    q1, q3 = _tukey_quartiles(finite)
    return cls(
      n=len(errors),
      median=_median_of(finite),
      q1=q1,
      q3=q3,
      n_infinite=n_infinite,
    )

  @property
  def mean(self) -> float:
    raise MetricRefused(
      "the mean of the tolerance prediction error is refused (R9). The distribution "
      "is heavy tailed: a single diverged episode moves the mean past all the others "
      "and the result prints as a typical error. Read median with the IQR beside it."
    )

  def format(self) -> str:
    tail = (
      f", {self.n_infinite} non-finite" if self.n_infinite else ""
    )
    return (
      f"TPE median {self.median:.4f} IQR ({self.q1:.4f}, {self.q3:.4f}) "
      f"in tolerance units over n={self.n}{tail}"
    )

  def to_dict(self) -> Dict[str, Any]:
    return {
      "n": self.n,
      "median": self.median,
      "q1": self.q1,
      "q3": self.q3,
      "n_infinite": self.n_infinite,
      "mean": "REFUSED (R9): heavy-tailed, the mean is not a summary",
    }


# -- the precision gap ---------------------------------------------------------


@dataclass(frozen=True)
class SuccessGap:
  """``LooseSuccess@k - TolAcc@k``, with the paired interval that makes it a claim.

  Both rates carry their counts, so both carry a Wilson interval, and the delta
  carries a paired bootstrap over the shared episode index. A delta printed without
  its interval is refusal rule R3, and this is the delta most likely to be quoted
  on its own because it is the one that restates a published finding.
  """

  task_id: str
  horizon: int
  loose: Rate
  strict: Rate
  delta: BootstrapCI

  @property
  def gap(self) -> Optional[float]:
    loose = self.loose.value
    strict = self.strict.value
    if loose is None or strict is None:
      return None
    return loose - strict

  def format(self) -> str:
    gap = self.gap
    head = (
      f"SuccessGap@{self.horizon}: UNDERPOWERED"
      if gap is None
      else f"SuccessGap@{self.horizon}: {gap:+.4f}"
    )
    return "\n".join([
      head,
      f"  {self.loose.format()}",
      f"  {self.strict.format()}",
      f"  {self.delta.format()}",
    ])

  def to_dict(self) -> Dict[str, Any]:
    return {
      "task_id": self.task_id,
      "horizon": self.horizon,
      "gap": self.gap,
      "loose_success": self.loose.to_dict(),
      "tolerance_accuracy": self.strict.to_dict(),
      "delta": self.delta.to_dict(),
    }


def success_gap(
  task_id: str,
  horizon: int,
  loose_hits: Sequence[int],
  strict_hits: Sequence[int],
  seed: int = 0,
  n_resamples: Optional[int] = None,
) -> SuccessGap:
  """The precision gap for one task at one horizon, with its paired interval."""
  if len(loose_hits) != len(strict_hits):
    raise MetricRefused(
      f"{len(loose_hits)} loose outcomes against {len(strict_hits)} strict ones. The "
      "gap is a paired difference over the same episodes and cannot be taken across "
      "two different sets."
    )
  loose = [int(bool(h)) for h in loose_hits]
  strict = [int(bool(h)) for h in strict_hits]
  kwargs: Dict[str, Any] = {"seed": seed}
  if n_resamples is not None:
    kwargs["n_resamples"] = int(n_resamples)
  delta = paired_bootstrap_ci(
    [float(v) for v in loose],
    [float(v) for v in strict],
    **kwargs
  )
  return SuccessGap(
    task_id=task_id,
    horizon=int(horizon),
    loose=Rate(sum(loose), len(loose), name=f"LooseSuccess@{horizon}"),
    strict=Rate(sum(strict), len(strict), name=f"TolAcc@{horizon}"),
    delta=delta,
  )


# -- the trust horizon ---------------------------------------------------------


def trust_horizon(
  risk_by_horizon: Dict[int, Optional[float]],
  threshold: float = TRUST_RISK,
) -> Optional[int]:
  """``k*``: the largest horizon whose risk at full coverage is still acceptable.

  A measurement and not a conformal guarantee, and it is deliberately contiguous
  from the shortest horizon rather than the plain maximum over satisfying
  horizons. A trust horizon with a hole in it -- acceptable at 1, unacceptable at 2,
  acceptable again at 4 -- is not a horizon, and reporting the 4 would license
  acting at a horizon the model had already been shown to fail at. ``None`` means no
  horizon was trustworthy, which is a reason not to build a screen rather than a
  screen that admits nothing.
  """
  if not risk_by_horizon:
    return None
  best: Optional[int] = None
  for horizon in sorted(risk_by_horizon):
    risk = risk_by_horizon[horizon]
    if risk is None or not math.isfinite(float(risk)) or float(risk) > float(threshold):
      break
    best = int(horizon)
  return best


# -- training overlap ----------------------------------------------------------


@dataclass(frozen=True)
class ScoringScope:
  """What is being scored, in the same vocabulary a training fingerprint declares in.

  A fingerprint names a corpus, a split, a seed epoch, a pack, a task, a seed range
  or an episode. Episode identifiers are the only one of those that a list of scored
  row identifiers can answer on its own; the rest are properties of the pack the rows
  came from, and the caller is the only thing that knows them. Handing the leak check
  the row identifiers alone is what made it structurally incapable of firing.

  ``corpus`` and ``split`` are required because they are the two axes a leak is
  actually declared on. The rest default to undeclared, and a declaration that
  narrows itself on an axis this scope left undeclared is reported as uncheckable
  rather than waved through: this object is a statement about what was checked, so
  it is not allowed to imply a comparison that did not happen.
  """

  corpus: str
  split: str
  seed_epoch: str = ""
  pack_id: str = ""
  task_id: str = ""
  seed_range: Optional[Tuple[int, int]] = None

  def __post_init__(self) -> None:
    for name in ("corpus", "split"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value:
        raise MetricRefused(
          f"ScoringScope.{name} is {value!r}. A scope that will not name its "
          f"{name} cannot have a declared training {name} compared against it, and "
          "the comparison that silently does not happen is the one this type exists "
          "to make impossible."
        )
    if self.seed_range is not None:
      low, high = (int(self.seed_range[0]), int(self.seed_range[1]))
      if low > high:
        raise MetricRefused(
          f"ScoringScope.seed_range is ({low}, {high}), which is empty. An empty "
          "range meets nothing, so every declared seed range would read as clean."
        )
      object.__setattr__(self, "seed_range", (low, high))

  def describe(self) -> str:
    parts = ["corpus={0}".format(self.corpus), "split={0}".format(self.split)]
    if self.seed_epoch:
      parts.append("epoch={0}".format(self.seed_epoch))
    if self.pack_id:
      parts.append("pack={0}".format(self.pack_id))
    if self.task_id:
      parts.append("task={0}".format(self.task_id))
    if self.seed_range is not None:
      parts.append("seeds={0}-{1}".format(self.seed_range[0], self.seed_range[1]))
    return " ".join(parts)


def _declared_seed_range(value: str) -> Optional[Tuple[int, int]]:
  """``lo-hi`` inclusive, or ``None`` when the text is not a range at all."""
  head, separator, tail = value.partition("-")
  if not separator:
    return None
  try:
    low = int(head)
    high = int(tail)
  except ValueError:
    return None
  if low > high:
    return None
  return (low, high)


def _ranges_meet(declared: Tuple[int, int], scored: Tuple[int, int]) -> bool:
  return declared[0] <= scored[1] and scored[0] <= declared[1]


def _rows_covered(
  entry: str,
  rows: Sequence[str],
  scope: Optional[ScoringScope],
) -> Optional[List[bool]]:
  """Which scored rows one declared fingerprint covers.

  ``None`` means the declaration could not be compared at all, which is a third
  answer and never a row of ``False``. Everything downstream of this function keeps
  those two apart, because collapsing them is the defect: an entry that named a
  split was compared against episode identifiers, matched nothing, and printed as
  proof that nothing had leaked.
  """
  text = str(entry)
  clean = [False] * len(rows)
  # An analytic model that was fit on nothing has made a complete declaration and
  # there is nothing left to compare it against.
  if text == NO_TRAINING_DATA:
    return clean
  if not text or scope is None:
    return None
  fields = text.split(FINGERPRINT_SEPARATOR)
  corpus = fields[0]
  if not corpus:
    return None
  if corpus != scope.corpus:
    # A model fit on some other corpus was not fit on these rows, whatever else its
    # declaration says about that corpus's own splits and seeds.
    return clean
  episodes: List[str] = []
  for term in fields[1:]:
    key, separator, value = term.partition("=")
    if not separator or key not in FINGERPRINT_KEYS or not value:
      return None
    if key == "episode":
      episodes.append(value)
      continue
    if key == "seeds":
      declared_range = _declared_seed_range(value)
      if declared_range is None or scope.seed_range is None:
        return None
      if not _ranges_meet(declared_range, scope.seed_range):
        return clean
      continue
    scored_value = {
      "split": scope.split,
      "epoch": scope.seed_epoch,
      "pack": scope.pack_id,
      "task": scope.task_id,
    }[key]
    if not scored_value:
      return None
    if scored_value != value:
      return clean
  if not episodes:
    # The declaration reached this point naming the corpus being scored and nothing
    # that excluded these rows, so it covers all of them.
    return [True] * len(rows)
  wanted = set(episodes)
  return [row in wanted for row in rows]


@dataclass(frozen=True)
class OverlapCount:
  """How much of what was scored the model had already been fit on.

  Three answers, kept apart on purpose. ``LEAK`` is an established overlap between a
  declaration and the scope being scored. ``LEAK-UNCHECKABLE`` is a declaration this
  harness had no way to compare, which withholds the row rather than clearing it.
  ``CLEAN`` is every declaration compared and none of them reaching these rows.

  The counts are taken twice and the duplicate-preserving one is the one that
  matters. Computing the intersection as the size of a ``frozenset`` collapses
  repeated rows: a fixture that presents the same transition forty times reports an
  overlap of one, and a fraction built on that reads as a rounding error.

  :attr:`leaked` is the single boolean the callers read, so it fails safe and is
  true for an uncheckable declaration as well as for an established one. A caller
  that wants the difference reads :attr:`verdict` or :attr:`established`; a caller
  that reads one boolean must not be handed the flattering half of a distinction it
  did not ask about.
  """

  n_rows: int
  n_rows_seen: int
  n_distinct_rows: int
  n_distinct_seen: int
  unresolved: Tuple[str, ...] = ()
  matched: Tuple[str, ...] = ()
  scope: str = ""

  @property
  def established(self) -> bool:
    """An overlap that was measured, not one that could not be ruled out."""
    return self.n_rows_seen > 0

  @property
  def leaked(self) -> bool:
    """Whether this row is anything other than clear. Fails safe."""
    return self.established or bool(self.unresolved)

  @property
  def verdict(self) -> str:
    if self.established:
      return "LEAK"
    if self.unresolved:
      return "LEAK-UNCHECKABLE"
    return "CLEAN"

  @property
  def fraction_rows_seen(self) -> float:
    return self.n_rows_seen / self.n_rows if self.n_rows else 0.0

  def format(self) -> str:
    parts: List[str] = []
    if self.established:
      parts.append(
        f"LEAK: {self.n_rows_seen} of {self.n_rows} scored rows "
        f"({self.fraction_rows_seen:.3f}) fall inside the declared training scope "
        f"{list(self.matched)}; {self.n_distinct_seen} of {self.n_distinct_rows} "
        "distinct rows. The duplicate-preserving count is the left-hand one and it "
        "is the one to read."
      )
    if self.unresolved:
      parts.append(
        f"LEAK-UNCHECKABLE: the declared training fingerprints {list(self.unresolved)} "
        "name a scope this call was not given, so they were not compared against the "
        f"{self.n_rows} scored rows. The row is withheld and not cleared: a leak "
        "check that could not read a declaration has established nothing. Pass "
        "scope=ScoringScope(...) naming the corpus, split, seed epoch, pack, task "
        "and seed range being scored."
      )
    if not parts:
      where = f" ({self.scope})" if self.scope else ""
      parts.append(
        f"training overlap: none of {self.n_rows} scored rows falls inside any "
        f"declared training scope{where}"
      )
    return " ".join(parts)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "n_rows": self.n_rows,
      "n_rows_seen": self.n_rows_seen,
      "n_distinct_rows": self.n_distinct_rows,
      "n_distinct_seen": self.n_distinct_seen,
      "leaked": self.leaked,
      "established": self.established,
      "verdict": self.verdict,
      "unresolved": list(self.unresolved),
      "matched": list(self.matched),
      "scope": self.scope,
      "fraction_rows_seen": self.fraction_rows_seen,
    }


def training_overlap(
  scored_rows: Sequence[str],
  training_fingerprints: Sequence[str],
  scope: Optional[ScoringScope] = None,
) -> OverlapCount:
  """Compare what the model declared it was fit on against what is being scored.

  Each declared fingerprint is read as ``corpus`` followed by zero or more
  ``key=value`` terms drawn from :data:`FINGERPRINT_KEYS`. A term that names a split,
  a seed epoch, a pack, a task or a seed range is compared against the matching field
  of ``scope``; a term that names an episode is compared against the scored row
  identifiers. Terms within one fingerprint are conjunctive, repeated ``episode``
  terms are a list, and separate fingerprints are a union over rows. The count is
  duplicate preserving: forty scored rows covered by one declaration are forty rows.

  A declaration whose corpus is not the scope's is clean and not withheld. That is
  the direction a leak check is easiest to get wrong in the flattering way and also
  in the useless way: a corpus named ``droid`` is a statement that the model was fit
  on something else, and a check that withheld every row for it would refuse every
  submission that honestly declared any outside training data at all.

  Without a ``scope`` the only decidable declaration is
  :data:`NO_TRAINING_DATA`, and every other one comes back ``LEAK-UNCHECKABLE``.
  That is the intended behaviour and it is the whole point of the change: the
  previous version took a sequence of episode identifiers and a sequence of
  corpus, split and seed-range names, intersected them, and found nothing every
  single time, so a model that had declared it was fit on the split being scored
  was printed as clean.
  """
  if isinstance(scored_rows, str) or isinstance(training_fingerprints, str):
    raise MetricRefused(
      "training_overlap takes two sequences of strings and one of these arguments is "
      "a single string. Iterating it would read one declaration as one corpus per "
      "character, none of which matches anything, and the answer would come back "
      "CLEAN: a false clearance produced by parsing the declaration in the wrong "
      "vocabulary, which is the exact defect this function was rewritten to remove."
    )
  rows = [str(r) for r in scored_rows]
  entries = sorted(set(str(f) for f in training_fingerprints))
  seen = [False] * len(rows)
  unresolved: List[str] = []
  matched: List[str] = []
  for entry in entries:
    coverage = _rows_covered(entry, rows, scope)
    if coverage is None:
      unresolved.append(entry)
      continue
    if any(coverage):
      matched.append(entry)
    for index, covered in enumerate(coverage):
      if covered:
        seen[index] = True
  return OverlapCount(
    n_rows=len(rows),
    n_rows_seen=sum(1 for flag in seen if flag),
    n_distinct_rows=len(set(rows)),
    n_distinct_seen=len({rows[i] for i, flag in enumerate(seen) if flag}),
    unresolved=tuple(unresolved),
    matched=tuple(matched),
    scope=scope.describe() if scope is not None else "",
  )


# -- declared compute ----------------------------------------------------------


@dataclass(frozen=True)
class ComputeDeclaration:
  """Median forward passes, median wall clock, and the submitter's hardware string.

  ``scored`` is a field rather than a docstring sentence so that a sort written
  against this object has to read a ``False`` before it can use any of these
  numbers as a key. A leaderboard that hides compute lets a sixty-four-card entry
  look like a single-card one, and the published latency tax between those two
  regimes is three to four times.
  """

  n: int
  median_forward_passes: float
  median_wall_clock_s: float
  hardware: str = UNDECLARED_HARDWARE
  scored: bool = False

  @classmethod
  def of(
    cls,
    forward_passes: Sequence[int],
    wall_clock_s: Sequence[float],
    hardware: str = UNDECLARED_HARDWARE,
  ) -> "ComputeDeclaration":
    if len(forward_passes) != len(wall_clock_s):
      raise MetricRefused(
        f"{len(forward_passes)} forward-pass counts against {len(wall_clock_s)} wall "
        "clock readings; they are two columns of the same rows."
      )
    if not forward_passes:
      return cls(n=0, median_forward_passes=0.0, median_wall_clock_s=0.0, hardware=hardware)
    return cls(
      n=len(forward_passes),
      median_forward_passes=_median_of([float(v) for v in forward_passes]),
      median_wall_clock_s=_median_of([float(v) for v in wall_clock_s]),
      hardware=str(hardware) if hardware else UNDECLARED_HARDWARE,
    )

  def format(self) -> str:
    return (
      f"declared compute ({NOT_RANKED}): median {self.median_forward_passes:g} forward "
      f"passes, median {self.median_wall_clock_s:.4f} s wall clock, hardware "
      f"{self.hardware}"
    )

  def to_dict(self) -> Dict[str, Any]:
    return {
      "n": self.n,
      "median_forward_passes": self.median_forward_passes,
      "median_wall_clock_s": self.median_wall_clock_s,
      "hardware": self.hardware,
      "scored": self.scored,
      "header": NOT_RANKED,
    }


# -- the pixel column ----------------------------------------------------------


@dataclass(frozen=True)
class PixelColumn:
  """PSNR and SSIM, printed under a header literally reading ``NOT RANKED``.

  The column exists for exactly one demonstration: the ``PixelPerfect`` control
  replays ground-truth frames, tops this column outright, and loses the E-AURC
  ranking anyway. Shipping no pixel metric at all is defensible hygiene and forfeits
  that demonstration, and a demonstration convinces a video-model author where a
  citation does not.

  ``rankable`` is a property that returns ``False`` and takes no arguments, so there
  is no configuration under which this column enters an ordering.
  """

  psnr: Optional[float] = None
  ssim: Optional[float] = None
  declared_by: str = "not declared"

  @property
  def rankable(self) -> bool:
    return False

  @property
  def declared(self) -> bool:
    return self.psnr is not None or self.ssim is not None

  def format(self) -> str:
    if not self.declared:
      return f"pixel ({NOT_RANKED}): not declared"
    psnr = "n/a" if self.psnr is None else f"{self.psnr:.2f} dB"
    ssim = "n/a" if self.ssim is None else f"{self.ssim:.4f}"
    return f"pixel ({NOT_RANKED}): PSNR {psnr}, SSIM {ssim}, from {self.declared_by}"

  def to_dict(self) -> Dict[str, Any]:
    return {
      "header": NOT_RANKED,
      "psnr": self.psnr,
      "ssim": self.ssim,
      "declared_by": self.declared_by,
      "rankable": self.rankable,
    }


# -- below the fold ------------------------------------------------------------


def ece_not_ranked(
  confidences: Sequence[float],
  hits: Sequence[int],
  n_bins: int = 10,
) -> float:
  """Expected calibration error, on equal-width bins, printed below the fold.

  Not a headline and not a ranking key. The published position is that ECE and
  AUROC do not directly test whether uncertainty changes act or defer decisions,
  which is the only question this benchmark asks, so both are available and both
  print under ``NOT RANKED``. The bin count is an absolute constant argument and the
  bins are equal width rather than equal mass, because equal-mass bins are a
  quantile of the data being scored.
  """
  if len(confidences) != len(hits):
    raise MetricRefused("ECE needs one confidence per outcome")
  n = len(confidences)
  if n == 0:
    raise MetricRefused("ECE over zero episodes is not a calibration")
  if int(n_bins) < 1:
    raise MetricRefused("ECE needs at least one bin")
  bins = int(n_bins)
  totals = [0] * bins
  correct = [0] * bins
  confidence_sum = [0.0] * bins
  for c, hit in zip(confidences, hits):
    value = float(c)
    index = min(bins - 1, max(0, int(value * bins)))
    totals[index] += 1
    correct[index] += int(bool(hit))
    confidence_sum[index] += value
  error = 0.0
  for index in range(bins):
    if totals[index] == 0:
      continue
    accuracy = correct[index] / totals[index]
    mean_confidence = confidence_sum[index] / totals[index]
    error += (totals[index] / n) * abs(accuracy - mean_confidence)
  return error


def auroc_not_ranked(confidences: Sequence[float], hits: Sequence[int]) -> Optional[float]:
  """Area under the ROC for confidence against correctness, printed below the fold.

  ``None`` when one class is empty, which is a real state and not a 0.5: with no
  errors there is nothing for the confidence ordering to separate, and printing 0.5
  there reads as a measurement of a model that was never tested.
  """
  if len(confidences) != len(hits):
    raise MetricRefused("AUROC needs one confidence per outcome")
  positives = [float(c) for c, h in zip(confidences, hits) if bool(h)]
  negatives = [float(c) for c, h in zip(confidences, hits) if not bool(h)]
  if not positives or not negatives:
    return None
  wins = 0.0
  for p in positives:
    for q in negatives:
      if p > q:
        wins += 1.0
      elif p == q:
        wins += 0.5
  return wins / (len(positives) * len(negatives))


def median_of(values: Sequence[float]) -> float:
  """The build's one median, re-exported so no module writes a second one."""
  if not values:
    raise MetricRefused("no median over an empty sample")
  return _median_of([float(v) for v in values])


__all__ = [
  "BENCH_CORPUS",
  "FINGERPRINT_KEYS",
  "FINGERPRINT_SEPARATOR",
  "LOOSE_TOLERANCE",
  "NOT_RANKED",
  "NO_TRAINING_DATA",
  "TRUST_RISK",
  "UNDECLARED_HARDWARE",
  "ComputeDeclaration",
  "LooseCriterionMissing",
  "OverlapCount",
  "PixelColumn",
  "ScoringScope",
  "SuccessGap",
  "TolerancePredictionError",
  "auroc_not_ranked",
  "ece_not_ranked",
  "in_tolerance",
  "loose_hit",
  "loose_tolerance_for",
  "median_of",
  "success_gap",
  "tolerance_error",
  "training_overlap",
  "trust_horizon",
]
