"""The five DeferLab tasks, and the one structural property that decides whether
the abstention half of the benchmark measures a model or measures a label.

A task here is a declaration, not a generator. It fixes the state fields and their
units, the per-field tolerance every downstream metric divides by, the horizons
that may be scored, and the episode count that buys a rankable row. The episodes
themselves arrive as a versioned scene pack (:mod:`plr_lr.learn.bench.packio`), so
nothing in this module renders, simulates or imports anything from the vision side
of the build.

Two refusals live here.

**Unknowable episodes are interleaved inside every task and never split out into a
task of their own.** This is the whole reason the abstention metrics are worth
computing. A dedicated "ambiguous" task is won outright by ``if task_id == "T6":
abstain``, which is four tokens long, needs no model, and would top the abstention
column of any leaderboard that shipped one. :func:`check_interleaving` refuses a
pack whose unknowable episodes sit in a contiguous block, whose episode
identifiers carry a knowability token, or whose unknowable fraction leaves
:data:`plr_lr.learn.selective.UNKNOWABLE_BAND`. Outside that band the abstention
numbers are a statement about the base rate rather than about the model, which is
refusal rule R6.

**A task declares its hidden latent and the exact ceiling that latent buys.**
``hidden_latent`` names the coefficient keys an unknowable episode redraws and
``hidden_branches`` says how many values the support offers, so the ceiling a
non-abstaining model faces on those episodes is ``1 / hidden_branches`` and is a
number in the declaration rather than an adjective in a docstring. The support
itself lives with the transition constants in :mod:`plr_lr.learn.bench.packio`,
which is also where the build-time separation gate enforces the claim: every pair
of branches more than two tolerances apart, every branch more than one tolerance
from the nominal. An earlier version of this file said only that "a non-abstaining
model's ceiling is chance" and nothing checked it; on the pack the CLI actually
scored, 80 of 80 S1 horizon-1 episodes labelled unknowable had a terminal state
that was byte-identical for every value of the hidden latent.

The claim the gate establishes is exact and it is worth stating exactly. Holding an
episode's other coefficients where they are, its terminal state takes
``hidden_branches`` values more than two tolerances apart, so any prediction is
inside tolerance of at most one of them and the hidden value is drawn uniformly and
independently of everything observable: the hit probability is at most
``1 / hidden_branches``. It does not say that a particular model scores zero on
these episodes. A model whose own error is wide can land on a branch by
cancellation, and the declared floor does on 3 of the 80 unknowable S1 horizon-1
episodes of the pack ``python -m plr_lr.learn.bench run`` builds by default (400
episodes per task, seed 20260803), which is its noise rather than a signal it
recovered. The count is a property of the draw and not a constant: the same floor
on the same seed lands on 4 of 40 at ``--episodes 200``, so a citation of it
without the episode count beside it is not reproducible.

**Tolerance is per field and is never collapsed into one weight.** S4 asks for half
a millimetre and half a degree jointly. Adding a millimetre to a degree needs a
weight, a weight chosen after seeing the errors is a fitted parameter, and the
resulting scalar is labelled in whichever unit was written first. Every criterion
in :mod:`plr_lr.learn.bench.metrics` is therefore per-field error divided by that
field's tolerance and then a max over fields, which has no weight in it at all.

The action layout is :data:`plr_lr.learn.types.ACTION_FIELDS` for every task,
including the two whose physical action space is a discrete lab verb. The discrete
verb rides in the ``p3`` column and the continuous magnitudes ride in ``p0``,
``p1`` and ``p2``, so a task-scalar task still answers the interventional gate,
which perturbs the first three columns. A task whose terminal state ignored those
columns would pass the gate only by accident.

Two later tiers are named so nobody rebuilds them: **Tier D** replays recorded
episodes offline with no arm present at score time, and **Tier H** closes the loop
on hardware. Neither ships here, and every row this module feeds reads
``basis: simulated`` until a hardware run record with a device serial hash exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..selective import MIN_N_RANKABLE, UNKNOWABLE_BAND
from ..types import ACTION_FIELDS

#: The closed enum of knowability reasons recorded per episode. A free-text reason
#: would let a fixture author write "probably fine" and have it counted as a
#: construction, so the set is closed and a pack carrying anything else is refused.
KNOWABLE = "knowable"
INDEPENDENT_LATENT = "independent_latent"
FULL_OCCLUSION = "full_occlusion"

KNOWABILITY_REASONS: Tuple[str, ...] = (KNOWABLE, INDEPENDENT_LATENT, FULL_OCCLUSION)
UNKNOWABLE_REASONS: Tuple[str, ...] = (INDEPENDENT_LATENT, FULL_OCCLUSION)

#: Substrings an episode identifier may not contain. An identifier that names the
#: answer is the same exploit as a dedicated unknowable task, one layer down.
FORBIDDEN_ID_TOKENS: Tuple[str, ...] = (
  "unknow",
  "knowable",
  "ambiguous",
  "occlud",
  "occlusion",
  "independent",
  "abstain",
  "ceiling",
)

#: S3's dispense target, in microlitres. Published because it is the task
#: definition rather than a transition constant.
S3_TARGET_UL = 10.0

#: S2's physically reachable residual range, in microlitres.
S2_RESIDUAL_RANGE: Tuple[float, float] = (0.0, 5.0)

#: S5 counts a tip as engaged inside this band, in millimetres. The band is the
#: task definition; the tolerance below is derived from it and says so.
S5_ENGAGEMENT_BAND: Tuple[float, float] = (1.5, 3.0)


class TaskDefinitionError(ValueError):
  """Raised when a task declaration could not support a scored row."""


class InterleavingRefused(ValueError):
  """Raised when a task's unknowable episodes are separable without a model.

  The permissive alternative is to trust the fixture author. That has already been
  measured to fail in this shape: an unknowable episode set that sits in one block
  at the end of a pack is recoverable from the episode index alone, and a
  submission that noticed would score a perfect refusal recall while predicting
  nothing.
  """


@dataclass(frozen=True)
class BenchTask:
  """One scored task: its state layout, its tolerance, and what may be asked of it.

  Args:
    task_id: the short identifier printed on every row. Never encodes knowability.
    title: the human-readable name.
    state_fields: field order of the terminal state vector. Declared here and
      nowhere else, because a state vector whose column order is implicit is the
      fastest way to score a model against scrambled columns.
    state_units: one unit per field, carried so a millimetre and a microlitre can
      never be compared without going through ``tolerance`` first.
    tolerance: per-field tolerance in the field's own units. Strictly positive:
      every metric divides by it, so a zero entry reports either a perfect score or
      a NaN depending on rounding, and both read as a result.
    horizons: the action-sequence lengths this task may be scored at. Rows at
      different horizons are different measurements and never share a table.
    n_episodes: how many episodes the pack is expected to carry for this task.
    action_space: how the five action columns are meant to be read.
    action_verbs: the discrete lab verbs this task's actions carry, in code order.
      Empty for a task whose action is purely continuous.
    verb_column: which action column holds the verb code. Never ``p0``, ``p1`` or
      ``p2``: those three are what the interventional gate perturbs, and a gate
      that perturbed a discrete code would be asking a model to extrapolate a verb
      that does not exist rather than asking whether it reads its actions at all.
    unknowable_construction: which of the two constructions in the plan builds this
      task's unknowable episodes. Recorded so a reader can check the claim rather
      than take it.
    hidden_latent: the coefficient keys an unknowable episode of this task redraws.
      Named here so the claim is in the declaration; the values live with the
      transition constants and the gate that enforces their separation.
    hidden_branches: how many distinct hidden values the support offers. The
      ceiling a non-abstaining model faces on an unknowable episode of this task is
      one over this number, and that only holds while the separation gate holds, so
      a task declaring fewer than two branches is refused: one branch is a
      prediction, not an ambiguity.
    renderer: the name of the observation model that produced the frames. A string,
      never an import: the entire interface to the vision side is a file format.
    loose_criterion: the binary criterion a policy layer would use, against which
      the tolerance criterion is differenced to produce SuccessGap.
  """

  task_id: str
  title: str
  state_fields: Tuple[str, ...]
  state_units: Tuple[str, ...]
  tolerance: Tuple[float, ...]
  horizons: Tuple[int, ...]
  n_episodes: int
  action_space: str
  action_verbs: Tuple[str, ...]
  unknowable_construction: str
  renderer: str
  loose_criterion: str
  verb_column: int = 3
  hidden_latent: Tuple[str, ...] = ()
  hidden_branches: int = 0
  notes: Tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if self.action_verbs and int(self.verb_column) not in (3, 4):
      raise TaskDefinitionError(
        f"task {self.task_id} puts its verb code in column {self.verb_column}. Only "
        "p3 and the gripper column may carry one; p0, p1 and p2 are what the "
        "interventional gate perturbs."
      )
    if not isinstance(self.task_id, str) or not self.task_id:
      raise TaskDefinitionError("BenchTask.task_id must be a non-empty string")
    lowered = self.task_id.lower()
    for token in FORBIDDEN_ID_TOKENS:
      if token in lowered:
        raise TaskDefinitionError(
          f"task_id {self.task_id!r} contains {token!r}. A task whose name states the "
          "knowability of its episodes can be answered by reading the name."
        )
    dim = len(self.state_fields)
    if dim < 1:
      raise TaskDefinitionError(f"task {self.task_id} declares no state fields")
    if len(self.state_units) != dim or len(self.tolerance) != dim:
      raise TaskDefinitionError(
        f"task {self.task_id} declares {dim} fields, {len(self.state_units)} units and "
        f"{len(self.tolerance)} tolerances. All three are the same list read three ways."
      )
    for name, value in zip(self.state_fields, self.tolerance):
      number = float(value)
      if not number > 0.0 or number != number or number in (float("inf"),):
        raise TaskDefinitionError(
          f"task {self.task_id} field {name!r} declares tolerance {value!r}. Every "
          "criterion here divides by this number, so a zero or non-finite entry "
          "reports a perfect score or a NaN depending on rounding."
        )
    if not self.horizons:
      raise TaskDefinitionError(f"task {self.task_id} declares no horizons")
    if list(self.horizons) != sorted(set(self.horizons)) or self.horizons[0] < 1:
      raise TaskDefinitionError(
        f"task {self.task_id} horizons {list(self.horizons)} must be distinct, "
        "ascending and at least 1"
      )
    if int(self.n_episodes) < MIN_N_RANKABLE:
      raise TaskDefinitionError(
        f"task {self.task_id} declares n={self.n_episodes} episodes, below the "
        f"rankable floor of {MIN_N_RANKABLE}. A task that can never produce a "
        "rankable row does not belong in the task list; run it as a pilot instead."
      )
    if self.unknowable_construction not in UNKNOWABLE_REASONS:
      raise TaskDefinitionError(
        f"task {self.task_id} declares unknowable_construction="
        f"{self.unknowable_construction!r}, which is not one of "
        f"{list(UNKNOWABLE_REASONS)}. Unknowability is constructed, not asserted."
      )
    object.__setattr__(self, "hidden_latent", tuple(str(k) for k in self.hidden_latent))
    object.__setattr__(self, "hidden_branches", int(self.hidden_branches))
    if not self.hidden_latent:
      raise TaskDefinitionError(
        f"task {self.task_id} names no hidden latent. Its unknowable episodes would "
        "then differ from its knowable ones in nothing a checker can evaluate, which "
        "is the state this declaration exists to make impossible: the label would be "
        "an assertion by the fixture author and the abstention columns would measure "
        "that author rather than the model."
      )
    if self.hidden_branches < 2:
      raise TaskDefinitionError(
        f"task {self.task_id} declares {self.hidden_branches} hidden branches. One "
        "branch is a prediction, not an ambiguity: a model that knows the "
        "construction predicts it and is right every time, so the episode is "
        "knowable and the label is false."
      )

  @property
  def state_dim(self) -> int:
    return len(self.state_fields)

  @property
  def chance_ceiling(self) -> float:
    """The best a non-abstaining model can do on this task's unknowable episodes.

    One over the number of hidden branches, and true only because the separation
    gate in :mod:`plr_lr.learn.bench.packio` refuses any episode whose branches land
    within two tolerances of each other: two branches closer than that are both hit
    by one prediction placed between them, and then the ceiling is two over the
    branch count and this number is an overstatement of the difficulty.
    """
    return 1.0 / float(self.hidden_branches)

  @property
  def max_horizon(self) -> int:
    return int(self.horizons[-1])

  @property
  def unknowable_band(self) -> Tuple[float, float]:
    """The band the per-task unknowable fraction has to land in.

    Read from :data:`plr_lr.learn.selective.UNKNOWABLE_BAND` rather than restated,
    so the benchmark and the metric core cannot drift into two different bands and
    disagree about which rows were refused.
    """
    return UNKNOWABLE_BAND

  def tolerance_for(self, field: str) -> float:
    index = self.state_fields.index(field)
    return float(self.tolerance[index])

  def describe(self) -> str:
    fields = ", ".join(
      f"{name} ({unit}) +/- {tol:g}"
      for name, unit, tol in zip(self.state_fields, self.state_units, self.tolerance)
    )
    lines = [
      f"{self.task_id} {self.title}",
      f"  state:      {fields}",
      f"  horizons:   {list(self.horizons)}, n={self.n_episodes} episodes",
      f"  actions:    {self.action_space} in the {list(ACTION_FIELDS)} layout"
      + (
        f", verbs {list(self.action_verbs)} in {ACTION_FIELDS[self.verb_column]}"
        if self.action_verbs
        else ""
      ),
      f"  unknowable: {self.unknowable_construction}, interleaved, band "
      f"[{self.unknowable_band[0]:.2f}, {self.unknowable_band[1]:.2f}]",
      f"  hidden:     {list(self.hidden_latent)} over {self.hidden_branches} "
      f"branches, non-abstaining ceiling {self.chance_ceiling:.4f}",
      f"  frames:     {self.renderer}",
      f"  loose:      {self.loose_criterion}",
    ]
    lines.extend(f"  note: {note}" for note in self.notes)
    return "\n".join(lines)

  def to_dict(self) -> Dict[str, object]:
    return {
      "task_id": self.task_id,
      "title": self.title,
      "state_fields": list(self.state_fields),
      "state_units": list(self.state_units),
      "tolerance": [float(t) for t in self.tolerance],
      "horizons": [int(k) for k in self.horizons],
      "n_episodes": int(self.n_episodes),
      "action_space": self.action_space,
      "action_verbs": list(self.action_verbs),
      "verb_column": int(self.verb_column),
      "unknowable_construction": self.unknowable_construction,
      "hidden_latent": list(self.hidden_latent),
      "hidden_branches": int(self.hidden_branches),
      "chance_ceiling": self.chance_ceiling,
      "renderer": self.renderer,
      "loose_criterion": self.loose_criterion,
      "unknowable_band": [float(self.unknowable_band[0]), float(self.unknowable_band[1])],
    }


S1 = BenchTask(
  task_id="S1",
  title="reach-regime",
  state_fields=("x", "y", "z", "rz"),
  state_units=("mm", "mm", "mm", "rad"),
  tolerance=(2.0, 2.0, 2.0, 0.05),
  horizons=(1, 2, 4, 8),
  n_episodes=400,
  action_space="ee_delta_mm",
  action_verbs=(),
  unknowable_construction=INDEPENDENT_LATENT,
  hidden_latent=("bias_mm",),
  hidden_branches=4,
  renderer="none: state-only, the tracking regime is not a visual quantity",
  loose_criterion="terminal pose within 10 mm, the criterion a reach policy reports",
  notes=(
    "the tracking gain switches at a workspace boundary the observed prefix does "
    "not cross, and a drift onset step is hidden inside the horizon",
    "the unknowable episodes carry a hidden step change in the tracking reference "
    "that lands on the first commanded move, so it is in the terminal state at "
    "every horizon this task is scored at, horizon 1 included",
    "the earlier construction hid the latent inside a drift that only fired from "
    "the second step, which left all 80 unknowable horizon-1 episodes byte-identical "
    "for every value of the latent while still labelled unknowable. That is why the "
    "latent now acts from the first step and why the build gate exists",
    "four hidden values, none of them the published nominal, so the terminal a "
    "prediction with the latent at its nominal produces is more than a tolerance "
    "from every branch and a model that guesses a branch is right one time in four",
  ),
)

S2 = BenchTask(
  task_id="S2",
  title="residual-drawdown",
  state_fields=("residual_uL",),
  state_units=("uL",),
  tolerance=(0.30,),
  horizons=(1, 2, 3),
  n_episodes=400,
  action_space="task_scalar",
  action_verbs=("PROCEED", "REWASH", "EXTEND_DRY"),
  unknowable_construction=INDEPENDENT_LATENT,
  hidden_latent=("wet_uL",),
  hidden_branches=3,
  renderer="tip_view",
  loose_criterion="reads dry: residual below 1.0 uL, the criterion a QC step reports",
  notes=(
    "p0 is wash volume in uL, p1 is blot displacement in mm, p2 is dry height in "
    "mm, and p3 carries the verb code. All three continuous columns move the "
    "residual, so the interventional gate is answerable on a task-scalar task",
    "the wet-tip regime is a retained film, in uL, drawn independently of everything "
    "the render reads and added to the residual at the end of the sequence. It is "
    "added at the end rather than carried through because the drawdown is "
    "multiplicative: a carry added at the first step is shrunk by every step after "
    "it, and by the third step two branches that started a tolerance apart are not",
    "three hidden values, none of them the published nominal of zero retention",
  ),
)

S3 = BenchTask(
  task_id="S3",
  title="dispense-volume",
  state_fields=("volume_uL",),
  state_units=("uL",),
  tolerance=(0.50,),
  horizons=(1, 2, 3),
  n_episodes=400,
  action_space="task_scalar",
  action_verbs=("PROCEED", "TOP_UP", "ASPIRATE"),
  unknowable_construction=INDEPENDENT_LATENT,
  hidden_latent=("bias_uL",),
  hidden_branches=4,
  renderer="column_view",
  loose_criterion=f"looks filled: within 2.0 uL of the {S3_TARGET_UL:g} uL target",
  notes=(
    "p0 is the commanded volume delta in uL, p1 a lateral offset in mm that costs "
    "wall loss, p2 the tip height in mm that costs droplet retention",
    "the hidden latent is a per-episode dispense calibration offset in uL that lands "
    "on the first commanded step. The frame is a function of the volume before that "
    "step, so it is independent of the offset by construction and the ceiling is a "
    "proof rather than a claim about how hard the frame is to read",
    "it replaced a concentration multiplier on the commanded stroke, which had no "
    "effect at all on the third of the steps whose commanded stroke is zero: 29 of "
    "80 unknowable horizon-1 episodes were fully determined under that construction",
    "four hidden values, none of them the published nominal of zero offset",
  ),
)

S4 = BenchTask(
  task_id="S4",
  title="plate-seating",
  state_fields=("dx", "dy", "drz", "seated"),
  state_units=("mm", "mm", "deg", "bool"),
  tolerance=(0.5, 0.5, 0.5, 0.5),
  horizons=(1, 2, 4),
  n_episodes=400,
  action_space="ee_delta_mm",
  action_verbs=("PROCEED", "RESEAT"),
  unknowable_construction=FULL_OCCLUSION,
  hidden_latent=("occ_dx_mm", "occ_dy_mm"),
  hidden_branches=4,
  renderer="microplate",
  loose_criterion="looks seated: the plate reads flush at the corner the camera sees",
  verb_column=4,
  notes=(
    "p0 and p1 are the corrective dx and dy in mm, p2 is press depth in mm, p3 is "
    "drz in degrees, and the gripper column carries the RESEAT verb",
    "half a millimetre and half a degree jointly, as one number, because the "
    "criterion is per-field error over that field's tolerance and then a max",
    "the unknowable episodes place an occluder covering 100 percent of the target "
    "well ROI in every frame, checked pixel-wise against the boxes the renderer "
    "returns, so the deciding offset is never in view",
    "the hidden offset is a nest offset at the covered corner, added to the terminal "
    "before the seating test. It is not an initial condition the corrections then "
    "act on: a RESEAT multiplies the offset by about a third, so two branches that "
    "started four millimetres apart land a quarter of a millimetre apart after two "
    "reseats and the tolerance is half a millimetre",
    "two values on each axis, neither of them the published nominal of zero, so four "
    "branches and a ceiling of 1/4",
  ),
)

S5 = BenchTask(
  task_id="S5",
  title="tip-engagement",
  state_fields=("z_engage_mm", "attached"),
  state_units=("mm", "bool"),
  tolerance=(0.75, 0.5),
  horizons=(1, 2, 4),
  n_episodes=400,
  action_space="ee_delta_mm",
  action_verbs=("PROCEED", "RELEASE"),
  unknowable_construction=INDEPENDENT_LATENT,
  hidden_latent=("gap_mm",),
  hidden_branches=4,
  renderer="tip_view",
  loose_criterion=(
    f"engagement inside [{S5_ENGAGEMENT_BAND[0]:g}, {S5_ENGAGEMENT_BAND[1]:g}] mm"
  ),
  verb_column=4,
  notes=(
    "p2 is the commanded delta z in mm, p0 and p1 are lateral offsets that cost "
    "engagement depth and prevent attachment, and the gripper column carries the "
    "RELEASE verb",
    "the engagement tolerance is half the width of the "
    f"[{S5_ENGAGEMENT_BAND[0]:g}, {S5_ENGAGEMENT_BAND[1]:g}] mm band, so a "
    "prediction inside tolerance cannot move the engagement verdict across more "
    "than one band edge. It is derived from the band and is not a percentile",
    "the apparent engagement depth is visible and the seating gap behind the tip "
    "shoulder is not. The unknowable episodes draw that gap independently of the "
    "frame and add it to the terminal depth before the attachment test, which is "
    "the honest ceiling rather than a hard task",
    "the earlier construction hid a boolean seal instead. A boolean can only take "
    "the truth to two values, one of which was the nominal, and on any episode whose "
    "depth already sat outside the band it took it to neither: 63 of 80 unknowable "
    "horizon-4 episodes were fully determined under it",
    "four hidden values, none of them the published nominal of no gap: a tip short "
    "of the mandrel and a tip driven past its shoulder, in two magnitudes each",
  ),
)

#: The five scored tasks, in order. Tier S, CPU-only.
TASKS: Tuple[BenchTask, ...] = (S1, S2, S3, S4, S5)

TASKS_BY_ID: Dict[str, BenchTask] = {task.task_id: task for task in TASKS}


def task_for(task_id: str) -> BenchTask:
  """Look one task up by id, refusing an unknown one rather than inventing it."""
  try:
    return TASKS_BY_ID[task_id]
  except KeyError:
    raise TaskDefinitionError(
      f"no task {task_id!r}. Known tasks are {sorted(TASKS_BY_ID)}. A pack naming a "
      "task this harness does not define cannot have its tolerance checked, and a "
      "tolerance guessed at the scoring step is not the task's tolerance."
    ) from None


@dataclass(frozen=True)
class InterleavingReport:
  """Whether a task's unknowable episodes can be found without predicting anything.

  ``ok`` is False for four different exploits and each one is named, because the
  repair differs: a fraction outside the band is a fixture-mix problem, a token in
  the identifier is a naming problem, a contiguous block is an ordering problem,
  and an all-or-nothing split is a construction problem.
  """

  task_id: str
  n: int
  n_unknowable: int
  fraction: float
  in_band: bool
  contiguous: bool
  id_tell: Optional[str]
  band: Tuple[float, float] = UNKNOWABLE_BAND

  @property
  def ok(self) -> bool:
    if self.n <= 0:
      return False
    if self.n_unknowable <= 0 or self.n_unknowable >= self.n:
      return False
    return self.in_band and not self.contiguous and self.id_tell is None

  def reasons(self) -> Tuple[str, ...]:
    out: List[str] = []
    if self.n <= 0:
      out.append("the task carried no episodes, so nothing was checked")
    if self.n_unknowable <= 0:
      out.append(
        "no episode is unknowable, so the abstention columns measure nothing and a "
        "model that never refuses is indistinguishable from one that refuses well"
      )
    elif self.n_unknowable >= self.n:
      out.append(
        "every episode is unknowable, which is the dedicated-unknowable-task exploit "
        "wearing a task id that does not say so"
      )
    if not self.in_band:
      out.append(
        f"unknowable fraction {self.fraction:.3f} is outside "
        f"[{self.band[0]:.2f}, {self.band[1]:.2f}], so the abstention numbers are "
        "dominated by the base rate rather than by the model (R6)"
      )
    if self.contiguous:
      out.append(
        "the unknowable episodes form one contiguous block, so they are recoverable "
        "from the episode index without predicting anything"
      )
    if self.id_tell is not None:
      out.append(f"episode identifiers give it away: {self.id_tell}")
    return tuple(out)

  def format(self) -> str:
    verdict = "ok" if self.ok else "REFUSED"
    head = (
      f"interleaving {self.task_id}: {verdict} "
      f"({self.n_unknowable}/{self.n} unknowable, fraction {self.fraction:.3f})"
    )
    return "\n".join([head] + [f"  {reason}" for reason in self.reasons()])


def check_interleaving(
  task_id: str,
  episode_ids: Sequence[str],
  knowability: Sequence[str],
) -> InterleavingReport:
  """Can this task's unknowable episodes be picked out without a model?

  Four exploits, all of which have to be closed at once for the abstention columns
  to be about the model. Anything this function refuses would otherwise be scored,
  and it would be scored generously: every one of the four lets a submission
  collect refusal credit for free.
  """
  ids = [str(e) for e in episode_ids]
  labels = [str(k) for k in knowability]
  if len(ids) != len(labels):
    raise InterleavingRefused(
      f"{len(ids)} episode ids against {len(labels)} knowability labels for task "
      f"{task_id}. A pack whose two columns disagree in length has already lost the "
      "correspondence the whole check is about."
    )
  for label in labels:
    if label not in KNOWABILITY_REASONS:
      raise InterleavingRefused(
        f"task {task_id} carries knowability {label!r}, which is not one of "
        f"{list(KNOWABILITY_REASONS)}. The enum is closed so a fixture cannot invent "
        "a reason that no check knows how to verify."
      )
  n = len(ids)
  flags = [label in UNKNOWABLE_REASONS for label in labels]
  n_unknowable = sum(1 for f in flags if f)
  fraction = (n_unknowable / n) if n else 0.0
  low, high = UNKNOWABLE_BAND
  in_band = bool(n) and low <= fraction <= high

  indices = [i for i, f in enumerate(flags) if f]
  contiguous = bool(indices) and indices == list(range(indices[0], indices[0] + len(indices)))

  tell: Optional[str] = None
  for episode_id, flag in zip(ids, flags):
    lowered = episode_id.lower()
    for token in FORBIDDEN_ID_TOKENS:
      if token in lowered:
        tell = f"{episode_id!r} contains {token!r}"
        break
    if tell is not None:
      break
  if tell is None and indices and n_unknowable < n:
    unknowable_prefixes = {ids[i].rsplit("-", 1)[0] for i in indices}
    knowable_prefixes = {ids[i].rsplit("-", 1)[0] for i in range(n) if not flags[i]}
    only_unknowable = unknowable_prefixes - knowable_prefixes
    if only_unknowable:
      tell = (
        f"prefixes {sorted(only_unknowable)} appear on unknowable episodes and on no "
        "knowable one"
      )

  return InterleavingReport(
    task_id=task_id,
    n=n,
    n_unknowable=n_unknowable,
    fraction=fraction,
    in_band=in_band,
    contiguous=contiguous,
    id_tell=tell,
  )


def require_interleaving(
  task_id: str,
  episode_ids: Sequence[str],
  knowability: Sequence[str],
) -> InterleavingReport:
  """Run :func:`check_interleaving` and raise unless it passed.

  The checking form returns its verdict so a report can record it, which means a
  caller can also ignore it. This one cannot be ignored.
  """
  report = check_interleaving(task_id, episode_ids, knowability)
  if not report.ok:
    raise InterleavingRefused(report.format())
  return report
