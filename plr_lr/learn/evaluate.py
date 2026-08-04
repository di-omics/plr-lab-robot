"""Score a policy in simulation, and check the score means anything.

The failure mode this module exists to prevent is a success criterion that
something trivial can satisfy. A reach scored on "did the tool ever come within
tolerance at any point" is passed by a random walk given enough steps. A grasp
scored on "is the gripper closed" is passed by closing the gripper on air. The
number that comes out looks the same either way, and it is confidently wrong.

So every :func:`evaluate` runs the same task against two policies that cannot
possibly solve it, :class:`~plr_lr.learn.policy.NoOpPolicy` and
:class:`~plr_lr.learn.policy.RandomPolicy`, and reports the result alongside the
real number. If either null passes, ``EvalReport.valid`` is ``False`` and the
summary leads with that, because at that point the success rate is measuring the
criterion rather than the policy.

    report = await evaluate(lambda: my_policy, task, n_trials=10)
    print(report.summary())
    assert report.valid          # the criterion discriminates and the trials vary
    assert report.success_rate > 0.8

Success is judged only on the final observation, never on the best moment during
the run, for the same reason: a maximum taken over frames is a statistic about
how many frames there were.

``valid`` also refuses a report whose trials did not actually differ. Everything
here is seeded, so a deterministic policy driven from a fixed start replays one
identical run: ten trials, one measurement, and a success rate of 1.00 with an
interval of zero width. That is why :attr:`ReachTask.start_jitter_mm` defaults to
a real displacement rather than to zero, and why "these trials were one run" is a
hard term in ``valid`` rather than a note underneath the number.

"One run" is decided on the commanded actions, which every trial now records, and
not on the outcome. Different trajectories converging to the same outcome is a
fixed point, not a duplicate, and it is the most useful result this harness
produces: a systematic failure rather than a noisy one. That case prints a note
and stays valid.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Callable, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from pylabrobot.arms import ExperimentalSCARA

from ..arm.sim_backend import SimulationArmBackend, Workspace
from .dataset import Dataset, episodes_from
from .dynamics import (
  IMPROVEMENT_MARGIN,
  N_FEATURES,
  DynamicsModel,
  InsufficientData,
  KinematicDynamics,
  _fingerprint,
  _rmse,
)
from .planner import Goal
from .policy import NoOpPolicy, RandomPolicy
from .rollout import HardwarePermissionRequired, SafetyEnvelope, rollout
from .synthetic import park
from .types import Episode

#: A policy factory takes no arguments, or one integer trial seed. The second
#: form exists so that a stochastic policy can differ between trials instead of
#: replaying one identical run n times.
PolicyFactory = Callable[..., Any]

#: Default per-trial start displacement for :class:`ReachTask`, in mm. Not zero,
#: because zero is what turned five trials of a seeded planner into one run
#: printed five times under a success rate of 1.00. Sized at the default
#: :attr:`~plr_lr.learn.rollout.SafetyEnvelope.max_step_mm` so the displacement is
#: a whole commanded step: smaller and a policy absorbs it on its first move, at
#: which point the trials reconverge and the jitter is decorative.
DEFAULT_START_JITTER_MM = 25.0

#: Fewest trials that can carry a rate at all. One trial is a measurement, not a
#: rate, and reporting it as 0.00 or 1.00 states a precision that does not exist.
MINIMUM_TRIALS = 2

#: Fewest distinct action sequences a report must contain. Two trials that
#: commanded the same motion are one sample of that motion, however different
#: their starts were.
MINIMUM_DISTINCT_ACTIONS = 2


def _build(factory: PolicyFactory, trial_seed: int) -> Any:
  """Call a factory that may or may not want a per-trial seed."""
  try:
    params = inspect.signature(factory).parameters
  except (TypeError, ValueError):
    return factory()
  takes_seed = any(
    p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.VAR_POSITIONAL)
    for p in params.values()
  )
  return factory(trial_seed) if takes_seed else factory()


@dataclass
class ReachTask:
  """Drive the tool to a goal pose from a fixed start, in simulation.

  ``start_jitter_mm`` displaces the start pose per trial from a seeded generator,
  so repeated trials of a deterministic policy measure robustness to where the
  arm happened to be rather than repeating one identical run. It defaults to
  :data:`DEFAULT_START_JITTER_MM` and not to zero. At zero, every object in this
  package is seeded end to end: the task, the backend, and the planner all
  reproduce exactly, so ``n_trials=5`` runs the same trajectory five times and
  reports a success rate over one sample. Set it to ``0.0`` deliberately when a
  test needs the arm to start in an exactly known place, and expect the resulting
  report to be marked invalid.

  ``backend_factory`` swaps in a different simulated arm. Pass
  :class:`~plr_lr.learn.synthetic.MistrackingBackend` to score a policy against
  an arm that does not track perfectly, which is where a fitted dynamics model
  has something to contribute.
  """

  goal: Goal
  start: Tuple[float, float, float, float] = (0.0, 0.0, 100.0, 0.0)
  max_steps: int = 60
  start_jitter_mm: float = DEFAULT_START_JITTER_MM
  workspace: Optional[Workspace] = None
  name: str = "reach"
  backend_factory: Optional[Callable[[], SimulationArmBackend]] = None

  async def reset(self, trial: int = 0, seed: int = 0) -> ExperimentalSCARA:
    """Build a fresh simulated arm parked at this trial's start pose."""
    if self.backend_factory is not None:
      backend = self.backend_factory()
    else:
      backend = SimulationArmBackend(workspace=self.workspace or Workspace())
    if not isinstance(backend, SimulationArmBackend):
      raise HardwarePermissionRequired(
        f"ReachTask evaluation is simulation-only, but backend_factory returned "
        f"{type(backend).__name__}. Refusing before setup or home; use rollout with "
        "an explicit hardware permission only after simulation evaluation."
      )
    arm = ExperimentalSCARA(backend=backend)
    await arm.setup()
    await arm.home()

    start = np.array(self.start, float)
    if self.start_jitter_mm:
      rng = np.random.default_rng((seed, trial))
      start[:3] = start[:3] + rng.uniform(-self.start_jitter_mm, self.start_jitter_mm, size=3)

    # Placed, not driven. Commanding the move would run it through whatever
    # defect the backend models, so the arm would not actually be at self.start
    # and two backends under comparison would start in different places.
    park(backend, start)
    return arm

  def succeeded(self, episode: Episode) -> bool:
    """Judged on where the arm finished, not on the best frame along the way."""
    final = episode.final_observation
    if final is None:
      return False
    return self.goal.reached(final.pose_vector())

  def final_error_mm(self, episode: Episode) -> float:
    final = episode.final_observation
    if final is None:
      return float("inf")
    return float(self.goal.translation_error(final.pose_vector()))


def action_digest(episode: Episode) -> str:
  """A fingerprint of the action sequence one trial actually commanded.

  Carried per trial so that a report can tell "five trials" from "one trial run
  five times". The aggregate outcome cannot do that on its own in either
  direction: two genuinely different runs can land on the same step count and
  error, and two identical runs obviously agree on everything. Only the commanded
  actions distinguish them, and comparing whole sequences pairwise is what this
  reduces to a set membership test.

  Hashed rather than stored because the sequences are up to ``max_steps`` long
  per trial and nothing downstream needs to read one back; what is needed is
  whether two of them differ.
  """
  parts = []
  for step in episode.steps:
    action = step.action
    joints = (
      ()
      if action.joints is None
      else tuple(sorted((int(k), float(v)) for k, v in action.joints.items()))
    )
    parts.append(
      (
        action.mode.value,
        tuple(float(v) for v in action.pose),
        None if action.gripper is None else float(action.gripper),
        joints,
        bool(action.terminate),
      )
    )
  return hashlib.sha256(repr(tuple(parts)).encode("ascii")).hexdigest()


@dataclass
class TrialResult:
  success: bool
  steps: int
  final_error_mm: float
  clamped_steps: int
  stop_reason: str
  #: See :func:`action_digest`. Defaulted so a hand-built TrialResult stays easy
  #: to write, and defaulted to a value that reads as "not recorded": a report of
  #: hand-built trials therefore has one distinct sequence and is not valid,
  #: which is the right answer for trials that were never run.
  action_digest: str = ""


class ControlOutcome(NamedTuple):
  """One null control's outcome. ``n_trials`` travels with the rate on purpose."""

  name: str
  success_rate: float
  n_trials: int


@dataclass
class NullTestReport:
  """What the policies that should never succeed actually did."""

  results: List[ControlOutcome] = field(default_factory=list)

  @property
  def passed(self) -> bool:
    """True when every null control was actually run and failed every trial.

    An empty result list is not a pass. Neither is a control that ran zero
    trials: "nothing was checked" and "the check succeeded" have to be different
    answers, or the flag means nothing on exactly the runs where it matters.
    """
    if not self.results:
      return False
    return all(rate == 0.0 and n > 0 for _, rate, n in self.results)

  def summary(self) -> str:
    if not self.results:
      return "null tests: NOT RUN (nothing was checked, so nothing is established)"
    verdict = "ok" if self.passed else "BROKEN CRITERION"
    lines = [f"null tests: {verdict}"]
    for name, rate, n in self.results:
      mark = "ok " if (rate == 0.0 and n > 0) else "XX "
      lines.append(f"  [{mark}] {name}: success rate {rate:.2f} over {n} trials (must be 0.00)")
    return "\n".join(lines)


@dataclass
class EvalReport:
  """A policy's score, and whether the score is trustworthy."""

  task_name: str
  policy_name: str
  trials: List[TrialResult] = field(default_factory=list)
  null: Optional[NullTestReport] = None

  @property
  def n_trials(self) -> int:
    return len(self.trials)

  @property
  def success_rate(self) -> float:
    if not self.trials:
      return 0.0
    return sum(1 for t in self.trials if t.success) / len(self.trials)

  @property
  def mean_final_error_mm(self) -> float:
    if not self.trials:
      return float("inf")
    return float(np.mean([t.final_error_mm for t in self.trials]))

  @property
  def mean_steps_to_success(self) -> float:
    """Steps taken on successful trials only. Infinite when none succeeded.

    This is where a better dynamics model shows up. Replanning every step lets
    model-predictive control reach the goal even with a mediocre model, so
    success rate saturates while step count keeps improving.
    """
    wins = [t.steps for t in self.trials if t.success]
    return float(np.mean(wins)) if wins else float("inf")

  @property
  def clamped_trials(self) -> int:
    return sum(1 for t in self.trials if t.clamped_steps > 0)

  @property
  def null_checked(self) -> bool:
    """Whether null controls were run against this task at all."""
    return self.null is not None

  @property
  def n_distinct_action_sequences(self) -> int:
    """How many different action sequences the trials in this report contain.

    The count a trial number is only allowed to claim to be. ``n_trials`` counts
    executions; this counts evidence.
    """
    return len({t.action_digest for t in self.trials})

  @property
  def outcomes_are_identical(self) -> bool:
    """True when every trial produced the same success, step count and error.

    Deliberately **not** a validity term, and this is the one judgement call in
    the block. Trials that genuinely differ can converge: a planner driven by a
    model that is wrong about the arm's gain reaches the same fixed point from
    every jittered start, so five different trajectories finish at the same error
    to the last bit. Measured here at a 0.5 mm tolerance against a gain-0.85 arm:
    five distinct trajectories, five different starts, final error 0.9189 mm in
    all five. That is a real result and it is the most useful kind, a systematic
    failure rather than a noisy one. Refusing it would leave this harness unable
    to report reproducibility at all, so it prints as a note.

    What makes it safe to demote is that the commanded actions are now recorded
    per trial, so "the same run repeated" no longer has to be inferred from the
    outcome. See :attr:`trials_are_identical`.
    """
    if self.n_trials < MINIMUM_TRIALS:
      return False
    first = self.trials[0]
    return all(
      t.success == first.success
      and t.steps == first.steps
      and abs(t.final_error_mm - first.final_error_mm) < 1e-12
      for t in self.trials[1:]
    )

  @property
  def trials_are_identical(self) -> bool:
    """True when the trials were one run repeated: same actions, same outcome.

    The measured defect this names: with ``start_jitter_mm`` at zero, a seeded
    planner against a seeded backend produced five trials that agreed on every
    recorded byte, ``success_rate=1.00``, and ``valid=True``. The identical trials
    were mentioned in a note under the number, and a note is not a term in
    anything a caller asserts on. This is a hard term in :attr:`valid` now.
    """
    if self.n_trials < MINIMUM_TRIALS:
      return False
    return self.n_distinct_action_sequences == 1 and self.outcomes_are_identical

  @property
  def trials_varied(self) -> bool:
    """Whether these trials are more than one measurement reported n times.

    Three conditions, all required: enough trials to form a rate at all; at least
    two distinct commanded action sequences, so the runs were not one trajectory
    replayed from n starts; and not one run repeated.

    The last condition is implied by the second, and is written out anyway. It is
    the condition that names the measured defect, so it belongs where a reader
    looking for it will find it, and neither definition can be loosened later
    without the other one still being read.
    """
    if self.n_trials < MINIMUM_TRIALS:
      return False
    if self.n_distinct_action_sequences < MINIMUM_DISTINCT_ACTIONS:
      return False
    return not self.trials_are_identical

  @property
  def valid(self) -> bool:
    """Whether the success rate is worth quoting.

    Requires that null controls were run, that they failed, and that the trials
    behind the rate actually varied. Returning True when nothing was checked
    would make this flag say "trustworthy" precisely on the runs where nothing
    established that, and returning True for n copies of one run would make the
    rate a statistic about a single trajectory.
    """
    return self.null is not None and self.null.passed and self.trials_varied

  def variation_note(self) -> Optional[str]:
    """Why :attr:`trials_varied` is False, or None when it is True."""
    if self.n_trials < MINIMUM_TRIALS:
      return (
        f"{self.n_trials} trial(s): a rate needs at least {MINIMUM_TRIALS}. One run "
        "is a measurement, and printing it as 0.00 or 1.00 claims a precision that "
        "was never measured."
      )
    if self.trials_are_identical:
      return (
        f"all {self.n_trials} trials were the same run: the same action sequence and "
        "the same outcome. This is one measurement printed "
        f"{self.n_trials} times. Vary the start (ReachTask.start_jitter_mm) or the "
        "per-trial policy seed."
      )
    if self.n_distinct_action_sequences < MINIMUM_DISTINCT_ACTIONS:
      return (
        f"all {self.n_trials} trials commanded the same action sequence, so this is "
        "one trajectory replayed from "
        f"{self.n_trials} starts. Vary the per-trial policy seed to sample the policy "
        "as well as the start."
      )
    return None

  def summary(self) -> str:
    lines = []
    if self.null is not None and not self.null.passed:
      lines.append(
        "INVALID: a null policy passed this task, so the success rate below "
        "describes the success criterion, not the policy. Fix the criterion."
      )
    elif self.null is None:
      lines.append(
        "UNVALIDATED: null controls were not run, so nothing here establishes "
        "that this task is one a do-nothing policy would fail."
      )
    note = self.variation_note()
    if note is not None:
      lines.append(f"INVALID: {note}")
    lines.append(f"{self.policy_name} on {self.task_name}: {self.n_trials} trials")
    lines.append(
      f"  distinct actions: {self.n_distinct_action_sequences}/{self.n_trials} "
      "(trials commanding a different sequence)"
    )
    lines.append(f"  success rate:     {self.success_rate:.2f}")
    lines.append(f"  mean final error: {self.mean_final_error_mm:.3f} mm")
    n_wins = sum(1 for t in self.trials if t.success)
    steps = self.mean_steps_to_success
    shown = "n/a" if steps == float("inf") else format(steps, ".1f")
    lines.append(f"  steps to success: {shown} (over {n_wins}/{self.n_trials} successful)")
    lines.append(f"  trials clamped:   {self.clamped_trials}/{self.n_trials}")
    if self.outcomes_are_identical and not self.trials_are_identical:
      # Name the outcome that was actually reached. This note used to say "success"
      # unconditionally, so a task on which every trial failed printed "converged to
      # the same success" directly under a success rate of 0.00. A systematic failure
      # is the more useful of the two results and it is the one a reader is most
      # likely to skim, so it has to be the word on the line.
      outcome = "success" if self.trials[0].success else "failure"
      lines.append(
        f"  note: all {self.n_trials} trials converged to the same {outcome}, step "
        "count and final error from different trajectories. That is a fixed point, "
        "which is a result; it is not n independent samples of a spread."
      )
    if self.null is not None:
      lines.append("  " + self.null.summary().replace("\n", "\n  "))
    return "\n".join(lines)


async def _run_trials(
  policy_factory: PolicyFactory,
  task: ReachTask,
  n_trials: int = 5,
  envelope: Optional[SafetyEnvelope] = None,
  seed: int = 0,
) -> Tuple[List[TrialResult], str]:
  """Run trials and retain the concrete policy type built for the first one."""
  if n_trials < 1:
    raise ValueError(
      f"n_trials={n_trials} produces no evidence. A report built from zero trials "
      "reads as a clean pass because nothing failed, which is the opposite of what "
      "it means."
    )
  # "or Workspace()" matches what reset() gives the arm. Passing task.workspace
  # straight through means a task left at the default gets an envelope that never
  # clips while the arm still enforces its own box, so a null control walks out of
  # bounds and WorkspaceViolation escapes the harness instead of being scored.
  envelope = envelope or SafetyEnvelope(
    max_steps=task.max_steps, workspace=task.workspace or Workspace()
  )
  results: List[TrialResult] = []
  built_policy_name: Optional[str] = None
  for trial in range(n_trials):
    arm = await task.reset(trial=trial, seed=seed)
    policy = _build(policy_factory, seed * 100_000 + trial)
    if built_policy_name is None:
      built_policy_name = type(policy).__name__
    episode = await rollout(arm, policy, envelope=envelope, task=task.name)
    results.append(
      TrialResult(
        success=task.succeeded(episode),
        steps=len(episode),
        final_error_mm=task.final_error_mm(episode),
        clamped_steps=int(episode.metadata.get("clamped_steps", 0)),
        stop_reason=str(episode.metadata.get("stop_reason", "")),
        action_digest=action_digest(episode),
      )
    )
  # n_trials >= 1 above guarantees that a policy was built.
  assert built_policy_name is not None
  return results, built_policy_name


async def run_trials(
  policy_factory: PolicyFactory,
  task: ReachTask,
  n_trials: int = 5,
  envelope: Optional[SafetyEnvelope] = None,
  seed: int = 0,
) -> List[TrialResult]:
  """Run ``n_trials`` rollouts and score each one. Simulation only."""
  trials, _ = await _run_trials(
    policy_factory, task, n_trials=n_trials, envelope=envelope, seed=seed
  )
  return trials


async def null_test(
  task: ReachTask,
  n_trials: int = 5,
  envelope: Optional[SafetyEnvelope] = None,
  seed: int = 0,
) -> NullTestReport:
  """Run the policies that must fail, and report whether they did.

  A task that a null control can pass is not a task. Run this against any new
  success criterion before trusting a number that came out of it.
  """
  nulls: List[Tuple[str, PolicyFactory]] = [
    ("NoOpPolicy", NoOpPolicy),
    # Takes the per-trial seed, so the n trials are n different random walks
    # rather than one walk replayed n times under a report that says "n trials".
    ("RandomPolicy", lambda trial_seed: RandomPolicy(step_mm=25.0, seed=trial_seed)),
  ]
  report = NullTestReport()
  for name, factory in nulls:
    trials = await run_trials(factory, task, n_trials=n_trials, envelope=envelope, seed=seed)
    rate = sum(1 for t in trials if t.success) / len(trials)
    report.results.append(ControlOutcome(name, rate, len(trials)))
  return report


async def evaluate(
  policy_factory: PolicyFactory,
  task: ReachTask,
  n_trials: int = 5,
  envelope: Optional[SafetyEnvelope] = None,
  null_check: bool = True,
  policy_name: Optional[str] = None,
  seed: int = 0,
) -> EvalReport:
  """Score a policy, with the null controls run alongside it by default.

  ``null_check=False`` skips them. It exists for the inner loop of a sweep where
  the criterion was already validated, and it is the wrong default for anything
  whose number will be quoted.
  """
  trials, built_policy_name = await _run_trials(
    policy_factory, task, n_trials=n_trials, envelope=envelope, seed=seed
  )
  name = policy_name or built_policy_name
  null = (
    await null_test(task, n_trials=n_trials, envelope=envelope, seed=seed)
    if null_check
    else None
  )
  return EvalReport(task_name=task.name, policy_name=name, trials=trials, null=null)


# -- dynamics ----------------------------------------------------------------


@dataclass
class DynamicsScore:
  """How well a dynamics model predicted transitions it was not fit on."""

  n: int
  rmse: float  # translation, mm
  baseline_rmse: float  # translation, mm
  rotation_rmse: float = 0.0  # rad
  baseline_rotation_rmse: float = 0.0  # rad
  n_seen_in_training: int = 0

  @property
  def improved(self) -> bool:
    """Beat the baseline by a margin, on rows the model was not fit on."""
    if self.n_seen_in_training:
      return False
    if self.baseline_rmse <= 1e-12:
      return False
    if self.rmse > self.baseline_rmse * (1.0 - IMPROVEMENT_MARGIN):
      return False
    return self.rotation_rmse <= self.baseline_rotation_rmse + 1e-12

  def summary(self) -> str:
    verdict = "improved" if self.improved else "NO IMPROVEMENT"
    lines = [
      f"dynamics score: {verdict} over {self.n} transitions",
      f"  model:    {self.rmse:.4f} mm, {self.rotation_rmse:.5f} rad",
      f"  baseline: {self.baseline_rmse:.4f} mm, {self.baseline_rotation_rmse:.5f} rad",
    ]
    if self.n_seen_in_training:
      lines.append(
        f"  LEAK: {self.n_seen_in_training} of these {self.n} transitions were in the "
        "model's own training set, so this measures memorisation. Score on episodes "
        "the fit never saw."
      )
    return "\n".join(lines)


def evaluate_dynamics(
  model: DynamicsModel,
  source: Union[Dataset, Sequence[Episode]],
  baseline: Optional[DynamicsModel] = None,
  allow_training_data: bool = False,
) -> DynamicsScore:
  """Score a dynamics model against episodes it was not fit on.

  Scoring a model on its training data measures memorisation, and for a least
  squares fit it always looks good. ``ResidualDynamics.fit`` records which rows
  it trained on, so this detects the overlap rather than trusting the caller to
  remember: any overlap makes ``improved`` False and is named in the summary.
  Pass ``allow_training_data=True`` to count it anyway, deliberately.
  """
  baseline = baseline or KinematicDynamics()
  poses, deltas, nexts = [], [], []
  for ep in episodes_from(source):
    for obs, action, nxt in ep.transitions():
      pose = obs.pose_vector()
      try:
        delta = action.as_delta(pose)
      except ValueError:
        continue
      poses.append(pose)
      deltas.append(delta)
      nexts.append(nxt.pose_vector())
  if not poses:
    raise ValueError("no Cartesian transitions to score against")
  if len(poses) < N_FEATURES:
    raise InsufficientData(
      f"{len(poses)} transitions is too few to say anything about a model with "
      f"{N_FEATURES} parameters. ResidualDynamics.fit refuses to fit on this "
      "little; scoring on this little is no more informative."
    )

  P, D, N = np.array(poses), np.array(deltas), np.array(nexts)

  overlap = 0
  trained_on = getattr(model, "trained_on", None)
  if trained_on and not allow_training_data:
    overlap = len(_fingerprint(P, D) & trained_on)

  model_t, model_r = _rmse(np.atleast_2d(model.predict(P, D)), N)
  base_t, base_r = _rmse(np.atleast_2d(baseline.predict(P, D)), N)
  return DynamicsScore(
    n=len(poses),
    rmse=model_t,
    baseline_rmse=base_t,
    rotation_rmse=model_r,
    baseline_rotation_rmse=base_r,
    n_seen_in_training=overlap,
  )
