"""The command line: six verbs, and not one of them has a hardware flag.

That absence is load-bearing and is asserted by an AST test rather than left to
review. This package has exactly one hardware permission gate, it lives on the
rollout call in :mod:`plr_lr.learn.rollout`, and it needs three independent
conditions that all default to off. A benchmark runner that grew a
``--i-am-at-the-arm`` of its own would be a second gate, and a second gate is the
documented way the first one stops being the one that matters. Scoring a world model
never needs an arm: every verb here reads a file, computes, and prints.

The verbs:

``run``         gate, then controls, then rows. The scorecard.
``controls``    the null suite on its own, for when only the suite is in question.
``probe``       the action-conditioning gate on its own, before anything else runs.
``score``       the same as ``run`` and emits only the receipt JSON.
``screen-fit``  fit the frozen act/defer threshold on a disjoint calibration pack
                and print the calibration string that will be reprinted verbatim on
                every verdict it later decides.
``receipt``     the declaration block, the seeds and the refusals, without the rows.

The threshold is fitted on a **separate** pack drawn from a different seed. Fitting
it on the pack being scored would be fitting a decision rule to the answer key, and
the resulting act/defer table would be a statement about the answer key rather than
about the model.

Two properties of the parser itself are load-bearing.

**A shared option means the same thing on either side of the verb.** Every option
below is attached twice, once to the top-level parser and once to each subparser,
so that ``--split scoring run`` and ``run --split scoring`` are the same command.
The subparser copies default to :data:`argparse.SUPPRESS`, which is the whole fix:
with real defaults on both copies, argparse re-applies the subparser's defaults over
whatever the top level had already parsed, and every pre-verb option is discarded in
silence. Measured before the fix: ``--split scoring --seed 7 --epoch E9 run`` scored
``pack builtin-demo-20260803 | epoch demo-2026-08-03 | split dev``. Worse than the
wrong label, ``--episodes 40 run`` printed a sixteen-row scorecard while ``run
--episodes 40`` printed ``INTERVENTION-REFUSED`` and no rows, so the verdict itself
depended on where the option was typed. Under R11 that is a receipt for a pack the
operator did not ask for, which is worse than either an error or a correct answer.

**A pack chooses the tasks, and never the other way round.** The task list scored is
the intersection of what this harness declares with what the pack carries, so a pack
holding a subset scores that subset and the tasks it does not carry are named in the
declaration block rather than silently absent. A pack naming a task this harness does
not define is refused by name before anything runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..screen import ALPHA, MIN_CALIBRATION, CalibrationPoint, NotCalibrated, Screen
from ..selective import PERMUTATION_RESAMPLES
from . import metrics
from .controls import evaluate_rows, run_control_suite
from .packio import (
  PackFormatError,
  ScenePack,
  builtin_demo_pack,
  load_manifest,
  tasks_in,
)
from .report import BenchReport
from .runner import SeedPlan, floor_predictors, risk_at_full_coverage, run_bench
from .tasks import TASKS, BenchTask

PROGRAM = "python -m plr_lr.learn.bench"

#: The staleness bound a screen fitted here declares. An absolute constant and not a
#: default that a caller can leave unset: a staleness bound nobody chose is what
#: decides whether a prediction is about the plate in front of the arm now or one
#: from thirty seconds ago.
MAX_FRAME_AGE_S = 1.0

#: How far the calibration pack's seed is moved from the scoring pack's. Any
#: separation would do; what matters is that it is a separation and that it is
#: printed, so a reader can check that the threshold was not fitted on the episodes
#: it then judged.
CALIBRATION_SEED_OFFSET = 977


def _pack(args: argparse.Namespace) -> ScenePack:
  """The pack to score, refusing a manifest by name rather than by traceback.

  Every refusal :func:`plr_lr.learn.bench.packio.load_manifest` raises is a stated
  reason a file cannot be scored, so it is printed as one. The alternative is what
  this used to do with a real vision-side export: an unhandled ``ValueError`` from
  four frames deeper, naming a task the operator never mentioned and neither of the
  two vocabularies involved.
  """
  if getattr(args, "pack", None):
    try:
      return load_manifest(args.pack)
    except PackFormatError as exc:
      raise SystemExit(
        f"this manifest was not scored: {exc}\nNothing is emitted. A pack that cannot "
        "be read, whose task ids this harness does not define, or whose unknowable "
        "labels the evidence refutes is a refusal and not a low score."
      ) from None
  return builtin_demo_pack(
    tasks=TASKS,
    n_episodes=args.episodes,
    seed=args.seed,
    seed_epoch=args.epoch,
    split=args.split,
  )


def _tasks(pack: ScenePack) -> Tuple[BenchTask, ...]:
  """The declared tasks this pack carries, refused by name when it carries none."""
  try:
    return tasks_in(pack, TASKS)
  except PackFormatError as exc:
    raise SystemExit(
      f"this pack was not scored: {exc}\nNothing is emitted."
    ) from None


def _calibration_pack(
  args: argparse.Namespace,
  tasks: Sequence[BenchTask] = TASKS,
) -> ScenePack:
  """The disjoint pack the threshold is fitted on, over the tasks being scored.

  Fitted over the same task list, because the trust horizon carried on one screen is
  the minimum across the tasks it was fitted over: a screen calibrated over five
  tasks and used to judge two carries the shortest horizon of three tasks nobody
  scored.
  """
  return builtin_demo_pack(
    tasks=tuple(tasks),
    n_episodes=max(MIN_CALIBRATION * 4, int(args.episodes or 200)),
    seed=int(args.seed) + CALIBRATION_SEED_OFFSET,
    seed_epoch=f"{args.epoch}-calibration",
    split="dev",
  )


def fit_screen(
  pack: ScenePack,
  predictors: Dict[str, Any],
  tasks: Sequence[BenchTask] = TASKS,
  alpha: float = ALPHA,
) -> Screen:
  """Split conformal on held-out successful episodes, and nothing else.

  ``successes`` really means successes: episodes the floor predictor got inside
  tolerance. Mixing failures in drags the quantile down until the threshold admits
  them too, and the question the threshold answers is what a confident success looks
  like, not what the average episode looks like.

  The trust horizon carried on the resulting screen is the **minimum** measured
  across the tasks, because one screen carries one horizon and the conservative
  choice is the shortest horizon any task was trustworthy at.

  ``metrics.trust_horizon`` returns ``None`` to mean that no horizon was trustworthy,
  and one screen covers every task, so a single ``None`` anywhere in that minimum
  raises :class:`plr_lr.learn.screen.NotCalibrated` and no screen is built. Dropping
  the ``None`` out of the minimum and carrying on with the other tasks would hand
  back a live screen that admits horizon-one predictions on the very task nothing
  was trustworthy at, and it would read in every log exactly like a screen that had
  measured a horizon of one.

  Raises:
    NotCalibrated: too few calibration successes, or any task with no trustworthy
      horizon.
  """
  points: List[CalibrationPoint] = []
  horizons: List[Optional[int]] = []
  for task in tasks:
    risk_by_horizon: Dict[int, Optional[float]] = {}
    for horizon in task.horizons:
      rows = evaluate_rows(predictors[task.task_id], task, pack, int(horizon))
      risk_by_horizon[int(horizon)] = risk_at_full_coverage(rows)
      if int(horizon) != int(task.horizons[0]):
        continue
      for index in range(rows.n):
        if rows.abstained[index] or not rows.hits[index]:
          continue
        confidence = rows.confidences[index]
        if confidence is None:
          continue
        points.append(
          CalibrationPoint(
            episode_id=f"{task.task_id}/{rows.episode_ids[index]}",
            confidence=float(confidence),
          )
        )
    horizons.append(metrics.trust_horizon(risk_by_horizon))
  if not horizons:
    raise NotCalibrated(
      "no tasks were measured, so no horizon was measured either. A screen fitted "
      "over nothing carries a trust horizon nothing established."
    )
  untrusted = [t.task_id for t, h in zip(tasks, horizons) if h is None]
  if untrusted:
    raise NotCalibrated(
      f"tasks {sorted(untrusted)} measured no trustworthy horizon on calibration pack "
      f"{pack.pack_id}: risk at full coverage was above the bound at every horizon "
      "offered, including the shortest. No screen is built. A screen defaulted to a "
      "trust horizon of one is not a screen that admits nothing, it is a screen that "
      "admits every single-step prediction on a task the model was never shown to be "
      "safe at, and it prints the same calibration string a measured one would."
    )
  trust = min(int(h) for h in horizons if h is not None)
  return Screen.calibrate(
    successes=points,
    source=f"calibration pack {pack.pack_id} epoch {pack.seed_epoch}",
    trust_horizon=int(trust),
    max_frame_age_s=MAX_FRAME_AGE_S,
    alpha=alpha,
  )


def _seeds(args: argparse.Namespace, pack: ScenePack) -> SeedPlan:
  return SeedPlan(
    epoch=pack.seed_epoch,
    tie_break_seed=int(args.tie_break_seed),
    gate_seed=int(args.gate_seed),
    control_seed=int(args.control_seed),
    bootstrap_seed=int(args.bootstrap_seed),
  )


def pack_declarations(pack: ScenePack, tasks: Sequence[BenchTask]) -> Dict[str, Any]:
  """What the run has to say about the pack itself, printed with the declarations.

  Two lines, both of which used to be absent and both of which change how a reader
  should take the rows underneath.

  ``separation`` is the status of the unknowable-label gate on **this** pack. It
  reads ``UNCHECKED`` whenever the pack withheld the coefficients the gate needs,
  which a scoring pack does deliberately, and that is a refusal to certify rather
  than a pass. Before the gate ran on load it printed nothing at all for every pack
  this repository did not generate.

  ``tasks_not_in_pack`` names the declared tasks the pack carries no episodes for. A
  two-row scorecard from a five-task benchmark reads like a five-task result unless
  the three missing ones are named, and R10 means there is no aggregate row where
  their absence would otherwise show up.
  """
  scored = [task.task_id for task in tasks]
  missing = [task.task_id for task in TASKS if task.task_id not in scored]
  separation = pack.separation
  return {
    "pack_separation": (
      "NOT RUN: the gate never read this pack, so nothing about its unknowable "
      "labels is established"
      if separation is None
      else separation.line()
    ),
    "tasks_scored": scored,
    "tasks_not_in_pack": missing or ["none: the pack carries every declared task"],
  }


def _report(args: argparse.Namespace) -> Tuple[BenchReport, ScenePack]:
  pack = _pack(args)
  tasks = _tasks(pack)
  predictors = floor_predictors(TASKS)
  try:
    screen = fit_screen(_calibration_pack(args, tasks), predictors, tasks)
  except NotCalibrated as exc:
    raise SystemExit(
      f"no screen could be fitted: {exc}\nNo scorecard is emitted. A screen with a "
      "placeholder threshold admits every prediction ever made, a screen with a "
      "placeholder trust horizon admits every single-step one, and both read in every "
      "log exactly like a calibrated screen."
    ) from None
  report = run_bench(
    predictors,
    tasks,
    pack,
    screen,
    _seeds(args, pack),
    n_permutations=int(args.permutations),
    n_resamples=None if args.resamples is None else int(args.resamples),
    model_name=args.model,
  )
  declared = dict(report.declared)
  declared.update(pack_declarations(pack, tasks))
  return dataclasses.replace(report, declared=declared), pack


def cmd_run(args: argparse.Namespace) -> int:
  report, _pack_used = _report(args)
  if args.json:
    print(report.to_json())
  else:
    print(report.summary())
  return 0 if report.valid else 1


def cmd_score(args: argparse.Namespace) -> int:
  report, _pack_used = _report(args)
  print(report.to_json())
  return 0 if report.valid else 1


def cmd_controls(args: argparse.Namespace) -> int:
  pack = _pack(args)
  tasks = _tasks(pack)
  predictors = floor_predictors(TASKS)
  suite = run_control_suite(
    predictors,
    pack,
    tasks=tasks,
    seed=int(args.control_seed),
    n_permutations=int(args.permutations),
    n_resamples=None if args.resamples is None else int(args.resamples),
  )
  print(suite.summary())
  for key, value in sorted(pack_declarations(pack, tasks).items()):
    print(f"  declared {key}: {value}")
  return 0 if suite.passed else 1


def cmd_probe(args: argparse.Namespace) -> int:
  from ..predict import interventional_check

  from .controls import build_request

  pack = _pack(args)
  tasks = _tasks(pack)
  predictors = floor_predictors(TASKS)
  ok = True
  for task in tasks:
    episodes = pack.for_task(task.task_id)
    requests = [build_request(task, e, int(task.horizons[0])) for e in episodes]
    result = interventional_check(
      predictors[task.task_id], requests, seed=int(args.gate_seed)
    )
    print(f"{task.task_id}: " + result.summary())
    ok = ok and result.passed
  if not ok:
    print(
      "at least one task refused. A model that does not clear this gate is not "
      "scored at all: the failure is a refusal and not a low score, because a low "
      "score still gets ranked."
    )
  return 0 if ok else 1


def cmd_screen_fit(args: argparse.Namespace) -> int:
  tasks = _tasks(_pack(args)) if getattr(args, "pack", None) else TASKS
  pack = _calibration_pack(args, tasks)
  predictors = floor_predictors(TASKS)
  try:
    screen = fit_screen(pack, predictors, tasks)
  except NotCalibrated as exc:
    print(f"NOT CALIBRATED: {exc}")
    return 1
  print(screen.summary())
  print(
    "  this string is reprinted verbatim on every verdict this threshold decides, "
    "because a frozen threshold whose provenance is a second lookup is a frozen "
    "threshold nobody checks."
  )
  return 0


def cmd_receipt(args: argparse.Namespace) -> int:
  report, pack = _report(args)
  receipt = report.to_dict()
  receipt["rows"] = f"{len(report.rows)} rows withheld from this view; run score for them"
  receipt["pack"] = {
    "pack_id": pack.pack_id,
    "seed_epoch": pack.seed_epoch,
    "split": pack.split,
    "source": pack.source,
    "basis": pack.basis,
    "n_episodes": len(pack.episodes),
    "n_frame_hashes": len(pack.frames_sha256()),
    "task_ids": list(pack.task_ids),
    "separation": (
      None if pack.separation is None else pack.separation.to_dict()
    ),
  }
  import json

  print(json.dumps(receipt, indent=2, sort_keys=True))
  return 0 if report.valid else 1


#: Defaults for the shared options, in one place so the two copies of the option
#: table cannot disagree about them. The top-level copy carries these; the subparser
#: copy carries :data:`argparse.SUPPRESS` instead, which is what makes an option
#: written before the verb survive.
DEFAULTS: Dict[str, Any] = {
  "pack": None,
  "episodes": None,
  "seed": 20260803,
  "epoch": "demo-2026-08-03",
  "split": "dev",
  "model": None,
  "permutations": PERMUTATION_RESAMPLES,
  "resamples": None,
  "tie_break_seed": 0,
  "gate_seed": 0,
  "control_seed": 0,
  "bootstrap_seed": 0,
  "json": False,
}


def _shared_options(defaults: bool) -> argparse.ArgumentParser:
  """Options every verb accepts. Not one of them names hardware, an arm or a port.

  Kept in one place and attached to every subparser so that adding an option means
  adding it once. That matters more than it looks: an option added to four of six
  verbs is an option that behaves differently depending on which verb you typed, and
  the two verbs that lack it are the ones whose output somebody then compares.

  ``defaults`` decides which of the two copies this is, and it is the fix for a
  measured defect rather than a style choice. Attaching one option table with real
  defaults to both the top-level parser and every subparser means argparse parses
  the pre-verb options, then parses the verb, then writes the subparser's defaults
  over everything it already had. Every option written before the verb was discarded
  in silence: ``--split scoring --seed 7 --epoch E9 run`` scored the default pack at
  the default epoch on the default split and printed a receipt for it. The subparser
  copy therefore defaults to :data:`argparse.SUPPRESS`, so it sets only what was
  actually typed after the verb, and a post-verb value still wins over a pre-verb one
  because it is the later of the two writes.
  """

  def default(name: str) -> Any:
    return DEFAULTS[name] if defaults else argparse.SUPPRESS

  shared = argparse.ArgumentParser(add_help=False)
  shared.add_argument("--pack", default=default("pack"),
                      help="path to a scenes manifest.json")
  shared.add_argument("--episodes", type=int, default=default("episodes"),
                      help="episodes per task for the built-in demonstration pack")
  shared.add_argument("--seed", type=int, default=default("seed"), help="generator seed")
  shared.add_argument("--epoch", default=default("epoch"), help="seed epoch label")
  shared.add_argument("--split", default=default("split"), choices=("dev", "scoring"))
  shared.add_argument("--model", default=default("model"),
                      help="name printed on the scorecard")
  shared.add_argument("--permutations", type=int, default=default("permutations"))
  shared.add_argument("--resamples", type=int, default=default("resamples"),
                      help="paired bootstrap resamples; the default is the full count")
  shared.add_argument("--tie-break-seed", type=int, default=default("tie_break_seed"),
                      dest="tie_break_seed")
  shared.add_argument("--gate-seed", type=int, default=default("gate_seed"),
                      dest="gate_seed")
  shared.add_argument("--control-seed", type=int, default=default("control_seed"),
                      dest="control_seed")
  shared.add_argument("--bootstrap-seed", type=int, default=default("bootstrap_seed"),
                      dest="bootstrap_seed")
  shared.add_argument("--json", action="store_true", default=default("json"),
                      help="emit the receipt JSON")
  return shared


def build_parser() -> argparse.ArgumentParser:
  """Six verbs and shared options. No verb takes a hardware flag, ever.

  Adding one would create a second hardware permission gate in a package that has
  exactly one, and the second gate is always the one somebody finds first.

  The two option tables are not the same object: the top-level one holds the
  defaults and the subparser one holds :data:`argparse.SUPPRESS`, which is what
  makes ``--split scoring run`` and ``run --split scoring`` the same command.
  """
  parser = argparse.ArgumentParser(
    prog=PROGRAM,
    parents=[_shared_options(defaults=True)],
    description=(
      "DeferLab v0.1: score a world model on the decision not to predict. CPU only, "
      "no network, no hardware. Every row reads basis: simulated until a hardware run "
      "record with a device serial hash exists. Shared options mean the same thing "
      "before and after the verb."
    ),
  )
  verb_options = _shared_options(defaults=False)
  sub = parser.add_subparsers(dest="verb")
  for verb, help_text in (
    ("run", "gate, controls, then rows: the scorecard"),
    ("controls", "the null control suite on its own"),
    ("probe", "the action-conditioning gate on its own"),
    ("score", "the scorecard as receipt JSON"),
    ("screen-fit", "fit the frozen act/defer threshold on a disjoint pack"),
    ("receipt", "declarations, seeds and refusals without the rows"),
  ):
    sub.add_parser(verb, parents=[verb_options], help=help_text)
  return parser


VERBS = {
  "run": cmd_run,
  "controls": cmd_controls,
  "probe": cmd_probe,
  "score": cmd_score,
  "screen-fit": cmd_screen_fit,
  "receipt": cmd_receipt,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = build_parser()
  args = parser.parse_args(list(argv) if argv is not None else sys.argv[1:])
  verb = args.verb or "run"
  return VERBS[verb](args)


__all__ = [
  "DEFAULTS",
  "MAX_FRAME_AGE_S",
  "VERBS",
  "build_parser",
  "fit_screen",
  "main",
  "pack_declarations",
]
