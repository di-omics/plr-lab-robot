"""DeferLab v0.1: score a world model on the decision not to produce a rollout.

Every world-model benchmark in the current survey scores a rollout that was
produced. This one scores the decision not to produce one, on bench-tolerance
outcomes rather than pixels, credits a correct refusal exactly as much as a correct
prediction, and refuses to publish any score at all for a model that ignores its
action argument.

The order of operations is the design:

    interventional gate  ->  null control suite  ->  per-task rows

The gate runs before any episode is scored and its failure is a refusal, not a low
score. The suite runs second and no scorecard is emitted unless it passes. Only then
does a single number exist. Reversing any two of those lets a number exist before
the thing that would have withheld it, and once a number exists somebody quotes it.

Three things this package will not do, at any setting:

* print a cross-task aggregate scalar (a composite index lets a strong task hide a
  broken one);
* print an excess-AURC headline beside the control that refutes it (the number gets
  quoted and the refutation gets dropped, so the number is withheld instead);
* describe a run as hardware-validated without a device serial hash behind it.

The entire interface to the vision side of the build is the versioned scene-pack
file format in :mod:`plr_lr.learn.bench.packio`. There is no import of that package
here in either direction, and a test asserts it.
"""

from __future__ import annotations

from .controls import (
  BANDS,
  ControlResult,
  ControlSuiteReport,
  EpisodeRows,
  evaluate_rows,
  run_control_suite,
)
from .metrics import (
  NOT_RANKED,
  ComputeDeclaration,
  PixelColumn,
  SuccessGap,
  TolerancePredictionError,
  in_tolerance,
  tolerance_error,
  training_overlap,
  trust_horizon,
)
from .packio import (
  SCENES_FORMAT_VERSION,
  ScenePack,
  builtin_demo_pack,
  load_manifest,
  write_manifest,
)
from .report import REFUSAL_RULES, BenchReport, BenchRow, Refusal
from .runner import SeedPlan, floor_predictors, run_bench
from .tasks import TASKS, TASKS_BY_ID, BenchTask, check_interleaving, task_for

__all__ = [
  "BANDS",
  "NOT_RANKED",
  "REFUSAL_RULES",
  "SCENES_FORMAT_VERSION",
  "TASKS",
  "TASKS_BY_ID",
  "BenchReport",
  "BenchRow",
  "BenchTask",
  "ComputeDeclaration",
  "ControlResult",
  "ControlSuiteReport",
  "EpisodeRows",
  "PixelColumn",
  "Refusal",
  "ScenePack",
  "SeedPlan",
  "SuccessGap",
  "TolerancePredictionError",
  "builtin_demo_pack",
  "check_interleaving",
  "evaluate_rows",
  "floor_predictors",
  "in_tolerance",
  "load_manifest",
  "run_bench",
  "run_control_suite",
  "task_for",
  "tolerance_error",
  "training_overlap",
  "trust_horizon",
  "write_manifest",
]
