"""The versioned file format that is the only interface between this harness and
the vision side of the build.

There is no import of ``labcv`` anywhere under :mod:`plr_lr.learn.bench`, and there
is no import of ``plr_lr`` in the exporter that writes these files. That is not
tidiness. Two packages that import each other end up sharing a type, then sharing a
constant, then sharing a bug, and the one thing that must stay checkable is whether
a submission ran on the fixtures it claims: that check is a set of frame hashes in a
manifest, and a manifest is a file, so the coupling is a file. A test asserts the
absence of the import in this direction, which is the direction this repository can
actually enforce.

``SCENES_FORMAT_VERSION`` is checked on load and a mismatch is a refusal rather than
a best-effort read. A format that silently accepts an older manifest reads whatever
fields it recognises and defaults the rest, and the defaulted field here would be
``knowability``: an episode whose knowability defaulted to "knowable" turns a
correct refusal into a false alarm and scores it against the model.

One refusal lives outside the format layer and is the reason this module also owns
the declared transitions: :func:`require_unknowable_separation` advances every
episode labelled unknowable under every value in its hidden grid, at every horizon
it will be scored at, and fails the build with
:class:`UnknowableSeparationError` unless the branches land apart. It is a
falsifier and only a falsifier. A collapse proves an episode is knowable and the
label is refused; a separation proves nothing about the label, because the latent
could still be recoverable from the frames, which is a question about pixels and
lives on the vision side. Only the conclusive direction is read.

**That gate runs on load, not only on build.** It used to be wired into
:func:`builtin_demo_pack` alone, which meant it protected the one pack whose labels
this repository generated and skipped every externally supplied pack: exactly the
case a file format exists for, and exactly the case where the labels were written by
a different codebase. Measured with a spy on a real vision-side export: "labcv pack
loaded, separation calls: 0". :func:`load_manifest` now certifies every pack it
reads. Where a pack legitimately withholds the coefficients the gate needs -- a
scoring pack does, on purpose -- the outcome is a stated
:data:`SEPARATION_UNCHECKED` status that travels to the scorecard on
:attr:`ScenePack.separation`, and never a silent pass. An uncheckable claim and a
checked one are different answers and the receipt prints which one it got.

Task identifiers are this harness's, and a pack naming one it does not define is
refused by :func:`require_known_tasks` with :class:`UnknownTaskIds` naming both
vocabularies. The identifier is the key that selects a tolerance, a state-field
layout, a horizon list and a hidden-branch count, and all four are declarations in
:mod:`plr_lr.learn.bench.tasks`. A producer that names its own vocabulary is asking
the scorer to guess a tolerance, and a tolerance guessed at the scoring step is not
the task's tolerance. Before this refusal existed the failure was an unhandled
``ValueError`` from the interventional gate reading "task S1 carries 0 episodes",
which names neither the ids that arrived nor the ids that were wanted.

That gate exists because the permissive version of this module shipped labels that
were false: 80 of 80 S1 horizon-1 episodes labelled unknowable had a terminal state
that was byte-identical for every value of the hidden latent, and 63 of 80 S5
horizon-4 episodes did. Interleaving was checked, fractions were checked, episode
identifiers were checked, and nothing checked whether the hidden latent decided
anything at all, so ACR, RefusalRecall and FAR measured the labeller.

Three refusals live in :meth:`ScenePack.__post_init__`.

* **Duplicate episode identifiers.** The identifier is what the risk-coverage sort,
  the calibration exclusion list and the leak check all key on. Two episodes sharing
  one is not a naming inconvenience, it silently halves one of those three.
* **Duplicate frame content across episodes.** ``len(set(frame hashes)) ==
  n_episodes`` is the fixture-duplication check. A pack that rendered the same frame
  twice is measuring how often the harness sees that frame, not how often the model
  is right.
* **A knowability outside the closed enum.** Free text there would let a fixture
  author write a reason no checker knows how to verify.

Everything here is stdlib. The format layer has no numpy in it on purpose: an
outside submitter has to be able to read one of these manifests and check a hash
without taking on a dependency tree to do it, which is the same reason
``scripts/score_standalone.py`` exists.
"""

from __future__ import annotations

import dataclasses
import hashlib
import itertools
import json
import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..types import ACTION_FIELDS, GRIPPER_UNCHANGED
from .metrics import tolerance_error
from .tasks import (
  FULL_OCCLUSION,
  INDEPENDENT_LATENT,
  KNOWABILITY_REASONS,
  KNOWABLE,
  S2_RESIDUAL_RANGE,
  S3_TARGET_UL,
  S5_ENGAGEMENT_BAND,
  TASKS,
  UNKNOWABLE_REASONS,
  BenchTask,
  require_interleaving,
  task_for,
)

#: The manifest schema version. Bumped whenever a field changes meaning, never
#: whenever one is added: a reader that guessed at an unknown version would default
#: exactly the fields whose absence changes a score.
SCENES_FORMAT_VERSION = 1

#: What a pack that has never touched hardware reads. Every row printed from such a
#: pack reads ``basis: simulated`` and may never be called bench-validated. The lab
#: OT-2 that would not power on is the reason this is a constant and not a flag.
BASIS_SIMULATED = "simulated"
BASIS_HARDWARE = "hardware"

#: Splits a pack may declare. ``dev`` publishes its transition coefficients so a
#: submitter can develop against them; ``scoring`` draws them per episode and does
#: not publish them, because publishing them turns the board into a regression
#: contest that never touches a world model.
SPLITS: Tuple[str, ...] = ("dev", "scoring")

#: Where a frame's timestamp came from, mirroring
#: :data:`plr_lr.learn.predict.CLOCK_SOURCES`. Restated rather than imported so the
#: format layer stays free of numpy, and pinned by a test so the two cannot drift.
CLOCK_SOURCES: Tuple[str, ...] = ("device_30000", "host_arrival", "rendered")


class PackFormatError(ValueError):
  """Raised when a manifest cannot be read as a scene pack at all."""


class PackVersionMismatch(PackFormatError):
  """Raised when a manifest declares a format version this harness does not speak.

  A best-effort read is the permissive alternative and it is worse than a crash: the
  fields a newer manifest adds are exactly the ones a scorer would default, and the
  default for ``knowability`` scores a correct refusal as a false alarm.
  """


class FixtureDuplication(PackFormatError):
  """Raised when two episodes carry the same frames. Trap 4's duplication check."""


class PackContentError(PackFormatError):
  """Raised when a manifest is well formed but describes something unscoreable."""


class UnknownTaskIds(PackContentError):
  """Raised when a pack names a task this harness does not define.

  Names the ids it found and the ids it expected, in that order, because the two
  vocabularies are the whole content of the failure and a message carrying neither
  sends the reader to the wrong repository. The identifier selects the tolerance,
  the state-field layout, the horizons and the hidden-branch count; a pack that
  names its own vocabulary is asking for all four to be guessed.
  """


class UnknowableSeparationError(PackContentError):
  """Raised when an episode labelled unknowable is decided by nothing hidden.

  The build stops here and no manifest is written. Emitting the pack and noting the
  problem somewhere would produce a scoreboard whose abstention columns are about
  the fixture author: an episode whose terminal state is the same for every value of
  the hidden latent has one right answer, a model that predicts it is right, and
  crediting a refusal on it means crediting a refusal to answer a question that was
  never open.
  """


class SeparationEvidenceWithheld(UnknowableSeparationError):
  """Raised when the separation gate has nothing to run on, rather than a bad label.

  Kept apart from its parent because the two demand opposite handling and the
  difference is the whole of MAJOR 1. A failed label is a fixture that must not
  reach a scoreboard, so it raises out of the load. Withheld evidence is a scoring
  pack behaving exactly as designed, so it produces a stated ``UNCHECKED`` status
  that prints on the receipt. Collapsing them either way is wrong in a different
  direction: raising on a scoring pack makes the format unreadable, and swallowing a
  failed label is the silent pass this gate exists to prevent.
  """


class FixtureMismatch(PackFormatError):
  """Raised when a submission's frame hashes do not match the pack. Rule R16.

  Makes "did you run on our data" mechanically checkable without hosting the data,
  which is the only version of that check a solo public build can actually operate.
  """


@dataclass(frozen=True)
class SceneFrame:
  """One rendered or captured frame, identified well enough to argue about later."""

  path: str
  sha256: str
  t_capture_s: float
  clock_source: str = "rendered"

  def __post_init__(self) -> None:
    if not isinstance(self.path, str) or not self.path:
      raise PackContentError("SceneFrame.path must be a non-empty string")
    if not isinstance(self.sha256, str) or len(self.sha256) < 8:
      raise PackContentError(
        f"SceneFrame.sha256 is {self.sha256!r}. An unhashed frame cannot be matched "
        "against a manifest, so a submission carrying one can never be shown to have "
        "run on the data it claims (R16)."
      )
    if self.clock_source not in CLOCK_SOURCES:
      raise PackContentError(
        f"SceneFrame.clock_source must be one of {list(CLOCK_SOURCES)}, got "
        f"{self.clock_source!r}. An unnamed clock reads like a device clock in a table "
        "and is usually a host arrival time."
      )
    t = float(self.t_capture_s)
    if not math.isfinite(t):
      raise PackContentError("SceneFrame.t_capture_s must be finite")
    object.__setattr__(self, "t_capture_s", t)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "path": self.path,
      "sha256": self.sha256,
      "t_capture_s": self.t_capture_s,
      "clock_source": self.clock_source,
    }

  @classmethod
  def from_dict(cls, raw: Dict[str, Any]) -> "SceneFrame":
    try:
      return cls(
        path=str(raw["path"]),
        sha256=str(raw["sha256"]),
        t_capture_s=float(raw["t_capture_s"]),
        clock_source=str(raw.get("clock_source", "rendered")),
      )
    except KeyError as exc:
      raise PackFormatError(f"frame entry is missing {exc.args[0]!r}") from None


@dataclass(frozen=True)
class SceneEpisode:
  """One episode: what the model may condition on, and what it will be scored on.

  ``hidden`` records the invisible latent for an independence-constructed episode.
  It is written into the pack because a claim that an episode is unknowable has to
  be checkable by somebody who did not build it, and it is never handed to a
  predictor: :class:`plr_lr.learn.predict.RolloutRequest` has no field it could
  travel in, which is a stronger guarantee than a rule about not passing it.
  """

  episode_id: str
  task_id: str
  state0: Tuple[float, ...]
  actions: Tuple[Tuple[float, ...], ...]
  terminal_by_horizon: Dict[int, Tuple[float, ...]]
  knowability: str
  frames: Tuple[SceneFrame, ...] = ()
  coefficients: Optional[Dict[str, float]] = None
  hidden: Optional[Dict[str, float]] = None
  hidden_keys: Tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if not isinstance(self.episode_id, str) or not self.episode_id:
      raise PackContentError("SceneEpisode.episode_id must be a non-empty string")
    if self.knowability not in KNOWABILITY_REASONS:
      raise PackContentError(
        f"episode {self.episode_id} declares knowability {self.knowability!r}, which "
        f"is not one of {list(KNOWABILITY_REASONS)}. The enum is closed so a fixture "
        "cannot invent a reason no checker knows how to verify."
      )
    object.__setattr__(self, "state0", tuple(float(v) for v in self.state0))
    rows: List[Tuple[float, ...]] = []
    for row in self.actions:
      values = tuple(float(v) for v in row)
      if len(values) != len(ACTION_FIELDS):
        raise PackContentError(
          f"episode {self.episode_id} has an action row of width {len(values)}; the "
          f"layout is {list(ACTION_FIELDS)}."
        )
      if not all(math.isfinite(v) for v in values):
        raise PackContentError(f"episode {self.episode_id} carries a non-finite action")
      rows.append(values)
    if not rows:
      raise PackContentError(f"episode {self.episode_id} carries no actions")
    object.__setattr__(self, "actions", tuple(rows))
    terminals: Dict[int, Tuple[float, ...]] = {}
    for key, value in self.terminal_by_horizon.items():
      horizon = int(key)
      if horizon < 1 or horizon > len(rows):
        raise PackContentError(
          f"episode {self.episode_id} declares a terminal at horizon {horizon} but "
          f"carries {len(rows)} actions. The horizon is the number of actions."
        )
      terminals[horizon] = tuple(float(v) for v in value)
    if not terminals:
      raise PackContentError(f"episode {self.episode_id} declares no terminal state")
    object.__setattr__(self, "terminal_by_horizon", terminals)
    object.__setattr__(self, "frames", tuple(self.frames))
    object.__setattr__(self, "hidden_keys", tuple(str(k) for k in self.hidden_keys))

  @property
  def unknowable(self) -> bool:
    return self.knowability in UNKNOWABLE_REASONS

  @property
  def frames_digest(self) -> str:
    """One hash standing for this episode's frames, in order.

    Order is part of the identity. Two episodes that rendered the same frames in a
    different sequence are different observations of different runs, and folding
    them together would make the duplication check pass on a pack that duplicated.
    """
    joined = "|".join(f.sha256 for f in self.frames)
    return hashlib.sha256(joined.encode("ascii")).hexdigest()

  def terminal(self, horizon: int) -> Tuple[float, ...]:
    try:
      return self.terminal_by_horizon[int(horizon)]
    except KeyError:
      raise PackContentError(
        f"episode {self.episode_id} carries no truth at horizon {horizon}; it has "
        f"{sorted(self.terminal_by_horizon)}. Scoring against an interpolated truth "
        "would be scoring against a number this pack never generated."
      ) from None

  def to_dict(self) -> Dict[str, Any]:
    out: Dict[str, Any] = {
      "episode_id": self.episode_id,
      "task_id": self.task_id,
      "state0": list(self.state0),
      "actions": [list(row) for row in self.actions],
      "terminal_by_horizon": {
        str(k): list(v) for k, v in sorted(self.terminal_by_horizon.items())
      },
      "knowability": self.knowability,
      "frames": [f.to_dict() for f in self.frames],
      "hidden_keys": list(self.hidden_keys),
    }
    if self.coefficients is not None:
      out["coefficients"] = {k: float(v) for k, v in sorted(self.coefficients.items())}
    if self.hidden is not None:
      out["hidden"] = {k: float(v) for k, v in sorted(self.hidden.items())}
    return out

  @classmethod
  def from_dict(cls, raw: Dict[str, Any]) -> "SceneEpisode":
    try:
      return cls(
        episode_id=str(raw["episode_id"]),
        task_id=str(raw["task_id"]),
        state0=tuple(float(v) for v in raw["state0"]),
        actions=tuple(tuple(float(v) for v in row) for row in raw["actions"]),
        terminal_by_horizon={
          int(k): tuple(float(v) for v in row)
          for k, row in raw["terminal_by_horizon"].items()
        },
        knowability=str(raw["knowability"]),
        frames=tuple(SceneFrame.from_dict(f) for f in raw.get("frames", ())),
        coefficients=(
          None if raw.get("coefficients") is None
          else {str(k): float(v) for k, v in raw["coefficients"].items()}
        ),
        hidden=(
          None if raw.get("hidden") is None
          else {str(k): float(v) for k, v in raw["hidden"].items()}
        ),
        hidden_keys=tuple(str(k) for k in raw.get("hidden_keys", ())),
      )
    except KeyError as exc:
      raise PackFormatError(f"episode entry is missing {exc.args[0]!r}") from None


@dataclass(frozen=True)
class ScenePack:
  """A whole fixture: episodes, the seed epoch they came from, and their basis.

  ``seed_epoch`` prints on every row and cross-epoch tables are refused (R11).
  Seed secrecy is a speed bump and this build says so out loud: the generator is
  public, so anyone can enumerate seeds. The real defences are the interventional
  gate, the seed-leak control and the declared training fingerprints, and pretending
  the epoch is a guarantee would be the dishonest part.
  """

  pack_id: str
  seed_epoch: str
  split: str
  source: str
  episodes: Tuple[SceneEpisode, ...]
  format_version: int = SCENES_FORMAT_VERSION
  created_utc: str = ""
  device_serial_sha256: Optional[str] = None
  #: What the separation gate was able to establish about this pack's unknowable
  #: labels. Derived at load or at build and deliberately not a manifest field: a
  #: pack that carried its own certificate would be certifying itself, which is the
  #: claim the gate exists to replace. ``None`` means the gate has not been run at
  #: all, which is a third answer and is printed as one.
  separation: Optional["SeparationCertificate"] = None

  def __post_init__(self) -> None:
    if int(self.format_version) != SCENES_FORMAT_VERSION:
      raise PackVersionMismatch(
        f"this pack declares scenes format {self.format_version} and this harness "
        f"speaks {SCENES_FORMAT_VERSION}. Reading it anyway would default whichever "
        "fields changed, and the field whose default changes a score is knowability."
      )
    if self.split not in SPLITS:
      raise PackContentError(
        f"ScenePack.split must be one of {list(SPLITS)}, got {self.split!r}. A pack "
        "that will not say which side of the dev/scoring partition it is on cannot "
        "have the partition checked, and the partition is what the seed-leak control "
        "measures."
      )
    for name in ("pack_id", "seed_epoch", "source"):
      value = getattr(self, name)
      if not isinstance(value, str) or not value:
        raise PackContentError(f"ScenePack.{name} must be a non-empty string")
    episodes = tuple(self.episodes)
    if not episodes:
      raise PackContentError(
        "a pack with no episodes is not a fixture. Scoring against it would produce "
        "an empty result that reads on a scorecard exactly like a clean run."
      )
    ids = [e.episode_id for e in episodes]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
      raise PackContentError(
        f"episode identifiers {duplicates} appear more than once. The identifier is "
        "what the risk-coverage sort, the calibration exclusion list and the leak "
        "check all key on."
      )
    with_frames = [e for e in episodes if e.frames]
    digests = {e.frames_digest for e in with_frames}
    if with_frames and len(digests) != len(with_frames):
      raise FixtureDuplication(
        f"{len(with_frames)} episodes carry frames but only {len(digests)} distinct "
        "frame sequences. A pack that rendered the same frames twice measures how "
        "often the harness saw that frame, not how often the model was right."
      )
    object.__setattr__(self, "episodes", episodes)
    object.__setattr__(self, "format_version", int(self.format_version))

  # -- what the pack is entitled to say about itself ---------------------------

  @property
  def basis(self) -> str:
    """``simulated`` until a hardware run record with a device serial hash exists.

    Refusal rule R13. There is no argument that overrides this and no flag that sets
    it: the label follows the presence of a device serial hash and nothing else,
    because every other way of deciding it has already been a claim somebody made
    about a run that never touched an arm.
    """
    if self.device_serial_sha256:
      return BASIS_HARDWARE
    return BASIS_SIMULATED

  @property
  def is_simulation(self) -> bool:
    return self.basis == BASIS_SIMULATED

  @property
  def task_ids(self) -> Tuple[str, ...]:
    seen: List[str] = []
    for episode in self.episodes:
      if episode.task_id not in seen:
        seen.append(episode.task_id)
    return tuple(seen)

  def for_task(self, task_id: str) -> Tuple[SceneEpisode, ...]:
    return tuple(e for e in self.episodes if e.task_id == task_id)

  def frames_sha256(self) -> Tuple[str, ...]:
    """Every frame hash in the pack, in pack order. The R16 comparison key."""
    out: List[str] = []
    for episode in self.episodes:
      out.extend(f.sha256 for f in episode.frames)
    return tuple(out)

  def check_interleaved(self) -> Tuple[Any, ...]:
    """Require every task in this pack to interleave its unknowable episodes."""
    reports = []
    for task_id in self.task_ids:
      episodes = self.for_task(task_id)
      reports.append(
        require_interleaving(
          task_id,
          [e.episode_id for e in episodes],
          [e.knowability for e in episodes],
        )
      )
    return tuple(reports)

  def to_dict(self) -> Dict[str, Any]:
    return {
      "format_version": int(self.format_version),
      "pack_id": self.pack_id,
      "seed_epoch": self.seed_epoch,
      "split": self.split,
      "source": self.source,
      "created_utc": self.created_utc,
      "device_serial_sha256": self.device_serial_sha256,
      "basis": self.basis,
      "episodes": [e.to_dict() for e in self.episodes],
    }

  @classmethod
  def from_dict(cls, raw: Dict[str, Any]) -> "ScenePack":
    if not isinstance(raw, dict):
      raise PackFormatError(f"a manifest is a JSON object, got {type(raw).__name__}")
    if "format_version" not in raw:
      raise PackVersionMismatch(
        "this manifest declares no format_version. An undeclared version is read as "
        "whichever one the reader happens to implement, which is how a knowability "
        "field goes missing without anybody noticing."
      )
    try:
      return cls(
        format_version=int(raw["format_version"]),
        pack_id=str(raw["pack_id"]),
        seed_epoch=str(raw["seed_epoch"]),
        split=str(raw["split"]),
        source=str(raw["source"]),
        created_utc=str(raw.get("created_utc", "")),
        device_serial_sha256=(
          None if raw.get("device_serial_sha256") in (None, "")
          else str(raw["device_serial_sha256"])
        ),
        episodes=tuple(SceneEpisode.from_dict(e) for e in raw["episodes"]),
      )
    except KeyError as exc:
      raise PackFormatError(f"manifest is missing {exc.args[0]!r}") from None


def load_manifest(path: str) -> ScenePack:
  """Read one ``manifest.json`` into a :class:`ScenePack`, or refuse it.

  Three checks, in the order a failure is cheapest to explain: the shape, then the
  task vocabulary, then the unknowable labels. The last two are the ones that used
  to be missing here, and both were missing in the same way -- wired into
  :func:`builtin_demo_pack`, so they held for the pack this repository generates and
  for no other. An externally produced pack is the case a file format exists for and
  was the case neither one covered.

  A pack that withholds the coefficients the separation gate needs is not refused.
  It is loaded carrying :data:`SEPARATION_UNCHECKED` on
  :attr:`ScenePack.separation`, which the runner prints on the scorecard.

  Raises:
    PackFormatError: unreadable, or a shape this format version does not describe.
    UnknownTaskIds: a task id this harness does not define.
    UnknowableSeparationError: an unknowable label the evidence refutes.
  """
  try:
    with open(path, "r", encoding="ascii") as handle:
      raw = json.load(handle)
  except (OSError, ValueError) as exc:
    raise PackFormatError(f"cannot read a scene manifest at {path}: {exc}") from exc
  pack = ScenePack.from_dict(raw)
  require_known_tasks(pack)
  return dataclasses.replace(pack, separation=certify_separation(pack))


def write_manifest(pack: ScenePack, path: str) -> None:
  """Write a pack out in the one format anything else is allowed to read."""
  with open(path, "w", encoding="ascii") as handle:
    json.dump(pack.to_dict(), handle, indent=2, sort_keys=True)
    handle.write("\n")


def require_fixture_match(pack: ScenePack, submitted_sha256: Sequence[str]) -> None:
  """Refuse a submission whose frame hashes are not this pack's. Rule R16.

  Prints ``FIXTURE-MISMATCH`` and scores nothing. The permissive alternative is to
  score whatever arrived and note the discrepancy in a footnote, which produces a
  ranked row for a run on data nobody can identify.
  """
  want = list(pack.frames_sha256())
  got = [str(s) for s in submitted_sha256]
  if want == got:
    return
  want_set = set(want)
  got_set = set(got)
  missing = sorted(want_set - got_set)[:3]
  extra = sorted(got_set - want_set)[:3]
  raise FixtureMismatch(
    f"FIXTURE-MISMATCH: pack {pack.pack_id} carries {len(want)} frame hashes and the "
    f"submission carries {len(got)}; {len(want_set - got_set)} expected hashes are "
    f"absent and {len(got_set - want_set)} unexpected ones are present. Nothing is "
    f"scored. First missing {missing}, first unexpected {extra}."
  )


# -- the declared transitions --------------------------------------------------
#
# One function per task, and the same function generates a pack and reproduces a
# truth for the shuffled-action and oracle controls. Two implementations of a
# transition drift, and the direction they drift in is always the one that makes
# the control look like it passed.


#: The published nominal coefficients. These are the dev-pack constants and a
#: submitter may develop against them. A scoring pack draws its coefficients per
#: episode around these and does not publish them: publishing the constants for a
#: scored task turns the board into a twenty-line regression contest that never
#: touches a world model.
NOMINAL: Dict[str, Dict[str, float]] = {
  "S1": {
    "gain_lo": 0.94,
    "gain_hi": 0.86,
    "boundary_mm": 0.0,
    "drift_mm": -0.35,
    "drift_onset": 1.0,
    "bias_mm": 0.0,
  },
  "S2": {
    "c_wash": 0.08,
    "c_blot": 0.02,
    "c_dry": 0.04,
    "c_quad": 0.80,
    "leave": 0.72,
    "drydown": 0.80,
    "wet_uL": 0.0,
  },
  "S3": {
    "c_cmd": 0.96,
    "c_wall": 0.06,
    "c_drop": 0.04,
    "c_quad": 0.80,
    "topup": 0.15,
    "aspirate": 0.15,
    "bias_uL": 0.0,
  },
  "S4": {
    "c_corr": 0.82,
    "c_press": 0.02,
    "reseat": 0.35,
    "press_lo": 0.5,
    "press_hi": 3.0,
    "seat_mm": 0.5,
    "seat_deg": 0.5,
    "occ_dx_mm": 0.0,
    "occ_dy_mm": 0.0,
  },
  "S5": {
    "gain": 0.90,
    "c_mag": 0.05,
    "max_lateral_mm": 1.0,
    "band_lo": S5_ENGAGEMENT_BAND[0],
    "band_hi": S5_ENGAGEMENT_BAND[1],
    "gap_mm": 0.0,
  },
}

#: The values an unknowable episode draws its hidden latent from, per task. The
#: nominal is never in a support: an episode that drew it would be exactly the
#: episode a model holding the published constants predicts correctly, and a fifth
#: of those would be labelled unknowable and credited to whoever refused them.
#:
#: The supports are wide for a reason that is arithmetic rather than taste. Two
#: branches a single tolerance apart are both inside tolerance of one prediction
#: placed between them, so the ceiling a task claims is one over the branch count
#: only when the branches are more than *two* tolerances apart. Every gap below
#: clears that, and :func:`require_unknowable_separation` checks the terminal states
#: rather than these numbers, because what has to be separated is the outcome and
#: not the coefficient.
HIDDEN_SUPPORT: Dict[str, Dict[str, Tuple[float, ...]]] = {
  # tolerance 2.0 mm on z: gaps of 4.5 mm and 5.0 mm, nearest branch 2.5 mm off nominal
  "S1": {"bias_mm": (-7.0, -2.5, 2.5, 7.0)},
  # tolerance 0.30 uL: gaps of 0.65 uL, nearest branch 0.45 uL off nominal. Retention
  # is one-signed because a wet tip carries extra liquid and never less than none,
  # and the largest of them keeps the terminal inside S2_RESIDUAL_RANGE.
  "S2": {"wet_uL": (0.45, 1.10, 1.75)},
  # tolerance 0.50 uL: gaps of 1.2 uL, nearest branch 1.2 uL off nominal
  "S3": {"bias_uL": (-2.4, -1.2, 1.2, 2.4)},
  # tolerance 0.5 mm on dx and dy: two values per axis, 3.2 mm apart, so any two of
  # the four corner branches differ by more than two tolerances on at least one axis
  "S4": {"occ_dx_mm": (-1.6, 1.6), "occ_dy_mm": (-1.6, 1.6)},
  # tolerance 0.75 mm on the engagement depth: gaps of 1.8 mm, nearest branch 1.8 mm
  # off nominal. Both signs are real failures: a tip short of the mandrel and a tip
  # driven past its shoulder.
  "S5": {"gap_mm": (-3.6, -1.8, 1.8, 3.6)},
}

#: A branch pair closer than this many tolerances is refused. One prediction placed
#: between two such branches is inside tolerance of both, so the episode has a
#: partial answer and the chance ceiling the task declares is an overstatement.
MIN_BRANCH_SEPARATION = 2.0

#: A branch closer than this many tolerances to the nominal-latent terminal is
#: refused. That episode is answered correctly by a model that holds the published
#: constants and ignores the hidden latent entirely, which is the one thing an
#: unknowable label promises cannot happen.
MIN_NOMINAL_SEPARATION = 1.0


def _norm3(p0: float, p1: float, p2: float) -> float:
  return math.sqrt(p0 * p0 + p1 * p1 + p2 * p2)


def _advance_s1(
  state: Sequence[float],
  actions: Sequence[Sequence[float]],
  c: Dict[str, float],
) -> Tuple[float, ...]:
  x, y, z, rz = (float(v) for v in state)
  onset = int(round(c["drift_onset"]))
  for t, row in enumerate(actions):
    p0, p1, p2, p3 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    gain = c["gain_hi"] if x >= c["boundary_mm"] else c["gain_lo"]
    x += gain * p0
    y += gain * p1
    z += gain * p2
    rz += gain * p3
    if t == 0:
      # The hidden step change lands on the first commanded move and on no other, so
      # it is in the terminal state at horizon 1 as much as at horizon 8. The drift
      # below is the visible half of the story and fires from its onset step; a
      # latent that only acted from the onset could not make a horizon-1 episode
      # unknowable, because at horizon 1 it would not have acted at all.
      z += c["bias_mm"]
    if t >= onset:
      z += c["drift_mm"]
  return (x, y, z, rz)


def _advance_s2(
  state: Sequence[float],
  actions: Sequence[Sequence[float]],
  c: Dict[str, float],
) -> Tuple[float, ...]:
  r = float(state[0])
  for row in actions:
    p0, p1, p2, p3 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    # Multiplicative drawdown, not subtractive. A subtractive one drives the residual
    # through zero on a long sequence, and the two ways of dealing with that are both
    # worse: leaving it negative publishes an unphysical truth, and clamping it at
    # zero makes the terminal state stop responding to the action, which the
    # interventional gate cannot tell apart from a model that ignores its actions.
    exponent = (
      c["c_wash"] * p0
      + c["c_blot"] * abs(p1)
      + c["c_dry"] * abs(p2)
      + c["c_quad"] * (p0 * p0 + p1 * p1 + p2 * p2) / 100.0
    )
    r *= math.exp(-exponent)
    verb = int(round(p3))
    if verb == 1:
      r *= c["leave"]
    elif verb == 2:
      r *= c["drydown"]
  # The retained film of a wet tip, added at the end. Adding it at the first step
  # instead would put it through every multiplicative drawdown that follows, and at
  # horizon 3 two branches that started 0.65 uL apart can land inside one 0.30 uL
  # tolerance of each other, which is an unknowable episode with a knowable answer.
  return (r + c["wet_uL"],)


def _advance_s3(
  state: Sequence[float],
  actions: Sequence[Sequence[float]],
  c: Dict[str, float],
) -> Tuple[float, ...]:
  v = float(state[0])
  for t, row in enumerate(actions):
    p0, p1, p2, p3 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
    if t == 0:
      # The hidden calibration offset lands on the first commanded step, whatever
      # that step was. A multiplier on the commanded stroke was the earlier
      # construction and it had no effect at all on a PROCEED step, whose commanded
      # stroke is zero, so a third of the horizon-1 episodes were decided by nothing.
      v += c["bias_uL"]
    v += c["c_cmd"] * p0
    v -= c["c_wall"] * abs(p1) + c["c_drop"] * abs(p2)
    v -= c["c_quad"] * (p0 * p0 + p1 * p1 + p2 * p2) / 100.0
    verb = int(round(p3))
    if verb == 1:
      v += c["topup"]
    elif verb == 2:
      v -= c["aspirate"]
  return (v,)


def _advance_s4(
  state: Sequence[float],
  actions: Sequence[Sequence[float]],
  c: Dict[str, float],
) -> Tuple[float, ...]:
  dx, dy, drz = float(state[0]), float(state[1]), float(state[2])
  press = 0.0
  for row in actions:
    p0, p1, p2, p3, grip = (float(row[i]) for i in range(5))
    dx -= c["c_corr"] * p0
    dy -= c["c_corr"] * p1
    drz -= c["c_corr"] * p3
    drz += c["c_press"] * p2
    press += p2
    if int(round(grip)) == 1:
      dx *= c["reseat"]
      dy *= c["reseat"]
      drz *= c["reseat"]
      press = 0.0
  # The nest offset at the occluded corner, added to the terminal and therefore not
  # subject to the reseat contraction above. Carrying it as an initial condition put
  # it through that contraction: two reseats take a 3.2 mm branch separation to
  # 0.39 mm, and the tolerance on this task is 0.5 mm.
  dx += c["occ_dx_mm"]
  dy += c["occ_dy_mm"]
  seated = 1.0
  if abs(dx) > c["seat_mm"] or abs(dy) > c["seat_mm"] or abs(drz) > c["seat_deg"]:
    seated = 0.0
  elif not c["press_lo"] <= press <= c["press_hi"]:
    seated = 0.0
  return (dx, dy, drz, seated)


def _advance_s5(
  state: Sequence[float],
  actions: Sequence[Sequence[float]],
  c: Dict[str, float],
) -> Tuple[float, ...]:
  z = float(state[0])
  lateral = 0.0
  for row in actions:
    p0, p1, p2, grip = float(row[0]), float(row[1]), float(row[2]), float(row[4])
    z += c["gain"] * p2
    z -= c["c_mag"] * _norm3(p0, p1, p2)
    lateral += math.sqrt(p0 * p0 + p1 * p1)
    if int(round(grip)) == 1:
      # A release retains a tenth of the engagement and a tenth of the accumulated
      # lateral offset rather than resetting both to zero. Resetting to a constant
      # would make an episode that ends in a release have a terminal state that does
      # not depend on its actions at all, and the interventional gate cannot tell
      # that apart from a model that ignores its action argument.
      z *= 0.1
      lateral *= 0.1
  # The seating gap behind the tip shoulder: invisible in the tip view, added to the
  # apparent depth before the attachment test, and added at the end because a
  # release multiplies the depth by a tenth and would take a 3.6 mm separation to
  # 0.36 mm against a 0.75 mm tolerance. The earlier construction hid a boolean seal
  # here instead, which moved the terminal on an episode only when the depth already
  # sat inside the band, so most of the episodes it labelled were decided by nothing.
  z += c["gap_mm"]
  engaged = c["band_lo"] <= z <= c["band_hi"] and lateral <= c["max_lateral_mm"]
  attached = 1.0 if engaged else 0.0
  return (z, attached)


_ADVANCE = {
  "S1": _advance_s1,
  "S2": _advance_s2,
  "S3": _advance_s3,
  "S4": _advance_s4,
  "S5": _advance_s5,
}


def advance(
  task_id: str,
  state: Sequence[float],
  actions: Sequence[Sequence[float]],
  coefficients: Dict[str, float],
) -> Tuple[float, ...]:
  """Roll one task's declared transition forward over a whole action sequence.

  The single implementation. The pack generator calls it to produce truth, the
  shuffled-action control calls it to produce truth under somebody else's actions,
  and a submitter can call it to check a number. A second copy of this arithmetic
  anywhere would be the version that quietly disagrees, and it would disagree in
  whichever direction made a control look like it passed.
  """
  try:
    fn = _ADVANCE[task_id]
  except KeyError:
    raise PackContentError(
      f"no declared transition for task {task_id!r}; known tasks are "
      f"{sorted(_ADVANCE)}."
    ) from None
  merged = dict(NOMINAL[task_id])
  merged.update({str(k): float(v) for k, v in coefficients.items()})
  return fn(state, actions, merged)


# -- the separation gate -------------------------------------------------------
#
# The falsifier. It answers one question per episode per horizon: with everything
# else held exactly as this episode has it, does the value of the hidden latent
# change the terminal state by more than the tolerance the episode is scored at?
# A no means the episode is knowable and the label is refused. A yes means only
# that this construction did not fail, never that the label is established: the
# frames could still carry the latent, which is a question about pixels and is
# checked by the identifiability oracle on the vision side. Nothing here promotes
# an episode to unknowable, and nothing here reads a knowable episode at all.


def hidden_grid(task_id: str) -> Tuple[Tuple[Tuple[str, float], ...], ...]:
  """Every assignment of this task's hidden latent, as (key, value) pairs.

  The product over keys, so S4's two axes give four corner branches. Sorted by key
  so the branch order is the same on every run, which is what lets a refusal message
  be compared against the one somebody else got.
  """
  try:
    support = HIDDEN_SUPPORT[task_id]
  except KeyError:
    raise PackContentError(
      f"no declared hidden support for task {task_id!r}; known tasks are "
      f"{sorted(HIDDEN_SUPPORT)}. An unknowable label on a task with no declared "
      "support is a claim with nothing behind it."
    ) from None
  keys = sorted(support)
  return tuple(
    tuple(zip(keys, values)) for values in itertools.product(*(support[k] for k in keys))
  )


def _label(assignment: Sequence[Tuple[str, float]]) -> str:
  return "{" + ", ".join(f"{key}={value:g}" for key, value in assignment) + "}"


@dataclass(frozen=True)
class SeparationReport:
  """What the hidden latent is worth on one episode at one horizon.

  Both distances are in tolerance units, per field over that field's tolerance and
  then a max over fields, which is the same criterion
  :func:`plr_lr.learn.bench.metrics.tolerance_error` scores a prediction with. A
  separation computed any other way would be a number that does not answer the
  question the score asks.
  """

  episode_id: str
  task_id: str
  horizon: int
  n_branches: int
  min_branch_separation: float
  min_nominal_separation: float
  closest_pair: str = ""
  closest_branch: str = ""

  @property
  def decides(self) -> bool:
    """Does the latent move the truth by more than one tolerance at all?

    The minimum the label needs. Below it the episode has one answer and the model
    that gives it is right, so the label is false and a refusal on that episode is
    credit for refusing a question nobody asked.
    """
    return self.min_branch_separation > 1.0

  @property
  def ok(self) -> bool:
    return (
      self.n_branches >= 2
      and self.min_branch_separation > MIN_BRANCH_SEPARATION
      and self.min_nominal_separation > MIN_NOMINAL_SEPARATION
    )

  def reasons(self) -> Tuple[str, ...]:
    out: List[str] = []
    if self.n_branches < 2:
      out.append(
        f"the hidden grid offers {self.n_branches} branch(es); one branch is a "
        "prediction, not an ambiguity"
      )
    if not self.decides:
      out.append(
        f"the closest branch pair {self.closest_pair} lands "
        f"{self.min_branch_separation:.4f} tolerances apart, so every value of the "
        "hidden latent produces the same terminal state to within the tolerance this "
        "episode is scored at. The episode is knowable and the label is false"
      )
    elif self.min_branch_separation <= MIN_BRANCH_SEPARATION:
      out.append(
        f"the closest branch pair {self.closest_pair} lands "
        f"{self.min_branch_separation:.4f} tolerances apart, under the "
        f"{MIN_BRANCH_SEPARATION:g} this gate requires. One prediction placed between "
        "them is inside tolerance of both, so part of the answer is available without "
        "the latent and the declared chance ceiling is an overstatement"
      )
    if self.min_nominal_separation <= MIN_NOMINAL_SEPARATION:
      out.append(
        f"branch {self.closest_branch} lands {self.min_nominal_separation:.4f} "
        "tolerances from the terminal the published nominal produces, so a model that "
        "holds the nominal and ignores the hidden latent scores on this episode"
      )
    return tuple(out)

  def format(self) -> str:
    verdict = "ok" if self.ok else "REFUSED"
    head = (
      f"separation {self.episode_id}@{self.horizon}: {verdict} "
      f"({self.n_branches} branches, closest pair "
      f"{self.min_branch_separation:.4f} tol, closest to nominal "
      f"{self.min_nominal_separation:.4f} tol)"
    )
    return "\n".join([head] + [f"  {reason}" for reason in self.reasons()])


def separation_report(
  task: BenchTask,
  episode: SceneEpisode,
  coefficients: Dict[str, float],
  horizon: int,
) -> SeparationReport:
  """Advance one episode under every value in its hidden grid, at one horizon."""
  grid = hidden_grid(task.task_id)
  base = {str(k): float(v) for k, v in coefficients.items()}
  actions = episode.actions[: int(horizon)]
  nominal_coefficients = dict(base)
  for key in sorted(HIDDEN_SUPPORT[task.task_id]):
    nominal_coefficients[key] = float(NOMINAL[task.task_id][key])
  nominal = advance(task.task_id, episode.state0, actions, nominal_coefficients)

  terminals: List[Tuple[float, ...]] = []
  for assignment in grid:
    branch = dict(base)
    branch.update({key: float(value) for key, value in assignment})
    terminal = advance(task.task_id, episode.state0, actions, branch)
    if not all(math.isfinite(v) for v in terminal):
      raise UnknowableSeparationError(
        f"episode {episode.episode_id} at horizon {horizon} advances to a non-finite "
        f"terminal under {_label(assignment)}. An unscoreable truth cannot be shown "
        "to be separated from anything."
      )
    terminals.append(terminal)

  min_pair = float("inf")
  closest_pair = ""
  for i, j in itertools.combinations(range(len(terminals)), 2):
    distance = tolerance_error(terminals[i], terminals[j], task.tolerance)
    if distance < min_pair:
      min_pair = distance
      closest_pair = f"{_label(grid[i])} vs {_label(grid[j])}"
  min_nominal = float("inf")
  closest_branch = ""
  for index, terminal in enumerate(terminals):
    distance = tolerance_error(terminal, nominal, task.tolerance)
    if distance < min_nominal:
      min_nominal = distance
      closest_branch = _label(grid[index])
  return SeparationReport(
    episode_id=episode.episode_id,
    task_id=task.task_id,
    horizon=int(horizon),
    n_branches=len(terminals),
    min_branch_separation=min_pair,
    min_nominal_separation=min_nominal,
    closest_pair=closest_pair,
    closest_branch=closest_branch,
  )


def _coefficients_for(
  episode: SceneEpisode,
  coefficients: Optional[Dict[str, Dict[str, float]]],
) -> Dict[str, float]:
  if coefficients is not None and episode.episode_id in coefficients:
    return dict(coefficients[episode.episode_id])
  if episode.coefficients is not None:
    return dict(episode.coefficients)
  raise SeparationEvidenceWithheld(
    f"episode {episode.episode_id} is labelled {episode.knowability} and carries no "
    "coefficients, so the claim cannot be checked from this manifest alone. A scoring "
    "pack does not publish them; the gate runs at build time, where they are in hand, "
    "and a pack whose build never ran it should not be scored."
  )


def check_unknowable_separation(
  pack: ScenePack,
  coefficients: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[SeparationReport, ...]:
  """One report per unknowable episode per scored horizon. Never promotes a label.

  Only episodes already labelled unknowable are read. That asymmetry is the whole
  discipline: a collapse falsifies an unknowable label, and no measurement here can
  establish one, because separation of the hidden branches says nothing about
  whether the frames carry the latent. Relabelling a knowable episode as unknowable
  because its hidden grid happened to separate would be reading the inconclusive
  direction, and the label it produced would be the labeller's.
  """
  reports: List[SeparationReport] = []
  for episode in pack.episodes:
    if not episode.unknowable:
      continue
    task = task_for(episode.task_id)
    support = HIDDEN_SUPPORT.get(task.task_id, {})
    declared = tuple(sorted(support))
    if tuple(sorted(episode.hidden_keys)) != declared:
      raise UnknowableSeparationError(
        f"episode {episode.episode_id} declares hidden keys "
        f"{list(episode.hidden_keys)} and task {task.task_id} declares "
        f"{list(declared)}. A hidden latent nobody declared has no grid to be "
        "separated over, so the label cannot be checked and is refused."
      )
    hidden = episode.hidden or {}
    for key in declared:
      if key not in hidden:
        raise UnknowableSeparationError(
          f"episode {episode.episode_id} records no value for its hidden {key!r}. The "
          "invisible latent is written into the pack precisely so that somebody who "
          "did not build the fixture can check the claim."
        )
      if not any(math.isclose(hidden[key], v) for v in support[key]):
        raise UnknowableSeparationError(
          f"episode {episode.episode_id} drew {key}={hidden[key]!r}, which is not in "
          f"the declared support {list(support[key])}. The ceiling this task prints is "
          "one over the size of that support, and a value outside it makes the ceiling "
          "a number about a distribution nobody wrote down."
        )
    horizons = [h for h in task.horizons if h in episode.terminal_by_horizon]
    if not horizons:
      raise UnknowableSeparationError(
        f"episode {episode.episode_id} carries no truth at any horizon task "
        f"{task.task_id} is scored at ({list(task.horizons)}), so there is no horizon "
        "at which the label could be checked."
      )
    resolved = _coefficients_for(episode, coefficients)
    for horizon in horizons:
      reports.append(separation_report(task, episode, resolved, int(horizon)))
  return tuple(reports)


def require_unknowable_separation(
  pack: ScenePack,
  coefficients: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[SeparationReport, ...]:
  """Run the gate and fail the build unless every unknowable label survives it.

  Raising rather than returning a verdict is the point. The permissive version
  writes the manifest and records the problem in a field, and then the manifest
  exists, gets scored, and produces an ACR, a RefusalRecall and an FAR that are
  about the fixture. Refusing to emit is what makes a bad label unable to reach a
  scoreboard at all.
  """
  reports = check_unknowable_separation(pack, coefficients)
  failures = [report for report in reports if not report.ok]
  if failures:
    head = (
      f"{len(failures)} of {len(reports)} unknowable episode-horizons are decided by "
      "nothing hidden. No manifest is emitted."
    )
    shown = "\n".join(report.format() for report in failures[:5])
    tail = "" if len(failures) <= 5 else f"\n... and {len(failures) - 5} more"
    raise UnknowableSeparationError(f"{head}\n{shown}{tail}")
  return reports


# -- certifying a pack this repository did not generate ------------------------

#: The gate ran on every unknowable episode in the pack and every label survived it.
SEPARATION_CHECKED = "CHECKED"

#: The gate could not run, and the pack is entitled to that: a scoring pack
#: withholds the per-episode coefficients on purpose. The status is printed and
#: propagates to the scorecard, because an unchecked label and a checked one are
#: different answers and a run that cannot tell them apart is the silent pass.
SEPARATION_UNCHECKED = "UNCHECKED"


@dataclass(frozen=True)
class SeparationCertificate:
  """What the separation gate established about one pack, and what it could not.

  This is a refusal to certify rather than a verdict of "fine". Nothing here can
  say a label is *right*: a surviving separation only says this construction did
  not collapse, and whether the frames carry the latent is a question about pixels
  that lives on the vision side. What it can say, and now does say on every pack
  rather than on the one this repository generates, is whether anybody checked.
  """

  status: str
  n_unknowable: int
  n_reports: int
  reason: str = ""
  reports: Tuple[SeparationReport, ...] = ()

  @property
  def checked(self) -> bool:
    return self.status == SEPARATION_CHECKED

  def line(self) -> str:
    if self.checked:
      return (
        f"{SEPARATION_CHECKED}: {self.n_reports} episode-horizons over "
        f"{self.n_unknowable} unknowable episodes, every branch pair more than "
        f"{MIN_BRANCH_SEPARATION:g} tolerances apart and every branch more than "
        f"{MIN_NOMINAL_SEPARATION:g} from the nominal"
      )
    return (
      f"{SEPARATION_UNCHECKED}: {self.n_unknowable} unknowable episodes carry a label "
      f"this run could not check. {self.reason} The abstention columns built from "
      "these labels are therefore uncertified, and this line prints beside them "
      "rather than the run reading as though the gate had passed."
    )

  def to_dict(self) -> Dict[str, Any]:
    return {
      "status": self.status,
      "n_unknowable": int(self.n_unknowable),
      "n_episode_horizons_checked": int(self.n_reports),
      "reason": self.reason,
    }


def certify_separation(
  pack: ScenePack,
  coefficients: Optional[Dict[str, Dict[str, float]]] = None,
) -> SeparationCertificate:
  """Run the gate on any pack, and say plainly when it could not run.

  Three outcomes, and the middle one is the point. A label that fails raises, so a
  bad fixture cannot reach a scoreboard through the front door any more than through
  the build. A pack that withholds the evidence gets
  :data:`SEPARATION_UNCHECKED` and carries it onto the receipt. Only a pack whose
  every unknowable episode was advanced under its whole hidden grid at every scored
  horizon gets :data:`SEPARATION_CHECKED`.

  Raises:
    UnknowableSeparationError: a label the evidence refutes, or an unknowable claim
      whose hidden latent is not the one its task declares. Withheld evidence is
      :class:`SeparationEvidenceWithheld` and is caught here instead.
  """
  unknowable = [episode for episode in pack.episodes if episode.unknowable]
  if not unknowable:
    return SeparationCertificate(
      status=SEPARATION_UNCHECKED,
      n_unknowable=0,
      n_reports=0,
      reason=(
        "the pack labels no episode unknowable, so there was nothing for the gate to "
        "read; that is a pack the interleaving check refuses separately, and it is "
        "not a pack whose labels were verified."
      ),
    )
  try:
    reports = require_unknowable_separation(pack, coefficients)
  except SeparationEvidenceWithheld as exc:
    return SeparationCertificate(
      status=SEPARATION_UNCHECKED,
      n_unknowable=len(unknowable),
      n_reports=0,
      reason=str(exc),
    )
  return SeparationCertificate(
    status=SEPARATION_CHECKED,
    n_unknowable=len(unknowable),
    n_reports=len(reports),
    reports=reports,
  )


# -- the task vocabulary -------------------------------------------------------


def known_task_ids(tasks: Sequence[BenchTask] = TASKS) -> Tuple[str, ...]:
  return tuple(task.task_id for task in tasks)


def require_known_tasks(
  pack: ScenePack,
  tasks: Sequence[BenchTask] = TASKS,
) -> Tuple[str, ...]:
  """Every task id in the pack has to be one this harness declares.

  The harness owns the vocabulary and it owns it for a reason that is not
  territorial: the identifier is the key that selects a tolerance, a state-field
  order, a horizon list and a hidden-branch count, and those four are declarations
  in :mod:`plr_lr.learn.bench.tasks` with no counterpart on the producing side. A
  producer free to name its own tasks is a producer asking this module to invent a
  tolerance for a name it has never seen, and every metric here divides by that
  number.

  Returns the pack's task ids in pack order.

  Raises:
    UnknownTaskIds: naming what arrived and what was expected.
  """
  expected = known_task_ids(tasks)
  found = pack.task_ids
  unknown = [task_id for task_id in found if task_id not in expected]
  if unknown:
    raise UnknownTaskIds(
      f"pack {pack.pack_id} carries task ids {sorted(unknown)} that this harness does "
      f"not define. It found {list(found)} and it expects ids drawn from "
      f"{list(expected)}. Nothing is scored: the id selects the tolerance, the state "
      "fields, the horizons and the hidden-branch count, so scoring an unrecognised "
      "one would mean guessing all four, and a guessed tolerance is not the task's "
      "tolerance."
    )
  return found


def tasks_in(
  pack: ScenePack,
  tasks: Sequence[BenchTask] = TASKS,
) -> Tuple[BenchTask, ...]:
  """The declared tasks this pack actually carries episodes for, in declared order.

  A pack is allowed to carry a subset. It is not allowed to carry a name from
  somewhere else, which :func:`require_known_tasks` settles first, and it is not
  allowed to carry nothing: a run over zero tasks emits an empty scorecard that
  reads exactly like a clean one.
  """
  found = set(require_known_tasks(pack, tasks))
  selected = tuple(task for task in tasks if task.task_id in found)
  if not selected:
    raise UnknownTaskIds(
      f"pack {pack.pack_id} carries episodes for none of the declared tasks "
      f"{list(known_task_ids(tasks))}. An empty selection produces a scorecard with no "
      "rows, and a scorecard with no rows reads like a clean run rather than like a "
      "pack that was never scored."
    )
  return selected


# -- the built-in demonstration pack -------------------------------------------


def _digest(*parts: str) -> str:
  return hashlib.sha256("|".join(parts).encode("ascii")).hexdigest()


def _jitter(rng: random.Random, nominal: float, spread: float) -> float:
  return nominal * (1.0 + rng.uniform(-spread, spread))


#: Coefficients that are structure rather than magnitude: a boundary, an onset, a
#: band edge, a seating limit. Jittering one of these does not make an episode
#: harder, it makes the task a different task, so they are held at their declared
#: value on every episode. The hidden keys are held too, and for the opposite
#: reason: their value is the whole experiment, and a knowable episode has to carry
#: exactly the nominal or "a model holding the nominal is wrong on this episode"
#: stops being checkable.
_FIXED_KEYS: Tuple[str, ...] = (
  "drift_onset",
  "boundary_mm",
  "band_lo",
  "band_hi",
  "seat_mm",
  "seat_deg",
  "press_lo",
  "press_hi",
  "max_lateral_mm",
)


def _draw_coefficients(
  task: BenchTask,
  rng: random.Random,
  unknowable: bool,
  spread: float,
) -> Tuple[Dict[str, float], Dict[str, float], Tuple[str, ...]]:
  """Per-episode coefficients, plus the invisible latent when there is one.

  A knowable episode holds every hidden key at its published nominal. An unknowable
  one redraws each of them from :data:`HIDDEN_SUPPORT`, which never contains the
  nominal, and that is the only difference between the two: same actions, same
  initial state, same jitter on everything else, same frames. Nothing in an
  unknowable episode is separable from a knowable one except the thing no observer
  can see.

  The draw is not the guarantee. The guarantee is
  :func:`require_unknowable_separation`, which advances the episode under every
  value in the grid at every horizon it will be scored at and fails the build if any
  pair collapses. An earlier version of this function claimed in its docstring that
  the grid was "disjoint in effect from the nominal by more than the task
  tolerance", nothing checked it, and on the pack the CLI actually scored the claim
  was false on 80 of 80 S1 horizon-1 episodes.
  """
  base = dict(NOMINAL[task.task_id])
  hidden: Dict[str, float] = {}
  keys: Tuple[str, ...] = ()
  support = HIDDEN_SUPPORT[task.task_id]
  for key, value in list(base.items()):
    if key in _FIXED_KEYS or key in support:
      continue
    base[key] = _jitter(rng, value, spread)
  if not unknowable:
    return base, hidden, keys

  keys = tuple(sorted(support))
  for key in keys:
    drawn = float(rng.choice(support[key]))
    base[key] = drawn
    hidden[key] = drawn
  return base, hidden, keys


def _draw_actions(
  task: BenchTask,
  rng: random.Random,
) -> Tuple[Tuple[float, ...], ...]:
  rows: List[Tuple[float, ...]] = []
  for _ in range(task.max_horizon):
    if task.task_id == "S1":
      row = (
        rng.gauss(0.0, 8.0),
        rng.gauss(0.0, 8.0),
        rng.gauss(0.0, 6.0),
        rng.gauss(0.0, 0.03),
        GRIPPER_UNCHANGED,
      )
    elif task.task_id == "S2":
      verb = rng.randrange(3)
      row = (
        rng.uniform(0.0, 5.0) if verb == 1 else 0.0,
        rng.uniform(-1.0, 1.0),
        rng.uniform(0.0, 3.0) if verb == 2 else rng.uniform(0.0, 1.0),
        float(verb),
        GRIPPER_UNCHANGED,
      )
    elif task.task_id == "S3":
      verb = rng.randrange(3)
      magnitude = rng.uniform(0.2, 2.5)
      row = (
        magnitude if verb == 1 else (-magnitude if verb == 2 else 0.0),
        rng.uniform(-0.5, 0.5),
        rng.uniform(0.0, 4.0),
        float(verb),
        GRIPPER_UNCHANGED,
      )
    elif task.task_id == "S4":
      reseat = 1.0 if rng.random() < 0.2 else 0.0
      row = (
        rng.uniform(-1.5, 1.5),
        rng.uniform(-1.5, 1.5),
        rng.uniform(0.4, 1.6),
        rng.uniform(-1.0, 1.0),
        reseat,
      )
    else:
      release = 1.0 if rng.random() < 0.1 else 0.0
      row = (0.0, 0.0, rng.uniform(-2.0, 2.5), 0.0, release)
    rows.append(tuple(float(v) for v in row))
  return tuple(rows)


def _draw_state0(task: BenchTask, rng: random.Random) -> Tuple[float, ...]:
  if task.task_id == "S1":
    return (rng.uniform(-40.0, 40.0), rng.uniform(-40.0, 40.0), rng.uniform(20.0, 90.0), 0.0)
  if task.task_id == "S2":
    # Bounded so that the largest retained film in HIDDEN_SUPPORT["S2"] cannot push
    # the terminal past the physically reachable ceiling: the drawdown never
    # increases the residual, so the terminal is at most this bound plus 1.75 uL,
    # inside S2_RESIDUAL_RANGE. The transition deliberately does not clamp: a clamped
    # truth stops responding to the action once it saturates, and a saturated truth
    # is indistinguishable from a model that ignores its actions.
    return (rng.uniform(1.8, 3.0),)
  if task.task_id == "S3":
    return (rng.uniform(S3_TARGET_UL - 2.5, S3_TARGET_UL + 2.5),)
  if task.task_id == "S4":
    return (rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), rng.uniform(-2.0, 2.0), 0.0)
  return (rng.uniform(0.0, 3.5), 0.0)


def builtin_demo_pack(
  tasks: Sequence[BenchTask] = TASKS,
  n_episodes: Optional[int] = None,
  seed: int = 20260803,
  seed_epoch: str = "demo-2026-08-03",
  split: str = "dev",
  unknowable_fraction: float = 0.20,
  coefficient_spread: float = 0.12,
) -> ScenePack:
  """A self-contained pack so the harness can be exercised with nothing installed.

  This is **not** the scored fixture. The scored fixtures come from the vision
  side's exporter, and this pack says so in its ``source`` field, which is printed
  on every row built from it. It exists because a harness that cannot run until an
  unrelated repository ships is a harness whose refusals have never been executed,
  and every refusal in this build is supposed to be a code path with a test rather
  than a paragraph.

  The unknowable episodes are placed by a seeded shuffle across the whole task, not
  in a block, and their identifiers are drawn from the same sequence as everything
  else. :meth:`ScenePack.check_interleaved` is run before the pack is returned, so
  a generator change that accidentally clustered them fails here rather than at the
  point somebody reads a refusal recall of 1.0 and believes it.

  :func:`require_unknowable_separation` is run on the same line of reasoning and is
  the stronger of the two: interleaving asks whether the unknowable set can be found
  without a model, and separation asks whether there was anything to find. It runs
  on both splits, over the coefficients as drawn, because a scoring pack does not
  publish them and a check that only runs when the numbers happen to be published is
  a check that never runs on the pack that gets scored.
  """
  if split not in SPLITS:
    raise PackContentError(f"split must be one of {list(SPLITS)}, got {split!r}")
  low, high = TASKS[0].unknowable_band
  if not low <= float(unknowable_fraction) <= high:
    raise PackContentError(
      f"unknowable_fraction {unknowable_fraction} is outside [{low:.2f}, {high:.2f}], "
      "and a pack built outside the band produces abstention numbers this harness "
      "will refuse to print anyway (R6)."
    )
  rng = random.Random(seed)
  episodes: List[SceneEpisode] = []
  drawn: Dict[str, Dict[str, float]] = {}
  # Drawn per episode on both splits. The split decides whether the draws are
  # published, not whether they happen: a dev pack whose coefficients were all the
  # published nominal would make the analytic floor an exact oracle on it, and a
  # floor that is exact on the development pack teaches a submitter nothing except
  # that the pack is not the task.
  spread = float(coefficient_spread)
  for task in tasks:
    count = int(task.n_episodes if n_episodes is None else n_episodes)
    if count < 1:
      raise PackContentError("n_episodes must be at least 1")
    n_unknowable = max(1, int(round(count * float(unknowable_fraction))))
    flags = [True] * n_unknowable + [False] * (count - n_unknowable)
    rng.shuffle(flags)
    for index, unknowable in enumerate(flags):
      episode_id = f"{task.task_id}-e{index:04d}"
      coefficients, hidden, keys = _draw_coefficients(task, rng, unknowable, spread)
      actions = _draw_actions(task, rng)
      state0 = _draw_state0(task, rng)
      drawn[episode_id] = coefficients
      # Full occlusion on S4: the target well ROI is covered in every frame, so the
      # nest offset the truth carries never reaches the observation. The reported
      # initial offset is the same draw every other episode gets and the hidden one
      # is added to the terminal, which is a coefficient like every other hidden
      # latent here rather than a swapped initial condition. Reporting zeros for the
      # observed offset instead would be a tell: an exactly-zero offset occurs on no
      # other episode, so "if the offset is exactly zero, abstain" would collect the
      # whole occluded set without predicting anything.
      terminals = {
        k: advance(task.task_id, state0, actions[:k], coefficients)
        for k in task.horizons
      }
      frames = tuple(
        SceneFrame(
          path=f"frames/{task.task_id}/{episode_id}-{step:02d}.png",
          sha256=_digest(seed_epoch, task.renderer, episode_id, str(step)),
          t_capture_s=0.10 * step,
          clock_source="rendered",
        )
        for step in range(2)
      )
      episodes.append(
        SceneEpisode(
          episode_id=episode_id,
          task_id=task.task_id,
          state0=state0,
          actions=actions,
          terminal_by_horizon=terminals,
          knowability=(
            (FULL_OCCLUSION if task.unknowable_construction == FULL_OCCLUSION
             else INDEPENDENT_LATENT)
            if unknowable
            else KNOWABLE
          ),
          frames=frames,
          coefficients=coefficients if split == "dev" else None,
          hidden=hidden or None,
          hidden_keys=keys,
        )
      )
  pack = ScenePack(
    pack_id=f"builtin-demo-{seed}",
    seed_epoch=seed_epoch,
    split=split,
    source=(
      "plr_lr.learn.bench.packio.builtin_demo_pack -- a demonstration fixture, not "
      "the scored one. The scored packs are exported by the vision side at "
      f"scenes format {SCENES_FORMAT_VERSION}."
    ),
    episodes=tuple(episodes),
    created_utc="",
  )
  pack.check_interleaved()
  reports = require_unknowable_separation(pack, drawn)
  return dataclasses.replace(
    pack,
    separation=SeparationCertificate(
      status=SEPARATION_CHECKED,
      n_unknowable=sum(1 for episode in pack.episodes if episode.unknowable),
      n_reports=len(reports),
      reports=reports,
    ),
  )


def residual_in_range(pack: ScenePack) -> bool:
  """Did S2's planted truth stay inside its physically reachable range?

  Checked rather than clamped. Clamping the truth to ``[0, 5]`` would make the
  terminal state stop responding to the action once it saturated, and a saturated
  truth is one the interventional gate cannot distinguish from a model that ignores
  its actions. Drawing initial conditions that keep the trajectory inside the range
  is the honest version, and this is the assertion that says it worked.
  """
  low, high = S2_RESIDUAL_RANGE
  for episode in pack.for_task("S2"):
    for terminal in episode.terminal_by_horizon.values():
      if not low <= terminal[0] <= high:
        return False
  return True
