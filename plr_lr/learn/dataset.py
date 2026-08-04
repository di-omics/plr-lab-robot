"""An on-disk dataset of recorded episodes, in plain JSON.

The format is deliberately boring: a directory, one JSON file per episode, and a
``meta.json`` that names every field of the state and action vectors. No parquet,
no database, no dependency beyond the standard library, because the point is that
a demonstration recorded on a laptop today can still be read in a year by
something that was not written yet.

    ds = Dataset("demos/plate_transfer")
    ds.add(episode)                 # assigns the next episode index
    ds.export_frames("frames.jsonl")  # project-native JSONL interchange records

``meta.json`` records fields, units, and the project-native frame schema from the
first episode written. If a later observation has a different state layout (a
different arm, or a different joint count) the write is refused rather than
silently appending rows that mean something else.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from .types import ACTION_FIELDS, ACTION_UNITS, Episode

FORMAT_VERSION = 1

FRAME_FIELDS: Tuple[str, ...] = (
  "index",
  "observation.state",
  "action",
  "action.mode",
  "timestamp",
  "frame_index",
  "episode_index",
  "task",
)

FRAME_SCHEMA = {
  "name": "plr-lr-learning-frames",
  "kind": "project-native-interchange",
  "encoding": "jsonl",
  "fields": list(FRAME_FIELDS),
  "action_modes": {
    "delta": {"p0-p2": "translation_delta", "p3": "rotation_delta"},
    "absolute": {"p0-p2": "position", "p3": "rotation"},
  },
}


def _json_safe(value: Any) -> Any:
  """Replace non-finite floats with None so the result is valid JSON."""
  if isinstance(value, float):
    return None if math.isnan(value) or math.isinf(value) else value
  if isinstance(value, dict):
    return {k: _json_safe(v) for k, v in value.items()}
  if isinstance(value, (list, tuple)):
    return [_json_safe(v) for v in value]
  return value


class DatasetLayoutError(ValueError):
  """Raised when an episode's vector layout disagrees with the dataset's."""


class Dataset:
  """A directory of recorded episodes."""

  def __init__(self, root: Union[str, Path]) -> None:
    self.root = Path(root)
    self.episode_dir = self.root / "episodes"

  # -- paths -----------------------------------------------------------------

  @property
  def meta_path(self) -> Path:
    return self.root / "meta.json"

  def _episode_path(self, index: int) -> Path:
    return self.episode_dir / f"episode_{index:06d}.json"

  def _episode_paths(self) -> List[Path]:
    if not self.episode_dir.exists():
      return []

    indexed = []
    for path in self.episode_dir.glob("episode_*.json"):
      suffix = path.stem[len("episode_") :]
      if not suffix.isdigit():
        raise DatasetLayoutError(
          f"cannot read episode index from {path.name!r}; expected episode_000000.json"
        )
      indexed.append((int(suffix), path))
    return [path for _, path in sorted(indexed)]

  # -- meta ------------------------------------------------------------------

  def meta(self) -> Dict[str, Any]:
    if not self.meta_path.exists():
      return {}
    return json.loads(self.meta_path.read_text())

  def _write_meta(self, meta: Dict[str, Any]) -> None:
    self.root.mkdir(parents=True, exist_ok=True)
    self.meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")

  def _state_fields_of(self, episode: Episode) -> Tuple[str, ...]:
    obs = episode.observations()
    if not obs:
      raise DatasetLayoutError("cannot write an episode with no observations")
    fields = obs[0].state_fields()
    for i, observation in enumerate(obs[1:], start=1):
      other = observation.state_fields()
      if other != fields:
        raise DatasetLayoutError(
          f"observation {i} state layout {other} does not match observation 0 layout "
          f"{fields}. One episode cannot change vector meaning between frames."
        )
    return fields

  def _layout_meta(self, episode: Episode, fields: Tuple[str, ...]) -> Dict[str, Any]:
    observation = episode.observations()[0]
    return {
      "format_version": FORMAT_VERSION,
      "schema": FRAME_SCHEMA,
      "state_fields": list(fields),
      "state_units": list(observation.state_units()),
      "action_fields": list(ACTION_FIELDS),
      "action_units": list(ACTION_UNITS),
      "timestamp_unit": "s",
    }

  # -- writing ---------------------------------------------------------------

  def next_index(self) -> int:
    paths = self._episode_paths()
    if not paths:
      return 0
    return max(int(path.stem[len("episode_") :]) for path in paths) + 1

  def add(self, episode: Episode) -> Path:
    """Write ``episode`` into the dataset and return the file it landed in.

    The episode's ``episode_index`` is overwritten with an index greater than
    every one already on disk. Gaps are preserved: reusing a missing index risks
    overwriting a file if the directory and metadata ever disagree.
    """
    fields = self._state_fields_of(episode)
    meta = self.meta()
    expected_layout = self._layout_meta(episode, fields)
    if meta:
      known = tuple(meta.get("state_fields", ()))
      if known and known != fields:
        raise DatasetLayoutError(
          f"episode state layout {fields} does not match dataset layout {known}. "
          "Writing it would mix two different meanings into one 'observation.state' "
          "column. Start a new dataset, or record with the same arm configuration."
        )
      for key, value in expected_layout.items():
        if key in meta and meta[key] != value:
          raise DatasetLayoutError(
            f"dataset metadata {key!r} is {meta[key]!r}, but this writer requires "
            f"{value!r}. Refusing to reinterpret an existing dataset."
          )
        meta.setdefault(key, value)
    else:
      meta = expected_layout
      meta.update({"n_episodes": 0, "tasks": []})

    index = self.next_index()
    episode.episode_index = index
    self.episode_dir.mkdir(parents=True, exist_ok=True)
    path = self._episode_path(index)
    # allow_nan=False so a NaN reaches the caller as an exception rather than as
    # a file that only Python can read back.
    encoded = json.dumps(episode.to_dict(), indent=2, allow_nan=False) + "\n"
    # Exclusive creation keeps the no-overwrite guarantee if another writer
    # creates this index after next_index() inspected the directory.
    with path.open("x") as fh:
      fh.write(encoded)

    meta["n_episodes"] = len(self._episode_paths())
    tasks = list(meta.get("tasks", []))
    if episode.task and episode.task not in tasks:
      tasks.append(episode.task)
    meta["tasks"] = tasks
    self._write_meta(meta)
    return path

  # -- reading ---------------------------------------------------------------

  def __len__(self) -> int:
    return len(self._episode_paths())

  def episodes(self) -> Iterator[Episode]:
    """Yield every episode in index order."""
    for path in self._episode_paths():
      yield Episode.from_dict(json.loads(path.read_text()))

  def load(self, index: int) -> Episode:
    path = self._episode_path(index)
    if not path.exists():
      raise FileNotFoundError(f"no episode {index} in {self.root}")
    return Episode.from_dict(json.loads(path.read_text()))

  def transitions(self) -> List[Tuple[Any, Any, Any]]:
    """Every complete ``(observation, action, next observation)`` in the dataset."""
    out: List[Tuple[Any, Any, Any]] = []
    for ep in self.episodes():
      out.extend(ep.transitions())
    return out

  # -- export ----------------------------------------------------------------

  def frames(self) -> Iterator[Dict[str, Any]]:
    """Flat per-frame records across every episode, with a global ``index``."""
    i = 0
    for ep in self.episodes():
      for frame in ep.frames():
        frame["index"] = i
        i += 1
        yield frame

  def export_frames(self, path: Optional[Union[str, Path]] = None) -> Path:
    """Write project-native interchange frames as JSON Lines.

    These records are not a native LeRobot or other external-library export.
    Their explicit ``action.mode`` field is part of the schema and must be
    preserved by any adapter before the action vector is interpreted.

    Unmeasured values are written as ``null``, not as ``NaN``. Python's json
    module emits a bare ``NaN`` token by default and reads it back happily,
    which makes a Python round trip look fine while the file is not JSON at all:
    ``JSON.parse``, ``serde_json`` and ``encoding/json`` all reject the line.
    That would break exactly the non-Python loaders this export exists to feed.
    """
    out = Path(path) if path is not None else self.root / "frames.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
      for frame in self.frames():
        fh.write(json.dumps(_json_safe(frame), allow_nan=False) + "\n")
    return out


def episodes_from(source: Union[Dataset, Sequence[Episode]]) -> List[Episode]:
  """Accept either a ``Dataset`` or a plain list of episodes."""
  if isinstance(source, Dataset):
    return list(source.episodes())
  return list(source)
