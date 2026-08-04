"""Tests for plr_lr.learn.bench.packio, the file format that is the whole interface.

The load-bearing claim in this module is an absence: no module under
``plr_lr.learn.bench`` imports the vision package, at any depth, and nothing in it
appears in ``sys.modules`` after the harness has been imported and a pack has been
built. That is asserted rather than reviewed, because the coupling this prevents is
the kind that arrives one convenient shared constant at a time.

Everything else here is a refusal that would otherwise have produced a score:
a manifest read at the wrong format version, two episodes sharing an identifier,
two episodes sharing their frames, an unknowable set that sits in one block, and a
submission whose frame hashes are not the fixture's.

The heaviest of them is :class:`UnknowableSeparationError`. Interleaving, fractions
and identifier tokens were all checked before it existed and none of them asked
whether the hidden latent decided anything, so a pack shipped in which 80 of 80 S1
horizon-1 episodes labelled unknowable had a terminal state that was identical for
every value of that latent. The tests below assert the gate fires on a seeded bad
construction in each of its three shapes, that it reads only the conclusive
direction, and that on the built-in pack every unknowable episode is decided by its
hidden latent at every horizon it is scored at.
"""

from __future__ import annotations

import ast
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

from plr_lr.learn.bench import packio
from plr_lr.learn.bench.packio import (
  BASIS_HARDWARE,
  BASIS_SIMULATED,
  CLOCK_SOURCES,
  HIDDEN_SUPPORT,
  MIN_BRANCH_SEPARATION,
  MIN_NOMINAL_SEPARATION,
  NOMINAL,
  SCENES_FORMAT_VERSION,
  FixtureDuplication,
  FixtureMismatch,
  PackContentError,
  PackFormatError,
  PackVersionMismatch,
  SceneEpisode,
  SceneFrame,
  ScenePack,
  SeparationReport,
  UnknowableSeparationError,
  advance,
  builtin_demo_pack,
  check_unknowable_separation,
  hidden_grid,
  load_manifest,
  require_fixture_match,
  require_unknowable_separation,
  residual_in_range,
  write_manifest,
)
from plr_lr.learn.bench.tasks import (
  FULL_OCCLUSION,
  INDEPENDENT_LATENT,
  KNOWABLE,
  TASKS,
  TASKS_BY_ID,
  InterleavingRefused,
  check_interleaving,
)
from plr_lr.learn.predict import CLOCK_SOURCES as PREDICT_CLOCK_SOURCES
from plr_lr.learn.selective import AbstentionBandRefused, AbstentionCredit

BENCH_DIR = Path(packio.__file__).parent

#: Kept small so the whole module runs in well under a second. Above the thirty-
#: episode reporting floor so the rate-carrying assertions are about a rate rather
#: than about the floor.
DEMO_N = 60


def _episode(
  episode_id: str,
  knowability: str = KNOWABLE,
  frame_hash: str = "a",
) -> SceneEpisode:
  frame_hash = frame_hash.ljust(64, "0")
  return SceneEpisode(
    episode_id=episode_id,
    task_id="S2",
    state0=(2.0,),
    actions=((0.0, 0.0, 0.0, 0.0, -1.0),),
    terminal_by_horizon={1: (2.0,)},
    knowability=knowability,
    frames=(SceneFrame(path=f"{episode_id}.png", sha256=frame_hash, t_capture_s=0.0),),
  )


def _pack(episodes: List[SceneEpisode]) -> ScenePack:
  return ScenePack(
    pack_id="test-pack",
    seed_epoch="test-epoch",
    split="dev",
    source="tests",
    episodes=tuple(episodes),
  )


S1_TASK = TASKS_BY_ID["S1"]


def _s1_episode(
  episode_id: str = "S1-e0000",
  hidden: Optional[Dict[str, float]] = None,
  hidden_keys: Tuple[str, ...] = ("bias_mm",),
  knowability: str = INDEPENDENT_LATENT,
  publish_coefficients: bool = True,
  horizons: Optional[Tuple[int, ...]] = None,
) -> SceneEpisode:
  """One hand-made S1 episode, built the way the generator builds them.

  The truth is produced by the same :func:`advance` the generator calls, so an
  episode assembled here is separated or collapsed for the same reason a generated
  one would be, rather than because the fixture wrote its own arithmetic.
  """
  actions = tuple((1.0, 0.5, 0.5, 0.0, -1.0) for _ in range(S1_TASK.max_horizon))
  coefficients = dict(NOMINAL["S1"])
  coefficients.update(hidden or {})
  state0 = (1.0, 2.0, 30.0, 0.0)
  wanted = S1_TASK.horizons if horizons is None else horizons
  terminals = {k: advance("S1", state0, actions[:k], coefficients) for k in wanted}
  return SceneEpisode(
    episode_id=episode_id,
    task_id="S1",
    state0=state0,
    actions=actions,
    terminal_by_horizon=terminals,
    knowability=knowability,
    frames=(
      SceneFrame(
        path=f"{episode_id}.png",
        sha256=episode_id.ljust(64, "0"),
        t_capture_s=0.0,
      ),
    ),
    coefficients=coefficients if publish_coefficients else None,
    hidden=hidden,
    hidden_keys=hidden_keys,
  )


class NoVisionImportTest(unittest.TestCase):
  """The interface is a file. It is not a package import, in either direction."""

  def test_no_bench_module_imports_the_vision_package(self) -> None:
    offenders = []
    for path in sorted(BENCH_DIR.glob("*.py")):
      tree = ast.parse(path.read_text(encoding="ascii"))
      for node in ast.walk(tree):
        if isinstance(node, ast.Import):
          for alias in node.names:
            if alias.name.split(".")[0] == "labcv":
              offenders.append(f"{path.name}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
          module = node.module or ""
          if module.split(".")[0] == "labcv":
            offenders.append(f"{path.name}: from {module}")
    self.assertEqual(
      offenders,
      [],
      "the entire interface to the vision side is the versioned scene-pack file "
      "format; a package import would make it two interfaces, and the second one is "
      "the one that quietly grows a shared constant",
    )

  def test_building_a_pack_loads_no_vision_module(self) -> None:
    builtin_demo_pack(n_episodes=DEMO_N)
    self.assertNotIn("labcv", sys.modules)

  def test_no_bench_module_mentions_a_hardware_permission_token(self) -> None:
    """The benchmark has no hardware path, so it may not carry the word that opens one."""
    for path in sorted(BENCH_DIR.glob("*.py")):
      source = path.read_text(encoding="ascii")
      self.assertNotIn("allow_hardware", source, path.name)


class FormatVersionTest(unittest.TestCase):
  def test_the_version_is_declared_and_checked(self) -> None:
    self.assertEqual(SCENES_FORMAT_VERSION, 1)

  def test_a_future_version_is_refused_rather_than_read_best_effort(self) -> None:
    raw = _pack([_episode("a")]).to_dict()
    raw["format_version"] = SCENES_FORMAT_VERSION + 1
    with self.assertRaises(PackVersionMismatch):
      ScenePack.from_dict(raw)

  def test_a_manifest_with_no_version_is_refused(self) -> None:
    raw = _pack([_episode("a")]).to_dict()
    del raw["format_version"]
    with self.assertRaises(PackVersionMismatch):
      ScenePack.from_dict(raw)

  def test_a_missing_field_names_itself(self) -> None:
    raw = _pack([_episode("a")]).to_dict()
    del raw["seed_epoch"]
    with self.assertRaises(PackFormatError) as caught:
      ScenePack.from_dict(raw)
    self.assertIn("seed_epoch", str(caught.exception))

  def test_the_clock_sources_match_the_adapter_contract(self) -> None:
    """Restated in the format layer to keep it numpy-free, pinned so it cannot drift."""
    self.assertEqual(CLOCK_SOURCES, PREDICT_CLOCK_SOURCES)


class PackRefusalTest(unittest.TestCase):
  def test_duplicate_episode_identifiers_are_refused(self) -> None:
    with self.assertRaises(PackContentError):
      _pack([_episode("a", frame_hash="h1"), _episode("a", frame_hash="h2")])

  def test_duplicate_frames_across_episodes_are_refused(self) -> None:
    with self.assertRaises(FixtureDuplication):
      _pack([_episode("a", frame_hash="same"), _episode("b", frame_hash="same")])

  def test_an_unknown_knowability_is_refused(self) -> None:
    with self.assertRaises(PackContentError):
      SceneEpisode(
        episode_id="a",
        task_id="S2",
        state0=(1.0,),
        actions=((0.0, 0.0, 0.0, 0.0, -1.0),),
        terminal_by_horizon={1: (1.0,)},
        knowability="probably fine",
      )

  def test_an_empty_pack_is_refused(self) -> None:
    with self.assertRaises(PackContentError):
      _pack([])

  def test_an_unnamed_split_is_refused(self) -> None:
    with self.assertRaises(PackContentError):
      ScenePack(
        pack_id="p",
        seed_epoch="e",
        split="whatever",
        source="tests",
        episodes=(_episode("a"),),
      )

  def test_an_unhashed_frame_is_refused(self) -> None:
    with self.assertRaises(PackContentError):
      SceneFrame(path="a.png", sha256="", t_capture_s=0.0)

  def test_an_unnamed_clock_is_refused(self) -> None:
    with self.assertRaises(PackContentError):
      SceneFrame(path="a.png", sha256="sha-abc123", t_capture_s=0.0, clock_source="wall")

  def test_a_horizon_with_no_truth_is_refused_rather_than_interpolated(self) -> None:
    episode = _episode("a")
    with self.assertRaises(PackContentError):
      episode.terminal(2)


class BasisTest(unittest.TestCase):
  def test_a_pack_with_no_device_serial_reads_simulated(self) -> None:
    pack = _pack([_episode("a")])
    self.assertEqual(pack.basis, BASIS_SIMULATED)
    self.assertTrue(pack.is_simulation)

  def test_only_a_device_serial_hash_changes_the_basis(self) -> None:
    pack = ScenePack(
      pack_id="p",
      seed_epoch="e",
      split="dev",
      source="tests",
      episodes=(_episode("a"),),
      device_serial_sha256="deadbeef",
    )
    self.assertEqual(pack.basis, BASIS_HARDWARE)
    self.assertFalse(pack.is_simulation)


class RoundTripTest(unittest.TestCase):
  def test_a_pack_survives_a_write_and_a_read(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N)
    with tempfile.TemporaryDirectory() as directory:
      path = str(Path(directory) / "manifest.json")
      write_manifest(pack, path)
      reloaded = load_manifest(path)
    self.assertEqual(reloaded.pack_id, pack.pack_id)
    self.assertEqual(len(reloaded.episodes), len(pack.episodes))
    self.assertEqual(reloaded.frames_sha256(), pack.frames_sha256())
    self.assertEqual(reloaded.episodes[0].terminal(1), pack.episodes[0].terminal(1))

  def test_the_manifest_is_ascii_json(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N)
    text = json.dumps(pack.to_dict())
    text.encode("ascii")

  def test_an_unreadable_path_is_refused(self) -> None:
    with self.assertRaises(PackFormatError):
      load_manifest("/nonexistent/manifest.json")


class FixtureMatchTest(unittest.TestCase):
  def test_matching_hashes_pass(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N)
    require_fixture_match(pack, pack.frames_sha256())

  def test_a_mismatch_prints_the_token_and_scores_nothing(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N)
    wrong = list(pack.frames_sha256())
    wrong[0] = "0" * 64
    with self.assertRaises(FixtureMismatch) as caught:
      require_fixture_match(pack, wrong)
    self.assertIn("FIXTURE-MISMATCH", str(caught.exception))


class DemoPackTest(unittest.TestCase):
  def setUp(self) -> None:
    self.pack = builtin_demo_pack(n_episodes=DEMO_N)

  def test_the_source_says_it_is_not_the_scored_fixture(self) -> None:
    self.assertIn("demonstration fixture", self.pack.source)

  def test_every_task_is_present_with_the_requested_count(self) -> None:
    for task in TASKS:
      self.assertEqual(len(self.pack.for_task(task.task_id)), DEMO_N)

  def test_every_task_interleaves_its_unknowable_episodes(self) -> None:
    for report in self.pack.check_interleaved():
      self.assertTrue(report.ok, report.format())

  def test_no_episode_identifier_names_its_own_knowability(self) -> None:
    for episode in self.pack.episodes:
      self.assertNotIn("unknow", episode.episode_id.lower())

  def test_the_occlusion_task_reports_an_offset_drawn_like_every_other(self) -> None:
    """A constant reported offset on the occluded episodes would be a free tell.

    An exactly-zero offset occurs on no knowable episode, so reporting zeros there
    would let ``if the offset is exactly zero, abstain`` collect the whole occluded
    set without predicting anything.
    """
    occluded = [e for e in self.pack.for_task("S4") if e.knowability == FULL_OCCLUSION]
    self.assertTrue(occluded)
    for episode in occluded:
      self.assertNotEqual((episode.state0[0], episode.state0[1]), (0.0, 0.0))

  def test_the_invisible_latent_is_recorded_and_lives_only_on_the_episode(self) -> None:
    unknowable = [e for e in self.pack.episodes if e.unknowable]
    self.assertTrue(unknowable)
    for episode in unknowable:
      self.assertTrue(episode.hidden)
      self.assertTrue(episode.hidden_keys)

  def test_knowable_episodes_carry_no_invisible_latent(self) -> None:
    for episode in self.pack.episodes:
      if episode.knowability == KNOWABLE:
        self.assertIn(episode.hidden, (None, {}))

  def test_the_residual_task_stays_inside_its_physical_range(self) -> None:
    self.assertTrue(residual_in_range(self.pack))

  def test_the_unknowable_construction_matches_the_task_declaration(self) -> None:
    for task in TASKS:
      wanted = task.unknowable_construction
      for episode in self.pack.for_task(task.task_id):
        if episode.unknowable:
          self.assertEqual(episode.knowability, wanted, task.task_id)

  def test_an_out_of_band_fraction_is_refused_at_build_time(self) -> None:
    with self.assertRaises(PackContentError):
      builtin_demo_pack(n_episodes=DEMO_N, unknowable_fraction=0.90)


class HiddenGridDeclarationTest(unittest.TestCase):
  """The task declares the ceiling; the pack declares the values that buy it."""

  def test_the_grid_matches_what_each_task_declares(self) -> None:
    for task in TASKS:
      support = HIDDEN_SUPPORT[task.task_id]
      self.assertEqual(tuple(sorted(support)), tuple(sorted(task.hidden_latent)))
      self.assertEqual(len(hidden_grid(task.task_id)), task.hidden_branches, task.task_id)
      self.assertAlmostEqual(task.chance_ceiling, 1.0 / task.hidden_branches)

  def test_no_support_contains_the_nominal(self) -> None:
    """A branch that is the nominal is answered by a model that ignores the latent."""
    for task in TASKS:
      for key, values in HIDDEN_SUPPORT[task.task_id].items():
        self.assertNotIn(float(NOMINAL[task.task_id][key]), [float(v) for v in values])

  def test_a_knowable_episode_holds_every_hidden_key_at_the_nominal(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N)
    for episode in pack.episodes:
      if episode.unknowable:
        continue
      for key in HIDDEN_SUPPORT[episode.task_id]:
        self.assertEqual(
          episode.coefficients[key],
          NOMINAL[episode.task_id][key],
          f"{episode.episode_id} jittered its hidden {key}, which makes 'a model "
          "holding the nominal is wrong on the unknowable episodes' uncheckable",
        )

  def test_the_two_thresholds_are_absolute_constants(self) -> None:
    """Neither is derived from the pack it judges, which is the R9 failure mode."""
    self.assertEqual(MIN_BRANCH_SEPARATION, 2.0)
    self.assertEqual(MIN_NOMINAL_SEPARATION, 1.0)

  def test_an_unknown_task_has_no_grid_to_invent(self) -> None:
    with self.assertRaises(PackContentError):
      hidden_grid("S99")


class UnknowableSeparationTest(unittest.TestCase):
  """Does the hidden latent decide the terminal state, at every scored horizon?

  The question nothing asked before this gate existed. Interleaving, the fraction
  band and the identifier tokens were all green on a pack where 80 of 80 S1
  horizon-1 unknowable episodes had one possible answer.
  """

  def setUp(self) -> None:
    self.pack = builtin_demo_pack(n_episodes=DEMO_N)

  def test_every_unknowable_episode_is_decided_by_its_latent_at_every_horizon(self) -> None:
    reports = check_unknowable_separation(self.pack)
    self.assertTrue(reports)
    seen = set()
    for report in reports:
      self.assertTrue(report.decides, report.format())
      self.assertTrue(report.ok, report.format())
      seen.add((report.task_id, report.horizon))
    for task in TASKS:
      for horizon in task.horizons:
        self.assertIn((task.task_id, horizon), seen)
    expected = sum(
      len(TASKS_BY_ID[e.task_id].horizons) for e in self.pack.episodes if e.unknowable
    )
    self.assertEqual(len(reports), expected)

  def test_only_episodes_already_labelled_unknowable_are_read(self) -> None:
    reports = check_unknowable_separation(self.pack)
    labelled = {e.episode_id for e in self.pack.episodes if e.unknowable}
    self.assertEqual({r.episode_id for r in reports}, labelled)

  def test_a_knowable_episode_is_never_promoted_by_a_separation(self) -> None:
    """The asymmetry, executed: only the conclusive direction is read.

    A collapse falsifies an unknowable label. A separation establishes nothing,
    because whether the frames carry the latent is a question about pixels and is
    not asked here. An episode labelled knowable whose hidden grid would separate
    stays knowable and produces no report at all.
    """
    knowable = _s1_episode(
      "S1-e0000", hidden={"bias_mm": 7.0}, knowability=KNOWABLE
    )
    unknowable = _s1_episode("S1-e0001", hidden={"bias_mm": 2.5})
    pack = _pack([knowable, unknowable])
    reports = require_unknowable_separation(pack)
    self.assertEqual(sorted({r.episode_id for r in reports}), ["S1-e0001"])
    self.assertEqual(pack.episodes[0].knowability, KNOWABLE)
    self.assertFalse(pack.episodes[0].unknowable)

  def test_a_latent_that_changes_nothing_fails_the_build(self) -> None:
    """The measured defect, seeded back in: a latent whose branches collapse."""
    with mock.patch.dict(
      packio.HIDDEN_SUPPORT, {"S1": {"bias_mm": (-0.05, 0.05)}}, clear=False
    ):
      with self.assertRaises(UnknowableSeparationError) as caught:
        builtin_demo_pack(tasks=(S1_TASK,), n_episodes=DEMO_N)
    message = str(caught.exception)
    self.assertIn("No manifest is emitted", message)
    self.assertIn("the label is false", message)

  def test_a_branch_that_is_the_nominal_fails_the_build(self) -> None:
    with mock.patch.dict(
      packio.HIDDEN_SUPPORT, {"S1": {"bias_mm": (0.0, 7.0)}}, clear=False
    ):
      with self.assertRaises(UnknowableSeparationError) as caught:
        builtin_demo_pack(tasks=(S1_TASK,), n_episodes=DEMO_N)
    self.assertIn("holds the nominal", str(caught.exception))

  def test_a_pair_inside_two_tolerances_fails_even_though_the_latent_decides(self) -> None:
    """More than one tolerance apart is the label's floor; the ceiling needs two.

    Two branches 1.75 tolerances apart are both inside tolerance of one prediction
    placed between them, so the episode has half an answer and the 1/4 ceiling the
    task declares is an overstatement of the difficulty.
    """
    with mock.patch.dict(
      packio.HIDDEN_SUPPORT, {"S1": {"bias_mm": (2.5, 6.0)}}, clear=False
    ):
      with self.assertRaises(UnknowableSeparationError) as caught:
        builtin_demo_pack(tasks=(S1_TASK,), n_episodes=DEMO_N)
    message = str(caught.exception)
    self.assertIn("overstatement", message)
    self.assertNotIn("holds the nominal", message)

  def test_the_report_names_which_of_the_two_claims_failed(self) -> None:
    def report(pair: float, nominal: float) -> SeparationReport:
      return SeparationReport(
        episode_id="e",
        task_id="S1",
        horizon=1,
        n_branches=2,
        min_branch_separation=pair,
        min_nominal_separation=nominal,
      )

    collapsed = report(0.5, 3.0)
    self.assertFalse(collapsed.decides)
    self.assertFalse(collapsed.ok)
    middle = report(1.5, 3.0)
    self.assertTrue(middle.decides)
    self.assertFalse(middle.ok)
    on_nominal = report(3.0, 0.5)
    self.assertTrue(on_nominal.decides)
    self.assertFalse(on_nominal.ok)
    self.assertTrue(report(3.0, 3.0).ok)

  def test_an_undeclared_hidden_key_is_refused(self) -> None:
    pack = _pack([_s1_episode(hidden={"nowhere": 3.0}, hidden_keys=("nowhere",))])
    with self.assertRaises(UnknowableSeparationError) as caught:
      require_unknowable_separation(pack)
    self.assertIn("hidden keys", str(caught.exception))

  def test_a_value_outside_the_declared_support_is_refused(self) -> None:
    pack = _pack([_s1_episode(hidden={"bias_mm": 3.7})])
    with self.assertRaises(UnknowableSeparationError) as caught:
      require_unknowable_separation(pack)
    self.assertIn("declared support", str(caught.exception))

  def test_an_unknowable_episode_with_no_recorded_latent_is_refused(self) -> None:
    pack = _pack([_s1_episode(hidden=None)])
    with self.assertRaises(UnknowableSeparationError):
      require_unknowable_separation(pack)

  def test_an_episode_with_no_truth_at_a_scored_horizon_is_refused(self) -> None:
    pack = _pack([_s1_episode(hidden={"bias_mm": 2.5}, horizons=(3,))])
    with self.assertRaises(UnknowableSeparationError) as caught:
      require_unknowable_separation(pack)
    self.assertIn("no truth at any horizon", str(caught.exception))

  def test_a_scoring_pack_is_gated_at_build_time_and_says_so_afterwards(self) -> None:
    """A pack that does not publish its coefficients cannot be re-checked later.

    The gate runs at build time, where the draws are in hand. Reading the manifest
    afterwards and finding nothing to check is not a pass, so it refuses rather than
    returning an empty verdict that reads like one.
    """
    pack = builtin_demo_pack(n_episodes=DEMO_N, split="scoring")
    self.assertIsNone(pack.episodes[0].coefficients)
    with self.assertRaises(UnknowableSeparationError) as caught:
      check_unknowable_separation(pack)
    self.assertIn("carries no coefficients", str(caught.exception))

  def test_the_gate_runs_inside_the_generator(self) -> None:
    """Not a check a caller has to remember: the build itself refuses."""
    calls: List[int] = []
    original = packio.require_unknowable_separation

    def spy(pack: ScenePack, coefficients: Any = None) -> Any:
      calls.append(len(pack.episodes))
      return original(pack, coefficients)

    with mock.patch.object(packio, "require_unknowable_separation", spy):
      builtin_demo_pack(tasks=(S1_TASK,), n_episodes=DEMO_N)
    self.assertEqual(calls, [DEMO_N])


class NoUnknowableEpisodesTest(unittest.TestCase):
  """A task with nothing unknowable in it reports no abstention numbers at all."""

  def test_a_pack_with_no_unknowable_episodes_is_refused_at_build_time(self) -> None:
    with self.assertRaises(PackContentError):
      builtin_demo_pack(n_episodes=DEMO_N, unknowable_fraction=0.0)

  def test_zero_unknowable_episodes_refuse_the_metrics_rather_than_scoring_zero(self) -> None:
    """R6, at the far end of the band. Not a division by zero and not a silent 0.0.

    ``RefusalRecall`` over an empty unknowable set is 0/0. The permissive readings
    are a crash and a zero, and the zero is worse: it prints as a model that never
    refused when it should have, on a task where there was nothing to refuse.
    """
    credit = AbstentionCredit(
      n=200,
      n_unknowable=0,
      n_abstained=0,
      n_unknowable_abstained=0,
      n_knowable_abstained=0,
      n_credited=160,
    )
    self.assertFalse(credit.in_band)
    with self.assertRaises(AbstentionBandRefused):
      credit.credit_and_false_alarm()
    with self.assertRaises(AbstentionBandRefused):
      credit.recall_and_precision()
    self.assertIn("REFUSED", credit.format())
    self.assertIn("R6", credit.to_dict()["refused"])


class InterleavingTest(unittest.TestCase):
  def test_a_contiguous_block_is_refused(self) -> None:
    ids = [f"S2-e{i:03d}" for i in range(10)]
    labels = [INDEPENDENT_LATENT] * 3 + [KNOWABLE] * 7
    report = check_interleaving("S2", ids, labels)
    self.assertFalse(report.ok)
    self.assertTrue(report.contiguous)

  def test_an_identifier_that_names_the_answer_is_refused(self) -> None:
    ids = ["S2-a", "S2-unknowable-b", "S2-c", "S2-d", "S2-e", "S2-f"]
    labels = [KNOWABLE, INDEPENDENT_LATENT, KNOWABLE, KNOWABLE, KNOWABLE, KNOWABLE]
    report = check_interleaving("S2", ids, labels)
    self.assertFalse(report.ok)
    self.assertIsNotNone(report.id_tell)

  def test_an_all_unknowable_task_is_refused(self) -> None:
    ids = [f"S2-e{i}" for i in range(10)]
    report = check_interleaving("S2", ids, [INDEPENDENT_LATENT] * 10)
    self.assertFalse(report.ok)

  def test_a_task_with_no_unknowable_episodes_is_refused(self) -> None:
    ids = [f"S2-e{i}" for i in range(10)]
    report = check_interleaving("S2", ids, [KNOWABLE] * 10)
    self.assertFalse(report.ok)

  def test_an_unknown_label_raises_rather_than_scoring(self) -> None:
    with self.assertRaises(InterleavingRefused):
      check_interleaving("S2", ["a"], ["maybe"])

  def test_mismatched_columns_raise(self) -> None:
    with self.assertRaises(InterleavingRefused):
      check_interleaving("S2", ["a", "b"], [KNOWABLE])

  def test_an_interleaved_task_in_band_passes(self) -> None:
    ids = [f"S2-e{i:03d}" for i in range(10)]
    labels = [KNOWABLE] * 10
    labels[2] = INDEPENDENT_LATENT
    labels[7] = INDEPENDENT_LATENT
    report = check_interleaving("S2", ids, labels)
    self.assertTrue(report.ok, report.format())


class AdvanceTest(unittest.TestCase):
  def test_there_is_one_transition_per_task_and_it_is_deterministic(self) -> None:
    for task in TASKS:
      state = tuple(0.5 for _ in range(task.state_dim))
      actions = ((0.3, 0.2, 0.4, 0.0, -1.0),)
      first = advance(task.task_id, state, actions, dict(NOMINAL[task.task_id]))
      second = advance(task.task_id, state, actions, dict(NOMINAL[task.task_id]))
      self.assertEqual(first, second)
      self.assertEqual(len(first), task.state_dim)

  def test_an_unknown_task_is_refused(self) -> None:
    with self.assertRaises(PackContentError):
      advance("S99", (0.0,), ((0.0, 0.0, 0.0, 0.0, 0.0),), {})

  def test_the_terminal_state_moves_when_the_actions_move(self) -> None:
    """The property the interventional gate measures, asserted on the truth itself."""
    for task in TASKS:
      state = tuple(1.0 for _ in range(task.state_dim))
      quiet = advance(task.task_id, state, ((0.1, 0.1, 0.1, 0.0, -1.0),), {})
      loud = advance(task.task_id, state, ((3.1, 3.1, 3.1, 0.0, -1.0),), {})
      self.assertNotEqual(quiet, loud, task.task_id)

  def test_a_release_does_not_erase_the_action_dependence(self) -> None:
    """Resetting to a constant would make an episode's terminal ignore its actions."""
    a = advance("S5", (2.0, 0.0), ((0.0, 0.0, 0.5, 0.0, 1.0),), {})
    b = advance("S5", (2.0, 0.0), ((0.0, 0.0, 2.5, 0.0, 1.0),), {})
    self.assertNotEqual(a[0], b[0])

  def test_the_published_nominal_covers_every_task(self) -> None:
    for task in TASKS:
      self.assertIn(task.task_id, NOMINAL)


class SplitPolicyTest(unittest.TestCase):
  def test_a_dev_pack_publishes_its_coefficients(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N, split="dev")
    self.assertIsNotNone(pack.episodes[0].coefficients)

  def test_a_scoring_pack_does_not(self) -> None:
    """Publishing them for a scored task turns the board into a regression contest."""
    pack = builtin_demo_pack(n_episodes=DEMO_N, split="scoring")
    self.assertIsNone(pack.episodes[0].coefficients)

  def test_coefficients_are_drawn_per_episode_on_both_splits(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N, split="dev")
    values = {
      round(e.coefficients["gain_lo"], 9) for e in pack.for_task("S1")
    }
    self.assertGreater(len(values), 1)

  def test_the_dict_form_is_json_safe(self) -> None:
    pack = builtin_demo_pack(n_episodes=DEMO_N)
    raw: Dict[str, Any] = copy.deepcopy(pack.to_dict())
    json.dumps(raw)


if __name__ == "__main__":
  unittest.main()
