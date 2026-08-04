"""Phase 4: record what the arm actually did, fit it, plan against the fit, then refuse.

The arm below reaches 85 percent of every commanded move and adds a small fixed
offset. That stands in for a real arm that is slightly out of calibration:
nothing is broken, every move just lands a little short, always in the same way.
It is the kind of error that survives a visual check and then fails a tight
tolerance.

Four moves close the loop on it: record, fit, plan and score, refuse. Scoring is
what keeps the first three honest, so it runs null controls that must fail and
judges success only on where the arm finished. Refusing is what stands between
all of it and a real arm.

Everything here runs on a CPU with no hardware, no weights, no network, and
nothing installed beyond pylabrobot and numpy.

What this file does **not** show, written down rather than left to be inferred,
because an example that overstates its own reach is the expensive kind:

* No vendor driver is exercised, and nothing below has ever moved a physical
  arm. This repository publishes no hardware backend, so every pose printed here
  came out of ``SimulationArmBackend`` with a defect planted on top of it.
* The capture path is nonetheless the one a driver would use.
  :func:`plr_lr.learn.record.observe` reads an ``ExperimentalSCARA`` through the
  public arm API and never names a backend type, so the recorded episode format
  is a property of the front end rather than of the simulator. That is an
  argument from the code, not a measurement, and it is worth exactly as much as
  one. The refusal in section 4 is the part that is checked mechanically: it is
  run against a real PyLabRobot hardware driver class.
* The freedrive capture injects the poses a hand would have dragged the tool
  through, because a simulated arm in freedrive does not move on its own. The
  session labels the result ``pose_source=injected`` and ``source=synthetic`` for
  exactly that reason, and both labels are printed below.
* The prediction screened in section 4 is hand-built and its calibration points
  are hand-written. It exercises the gate. It does not evaluate a world model,
  and no number it prints is a measurement of one.

    python examples/05_learn_from_demonstration.py
"""

from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
from pylabrobot.arms import ExperimentalSCARA, PreciseFlex400Backend

from plr_lr import Workspace
from plr_lr.learn import (
  Dataset,
  DynamicsModel,
  Goal,
  HardwarePermissionRequired,
  KinematicDynamics,
  KinestheticSession,
  MistrackingBackend,
  NoOpPolicy,
  ReachTask,
  ResidualDynamics,
  SafetyEnvelope,
  SamplingPlanner,
  evaluate,
  random_walk_actions,
  rollout,
  simulate_episode,
)
from plr_lr.learn.predict import FrameRef, RolloutPrediction, RolloutRequest
from plr_lr.learn.readout import ReadoutResult
from plr_lr.learn.screen import CalibrationPoint, Screen

GAIN = 0.85
BIAS = (0.6, -0.4, 0.3, 0.0)
START = (0.0, 0.0, 100.0, 0.0)
STEP_MM = 25.0
N_TRIALS = 5
#: Displaces each trial's start pose. Both the task and the planner are seeded,
#: so without this every trial replays one identical run and a success rate over
#: five of them is a single measurement printed five times.
JITTER_MM = 25.0

#: Reserved for documentation by RFC 5737 and routed nowhere. The driver in
#: section 4 is constructed and never connected, and the address is picked so
#: that a copy of this file which somehow did try to connect still cannot reach
#: anybody's instrument.
UNREACHABLE_HOST = "192.0.2.10"


def mistracked() -> MistrackingBackend:
  return MistrackingBackend(gain=GAIN, bias=BIAS, workspace=Workspace())


def drag_path(i: int, t: float) -> Optional[Tuple[float, float, float, float]]:
  """Poses a hand would have dragged the tool through, for the freedrive capture.

  The simulation backend reports the pose it was last commanded, so an arm in
  freedrive here never moves on its own and a capture without this would record
  sixteen copies of one pose. Injecting the path is how the capture code gets
  exercised with no hardware, and it is why the resulting episode is synthetic
  rather than demonstrated.
  """
  if i >= 16:
    return None
  return (9.0 * i, 2.0 * i, 100.0 - 3.0 * i, 0.0)


def probe(model: DynamicsModel, pose: Sequence[float]) -> Tuple[np.ndarray, np.ndarray]:
  """Read a model's tracking gain and offset back out by asking it two questions.

  A fit is only checked once its numbers are held against the ones that were
  planted. Commanding nothing exposes the offset; commanding a known step on one
  axis exposes that axis's gain.
  """
  p = np.asarray(pose, float)
  rest = np.asarray(model.predict(p, np.zeros(4)), float)
  gains = []
  for axis in range(3):
    delta = np.zeros(4)
    delta[axis] = 10.0
    moved = np.asarray(model.predict(p, delta), float)
    gains.append((moved[axis] - rest[axis]) / 10.0)
  return np.array(gains), (rest - p)[:3]


def indent(text: str) -> str:
  return "  " + text.replace("\n", "\n  ")


def demo_screen() -> Screen:
  """A calibrated screen built from hand-written confidences.

  The numbers are invented and are labelled as such on every line this screen
  goes on to print, because ``calibrated_on`` is reprinted verbatim on every
  verdict. Nothing in this repository produces model confidences yet, and the
  alternative to saying so is a threshold that reads like a fitted one.
  """
  successes = tuple(
    CalibrationPoint(episode_id="cal-%02d" % i, confidence=0.80 + 0.015 * i) for i in range(12)
  )
  return Screen.calibrate(
    successes,
    source="HAND-WRITTEN example confidences, not measured",
    trust_horizon=2,
    max_frame_age_s=0.25,
    name="reach screen",
  )


def screened_request(episode_id: str, t_capture_s: float) -> RolloutRequest:
  """One two-step request against the same 0.5 mm reach the planner was scored on."""
  return RolloutRequest(
    task_id="reach-0.5mm",
    episode_id=episode_id,
    state=np.array(START, float),
    state_fields=("x", "y", "z", "rz"),
    state_units=("mm", "mm", "mm", "rad"),
    frames=(
      FrameRef(
        path="frames/%s.png" % episode_id,
        sha256="0" * 64,
        t_capture_s=t_capture_s,
        clock_source="device_30000",
      ),
    ),
    actions=np.zeros((2, 5)),
    horizon=2,
    tolerance=np.array([0.5, 0.5, 0.5, 0.02]),
  )


def prediction(confidence: float) -> RolloutPrediction:
  return RolloutPrediction(
    state=np.zeros((2, 4)),
    confidence=confidence,
    abstain=False,
    flag=False,
    readout="aruco",
    n_forward_passes=1,
    wall_clock_s=0.012,
  )


async def main() -> None:
  root = Path(tempfile.mkdtemp(prefix="plr_lr_demos_"))
  try:
    truth = mistracked().true_dynamics()
    print(f"planted defect: gain {GAIN}, offset {list(BIAS[:3])} mm per move")

    # --- 1. record ------------------------------------------------------------
    explore = Dataset(root / "exploration")
    for seed in range(4):
      explore.add(
        simulate_episode(
          truth, random_walk_actions(40, step_mm=STEP_MM, seed=seed),
          start=START, task="explore the workspace",
        )
      )
    print(f"\nexploration: {len(explore)} episodes, {len(explore.transitions())} transitions")

    arm = ExperimentalSCARA(backend=mistracked())
    await arm.setup()
    await arm.home()
    session = KinestheticSession(arm, task="drag to the reader", hz=100.0, pose_source=drag_path)
    demo = await session.capture(duration_s=5.0)
    # A separate dataset, not a stylistic choice: the synthetic episodes carry no
    # joint angles and the captured one does, so the two state layouts differ and
    # Dataset refuses to mix them into one column.
    Dataset(root / "kinesthetic").add(demo)
    print(f"kinesthetic: {demo.metadata['n_samples']} freedrive samples at "
          f"{demo.metadata['hz']:.0f} Hz -> {len(demo)} delta actions, task {demo.task!r}")
    print(f"  provenance: pose_source {session.pose_provenance!r}, "
          f"source {demo.metadata['source']!r}. No hand touched anything here, and the "
          "capture says so\n  itself rather than leaving it to this comment.")
    print("  the capture read the arm through ExperimentalSCARA and never named a backend "
          "type,\n  which is why the same call would record a vendor driver. None was run "
          "here; see the docstring.")

    frames_path = explore.export_frames()
    frame = next(iter(explore.frames()))
    print(f"flat export: {frames_path.name}, fields {explore.meta()['state_fields']}")
    print(f"  frame 0: state {np.round(frame['observation.state'], 2).tolist()} "
          f"action {np.round(frame['action'], 2).tolist()}")
    print("  the in-memory nan is the gripper width, which these frames never measured. "
          "The JSONL export writes null; 0.0 would train as a real measurement.")

    # --- 2. fit ---------------------------------------------------------------
    model, fit = ResidualDynamics.fit(explore, seed=0)
    print()
    print(fit.summary())
    gains, offset = probe(model, START)
    print(f"  recovered gain:   {np.round(gains, 4).tolist()} (planted {GAIN})")
    print(f"  recovered offset: {np.round(offset, 4).tolist()} mm (planted {list(BIAS[:3])})")
    print("  the zero rmse is honest and not impressive: this arm's defect is exactly the "
          "shape\n  the fit searches, and nothing here is noisy. Real demos are neither.")

    perfect = [
      simulate_episode(
        KinematicDynamics(), random_walk_actions(40, step_mm=STEP_MM, seed=100 + s), start=START
      )
      for s in range(4)
    ]
    _, honest = ResidualDynamics.fit(perfect, seed=0)
    print("\nthe same fit on episodes from an arm that tracks perfectly:")
    print(indent(honest.summary()))

    # The third verdict, and the one that is easy to forget exists. `improved` is
    # three-valued: a single episode cannot be split by episode, the holdout
    # becomes neighbouring frames of the same motion, and the fit is refused a
    # verdict rather than given a flattering one. `not fit.improved` reads
    # correctly for all three values, which is why the guard below is written
    # that way and not as `fit.improved is False`.
    _, unsound = ResidualDynamics.fit(
      [simulate_episode(truth, random_walk_actions(40, step_mm=STEP_MM, seed=0), start=START)],
      seed=0,
    )
    print("\nthe same fit given one episode instead of four:")
    print(indent(unsound.summary()))
    print(f"  improved is {unsound.improved!r}, not False. The data did not refute the "
          "correction,\n  it was never in a position to test it.")
    for label, candidate in (("four episodes", fit), ("one episode", unsound)):
      verdict = "plan against the fit" if candidate.improved else "keep the baseline"
      print(f"  {label:<14}: improved={str(candidate.improved):<5} -> {verdict}")

    # --- 3. plan and score ----------------------------------------------------
    def planner_for(task: ReachTask, dynamics: DynamicsModel):
      return lambda: SamplingPlanner(task.goal, dynamics=dynamics, step_mm=STEP_MM, seed=0)

    def reach(tolerance_mm: float) -> ReachTask:
      return ReachTask(
        goal=Goal.at(200.0, 0.0, 40.0, tolerance_mm=tolerance_mm),
        start=START, max_steps=40, workspace=Workspace(), start_jitter_mm=JITTER_MM,
        backend_factory=mistracked, name=f"reach ({tolerance_mm} mm)",
      )

    models = (("naive", KinematicDynamics()), ("fitted", model))
    envelope = SafetyEnvelope(max_step_mm=STEP_MM, workspace=Workspace(), max_steps=40)

    loose = reach(2.0)
    print("\nat a 2.0 mm tolerance both models pass, so the success rate cannot rank them:")
    for label, dynamics in models:
      r = await evaluate(planner_for(loose, dynamics), loose, n_trials=N_TRIALS, envelope=envelope)
      print(f"  {label:<6}: success {r.success_rate:.2f}, {r.mean_steps_to_success:.1f} steps, "
            f"final error {r.mean_final_error_mm:.3f} mm, nulls "
            f"{'ok' if r.valid else 'BROKEN CRITERION'}")
    print("  the error column already separates them. Tighten the tolerance until the "
          "success rate does too.")

    tight = reach(0.5)
    print("\nat 0.5 mm the model is the whole difference:")
    for label, dynamics in models:
      r = await evaluate(planner_for(tight, dynamics), tight, n_trials=N_TRIALS, envelope=envelope,
                         policy_name=f"SamplingPlanner + {label} dynamics")
      print(indent(r.summary()))

    # --- 4. refuse ------------------------------------------------------------
    # Two gates, and they refuse different things. The screen refuses one
    # prediction. The hardware gate refuses the whole run, whatever the screen
    # thought, because permission to move a real arm is not something a
    # confidence number is allowed to grant.
    screen = demo_screen()
    print()
    print(screen.summary())

    good = screen.check(
      prediction(0.93), screened_request("ep-0001", t_capture_s=10.0), now_s=10.05,
      readout=ReadoutResult(
        name="aruco", ok=True, state=np.array(START, float), reason="4 markers"),
    )
    print(indent(good.summary()))

    stale = screen.check(
      prediction(0.62), screened_request("ep-0002", t_capture_s=10.0), now_s=10.90,
      readout=ReadoutResult(
        name="aruco", ok=True, state=np.array(START, float), reason="4 markers"),
    )
    print("\nthe same screen on a prediction that is both late and unsure:")
    print(indent(stale.summary()))
    print("  both reasons are reported, not the first one hit. Repairing the staleness and "
          "re-running\n  would otherwise be how the low confidence gets discovered, one "
          "round trip at a time.")

    # A real PyLabRobot hardware driver, constructed and never connected: the gate
    # keys on "is not the simulation backend", so it does not need this repository
    # to ship a driver of its own to be exercised honestly.
    hardware = ExperimentalSCARA(backend=PreciseFlex400Backend(host=UNREACHABLE_HOST))
    try:
      await rollout(hardware, planner_for(tight, model)(), envelope=envelope)
    except HardwarePermissionRequired as exc:
      print(f"\nrefused on a real backend: {type(exc).__name__}")
      print(indent(str(exc)))
    # Nothing was connected and nothing was commanded. rollout raises on the
    # backend type before it observes, so setup() was never called on the driver
    # above and no socket was opened to UNREACHABLE_HOST.
    try:
      await rollout(hardware, NoOpPolicy(), envelope=envelope)
    except HardwarePermissionRequired:
      print("  the same refusal for a policy that does nothing at all: the gate is about "
            "the backend,\n  not about how dangerous the policy looks.")
  finally:
    shutil.rmtree(root)


if __name__ == "__main__":
  asyncio.run(main())
