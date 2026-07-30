# plr-lab-robot

PyLabRobot LabRobot: bring a robotic arm under PyLabRobot control, then give it eyes and hands.

> The Python package is `plr_lr` (short, for imports). The repository is named
> `plr-lab-robot` so its purpose is obvious at a glance; they are the same project.

PyLabRobot ships an arm front end (`ExperimentalSCARA`) and a hardware driver
(PreciseFlex), but no simulation backend, so the arm API cannot be exercised
without a physical robot. `plr-lr` closes that gap and builds two layers on top,
in three phases:

```
  arm  ->  vision  ->  manipulation
```

1. **arm** - a simulation backend for `ExperimentalSCARA`: joint and Cartesian
   state, gripper state, a home interlock, a workspace envelope, and a command
   trace. Stand the arm up and move plates with no hardware.
2. **vision** - an eye-in-hand (wrist-mounted) camera: hand-eye calibration,
   a detector seam, and a closed-loop guided pick that corrects a taught pose
   from what the camera sees.
3. **manipulation** - dexterous skills beyond plate handling, starting with
   uncapping and recapping threaded vessels. Every skill is gated: precondition,
   execute, verify.

Everything runs simulation-first and CPU-only, and is validated by planting known
ground truth and scoring recovery. Swap the simulation backend for a hardware
backend and the same scripts drive a real arm.

## Install

```bash
pip install -e .
```

Runtime dependencies are `pylabrobot` and `numpy`. Real SOTA detectors
(RF-DETR, SAM2) are an optional `vision` extra; the simulation path needs
neither weights nor a GPU.

## Quickstart

### Phase 1: stand up the arm

```python
import asyncio
from plr_lr import Labware, Workcell

async def main():
    wc = Workcell.sim()          # ExperimentalSCARA + simulation backend
    await wc.setup()             # connect + home

    plate = Labware(name="assay_plate")
    wc.add_site("incubator_out", x=180, y=0, z=12, occupant=plate)
    wc.add_site("reader_in", x=-180, y=40, z=12)
    await wc.move_plate("incubator_out", "reader_in")

asyncio.run(main())
```

### Phase 2: vision-guided pick

Calibrate the wrist camera, detect the plate, and shift the taught pick to the
true position. A 7 mm placement error is corrected to sub-millimeter residual
even with pixel noise (see `examples/02_vision_guided_pick.py`).

```python
from plr_lr.vision import calibrate_eye_in_hand, estimate_offset, vision_guided_pick
```

### Phase 3: dexterous manipulation

```python
from plr_lr.manipulation import DecapSkill, RecapSkill

decap = DecapSkill(wc.arm, wc.world, "sample_tube",
                   coords=wc.coords_for("rack_a1"), cap_park=wc.coords_for("cap_park"))
result = await decap.run()       # rotates while lifting by thread pitch, then verifies
```

Run the three worked examples end to end:

```bash
PYTHONPATH=. python examples/01_stand_up_arm.py
PYTHONPATH=. python examples/02_vision_guided_pick.py
PYTHONPATH=. python examples/03_decap_recap.py
```

## Kinematics: telling the arm where to go

Until now this repo could exercise an arm API and could not tell an arm where to go.
`_joints` and `_cartesian` were independent variables: `move_to` with joint angles never
updated the Cartesian pose, and `move_to` with a Cartesian pose never updated the joints.
The two could disagree without limit and nothing noticed.

`plr_lr.arm.kinematics` closes that with modified Craig D-H for the xArm 6 and 7, joint
limits taken from the SDK's own `x_config.py`, and a reach envelope derived from that
geometry rather than quoted.

```python
from plr_lr.arm import xarm6, forward, inverse

m = xarm6()
pose = forward(m, [0.3, -0.5, 0.4, 0.2, 0.6, 0.1])
result = inverse(m, pose)          # 8 solutions, not one
```

The asymmetry between the two directions is the whole design.

**Forward kinematics is exact.** Given joints and geometry, the pose is determined. Round
tripping 25 random configurations through FK then IK recovers every pose to within 1e-9 mm
across every solution returned.

**Inverse kinematics is not.** A 6-DOF arm reaches the same Cartesian point in several
distinct configurations, so `inverse` returns **all** of them and choosing is a named
policy rather than a hidden default. That matters physically: the wrong branch reaches the
identical point by swinging the elbow through the deck.

Four things it refuses rather than approximating:

- **A pose outside the envelope.** It reports the distance and the bound and offers no
  nearest-reachable alternative, because silently moving somewhere else is how a plate ends
  up next to the nest instead of in it. The bound is derived from the shipped D-H over the
  declared joint limits and ignores self-collision, so it is necessary and not sufficient.
- **Geometry it cannot source.** `unverified_model` carries provenance and refuses to
  compute a pose at all. An arm driven on invented link lengths moves confidently to the
  wrong place.
- **A solution outside a joint limit.** Returned flagged, never silently.
- **A near-singular pose**, where the joint velocities required for a bounded Cartesian
  velocity diverge.

And `reachable()` says nothing about whether a path exists. Conflating the two is how an arm
gets commanded into a move that is valid at both endpoints and passes through the bench.

### Repeatability is not accuracy

The distinction the datasheet invites you to get wrong, and the one that matters most for a
lab. A repeatability figure bounds returning to a **taught** pose. It does not bound the
error when moving to a **computed** one, which is what IK produces, because absolute
accuracy also carries link tolerance, joint offset, deflection and thermal error.

So a `Pose` knows its own origin. `PoseOrigin.TAUGHT` and `PoseOrigin.COMPUTED` are not
interchangeable for a placement that has to land inside a nest's lead-in chamfer, and
`stations` reports which of your stations are which.

## Stations: a plate move that is checked rather than assumed

`workcell.py` was a dict of taught poses and a `move_plate` that teleported occupancy, so
`incubator_out` and `reader_in` were string keys with no incubator and no reader behind
them. A workcell that cannot say what a station **is** cannot say whether a move between
two of them is legal.

`plr_lr.arm.stations` computes legality instead of asserting it. A move is refused when the
source holds nothing, the destination is full, either pose is missing, either pose is
unreachable, or the destination instrument is mid-cycle. That last one is not a queueing
problem: a plate placed into a reader that is still reading is a crash.

Handed two stations with nothing set, it returns four refusals rather than permission,
including the one worth quoting: *an untaught station is a name, not a place.*

`sequence()` checks a whole run, because a later move can be illegal only because an
earlier one left a station occupied, and reports the first that fails and why.

A completed move is also not a seated plate. `MoveOutcome` distinguishes a placement the
arm merely finished from one a sensor confirmed, the same way a provenance layer separates
software intent from an instrument's word.

## Layout

```
plr_lr/
  world.py            deck model: sites, labware, poses, capped/held state
  arm/
    sim_backend.py    SimulationArmBackend (implements SCARABackend)
    workcell.py       Workcell: taught sites + move_plate
  vision/
    transforms.py     rigid 4x4 transforms
    camera.py         pinhole intrinsics + eye-in-hand pose
    hand_eye.py       eye-in-hand calibration (Kabsch/Umeyama)
    detector.py       Detector seam + SimDetector + RF-DETR/SAM2 stubs
    guided_pick.py    ray-plane back-projection + closed-loop pick
  manipulation/
    skills.py         Skill base: precondition/execute/verify
    grasp.py          grasp planning for caps
    decap.py          DecapSkill / RecapSkill
  qc.py               readiness gate (setup, home, gripper, hand-eye rms)
examples/             one runnable script per phase
tests/                plant-and-recover validation
```

## Validation

```bash
python -m unittest discover -s tests -t .
```

The suite plants ground truth and scores recovery: the hand-eye transform is
recovered to sub-micron RMS, the guided pick recovers a known offset, and decap
flips and verifies the capped state. The readiness gate in `qc.py` mirrors the
di-omics wet-lab QC shape so the same pass/fail reporting carries into hardware
bring-up.

## On hardware

The vision and manipulation layers are backend-agnostic: they talk to the arm
through the PyLabRobot `ExperimentalSCARA` API. To move to a real arm, construct
`ExperimentalSCARA` with a hardware backend (for example `PreciseFlexBackend`)
instead of `SimulationArmBackend`; teach the sites, calibrate the camera, and the
same code path applies.

## License

MIT.
