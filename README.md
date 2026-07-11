# plr-lab-robot

PyLabRobot LabRobot: bring a robotic arm under PyLabRobot control, then give it eyes and hands.

> The Python package is `plr_lr` (short, for imports). The repository is named
> `plr-lab-robot` so its purpose is obvious at a glance; they are the same project.

PyLabRobot ships an arm front end (`ExperimentalSCARA`) and a hardware driver
(PreciseFlex), but no simulation backend, so the arm API cannot be exercised
without a physical robot. `plr-lr` closes that gap and builds two layers on top,
in three phases:

```
  arm  ─▶  vision  ─▶  manipulation
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
