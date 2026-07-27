# plr-lab-robot

Simulation-first lab robotics: bring a robotic arm under Python control, give it
eyes and hands, then execute domain-authored workcell tasks against visible state.

> The Python package is `plr_lr` (short, for imports). The repository is named
> `plr-lab-robot` so its purpose is obvious at a glance; they are the same project.

PyLabRobot ships an arm front end (`ExperimentalSCARA`) and a hardware driver
(PreciseFlex), but no simulation backend, so the arm API cannot be exercised
without a physical robot. `plr-lr` closes that gap and builds three layers on top,
in four phases:

```
  arm  ─▶  vision  ─▶  manipulation  ─▶  tasking
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
4. **tasking** - typed, domain-authored workcell contracts: explicit visible-state
   preconditions/postconditions, hard retry budgets, conservative recovery, and
   integrity-checkable task outcomes carrying protocol and sample identifiers.

Everything in this repository runs simulation-first and CPU-only, and is validated
by planting known ground truth and scoring recovery. PyLabRobot provides a seam
for later hardware integration; this repository makes no hardware-validation
claim.

## Install

```bash
pip install -e .
```

Runtime dependencies are `pylabrobot` and `numpy`; the optional `vision` extra
adds OpenCV. `RFDetrDetector` and `SAM2Detector` are adapter stubs only: no model
implementation, weights, or real-image validation is bundled. The simulation
path needs neither weights nor a GPU.

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

### Phase 4: auditable workcell task

The task layer keeps expert intent explicit instead of hiding it in a generic
planner. This contract says what resource may move, why it is moving, who or what
requested it, which protocol/sample identifiers travel with it, and the visible
state that must be observed on both sides of execution.

```python
from plr_lr.tasking import ProvenanceContext, TaskContract, TaskExecutor

contract = TaskContract.move_labware(
    task_id="reader-load-001",
    intent="Load the tracked assay plate into the reader",
    labware_name="assay_plate",
    source_site="incubator_out",
    destination_site="reader_in",
    provenance=ProvenanceContext(
        requested_by="assay-scientist",
        protocol_id="viability-read",
        protocol_version="1.0",
        sample_ids=("sample-A", "sample-B"),
    ),
)
outcome = await TaskExecutor(wc).execute(contract)
assert outcome.ok and outcome.verify_integrity()
```

Only failures explicitly classified as transient are retryable, and never more
than three attempts. Before retrying, the executor halts motion and proves from
visible state that the original preconditions still hold and the arm is not
holding a resource. A partial pick therefore stops for intervention instead of
blindly replaying the task.

Run the four worked examples end to end:

```bash
PYTHONPATH=. python examples/01_stand_up_arm.py
PYTHONPATH=. python examples/02_vision_guided_pick.py
PYTHONPATH=. python examples/03_decap_recap.py
PYTHONPATH=. python examples/04_auditable_workcell_task.py
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
  tasking.py          typed contracts, state gates, bounded recovery, audit outcomes
  qc.py               readiness gate (setup, home, gripper, hand-eye rms)
examples/             one runnable script per phase
tests/                plant, fault-inject, recover, and refuse-unsafe-retry validation
```

## Validation

```bash
python -m unittest discover -s tests -t .
```

The suite plants ground truth and scores recovery: the hand-eye transform is
recovered to sub-micron RMS, the guided pick recovers a known offset, and decap
flips and verifies the capped state. The readiness gate in `qc.py` mirrors the
di-omics wet-lab QC shape so the same pass/fail reporting carries into hardware
bring-up. Tasking tests inject pre-motion timeouts, unexpected adapter errors,
postcondition failures, and a disconnect after pick. They assert that recovery is
bounded, unsafe replay is refused, and the JSON outcome's event/trace digests
verify.

## Evidence boundary

The task artifact is an auditable record of software-observed state and commands.
Its hash chain makes accidental changes detectable; it is not a signed identity,
an external timestamp, or proof that the physical world matched the model.
Protocol, operator, and sample identifiers are labels supplied by the caller.

All repository validation is simulation-only. The simulator is kinematic and
geometric; it does **not** model or validate dynamics, collision avoidance,
contact force, torque, slip, liquid/sample identity, sensor performance, robot
safety functions, or hardware interlocks. Those require risk assessment,
calibration, sensing, and qualified hardware bring-up outside this repository.

## On hardware

The arm, vision, and manipulation interfaces are designed around PyLabRobot's
`ExperimentalSCARA` API. A hardware backend can be wired at that seam, but this
repository has not validated that integration. The task executor gates unknown
backends by default; enabling one is an explicit operator override and still does
not turn a software outcome into hardware-validation evidence.

## License

MIT.
