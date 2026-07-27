# Phases

The build is staged so each layer stands on a validated one below it.

## Phase 1: stand up the arm

Goal: exercise the full PyLabRobot arm API with no hardware, then move plates.

`SimulationArmBackend` implements every method of `pylabrobot.arms.backend.SCARABackend`:
`setup`, `stop`, `home`, `move_to_safe`, `move_to`, `approach`, `pick_up_resource`,
`drop_resource`, `open_gripper`, `close_gripper`, `is_gripper_closed`, `halt`,
`get_joint_position`, `get_cartesian_position`, `freedrive_mode`, `end_freedrive_mode`.

It adds three things a bring-up needs:

- a **home interlock**: motion before `home()` raises `ArmNotHomedError`, matching
  the interlock a real controller enforces;
- a **workspace envelope**: targets outside the configured box raise
  `WorkspaceViolation`;
- a **command trace**: every action is appended to `backend.trace` as a dict, so a
  protocol can be replayed and asserted on.

`Workcell` sits on top: it remembers the Cartesian pose taught for each named site
and turns pick + drop into `move_plate(src, dst)`, keeping the `World`'s site
occupancy in sync.

## Phase 2: computer vision

Goal: correct a taught pick from what a wrist camera sees, so hand placement drift
and calibration age stop causing misses.

- `camera.py` is a pinhole model with an eye-in-hand pose helper
  (`base_from_camera = base_from_tool @ tool_from_camera`).
- `hand_eye.py` recovers the wrist-to-camera transform by rigid point-set
  registration (Kabsch/Umeyama). Markers at known base positions are expressed in
  the tool frame from the known tool pose and paired with their measured
  camera-frame positions; stacking observations over several poses makes the
  problem well posed, and the RMS residual is reported for QC.
- `detector.py` defines the `Detector` seam. `SimDetector` projects known world
  points through the camera to synthesize detections (with optional pixel noise)
  so the loop can be validated without weights. `RFDetrDetector` and
  `SAM2Detector` are the SOTA adapter seams (optional `vision` extra).
- `guided_pick.py` back-projects a detection pixel onto the known deck plane
  (ray-plane intersection), measures the offset from the taught pose, and shifts
  the pick target before gripping.

## Phase 3: dexterous manipulation

Goal: actions past plate handling, starting with threaded caps.

`skills.py` defines a gated skill: `precondition -> execute -> verify`, returning a
`SkillResult`. `decap.py` implements `DecapSkill` (grip the cap, rotate
counter-clockwise while lifting by the thread pitch per turn, park the cap, verify
the vessel is open) and `RecapSkill` (the inverse). The motion stays within the
`ExperimentalSCARA` API to preserve a future hardware-adapter seam; this repository
does not validate the motion on a physical end-effector.

## Phase 4: domain-authored workcell execution

Goal: make expert intent executable without allowing an AI planner or transient
controller failure to bypass observable safety gates.

`tasking.py` adds four pieces:

- `TaskContract` and `MoveLabwareParameters` are a typed boundary around action,
  intent, protocol version, sample identifiers, and recovery budget;
- `StateCondition` checks named sites, labware held/capped state, cap parking, and
  arm readiness/homing/holding state before or after motion;
- `TaskExecutor` contains errors with `halt()`, retries only classified transient
  errors, and retries only if the original preconditions and empty gripper are
  observed again. Attempts are hard-limited to three;
- `TaskOutcome` captures before/after snapshots, condition results, the command
  trace slice, protocol/sample provenance, and a hash-linked event trail. Its
  JSON artifact can detect accidental mutation of recorded events or trace data.

`summarize_benchmark()` aggregates success, visible pre/postcondition pass rates,
recovered runs, and audit-integrity checks over deterministic simulation cases.
Fault-injection tests cover a pre-motion transient that recovers, an unexpected
adapter error that does not retry, a no-op that fails postconditions, and a
disconnect after pick that is refused a retry because the arm is still holding.

## What is validated, and what is not

The simulation is kinematic and geometric. It validates control flow, coordinate
math, calibration recovery, state bookkeeping, task gating, and bounded recovery
decisions. It does not model dynamics, collision avoidance, contact forces,
torque, slip, sample identity, sensor behavior, robot safety functions, or
hardware interlocks. The audit hash chain is an integrity check, not signed proof
or physical-world evidence. Those gaps belong to qualified hardware bring-up and
must not be inferred from the passing simulation suite.
