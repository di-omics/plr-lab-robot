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
`ExperimentalSCARA` API, so the same skill maps onto a real end-effector with a
rotating wrist.

## What is validated, and what is not

The simulation is kinematic and geometric. It validates control flow, coordinate
math, calibration recovery, and state bookkeeping. It does not model dynamics,
contact forces, torque, or slip; those belong to hardware bring-up and are gated
by the readiness checks in `qc.py`.
