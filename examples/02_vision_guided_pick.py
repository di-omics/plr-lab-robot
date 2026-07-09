"""Phase 2: calibrate the wrist camera, then correct a taught pick from a detection.

A plate is placed 7 mm off its taught position. Open loop, the arm would miss by
7 mm. With the eye-in-hand camera it detects the plate, back-projects onto the
deck, and shifts the pick target to the true position.

    python examples/02_vision_guided_pick.py
"""

import asyncio

import numpy as np

from plr_lr import Labware, Pose2D, Workcell, site_coords
from plr_lr.vision import (
  CameraIntrinsics,
  SimDetector,
  Transform,
  calibrate_eye_in_hand,
  estimate_offset,
  vision_guided_pick,
)


def overhead_camera(x, y, z_cam=250.0) -> Transform:
  R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)  # optical axis points down
  return Transform.from_rt(R, [x, y, z_cam])


async def main() -> None:
  intr = CameraIntrinsics(fx=1400.0, fy=1400.0, cx=640.0, cy=512.0)
  plane_z = 10.0

  # --- hand-eye calibration (plant a transform, recover it from observations) ---
  true_cam_from_tool = Transform.from_euler(0.15, -0.2, 0.9, t=(35.0, -10.0, 60.0))
  markers = np.array([[100, 100, 0], [140, 100, 0], [100, 140, 0], [140, 140, 6]], float)
  rng = np.random.default_rng(0)
  obs = []
  for _ in range(6):
    base_from_tool = Transform.from_euler(
      rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-np.pi, np.pi),
      t=rng.uniform([-40, -40, 160], [40, 40, 240]),
    )
    cam_from_base = true_cam_from_tool @ base_from_tool.inv()
    obs.append((base_from_tool, markers, cam_from_base.apply(markers)))
  cal = calibrate_eye_in_hand(obs)
  print(f"hand-eye: rms {cal.rms_mm:.2e} mm over {cal.n_points} points -> ok={cal.ok}")

  # --- vision-guided pick ---
  wc = Workcell.sim()
  await wc.setup()
  taught_xy = (150.0, 0.0)
  plate = Labware(name="plate")
  wc.add_site("stage", x=taught_xy[0], y=taught_xy[1], z=plane_z, occupant=plate)

  true_xy = (taught_xy[0] + 7.0, taught_xy[1] - 3.0)  # placed off-target
  base_from_cam = overhead_camera(*taught_xy)
  detector = SimDetector(intr, pixel_noise=0.3, seed=1)
  det = detector.observe({"plate": np.array([true_xy[0], true_xy[1], plane_z])}, base_from_cam)[0]

  nominal = estimate_offset(det, intr, base_from_cam, wc.coords_for("stage"), plane_z)
  print(f"detected offset from taught: dx={nominal.dx:+.2f} mm dy={nominal.dy:+.2f} mm")

  corr = await vision_guided_pick(
    wc.arm, det, intr, base_from_cam, wc.coords_for("stage"),
    plane_z=plane_z, plate_width=plate.plate_width,
  )
  descend = [e for e in wc.backend.trace if e["action"] == "descend_pick"][-1]
  print(f"open-loop target : ({taught_xy[0]:.2f}, {taught_xy[1]:.2f})  true ({true_xy[0]:.2f}, {true_xy[1]:.2f})")
  print(f"corrected target : ({descend['x']:.2f}, {descend['y']:.2f})  residual "
        f"{np.hypot(descend['x'] - true_xy[0], descend['y'] - true_xy[1]):.2f} mm")


if __name__ == "__main__":
  asyncio.run(main())
