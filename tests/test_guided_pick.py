import unittest

import numpy as np

from plr_lr import Labware, Pose2D, Workcell, site_coords
from plr_lr.vision import (
  CameraIntrinsics,
  SimDetector,
  Transform,
  estimate_offset,
  vision_guided_pick,
)


def overhead_camera(x, y, z_cam=250.0):
  # camera looking straight down (base -x maps to cam +? ) at (x,y)
  # base_from_cam: camera positioned above (x,y), optical axis pointing down (-z base)
  R = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)  # flip so +Z_cam points down
  return Transform.from_rt(R, [x, y, z_cam])


class GuidedPickTests(unittest.IsolatedAsyncioTestCase):
  def setUp(self):
    self.intr = CameraIntrinsics(fx=1400.0, fy=1400.0, cx=640.0, cy=512.0)
    self.plane_z = 10.0

  def test_offset_recovered_noiseless(self):
    taught_xy = (150.0, 0.0)
    true_offset = (6.5, -4.0)
    true_xy = (taught_xy[0] + true_offset[0], taught_xy[1] + true_offset[1])

    base_from_cam = overhead_camera(taught_xy[0], taught_xy[1])
    detector = SimDetector(self.intr)
    world_pts = {"plate": np.array([true_xy[0], true_xy[1], self.plane_z])}
    dets = detector.observe(world_pts, base_from_cam)
    self.assertEqual(len(dets), 1)

    taught = site_coords(Pose2D(taught_xy[0], taught_xy[1], self.plane_z))
    corr = estimate_offset(dets[0], self.intr, base_from_cam, taught, self.plane_z)
    self.assertAlmostEqual(corr.dx, true_offset[0], places=4)
    self.assertAlmostEqual(corr.dy, true_offset[1], places=4)

  def test_correction_beats_open_loop(self):
    taught_xy = (150.0, 20.0)
    true_offset = (5.0, 5.0)
    true_xy = (taught_xy[0] + true_offset[0], taught_xy[1] + true_offset[1])
    base_from_cam = overhead_camera(taught_xy[0], taught_xy[1])
    detector = SimDetector(self.intr, pixel_noise=0.5, seed=3)
    world_pts = {"plate": np.array([true_xy[0], true_xy[1], self.plane_z])}

    # average a few noisy estimates
    taught = site_coords(Pose2D(taught_xy[0], taught_xy[1], self.plane_z))
    est = []
    for _ in range(25):
      d = detector.observe(world_pts, base_from_cam)[0]
      c = estimate_offset(d, self.intr, base_from_cam, taught, self.plane_z)
      est.append(c.measured_xy)
    est = np.mean(est, axis=0)

    open_loop_err = np.hypot(taught_xy[0] - true_xy[0], taught_xy[1] - true_xy[1])
    closed_loop_err = np.hypot(est[0] - true_xy[0], est[1] - true_xy[1])
    self.assertLess(closed_loop_err, open_loop_err * 0.1)

  async def test_guided_pick_targets_true_position(self):
    wc = Workcell.sim()
    await wc.setup()
    taught_xy = (150.0, 0.0)
    plate = Labware(name="plate")
    wc.add_site("stage", x=taught_xy[0], y=taught_xy[1], z=self.plane_z, occupant=plate)

    true_offset = (7.0, -3.0)
    true_xy = (taught_xy[0] + true_offset[0], taught_xy[1] + true_offset[1])
    base_from_cam = overhead_camera(taught_xy[0], taught_xy[1])
    detector = SimDetector(self.intr)
    det = detector.observe({"plate": np.array([true_xy[0], true_xy[1], self.plane_z])}, base_from_cam)[0]

    corr = await vision_guided_pick(
      wc.arm, det, self.intr, base_from_cam, wc.coords_for("stage"),
      plane_z=self.plane_z, plate_width=plate.plate_width,
    )
    self.assertAlmostEqual(corr.dx, true_offset[0], places=3)
    self.assertTrue(wc.backend.holding)
    # the descend target should be the corrected (true) position
    descend = [e for e in wc.backend.trace if e["action"] == "descend_pick"][-1]
    self.assertAlmostEqual(descend["x"], true_xy[0], places=3)
    self.assertAlmostEqual(descend["y"], true_xy[1], places=3)


if __name__ == "__main__":
  unittest.main()
