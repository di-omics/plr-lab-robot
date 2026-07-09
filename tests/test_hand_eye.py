import unittest

import numpy as np

from plr_lr.vision import Transform, calibrate_eye_in_hand
from plr_lr.vision.camera import base_from_camera


class HandEyeTests(unittest.TestCase):
  def test_recovers_planted_hand_eye_transform(self):
    rng = np.random.default_rng(0)
    # ground-truth wrist-to-camera transform we will try to recover
    true_cam_from_tool = Transform.from_euler(0.2, -0.35, 1.1, t=(40.0, -15.0, 65.0))

    # a fixed calibration fixture: markers at known base-frame positions
    markers_base = np.array(
      [[100, 100, 0], [140, 100, 0], [100, 140, 0], [140, 140, 5], [120, 120, 12]],
      float,
    )

    observations = []
    for _ in range(6):
      base_from_tool = Transform.from_euler(
        rng.uniform(-0.3, 0.3),
        rng.uniform(-0.3, 0.3),
        rng.uniform(-np.pi, np.pi),
        t=rng.uniform([-50, -50, 150], [50, 50, 260]),
      )
      # camera measures each marker in its own frame
      cam_from_base = (true_cam_from_tool @ base_from_tool.inv())
      markers_cam = cam_from_base.apply(markers_base)
      observations.append((base_from_tool, markers_base, markers_cam))

    result = calibrate_eye_in_hand(observations)
    self.assertTrue(result.ok)
    self.assertLess(result.rms_mm, 1e-6)
    np.testing.assert_allclose(
      result.cam_from_tool.M, true_cam_from_tool.M, atol=1e-6
    )

  def test_noise_inflates_residual_but_stays_bounded(self):
    rng = np.random.default_rng(7)
    true_cam_from_tool = Transform.from_euler(-0.1, 0.2, -0.9, t=(30.0, 10.0, 70.0))
    markers_base = np.array(
      [[100, 100, 0], [150, 100, 0], [100, 150, 0], [150, 150, 8], [125, 125, 15]], float
    )
    observations = []
    for _ in range(8):
      base_from_tool = Transform.from_euler(
        rng.uniform(-0.2, 0.2), rng.uniform(-0.2, 0.2), rng.uniform(-np.pi, np.pi),
        t=rng.uniform([-40, -40, 160], [40, 40, 240]),
      )
      cam_from_base = true_cam_from_tool @ base_from_tool.inv()
      noisy = cam_from_base.apply(markers_base) + rng.normal(0, 0.05, size=markers_base.shape)
      observations.append((base_from_tool, markers_base, noisy))
    result = calibrate_eye_in_hand(observations)
    self.assertLess(result.rms_mm, 0.2)  # sub-0.2mm under 50um point noise

  def test_base_from_camera_roundtrip(self):
    base_from_tool = Transform.from_euler(0.1, 0.1, 0.5, t=(10, 20, 200))
    cam_from_tool = Transform.from_euler(0.0, 0.0, 0.0, t=(0, 0, 60))
    bc = base_from_camera(base_from_tool, cam_from_tool)
    # camera origin in base == tool applied to tool_from_cam translation
    expected = base_from_tool.apply(cam_from_tool.inv().t)
    np.testing.assert_allclose(bc.t, expected, atol=1e-9)


if __name__ == "__main__":
  unittest.main()
