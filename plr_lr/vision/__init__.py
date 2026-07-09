"""Phase 2: computer vision for the arm.

Camera model, eye-in-hand hand-eye calibration, a detector seam (with a
ground-truth simulation detector and SOTA adapter stubs), and a closed-loop
vision-guided pick.
"""

from .camera import CameraIntrinsics, base_from_camera
from .detector import Detection, Detector, RFDetrDetector, SAM2Detector, SimDetector
from .guided_pick import (
  PickCorrection,
  apply_correction,
  estimate_offset,
  pixel_to_plane,
  vision_guided_pick,
)
from .hand_eye import HandEyeResult, calibrate_eye_in_hand, kabsch
from .transforms import Transform

__all__ = [
  "Transform",
  "CameraIntrinsics",
  "base_from_camera",
  "Detection",
  "Detector",
  "SimDetector",
  "RFDetrDetector",
  "SAM2Detector",
  "kabsch",
  "calibrate_eye_in_hand",
  "HandEyeResult",
  "pixel_to_plane",
  "estimate_offset",
  "apply_correction",
  "vision_guided_pick",
  "PickCorrection",
]
