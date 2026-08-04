"""Tests for plr_lr.learn.readout, the step that turns a picture into millimetres.

Organised around the two claims this module makes rather than around its
functions. The first is that :class:`MarkerReadout` is a working classical
readout and not a placeholder, which is only demonstrable by planting a pose,
rendering it through the camera model, and getting it back: every test in
``TestMarkerReadoutRecoversAPlantedPose`` is that round trip, and the tolerance
is tight enough that an approximation would fail it.

The second is that a readout which cannot measure says so. Four ways of failing
are exercised separately, because they have four different remedies and a single
``ok=False`` with no reason attached sends every one of them to the same person.

``ConstantPoseReadout`` appears throughout as the control. It is expected to
report a pose it never measured, and the test that pins that behaviour is the one
that would catch a real readout quietly acquiring a fallback to the taught
nominal.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from plr_lr.learn.readout import (
  DEGENERATE,
  MEASURED,
  NO_MARKERS,
  RESIDUAL_TOO_LARGE,
  UNDERDETERMINED,
  ConstantPoseReadout,
  InjectedReadout,
  MarkerReadout,
  Readout,
  ReadoutResult,
  read_or_refuse,
  readout_error,
)
from plr_lr.learn.types import Observation
from plr_lr.vision.camera import CameraIntrinsics, base_from_camera
from plr_lr.vision.detector import Detection
from plr_lr.vision.transforms import Transform

#: Camera looking straight down, rigidly mounted 60 mm above the tool. The flip
#: is what makes +Z_cam point at the deck rather than at the ceiling; without it
#: every marker back-projects behind the camera and the whole fixture is silently
#: mirrored.
_R_FLIP = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
TOOL_FROM_CAM = Transform.from_rt(_R_FLIP, [0.0, 0.0, 60.0])
CAM_FROM_TOOL = TOOL_FROM_CAM.inv()

INTRINSICS = CameraIntrinsics(fx=1400.0, fy=1400.0, cx=640.0, cy=512.0)
PLANE_Z = 10.0

#: Three markers, deliberately not collinear and deliberately not symmetric, so a
#: fit that recovered the rotation only up to a half turn would be visible.
MODEL = {"a": (-30.0, -20.0), "b": (30.0, -20.0), "c": (0.0, 25.0)}

TOOL_POSE = (150.0, 20.0, 260.0, 0.0)


def marker_observation(
  object_pose=(150.0, 20.0, 0.0),
  tool_pose=TOOL_POSE,
  confidence: float = 0.9,
  labels=None,
  plane_z: float = PLANE_Z,
) -> Observation:
  """Render the marker layout at a known object pose into pixel detections.

  This is the inverse of what the readout does, written independently of it: the
  fixture composes the transforms and calls ``intrinsics.project``, while the
  readout calls ``pixel_to_plane`` and fits. A shared helper between the two would
  make the round trip prove only that one function is its own inverse.
  """
  base_from_tool = Transform.from_euler(0.0, 0.0, tool_pose[3], t=tool_pose[:3])
  cam_from_base = base_from_camera(base_from_tool, CAM_FROM_TOOL).inv()
  ox, oy, orz = object_pose
  cos_o, sin_o = math.cos(orz), math.sin(orz)
  detections = []
  for label, (mx, my) in MODEL.items():
    if labels is not None and label not in labels:
      continue
    point = np.array(
      [ox + cos_o * mx - sin_o * my, oy + sin_o * mx + cos_o * my, plane_z], float
    )
    uv = INTRINSICS.project(cam_from_base.apply(point))
    detections.append(
      Detection(label=label, uv=(float(uv[0]), float(uv[1])), confidence=confidence)
    )
  return Observation(
    t=0.0,
    x=tool_pose[0],
    y=tool_pose[1],
    z=tool_pose[2],
    rz=tool_pose[3],
    detections=tuple(detections),
  )


def readout() -> MarkerReadout:
  return MarkerReadout(INTRINSICS, CAM_FROM_TOOL, PLANE_Z, MODEL)


class TestMarkerReadoutRecoversAPlantedPose(unittest.TestCase):
  """The declared floor is a working measurement, not a stub with a docstring."""

  def test_a_planted_pose_comes_back_to_within_a_micrometre(self):
    truth = (152.0, 18.0, 0.35)
    result = readout().read(marker_observation(object_pose=truth))

    self.assertTrue(result.ok)
    self.assertEqual(result.reason, MEASURED)
    self.assertEqual(result.n_markers, 3)
    self.assertAlmostEqual(result.state[0], truth[0], places=6)
    self.assertAlmostEqual(result.state[1], truth[1], places=6)
    self.assertAlmostEqual(result.state[2], PLANE_Z, places=9)
    self.assertAlmostEqual(result.state[3], truth[2], places=8)
    self.assertLess(result.residual_mm, 1e-6)

  def test_it_works_from_a_rotated_tool_as_well(self):
    """The hand-eye composition is exercised, not just the plane intersection.

    With the tool at rz zero the camera rotation is the identity and a readout
    that ignored ``observation.rz`` entirely would pass every other test here.
    """
    truth = (5.0, -7.0, -0.9)
    result = readout().read(
      marker_observation(object_pose=truth, tool_pose=(0.0, 0.0, 260.0, 0.4))
    )
    self.assertTrue(result.ok)
    self.assertAlmostEqual(result.state[0], truth[0], places=6)
    self.assertAlmostEqual(result.state[1], truth[1], places=6)
    self.assertAlmostEqual(result.state[3], truth[2], places=8)

  def test_a_half_turn_is_not_confused_with_no_turn(self):
    for rz in (0.0, math.pi / 2.0, -math.pi / 2.0, 2.5):
      with self.subTest(rz=rz):
        result = readout().read(marker_observation(object_pose=(150.0, 20.0, rz)))
        self.assertTrue(result.ok)
        wrapped = math.atan2(math.sin(result.state[3] - rz), math.cos(result.state[3] - rz))
        self.assertAlmostEqual(wrapped, 0.0, places=8)

  def test_pixel_noise_degrades_the_fit_without_breaking_it(self):
    """A few tenths of a pixel is a fraction of a millimetre, not a refusal.

    Also confirms that ``residual_mm`` is a real number rather than a constant:
    it is near machine zero on the noiseless fixture above and measurably above
    zero here.
    """
    clean = marker_observation(object_pose=(150.0, 20.0, 0.2))
    rng = np.random.default_rng(0)
    noisy = Observation(
      t=clean.t,
      x=clean.x,
      y=clean.y,
      z=clean.z,
      rz=clean.rz,
      detections=tuple(
        Detection(
          label=d.label,
          uv=(d.uv[0] + float(rng.normal(0.0, 0.5)), d.uv[1] + float(rng.normal(0.0, 0.5))),
          confidence=d.confidence,
        )
        for d in clean.detections
      ),
    )
    result = readout().read(noisy)
    self.assertTrue(result.ok)
    self.assertGreater(result.residual_mm, 0.0)
    error = readout_error(result.state, np.array([150.0, 20.0, PLANE_Z, 0.2]))
    self.assertLess(error.mm, 1.0)
    self.assertLess(error.deg, 2.0)


class TestMarkerReadoutRefusals(unittest.TestCase):
  """Four ways of failing, four reasons, and a NaN pose in every one of them."""

  def test_no_recognised_markers(self):
    result = readout().read(Observation(t=0.0, x=150.0, y=20.0, z=260.0))
    self.assertFalse(result.ok)
    self.assertEqual(result.reason, NO_MARKERS)
    self.assertTrue(bool(np.isnan(result.state).all()))

  def test_one_marker_is_underdetermined_rather_than_a_pose_with_rz_zero(self):
    """The refusal that stops an invented wrist angle reaching the arm.

    One marker fixes x and y perfectly well. The tempting thing is to return that
    with ``rz=0.0``, and the resulting pose is four finite plausible numbers, one
    of which was never measured.
    """
    result = readout().read(marker_observation(labels={"a"}))
    self.assertFalse(result.ok)
    self.assertEqual(result.reason, UNDERDETERMINED)
    self.assertEqual(result.n_markers, 1)
    self.assertTrue(bool(np.isnan(result.state).all()))

  def test_detections_below_the_confidence_floor_are_dropped_not_downweighted(self):
    result = readout().read(marker_observation(confidence=0.1))
    self.assertFalse(result.ok)
    self.assertEqual(result.reason, NO_MARKERS)

  def test_unknown_labels_are_ignored(self):
    obs = marker_observation()
    extra = Observation(
      t=obs.t,
      x=obs.x,
      y=obs.y,
      z=obs.z,
      rz=obs.rz,
      detections=obs.detections + (Detection(label="coffee", uv=(1.0, 1.0), confidence=1.0),),
    )
    result = readout().read(extra)
    self.assertTrue(result.ok)
    self.assertEqual(result.n_markers, 3)

  def test_the_most_confident_detection_per_label_wins(self):
    obs = marker_observation(object_pose=(150.0, 20.0, 0.0))
    duplicate = Observation(
      t=obs.t,
      x=obs.x,
      y=obs.y,
      z=obs.z,
      rz=obs.rz,
      # A second, less confident claim about marker "a", 200 px away. Averaging
      # the two would drag the fit; the weaker claim is dropped instead.
      detections=obs.detections
      + (Detection(label="a", uv=(obs.detections[0].uv[0] + 200.0, 0.0), confidence=0.6),),
    )
    result = readout().read(duplicate)
    self.assertTrue(result.ok)
    self.assertEqual(result.n_markers, 3)
    self.assertAlmostEqual(result.state[0], 150.0, places=6)

  def test_a_layout_that_does_not_fit_is_refused_rather_than_averaged(self):
    """The wrong labware on the deck, or a bumped camera.

    One marker is moved 40 mm out of the layout. The rigid fit still has a
    least-squares answer, and that answer is a pose the arm would act on.
    """
    obs = marker_observation(object_pose=(150.0, 20.0, 0.0))
    displaced = Observation(
      t=obs.t,
      x=obs.x,
      y=obs.y,
      z=obs.z,
      rz=obs.rz,
      detections=(
        obs.detections[0],
        Detection(
          label="b",
          uv=(obs.detections[1].uv[0] + 900.0, obs.detections[1].uv[1]),
          confidence=0.9,
        ),
        obs.detections[2],
      ),
    )
    result = readout().read(displaced)
    self.assertFalse(result.ok)
    self.assertEqual(result.reason, RESIDUAL_TOO_LARGE)
    self.assertGreater(result.residual_mm, 2.0)
    self.assertTrue(bool(np.isnan(result.state).all()))

  def test_markers_that_all_land_in_one_place_are_degenerate(self):
    obs = marker_observation()
    stacked = Observation(
      t=obs.t,
      x=obs.x,
      y=obs.y,
      z=obs.z,
      rz=obs.rz,
      detections=tuple(
        Detection(label=d.label, uv=obs.detections[0].uv, confidence=0.9) for d in obs.detections
      ),
    )
    result = readout().read(stacked)
    self.assertFalse(result.ok)
    self.assertEqual(result.reason, DEGENERATE)


class TestMarkerReadoutConfiguration(unittest.TestCase):
  """A model that cannot determine a pose is refused at construction."""

  def test_a_single_point_model_is_refused(self):
    with self.assertRaises(ValueError) as ctx:
      MarkerReadout(INTRINSICS, CAM_FROM_TOOL, PLANE_Z, {"a": (0.0, 0.0)})
    self.assertIn("rotation", str(ctx.exception))

  def test_a_model_whose_points_coincide_is_refused(self):
    with self.assertRaises(ValueError):
      MarkerReadout(INTRINSICS, CAM_FROM_TOOL, PLANE_Z, {"a": (1.0, 1.0), "b": (1.0, 1.0)})

  def test_malformed_bounds_are_refused(self):
    with self.assertRaises(ValueError):
      MarkerReadout(INTRINSICS, CAM_FROM_TOOL, float("nan"), MODEL)
    with self.assertRaises(ValueError):
      MarkerReadout(INTRINSICS, CAM_FROM_TOOL, PLANE_Z, MODEL, min_confidence=1.5)
    with self.assertRaises(ValueError):
      MarkerReadout(INTRINSICS, CAM_FROM_TOOL, PLANE_Z, MODEL, max_residual_mm=0.0)


class TestReadoutResultInvariant(unittest.TestCase):
  """A refused measurement carries no pose, and a successful one carries no NaN."""

  def test_ok_with_a_nan_pose_is_refused_at_construction(self):
    with self.assertRaises(ValueError):
      ReadoutResult(name="x", ok=True, state=np.array([1.0, 2.0, 3.0, np.nan]), reason=MEASURED)

  def test_not_ok_with_a_finite_pose_is_refused_at_construction(self):
    with self.assertRaises(ValueError) as ctx:
      ReadoutResult(name="x", ok=False, state=np.zeros(4), reason=NO_MARKERS)
    self.assertIn("taught nominal", str(ctx.exception))

  def test_refused_builds_the_one_shape_the_invariant_accepts(self):
    result = ReadoutResult.refused("x", NO_MARKERS)
    self.assertFalse(result.ok)
    self.assertTrue(bool(np.isnan(result.state).all()))

  def test_a_pose_of_the_wrong_width_is_refused(self):
    with self.assertRaises(ValueError):
      ReadoutResult(name="x", ok=True, state=np.zeros(3), reason=MEASURED)


class TestReadoutError(unittest.TestCase):
  """Millimetres and degrees, named, never summed."""

  def test_translation_and_rotation_are_reported_separately(self):
    error = readout_error((3.0, 4.0, 0.0, 0.0), (0.0, 0.0, 0.0, math.radians(30.0)))
    self.assertAlmostEqual(error.mm, 5.0, places=12)
    self.assertAlmostEqual(error.deg, 30.0, places=9)

  def test_the_fields_are_named_so_the_pair_cannot_be_unpacked_backwards(self):
    error = readout_error((0.0, 0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    self.assertEqual(error._fields, ("mm", "deg"))
    self.assertEqual(tuple(error), (error.mm, error.deg))

  def test_rotation_wraps_the_short_way(self):
    error = readout_error((0.0, 0.0, 0.0, math.pi - 0.01), (0.0, 0.0, 0.0, -math.pi + 0.01))
    self.assertLess(error.deg, 2.0)

  def test_a_refused_readout_has_no_error_and_raises_rather_than_returning_nan(self):
    refused = ReadoutResult.refused("x", NO_MARKERS)
    with self.assertRaises(ValueError) as ctx:
      readout_error(refused.state, np.zeros(4))
    self.assertIn("refusal", str(ctx.exception))


class TestInjectedReadout(unittest.TestCase):
  """A checkpoint is not a readout, and it says so on the first call."""

  def test_an_empty_readout_satisfies_the_protocol_and_still_cannot_measure(self):
    empty = InjectedReadout()
    self.assertIsInstance(empty, Readout)
    with self.assertRaises(NotImplementedError) as ctx:
      empty.read(Observation())
    self.assertIn("not a readout", str(ctx.exception))

  def test_the_checkpoint_is_named_in_the_refusal_when_one_was_recorded(self):
    with self.assertRaises(NotImplementedError) as ctx:
      InjectedReadout(checkpoint="some-weights").read(Observation())
    self.assertIn("Checkpoint metadata was recorded", str(ctx.exception))

  def test_a_bare_pose_is_refused_because_it_carries_no_ok_flag(self):
    injected = InjectedReadout(readout=lambda obs: np.zeros(4))
    with self.assertRaises(ValueError) as ctx:
      injected.read(Observation())
    self.assertIn("ok flag", str(ctx.exception))

  def test_a_non_callable_is_refused_at_construction(self):
    with self.assertRaises(TypeError):
      InjectedReadout(readout="weights.pt")


class TestReadOrRefuse(unittest.TestCase):
  """A readout that raises has still failed to measure, and the run continues."""

  def test_an_exception_becomes_a_refusal_the_screen_can_read(self):
    class Broken:
      name = "broken"

      def read(self, observation):
        raise RuntimeError("camera unplugged")

    result = read_or_refuse(Broken(), Observation())
    self.assertFalse(result.ok)
    self.assertEqual(result.reason, "raised:RuntimeError")

  def test_a_wrong_return_type_becomes_a_refusal(self):
    class Wrong:
      name = "wrong"

      def read(self, observation):
        return (1.0, 2.0, 3.0, 4.0)

    self.assertEqual(read_or_refuse(Wrong(), Observation()).reason, "returned:tuple")

  def test_an_uninjected_seam_still_raises_rather_than_being_swallowed(self):
    """A missing adapter is a wiring error, not a failed measurement.

    Converting it to an abstention would let a run proceed to the end reporting
    that the readout kept failing, when what happened is that nobody connected
    one.
    """
    with self.assertRaises(NotImplementedError):
      read_or_refuse(InjectedReadout(), Observation())

  def test_a_working_readout_passes_straight_through(self):
    result = read_or_refuse(readout(), marker_observation())
    self.assertTrue(result.ok)


class TestConstantPoseNull(unittest.TestCase):
  """The null reports a pose it never measured, and must fail every real test."""

  def test_it_succeeds_on_an_empty_observation(self):
    result = ConstantPoseReadout((1.0, 2.0, 3.0, 0.4)).read(Observation())
    self.assertTrue(result.ok)
    np.testing.assert_allclose(result.state, (1.0, 2.0, 3.0, 0.4), atol=1e-12)

  def test_it_returns_the_same_pose_however_the_bench_moves(self):
    null = ConstantPoseReadout((150.0, 20.0, PLANE_Z, 0.0))
    real = readout()
    truths = ((150.0, 20.0, 0.0), (160.0, 30.0, 0.4), (140.0, 10.0, -0.6))
    null_errors = []
    real_errors = []
    for truth in truths:
      obs = marker_observation(object_pose=truth)
      expected = np.array([truth[0], truth[1], PLANE_Z, truth[2]])
      null_errors.append(readout_error(null.read(obs).state, expected).mm)
      real_errors.append(readout_error(real.read(obs).state, expected).mm)
    # The null is exactly right on the one pose it was handed and wrong on the
    # others; the real readout is right on all three. A test that only ever
    # planted one pose could not tell these apart, which is the point of the null.
    self.assertAlmostEqual(null_errors[0], 0.0, places=9)
    self.assertGreater(max(null_errors), 10.0)
    self.assertLess(max(real_errors), 1e-6)

  def test_it_satisfies_the_protocol_which_is_why_the_protocol_proves_nothing(self):
    self.assertIsInstance(ConstantPoseReadout(), Readout)
    self.assertIsInstance(readout(), Readout)


if __name__ == "__main__":
  unittest.main()
