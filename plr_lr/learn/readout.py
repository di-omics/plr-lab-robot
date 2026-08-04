"""Turning what a model produced back into a pose the bench can be measured in.

A world model predicts pixels or latents. Nothing downstream of it can act on
either one: the safety envelope bounds millimetres, the goal is a pose, and the
tolerance that decides whether a prediction was right is stated in millimetres
and radians. The readout is the step that crosses that gap, and it is a separate,
named, independently scorable component for one reason: it is fitted often enough
that leaving it inside "the model" hides a probe fit on the very episodes being
scored.

Three things live here, and the middle one is the point:

* :class:`Readout` is the seam. One method, observation in, pose out, plus an
  ``ok`` flag that the screen reads. A readout that cannot measure says so
  instead of returning a plausible pose, because a pose is four ordinary floats
  and a guessed one is indistinguishable from a measured one at every later step.
* :class:`MarkerReadout` is classical, has no weights, and is **actually
  implemented**. It is the declared floor: a world model that does not beat it on
  a task has produced no evidence of value on that task. Shipping it as a stub
  would quietly convert that claim into an untested assertion, so it is a real
  back-projection and a real rigid fit, and the tests recover a planted pose
  through it.
* :class:`ConstantPoseReadout` is the null. It returns a pose it never measured.
  It exists so that a test which passes with it is known to be measuring nothing,
  and it is expected to fail every test that varies the true pose.

:func:`readout_error` carries its units in the name of what it returns, because
the two numbers it produces are millimetres and degrees and adding them is the
error the rest of this package refuses everywhere else.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, NamedTuple, Optional, Protocol, Sequence, Tuple
from typing import runtime_checkable

import numpy as np

from ..vision.camera import CameraIntrinsics, base_from_camera
from ..vision.guided_pick import pixel_to_plane
from ..vision.transforms import Transform
from .types import POSE_FIELDS, Observation

#: Reason strings a readout may give for refusing. A closed set, so that a caller
#: can branch on why the measurement failed rather than on the wording.
NO_MARKERS = "no_markers"
UNDERDETERMINED = "underdetermined"
DEGENERATE = "degenerate"
RESIDUAL_TOO_LARGE = "residual_too_large"
RAY_PARALLEL_TO_PLANE = "ray_parallel_to_plane"
MEASURED = "measured"

REFUSAL_REASONS: Tuple[str, ...] = (
  NO_MARKERS,
  UNDERDETERMINED,
  DEGENERATE,
  RESIDUAL_TOO_LARGE,
  RAY_PARALLEL_TO_PLANE,
)

#: Two markers is the floor for a planar pose. One marker fixes a position and
#: says nothing at all about rotation, and a readout that filled rz with 0.0 in
#: that case would report a wrist angle it never measured.
MIN_MARKERS = 2

#: Below this the observed marker cloud has no usable extent and the rotation it
#: implies is whatever the pixel noise happened to be, in mm.
MIN_SPREAD_MM = 1e-6


class ReadoutError(NamedTuple):
  """Translation error in mm and wrist error in degrees, never summed.

  Named fields rather than a bare tuple because the two are routinely unpacked in
  the wrong order and both are small positive floats, so the swap survives review
  and shows up as a tolerance that is wrong by a factor of about sixty.
  """

  mm: float
  deg: float


@dataclass(frozen=True)
class ReadoutResult:
  """One attempt at measuring a pose, including the attempts that failed.

  ``state`` is all-NaN whenever ``ok`` is False, and that is checked rather than
  documented. The permissive version returns the last good pose, or zeros, or the
  nominal taught pose, and every one of those reads downstream as a measurement
  of the thing that is currently in front of the camera.
  """

  name: str
  ok: bool
  state: np.ndarray  # (4,) x, y, z in mm and rz in rad
  reason: str
  n_markers: int = 0
  residual_mm: float = float("nan")

  def __post_init__(self) -> None:
    if not isinstance(self.name, str) or not self.name:
      raise ValueError("ReadoutResult.name must be a non-empty string")
    try:
      state = np.asarray(self.state, dtype=float).reshape(len(POSE_FIELDS))
    except (TypeError, ValueError) as exc:
      raise ValueError(
        f"ReadoutResult.state must be {len(POSE_FIELDS)} numbers in the "
        f"{list(POSE_FIELDS)} layout"
      ) from exc
    object.__setattr__(self, "state", state)
    if self.ok:
      if not np.all(np.isfinite(state)):
        raise ValueError(
          f"ReadoutResult.ok is True but state is {state.tolist()}. A readout that "
          "succeeded produced four finite numbers; a NaN riding inside a successful "
          "measurement is dropped by every aggregate that touches it."
        )
    elif not bool(np.isnan(state).all()):
      raise ValueError(
        f"ReadoutResult.ok is False but state is {state.tolist()}. A refused "
        "measurement carries no pose. Returning the last good one, or zeros, or the "
        "taught nominal is indistinguishable from having measured it."
      )
    if not isinstance(self.reason, str) or not self.reason:
      raise ValueError("ReadoutResult.reason must be a non-empty string")

  @classmethod
  def refused(cls, name: str, reason: str, n_markers: int = 0) -> "ReadoutResult":
    """The one refusal shape the invariant accepts. Build refusals through this."""
    return cls(
      name=name,
      ok=False,
      state=np.full(len(POSE_FIELDS), np.nan, dtype=float),
      reason=reason,
      n_markers=int(n_markers),
    )


@runtime_checkable
class Readout(Protocol):
  """Maps one observation to a task pose, or declines to.

  Structural, like every other Protocol in this package, so satisfying it says
  only that the attribute names line up. :class:`InjectedReadout` satisfies it
  with nothing injected and raises on the first call.
  """

  name: str

  def read(self, observation: Observation) -> ReadoutResult:
    ...


def readout_error(pred: Sequence[float], true: Sequence[float]) -> ReadoutError:
  """Translation error in mm and wrist error in degrees between two poses.

  Kept as two numbers on purpose. A single combined figure adds radians to
  millimetres, and at that point a full radian of wrist error counts for about as
  much as one millimetre of travel while the result is labelled in mm.

  Non-finite input raises rather than propagating a NaN. A refused readout has no
  pose, so asking for its error is a question with no answer, and a NaN that
  reaches a median or a mean is silently dropped by numpy and shrinks the sample
  without shrinking the reported n.
  """
  p = np.asarray(pred, dtype=float).reshape(len(POSE_FIELDS))
  t = np.asarray(true, dtype=float).reshape(len(POSE_FIELDS))
  if not np.all(np.isfinite(p)) or not np.all(np.isfinite(t)):
    raise ValueError(
      "readout_error needs two finite poses. A refused readout carries an all-NaN "
      "state and has no error to report; count it as a refusal instead of feeding "
      "a NaN into an aggregate that will drop it without adjusting n."
    )
  mm = float(np.linalg.norm(p[:3] - t[:3]))
  drz = float(np.arctan2(np.sin(p[3] - t[3]), np.cos(p[3] - t[3])))
  return ReadoutError(mm=mm, deg=float(math.degrees(abs(drz))))


def _planar_rigid_fit(
  model: np.ndarray,
  observed: np.ndarray,
) -> Tuple[float, float, float, float]:
  """Least-squares (x, y, rz) taking ``model`` points onto ``observed`` points.

  The planar special case of Kabsch, written out rather than pulled from a
  library so the whole readout stays inside numpy. Rotation first from the
  cross- and dot-product sums, then translation from the centroids, which is the
  ordering that keeps the fit exact for a noise-free correspondence.
  """
  m_bar = model.mean(axis=0)
  o_bar = observed.mean(axis=0)
  m = model - m_bar
  o = observed - o_bar
  dot = float(np.sum(m[:, 0] * o[:, 0] + m[:, 1] * o[:, 1]))
  cross = float(np.sum(m[:, 0] * o[:, 1] - m[:, 1] * o[:, 0]))
  rz = float(np.arctan2(cross, dot))
  cos_rz, sin_rz = math.cos(rz), math.sin(rz)
  rotated = np.stack(
    [
      model[:, 0] * cos_rz - model[:, 1] * sin_rz,
      model[:, 0] * sin_rz + model[:, 1] * cos_rz,
    ],
    axis=1,
  )
  translation = o_bar - rotated.mean(axis=0)
  residual = float(np.sqrt(np.mean(np.sum((rotated + translation - observed) ** 2, axis=1))))
  return float(translation[0]), float(translation[1]), rz, residual


class MarkerReadout:
  """Classical planar pose readout from labelled markers. The declared floor.

  Back-projects every recognised detection onto the deck plane through the
  eye-in-hand camera model, then fits the rigid planar transform that best takes
  the labware's own known marker layout onto where those markers actually landed.
  Thirty lines of geometry, no weights, and it runs on a laptop in microseconds.

  It is here to be beaten. A video world model scored on a small synthetic render
  is competing against this, and if it does not win, the honest reading is that
  the task measured perception difficulty that was not there rather than that the
  model was good. That claim is only worth making because this is a working
  implementation: a stub in this slot would make the floor unmeasurable and the
  comparison rhetorical.

  Every way this can fail returns ``ok=False`` with a reason from a closed set:

  * fewer than two recognised markers, because one marker fixes a position and
    determines no rotation at all, and filling ``rz`` with zero there reports a
    wrist angle that was never measured;
  * a marker cloud with no extent, where the implied rotation is pixel noise;
  * a fit whose residual exceeds ``max_residual_mm``, which is what a wrong
    correspondence, a bumped camera, or the wrong labware on the deck looks like;
  * a view ray parallel to the deck plane, which has no intersection to report.

  Args:
    intrinsics: pinhole model of the wrist camera.
    cam_from_tool: hand-eye transform. Wrong by a few millimetres and every pose
      here is wrong by a few millimetres, which is why this is an argument rather
      than a default.
    plane_z: height of the deck plane the markers sit on, mm in the base frame.
    model: label -> (x, y) position of that marker in the labware's own frame,
      mm, at the labware's nominal zero rotation.
    min_confidence: detections below this are ignored rather than downweighted.
    max_residual_mm: how far the fitted layout may sit from the observed markers
      before the measurement is refused.
  """

  name = "marker"

  def __init__(
    self,
    intrinsics: CameraIntrinsics,
    cam_from_tool: Transform,
    plane_z: float,
    model: Dict[str, Tuple[float, float]],
    min_confidence: float = 0.5,
    max_residual_mm: float = 2.0,
  ) -> None:
    layout = {str(label): (float(xy[0]), float(xy[1])) for label, xy in dict(model).items()}
    if len(layout) < MIN_MARKERS:
      raise ValueError(
        f"a marker model needs at least {MIN_MARKERS} labelled points and this one has "
        f"{len(layout)}. A single-point model can never determine a rotation, so the "
        "readout built from it would report an rz it did not measure on every frame."
      )
    points = np.array([layout[label] for label in sorted(layout)], dtype=float)
    if not np.all(np.isfinite(points)):
      raise ValueError("marker model positions must all be finite")
    if float(np.max(np.linalg.norm(points - points.mean(axis=0), axis=1))) <= MIN_SPREAD_MM:
      raise ValueError(
        "every marker in this model sits at the same place, so no arrangement of "
        "them can determine a rotation"
      )
    if not math.isfinite(float(plane_z)):
      raise ValueError("plane_z must be finite")
    if not 0.0 <= float(min_confidence) <= 1.0:
      raise ValueError("min_confidence must be in [0, 1]")
    if not float(max_residual_mm) > 0.0 or not math.isfinite(float(max_residual_mm)):
      raise ValueError("max_residual_mm must be positive and finite")
    self.intrinsics = intrinsics
    self.cam_from_tool = cam_from_tool
    self.plane_z = float(plane_z)
    self.model = layout
    self.min_confidence = float(min_confidence)
    self.max_residual_mm = float(max_residual_mm)

  def _best_per_label(self, observation: Observation) -> Dict[str, Tuple[float, float]]:
    """Keep the most confident detection per known label, drop everything else.

    Two detections carrying the same label are two claims about one marker.
    Averaging them assumes both are of the same thing, which is exactly what a
    duplicate detection puts in doubt, so the weaker claim is dropped instead.
    """
    best: Dict[str, Tuple[float, float]] = {}
    scores: Dict[str, float] = {}
    for det in observation.detections:
      if det.label not in self.model:
        continue
      confidence = float(det.confidence)
      if not math.isfinite(confidence) or confidence < self.min_confidence:
        continue
      if det.label not in scores or confidence > scores[det.label]:
        scores[det.label] = confidence
        best[det.label] = (float(det.uv[0]), float(det.uv[1]))
    return best

  def read(self, observation: Observation) -> ReadoutResult:
    """Measure the labware pose, or refuse and say which way it failed."""
    seen = self._best_per_label(observation)
    if not seen:
      return ReadoutResult.refused(self.name, NO_MARKERS, 0)
    if len(seen) < MIN_MARKERS:
      return ReadoutResult.refused(self.name, UNDERDETERMINED, len(seen))

    base_from_tool = Transform.from_euler(
      0.0, 0.0, float(observation.rz), t=(observation.x, observation.y, observation.z)
    )
    base_from_cam = base_from_camera(base_from_tool, self.cam_from_tool)

    labels = sorted(seen)
    model_points = np.array([self.model[label] for label in labels], dtype=float)
    observed = np.empty((len(labels), 2), dtype=float)
    for row, label in enumerate(labels):
      try:
        point = pixel_to_plane(seen[label], self.intrinsics, base_from_cam, self.plane_z)
      except ValueError:
        return ReadoutResult.refused(self.name, RAY_PARALLEL_TO_PLANE, len(labels))
      if not np.all(np.isfinite(point)):
        return ReadoutResult.refused(self.name, RAY_PARALLEL_TO_PLANE, len(labels))
      observed[row] = point[:2]

    spread = float(np.max(np.linalg.norm(observed - observed.mean(axis=0), axis=1)))
    if spread <= MIN_SPREAD_MM:
      return ReadoutResult.refused(self.name, DEGENERATE, len(labels))

    x, y, rz, residual = _planar_rigid_fit(model_points, observed)
    if not math.isfinite(residual) or residual > self.max_residual_mm:
      return ReadoutResult(
        name=self.name,
        ok=False,
        state=np.full(len(POSE_FIELDS), np.nan, dtype=float),
        reason=RESIDUAL_TOO_LARGE,
        n_markers=len(labels),
        residual_mm=residual,
      )
    return ReadoutResult(
      name=self.name,
      ok=True,
      state=np.array([x, y, self.plane_z, rz], dtype=float),
      reason=MEASURED,
      n_markers=len(labels),
      residual_mm=residual,
    )

  def __repr__(self) -> str:
    return f"MarkerReadout({len(self.model)} markers, plane_z={self.plane_z:.1f})"


class InjectedReadout:
  """The seam a learned pixel-to-state or latent-to-state readout plugs into.

  Empty until something is injected, and it raises when called empty for the same
  reason the predictor seam does: naming a checkpoint establishes no mapping from
  whatever that model emits onto millimetres and radians in this arm's base
  frame, and a readout that silently returns zeros looks exactly like one that
  works.

  A readout fitted on data is part of the model. Whoever injects one is expected
  to declare it as a fitted component on the adapter, because the predictor's own
  training declaration says nothing about a probe bolted onto its output.
  """

  def __init__(
    self,
    name: str = "injected",
    readout: Optional[Callable[[Observation], ReadoutResult]] = None,
    checkpoint: Optional[Any] = None,
  ) -> None:
    if readout is not None and not callable(readout):
      raise TypeError("readout must be a synchronous callable or None")
    if not isinstance(name, str) or not name:
      raise ValueError("name must be a non-empty string")
    self.name = name
    self.readout = readout
    self.checkpoint = checkpoint

  def read(self, observation: Observation) -> ReadoutResult:
    if self.readout is None:
      note = " Checkpoint metadata was recorded." if self.checkpoint is not None else ""
      raise NotImplementedError(
        f"the {self.name} readout has no callable.{note} A checkpoint or an installed "
        "package alone is not a readout: it establishes no mapping from predicted "
        "pixels or latents onto millimetres and radians in this arm's base frame. "
        "Inject a synchronous read(observation) -> ReadoutResult."
      )
    result = self.readout(observation)
    if not isinstance(result, ReadoutResult):
      raise ValueError(
        f"the {self.name} readout returned {type(result).__name__}, not a "
        "ReadoutResult. A bare pose has no ok flag, so a failed measurement would "
        "reach the screen wearing the shape of a successful one."
      )
    return result

  def __repr__(self) -> str:
    state = "empty" if self.readout is None else "injected"
    return f"InjectedReadout({self.name!r}, {state})"


class ConstantPoseReadout:
  """Reports the same pose for every frame. A null, and never a measurement.

  It succeeds on every call, which is the whole point: a test suite that stays
  green with this substituted for a real readout has established nothing about
  the readout, and a screen that admits on its output is admitting on a number
  that never looked at the bench.

  This is also the honest model of the most common real failure, a readout that
  falls back to the taught nominal pose whenever detection fails. That fallback
  is invisible in the pose and fatal in the arm.
  """

  name = "constant-pose-null"

  def __init__(self, pose: Sequence[float] = (0.0, 0.0, 0.0, 0.0)) -> None:
    p = np.asarray(pose, dtype=float).reshape(len(POSE_FIELDS))
    if not np.all(np.isfinite(p)):
      raise ValueError("ConstantPoseReadout pose must be finite")
    self.pose = p

  def read(self, observation: Observation) -> ReadoutResult:
    return ReadoutResult(
      name=self.name,
      ok=True,
      state=self.pose.copy(),
      reason=MEASURED,
      n_markers=0,
      residual_mm=0.0,
    )

  def __repr__(self) -> str:
    return f"ConstantPoseReadout({np.round(self.pose, 3).tolist()})"


def read_or_refuse(readout: Any, observation: Observation) -> ReadoutResult:
  """Call a readout and turn any raised failure into a refusal the screen can read.

  The screen has one question, is the readout ok, and an exception escaping past
  it aborts the run instead of producing the abstention the design calls for. A
  readout that raises has still failed to measure, so it is recorded as a refusal
  with the exception type as the reason rather than being allowed to unwind
  through the arm loop.
  """
  name = str(getattr(readout, "name", type(readout).__name__))
  try:
    result = readout.read(observation)
  except NotImplementedError:
    raise
  except Exception as exc:  # noqa: BLE001 - deliberate: any failure is a refusal
    return ReadoutResult.refused(name, f"raised:{type(exc).__name__}")
  if not isinstance(result, ReadoutResult):
    return ReadoutResult.refused(name, f"returned:{type(result).__name__}")
  return result
