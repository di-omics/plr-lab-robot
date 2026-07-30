"""Forward and inverse kinematics, and the refusals that keep them honest.

WHY THIS MODULE EXISTS
----------------------
``SimulationArmBackend`` tracks ``_joints`` and ``_cartesian`` as two independent
variables. ``move_to`` with a joint dict never touches the Cartesian pose;
``move_to`` with a Cartesian pose never touches the joints. The two can therefore
drift apart without limit and nothing notices, because nothing in the backend
knows the geometry that ties them together. That is the difference between
exercising an arm API and being able to tell an arm where to go. This module
supplies the missing tie: a link geometry, a forward map, an inverse search, and
-- more importantly -- a set of refusals for the cases where an answer would be a
guess dressed up as a number.

THE ASYMMETRY THAT SHAPES EVERYTHING BELOW
------------------------------------------
Forward kinematics is exact. Given joint angles and link geometry the tool pose
is determined; there is one answer and it is a matter of multiplying matrices.
The only way to get it wrong is to have the wrong geometry, which is why
``forward`` refuses outright on a model whose D-H parameters were not confirmed
from a first-party source. An arm driven on invented link lengths moves
confidently to the wrong place, and it does so with no error and no warning.

Inverse kinematics is not exact and is not even single-valued. A general 6R arm
has at most 16 discrete solutions for a pose (Elias and Wen, IK-Geo,
arXiv:2211.05737); a Pieper-condition arm collapses to eight; a 7-DOF arm has a
one-dimensional continuum per branch, because six equations in seven unknowns
generically cut out a curve. So ``inverse`` returns every solution it finds
rather than one, and picking among them is a separate, named policy decision
made by ``select``. That separation is the point. The wrong branch reaches the
same Cartesian point by swinging the elbow through the deck, and no property of
the equation ``forward(q) == T`` distinguishes the two -- both satisfy it
exactly. Ranking them requires criteria the kinematics does not contain, and the
industry concedes this: ABB stores a configuration quadrant with each target,
Corke's toolbox takes a config string. The branch is an authored input, not a
derived result.

WHAT A SOLUTION DOES AND DOES NOT ASSERT
----------------------------------------
A returned ``Solution`` asserts exactly two things: forward kinematics of those
joints reproduces the target within the stated residual, and the joints were
checked against a declared joint-limit row. It asserts nothing about the path
from wherever the arm is now to those joints, because no path appears anywhere
in the problem statement. A joint-interpolated move drives all joints
simultaneously and the tool traces whatever curve the geometry maps that
joint-space straight line to. If the chosen solution sits in a different branch
from the current configuration, the excursion sweeps a large volume, and the
volume it sweeps is what intersects the bench. ``reachable`` therefore says
nothing at all about reachability of a path, and says so in its own docstring
and in ``Reachability.uncovered_gates``.

ACCURACY IS NOT REPEATABILITY
-----------------------------
This is the distinction datasheets invite people to get wrong, and it is the one
that matters most for a lab. ISO 9283 defines pose repeatability RP against the
barycentre of the robot's own attained poses -- the defining quantity contains no
term referencing the commanded pose. RP is scatter about wherever the arm
actually goes and is mathematically incapable of bounding how far that place is
from where it was told to go. Pose accuracy AP as vendors publish it is measured
against a hand-TAUGHT position, not a COMPUTED one; ABB's IRB 140 datasheet
prints AP = 0.02 mm alongside RP = 0.03 mm, an "accuracy" smaller than the
repeatability, which is only possible because the datum is the taught point.

Inverse kinematics produces computed poses. A computed pose inherits absolute
accuracy error that the repeatability figure does not cover: measured
uncalibrated absolute errors run one to three orders of magnitude worse than
datasheet repeatability, and the spread between two arms of the same make on the
same rig was a factor of 9.4. So ``Pose`` carries whether it was taught or
computed, and asking a computed pose for a positional uncertainty raises
``UnquantifiedAccuracy`` rather than handing back the repeatability number.

CONVENTION DECLARED
-------------------
This module consumes MODIFIED (Craig) D-H, with the link transform

    T_i = Rx(alpha_i) Tx(a_i) Rz(theta_i) Tz(d_i),   theta_i = q_i + offset_i

UFACTORY publishes both a modified and a standard table for the same arm and the
two are NOT interchangeable row for row -- alpha and a shift between rows. Feeding
a standard-D-H table to this module produces a wrong arm that computes cleanly.
``DHConvention`` exists so the table has to say which it is.

SOURCES FOR THE SHIPPED GEOMETRY
--------------------------------
xArm 6 and xArm 7 modified-D-H tables, offset constants, and joint limits are
triple-sourced and agree exactly across the UFACTORY docs site, the xArm User
Manual V2.0.0 Appendix 8, and the official xarm_ros kinematics yaml; the SDK
radian joint limits (xArm-Python-SDK v1.18.5, xarm/core/config/x_config.py) match
the manual's degree figures to three decimal places. Two independent self-checks
on the tables as transcribed here are asserted by the test suite: forward
kinematics maximised over the joint limits reproduces the separately derived
reach figures (763.87 mm and 772.05 mm flange radius) to the digit, and the
maximum flange height comes out at exactly shoulder height plus that radius,
which is what full extension means.

The published geometry is NOMINAL. It is the design, not the as-built unit.
UFACTORY ships a readback of per-unit calibrated kinematics over TCP port 502
precisely because nominal D-H is not accurate to the +/-0.1 mm repeatability
spec, and that readback is the only correct source of truth for a specific arm.
``Provenance.NOMINAL_FIRST_PARTY`` is computable; it is not a claim of accuracy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

# -- refusals ------------------------------------------------------------------


class KinematicsRefusal(RuntimeError):
  """Base for every refusal in this module.

  A refusal is not a failure to compute. It is a computation whose result is
  that no trustworthy answer exists, which is a different and more useful thing
  than a number with no provenance.
  """


class UnverifiedGeometry(KinematicsRefusal):
  """The model's link geometry was not confirmed from a first-party source."""


class UnknownDeviceType(KinematicsRefusal):
  """No joint-limit row is declared for this controller device type."""


class NotThisArm(KinematicsRefusal):
  """The device type belongs to a different arm than the axis count suggests."""


class OutsideEnvelope(KinematicsRefusal):
  """The pose lies outside a geometric bound derived from the joint limits."""


class SingularityRefusal(KinematicsRefusal):
  """Every solution found sits too close to a Jacobian rank deficiency."""


class RedundancyUnresolved(KinematicsRefusal):
  """A redundant arm was asked for "all" solutions, which is a continuum."""


class NoSolution(KinematicsRefusal):
  """The multi-start search converged on nothing for this pose."""


class AmbiguousSelection(KinematicsRefusal):
  """The stated selection policy does not pick a unique solution."""


class UnquantifiedAccuracy(KinematicsRefusal):
  """The requested error bound is not covered by any figure this model carries."""


# -- enums ---------------------------------------------------------------------


class DHConvention(str, Enum):
  """Which D-H table shape a row set is written in.

  UFACTORY publishes both for every arm and they describe the same hardware, so
  the choice is pure convention -- but the two tables are not interchangeable row
  for row, and mixing them silently yields a different robot.
  """

  MODIFIED_CRAIG = "modified_craig"
  STANDARD = "standard"

  @property
  def alpha_leads_the_joint(self) -> bool:
    """True when row i's alpha and a precede joint i's rotation in the transform.

    This is the whole difference. Under Craig's modified form the twist and link
    length on row i apply before theta_i; under the standard form they apply
    after. Swapping the tables shifts alpha and a by one row, which is why a
    standard table fed to a modified consumer produces a plausible-looking arm
    that is not the one on the bench.
    """
    return self is DHConvention.MODIFIED_CRAIG


class Provenance(str, Enum):
  """Where a model's link geometry came from, and therefore whether to use it."""

  NOMINAL_FIRST_PARTY = "nominal_first_party"
  PER_UNIT_CALIBRATED = "per_unit_calibrated"
  UNVERIFIED = "unverified"

  @property
  def computable(self) -> bool:
    """Whether kinematics may be evaluated on this geometry at all.

    Unverified geometry is refused rather than approximated. There is no
    degraded mode: link lengths are either measured or invented, and an arm
    driven on invented ones has no error signal to notice.
    """
    return self is not Provenance.UNVERIFIED


class PoseOrigin(str, Enum):
  """How a pose came to exist, which decides which error figure applies to it."""

  TAUGHT = "taught"
  COMPUTED = "computed"

  @property
  def bounded_by_repeatability(self) -> bool:
    """Whether the vendor repeatability figure bounds this pose's error.

    Only for a taught pose. Repeatability is scatter about the arm's own
    barycentre of attained poses and is measured by returning to a hand-taught
    point. A computed pose -- anything inverse kinematics produced -- carries
    absolute accuracy error instead, which the repeatability figure does not
    cover and which vendors mostly do not publish.
    """
    return self is PoseOrigin.TAUGHT


class CartesianLimitSource(str, Enum):
  """Which first-party Cartesian box a model declares, because they disagree.

  The xArm User Manual V2.0.0 common spec table gives X and Y +/-700 mm and Z
  -400 to 951.5 mm. The SDK's TCP_LIMITS for the same standard variants gives
  +/-750 mm and -400 to 1000 mm. That is 50 mm horizontally and 48.5 mm
  vertically between two UFACTORY sources, and neither is derivable from the
  other. Pick deliberately; the model records which.
  """

  MANUAL_COMMON_SPEC = "manual_common_spec"
  SDK_TCP_LIMITS = "sdk_tcp_limits"

  @property
  def bounds_the_geometry(self) -> bool:
    """Always False. Neither box is a kinematic bound on where the flange goes.

    Forward kinematics maximised over the published joint limits puts the xArm 6
    flange 763.87 mm out and the xArm 7 flange 772.05 mm out, past both boxes. A
    workspace check written as "the arm cannot exceed this" is simply false; use
    ``ReachEnvelope`` for the geometric bound and these boxes as declared
    operating limits, which is a different claim.
    """
    return False


class SelectionPolicy(str, Enum):
  """Named rules for choosing one solution out of many. There is no default.

  Nothing in ``forward(q) == T`` ranks its own solutions -- all of them satisfy
  it exactly. Every ranking imports a criterion from outside the kinematics, and
  the criteria conflict pairwise: the solution furthest from a singularity is
  routinely not the one nearest the current joint state. So the rule is stated
  at the call site or no solution is returned.
  """

  NEAREST_WEIGHTED = "nearest_weighted"
  DECLARED_CONFIGURATION = "declared_configuration"
  MAX_LIMIT_MARGIN = "max_limit_margin"
  MAX_SINGULAR_VALUE = "max_singular_value"

  @property
  def depends_on_current_state(self) -> bool:
    """Whether the choice is a function of where the arm happens to be.

    A state-dependent rule makes the result path-dependent and non-cyclic: a
    closed end-effector path need not return the arm to its starting joint
    configuration, so joint-space drift accumulates over repeated cycles (Klein
    and Huang, 1983). For a lab arm running the same pick-place cycle thousands
    of times, whether it drifts into a joint limit on run 400 is decided by this
    property and not by the geometry.
    """
    return self is SelectionPolicy.NEAREST_WEIGHTED


class Shoulder(str, Enum):
  FRONT = "front"
  BACK = "back"


class Elbow(str, Enum):
  UP = "up"
  DOWN = "down"


class Wrist(str, Enum):
  NOFLIP = "noflip"
  FLIP = "flip"

  @property
  def tool_orientation_preserved(self) -> bool:
    """Whether the tool frame survives the branch change intact.

    A wrist flip is a 180 deg rotation that reaches the same Cartesian pose, and
    it is commonly described as free because a symmetric two-finger grasp does
    not care. A pipette, a filled open vessel, a keyed connector, or anything
    with a tubing tail does care. For those the wrist branch is a hard
    constraint, not a tie-break.
    """
    return self is Wrist.NOFLIP


class PathGate(str, Enum):
  """Validity gates a pose-level solution does not cover.

  Listed as an enum so the omission is enumerable rather than implied. Each
  member is a check that must pass before a solution becomes a motion, and none
  of them is a function of the target pose alone.
  """

  SWEPT_VOLUME = "swept_volume"
  SINGULAR_LOCUS = "singular_locus"
  BRANCH_CONTINUITY = "branch_continuity"
  TOOL_ORIENTATION_IN_TRANSIT = "tool_orientation_in_transit"

  @property
  def checked_by_ik(self) -> bool:
    """Always False, for every member. That is the point of the enum.

    SWEPT_VOLUME: every link plus payload over the whole interpolated path stays
    in the safe half-space, checked densely and not just at the endpoints.
    SINGULAR_LOCUS: the smallest singular value stays above a floor along the
    whole path, not only at the ends. For a noncuspidal arm a branch change
    requires leaving the aspect, so "dives through the table" and "passes
    through a singularity" are often the same event. BRANCH_CONTINUITY: the
    configuration matches the declared policy so the motion is reproducible run
    to run. TOOL_ORIENTATION_IN_TRANSIT: an open vessel stays upright the whole
    way, which is a constraint on the path and invisible to a pose query.
    """
    return False


# -- geometry rows and limits ---------------------------------------------------


@dataclass(frozen=True)
class DHRow:
  """One link row. Lengths in mm, angles in rad. ``offset_rad`` is added to q.

  The offset carries the difference between the mathematical zero of the D-H
  chain and the mechanical zero the controller reports. It is not cosmetic: the
  xArm 6 URDF bakes the same quantity into link geometry instead, so mixing a
  URDF-zero joint vector with this table without applying the offsets puts
  joints 2 and 3 off by 79.34995 deg.
  """

  theta_rad: float
  d_mm: float
  alpha_rad: float
  a_mm: float
  offset_rad: float = 0.0

  def theta_for(self, q_rad: float) -> float:
    return self.theta_rad + q_rad + self.offset_rad


@dataclass(frozen=True)
class LimitViolation:
  """One axis outside its declared range. Carries the amount, not just the fact."""

  axis: int  # 1-indexed, matching the manual and the SDK
  value_rad: float
  lower_rad: float
  upper_rad: float

  def overshoot_rad(self) -> float:
    if self.value_rad < self.lower_rad:
      return self.lower_rad - self.value_rad
    return self.value_rad - self.upper_rad

  def describe(self) -> str:
    return (
      f"axis {self.axis}: {math.degrees(self.value_rad):.4f} deg outside "
      f"[{math.degrees(self.lower_rad):.4f}, {math.degrees(self.upper_rad):.4f}] deg "
      f"by {math.degrees(self.overshoot_rad()):.4f} deg"
    )


@dataclass(frozen=True)
class JointLimits:
  """A declared joint range row, keyed by controller device type.

  Joint limits are not one table per model. They vary by hardware variant, and
  keying on axis count alone silently loads the wrong envelope. The controller
  reports its own device type; read it and select the matching row.

  Note also that the SDK's own client-side check is not a safety guarantee: it
  is skipped entirely when the device type is absent from its table, and when it
  does run it tolerates 0.1 deg of overshoot on each side. This module checks
  strictly, with no slack, because the slack is an implementation tolerance for
  float noise and not a published mechanical allowance.
  """

  device_type: int
  label: str
  lower_rad: Tuple[float, ...]
  upper_rad: Tuple[float, ...]
  source: str = ""
  # Axes (1-indexed) whose range was reported only as a difference from the base
  # row for this arm, rather than stated axis by axis in the source.
  overridden_axes: Tuple[int, ...] = ()

  def dof(self) -> int:
    return len(self.lower_rad)

  def fully_stated(self) -> bool:
    """Whether every axis of this row was read out individually from the source."""
    return not self.overridden_axes

  def violations(self, joints_rad: Sequence[float]) -> Tuple[LimitViolation, ...]:
    out: List[LimitViolation] = []
    for i, q in enumerate(joints_rad):
      lo, hi = self.lower_rad[i], self.upper_rad[i]
      if q < lo or q > hi:
        out.append(LimitViolation(axis=i + 1, value_rad=float(q), lower_rad=lo, upper_rad=hi))
    return tuple(out)

  def contains(self, joints_rad: Sequence[float]) -> bool:
    return not self.violations(joints_rad)

  def margins_rad(self, joints_rad: Sequence[float]) -> Tuple[float, ...]:
    return tuple(
      min(q - self.lower_rad[i], self.upper_rad[i] - q) for i, q in enumerate(joints_rad)
    )

  def min_margin_rad(self, joints_rad: Sequence[float]) -> float:
    return min(self.margins_rad(joints_rad))


@dataclass(frozen=True)
class ReachEnvelope:
  """A geometric bound on flange position, DERIVED rather than published.

  The published 700 mm reach figure is not the geometric maximum and must not be
  used as a kinematic bound: it cannot be reconciled with what the published D-H
  actually reaches, and the measurement convention behind it (flange face,
  flange centre, tool plate, safety radius) is not stated anywhere first-party.
  The numbers here come from maximising flange radius over the joint limits of
  one specific device type using the D-H in this module, so they are reproducible
  but they are a computation, not a vendor figure.

  They ignore self-collision, so parts of the enclosed volume are not attainable.
  This is a necessary condition for reachability and never a sufficient one.
  """

  max_flange_radius_mm: float
  max_flange_height_mm: float
  shoulder_height_mm: float
  derived_for_device_type: int
  full_extension_witness_rad: Tuple[float, ...] = ()
  note: str = ""

  def applies_to(self, device_type: int) -> bool:
    """Whether this envelope was derived over the limits of that device type.

    A mirrored or narrowed variant has a different envelope. Reusing one across
    variants would reinstate exactly the hard-coded-limits trap the device-type
    keying exists to avoid.
    """
    return device_type == self.derived_for_device_type

  def shoulder_distance_mm(self, xyz: Sequence[float]) -> float:
    dx, dy = float(xyz[0]), float(xyz[1])
    dz = float(xyz[2]) - self.shoulder_height_mm
    return math.sqrt(dx * dx + dy * dy + dz * dz)

  def excludes(self, xyz: Sequence[float]) -> Optional[str]:
    """Return why the point is provably out of reach, or None if it may be in.

    The spherical test is the load-bearing one: joint 1 rotates about the base z
    axis through the shoulder point, so no joint setting moves the flange
    further from that point than the maximum found over the limits. That the
    maximum flange height equals shoulder height plus this radius is the
    self-check that the two agree.
    """
    reach = self.shoulder_distance_mm(xyz)
    if reach > self.max_flange_radius_mm:
      return (
        f"{reach:.3f} mm from the shoulder point exceeds the derived "
        f"{self.max_flange_radius_mm:.2f} mm maximum"
      )
    if float(xyz[2]) > self.max_flange_height_mm:
      return (
        f"z = {float(xyz[2]):.3f} mm exceeds the derived "
        f"{self.max_flange_height_mm:.2f} mm maximum flange height"
      )
    return None


@dataclass(frozen=True)
class CartesianBox:
  """A declared Cartesian operating box. Not a kinematic bound -- see the source."""

  source: CartesianLimitSource
  x_mm: Tuple[float, float]
  y_mm: Tuple[float, float]
  z_mm: Tuple[float, float]

  def contains(self, xyz: Sequence[float]) -> bool:
    return (
      self.x_mm[0] <= xyz[0] <= self.x_mm[1]
      and self.y_mm[0] <= xyz[1] <= self.y_mm[1]
      and self.z_mm[0] <= xyz[2] <= self.z_mm[1]
    )


# -- policies -------------------------------------------------------------------


@dataclass(frozen=True)
class SingularityPolicy:
  """A stated proximity threshold. Both numbers are choices, not measurements.

  ``characteristic_length_mm`` scales the rotational rows of the Jacobian so the
  singular values of a matrix that mixes mm/rad with rad/rad are commensurable.
  There is no physically correct value; it is the lever arm at which you care
  about orientation error as much as position error.

  ``min_singular_value`` is in mm per rad after that scaling. The joint-rate gain
  in the degenerate direction is its reciprocal, so at the default of 1.0 the arm
  needs 1 rad/s -- roughly a third of the published 180 deg/s maximum joint speed
  -- to deliver 1 mm/s of tool motion in the worst direction, against a published
  1 m/s maximum end-effector speed. Below that the controller cannot deliver what
  is asked and what happens instead is saturation and a lurch, not a smooth
  degradation. ``singularity_floor_for_speed`` derives a stricter floor from a
  task speed if you have one.
  """

  min_singular_value: float = 1.0
  characteristic_length_mm: float = 100.0

  def describe(self) -> str:
    return (
      f"sigma_min >= {self.min_singular_value} mm/rad at a characteristic length "
      f"of {self.characteristic_length_mm} mm (both are declared choices)"
    )


@dataclass(frozen=True)
class RedundancyLock:
  """A declared value for one joint of a redundant arm, 1-indexed.

  Locking a joint is a redundancy parameterization and it restores a finite
  solution count, but it does not come free. Any parameterization of a 7R's
  redundancy has an unavoidable algorithmic singularity -- that result is
  topological, not a property of a particular scheme -- and near one the elbow
  swings hard for small task motions. Locking a joint puts that singularity
  wherever the task needs the locked axis.
  """

  axis: int
  value_rad: float


# -- poses, configurations, solutions -------------------------------------------


@dataclass(frozen=True)
class Pose:
  """A rigid FLANGE pose plus the one fact that decides which error figure applies.

  This is the tool plate, not the gripper tip. No tool transform lives in a D-H
  table, so the offset from flange to the point that actually touches a plate is
  the caller's to apply and to measure. Treating a flange pose as a grasp pose is
  a fixed error of exactly the tool length, in a direction that rotates with the
  wrist, which is why it does not look like a calibration error.

  ``matrix`` is 16 floats, row-major, translation in mm. It is stored flat so the
  dataclass stays frozen and hashable; ``as_matrix`` rebuilds the array.
  """

  matrix: Tuple[float, ...]
  origin: PoseOrigin = PoseOrigin.COMPUTED

  @classmethod
  def from_matrix(cls, m, origin: PoseOrigin = PoseOrigin.COMPUTED) -> "Pose":
    a = np.asarray(m, dtype=float).reshape(4, 4)
    return cls(tuple(float(v) for v in a.ravel()), origin)

  @classmethod
  def taught(cls, m) -> "Pose":
    """Build a pose the arm was driven to by hand and told to remember."""
    return cls.from_matrix(m, origin=PoseOrigin.TAUGHT)

  def as_matrix(self) -> np.ndarray:
    return np.asarray(self.matrix, dtype=float).reshape(4, 4)

  def position_mm(self) -> Tuple[float, float, float]:
    m = self.matrix
    return (m[3], m[7], m[11])

  def rotation(self) -> np.ndarray:
    return self.as_matrix()[:3, :3]

  def radius_mm(self) -> float:
    m = self.matrix
    return math.hypot(m[3], m[7])

  def uncertainty_mm(self, model: "KinematicModel") -> float:
    """Positional error bound for this pose, or a refusal saying why there is none.

    A taught pose is bounded by the arm's repeatability, which is what
    repeatability measures: returning to a point that was demonstrated. A
    computed pose is not. Handing back the repeatability figure for a computed
    pose would be the exact error this module exists to prevent, so it raises.
    """
    if not self.origin.bounded_by_repeatability:
      lo, hi = ABSOLUTE_TO_REPEATABILITY_RATIO_RANGE
      raise UnquantifiedAccuracy(
        "a computed pose is not bounded by the repeatability figure. Repeatability "
        "is scatter about the arm's own attained poses; this pose was produced by "
        "inverse kinematics from nominal link geometry and carries absolute "
        f"accuracy error instead. Measured uncalibrated absolute error runs {lo:.0f}x "
        f"to {hi:.0f}x the datasheet repeatability across published laser-tracker "
        "studies, varies by a factor of nine between two arms of the same make, and "
        "is not published per unit. Read the per-unit calibrated kinematics off the "
        "controller, or teach the pose."
      )
    if model.repeatability_mm is None:
      raise UnquantifiedAccuracy(
        f"{model.name} declares no repeatability figure. The xArm User Manual "
        "V2.0.0 states +/-0.1 mm in the common spec table and in the xArm 6 "
        "section, and omits the line entirely from the xArm 7 section. This "
        "module does not carry a figure across from a different model."
      )
    return model.repeatability_mm


@dataclass(frozen=True)
class Configuration:
  """Which discrete branch a solution sits in, plus any locked redundancy.

  The three-way split is a derived mathematical fact for Pieper-condition arms:
  the geometric decomposition branches quadratically three times, giving
  2 x 2 x 2. The LABELS are not. ABB names configurations by axis quadrant, KUKA
  by status and turn bits, Corke by an l/r, u/d, n/f string, and none of them
  means the same joint sign convention on another controller. Treat these as
  local names for local branches.
  """

  shoulder: Shoulder
  elbow: Elbow
  wrist: Wrist
  redundancy_rad: Optional[float] = None

  def label(self) -> str:
    base = f"{self.shoulder.value}/{self.elbow.value}/{self.wrist.value}"
    if self.redundancy_rad is None:
      return base
    return f"{base}@{math.degrees(self.redundancy_rad):.3f}deg"

  def matches(self, other: "Configuration") -> bool:
    """Branch equality, ignoring the redundancy parameter.

    Redundancy is a continuous coordinate and comparing it for equality would
    make a declared configuration unmatchable in floating point.
    """
    return (
      self.shoulder is other.shoulder
      and self.elbow is other.elbow
      and self.wrist is other.wrist
    )


@dataclass(frozen=True)
class Solution:
  """One joint vector that reaches the target, carrying everything against it.

  A solution outside the joint limits is not a solution, but it is returned all
  the same -- flagged -- because dropping it silently hides the fact that the
  pose is only reachable by a branch the hardware cannot enter. Callers filter
  with ``usable``; nothing filters behind their back.
  """

  joints_rad: Tuple[float, ...]
  configuration: Configuration
  limits: JointLimits
  position_residual_mm: float
  orientation_residual_rad: float
  min_singular_value: float

  def violations(self) -> Tuple[LimitViolation, ...]:
    return self.limits.violations(self.joints_rad)

  def within_limits(self) -> bool:
    return not self.violations()

  def min_margin_rad(self) -> float:
    return self.limits.min_margin_rad(self.joints_rad)

  def near_singular(self, policy: SingularityPolicy) -> bool:
    return self.min_singular_value < policy.min_singular_value

  def joint_rate_gain(self) -> float:
    """Joint rate demanded per unit tool rate in the worst direction, 1/sigma_min."""
    if self.min_singular_value <= 0.0:
      return float("inf")
    return 1.0 / self.min_singular_value

  def usable(self, policy: SingularityPolicy) -> bool:
    return self.within_limits() and not self.near_singular(policy)

  def joints_deg(self) -> Tuple[float, ...]:
    return tuple(math.degrees(q) for q in self.joints_rad)


@dataclass(frozen=True)
class IKResult:
  """Everything the search found, valid and invalid, plus what it cannot promise."""

  model: "KinematicModel"
  target: Pose
  solutions: Tuple[Solution, ...]
  singularity: SingularityPolicy
  seeds_tried: int
  redundancy: Optional[RedundancyLock] = None

  def count(self) -> int:
    return len(self.solutions)

  def valid(self) -> Tuple[Solution, ...]:
    return tuple(s for s in self.solutions if s.usable(self.singularity))

  def flagged(self) -> Tuple[Solution, ...]:
    return tuple(s for s in self.solutions if not s.usable(self.singularity))

  def exhaustive(self) -> bool:
    """Always False. A multi-start numerical search is not a completeness proof.

    Raising the seed count changes the count found, in both directions: seeds
    that converge to the same basin are deduplicated away and rare basins are
    missed. Nor is there a fixed number to check against. Eight solutions is a
    generic-pose statement about arms whose last three axes intersect; the xArm 6
    wrist does not intersect (axis 6 carries a = 76 mm and d = 97 mm off axis 5),
    so it falls back toward the general 6R bound of 16 and this search routinely
    returns ten. At an internal singularity the true answer is a continuum and
    the count is not even well defined.
    """
    return False

  def configurations(self) -> Tuple[str, ...]:
    return tuple(s.configuration.label() for s in self.solutions)


@dataclass(frozen=True)
class Reachability:
  """The answer to a question about a POSE. It is not an answer about a path."""

  model: "KinematicModel"
  pose: Pose
  solutions_found: int
  valid_solutions: int
  refusal: Optional[str] = None

  def reachable(self) -> bool:
    return self.valid_solutions > 0

  def implies_path(self) -> bool:
    """Always False. See ``reachable`` for why this is a separate question."""
    return False

  def uncovered_gates(self) -> Tuple[PathGate, ...]:
    """The checks that still have to pass before this pose becomes a motion."""
    return tuple(PathGate)


# -- the model -------------------------------------------------------------------


@dataclass(frozen=True)
class KinematicModel:
  """A complete, self-describing arm geometry. Refuses to be used half-declared."""

  name: str
  convention: DHConvention
  rows: Tuple[DHRow, ...]
  provenance: Provenance
  limits: JointLimits
  envelope: Optional[ReachEnvelope] = None
  cartesian_box: Optional[CartesianBox] = None
  repeatability_mm: Optional[float] = None
  # 1-indexed axes used to label the shoulder, elbow and wrist branches.
  branch_axes: Tuple[int, int, int] = (1, 3, 5)
  source: str = ""

  def dof(self) -> int:
    return len(self.rows)

  def redundant(self) -> bool:
    """More joints than the six constraints a rigid pose imposes."""
    return self.dof() > 6

  def shoulder_height_mm(self) -> float:
    return self.rows[0].d_mm

  def require_computable(self) -> None:
    if not self.provenance.computable:
      raise UnverifiedGeometry(
        f"{self.name} carries provenance {self.provenance.value}: its link geometry "
        "was not confirmed from a first-party source, so this module will not "
        "compute a pose from it. Guessed link lengths do not produce an error "
        "signal -- the arm moves confidently to the wrong place. Supply a "
        "first-party D-H table, or read the per-unit calibrated kinematics off the "
        "controller."
      )

  def require_dof(self, joints_rad: Sequence[float]) -> None:
    if len(joints_rad) != self.dof():
      raise ValueError(
        f"{self.name} has {self.dof()} joints; got {len(joints_rad)} values"
      )
    if self.limits.dof() != self.dof():
      raise ValueError(
        f"{self.name} limit row {self.limits.label} declares {self.limits.dof()} axes "
        f"for a {self.dof()} axis arm"
      )


# -- verified geometry ------------------------------------------------------------

# Computed from the formulas the docs page and the manual print verbatim, not
# transcribed from their rounded results: the two sources round a2 differently
# (289.48866 against 289.4866) and only one of those can be right. sqrt and atan
# settle it. a2 is the effective link 2 length of the xArm 6, whose physical link
# is offset 53.5 mm from the straight line between the joints, and T2/T3 are the
# rotations that carry the mathematical zero onto the mechanical zero.
_XARM6_A2_MM = math.sqrt(284.5 ** 2 + 53.5 ** 2)
_XARM6_T2_OFFSET_RAD = -math.atan(284.5 / 53.5)
_XARM6_T3_OFFSET_RAD = -_XARM6_T2_OFFSET_RAD

_TWO_PI = 2.0 * math.pi

XARM6_MDH_ROWS: Tuple[DHRow, ...] = (
  DHRow(0.0, 267.0, 0.0, 0.0, 0.0),
  DHRow(0.0, 0.0, -math.pi / 2, 0.0, _XARM6_T2_OFFSET_RAD),
  DHRow(0.0, 0.0, 0.0, _XARM6_A2_MM, _XARM6_T3_OFFSET_RAD),
  DHRow(0.0, 342.5, -math.pi / 2, 77.5, 0.0),
  DHRow(0.0, 0.0, math.pi / 2, 0.0, 0.0),
  DHRow(0.0, 97.0, -math.pi / 2, 76.0, 0.0),
)

# Every xArm 7 offset is exactly zero; only the xArm 6 needs T2 and T3.
XARM7_MDH_ROWS: Tuple[DHRow, ...] = (
  DHRow(0.0, 267.0, 0.0, 0.0, 0.0),
  DHRow(0.0, 0.0, -math.pi / 2, 0.0, 0.0),
  DHRow(0.0, 293.0, math.pi / 2, 0.0, 0.0),
  DHRow(0.0, 0.0, math.pi / 2, 52.5, 0.0),
  DHRow(0.0, 342.5, math.pi / 2, 77.5, 0.0),
  DHRow(0.0, 0.0, math.pi / 2, 0.0, 0.0),
  DHRow(0.0, 97.0, -math.pi / 2, 76.0, 0.0),
)

_SDK_SOURCE = "xArm-Python-SDK v1.18.5 xarm/core/config/x_config.py JOINT_LIMITS"

# Base rows carry the SDK's verbatim radian constants. Variant rows below were
# reported as differences in degrees, so they are converted here and carry
# conversion rounding at the 1e-6 rad level; ``overridden_axes`` records which
# axes those are and ``fully_stated`` reports the row as partial because of them.
_XARM6_DT6 = JointLimits(
  device_type=6,
  label="xArm6 XARM6_X4",
  lower_rad=(-_TWO_PI, -2.059488, -3.92699, -_TWO_PI, -1.692969, -_TWO_PI),
  upper_rad=(_TWO_PI, 2.094395, 0.191986, _TWO_PI, math.pi, _TWO_PI),
  source=_SDK_SOURCE,
)

_XARM7_DT7 = JointLimits(
  device_type=7,
  label="xArm7 XARM7_X4",
  lower_rad=(-_TWO_PI, -2.059488, -_TWO_PI, -0.191986, -_TWO_PI, -1.692969, -_TWO_PI),
  upper_rad=(_TWO_PI, 2.094395, _TWO_PI, 3.92699, _TWO_PI, math.pi, _TWO_PI),
  source=_SDK_SOURCE,
)


def _variant(
  base: JointLimits,
  device_type: int,
  label: str,
  overrides: Dict[int, Tuple[float, float]],
) -> JointLimits:
  lo = list(base.lower_rad)
  hi = list(base.upper_rad)
  for axis, (a, b) in overrides.items():
    lo[axis - 1], hi[axis - 1] = a, b
  return JointLimits(
    device_type=device_type,
    label=label,
    lower_rad=tuple(lo),
    upper_rad=tuple(hi),
    source=_SDK_SOURCE,
    overridden_axes=tuple(sorted(overrides)),
  )


_D = math.radians

# Device types 9 and 12 sit in the same six-axis bucket of the SDK's limit table
# but are not xArm 6 arms at all -- the ROS parameter generator maps them to
# lite6 and uf850. Keying on axis count would load their envelope onto an xArm 6.
_NOT_XARM: Dict[int, str] = {
  9: "lite6",
  12: "uf850",
}

JOINT_LIMIT_ROWS: Dict[int, JointLimits] = {
  6: _XARM6_DT6,
  8: _variant(_XARM6_DT6, 8, "xArm6 XARM6_X8 (mirrored)", {3: (-0.191986, 3.92699)}),
  61305: _variant(
    _XARM6_DT6,
    61305,
    "xArm6 SN 1305+",
    {2: (_D(-117.0), _D(116.0)), 3: (_D(-219.0), _D(10.0))},
  ),
  7: _XARM7_DT7,
  3: _variant(_XARM7_DT7, 3, "xArm7 XARM7_X3 (mirrored)", {4: (-3.92699, 0.191986)}),
  71305: _variant(
    _XARM7_DT7,
    71305,
    "xArm7 SN 1305+",
    {2: (_D(-117.0), _D(116.0)), 4: (_D(-6.0), _D(225.0))},
  ),
  13: _variant(
    _XARM7_DT7,
    13,
    "xArm7 XARM7_X13 (mirror arm)",
    {2: (_D(-120.0), _D(118.0)), 4: (-3.92699, 0.191986), 6: (_D(-180.0), _D(97.0))},
  ),
}


def limits_for(device_type: int) -> JointLimits:
  """Select the joint-limit row the controller's own device type declares.

  Refuses on an unknown device type rather than falling back to the datasheet
  row, because the datasheet row is wrong for four of the seven variants here --
  two of them mirror an axis end for end.
  """
  if device_type in _NOT_XARM:
    raise NotThisArm(
      f"device type {device_type} is a {_NOT_XARM[device_type]}, not an xArm, despite "
      "sharing the six-axis bucket in the SDK limit table. Its joint ranges belong to "
      "that arm and its link geometry is not the geometry in this module."
    )
  try:
    return JOINT_LIMIT_ROWS[device_type]
  except KeyError:
    raise UnknownDeviceType(
      f"no joint-limit row declared for device type {device_type}. Joint limits vary "
      "by hardware variant, not by model or axis count, so this module will not guess "
      f"one. Declared rows: {sorted(JOINT_LIMIT_ROWS)}."
    ) from None


# Derived by maximising flange radius over the joint limits of the named device
# type using the D-H rows above. Reproducible, and self-checking: maximum flange
# height comes out at shoulder height plus maximum radius, which is what full
# extension means. Not a UFACTORY-published result, and it ignores self-collision.
XARM6_ENVELOPE = ReachEnvelope(
  max_flange_radius_mm=763.87,
  max_flange_height_mm=1030.87,
  shoulder_height_mm=267.0,
  derived_for_device_type=6,
  full_extension_witness_rad=(-1.888, 1.384918, -2.733184, -3.141593, 0.887132, 2.683891),
  note="derived from this module's D-H over device type 6 limits; ignores self-collision",
)

XARM7_ENVELOPE = ReachEnvelope(
  max_flange_radius_mm=772.05,
  max_flange_height_mm=1039.05,
  shoulder_height_mm=267.0,
  derived_for_device_type=7,
  full_extension_witness_rad=(
    0.860556, -1.393559, -3.114886, 2.741763, 3.141592, 0.887132, 0.670012,
  ),
  note="derived from this module's D-H over device type 7 limits; ignores self-collision",
)

MANUAL_CARTESIAN_BOX = CartesianBox(
  source=CartesianLimitSource.MANUAL_COMMON_SPEC,
  x_mm=(-700.0, 700.0),
  y_mm=(-700.0, 700.0),
  z_mm=(-400.0, 951.5),
)

SDK_CARTESIAN_BOX = CartesianBox(
  source=CartesianLimitSource.SDK_TCP_LIMITS,
  x_mm=(-750.0, 750.0),
  y_mm=(-750.0, 750.0),
  z_mm=(-400.0, 1000.0),
)

# The two first-party boxes disagree by 50 mm horizontally. The manual's is the
# tighter of the two, so a module that refuses rather than approximates takes it
# and accepts that it will decline poses the controller would have accepted.
DEFAULT_CARTESIAN_BOX = MANUAL_CARTESIAN_BOX

XARM6_REPEATABILITY_MM = 0.1
# The manual's xArm 7 section omits the repeatability line. None means refuse,
# not "assume the xArm 6 figure".
XARM7_REPEATABILITY_MM: Optional[float] = None

XARM_MAX_JOINT_SPEED_RAD_S = math.radians(180.0)
XARM_MAX_TCP_SPEED_MM_S = 1000.0

# Measured spread of uncalibrated absolute error against datasheet repeatability
# across published laser-tracker studies. Carried as a range because every fixed
# multiplier people quote (the "about 10x" rule) is contradicted by the data.
ABSOLUTE_TO_REPEATABILITY_RATIO_RANGE = (12.0, 85.0)
UNCALIBRATED_ABSOLUTE_ERROR_RANGE_MM = (0.36, 12.5)

DEFAULT_SINGULARITY = SingularityPolicy()


def xarm6(device_type: int = 6) -> KinematicModel:
  """The xArm 6 in modified (Craig) D-H, with the limit row for that variant."""
  lim = limits_for(device_type)
  if lim.dof() != 6:
    raise NotThisArm(f"device type {device_type} declares {lim.dof()} axes, not 6")
  return KinematicModel(
    name=f"xArm6 (device_type {device_type})",
    convention=DHConvention.MODIFIED_CRAIG,
    rows=XARM6_MDH_ROWS,
    provenance=Provenance.NOMINAL_FIRST_PARTY,
    limits=lim,
    envelope=XARM6_ENVELOPE if XARM6_ENVELOPE.applies_to(device_type) else None,
    cartesian_box=DEFAULT_CARTESIAN_BOX,
    repeatability_mm=XARM6_REPEATABILITY_MM,
    branch_axes=(1, 3, 5),
    source="UFACTORY docs + xArm User Manual V2.0.0 Appendix 8 + xarm_ros yaml",
  )


def xarm7(device_type: int = 7) -> KinematicModel:
  """The xArm 7 in modified (Craig) D-H. Redundant: see ``inverse``."""
  lim = limits_for(device_type)
  if lim.dof() != 7:
    raise NotThisArm(f"device type {device_type} declares {lim.dof()} axes, not 7")
  return KinematicModel(
    name=f"xArm7 (device_type {device_type})",
    convention=DHConvention.MODIFIED_CRAIG,
    rows=XARM7_MDH_ROWS,
    provenance=Provenance.NOMINAL_FIRST_PARTY,
    limits=lim,
    envelope=XARM7_ENVELOPE if XARM7_ENVELOPE.applies_to(device_type) else None,
    cartesian_box=DEFAULT_CARTESIAN_BOX,
    repeatability_mm=XARM7_REPEATABILITY_MM,
    branch_axes=(1, 4, 6),
    source="UFACTORY docs + xArm User Manual V2.0.0 Appendix 8 + xarm_ros yaml",
  )


def unverified_model(
  name: str,
  dof: int,
  limits: Optional[JointLimits] = None,
) -> KinematicModel:
  """A model whose geometry is not sourced. Every kinematic call on it refuses.

  This is the honest representation of an arm whose D-H table you could not
  confirm. It exists so that "we do not know the link lengths" is a state the
  code can hold and act on, rather than a comment above some plausible numbers.
  """
  rows = tuple(DHRow(0.0, 0.0, 0.0, 0.0, 0.0) for _ in range(dof))
  lim = limits or JointLimits(
    device_type=-1,
    label=f"{name} undeclared",
    lower_rad=tuple(-math.pi for _ in range(dof)),
    upper_rad=tuple(math.pi for _ in range(dof)),
    source="none",
  )
  return KinematicModel(
    name=name,
    convention=DHConvention.MODIFIED_CRAIG,
    rows=rows,
    provenance=Provenance.UNVERIFIED,
    limits=lim,
    source="none",
  )


# -- forward kinematics -----------------------------------------------------------


def link_transform(row: DHRow, q_rad: float) -> np.ndarray:
  """Modified (Craig) link transform: Rx(alpha) Tx(a) Rz(theta) Tz(d)."""
  th = row.theta_for(q_rad)
  ct, st = math.cos(th), math.sin(th)
  ca, sa = math.cos(row.alpha_rad), math.sin(row.alpha_rad)
  d, a = row.d_mm, row.a_mm
  return np.array(
    [
      [ct, -st, 0.0, a],
      [st * ca, ct * ca, -sa, -d * sa],
      [st * sa, ct * sa, ca, d * ca],
      [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=float,
  )


def _chain(model: KinematicModel, joints_rad: Sequence[float]) -> List[np.ndarray]:
  t = np.eye(4)
  out: List[np.ndarray] = []
  for row, q in zip(model.rows, joints_rad):
    t = t @ link_transform(row, q)
    out.append(t.copy())
  return out


def forward(
  model: KinematicModel,
  joints_rad: Sequence[float],
  origin: PoseOrigin = PoseOrigin.COMPUTED,
) -> Pose:
  """Flange pose in the base frame. Exact, given the geometry.

  Flange, not tool tip: no tool transform appears in a D-H table and none is
  applied here.

  Refuses on unverified geometry. Does NOT refuse on out-of-limit joints: where
  a set of angles would put the flange is a well-posed question even when the
  hardware cannot go there, and conflating the two would hide the distinction
  that ``Solution`` exists to keep visible.
  """
  model.require_computable()
  model.require_dof(joints_rad)
  chain = _chain(model, joints_rad)
  return Pose.from_matrix(chain[-1], origin=origin)


def jacobian(model: KinematicModel, joints_rad: Sequence[float]) -> np.ndarray:
  """Geometric Jacobian in the base frame, 6 x dof. Rows are mm/rad then rad/rad.

  For a modified-D-H chain the axis of joint i is the z axis of frame i, and Rz
  then Tz act along that same line, so the axis passes through the frame origin.
  """
  model.require_computable()
  model.require_dof(joints_rad)
  chain = _chain(model, joints_rad)
  o_n = chain[-1][:3, 3]
  j = np.zeros((6, model.dof()))
  for i, t in enumerate(chain):
    z = t[:3, 2]
    j[:3, i] = np.cross(z, o_n - t[:3, 3])
    j[3:, i] = z
  return j


def _scaled_jacobian(
  model: KinematicModel, joints_rad: Sequence[float], length_mm: float
) -> np.ndarray:
  # Unguarded twin of ``jacobian`` for the solver's inner loop, where the model
  # has already been validated once and revalidating per iteration is waste.
  chain = _chain(model, joints_rad)
  o_n = chain[-1][:3, 3]
  j = np.zeros((6, len(model.rows)))
  for i, t in enumerate(chain):
    z = t[:3, 2]
    j[:3, i] = np.cross(z, o_n - t[:3, 3])
    j[3:, i] = z * length_mm
  return j


def min_singular_value(
  model: KinematicModel,
  joints_rad: Sequence[float],
  policy: SingularityPolicy = DEFAULT_SINGULARITY,
) -> float:
  """Smallest singular value of the length-scaled Jacobian, in mm per rad.

  Its reciprocal is the joint rate demanded per unit tool rate in the worst
  Cartesian direction. Near a rank deficiency that reciprocal diverges, and the
  controller's answer to an impossible rate is saturation, not a smaller move.
  """
  model.require_computable()
  model.require_dof(joints_rad)
  scaled = _scaled_jacobian(model, joints_rad, policy.characteristic_length_mm)
  return float(np.linalg.svd(scaled, compute_uv=False)[-1])


def singularity_floor_for_speed(
  tcp_speed_mm_s: float,
  max_joint_speed_rad_s: float = XARM_MAX_JOINT_SPEED_RAD_S,
) -> float:
  """The sigma_min below which the arm cannot deliver that tool speed at all.

  Derived, not published: sustaining v mm/s along the degenerate direction costs
  v / sigma_min rad/s, so the floor is simply the ratio. Use it to replace the
  default threshold with one grounded in the move you actually intend.
  """
  if max_joint_speed_rad_s <= 0.0:
    raise ValueError("max joint speed must be positive")
  return tcp_speed_mm_s / max_joint_speed_rad_s


# -- branch labelling ---------------------------------------------------------------


def classify(
  model: KinematicModel,
  joints_rad: Sequence[float],
  redundancy: Optional[RedundancyLock] = None,
) -> Configuration:
  """Attach local branch labels to a joint vector.

  Shoulder is read from which side of the joint 1 frame the flange sits on;
  elbow and wrist from the sign of the D-H angle at the model's declared branch
  axes. These are names for branches, not a vendor-portable encoding.
  """
  model.require_computable()
  model.require_dof(joints_rad)
  chain = _chain(model, joints_rad)
  s_axis, e_axis, w_axis = model.branch_axes
  shoulder_frame = chain[s_axis - 1]
  flange_in_shoulder = np.linalg.inv(shoulder_frame) @ np.append(chain[-1][:3, 3], 1.0)
  theta_e = model.rows[e_axis - 1].theta_for(joints_rad[e_axis - 1])
  theta_w = model.rows[w_axis - 1].theta_for(joints_rad[w_axis - 1])
  return Configuration(
    shoulder=Shoulder.FRONT if flange_in_shoulder[0] >= 0.0 else Shoulder.BACK,
    elbow=Elbow.UP if theta_e <= 0.0 else Elbow.DOWN,
    wrist=Wrist.NOFLIP if theta_w >= 0.0 else Wrist.FLIP,
    redundancy_rad=None if redundancy is None else joints_rad[redundancy.axis - 1],
  )


# -- inverse kinematics ---------------------------------------------------------------


def _log_so3(r: np.ndarray) -> np.ndarray:
  c = (float(np.trace(r)) - 1.0) / 2.0
  c = max(-1.0, min(1.0, c))
  angle = math.acos(c)
  if angle < 1e-12:
    return np.zeros(3)
  if math.pi - angle < 1e-7:
    # Near pi the antisymmetric part vanishes; recover the axis from R + I.
    w = np.sqrt(np.maximum((np.diag(r) + 1.0) / 2.0, 0.0))
    k = int(np.argmax(w))
    axis = (r[:, k] + np.eye(3)[:, k]) / (2.0 * w[k]) if w[k] > 0 else np.eye(3)[:, k]
    axis = axis / np.linalg.norm(axis)
    return axis * angle
  v = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
  return v * (angle / (2.0 * math.sin(angle)))


def _pose_error(
  model: KinematicModel, joints_rad: np.ndarray, target: np.ndarray, length_mm: float
) -> np.ndarray:
  current = _chain(model, joints_rad)[-1]
  err = np.empty(6)
  err[:3] = target[:3, 3] - current[:3, 3]
  err[3:] = length_mm * _log_so3(target[:3, :3] @ current[:3, :3].T)
  return err


def _wrap(a: np.ndarray) -> np.ndarray:
  return (a + math.pi) % (2.0 * math.pi) - math.pi


def _solve_from_seed(
  model: KinematicModel,
  seed_q: np.ndarray,
  target: np.ndarray,
  length_mm: float,
  free: List[int],
  max_iterations: int,
  position_tolerance_mm: float,
  orientation_tolerance_rad: float,
) -> Tuple[np.ndarray, bool]:
  """Levenberg-Marquardt damped least squares.

  Damping replaces the pseudoinverse gain 1/sigma with sigma/(sigma^2 + lam^2),
  which caps joint-rate amplification at 1/(2 lam) and stays finite through a
  singularity. That is exactly why the solver can converge onto a configuration
  no controller should be driven through, and why the singular value is measured
  afterwards and carried on the solution rather than being fixed up here.
  """
  q = np.array(seed_q, dtype=float)
  lam = 1.0
  err = _pose_error(model, q, target, length_mm)
  cost = float(err @ err)
  eye6 = np.eye(6)
  for _ in range(max_iterations):
    if math.sqrt(cost) < 1e-9:
      break
    jf = _scaled_jacobian(model, q, length_mm)[:, free]
    step = jf.T @ np.linalg.solve(jf @ jf.T + (lam * lam) * eye6, err)
    biggest = float(np.abs(step).max())
    if biggest > 0.5:
      step *= 0.5 / biggest
    trial = q.copy()
    trial[free] += step
    trial_err = _pose_error(model, trial, target, length_mm)
    trial_cost = float(trial_err @ trial_err)
    if trial_cost < cost:
      q, err, cost = trial, trial_err, trial_cost
      lam = max(lam * 0.5, 1e-6)
    else:
      lam = min(lam * 3.0, 1e4)
      if lam >= 1e4:
        break
  final = _pose_error(model, q, target, length_mm)
  converged = (
    float(np.linalg.norm(final[:3])) < position_tolerance_mm
    and float(np.linalg.norm(final[3:])) / length_mm < orientation_tolerance_rad
  )
  return q, converged


def inverse(
  model: KinematicModel,
  pose: Pose,
  *,
  seeds: int = 48,
  seed: int = 0,
  singularity: SingularityPolicy = DEFAULT_SINGULARITY,
  redundancy: Optional[RedundancyLock] = None,
  max_iterations: int = 80,
  position_tolerance_mm: float = 1e-6,
  orientation_tolerance_rad: float = 1e-8,
  duplicate_tolerance_rad: float = 1e-4,
) -> IKResult:
  """Every joint vector this search finds for the pose, valid ones and flagged ones.

  Returns a set, never a single answer, because the problem has a set of answers
  and collapsing it here would hide the branch choice inside a library. Use
  ``select`` with a named policy to reduce the set to one.

  Refuses, in order, when: the geometry is unverified; the pose is provably
  outside the derived reach envelope; the arm is redundant and no redundancy
  parameter was declared; nothing converged; or everything that converged sits
  below the singularity floor.

  The search is deterministic for a given ``seed``. It is not exhaustive --
  ``IKResult.exhaustive`` says so and explains why no fixed solution count is the
  right thing to check against either.
  """
  model.require_computable()
  target = pose.as_matrix()
  xyz = target[:3, 3]

  if model.envelope is not None:
    reason = model.envelope.excludes(xyz)
    if reason is not None:
      raise OutsideEnvelope(
        f"{model.name}: pose at ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) mm is out of "
        f"reach -- {reason}. This bound is derived from the shipped D-H over the "
        "declared joint limits and it ignores self-collision, so it is necessary and "
        "not sufficient. No nearest-reachable pose is offered: silently moving "
        "somewhere else is how a plate ends up next to the nest instead of in it."
      )

  free = list(range(model.dof()))
  if redundancy is not None:
    if not 1 <= redundancy.axis <= model.dof():
      raise ValueError(f"redundancy axis {redundancy.axis} outside 1..{model.dof()}")
    free.remove(redundancy.axis - 1)
  elif model.redundant():
    raise RedundancyUnresolved(
      f"{model.name} has {model.dof()} joints against six pose constraints, so its "
      "solution set is a one-dimensional self-motion manifold per branch, not a finite "
      "list. Asking for all solutions is ill-posed. Declare a RedundancyLock -- a value "
      "for one axis -- and the search returns the finite set at that value. Note that "
      "any redundancy parameterization carries an unavoidable algorithmic singularity; "
      "locking an axis puts it wherever the task needs that axis."
    )

  rng = np.random.default_rng(seed)
  lo = np.array([max(v, -math.pi) for v in model.limits.lower_rad])
  hi = np.array([min(v, math.pi) for v in model.limits.upper_rad])
  length_mm = singularity.characteristic_length_mm

  found: List[np.ndarray] = []
  for _ in range(seeds):
    q0 = rng.uniform(lo, hi)
    if redundancy is not None:
      q0[redundancy.axis - 1] = redundancy.value_rad
    q, ok = _solve_from_seed(
      model,
      q0,
      target,
      length_mm,
      free,
      max_iterations,
      position_tolerance_mm,
      orientation_tolerance_rad,
    )
    if not ok:
      continue
    canonical = _wrap(q)
    if redundancy is not None:
      canonical[redundancy.axis - 1] = q[redundancy.axis - 1]
    if any(float(np.abs(_wrap(canonical - f)).max()) < duplicate_tolerance_rad for f in found):
      continue
    found.append(canonical)

  if not found:
    raise NoSolution(
      f"{model.name}: {seeds} seeded damped-least-squares runs converged on nothing for "
      f"the pose at ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) mm. That is evidence and "
      "not proof of unreachability -- a numerical multi-start search can miss a narrow "
      "basin. Raise the seed count before concluding anything."
    )

  solutions: List[Solution] = []
  for q in found:
    residual = _pose_error(model, q, target, length_mm)
    solutions.append(
      Solution(
        joints_rad=tuple(float(v) for v in q),
        configuration=classify(model, q, redundancy=redundancy),
        limits=model.limits,
        position_residual_mm=float(np.linalg.norm(residual[:3])),
        orientation_residual_rad=float(np.linalg.norm(residual[3:])) / length_mm,
        min_singular_value=min_singular_value(model, q, singularity),
      )
    )
  solutions.sort(key=lambda s: (not s.within_limits(), -s.min_singular_value))

  if all(s.near_singular(singularity) for s in solutions):
    worst = max(s.min_singular_value for s in solutions)
    raise SingularityRefusal(
      f"{model.name}: all {len(solutions)} solutions found for the pose at "
      f"({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) mm sit at or below sigma_min = "
      f"{worst:.3e} mm/rad, under the declared floor of "
      f"{singularity.min_singular_value} mm/rad. Holding a bounded tool velocity there "
      f"demands about {1.0 / worst if worst > 0 else float('inf'):.1f} rad/s per mm/s in "
      "the degenerate direction, which the controller answers with saturation and a "
      "lurch. Returning the joint vector anyway would be returning a number the arm "
      "cannot track. Near a singularity solution branches also merge into a double "
      "root, so which branch this is stops being decidable in double precision."
    )

  return IKResult(
    model=model,
    target=pose,
    solutions=tuple(solutions),
    singularity=singularity,
    seeds_tried=seeds,
    redundancy=redundancy,
  )


def reachable(
  model: KinematicModel,
  pose: Pose,
  *,
  seeds: int = 32,
  seed: int = 0,
  singularity: SingularityPolicy = DEFAULT_SINGULARITY,
  redundancy: Optional[RedundancyLock] = None,
) -> Reachability:
  """Whether the arm can HOLD this pose. It says NOTHING about reaching it.

  A pose-level query answers one question: does a joint vector exist that puts
  the flange here, inside the joint limits, far enough from a rank deficiency to
  be controllable. It does not answer, and cannot answer, whether a path to
  those joints exists. No path appears in the question; the path is manufactured
  later by an interpolator that this module never sees.

  Conflating the two is how an arm gets commanded into a move that is valid at
  both endpoints and passes through the bench. A joint-interpolated move drives
  every joint at once and the tool traces whatever curve the geometry maps that
  joint-space line to; if the endpoints are in different branches the arm cannot
  slide between them at all -- for a 6-DOF arm the solutions at a fixed pose are
  isolated points -- so the tool has to leave the pose, swing, and come back, and
  the swing is what hits the deck. ``Reachability.implies_path`` returns False
  unconditionally and ``uncovered_gates`` enumerates what still has to be checked
  somewhere else.
  """
  try:
    result = inverse(
      model,
      pose,
      seeds=seeds,
      seed=seed,
      singularity=singularity,
      redundancy=redundancy,
    )
  except KinematicsRefusal as exc:
    return Reachability(
      model=model,
      pose=pose,
      solutions_found=0,
      valid_solutions=0,
      refusal=f"{type(exc).__name__}: {exc}",
    )
  return Reachability(
    model=model,
    pose=pose,
    solutions_found=result.count(),
    valid_solutions=len(result.valid()),
  )


# -- selection ---------------------------------------------------------------------


def select(
  result: IKResult,
  policy: SelectionPolicy,
  *,
  reference_joints_rad: Optional[Sequence[float]] = None,
  weights: Optional[Sequence[float]] = None,
  declared: Optional[Configuration] = None,
  tie_tolerance: float = 1e-9,
) -> Solution:
  """Reduce a solution set to one under a NAMED rule. There is no default rule.

  Refuses when the rule does not pick a unique winner, rather than taking the
  first of a tie. A tie means two branches score identically under the stated
  criterion and the criterion is therefore not the one that should be deciding.
  """
  candidates = result.valid()
  if not candidates:
    flagged = result.flagged()
    detail = "; ".join(
      (
        f"{s.configuration.label()}: "
        + (
          "; ".join(v.describe() for v in s.violations())
          if s.violations()
          else f"sigma_min {s.min_singular_value:.3e} below floor"
        )
      )
      for s in flagged
    )
    raise NoSolution(
      f"none of the {len(flagged)} solutions found is usable. {detail}"
    )

  if policy is SelectionPolicy.DECLARED_CONFIGURATION:
    if declared is None:
      raise ValueError("DECLARED_CONFIGURATION requires declared=Configuration(...)")
    matches = [s for s in candidates if s.configuration.matches(declared)]
    if not matches:
      raise NoSolution(
        f"no usable solution in the declared configuration {declared.label()}. Available: "
        f"{sorted({s.configuration.label() for s in candidates})}. Reaching the pose in a "
        "different branch is a different motion, not a substitution."
      )
    if len(matches) > 1:
      raise AmbiguousSelection(
        f"{len(matches)} usable solutions share the configuration {declared.label()}. The "
        "branch labels do not separate them, so the declaration is not specific enough to "
        "decide."
      )
    return matches[0]

  if policy is SelectionPolicy.NEAREST_WEIGHTED:
    if reference_joints_rad is None:
      raise ValueError("NEAREST_WEIGHTED requires reference_joints_rad")
    if weights is None:
      raise ValueError(
        "NEAREST_WEIGHTED requires explicit weights: a distance on a joint torus is a "
        "chosen metric, not a physical quantity, and this module will not choose it. No "
        "physical unit makes a base-joint radian commensurable with a wrist-roll radian "
        "-- the joints differ in inertia, speed limit, swept volume, and consequence of "
        "error. Pass weights and own the choice."
      )
    ref = np.asarray(reference_joints_rad, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(w) != result.model.dof() or len(ref) != result.model.dof():
      raise ValueError("reference_joints_rad and weights must have one entry per joint")
    scores = [
      float(np.sum(w * _wrap(np.asarray(s.joints_rad) - ref) ** 2)) for s in candidates
    ]
    best = min(scores)
    winners = [c for c, sc in zip(candidates, scores) if sc - best <= tie_tolerance]
  elif policy is SelectionPolicy.MAX_LIMIT_MARGIN:
    scores = [s.min_margin_rad() for s in candidates]
    best = max(scores)
    winners = [c for c, sc in zip(candidates, scores) if best - sc <= tie_tolerance]
  elif policy is SelectionPolicy.MAX_SINGULAR_VALUE:
    scores = [s.min_singular_value for s in candidates]
    best = max(scores)
    winners = [c for c, sc in zip(candidates, scores) if best - sc <= tie_tolerance]
  else:  # pragma: no cover - the enum is closed
    raise ValueError(f"unhandled selection policy {policy}")

  if len(winners) > 1:
    raise AmbiguousSelection(
      f"{len(winners)} solutions tie under {policy.value} within {tie_tolerance}: "
      f"{[w.configuration.label() for w in winners]}. State a different rule rather than "
      "letting iteration order pick the branch the elbow travels in."
    )
  return winners[0]


# -- reporting ----------------------------------------------------------------------


def accuracy_statement(model: KinematicModel) -> str:
  """The sentence that has to accompany any computed pose this module produces."""
  lo, hi = ABSOLUTE_TO_REPEATABILITY_RATIO_RANGE
  elo, ehi = UNCALIBRATED_ABSOLUTE_ERROR_RANGE_MM
  rep = (
    f"{model.repeatability_mm} mm"
    if model.repeatability_mm is not None
    else "not declared for this model"
  )
  return (
    f"{model.name}: repeatability {rep}. That figure bounds returning to a TAUGHT pose "
    "and nothing else. A pose computed by inverse kinematics from nominal link geometry "
    f"inherits absolute accuracy error, measured at {lo:.0f}x to {hi:.0f}x datasheet "
    f"repeatability in published laser-tracker work ({elo} mm to {ehi} mm uncalibrated "
    "mean error, varying by a factor of nine between two arms of the same make). The "
    "geometry here is nominal design data; the per-unit calibrated kinematics readable "
    "off the controller is the only source that describes a specific arm. For placement "
    "into a plate nest, note also that the plate itself is allowed 1.0 mm of size "
    "variation per axis under ANSI/SLAS 1-2004 before the arm contributes anything, and "
    "that no standard specifies nest geometry at all."
  )
