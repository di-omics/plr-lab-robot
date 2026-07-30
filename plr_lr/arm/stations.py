"""Stations: what a place on the deck IS, and whether a move between two of them is legal.

``workcell.py`` is a dict of taught poses and a ``move_plate`` that teleports
occupancy. That is enough to script a demo and not enough to refuse one. In the
README's own example, ``incubator_out`` and ``reader_in`` are string keys: there
is no incubator behind the first and no reader behind the second, so nothing in
the model can notice that the reader is still reading when the arm arrives with a
plate. A workcell that cannot say what a station IS cannot say whether a move
between two of them is legal, and a workcell that cannot refuse a move is not a
safety layer, it is a sequencer with good manners.

This module models the place, not the motion. It adds nothing to the arm API and
touches no hardware. It answers one question -- may this plate go from here to
there, right now -- and it answers it by computing, never by being told.

WHY LEGALITY IS COMPUTED

``move_plate`` today checks two things: the source has an occupant and the
destination does not. Both are facts about a Python dict. Neither is a fact about
the cell. The checks that actually keep a plate off the floor are:

  1. the source really holds the plate you named;
  2. the destination has a free slot;
  3. both stations have a pose at all -- an untaught station is a name, not a place;
  4. both poses lie inside a reach envelope you have DECLARED (see below);
  5. neither instrument is mid-cycle.

Item 5 is the one people forget, and it is the one that breaks hardware. A plate
placed into a reader that is still reading is a crash, not a queue. There is no
back pressure in a gripper. The arm will happily drive a plate into a closed
drawer, and the first thing that yields is either the plate, the nest, or the
gripper fingers. So a station that is backed by an instrument carries that
instrument's cycle state, the default for that state is UNKNOWN rather than IDLE,
and UNKNOWN refuses. Unmeasured is untrusted: a status field nobody read is not
evidence that the reader is free.

WHY TAUGHT AND COMPUTED POSES ARE NOT INTERCHANGEABLE

This is the load-bearing distinction in the module, so it gets the long argument.

A taught pose was produced by driving the arm to the place and storing what the
encoders said. Its error is REPEATABILITY: how tightly the arm returns to a point
it has already been to. A computed pose was produced by a kinematic model --
forward kinematics from nominal DH, a CAD offset from another frame, a vision
correction expressed in the base frame. Its error is ABSOLUTE ACCURACY: how close
the arm gets to a Cartesian coordinate it was merely told about. These are
different quantities by construction, and the second is much larger.

ISO 9283:1998 makes the reason structural. RP (pose repeatability) is defined
against the barycentre of the robot's OWN attained poses; the quantity it
integrates contains no term referencing the commanded pose. AP (pose accuracy) is
defined against the command. RP therefore measures scatter about wherever the arm
actually goes and is mathematically incapable of bounding how far that place is
from where it was told to go. An arm can hold RP = 0.01 mm while sitting 5 mm off
the commanded point and be entirely in spec.

The vendors say so in their own footnotes. ABB's product specifications footnote
the published AP figure to the "teached position (position manually modified in
the cell)" -- so the number printed as "pose accuracy" is play-back error against
a hand-taught point, not the error of a computed Cartesian target. The tell is in
the arithmetic: the ABB IRB 140-6/0.8 datasheet prints AP = 0.02 mm and
RP = 0.03 mm (3HAC041346-001 Rev Q). An accuracy smaller than a repeatability is
impossible for a genuine absolute figure; it is only possible because the datum is
the taught pose. The same document's Absolute Accuracy option, on the same robot,
quotes average 0.35 mm and max 0.75 mm -- roughly 17x the printed AP, and that is
the CALIBRATED figure.

Peer-reviewed laser-tracker measurements against nominal kinematics land where
you would expect: UR5 mean 0.85 mm, max 2.01 mm (Jiang et al., Sensors 2020);
ABB IRB 4600 unidirectional mean 1.39 mm (Le Reun et al., Robotica 2022); KUKA
KR-16 mean 3.39 mm and KUKA KR-30 mean 0.36 mm (Sigron et al., IEEE T-ASE 2024).
Note the last pair: same study, same rig, same method, two arms from one vendor,
a factor of 9.4 apart. Absolute accuracy is a per-unit property, not a model
property, which is exactly why this module supplies no default figure for it and
refuses instead. A generic multiplier ("accuracy is about 10x repeatability")
would be an invention; the confirmed ratios span roughly 12x to 85x.

The operational consequence, and the refusal this module encodes: a station whose
pose is COMPUTED is not interchangeable with a taught one for a placement that has
to land inside a nest's lead-in. Moving to a computed destination is refused
unless the caller supplies a MEASURED per-unit absolute accuracy for this arm
(``CellCalibration``) and a MEASURED capture radius for that nest
(``PlacementTolerance``), and the first fits inside the second. Both refusals name
the measurement that would produce the missing number. ``provenance_table`` prints
which stations are which, because the caller is entitled to see the difference
before a plate is in the air.

WHAT THE NEST ABSORBS, AND WHY WE WILL NOT MAKE THE NUMBER UP

A printed nest and an arm are one system, not two. The lead-in chamfer is the
component that converts residual placement error into a seated plate, so the
chamfer is part of the arm's accuracy budget and the arm's accuracy is part of the
nest's design spec. That is the whole argument for owning both.

But there is a scope trap. All five ANSI/SLAS microplate standards constrain the
PLATE ONLY -- each states its scope as the dimensional requirements of a
microplate. There is no standard for nest geometry, deck position, handover pose,
approach vector, or placement tolerance. Worse, in ANSI/SLAS 3-2004 the word
"chamfer" means the CORNER NOTCH on the plate (the A1 keying feature), which is
optional and, per 4.1.3.1, "must not include the flange" -- so it sits above the
flange and cannot be used as a seating datum. A nest lead-in is a different thing
that shares a word.

What the standards DO give us is the slack the nest must absorb before the robot
contributes anything. ANSI/SLAS 1-2004 allows 127.76 mm +/- 0.5 mm and
85.48 mm +/- 0.5 mm measured at any point along the side. That is a 1.0 mm
per-axis spread between two conforming plates, which is arithmetic, not a
published figure, and is labelled as derived where it is used.

What no source gives, and what this module therefore refuses to invent, is a
capture range for a nest lead-in in millimetres. Harazono et al. (SLAS Technology
2024) say it in print: there is "no quantitative data indicating the misalignment
threshold that results in transfer failures". The nearest published threshold,
about 2.19 mm, is for a LID onto a PLATE, governed by roughly 1 mm of draft on the
plate side -- a different geometry, and not transferable. The best system-level
evidence is Scamarcio et al. (Front. Robot. AI 2024): +/-1.2 mm and +/-0.42 deg
gave 95 to 100 percent insertion success across a chamfered mockup and three real
instruments, while about 4.1 mm of placement error dropped success to about 69
percent on a chamfered target. Steep, and instrument-specific: the same paper
ranks a plate sealer's "sharp, chamfered edges" against a pipetting station's
"very small chamfered edges" that demand millimetre-level precision. Those figures
appear here as evidence in prose, never as a default. ``PlacementTolerance``
starts unmeasured and says what measurement would fix that.

COMPLETED IS NOT SEATED

A move that completed is not the same as a plate that is seated, and the
difference is a sensor. The arm reporting "I ran the trajectory and opened the
gripper" is the software's own word about its own intent. It is consistent with a
plate resting on a flange corner, cocked in the nest, or on the floor.

``MoveOutcome`` therefore separates the two. ASSERTED means the motion completed
and nothing observed the result. CONFIRMED means an instrument said the plate is
there, and constructing a CONFIRMED outcome requires naming that instrument --
this class will not let you record evidence without a witness. This mirrors how a
provenance layer distinguishes software intent from an instrument's word: the
software's claim about itself is metadata, the instrument's claim is data.

One specific trap is handled explicitly. Gripper-encoder-plus-force readback at
release is a real shipping feature (the PreciseFlex 400 advertises part detection
via gripper force control and encoder feedback) and it is genuinely useful, but it
detects GRIP STATE, not SEATING. It tells you the fingers reached the expected
width with the expected force, which is consistent with a plate that then tipped.
``MoveOutcome.from_gripper_release`` accordingly returns ASSERTED, with the
gripper check recorded in the detail rather than promoted to a witness.

SEQUENCES

Legality is not composable by inspection, because a move mutates the thing later
moves are judged against. The second move in a two-move sequence can be illegal
purely because the first one left a station occupied, and neither move is illegal
on its own. ``sequence`` therefore threads the station map through the moves,
stops at the first illegal one, and names the earlier move that caused the block.
Stations are frozen, so applying a move produces a new map rather than editing the
world; nothing is committed by asking whether a plan is legal.
"""

from __future__ import annotations

from collections import abc
from dataclasses import dataclass, replace
from enum import Enum
from typing import ClassVar, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from ..world import Pose2D

# Published system-level placement evidence, Scamarcio et al., Front. Robot. AI 2024
# (DOI 10.3389/frobt.2024.1462717). These are recorded so the refusal messages can cite
# the demonstrated regime. They are NOT defaults and NOT a nest capture range: the paper
# measured a whole system's fine-positioning accuracy, not what any lead-in absorbs.
DEMONSTRATED_PLACEMENT_MM = 1.2
DEMONSTRATED_PLACEMENT_DEG = 0.42
DEGRADED_PLACEMENT_MM = 4.1  # same paper: success fell to about 69 percent here


class PoseProvenance(Enum):
  """Where a station's pose came from, which decides which error term it carries."""

  TAUGHT = "taught"
  COMPUTED = "computed"

  @property
  def inherits_absolute_error(self) -> bool:
    """True when the pose's error is absolute accuracy rather than repeatability.

    A taught pose was reached and recorded, so returning to it is a repeatability
    problem. A computed pose was never visited, so reaching it is an absolute
    accuracy problem -- measured one to three orders of magnitude worse, per-unit,
    and not bounded by any datasheet repeatability figure.
    """
    return self is PoseProvenance.COMPUTED


class CycleState(Enum):
  """What an instrument behind a station is doing right now."""

  IDLE = "idle"
  BUSY = "busy"
  UNKNOWN = "unknown"
  FAULTED = "faulted"

  @property
  def accepts_handoff(self) -> bool:
    """True only for IDLE. UNKNOWN is a refusal, not a maybe.

    A drawer that might be closed is a closed drawer as far as a plate is
    concerned. Treating an unread status as permission is how a reader gets a
    plate driven into it mid-read.
    """
    return self is CycleState.IDLE


class PlacementEvidence(Enum):
  """Whether a placement is the software's own word or an instrument's."""

  ASSERTED = "asserted"
  CONFIRMED = "confirmed"

  @property
  def is_evidence(self) -> bool:
    """True only for CONFIRMED: a sensor outside the control path observed the plate.

    ASSERTED records that the commanded motion ran to completion. That is a claim
    the mover makes about itself, and it is compatible with every way a plate can
    fail to seat.
    """
    return self is PlacementEvidence.CONFIRMED


class RefusalReason(Enum):
  """Why a move was refused."""

  UNKNOWN_STATION = "unknown_station"
  SAME_STATION = "same_station"
  SOURCE_EMPTY = "source_empty"
  PLATE_NOT_AT_SOURCE = "plate_not_at_source"
  PLATE_AMBIGUOUS = "plate_ambiguous"
  DESTINATION_OCCUPIED = "destination_occupied"
  SOURCE_BUSY = "source_busy"
  DESTINATION_BUSY = "destination_busy"
  CYCLE_UNKNOWN = "cycle_unknown"
  POSE_MISSING = "pose_missing"
  REACH_UNDECLARED = "reach_undeclared"
  POSE_UNREACHABLE = "pose_unreachable"
  POSE_COMPUTED_UNCALIBRATED = "pose_computed_uncalibrated"
  TOLERANCE_UNMEASURED = "tolerance_unmeasured"
  PLACEMENT_BUDGET_EXCEEDED = "placement_budget_exceeded"

  @property
  def clears_by_waiting(self) -> bool:
    """True when running the cell forward can resolve this, false when a human must act.

    This is the line a scheduler needs. An occupied destination or a reader
    mid-cycle clears on its own; retrying is correct. A missing pose, an
    undeclared envelope, an unmeasured nest, or an unknown cycle state never
    clears by waiting -- someone has to teach, declare, measure, or read. Retrying
    those is a busy loop that hides a defect.
    """
    return self in (
      RefusalReason.SOURCE_EMPTY,
      RefusalReason.PLATE_NOT_AT_SOURCE,
      RefusalReason.DESTINATION_OCCUPIED,
      RefusalReason.SOURCE_BUSY,
      RefusalReason.DESTINATION_BUSY,
    )


@dataclass(frozen=True)
class ReachEnvelope:
  """An axis-aligned box the caller DECLARES the arm can reach, plus where it came from.

  The source string is required, because the first-party sources disagree with
  each other and the disagreement is larger than the tolerances people care
  about. For the xArm series, the User Manual V2.0.0 common specification table
  (Appendix 2 section 1.1) gives X +/-700 mm, Y +/-700 mm, Z -400 to 951.5 mm,
  while xArm-Python-SDK v1.18.5 ``XCONF.Robot.TCP_LIMITS`` gives X +/-750 mm,
  Y +/-750 mm, Z -400 to 1000 mm for the same standard variants. That is 50 mm
  horizontally and 48.5 mm vertically between two documents from one vendor.
  Neither is derivable from the other, so this class makes you pick one on the
  record instead of inheriting a default nobody chose.

  A box is a NECESSARY condition, not a sufficient one. The reachable set of a
  serial arm is bounded by joint limits and self-collision, and joint limits vary
  by hardware variant rather than by model -- the SDK carries different rows for
  xArm 6 device_type 6 and device_type 8, which mirror each other on J3. This
  class deliberately implements no radius test: the published 700 mm reach figure
  cannot be reconciled with the 763.87 mm (xArm 6) and 772.05 mm (xArm 7) maximum
  flange radii obtained by maximising forward kinematics over the published joint
  limits, and the reference point the 700 mm is measured to is not stated
  anywhere I could confirm. A radius gate written from an unconfirmed convention
  would reject valid poses while sounding authoritative.
  """

  x: Tuple[float, float]
  y: Tuple[float, float]
  z: Tuple[float, float]
  source: str

  def __post_init__(self) -> None:
    if not self.source.strip():
      raise ValueError(
        "a reach envelope must name its source; first-party limits disagree by tens of mm"
      )
    for axis, bounds in (("x", self.x), ("y", self.y), ("z", self.z)):
      lo, hi = bounds
      if hi <= lo:
        raise ValueError(f"{axis} bounds must be (low, high) with high > low, got {bounds}")

  @classmethod
  def xarm_manual_common_spec(cls) -> "ReachEnvelope":
    """The xArm User Manual V2.0.0 Appendix 2 section 1.1 Cartesian range."""
    return cls(
      x=(-700.0, 700.0),
      y=(-700.0, 700.0),
      z=(-400.0, 951.5),
      source="xArm User Manual V2.0.0 Appendix 2 sec 1.1 (common to xArm 5/6/7)",
    )

  @classmethod
  def xarm_sdk_tcp_limits(cls) -> "ReachEnvelope":
    """The xArm-Python-SDK TCP_LIMITS box, which is looser than the manual's."""
    return cls(
      x=(-750.0, 750.0),
      y=(-750.0, 750.0),
      z=(-400.0, 1000.0),
      source="xArm-Python-SDK v1.18.5 XCONF.Robot.TCP_LIMITS (device types 3/6/7/13)",
    )

  @classmethod
  def measured(
    cls,
    x: Tuple[float, float],
    y: Tuple[float, float],
    z: Tuple[float, float],
    source: str,
  ) -> "ReachEnvelope":
    """An envelope you characterised on this cell. Say how in ``source``."""
    return cls(x=x, y=y, z=z, source=source)

  def contains(self, pose: Pose2D) -> bool:
    return (
      self.x[0] <= pose.x <= self.x[1]
      and self.y[0] <= pose.y <= self.y[1]
      and self.z[0] <= pose.z <= self.z[1]
    )

  def margin_mm(self, pose: Pose2D) -> float:
    """Smallest distance from the pose to any face. Negative when outside."""
    return min(
      pose.x - self.x[0],
      self.x[1] - pose.x,
      pose.y - self.y[0],
      self.y[1] - pose.y,
      pose.z - self.z[0],
      self.z[1] - pose.z,
    )

  def disagreement_mm(self, other: "ReachEnvelope") -> Dict[str, float]:
    """Per-face difference against another envelope, for recording which you took."""
    return {
      "x_lo": other.x[0] - self.x[0],
      "x_hi": other.x[1] - self.x[1],
      "y_lo": other.y[0] - self.y[0],
      "y_hi": other.y[1] - self.y[1],
      "z_lo": other.z[0] - self.z[0],
      "z_hi": other.z[1] - self.z[1],
    }


@dataclass(frozen=True)
class PlacementTolerance:
  """What a specific nest absorbs, and the admission that nobody publishes it.

  ``capture_radius_mm`` is the largest planar placement error from which a plate
  seats in this nest every time. It starts None, and None means never measured,
  not zero and not fine. No standard defines it (ANSI/SLAS constrains the plate
  only), no vendor publishes it, and the peer-reviewed literature says outright
  that the number does not exist in print. It exists for your nest if and only if
  you measure your nest.

  The plate-size slack below is separate and is arithmetic from ANSI/SLAS 1-2004,
  not a published figure: two conforming plates can differ by 1.0 mm per axis at
  mid-side, and the nest has to swallow that before the arm contributes anything.
  """

  station: str
  capture_radius_mm: Optional[float] = None
  capture_source: str = ""

  # ANSI/SLAS 1-2004 (R2012) Footprint Dimensions, all at 20 deg C.
  SBS_LENGTH_MM: ClassVar[float] = 127.76
  SBS_WIDTH_MM: ClassVar[float] = 85.48
  SBS_SIDE_TOL_MM: ClassVar[float] = 0.5  # +/- measured at any point along the side
  SBS_CORNER_TOL_MM: ClassVar[float] = 0.25  # +/- within 12.7 mm of the outside corners

  def __post_init__(self) -> None:
    if self.capture_radius_mm is not None:
      if self.capture_radius_mm <= 0:
        raise ValueError("a measured capture radius must be positive")
      if not self.capture_source.strip():
        raise ValueError(
          "a capture radius without a stated measurement is not a measurement; "
          "set capture_source to how the number was obtained"
        )

  @classmethod
  def unmeasured(cls, station: str) -> "PlacementTolerance":
    """The honest default for a nest nobody has characterised."""
    return cls(station=station)

  def is_measured(self) -> bool:
    return self.capture_radius_mm is not None

  def plate_size_slack_mm(self) -> float:
    """DERIVED, not published: per-axis spread between two conforming SBS plates.

    ANSI/SLAS 1-2004 allows +/-0.5 mm at any point along the side, so the full
    spread is 1.0 mm on each of length and width.
    """
    return 2.0 * self.SBS_SIDE_TOL_MM

  def arm_budget_mm(self) -> Optional[float]:
    """How much placement error is left for the ARM. None when the nest is unmeasured.

    Conservative on purpose: the full +/-0.5 mm side tolerance is subtracted
    rather than half of it, because a plate is not guaranteed to sit centred in
    the nest and an oversize plate can consume the clearance on one side.
    """
    if self.capture_radius_mm is None:
      return None
    return max(self.capture_radius_mm - self.SBS_SIDE_TOL_MM, 0.0)

  def absorbs(self, placement_error_mm: float) -> Optional[bool]:
    """True/False when the nest is measured, None when it is not.

    None is the whole point. A boolean here would be a guess wearing a type.
    """
    budget = self.arm_budget_mm()
    if budget is None:
      return None
    return placement_error_mm <= budget

  def measurement_required(self) -> str:
    """The measurement that would replace this refusal with a number."""
    return (
      f"nest '{self.station}' has no measured capture radius: drive a conforming plate "
      "into it from a grid of deliberate planar offsets, record the largest offset from "
      "which it seats on every repeat, and pass that as capture_radius_mm with its "
      f"method in capture_source. No standard or vendor publishes this figure; the "
      f"nearest published evidence is a whole-system {DEMONSTRATED_PLACEMENT_MM} mm / "
      f"{DEMONSTRATED_PLACEMENT_DEG} deg regime at 95 to 100 percent success, degrading "
      f"to about 69 percent by {DEGRADED_PLACEMENT_MM} mm, which is not a capture range"
    )


@dataclass(frozen=True)
class Instrument:
  """The thing behind a station, and how fresh our knowledge of its state is.

  ``cycle`` defaults to UNKNOWN because a status nobody read is not a status. A
  stale IDLE is exactly as dangerous as an UNKNOWN one -- a reader that was idle
  thirty seconds ago is not a reader that is idle now -- so declaring
  ``max_state_age_s`` makes an old reading decay back to UNKNOWN instead of
  quietly authorising a handoff.
  """

  name: str
  cycle: CycleState = CycleState.UNKNOWN
  state_age_s: Optional[float] = None
  max_state_age_s: Optional[float] = None

  def is_stale(self) -> bool:
    """True when a freshness requirement is declared and this reading fails it."""
    if self.max_state_age_s is None:
      return False  # caller declined to declare a freshness requirement
    if self.state_age_s is None:
      return True  # a reading of unknown age cannot satisfy a freshness requirement
    return self.state_age_s > self.max_state_age_s

  def effective_cycle(self) -> CycleState:
    """The cycle state we are entitled to act on, after staleness."""
    if self.cycle is CycleState.IDLE and self.is_stale():
      return CycleState.UNKNOWN
    return self.cycle


@dataclass(frozen=True)
class StationPose:
  """The pose the arm uses to reach a station, bound to where that pose came from.

  Pose and provenance travel together because a bare Cartesian triple invites the
  mistake this module exists to prevent: reading a computed pose as if it were a
  taught one. There is no way to hold one without the other here.
  """

  pose: Pose2D
  provenance: PoseProvenance
  source: str = ""

  def __post_init__(self) -> None:
    # Pose2D is a mutable dataclass, so a station must not alias a pose the caller
    # keeps editing; the station's copy is the station's own.
    object.__setattr__(self, "pose", replace(self.pose))

  @classmethod
  def taught(
    cls, x: float, y: float, z: float = 0.0, rz: float = 0.0, source: str = ""
  ) -> "StationPose":
    """A pose reached by driving the arm there and recording the encoders."""
    return cls(pose=Pose2D(x, y, z, rz), provenance=PoseProvenance.TAUGHT, source=source)

  @classmethod
  def computed(
    cls, x: float, y: float, z: float = 0.0, rz: float = 0.0, source: str = ""
  ) -> "StationPose":
    """A pose produced by a model. Carries absolute-accuracy error, not repeatability."""
    return cls(pose=Pose2D(x, y, z, rz), provenance=PoseProvenance.COMPUTED, source=source)

  def is_taught(self) -> bool:
    return self.provenance is PoseProvenance.TAUGHT


@dataclass(frozen=True)
class Station:
  """A named place a plate can be: a pose, its provenance, a capacity, and occupants.

  Frozen, so occupancy changes make a new station rather than editing the world.
  That is what lets a legality question be asked about a plan without committing
  any part of it.
  """

  name: str
  pose: Optional[StationPose] = None
  capacity: int = 1
  occupants: Tuple[str, ...] = ()
  instrument: Optional[Instrument] = None
  tolerance: Optional[PlacementTolerance] = None

  def __post_init__(self) -> None:
    if self.capacity < 1:
      raise ValueError(f"station {self.name!r} must have capacity >= 1")
    object.__setattr__(self, "occupants", tuple(self.occupants))
    if len(set(self.occupants)) != len(self.occupants):
      raise ValueError(f"station {self.name!r} lists the same occupant twice")
    if len(self.occupants) > self.capacity:
      raise ValueError(
        f"station {self.name!r} holds {len(self.occupants)} items over capacity {self.capacity}"
      )

  def has_pose(self) -> bool:
    return self.pose is not None

  def pose_is_taught(self) -> Optional[bool]:
    """None when the station has no pose at all -- there is nothing to classify."""
    return None if self.pose is None else self.pose.is_taught()

  def is_empty(self) -> bool:
    return not self.occupants

  def free_slots(self) -> int:
    return self.capacity - len(self.occupants)

  def is_free(self) -> bool:
    """Free means there is room for one more, which is not the same as empty."""
    return self.free_slots() > 0

  def holds(self, plate: str) -> bool:
    return plate in self.occupants

  def cycle(self) -> Optional[CycleState]:
    """The instrument's actionable state, or None when the station is a passive nest."""
    return None if self.instrument is None else self.instrument.effective_cycle()

  def accepts_handoff(self) -> bool:
    """A passive nest is never mid-cycle; an instrument must say IDLE and say it recently."""
    cycle = self.cycle()
    return True if cycle is None else cycle.accepts_handoff

  def placement_tolerance(self) -> PlacementTolerance:
    """The declared tolerance, or an explicitly unmeasured one. Never a guess."""
    return self.tolerance or PlacementTolerance.unmeasured(self.name)

  def with_plate(self, plate: str) -> "Station":
    if plate in self.occupants:
      raise ValueError(f"station {self.name!r} already holds {plate!r}")
    if not self.is_free():
      raise ValueError(f"station {self.name!r} is full (capacity {self.capacity})")
    return replace(self, occupants=self.occupants + (plate,))

  def without_plate(self, plate: str) -> "Station":
    if plate not in self.occupants:
      raise ValueError(f"station {self.name!r} does not hold {plate!r}")
    return replace(self, occupants=tuple(p for p in self.occupants if p != plate))


@dataclass(frozen=True)
class CellCalibration:
  """What the caller has actually MEASURED about this arm in this cell.

  Empty is the correct default and it makes computed-pose moves refuse. No
  absolute accuracy figure is supplied here because none is defensible: measured
  uncalibrated means span roughly 0.36 mm to 12.5 mm across published studies,
  and two arms from one vendor measured on one rig in one study differed by a
  factor of 9.4. A number good enough to gate a placement has to come from this
  arm, either from a laser-tracker characterisation, from the per-unit calibrated
  kinematics the controller will hand you (UFACTORY ships a readback for this and
  says to use it), or from touch-probe localisation of the station itself.
  """

  envelope: Optional[ReachEnvelope] = None
  absolute_accuracy_mm: Optional[float] = None
  accuracy_source: str = ""

  def __post_init__(self) -> None:
    if self.absolute_accuracy_mm is not None:
      if self.absolute_accuracy_mm <= 0:
        raise ValueError("absolute_accuracy_mm must be positive")
      if not self.accuracy_source.strip():
        raise ValueError(
          "an absolute accuracy figure without a stated measurement is not a measurement; "
          "set accuracy_source to how this arm was characterised"
        )

  def has_envelope(self) -> bool:
    return self.envelope is not None

  def has_absolute_accuracy(self) -> bool:
    return self.absolute_accuracy_mm is not None


@dataclass(frozen=True)
class Refusal:
  """One computed reason a move may not run."""

  reason: RefusalReason
  detail: str

  def __str__(self) -> str:
    return f"{self.reason.value}: {self.detail}"


@dataclass(frozen=True)
class Legality:
  """The computed verdict on one move. Empty refusals means legal."""

  move: "Move"
  refusals: Tuple[Refusal, ...] = ()

  def is_legal(self) -> bool:
    return not self.refusals

  def __bool__(self) -> bool:
    return self.is_legal()

  def reasons(self) -> Tuple[RefusalReason, ...]:
    return tuple(r.reason for r in self.refusals)

  def has(self, reason: RefusalReason) -> bool:
    return reason in self.reasons()

  def blocked_only_by_waiting(self) -> bool:
    """True when every refusal is the kind that running the cell forward can clear."""
    return bool(self.refusals) and all(r.reason.clears_by_waiting for r in self.refusals)

  def summary(self) -> str:
    if self.is_legal():
      return f"legal: {self.move}"
    joined = "; ".join(str(r) for r in self.refusals)
    return f"illegal: {self.move} -- {joined}"


StationSource = Union[Mapping[str, Station], Iterable[Station]]


def station_map(stations: StationSource) -> Dict[str, Station]:
  """Accept either a name-keyed mapping or a bare iterable of stations."""
  if isinstance(stations, abc.Mapping):
    return dict(stations)
  built: Dict[str, Station] = {}
  for st in stations:
    if st.name in built:
      raise ValueError(f"duplicate station name {st.name!r}")
    built[st.name] = st
  return built


@dataclass(frozen=True)
class Move:
  """A plate going from one station to another. Legality is computed, never asserted."""

  source: str
  destination: str
  plate: Optional[str] = None

  def __str__(self) -> str:
    what = self.plate or "the occupant"
    return f"{what}: {self.source} -> {self.destination}"

  def moved_plate(self, stations: StationSource) -> Optional[str]:
    """Which plate this move refers to, or None when that cannot be determined."""
    known = station_map(stations)
    src = known.get(self.source)
    if src is None:
      return None
    if self.plate is not None:
      return self.plate if src.holds(self.plate) else None
    if len(src.occupants) == 1:
      return src.occupants[0]
    return None  # empty, or ambiguous across several occupants

  def legality(
    self,
    stations: StationSource,
    calibration: Optional[CellCalibration] = None,
  ) -> Legality:
    """Compute every reason this move may not run, right now, against this cell."""
    known = station_map(stations)
    cal = calibration or CellCalibration()
    refusals: List[Refusal] = []

    src = known.get(self.source)
    dst = known.get(self.destination)
    for role, name, station in (
      ("source", self.source, src),
      ("destination", self.destination, dst),
    ):
      if station is None:
        refusals.append(
          Refusal(RefusalReason.UNKNOWN_STATION, f"no {role} station named {name!r} in this cell")
        )
    if src is None or dst is None:
      # Nothing further is knowable without both stations.
      return Legality(move=self, refusals=tuple(refusals))

    if self.source == self.destination:
      refusals.append(
        Refusal(RefusalReason.SAME_STATION, f"source and destination are both {self.source!r}")
      )

    # -- the source actually holds a plate ------------------------------------
    plate = self.plate
    if plate is None:
      if src.is_empty():
        refusals.append(
          Refusal(RefusalReason.SOURCE_EMPTY, f"station {src.name!r} holds nothing to move")
        )
      elif len(src.occupants) > 1:
        refusals.append(
          Refusal(
            RefusalReason.PLATE_AMBIGUOUS,
            f"station {src.name!r} holds {len(src.occupants)} items "
            f"({', '.join(src.occupants)}); name the plate on the move",
          )
        )
      else:
        plate = src.occupants[0]
    elif not src.holds(plate):
      if src.is_empty():
        refusals.append(
          Refusal(RefusalReason.SOURCE_EMPTY, f"station {src.name!r} holds nothing to move")
        )
      else:
        refusals.append(
          Refusal(
            RefusalReason.PLATE_NOT_AT_SOURCE,
            f"station {src.name!r} holds {', '.join(src.occupants)}, not {plate!r}",
          )
        )

    # -- the destination is free ----------------------------------------------
    if not dst.is_free():
      refusals.append(
        Refusal(
          RefusalReason.DESTINATION_OCCUPIED,
          f"station {dst.name!r} holds {', '.join(dst.occupants)} at capacity {dst.capacity}",
        )
      )

    # -- both poses exist ------------------------------------------------------
    for role, station in (("source", src), ("destination", dst)):
      if not station.has_pose():
        refusals.append(
          Refusal(
            RefusalReason.POSE_MISSING,
            f"{role} station {station.name!r} has no pose; an untaught station is a name, "
            "not a place",
          )
        )

    # -- both poses are reachable ---------------------------------------------
    envelope = cal.envelope
    if envelope is None:
      refusals.append(
        Refusal(
          RefusalReason.REACH_UNDECLARED,
          "no reach envelope declared, so reachability cannot be computed; first-party "
          "sources disagree (xArm manual X/Y +/-700 mm vs SDK TCP_LIMITS +/-750 mm), so "
          "the envelope has to be chosen on the record via CellCalibration(envelope=...)",
        )
      )
    else:
      for role, station in (("source", src), ("destination", dst)):
        if station.pose is None:
          continue
        if not envelope.contains(station.pose.pose):
          margin = envelope.margin_mm(station.pose.pose)
          refusals.append(
            Refusal(
              RefusalReason.POSE_UNREACHABLE,
              f"{role} station {station.name!r} lies outside the declared envelope "
              f"({envelope.source}) by {abs(margin):.2f} mm",
            )
          )

    # -- neither instrument is mid-cycle --------------------------------------
    # The destination is the crash case: a plate driven into a reader that is still
    # reading. The source is the same failure wearing a different hat -- pulling a
    # plate out from under a running measurement -- so both are checked.
    for role, station, busy_reason in (
      ("source", src, RefusalReason.SOURCE_BUSY),
      ("destination", dst, RefusalReason.DESTINATION_BUSY),
    ):
      instrument = station.instrument
      if instrument is None:
        continue  # a passive nest has no cycle to be in the middle of
      cycle = instrument.effective_cycle()
      if cycle.accepts_handoff:
        continue
      if cycle is CycleState.BUSY:
        refusals.append(
          Refusal(
            busy_reason,
            f"{role} instrument {instrument.name!r} at {station.name!r} is mid-cycle; "
            "a plate into a busy instrument is a crash, not a queue",
          )
        )
      else:
        stale = instrument.is_stale() and instrument.cycle is CycleState.IDLE
        why = (
          f"last read {instrument.state_age_s} s ago against a "
          f"{instrument.max_state_age_s} s freshness requirement"
          if stale
          else f"reported {cycle.value}"
        )
        refusals.append(
          Refusal(
            RefusalReason.CYCLE_UNKNOWN,
            f"{role} instrument {instrument.name!r} at {station.name!r} {why}; "
            "an unread status is not permission",
          )
        )

    refusals.extend(self._placement_refusals(dst, cal))
    return Legality(move=self, refusals=tuple(refusals))

  def _placement_refusals(self, dst: Station, cal: CellCalibration) -> List[Refusal]:
    """The taught-versus-computed gate on the destination pose.

    A taught destination is exempt: the arm has been there, so the relevant error
    is repeatability, which is what the datasheet figure actually describes. A
    computed destination is not exempt, and no amount of repeatability spec
    substitutes for the absolute accuracy it inherits.
    """
    if dst.pose is None:
      return []  # already refused as POSE_MISSING; do not pile on
    if not dst.pose.provenance.inherits_absolute_error:
      return []

    tolerance = dst.placement_tolerance()
    accuracy = cal.absolute_accuracy_mm
    if accuracy is None:
      return [
        Refusal(
          RefusalReason.POSE_COMPUTED_UNCALIBRATED,
          f"destination {dst.name!r} has a COMPUTED pose ({dst.pose.source or 'no source given'}) "
          "and no measured absolute accuracy for this arm; a computed pose inherits "
          "absolute-accuracy error, which datasheet repeatability cannot bound (ISO 9283 RP is "
          "defined against the arm's own barycentre, not the command). Measure this unit -- "
          "laser tracker, the controller's per-unit calibrated kinematics, or touch-probe "
          "localisation of the station -- and pass it as CellCalibration(absolute_accuracy_mm=...)",
        )
      ]

    budget = tolerance.arm_budget_mm()
    if budget is None:
      return [Refusal(RefusalReason.TOLERANCE_UNMEASURED, tolerance.measurement_required())]

    if accuracy > budget:
      return [
        Refusal(
          RefusalReason.PLACEMENT_BUDGET_EXCEEDED,
          f"destination {dst.name!r} has a COMPUTED pose and this arm's measured absolute "
          f"accuracy {accuracy:.2f} mm exceeds the nest budget {budget:.2f} mm "
          f"(capture {tolerance.capture_radius_mm:.2f} mm less "
          f"{PlacementTolerance.SBS_SIDE_TOL_MM:.2f} mm of ANSI/SLAS plate side tolerance); "
          "teach this station or shrink the error before placing into it",
        )
      ]
    return []

  def apply(self, stations: StationSource) -> Dict[str, Station]:
    """Return the station map after this move. Does NOT re-check legality.

    Call ``legality`` first. This performs only the two checks it needs to avoid
    producing a corrupt map, and raises rather than fabricating one.
    """
    known = station_map(stations)
    src = known[self.source]
    dst = known[self.destination]
    plate = self.moved_plate(known)
    if plate is None:
      raise ValueError(f"cannot apply {self}: the plate to move is not determined at {src.name!r}")
    known[self.source] = src.without_plate(plate)
    known[self.destination] = dst.with_plate(plate)
    return known


@dataclass(frozen=True)
class MoveOutcome:
  """What the arm reports it did, and whether anything witnessed the result.

  Completion and seating are different claims. Construct via the classmethods:
  ``confirmed`` demands a named sensor, and there is no path to CONFIRMED that
  does not name one.
  """

  move: Move
  plate: str
  completed: bool
  placement: PlacementEvidence
  sensor: str = ""
  detail: str = ""

  def __post_init__(self) -> None:
    if self.placement is PlacementEvidence.CONFIRMED:
      if not self.sensor.strip():
        raise ValueError(
          "a CONFIRMED placement must name the sensor that observed the plate; "
          "evidence without a witness is an assertion"
        )
      if not self.completed:
        raise ValueError("a placement cannot be CONFIRMED by a move that did not complete")
    elif self.sensor.strip():
      raise ValueError(
        "an ASSERTED placement must not name a sensor; if a sensor spoke, the outcome is "
        "CONFIRMED or the move failed"
      )

  @classmethod
  def asserted(cls, move: Move, plate: str, detail: str = "") -> "MoveOutcome":
    """The motion ran and nothing observed the result. The default, and usually the truth."""
    return cls(
      move=move,
      plate=plate,
      completed=True,
      placement=PlacementEvidence.ASSERTED,
      detail=detail or "motion completed; no placement sensor",
    )

  @classmethod
  def confirmed(cls, move: Move, plate: str, sensor: str, detail: str = "") -> "MoveOutcome":
    """An instrument outside the control path says the plate is seated."""
    if not sensor.strip():
      raise ValueError("confirmed() requires the name of the sensor that observed the plate")
    return cls(
      move=move,
      plate=plate,
      completed=True,
      placement=PlacementEvidence.CONFIRMED,
      sensor=sensor,
      detail=detail or f"placement observed by {sensor}",
    )

  @classmethod
  def from_gripper_release(
    cls, move: Move, plate: str, width_mm: float, expected_width_mm: float
  ) -> "MoveOutcome":
    """A gripper-encoder check at release. Deliberately ASSERTED, never CONFIRMED.

    Encoder width plus force at release is real part detection (the PreciseFlex
    400 ships it) but it observes the GRIPPER, not the NEST. It is consistent with
    a plate released cleanly and then tipping, riding up on a flange, or sitting
    proud of the nest. Recording it as evidence of seating would be exactly the
    conflation this module exists to prevent, so it lands in the detail string and
    the placement stays ASSERTED.
    """
    delta = width_mm - expected_width_mm
    return cls.asserted(
      move=move,
      plate=plate,
      detail=(
        f"gripper released at {width_mm:.2f} mm vs expected {expected_width_mm:.2f} mm "
        f"(delta {delta:+.2f} mm); grip state only, not seating -- no nest sensor"
      ),
    )

  @classmethod
  def failed(cls, move: Move, plate: str, detail: str) -> "MoveOutcome":
    """The motion did not complete. Placement is asserted-and-false, never confirmed."""
    return cls(
      move=move,
      plate=plate,
      completed=False,
      placement=PlacementEvidence.ASSERTED,
      detail=detail,
    )

  def is_seated(self) -> Optional[bool]:
    """True only on sensed evidence, None when the placement is merely ASSERTED.

    None means nobody knows. A caller that wants a boolean here has to go and put
    a sensor on the nest.
    """
    if not self.completed:
      return False
    return True if self.placement.is_evidence else None

  def summary(self) -> str:
    state = "completed" if self.completed else "FAILED"
    witness = f" [{self.sensor}]" if self.sensor else ""
    return f"{self.move} {state} {self.placement.value.upper()}{witness}: {self.detail}"


@dataclass(frozen=True)
class SequenceReport:
  """Legality of a whole plan, stopping at the first move that cannot run.

  ``blocked_by[i]`` names the earlier move that caused move i to be refused, when
  one did. That is the case a per-move check cannot see: neither move is illegal
  alone, and the second is illegal only because the first ran.
  """

  legalities: Tuple[Legality, ...]
  blocked_by: Tuple[Optional[int], ...]
  stations_after: Tuple[Station, ...]

  def is_legal(self) -> bool:
    return all(lg.is_legal() for lg in self.legalities)

  def __bool__(self) -> bool:
    return self.is_legal()

  def station_map(self) -> Dict[str, Station]:
    """The cell as it stands after every legal move that was applied."""
    return {st.name: st for st in self.stations_after}

  def first_illegal_index(self) -> Optional[int]:
    for i, lg in enumerate(self.legalities):
      if not lg.is_legal():
        return i
    return None

  def first_illegal(self) -> Optional[Legality]:
    i = self.first_illegal_index()
    return None if i is None else self.legalities[i]

  def reason(self) -> str:
    """The first illegal move, why, and which earlier move is to blame."""
    i = self.first_illegal_index()
    if i is None:
      return f"all {len(self.legalities)} moves legal"
    lg = self.legalities[i]
    detail = "; ".join(str(r) for r in lg.refusals)
    text = f"move {i} ({lg.move}) is illegal -- {detail}"
    culprit = self.blocked_by[i]
    if culprit is not None:
      text += f"; blocked by move {culprit} ({self.legalities[culprit].move})"
    return text

  def summary(self) -> str:
    lines = [f"sequence: {'LEGAL' if self.is_legal() else 'ILLEGAL'}"]
    for i, lg in enumerate(self.legalities):
      mark = "ok " if lg.is_legal() else "XX "
      lines.append(f"  [{mark}] move {i}: {lg.summary()}")
      culprit = self.blocked_by[i]
      if culprit is not None:
        lines.append(f"         blocked by move {culprit} ({self.legalities[culprit].move})")
    lines.append(provenance_table(self.stations_after))
    return "\n".join(lines)


def taught_pose_stations(stations: StationSource) -> Tuple[str, ...]:
  known = station_map(stations)
  return tuple(sorted(n for n, st in known.items() if st.pose_is_taught() is True))


def computed_pose_stations(stations: StationSource) -> Tuple[str, ...]:
  known = station_map(stations)
  return tuple(sorted(n for n, st in known.items() if st.pose_is_taught() is False))


def unposed_stations(stations: StationSource) -> Tuple[str, ...]:
  known = station_map(stations)
  return tuple(sorted(n for n, st in known.items() if st.pose_is_taught() is None))


def provenance_table(stations: StationSource) -> str:
  """Show which stations are taught and which are computed, and what each nest knows.

  The caller is entitled to see this before a plate is in the air. A computed
  station is not a worse station; it is a station whose error term is different,
  and the difference does not show up anywhere else in a workcell dict.
  """
  known = station_map(stations)
  lines = ["  pose provenance:"]
  for name in sorted(known):
    st = known[name]
    if st.pose is None:
      mark, note = "??", "no pose taught or computed"
    elif st.pose.is_taught():
      mark = "T "
      note = "TAUGHT -- error is repeatability"
    else:
      mark = "C "
      note = "COMPUTED -- inherits absolute-accuracy error, not interchangeable with TAUGHT"
    tol = st.placement_tolerance()
    if tol.is_measured():
      note += f"; nest capture {tol.capture_radius_mm:.2f} mm measured"
    else:
      note += "; nest capture unmeasured"
    lines.append(f"    [{mark}] {name}: {note}")
  return "\n".join(lines)


def sequence(
  moves: Sequence[Move],
  stations: StationSource,
  calibration: Optional[CellCalibration] = None,
) -> SequenceReport:
  """Compute the legality of a whole plan, threading state through it.

  Each move is judged against the cell as the earlier moves left it, so a move can
  be refused for a condition its predecessor created. Evaluation stops at the
  first illegal move: everything after it is judged against a state that will
  never exist, and reporting speculative verdicts about an unreachable future is
  worse than saying nothing.
  """
  current = station_map(stations)
  legalities: List[Legality] = []
  blame: List[Optional[int]] = []
  placed_by: Dict[str, int] = {}
  emptied_by: Dict[str, int] = {}

  for i, mv in enumerate(moves):
    lg = mv.legality(current, calibration)
    culprit: Optional[int] = None
    if not lg.is_legal():
      if lg.has(RefusalReason.DESTINATION_OCCUPIED):
        culprit = placed_by.get(mv.destination)
      if culprit is None and (
        lg.has(RefusalReason.SOURCE_EMPTY) or lg.has(RefusalReason.PLATE_NOT_AT_SOURCE)
      ):
        culprit = emptied_by.get(mv.source)
    legalities.append(lg)
    blame.append(culprit)
    if not lg.is_legal():
      break
    current = mv.apply(current)
    placed_by[mv.destination] = i
    emptied_by[mv.source] = i

  return SequenceReport(
    legalities=tuple(legalities),
    blocked_by=tuple(blame),
    stations_after=tuple(current[n] for n in sorted(current)),
  )


__all__ = [
  "CellCalibration",
  "CycleState",
  "DEGRADED_PLACEMENT_MM",
  "DEMONSTRATED_PLACEMENT_DEG",
  "DEMONSTRATED_PLACEMENT_MM",
  "Instrument",
  "Legality",
  "Move",
  "MoveOutcome",
  "PlacementEvidence",
  "PlacementTolerance",
  "PoseProvenance",
  "ReachEnvelope",
  "Refusal",
  "RefusalReason",
  "SequenceReport",
  "Station",
  "StationPose",
  "computed_pose_stations",
  "provenance_table",
  "sequence",
  "station_map",
  "taught_pose_stations",
  "unposed_stations",
]
