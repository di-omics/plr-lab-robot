import unittest

from plr_lr import Pose2D
from plr_lr.arm.stations import (
  CellCalibration,
  CycleState,
  Instrument,
  Move,
  MoveOutcome,
  PlacementEvidence,
  PlacementTolerance,
  PoseProvenance,
  ReachEnvelope,
  RefusalReason,
  Station,
  StationPose,
  computed_pose_stations,
  provenance_table,
  sequence,
  station_map,
  taught_pose_stations,
)

# A declared envelope, so reachability is a computed answer rather than a refusal.
# The manual box is chosen deliberately here; the SDK box is 50 mm wider per side.
ENVELOPE = ReachEnvelope.xarm_manual_common_spec()
CAL = CellCalibration(envelope=ENVELOPE)


def taught_station(name, x=200.0, y=0.0, z=12.0, **kw):
  return Station(name=name, pose=StationPose.taught(x, y, z, source="jogged and stored"), **kw)


def computed_station(name, x=200.0, y=0.0, z=12.0, **kw):
  return Station(name=name, pose=StationPose.computed(x, y, z, source="CAD offset"), **kw)


def two_station_cell():
  """One plate at the incubator, an empty reader that is known idle."""
  return station_map(
    [
      taught_station("incubator_out", x=180.0, y=0.0, occupants=("assay_plate",)),
      taught_station(
        "reader_in",
        x=-180.0,
        y=40.0,
        instrument=Instrument("plate_reader", cycle=CycleState.IDLE),
      ),
    ]
  )


class LegalityTests(unittest.TestCase):
  def test_clean_move_is_legal(self):
    cell = two_station_cell()
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertTrue(verdict.is_legal(), verdict.summary())
    self.assertTrue(verdict)  # __bool__ mirrors SkillResult

  def test_move_into_occupied_station_is_refused(self):
    cell = two_station_cell()
    cell["reader_in"] = cell["reader_in"].with_plate("stale_plate")
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)

    self.assertFalse(verdict.is_legal())
    self.assertIn(RefusalReason.DESTINATION_OCCUPIED, verdict.reasons())
    self.assertIn("stale_plate", verdict.summary())
    # Occupancy clears when the cell runs on, so this one is worth retrying.
    self.assertTrue(RefusalReason.DESTINATION_OCCUPIED.clears_by_waiting)

  def test_move_from_empty_station_is_refused(self):
    cell = two_station_cell()
    cell["incubator_out"] = cell["incubator_out"].without_plate("assay_plate")
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)

    self.assertFalse(verdict.is_legal())
    self.assertIn(RefusalReason.SOURCE_EMPTY, verdict.reasons())

  def test_named_plate_not_at_source_is_refused(self):
    cell = two_station_cell()
    verdict = Move("incubator_out", "reader_in", plate="other_plate").legality(cell, CAL)
    self.assertIn(RefusalReason.PLATE_NOT_AT_SOURCE, verdict.reasons())

  def test_ambiguous_plate_at_a_hotel_is_refused(self):
    cell = station_map(
      [
        taught_station("hotel", x=180.0, capacity=4, occupants=("p1", "p2")),
        taught_station("reader_in", x=-180.0),
      ]
    )
    verdict = Move("hotel", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.PLATE_AMBIGUOUS, verdict.reasons())
    # Naming the plate resolves it; capacity 4 with two occupants is still free.
    self.assertTrue(Move("hotel", "reader_in", plate="p2").legality(cell, CAL).is_legal())

  def test_missing_pose_is_refused(self):
    cell = two_station_cell()
    cell["reader_in"] = Station(name="reader_in")  # a name, not a place
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.POSE_MISSING, verdict.reasons())
    self.assertFalse(RefusalReason.POSE_MISSING.clears_by_waiting)

  def test_unknown_station_short_circuits(self):
    cell = two_station_cell()
    verdict = Move("incubator_out", "nowhere").legality(cell, CAL)
    self.assertEqual(verdict.reasons(), (RefusalReason.UNKNOWN_STATION,))

  def test_undeclared_envelope_is_itself_a_refusal(self):
    # First-party Cartesian limits disagree, so an envelope has to be chosen on
    # the record. No declaration means reachability is unanswerable, not fine.
    verdict = Move("incubator_out", "reader_in").legality(two_station_cell())
    self.assertIn(RefusalReason.REACH_UNDECLARED, verdict.reasons())

  def test_pose_outside_declared_envelope_is_refused(self):
    cell = two_station_cell()
    cell["reader_in"] = taught_station("reader_in", x=-900.0, y=0.0)
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.POSE_UNREACHABLE, verdict.reasons())
    self.assertIn(ENVELOPE.source, verdict.summary())

  def test_published_envelopes_disagree_by_the_documented_amounts(self):
    delta = ReachEnvelope.xarm_manual_common_spec().disagreement_mm(
      ReachEnvelope.xarm_sdk_tcp_limits()
    )
    self.assertAlmostEqual(delta["x_hi"], 50.0, places=6)
    self.assertAlmostEqual(delta["y_hi"], 50.0, places=6)
    self.assertAlmostEqual(delta["z_hi"], 48.5, places=6)

  def test_envelope_requires_a_named_source(self):
    with self.assertRaises(ValueError):
      ReachEnvelope(x=(-1.0, 1.0), y=(-1.0, 1.0), z=(0.0, 1.0), source="  ")

  def test_retryable_and_unretryable_refusals_are_separable(self):
    cell = two_station_cell()
    cell["reader_in"] = cell["reader_in"].with_plate("stale_plate")
    retryable = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertTrue(retryable.blocked_only_by_waiting())

    cell["reader_in"] = Station(name="reader_in")  # untaught: waiting never fixes this
    stuck = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertFalse(stuck.blocked_only_by_waiting())

  def test_empty_calibration_reports_what_it_lacks(self):
    empty = CellCalibration()
    self.assertFalse(empty.has_envelope())
    self.assertFalse(empty.has_absolute_accuracy())
    self.assertTrue(CAL.has_envelope())


class InstrumentCycleTests(unittest.TestCase):
  def test_busy_destination_is_refused(self):
    cell = two_station_cell()
    cell["reader_in"] = taught_station(
      "reader_in",
      x=-180.0,
      y=40.0,
      instrument=Instrument("plate_reader", cycle=CycleState.BUSY),
    )
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.DESTINATION_BUSY, verdict.reasons())
    self.assertIn("crash, not a queue", verdict.summary())

  def test_busy_source_is_refused_too(self):
    cell = two_station_cell()
    cell["incubator_out"] = taught_station(
      "incubator_out",
      x=180.0,
      occupants=("assay_plate",),
      instrument=Instrument("incubator", cycle=CycleState.BUSY),
    )
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.SOURCE_BUSY, verdict.reasons())

  def test_unread_cycle_state_refuses_rather_than_assuming_idle(self):
    cell = two_station_cell()
    cell["reader_in"] = taught_station(
      "reader_in", x=-180.0, y=40.0, instrument=Instrument("plate_reader")
    )
    self.assertIs(cell["reader_in"].cycle(), CycleState.UNKNOWN)
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.CYCLE_UNKNOWN, verdict.reasons())
    self.assertFalse(RefusalReason.CYCLE_UNKNOWN.clears_by_waiting)

  def test_stale_idle_decays_to_unknown(self):
    fresh = Instrument("reader", cycle=CycleState.IDLE, state_age_s=1.0, max_state_age_s=5.0)
    stale = Instrument("reader", cycle=CycleState.IDLE, state_age_s=30.0, max_state_age_s=5.0)
    self.assertIs(fresh.effective_cycle(), CycleState.IDLE)
    self.assertIs(stale.effective_cycle(), CycleState.UNKNOWN)

    cell = two_station_cell()
    cell["reader_in"] = taught_station("reader_in", x=-180.0, y=40.0, instrument=stale)
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.CYCLE_UNKNOWN, verdict.reasons())

  def test_passive_nest_is_never_mid_cycle(self):
    nest = taught_station("bench_nest")
    self.assertIsNone(nest.cycle())
    self.assertTrue(nest.accepts_handoff())


class ComputedPoseTests(unittest.TestCase):
  def test_computed_destination_refuses_without_a_measured_arm(self):
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("assay_plate",)),
        computed_station("reader_in", x=-180.0, y=40.0),
      ]
    )
    verdict = Move("incubator_out", "reader_in").legality(cell, CAL)
    self.assertIn(RefusalReason.POSE_COMPUTED_UNCALIBRATED, verdict.reasons())
    self.assertIn("absolute-accuracy error", verdict.summary())

  def test_taught_destination_needs_no_absolute_accuracy(self):
    verdict = Move("incubator_out", "reader_in").legality(two_station_cell(), CAL)
    self.assertNotIn(RefusalReason.POSE_COMPUTED_UNCALIBRATED, verdict.reasons())
    self.assertTrue(verdict.is_legal())

  def test_measured_arm_still_refuses_an_unmeasured_nest(self):
    cal = CellCalibration(
      envelope=ENVELOPE, absolute_accuracy_mm=0.9, accuracy_source="laser tracker, 300 poses"
    )
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("assay_plate",)),
        computed_station("reader_in", x=-180.0, y=40.0),
      ]
    )
    verdict = Move("incubator_out", "reader_in").legality(cell, cal)
    self.assertIn(RefusalReason.TOLERANCE_UNMEASURED, verdict.reasons())
    self.assertIn("no measured capture radius", verdict.summary())

  def test_measured_arm_and_measured_nest_can_pass(self):
    cal = CellCalibration(
      envelope=ENVELOPE, absolute_accuracy_mm=0.9, accuracy_source="laser tracker, 300 poses"
    )
    nest = PlacementTolerance(
      station="reader_in",
      capture_radius_mm=2.0,
      capture_source="offset grid, 20 seats per offset, largest all-pass offset",
    )
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("assay_plate",)),
        computed_station("reader_in", x=-180.0, y=40.0, tolerance=nest),
      ]
    )
    self.assertTrue(Move("incubator_out", "reader_in").legality(cell, cal).is_legal())

  def test_measured_arm_outside_the_nest_budget_is_refused(self):
    cal = CellCalibration(
      envelope=ENVELOPE, absolute_accuracy_mm=3.4, accuracy_source="laser tracker, 300 poses"
    )
    nest = PlacementTolerance(
      station="reader_in", capture_radius_mm=2.0, capture_source="offset grid"
    )
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("assay_plate",)),
        computed_station("reader_in", x=-180.0, y=40.0, tolerance=nest),
      ]
    )
    verdict = Move("incubator_out", "reader_in").legality(cell, cal)
    self.assertIn(RefusalReason.PLACEMENT_BUDGET_EXCEEDED, verdict.reasons())

  def test_accuracy_without_a_stated_measurement_is_rejected(self):
    with self.assertRaises(ValueError):
      CellCalibration(envelope=ENVELOPE, absolute_accuracy_mm=0.5)

  def test_provenance_property_draws_the_line(self):
    self.assertTrue(PoseProvenance.COMPUTED.inherits_absolute_error)
    self.assertFalse(PoseProvenance.TAUGHT.inherits_absolute_error)


class PlacementToleranceTests(unittest.TestCase):
  def test_unmeasured_nest_answers_none_rather_than_guessing(self):
    tol = PlacementTolerance.unmeasured("reader_in")
    self.assertFalse(tol.is_measured())
    self.assertIsNone(tol.arm_budget_mm())
    self.assertIsNone(tol.absorbs(0.1))
    self.assertIn("capture_radius_mm", tol.measurement_required())

  def test_plate_size_slack_is_derived_from_ansi_slas(self):
    tol = PlacementTolerance.unmeasured("reader_in")
    self.assertAlmostEqual(tol.plate_size_slack_mm(), 1.0, places=6)

  def test_arm_budget_subtracts_the_plate_side_tolerance(self):
    tol = PlacementTolerance(
      station="reader_in", capture_radius_mm=2.0, capture_source="offset grid"
    )
    self.assertAlmostEqual(tol.arm_budget_mm(), 1.5, places=6)
    self.assertTrue(tol.absorbs(1.2))
    self.assertFalse(tol.absorbs(1.6))

  def test_capture_radius_without_a_method_is_rejected(self):
    with self.assertRaises(ValueError):
      PlacementTolerance(station="reader_in", capture_radius_mm=2.0)


class MoveOutcomeTests(unittest.TestCase):
  def test_completed_move_without_a_sensor_is_asserted_never_confirmed(self):
    move = Move("incubator_out", "reader_in", plate="assay_plate")
    outcome = MoveOutcome.asserted(move, "assay_plate")

    self.assertTrue(outcome.completed)
    self.assertIs(outcome.placement, PlacementEvidence.ASSERTED)
    self.assertIsNot(outcome.placement, PlacementEvidence.CONFIRMED)
    self.assertFalse(outcome.placement.is_evidence)
    self.assertIsNone(outcome.is_seated())  # nobody knows, and the type says so
    self.assertEqual(outcome.sensor, "")

  def test_gripper_release_check_is_grip_state_not_seating(self):
    move = Move("incubator_out", "reader_in", plate="assay_plate")
    outcome = MoveOutcome.from_gripper_release(move, "assay_plate", 90.0, 90.0)
    self.assertIs(outcome.placement, PlacementEvidence.ASSERTED)
    self.assertIsNone(outcome.is_seated())
    self.assertIn("not seating", outcome.detail)

  def test_confirmed_requires_a_named_witness(self):
    move = Move("incubator_out", "reader_in", plate="assay_plate")
    with self.assertRaises(ValueError):
      MoveOutcome.confirmed(move, "assay_plate", sensor="")
    with self.assertRaises(ValueError):
      MoveOutcome(
        move=move,
        plate="assay_plate",
        completed=True,
        placement=PlacementEvidence.CONFIRMED,
      )

  def test_confirmed_with_a_sensor_is_evidence(self):
    move = Move("incubator_out", "reader_in", plate="assay_plate")
    outcome = MoveOutcome.confirmed(move, "assay_plate", sensor="reader nest presence switch")
    self.assertTrue(outcome.placement.is_evidence)
    self.assertTrue(outcome.is_seated())

  def test_failed_move_is_not_seated(self):
    move = Move("incubator_out", "reader_in", plate="assay_plate")
    outcome = MoveOutcome.failed(move, "assay_plate", "gripper stalled on approach")
    self.assertFalse(outcome.completed)
    self.assertFalse(outcome.is_seated())


class SequenceTests(unittest.TestCase):
  def test_second_move_blocked_by_the_first_is_caught_and_named(self):
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, y=0.0, occupants=("plate_a",)),
        taught_station("bench", x=150.0, y=-90.0, occupants=("plate_b",)),
        taught_station("reader_in", x=-180.0, y=40.0),
      ]
    )
    plan = [Move("incubator_out", "reader_in"), Move("bench", "reader_in")]
    report = sequence(plan, cell, CAL)

    self.assertFalse(report.is_legal())
    self.assertEqual(report.first_illegal_index(), 1)
    self.assertTrue(report.legalities[0].is_legal())  # neither move is illegal alone
    self.assertIn(RefusalReason.DESTINATION_OCCUPIED, report.legalities[1].reasons())
    self.assertEqual(report.blocked_by[1], 0)
    self.assertIn("move 1", report.reason())
    self.assertIn("blocked by move 0", report.reason())
    self.assertIn("plate_a", report.reason())

  def test_sequence_names_the_move_that_emptied_the_source(self):
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("plate_a",)),
        taught_station("bench", x=150.0, y=-90.0),
        taught_station("reader_in", x=-180.0, y=40.0),
      ]
    )
    plan = [Move("incubator_out", "bench"), Move("incubator_out", "reader_in")]
    report = sequence(plan, cell, CAL)

    self.assertEqual(report.first_illegal_index(), 1)
    self.assertIn(RefusalReason.SOURCE_EMPTY, report.legalities[1].reasons())
    self.assertEqual(report.blocked_by[1], 0)

  def test_sequence_stops_at_the_first_illegal_move(self):
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("plate_a",)),
        taught_station("bench", x=150.0, y=-90.0, occupants=("plate_b",)),
        taught_station("reader_in", x=-180.0, y=40.0),
      ]
    )
    plan = [
      Move("incubator_out", "reader_in"),
      Move("bench", "reader_in"),
      Move("reader_in", "bench"),
    ]
    report = sequence(plan, cell, CAL)
    self.assertEqual(len(report.legalities), 2)  # the third is never judged
    self.assertEqual(len(report.blocked_by), 2)

  def test_legal_sequence_threads_state_through(self):
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("plate_a",)),
        taught_station("bench", x=150.0, y=-90.0),
        taught_station("reader_in", x=-180.0, y=40.0),
      ]
    )
    plan = [Move("incubator_out", "bench"), Move("bench", "reader_in")]
    report = sequence(plan, cell, CAL)

    self.assertTrue(report.is_legal(), report.reason())
    after = report.station_map()
    self.assertEqual(after["reader_in"].occupants, ("plate_a",))
    self.assertTrue(after["incubator_out"].is_empty())
    self.assertTrue(after["bench"].is_empty())
    # The input map is untouched: asking is not doing.
    self.assertEqual(cell["incubator_out"].occupants, ("plate_a",))


class ProvenanceReportTests(unittest.TestCase):
  def test_computed_station_is_distinguishable_from_a_taught_one(self):
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("plate_a",)),
        computed_station("reader_in", x=-180.0, y=40.0),
        Station(name="unposed"),
      ]
    )
    self.assertEqual(taught_pose_stations(cell), ("incubator_out",))
    self.assertEqual(computed_pose_stations(cell), ("reader_in",))

    table = provenance_table(cell)
    self.assertIn("[T ] incubator_out", table)
    self.assertIn("[C ] reader_in", table)
    self.assertIn("[??] unposed", table)
    self.assertIn("not interchangeable with TAUGHT", table)

  def test_sequence_summary_carries_the_provenance_table(self):
    cell = station_map(
      [
        taught_station("incubator_out", x=180.0, occupants=("plate_a",)),
        computed_station("reader_in", x=-180.0, y=40.0),
      ]
    )
    report = sequence([Move("incubator_out", "reader_in")], cell, CAL)
    summary = report.summary()
    self.assertIn("ILLEGAL", summary)
    self.assertIn("[C ] reader_in", summary)
    self.assertIn("[T ] incubator_out", summary)


class StationModelTests(unittest.TestCase):
  def test_free_is_not_the_same_as_empty(self):
    hotel = taught_station("hotel", capacity=4, occupants=("p1",))
    self.assertTrue(hotel.is_free())
    self.assertFalse(hotel.is_empty())
    self.assertEqual(hotel.free_slots(), 3)

  def test_station_is_frozen_so_occupancy_changes_make_a_new_station(self):
    before = taught_station("bench")
    after = before.with_plate("p1")
    self.assertTrue(before.is_empty())
    self.assertEqual(after.occupants, ("p1",))
    with self.assertRaises(ValueError):
      after.with_plate("p2")  # capacity 1

  def test_station_pose_does_not_alias_a_caller_mutable_pose(self):
    caller_pose = Pose2D(100.0, 0.0, 12.0)
    st = Station(name="bench", pose=StationPose(caller_pose, PoseProvenance.TAUGHT))
    caller_pose.x = 9999.0  # Pose2D is mutable; the station kept its own copy
    self.assertEqual(st.pose.pose.x, 100.0)

  def test_overfilled_station_is_rejected_at_construction(self):
    with self.assertRaises(ValueError):
      Station(name="bench", occupants=("p1", "p2"))


if __name__ == "__main__":
  unittest.main()
