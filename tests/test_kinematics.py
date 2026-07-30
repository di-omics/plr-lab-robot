"""Tests for plr_lr.arm.kinematics.

The suite is organised around the module's claims rather than its functions,
because the claims are what would silently rot: that forward kinematics is exact
and its geometry is sourced, that inverse kinematics returns a set and never
picks from it, that a pose query is not a path query, and that repeatability
does not bound a computed pose.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from plr_lr.arm.kinematics import (
  DEFAULT_CARTESIAN_BOX,
  MANUAL_CARTESIAN_BOX,
  SDK_CARTESIAN_BOX,
  XARM6_MDH_ROWS,
  AmbiguousSelection,
  CartesianLimitSource,
  Configuration,
  DHConvention,
  Elbow,
  JointLimits,
  KinematicsRefusal,
  NoSolution,
  NotThisArm,
  OutsideEnvelope,
  PathGate,
  Pose,
  PoseOrigin,
  Provenance,
  RedundancyLock,
  RedundancyUnresolved,
  SelectionPolicy,
  Shoulder,
  SingularityPolicy,
  SingularityRefusal,
  UnknownDeviceType,
  UnquantifiedAccuracy,
  UnverifiedGeometry,
  Wrist,
  accuracy_statement,
  classify,
  forward,
  inverse,
  jacobian,
  limits_for,
  min_singular_value,
  reachable,
  select,
  singularity_floor_for_speed,
  unverified_model,
  xarm6,
  xarm7,
)

M6 = xarm6()
M7 = xarm7()

# A generic, well-conditioned pose comfortably inside the envelope. Chosen so the
# branch structure is visible: several solutions exist and several are out of range.
GENERIC_Q6 = (0.3, -0.4, -1.1, 0.5, 0.6, -0.2)


def _wrap(a):
  return (np.asarray(a, float) + math.pi) % (2.0 * math.pi) - math.pi


class ForwardKinematicsTests(unittest.TestCase):
  """Forward kinematics is exact, so the only thing to test is the geometry."""

  def test_shipped_geometry_is_first_party_and_computable(self):
    # This build is not in refusal mode: the D-H tables were confirmed from three
    # independent first-party sources that agree cell for cell.
    for model in (M6, M7):
      self.assertIs(model.provenance, Provenance.NOMINAL_FIRST_PARTY)
      self.assertTrue(model.provenance.computable)
      self.assertIs(model.convention, DHConvention.MODIFIED_CRAIG)
      self.assertTrue(model.convention.alpha_leads_the_joint)

  def test_xarm6_offsets_come_from_the_published_formulas(self):
    # Both sources print a2 and T2_offset as formulas and as rounded results, and
    # the two printed roundings of a2 disagree. The module computes rather than
    # transcribes, so the value has to match the formula and not either rounding.
    self.assertAlmostEqual(XARM6_MDH_ROWS[2].a_mm, math.sqrt(284.5 ** 2 + 53.5 ** 2), places=9)
    self.assertAlmostEqual(math.degrees(XARM6_MDH_ROWS[1].offset_rad), -79.34995, places=5)
    self.assertAlmostEqual(
      XARM6_MDH_ROWS[1].offset_rad, -XARM6_MDH_ROWS[2].offset_rad, places=12
    )
    # Every xArm 7 offset is exactly zero; only the xArm 6 needs T2 and T3.
    self.assertTrue(all(row.offset_rad == 0.0 for row in M7.rows))

  def test_full_extension_witness_reproduces_the_derived_envelope(self):
    # The reach figures in ReachEnvelope were derived by maximising flange radius
    # over the joint limits using these very rows. If a row were mistyped the
    # witness would stop reproducing them, so this ties the table to its own
    # provenance rather than trusting the transcription.
    for model in (M6, M7):
      witness = model.envelope.full_extension_witness_rad
      pose = forward(model, witness)
      self.assertAlmostEqual(pose.radius_mm(), model.envelope.max_flange_radius_mm, places=2)
      self.assertTrue(model.limits.contains(witness))

  def test_max_height_is_shoulder_height_plus_reach(self):
    # Full extension means exactly this. It is the self-check that the two derived
    # numbers describe the same arm.
    for model in (M6, M7):
      env = model.envelope
      self.assertAlmostEqual(
        env.max_flange_height_mm,
        env.shoulder_height_mm + env.max_flange_radius_mm,
        places=2,
      )

  def test_forward_is_deterministic_and_dof_checked(self):
    a = forward(M6, GENERIC_Q6).as_matrix()
    b = forward(M6, GENERIC_Q6).as_matrix()
    self.assertTrue(np.array_equal(a, b))
    with self.assertRaises(ValueError):
      forward(M6, (0.0, 0.0, 0.0))

  def test_forward_does_not_refuse_out_of_limit_joints(self):
    # Where a set of angles puts the flange is well posed even where the hardware
    # cannot go. Conflating that with the limit check would hide the distinction
    # Solution exists to keep visible.
    out_of_range = (0.0, 0.0, 0.5, 0.0, 0.0, 0.0)  # axis 3 upper bound is 0.191986 rad
    self.assertFalse(M6.limits.contains(out_of_range))
    self.assertIsInstance(forward(M6, out_of_range), Pose)

  def test_jacobian_matches_finite_differences(self):
    rng = np.random.default_rng(4)
    for _ in range(8):
      q = rng.uniform(-1.5, 1.5, M6.dof())
      analytic = jacobian(M6, q)
      numeric = np.zeros_like(analytic)
      base = forward(M6, q).as_matrix()
      h = 1e-6
      for i in range(M6.dof()):
        qp = q.copy()
        qp[i] += h
        nxt = forward(M6, qp).as_matrix()
        numeric[:3, i] = (nxt[:3, 3] - base[:3, 3]) / h
        r = nxt[:3, :3] @ base[:3, :3].T
        numeric[3:, i] = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]) / (
          2.0 * h
        )
      self.assertLess(float(np.abs(analytic - numeric).max()), 1e-3)


class UnverifiedGeometryTests(unittest.TestCase):
  """If the geometry was not sourced, the module refuses rather than guessing."""

  def setUp(self):
    self.unknown = unverified_model("arm of unknown provenance", dof=6)

  def test_provenance_flag_is_load_bearing(self):
    self.assertIs(self.unknown.provenance, Provenance.UNVERIFIED)
    self.assertFalse(self.unknown.provenance.computable)

  def test_forward_refuses_on_unverified_geometry(self):
    with self.assertRaises(UnverifiedGeometry) as ctx:
      forward(self.unknown, [0.0] * 6)
    self.assertIn("not confirmed from a first-party source", str(ctx.exception))

  def test_every_kinematic_entry_point_refuses(self):
    for call in (
      lambda: forward(self.unknown, [0.0] * 6),
      lambda: jacobian(self.unknown, [0.0] * 6),
      lambda: min_singular_value(self.unknown, [0.0] * 6),
      lambda: classify(self.unknown, [0.0] * 6),
      lambda: inverse(self.unknown, Pose.from_matrix(np.eye(4))),
    ):
      with self.assertRaises(UnverifiedGeometry):
        call()

  def test_reachable_reports_the_refusal_rather_than_raising(self):
    answer = reachable(self.unknown, Pose.from_matrix(np.eye(4)))
    self.assertFalse(answer.reachable())
    self.assertIn("UnverifiedGeometry", answer.refusal)


class JointLimitTests(unittest.TestCase):
  """Limits vary by hardware variant, so the module reads them, never assumes."""

  def test_unknown_device_type_is_refused_not_defaulted(self):
    with self.assertRaises(UnknownDeviceType):
      limits_for(999)

  def test_six_axis_bucket_contains_arms_that_are_not_xarms(self):
    # Device types 9 and 12 sit in the same SDK bucket as the xArm 6 but are a
    # Lite6 and a UF850. Keying on axis count would load their envelope silently.
    for device_type in (9, 12):
      with self.assertRaises(NotThisArm):
        limits_for(device_type)
      with self.assertRaises(NotThisArm):
        xarm6(device_type=device_type)

  def test_variants_differ_and_say_which_axes_were_reported(self):
    base = limits_for(6)
    mirrored = limits_for(8)
    self.assertTrue(base.fully_stated())
    self.assertFalse(mirrored.fully_stated())
    self.assertEqual(mirrored.overridden_axes, (3,))
    self.assertAlmostEqual(math.degrees(base.lower_rad[2]), -225.0, places=3)
    self.assertAlmostEqual(math.degrees(mirrored.upper_rad[2]), 225.0, places=3)
    self.assertNotEqual(base.lower_rad, mirrored.lower_rad)

  def test_a_variant_envelope_is_not_carried_across_device_types(self):
    self.assertIsNotNone(xarm6(device_type=6).envelope)
    self.assertIsNone(xarm6(device_type=8).envelope)

  def test_violations_carry_the_overshoot_not_just_the_fact(self):
    over = list(GENERIC_Q6)
    over[2] = 0.5  # axis 3 upper bound is 0.191986 rad
    violations = M6.limits.violations(over)
    self.assertEqual(len(violations), 1)
    self.assertEqual(violations[0].axis, 3)
    self.assertAlmostEqual(violations[0].overshoot_rad(), 0.5 - 0.191986, places=9)

  def test_module_applies_no_slack_around_a_limit(self):
    # The SDK tolerates 0.1 deg of overshoot on each side before rejecting. That
    # is a float-noise allowance in a client-side check, not a mechanical one.
    just_over = list(GENERIC_Q6)
    just_over[2] = M6.limits.upper_rad[2] + math.radians(0.05)
    self.assertFalse(M6.limits.contains(just_over))


class InverseKinematicsTests(unittest.TestCase):
  """Inverse kinematics returns a set, flags what is unusable, and picks nothing."""

  @classmethod
  def setUpClass(cls):
    cls.target = forward(M6, GENERIC_Q6)
    cls.result = inverse(M6, cls.target, seeds=32, seed=0)

  def test_forward_then_inverse_round_trips_to_the_same_pose(self):
    for solution in self.result.solutions:
      back = forward(M6, solution.joints_rad).as_matrix()
      self.assertLess(
        float(np.linalg.norm(back[:3, 3] - self.target.as_matrix()[:3, 3])), 1e-6
      )
      self.assertLess(solution.position_residual_mm, 1e-6)
      self.assertLess(solution.orientation_residual_rad, 1e-8)

  def test_the_original_joint_vector_is_among_the_solutions(self):
    distances = [
      float(np.abs(_wrap(np.asarray(s.joints_rad) - np.asarray(GENERIC_Q6))).max())
      for s in self.result.solutions
    ]
    self.assertLess(min(distances), 1e-6)

  def test_all_solutions_are_returned_not_one(self):
    self.assertGreater(self.result.count(), 1)
    self.assertEqual(
      self.result.count(), len(self.result.valid()) + len(self.result.flagged())
    )

  def test_the_search_never_claims_completeness(self):
    # Eight is a generic-pose statement about arms whose last three axes intersect.
    # The xArm 6 wrist does not intersect, so the count is not a fixed constant and
    # a multi-start numerical search is not a proof either way.
    self.assertFalse(self.result.exhaustive())
    denser = inverse(M6, self.target, seeds=96, seed=1)
    self.assertGreaterEqual(denser.count(), 1)
    self.assertFalse(denser.exhaustive())

  def test_the_search_is_deterministic_for_a_given_seed(self):
    # A numerical solver whose answer moves between runs cannot be reviewed, and
    # a branch that changes silently is the one that swings through the deck.
    repeat = inverse(M6, self.target, seeds=32, seed=0)
    self.assertEqual(
      [s.joints_rad for s in repeat.solutions], [s.joints_rad for s in self.result.solutions]
    )

  def test_a_solution_violating_a_joint_limit_is_never_valid(self):
    self.assertTrue(self.result.flagged(), "expected at least one unusable branch here")
    for solution in self.result.valid():
      self.assertEqual(solution.violations(), ())
      self.assertTrue(solution.within_limits())
    out_of_range = [s for s in self.result.solutions if not s.within_limits()]
    self.assertTrue(out_of_range)
    for solution in out_of_range:
      self.assertNotIn(solution, self.result.valid())

  def test_out_of_limit_solutions_are_still_returned_flagged(self):
    # Dropping them silently would hide that the pose is only reachable through a
    # branch the hardware cannot enter, which is a fact about the deck layout.
    self.assertTrue(any(not s.within_limits() for s in self.result.solutions))

  def test_branches_reaching_the_same_point_can_be_far_apart_in_joint_space(self):
    # This is why the branch choice is not cosmetic: the same Cartesian point,
    # reached by swinging the shoulder or elbow through most of its range.
    valid = self.result.valid()
    spread = max(
      float(np.abs(_wrap(np.asarray(a.joints_rad) - np.asarray(b.joints_rad))).max())
      for a in valid
      for b in valid
    )
    self.assertGreater(math.degrees(spread), 100.0)

  def test_configuration_labels_are_attached_to_every_solution(self):
    for solution in self.result.solutions:
      self.assertIsInstance(solution.configuration, Configuration)
      self.assertIn(solution.configuration.shoulder, tuple(Shoulder))
      self.assertIn(solution.configuration.elbow, tuple(Elbow))
      self.assertIn(solution.configuration.wrist, tuple(Wrist))

  def test_a_wrist_flip_is_not_a_free_tie_break(self):
    self.assertTrue(Wrist.NOFLIP.tool_orientation_preserved)
    self.assertFalse(Wrist.FLIP.tool_orientation_preserved)


class EnvelopeRefusalTests(unittest.TestCase):
  """A pose outside the derived envelope is refused, never approximated."""

  def test_pose_beyond_reach_is_refused(self):
    far = np.eye(4)
    far[:3, 3] = (1500.0, 0.0, 300.0)
    with self.assertRaises(OutsideEnvelope) as ctx:
      inverse(M6, Pose.from_matrix(far))
    self.assertIn("out of reach", str(ctx.exception))

  def test_refusal_offers_no_nearest_reachable_substitute(self):
    far = np.eye(4)
    far[:3, 3] = (1500.0, 0.0, 300.0)
    answer = reachable(M6, Pose.from_matrix(far))
    self.assertFalse(answer.reachable())
    self.assertEqual(answer.solutions_found, 0)
    self.assertIn("OutsideEnvelope", answer.refusal)

  def test_the_published_reach_figure_is_not_used_as_the_bound(self):
    # The flange geometrically passes the quoted 700 mm. A check written against
    # the published figure would reject poses the arm can hold.
    self.assertGreater(M6.envelope.max_flange_radius_mm, 700.0)
    self.assertGreater(M7.envelope.max_flange_radius_mm, 700.0)
    just_past_the_datasheet = np.eye(4)
    just_past_the_datasheet[:3, 3] = (730.0, 0.0, 267.0)
    self.assertIsNone(M6.envelope.excludes(just_past_the_datasheet[:3, 3]))

  def test_the_two_first_party_cartesian_boxes_disagree(self):
    self.assertNotEqual(MANUAL_CARTESIAN_BOX.x_mm, SDK_CARTESIAN_BOX.x_mm)
    self.assertEqual(DEFAULT_CARTESIAN_BOX.source, CartesianLimitSource.MANUAL_COMMON_SPEC)
    for source in tuple(CartesianLimitSource):
      self.assertFalse(source.bounds_the_geometry)


class SingularityTests(unittest.TestCase):
  """Near a rank deficiency the module refuses instead of returning a huge number."""

  @classmethod
  def setUpClass(cls):
    # A hair inside full extension: reachable, in limits, and badly conditioned.
    q = list(M6.envelope.full_extension_witness_rad)
    q[2] += 0.05
    cls.near_boundary_q = tuple(q)
    cls.pose = forward(M6, cls.near_boundary_q)

  def test_the_pose_is_inside_the_envelope(self):
    self.assertIsNone(M6.envelope.excludes(self.pose.position_mm()))

  def test_a_near_singular_pose_is_refused(self):
    strict = SingularityPolicy(min_singular_value=singularity_floor_for_speed(50.0))
    with self.assertRaises(SingularityRefusal) as ctx:
      inverse(M6, self.pose, seeds=24, singularity=strict)
    self.assertIn("sigma_min", str(ctx.exception))

  def test_the_threshold_is_a_declared_choice_not_a_property_of_the_arm(self):
    # Same pose, looser floor, no refusal. If the refusal were baked in this would
    # fail, and the module would be pretending a chosen number is a measurement.
    loose = SingularityPolicy(min_singular_value=0.5)
    result = inverse(M6, self.pose, seeds=24, singularity=loose)
    self.assertGreater(result.count(), 0)

  def test_speed_derived_floor_follows_from_the_published_joint_speed(self):
    # Sustaining v mm/s along the degenerate direction costs v / sigma_min rad/s.
    self.assertAlmostEqual(
      singularity_floor_for_speed(50.0), 50.0 / math.radians(180.0), places=9
    )
    self.assertGreater(singularity_floor_for_speed(100.0), singularity_floor_for_speed(50.0))

  def test_joint_rate_gain_is_the_reciprocal_of_the_singular_value(self):
    result = inverse(M6, self.pose, seeds=16, singularity=SingularityPolicy(0.5))
    solution = result.solutions[0]
    self.assertAlmostEqual(
      solution.joint_rate_gain(), 1.0 / solution.min_singular_value, places=9
    )

  def test_the_arms_own_mechanical_zero_is_singular(self):
    # Worth pinning: seeding a solver from home and trusting the conditioning
    # there is already wrong for this arm.
    self.assertLess(min_singular_value(M6, [0.0] * 6), 1e-9)


class SelectionTests(unittest.TestCase):
  """Choosing among solutions is a named policy, never a hidden default."""

  @classmethod
  def setUpClass(cls):
    cls.result = inverse(M6, forward(M6, GENERIC_Q6), seeds=32, seed=0)

  def test_select_requires_a_policy_argument(self):
    with self.assertRaises(TypeError):
      select(self.result)  # type: ignore[call-arg]

  def test_nearest_requires_explicit_weights(self):
    # There is no physical unit making a base-joint radian commensurable with a
    # wrist-roll radian, so the module will not invent a metric.
    with self.assertRaises(ValueError) as ctx:
      select(
        self.result,
        SelectionPolicy.NEAREST_WEIGHTED,
        reference_joints_rad=GENERIC_Q6,
      )
    self.assertIn("explicit weights", str(ctx.exception))

  def test_nearest_with_stated_weights_returns_a_usable_solution(self):
    chosen = select(
      self.result,
      SelectionPolicy.NEAREST_WEIGHTED,
      reference_joints_rad=GENERIC_Q6,
      weights=(1.0,) * M6.dof(),
    )
    self.assertTrue(chosen.within_limits())
    self.assertLess(
      float(np.abs(_wrap(np.asarray(chosen.joints_rad) - np.asarray(GENERIC_Q6))).max()), 1e-6
    )

  def test_state_dependent_policies_are_marked_as_such(self):
    self.assertTrue(SelectionPolicy.NEAREST_WEIGHTED.depends_on_current_state)
    self.assertFalse(SelectionPolicy.MAX_LIMIT_MARGIN.depends_on_current_state)
    self.assertFalse(SelectionPolicy.MAX_SINGULAR_VALUE.depends_on_current_state)

  def test_different_policies_can_disagree_on_the_same_set(self):
    by_margin = select(self.result, SelectionPolicy.MAX_LIMIT_MARGIN)
    by_conditioning = select(self.result, SelectionPolicy.MAX_SINGULAR_VALUE)
    nearest = select(
      self.result,
      SelectionPolicy.NEAREST_WEIGHTED,
      reference_joints_rad=GENERIC_Q6,
      weights=(1.0,) * M6.dof(),
    )
    self.assertGreater(
      len({by_margin.joints_rad, by_conditioning.joints_rad, nearest.joints_rad}), 1
    )

  def test_select_never_returns_a_limit_violating_solution(self):
    for policy in (SelectionPolicy.MAX_LIMIT_MARGIN, SelectionPolicy.MAX_SINGULAR_VALUE):
      self.assertTrue(select(self.result, policy).within_limits())

  def test_declared_configuration_that_is_unavailable_is_refused(self):
    available = {s.configuration.label() for s in self.result.valid()}
    missing = None
    for shoulder in Shoulder:
      for elbow in Elbow:
        for wrist in Wrist:
          candidate = Configuration(shoulder, elbow, wrist)
          if candidate.label() not in available:
            missing = candidate
            break
        if missing:
          break
      if missing:
        break
    self.assertIsNotNone(missing)
    with self.assertRaises(NoSolution):
      select(self.result, SelectionPolicy.DECLARED_CONFIGURATION, declared=missing)

  def test_a_tie_is_refused_rather_than_resolved_by_iteration_order(self):
    loose = inverse(M6, forward(M6, GENERIC_Q6), seeds=32, seed=0)
    with self.assertRaises(AmbiguousSelection):
      select(loose, SelectionPolicy.MAX_SINGULAR_VALUE, tie_tolerance=1e9)

  def test_branch_labels_do_not_always_separate_solutions(self):
    # Two usable solutions can share a label, which is why a declared
    # configuration can be ambiguous rather than merely unavailable.
    labels = [s.configuration.label() for s in self.result.solutions]
    self.assertGreater(len(labels), len(set(labels)))


class ReachabilityIsNotPathTests(unittest.TestCase):
  """reachable() answers a question about a pose and nothing else."""

  @classmethod
  def setUpClass(cls):
    cls.answer = reachable(M6, forward(M6, GENERIC_Q6), seeds=24)

  def test_a_reachable_pose_is_reported_reachable(self):
    self.assertTrue(self.answer.reachable())
    self.assertGreater(self.answer.valid_solutions, 0)

  def test_a_reachable_pose_still_has_no_declared_path(self):
    self.assertTrue(self.answer.reachable())
    self.assertFalse(self.answer.implies_path())

  def test_the_docstring_states_the_contract(self):
    doc = reachable.__doc__ or ""
    self.assertIn("NOTHING", doc)
    self.assertIn("path", doc)
    self.assertNotEqual(doc.strip(), "")

  def test_every_path_gate_is_uncovered_by_a_pose_query(self):
    gates = self.answer.uncovered_gates()
    self.assertEqual(set(gates), set(PathGate))
    for gate in gates:
      self.assertFalse(gate.checked_by_ik)


class RedundancyTests(unittest.TestCase):
  """A 7-DOF arm has a continuum, so "all solutions" has to be refused."""

  def test_seven_dof_without_a_declared_redundancy_is_refused(self):
    target = forward(M7, (0.2, -0.5, 0.4, 1.2, 0.3, 0.7, -0.4))
    self.assertTrue(M7.redundant())
    with self.assertRaises(RedundancyUnresolved) as ctx:
      inverse(M7, target, seeds=8)
    self.assertIn("self-motion manifold", str(ctx.exception))

  def test_locking_one_axis_restores_a_finite_set(self):
    q7 = (0.2, -0.5, 0.4, 1.2, 0.3, 0.7, -0.4)
    target = forward(M7, q7)
    result = inverse(M7, target, seeds=40, redundancy=RedundancyLock(axis=3, value_rad=0.4))
    self.assertGreater(result.count(), 1)
    for solution in result.solutions:
      self.assertAlmostEqual(solution.joints_rad[2], 0.4, places=9)
      self.assertLess(solution.position_residual_mm, 1e-6)
    distances = [
      float(np.abs(_wrap(np.asarray(s.joints_rad) - np.asarray(q7))).max())
      for s in result.solutions
    ]
    self.assertLess(min(distances), 1e-6)

  def test_six_dof_is_not_redundant(self):
    self.assertFalse(M6.redundant())


class AccuracyVersusRepeatabilityTests(unittest.TestCase):
  """Repeatability bounds a taught pose. It does not bound a computed one."""

  def test_pose_origin_draws_the_line(self):
    self.assertTrue(PoseOrigin.TAUGHT.bounded_by_repeatability)
    self.assertFalse(PoseOrigin.COMPUTED.bounded_by_repeatability)

  def test_a_taught_pose_is_bounded_by_the_repeatability_figure(self):
    taught = Pose.taught(forward(M6, GENERIC_Q6).as_matrix())
    self.assertIs(taught.origin, PoseOrigin.TAUGHT)
    self.assertEqual(taught.uncertainty_mm(M6), 0.1)

  def test_a_computed_pose_refuses_to_borrow_the_repeatability_figure(self):
    computed = forward(M6, GENERIC_Q6)
    self.assertIs(computed.origin, PoseOrigin.COMPUTED)
    with self.assertRaises(UnquantifiedAccuracy) as ctx:
      computed.uncertainty_mm(M6)
    message = str(ctx.exception)
    self.assertIn("absolute", message)
    self.assertIn("repeatability", message)

  def test_inverse_kinematics_produces_computed_poses(self):
    result = inverse(M6, forward(M6, GENERIC_Q6), seeds=16)
    chosen = select(result, SelectionPolicy.MAX_SINGULAR_VALUE)
    self.assertIs(forward(M6, chosen.joints_rad).origin, PoseOrigin.COMPUTED)

  def test_a_model_with_no_published_figure_refuses_even_for_a_taught_pose(self):
    # The manual states +/-0.1 mm for the xArm 6 and omits the line from the
    # xArm 7 section. The module does not carry the figure across.
    self.assertIsNone(M7.repeatability_mm)
    taught = Pose.taught(forward(M7, (0.1,) * 7).as_matrix())
    with self.assertRaises(UnquantifiedAccuracy):
      taught.uncertainty_mm(M7)

  def test_accuracy_statement_keeps_the_two_words_apart(self):
    text = accuracy_statement(M6)
    self.assertIn("TAUGHT", text)
    self.assertIn("absolute accuracy", text)
    self.assertIn("nominal", text)


class SimulationBackendGapTests(unittest.IsolatedAsyncioTestCase):
  """The gap this module closes: two state variables that never consult each other."""

  async def test_joint_and_cartesian_state_drift_apart_unnoticed(self):
    from plr_lr import SimulationArmBackend

    backend = SimulationArmBackend()
    await backend.setup()
    await backend.home()
    before = await backend.get_cartesian_position()
    await backend.move_to({1: 0.9, 2: -0.4, 3: -1.1})
    after = await backend.get_cartesian_position()
    # Commanding joints left the Cartesian readout untouched. Nothing raised.
    self.assertEqual(before.location.x, after.location.x)
    self.assertEqual(before.location.z, after.location.z)

  async def test_kinematics_can_tell_that_the_two_disagree(self):
    from plr_lr import SimulationArmBackend

    backend = SimulationArmBackend()
    await backend.setup()
    await backend.home()
    joints = (0.9, -0.4, -1.1, 0.2, 0.3, 0.0)
    await backend.move_to(dict(enumerate(joints, start=1)))
    reported = await backend.get_cartesian_position()
    truth = forward(M6, joints).position_mm()
    disagreement = math.dist(
      (reported.location.x, reported.location.y, reported.location.z), truth
    )
    # Two orders of magnitude past any repeatability figure the arm carries.
    self.assertGreater(disagreement, 100.0)


class RefusalTaxonomyTests(unittest.TestCase):
  """Every refusal is catchable as one kind, so a caller can gate on it."""

  def test_all_refusals_share_a_base(self):
    for kind in (
      UnverifiedGeometry,
      UnknownDeviceType,
      NotThisArm,
      OutsideEnvelope,
      SingularityRefusal,
      RedundancyUnresolved,
      NoSolution,
      AmbiguousSelection,
      UnquantifiedAccuracy,
    ):
      self.assertTrue(issubclass(kind, KinematicsRefusal))
      self.assertTrue(issubclass(kind, RuntimeError))

  def test_a_declared_limit_row_must_match_the_arms_axis_count(self):
    wrong = JointLimits(
      device_type=-2,
      label="five axis row",
      lower_rad=(-1.0,) * 5,
      upper_rad=(1.0,) * 5,
    )
    model = unverified_model("five", dof=5, limits=wrong)
    self.assertEqual(model.limits.dof(), 5)
    with self.assertRaises(ValueError):
      M6.require_dof((0.0,) * 5)


if __name__ == "__main__":
  unittest.main()
