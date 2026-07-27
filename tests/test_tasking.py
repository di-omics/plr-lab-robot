import json
import unittest
from dataclasses import replace

from plr_lr import Labware, Workcell
from plr_lr.tasking import (
  ConditionKind,
  ProvenanceContext,
  RecoveryPolicy,
  StateCondition,
  TaskContract,
  TaskExecutor,
  TaskStatus,
  TransientTaskError,
  summarize_benchmark,
)


def task_contract(max_attempts=2, extra_preconditions=()):
  return TaskContract.move_labware(
    task_id="reader-load-001",
    intent="Move the tracked assay plate into the reader",
    labware_name="assay_plate",
    source_site="staging",
    destination_site="reader",
    provenance=ProvenanceContext(
      requested_by="assay-scientist",
      protocol_id="viability-read",
      protocol_version="2026-07-27",
      sample_ids=("sample-A", "sample-B"),
    ),
    recovery=RecoveryPolicy(max_attempts=max_attempts),
    extra_preconditions=extra_preconditions,
  )


async def ready_workcell():
  workcell = Workcell.sim()
  await workcell.setup()
  plate = Labware(name="assay_plate")
  workcell.add_site("staging", x=150, y=0, z=10, occupant=plate)
  workcell.add_site("reader", x=-150, y=20, z=10)
  return workcell, plate


class TaskingTests(unittest.IsolatedAsyncioTestCase):
  async def test_success_has_state_evidence_provenance_and_integrity(self):
    workcell, _ = await ready_workcell()
    contract = task_contract()

    outcome = await TaskExecutor(
      workcell,
      run_id_factory=lambda: "run-fixed",
    ).execute(contract)

    self.assertTrue(outcome.ok, outcome.error)
    self.assertEqual(outcome.status, TaskStatus.SUCCEEDED)
    self.assertEqual(outcome.run_id, "run-fixed")
    self.assertEqual(outcome.contract_digest, contract.digest)
    self.assertEqual(outcome.provenance.sample_ids, ("sample-A", "sample-B"))
    self.assertEqual(outcome.evidence_mode, "simulation")
    self.assertFalse(outcome.hardware_validated)
    self.assertIn("Kinematic simulation evidence only", outcome.evidence_caveat)
    self.assertTrue(all(check.passed for check in outcome.precondition_checks))
    self.assertTrue(all(check.passed for check in outcome.postcondition_checks))
    self.assertEqual(outcome.after.site("reader").occupant, "assay_plate")
    self.assertTrue(outcome.verify_integrity())
    self.assertIn("descend_pick", [entry["action"] for entry in outcome.command_trace])

    artifact = json.loads(outcome.to_json())
    self.assertEqual(artifact["status"], "succeeded")
    self.assertEqual(artifact["provenance"]["protocol_id"], "viability-read")

  async def test_failed_precondition_blocks_all_motion(self):
    workcell, _ = await ready_workcell()
    blocker = Labware(name="reader_blocker")
    workcell.world.labware[blocker.name] = blocker
    workcell.world.sites["reader"].occupant = blocker.name
    trace_before = len(workcell.backend.trace)

    outcome = await TaskExecutor(workcell).execute(task_contract())

    self.assertEqual(outcome.status, TaskStatus.PRECONDITION_FAILED)
    self.assertEqual(outcome.attempts, 0)
    self.assertEqual(len(workcell.backend.trace), trace_before)
    failed = [check for check in outcome.precondition_checks if not check.passed]
    self.assertEqual(len(failed), 1)
    self.assertEqual(failed[0].condition.kind, ConditionKind.SITE_OCCUPANT)
    self.assertTrue(outcome.verify_integrity())

  async def test_one_transient_failure_recovers_then_succeeds(self):
    workcell, _ = await ready_workcell()
    calls = 0

    async def flaky_move(parameters):
      nonlocal calls
      calls += 1
      if calls == 1:
        raise TransientTaskError("simulated controller timeout", code="controller_timeout")
      await workcell.move_plate(parameters.source_site, parameters.destination_site)

    outcome = await TaskExecutor(workcell).execute(task_contract(), operation=flaky_move)

    self.assertTrue(outcome.ok, outcome.error)
    self.assertTrue(outcome.recovered)
    self.assertEqual(outcome.attempts, 2)
    phases = [event.phase for event in outcome.events]
    self.assertIn("motion_halted", phases)
    self.assertIn("moved_to_safe", phases)
    self.assertIn("retry_authorized", phases)
    self.assertEqual(phases.count("attempt_started"), 2)
    self.assertTrue(outcome.verify_integrity())

  async def test_partial_motion_prevents_unsafe_retry(self):
    workcell, plate = await ready_workcell()
    calls = 0

    async def fail_after_pick(parameters):
      nonlocal calls
      calls += 1
      await workcell.arm.pick_up_resource(
        workcell.coords_for(parameters.source_site), plate_width=plate.plate_width
      )
      plate.held = True
      raise TransientTaskError("simulated disconnect after pick", code="disconnect")

    outcome = await TaskExecutor(workcell).execute(task_contract(), operation=fail_after_pick)

    self.assertEqual(outcome.status, TaskStatus.RECOVERY_FAILED)
    self.assertEqual(outcome.attempts, 1)
    self.assertEqual(calls, 1)
    self.assertTrue(workcell.backend.holding)
    self.assertIn("recovery_blocked", [event.phase for event in outcome.events])
    self.assertTrue(outcome.verify_integrity())

  async def test_unexpected_error_is_contained_but_not_retried(self):
    workcell, _ = await ready_workcell()
    calls = 0

    async def broken_move(_parameters):
      nonlocal calls
      calls += 1
      raise RuntimeError("unexpected adapter failure")

    outcome = await TaskExecutor(workcell).execute(task_contract(), operation=broken_move)

    self.assertEqual(outcome.status, TaskStatus.EXECUTION_FAILED)
    self.assertEqual(outcome.attempts, 1)
    self.assertEqual(calls, 1)
    self.assertIn("halt", [entry["action"] for entry in outcome.command_trace])

  async def test_transient_failures_stop_at_declared_attempt_budget(self):
    workcell, _ = await ready_workcell()
    calls = 0

    async def unavailable_controller(_parameters):
      nonlocal calls
      calls += 1
      raise TransientTaskError("controller unavailable", code="controller_unavailable")

    outcome = await TaskExecutor(workcell).execute(
      task_contract(max_attempts=3),
      operation=unavailable_controller,
    )

    self.assertEqual(outcome.status, TaskStatus.EXECUTION_FAILED)
    self.assertEqual(outcome.attempts, 3)
    self.assertEqual(calls, 3)
    self.assertEqual(
      [event.phase for event in outcome.events].count("retry_authorized"),
      2,
    )

  async def test_visible_postcondition_failure_is_not_retried(self):
    workcell, _ = await ready_workcell()

    async def no_op(_parameters):
      return None

    outcome = await TaskExecutor(workcell).execute(task_contract(), operation=no_op)

    self.assertEqual(outcome.status, TaskStatus.POSTCONDITION_FAILED)
    self.assertEqual(outcome.attempts, 1)
    self.assertTrue(any(not check.passed for check in outcome.postcondition_checks))
    self.assertIn("postcondition_failed", [event.phase for event in outcome.events])

  async def test_integrity_check_detects_modified_trace(self):
    workcell, _ = await ready_workcell()
    outcome = await TaskExecutor(workcell).execute(task_contract())
    forged = replace(
      outcome,
      command_trace=outcome.command_trace + ({"action": "forged"},),
    )
    self.assertTrue(outcome.verify_integrity())
    self.assertFalse(forged.verify_integrity())

  async def test_extra_domain_condition_is_a_visible_gate(self):
    workcell, plate = await ready_workcell()
    plate.capped = True
    contract = task_contract(
      extra_preconditions=(
        StateCondition(ConditionKind.LABWARE_CAPPED, True, "assay_plate"),
      )
    )
    outcome = await TaskExecutor(workcell).execute(contract)
    self.assertTrue(outcome.ok)

  async def test_benchmark_summarizes_state_gate_and_recovery_metrics(self):
    success_workcell, _ = await ready_workcell()
    success = await TaskExecutor(success_workcell).execute(task_contract())

    blocked_workcell, _ = await ready_workcell()
    blocked_workcell.world.sites["staging"].occupant = None
    blocked = await TaskExecutor(blocked_workcell).execute(task_contract())

    summary = summarize_benchmark((success, blocked))
    self.assertEqual(summary.total, 2)
    self.assertEqual(summary.succeeded, 1)
    self.assertEqual(summary.precondition_passed, 1)
    self.assertEqual(summary.postcondition_passed, 1)
    self.assertEqual(summary.integrity_verified, 2)
    self.assertEqual(summary.success_rate, 0.5)

  def test_retry_budget_is_hard_bounded(self):
    with self.assertRaises(ValueError):
      RecoveryPolicy(max_attempts=4)


if __name__ == "__main__":
  unittest.main()
