"""Typed, observable, and auditable workcell task execution.

This module sits above :class:`plr_lr.arm.workcell.Workcell`. A domain expert
declares a typed task contract, including the visible state that must be true
before and after motion. ``TaskExecutor`` evaluates those gates, contains errors,
and retries only explicitly transient failures when the original safe state has
been re-established.

The implementation is simulation-first. Audit records describe what this Python
process observed and commanded; they are not proof of physical state and do not
replace hardware interlocks, sensing, or operator qualification.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Awaitable, Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Optional, Tuple, Union

from .arm.sim_backend import SimulationArmBackend
from .arm.workcell import Workcell

JsonScalar = Union[None, bool, int, float, str]
MoveOperation = Callable[["MoveLabwareParameters"], Awaitable[None]]


def _jsonable(value: Any) -> Any:
  """Convert task records into a stable JSON-compatible shape."""
  if isinstance(value, Enum):
    return value.value
  if is_dataclass(value):
    return {item.name: _jsonable(getattr(value, item.name)) for item in fields(value)}
  if isinstance(value, Mapping):
    return {str(key): _jsonable(item) for key, item in sorted(value.items())}
  if isinstance(value, (tuple, list)):
    return [_jsonable(item) for item in value]
  return value


def _canonical_json(value: Any) -> str:
  return json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _digest(value: Any) -> str:
  return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _utc_now() -> str:
  return datetime.now(timezone.utc).isoformat()


def _details(**values: object) -> Tuple[Tuple[str, str], ...]:
  return tuple(sorted((key, str(value)) for key, value in values.items()))


class TaskAction(str, Enum):
  MOVE_LABWARE = "move_labware"


class TaskStatus(str, Enum):
  SUCCEEDED = "succeeded"
  EVIDENCE_GATED = "evidence_gated"
  PRECONDITION_FAILED = "precondition_failed"
  EXECUTION_FAILED = "execution_failed"
  RECOVERY_FAILED = "recovery_failed"
  POSTCONDITION_FAILED = "postcondition_failed"


class ConditionKind(str, Enum):
  SITE_OCCUPANT = "site_occupant"
  LABWARE_HELD = "labware_held"
  LABWARE_CAPPED = "labware_capped"
  CAP_PARKED = "cap_parked"
  ARM_HOLDING = "arm_holding"
  ARM_HOMED = "arm_homed"
  BACKEND_READY = "backend_ready"


@dataclass(frozen=True)
class SiteState:
  name: str
  occupant: Optional[str]


@dataclass(frozen=True)
class LabwareState:
  name: str
  site: Optional[str]
  held: bool
  capped: bool
  cap_parked: bool
  pose_mm_rad: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ArmState:
  backend_type: str
  ready: Optional[bool]
  homed: Optional[bool]
  holding: Optional[bool]


@dataclass(frozen=True)
class WorkcellSnapshot:
  """State visible to the Python workcell model at one point in execution."""

  sites: Tuple[SiteState, ...]
  labware: Tuple[LabwareState, ...]
  arm: ArmState
  command_count: int

  @classmethod
  def capture(cls, workcell: Workcell) -> "WorkcellSnapshot":
    backend = workcell.arm.backend
    sites = tuple(
      SiteState(name=name, occupant=site.occupant)
      for name, site in sorted(workcell.world.sites.items())
    )
    labware_sites = {
      site.occupant: site.name for site in workcell.world.sites.values() if site.occupant
    }
    labware = tuple(
      LabwareState(
        name=name,
        site=labware_sites.get(name),
        held=item.held,
        capped=item.capped,
        cap_parked=name in workcell.world.parking,
        pose_mm_rad=(item.pose.x, item.pose.y, item.pose.z, item.pose.rz),
      )
      for name, item in sorted(workcell.world.labware.items())
    )
    trace = getattr(backend, "trace", ())
    return cls(
      sites=sites,
      labware=labware,
      arm=ArmState(
        backend_type=type(backend).__name__,
        ready=getattr(backend, "is_set_up", None),
        homed=getattr(backend, "homed", None),
        holding=getattr(backend, "holding", None),
      ),
      command_count=len(trace),
    )

  def site(self, name: str) -> Optional[SiteState]:
    return next((site for site in self.sites if site.name == name), None)

  def item(self, name: str) -> Optional[LabwareState]:
    return next((item for item in self.labware if item.name == name), None)


@dataclass(frozen=True)
class StateCondition:
  """One machine-checkable expectation over visible workcell state."""

  kind: ConditionKind
  expected: JsonScalar
  subject: str = ""

  def evaluate(self, snapshot: WorkcellSnapshot) -> "ConditionCheck":
    observed: JsonScalar
    if self.kind == ConditionKind.SITE_OCCUPANT:
      site = snapshot.site(self.subject)
      observed = site.occupant if site is not None else "<missing-site>"
    elif self.kind in (
      ConditionKind.LABWARE_HELD,
      ConditionKind.LABWARE_CAPPED,
      ConditionKind.CAP_PARKED,
    ):
      item = snapshot.item(self.subject)
      if item is None:
        observed = "<missing-labware>"
      elif self.kind == ConditionKind.LABWARE_HELD:
        observed = item.held
      elif self.kind == ConditionKind.LABWARE_CAPPED:
        observed = item.capped
      else:
        observed = item.cap_parked
    elif self.kind == ConditionKind.ARM_HOLDING:
      observed = snapshot.arm.holding
    elif self.kind == ConditionKind.ARM_HOMED:
      observed = snapshot.arm.homed
    else:
      observed = snapshot.arm.ready
    return ConditionCheck(condition=self, observed=observed, passed=observed == self.expected)


@dataclass(frozen=True)
class ConditionCheck:
  condition: StateCondition
  observed: JsonScalar
  passed: bool


def evaluate_conditions(
  conditions: Iterable[StateCondition], snapshot: WorkcellSnapshot
) -> Tuple[ConditionCheck, ...]:
  return tuple(condition.evaluate(snapshot) for condition in conditions)


@dataclass(frozen=True)
class ProvenanceContext:
  """Operator-supplied identifiers carried into every task outcome."""

  requested_by: str
  protocol_id: str
  protocol_version: str
  sample_ids: Tuple[str, ...] = ()

  def __post_init__(self) -> None:
    if not self.requested_by.strip():
      raise ValueError("requested_by must identify the requesting role or operator")
    if not self.protocol_id.strip() or not self.protocol_version.strip():
      raise ValueError("protocol_id and protocol_version are required")
    if any(not sample_id.strip() for sample_id in self.sample_ids):
      raise ValueError("sample_ids cannot contain blank identifiers")


@dataclass(frozen=True)
class RecoveryPolicy:
  """A small retry budget; no task may retry without bound."""

  max_attempts: int = 2
  move_to_safe_before_retry: bool = True

  def __post_init__(self) -> None:
    if not 1 <= self.max_attempts <= 3:
      raise ValueError("max_attempts must be between 1 and 3")


@dataclass(frozen=True)
class MoveLabwareParameters:
  labware_name: str
  source_site: str
  destination_site: str

  def __post_init__(self) -> None:
    if not self.labware_name.strip():
      raise ValueError("labware_name is required")
    if not self.source_site.strip() or not self.destination_site.strip():
      raise ValueError("source_site and destination_site are required")
    if self.source_site == self.destination_site:
      raise ValueError("source_site and destination_site must differ")


@dataclass(frozen=True)
class TaskContract:
  """Typed task intent plus explicit visible-state gates."""

  task_id: str
  intent: str
  action: TaskAction
  parameters: MoveLabwareParameters
  provenance: ProvenanceContext
  preconditions: Tuple[StateCondition, ...]
  postconditions: Tuple[StateCondition, ...]
  recovery: RecoveryPolicy = field(default_factory=RecoveryPolicy)

  def __post_init__(self) -> None:
    if not self.task_id.strip() or not self.intent.strip():
      raise ValueError("task_id and intent are required")
    if self.action != TaskAction.MOVE_LABWARE:
      raise ValueError(f"Unsupported task action: {self.action}")
    if not self.preconditions or not self.postconditions:
      raise ValueError("preconditions and postconditions cannot be empty")

  @classmethod
  def move_labware(
    cls,
    *,
    task_id: str,
    intent: str,
    labware_name: str,
    source_site: str,
    destination_site: str,
    provenance: ProvenanceContext,
    recovery: Optional[RecoveryPolicy] = None,
    extra_preconditions: Tuple[StateCondition, ...] = (),
    extra_postconditions: Tuple[StateCondition, ...] = (),
  ) -> "TaskContract":
    """Build a plate move with conservative default state gates."""
    parameters = MoveLabwareParameters(labware_name, source_site, destination_site)
    preconditions = (
      StateCondition(ConditionKind.BACKEND_READY, True),
      StateCondition(ConditionKind.ARM_HOMED, True),
      StateCondition(ConditionKind.ARM_HOLDING, False),
      StateCondition(ConditionKind.SITE_OCCUPANT, labware_name, source_site),
      StateCondition(ConditionKind.SITE_OCCUPANT, None, destination_site),
      StateCondition(ConditionKind.LABWARE_HELD, False, labware_name),
    ) + extra_preconditions
    postconditions = (
      StateCondition(ConditionKind.ARM_HOLDING, False),
      StateCondition(ConditionKind.SITE_OCCUPANT, None, source_site),
      StateCondition(ConditionKind.SITE_OCCUPANT, labware_name, destination_site),
      StateCondition(ConditionKind.LABWARE_HELD, False, labware_name),
    ) + extra_postconditions
    return cls(
      task_id=task_id,
      intent=intent,
      action=TaskAction.MOVE_LABWARE,
      parameters=parameters,
      provenance=provenance,
      preconditions=preconditions,
      postconditions=postconditions,
      recovery=recovery or RecoveryPolicy(),
    )

  @property
  def digest(self) -> str:
    return _digest(self)


class TaskExecutionError(RuntimeError):
  """A classified task error; only errors marked transient are retryable."""

  def __init__(self, message: str, *, code: str = "task_error", transient: bool = False) -> None:
    super().__init__(message)
    self.code = code
    self.transient = transient


class TransientTaskError(TaskExecutionError):
  def __init__(self, message: str, *, code: str = "transient_task_error") -> None:
    super().__init__(message, code=code, transient=True)


@dataclass(frozen=True)
class TaskEvent:
  sequence: int
  at: str
  phase: str
  attempt: int
  details: Tuple[Tuple[str, str], ...]
  previous_hash: str
  event_hash: str

  def unsigned_payload(self) -> dict:
    return {
      "sequence": self.sequence,
      "at": self.at,
      "phase": self.phase,
      "attempt": self.attempt,
      "details": self.details,
      "previous_hash": self.previous_hash,
    }


class _AuditTrail:
  def __init__(self, clock: Callable[[], str]) -> None:
    self._clock = clock
    self.events: list[TaskEvent] = []

  def append(self, phase: str, attempt: int = 0, **details: object) -> None:
    previous_hash = self.events[-1].event_hash if self.events else "GENESIS"
    event = TaskEvent(
      sequence=len(self.events),
      at=self._clock(),
      phase=phase,
      attempt=attempt,
      details=_details(**details),
      previous_hash=previous_hash,
      event_hash="",
    )
    event_hash = _digest(event.unsigned_payload())
    self.events.append(
      TaskEvent(
        sequence=event.sequence,
        at=event.at,
        phase=event.phase,
        attempt=event.attempt,
        details=event.details,
        previous_hash=event.previous_hash,
        event_hash=event_hash,
      )
    )


def verify_event_chain(events: Iterable[TaskEvent]) -> bool:
  previous_hash = "GENESIS"
  for sequence, event in enumerate(events):
    if event.sequence != sequence or event.previous_hash != previous_hash:
      return False
    if event.event_hash != _digest(event.unsigned_payload()):
      return False
    previous_hash = event.event_hash
  return True


@dataclass(frozen=True)
class TaskOutcome:
  run_id: str
  task_id: str
  status: TaskStatus
  contract_digest: str
  provenance: ProvenanceContext
  evidence_mode: str
  hardware_validated: bool
  evidence_caveat: str
  attempts: int
  error: str
  before: WorkcellSnapshot
  after: WorkcellSnapshot
  precondition_checks: Tuple[ConditionCheck, ...]
  postcondition_checks: Tuple[ConditionCheck, ...]
  command_trace: Tuple[Mapping[str, object], ...]
  events: Tuple[TaskEvent, ...]

  @property
  def ok(self) -> bool:
    return self.status == TaskStatus.SUCCEEDED

  @property
  def recovered(self) -> bool:
    return self.ok and self.attempts > 1

  def verify_integrity(self) -> bool:
    if not self.events or not verify_event_chain(self.events):
      return False
    final = self.events[-1]
    if final.phase != "task_finished":
      return False
    details = dict(final.details)
    return (
      details.get("status") == self.status.value
      and details.get("before_digest") == _digest(self.before)
      and details.get("after_digest") == _digest(self.after)
      and details.get("trace_digest") == _digest(self.command_trace)
    )

  def to_dict(self) -> dict:
    return _jsonable(self)

  def to_json(self, *, indent: int = 2) -> str:
    return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


@dataclass(frozen=True)
class BenchmarkSummary:
  """Aggregate state-gate and recovery metrics over task outcomes."""

  total: int
  succeeded: int
  recovered: int
  precondition_passed: int
  postcondition_passed: int
  integrity_verified: int

  @property
  def success_rate(self) -> float:
    return self.succeeded / self.total if self.total else 0.0

  @property
  def precondition_pass_rate(self) -> float:
    return self.precondition_passed / self.total if self.total else 0.0

  @property
  def postcondition_pass_rate(self) -> float:
    return self.postcondition_passed / self.total if self.total else 0.0


def summarize_benchmark(outcomes: Iterable[TaskOutcome]) -> BenchmarkSummary:
  collected = tuple(outcomes)
  return BenchmarkSummary(
    total=len(collected),
    succeeded=sum(outcome.ok for outcome in collected),
    recovered=sum(outcome.recovered for outcome in collected),
    precondition_passed=sum(
      bool(outcome.precondition_checks) and all(check.passed for check in outcome.precondition_checks)
      for outcome in collected
    ),
    postcondition_passed=sum(
      bool(outcome.postcondition_checks) and all(check.passed for check in outcome.postcondition_checks)
      for outcome in collected
    ),
    integrity_verified=sum(outcome.verify_integrity() for outcome in collected),
  )


class TaskExecutor:
  """Execute task contracts with state gates and conservative bounded recovery."""

  SIMULATION_CAVEAT = (
    "Kinematic simulation evidence only: no physical motion, contact, force, torque, slip, "
    "sensor, or hardware interlock was validated."
  )
  UNVALIDATED_RUNTIME_CAVEAT = (
    "This backend has no validation evidence in plr-lab-robot; execution is gated by default."
  )

  def __init__(
    self,
    workcell: Workcell,
    *,
    allow_unvalidated_hardware: bool = False,
    clock: Callable[[], str] = _utc_now,
    run_id_factory: Callable[[], str] = lambda: str(uuid.uuid4()),
  ) -> None:
    self.workcell = workcell
    self.allow_unvalidated_hardware = allow_unvalidated_hardware
    self.clock = clock
    self.run_id_factory = run_id_factory

  async def _default_move(self, parameters: MoveLabwareParameters) -> None:
    moved = await self.workcell.move_plate(parameters.source_site, parameters.destination_site)
    if moved != parameters.labware_name:
      raise TaskExecutionError(
        f"workcell moved {moved!r}, expected {parameters.labware_name!r}",
        code="wrong_labware",
      )

  async def _contain(self, trail: _AuditTrail, attempt: int) -> bool:
    try:
      await self.workcell.arm.halt()
      trail.append("motion_halted", attempt)
      return True
    except Exception as error:  # containment failure must be part of the audit record
      trail.append(
        "motion_halt_failed",
        attempt,
        error_type=type(error).__name__,
        error=str(error),
      )
      return False

  async def _recover_for_retry(
    self,
    contract: TaskContract,
    trail: _AuditTrail,
    attempt: int,
  ) -> bool:
    snapshot = WorkcellSnapshot.capture(self.workcell)
    checks = evaluate_conditions(contract.preconditions, snapshot)
    if snapshot.arm.holding is not False or not all(check.passed for check in checks):
      trail.append(
        "recovery_blocked",
        attempt,
        reason="original visible preconditions were not re-established",
        failed_checks=sum(not check.passed for check in checks),
        arm_holding=snapshot.arm.holding,
      )
      return False
    if contract.recovery.move_to_safe_before_retry:
      try:
        await self.workcell.arm.move_to_safe()
        trail.append("moved_to_safe", attempt)
      except Exception as error:
        trail.append(
          "move_to_safe_failed",
          attempt,
          error_type=type(error).__name__,
          error=str(error),
        )
        return False
    checks = evaluate_conditions(contract.preconditions, WorkcellSnapshot.capture(self.workcell))
    ok = all(check.passed for check in checks)
    trail.append("recovery_evaluated", attempt, retry_safe=ok)
    return ok

  async def execute(
    self,
    contract: TaskContract,
    *,
    operation: Optional[MoveOperation] = None,
  ) -> TaskOutcome:
    """Run one contract; injected operations are useful for simulation fault benchmarks."""
    run_id = self.run_id_factory()
    trail = _AuditTrail(self.clock)
    before = WorkcellSnapshot.capture(self.workcell)
    trace_start = before.command_count
    backend = self.workcell.arm.backend
    is_simulation = isinstance(backend, SimulationArmBackend)
    evidence_mode = "simulation" if is_simulation else "unvalidated_runtime"
    caveat = self.SIMULATION_CAVEAT if is_simulation else self.UNVALIDATED_RUNTIME_CAVEAT
    trail.append(
      "task_started",
      task_id=contract.task_id,
      run_id=run_id,
      contract_digest=contract.digest,
      evidence_mode=evidence_mode,
    )

    status = TaskStatus.EXECUTION_FAILED
    attempts = 0
    error_message = ""
    post_checks: Tuple[ConditionCheck, ...] = ()
    pre_checks = evaluate_conditions(contract.preconditions, before)

    if not is_simulation and not self.allow_unvalidated_hardware:
      status = TaskStatus.EVIDENCE_GATED
      error_message = "unvalidated backend blocked; explicit operator override required"
      trail.append("evidence_gate_blocked", reason=error_message)
    elif not all(check.passed for check in pre_checks):
      status = TaskStatus.PRECONDITION_FAILED
      error_message = "one or more visible preconditions failed"
      trail.append(
        "precondition_failed",
        failed_checks=sum(not check.passed for check in pre_checks),
      )
    else:
      trail.append("precondition_passed", checks=len(pre_checks))
      move = operation or self._default_move
      for attempt in range(1, contract.recovery.max_attempts + 1):
        attempts = attempt
        trail.append("attempt_started", attempt)
        try:
          await move(contract.parameters)
        except Exception as caught:
          transient = isinstance(caught, TaskExecutionError) and caught.transient
          code = caught.code if isinstance(caught, TaskExecutionError) else "unexpected_exception"
          error_message = str(caught)
          trail.append(
            "attempt_failed",
            attempt,
            error_type=type(caught).__name__,
            error_code=code,
            error=error_message,
            transient=transient,
          )
          contained = await self._contain(trail, attempt)
          if not contained:
            status = TaskStatus.RECOVERY_FAILED
            break
          can_retry = transient and attempt < contract.recovery.max_attempts
          if not can_retry:
            status = TaskStatus.EXECUTION_FAILED
            break
          if not await self._recover_for_retry(contract, trail, attempt):
            status = TaskStatus.RECOVERY_FAILED
            break
          trail.append("retry_authorized", attempt, next_attempt=attempt + 1)
          continue

        post_checks = evaluate_conditions(
          contract.postconditions, WorkcellSnapshot.capture(self.workcell)
        )
        if all(check.passed for check in post_checks):
          status = TaskStatus.SUCCEEDED
          trail.append("postcondition_passed", attempt, checks=len(post_checks))
        else:
          status = TaskStatus.POSTCONDITION_FAILED
          error_message = "one or more visible postconditions failed"
          trail.append(
            "postcondition_failed",
            attempt,
            failed_checks=sum(not check.passed for check in post_checks),
          )
          await self._contain(trail, attempt)
        break

    after = WorkcellSnapshot.capture(self.workcell)
    raw_trace = getattr(backend, "trace", ())
    command_trace = tuple(dict(entry) for entry in raw_trace[trace_start:])
    trail.append(
      "task_finished",
      attempts=attempts,
      status=status.value,
      before_digest=_digest(before),
      after_digest=_digest(after),
      trace_digest=_digest(command_trace),
    )
    return TaskOutcome(
      run_id=run_id,
      task_id=contract.task_id,
      status=status,
      contract_digest=contract.digest,
      provenance=contract.provenance,
      evidence_mode=evidence_mode,
      hardware_validated=False,
      evidence_caveat=caveat,
      attempts=attempts,
      error=error_message,
      before=before,
      after=after,
      precondition_checks=pre_checks,
      postcondition_checks=post_checks,
      command_trace=command_trace,
      events=tuple(trail.events),
    )
