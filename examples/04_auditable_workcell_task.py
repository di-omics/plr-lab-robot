"""Phase 4: execute a domain-authored workcell task contract in simulation.

The contract identifies the protocol and samples, declares visible preconditions
and postconditions, and produces an integrity-checkable JSON outcome. This is a
software/simulation artifact, not evidence that physical hardware was validated.

    python examples/04_auditable_workcell_task.py
"""

import asyncio

from plr_lr import Labware, Workcell
from plr_lr.tasking import ProvenanceContext, TaskContract, TaskExecutor


async def main() -> None:
  workcell = Workcell.sim()
  await workcell.setup()
  workcell.add_site(
    "incubator_out",
    x=180,
    y=0,
    z=12,
    occupant=Labware(name="assay_plate"),
  )
  workcell.add_site("reader_in", x=-180, y=40, z=12)

  contract = TaskContract.move_labware(
    task_id="reader-load-001",
    intent="Load the tracked assay plate into the reader",
    labware_name="assay_plate",
    source_site="incubator_out",
    destination_site="reader_in",
    provenance=ProvenanceContext(
      requested_by="assay-scientist",
      protocol_id="viability-read",
      protocol_version="1.0",
      sample_ids=("sample-A", "sample-B"),
    ),
  )

  outcome = await TaskExecutor(workcell).execute(contract)
  print(f"status: {outcome.status.value}; integrity: {outcome.verify_integrity()}")
  print(f"evidence: {outcome.evidence_mode}; hardware validated: {outcome.hardware_validated}")
  print(outcome.to_json())


if __name__ == "__main__":
  asyncio.run(main())
