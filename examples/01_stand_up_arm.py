"""Phase 1: stand up the arm and move a plate between two deck sites.

Runs entirely in simulation (no hardware). The readiness report and the last 2
prints read `is_set_up`, `homed`, `holding` and `trace`, which only
SimulationArmBackend provides.

    python examples/01_stand_up_arm.py
"""

import asyncio

from plr_lr import Labware, Workcell
from plr_lr.qc import arm_readiness


async def main() -> None:
  wc = Workcell.sim()
  await wc.setup()  # connect + home

  report = await arm_readiness(wc)
  print(report.summary())

  plate = Labware(name="assay_plate")
  wc.add_site("incubator_out", x=180, y=0, z=12, occupant=plate)
  wc.add_site("reader_in", x=-180, y=40, z=12)

  print(f"\nbefore: incubator_out holds {wc.world.sites['incubator_out'].occupant}")
  await wc.move_plate("incubator_out", "reader_in")
  print(f"after : reader_in holds {wc.world.sites['reader_in'].occupant}")
  print(f"arm holding: {wc.backend.holding}")
  print(f"commands issued: {len(wc.backend.trace)}")


if __name__ == "__main__":
  asyncio.run(main())
