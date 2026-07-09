"""Phase 3: uncap a threaded tube, then screw the cap back on.

Each skill is gated: it checks a precondition, runs the motion, and verifies the
result. The decap rotates the wrist counter-clockwise while lifting by the thread
pitch per turn; recap reverses it.

    python examples/03_decap_recap.py
"""

import asyncio

from plr_lr import Labware, Workcell
from plr_lr.manipulation import DecapSkill, RecapSkill


async def main() -> None:
  wc = Workcell.sim()
  await wc.setup()

  tube = Labware(
    name="sample_tube",
    kind="microtube",
    capped=True,
    cap_diameter=9.0,
    thread_pitch_mm=1.5,
    cap_turns=2.5,
    grip_z=6.0,
  )
  wc.add_site("rack_a1", x=120, y=0, z=8, occupant=tube)
  wc.add_site("cap_park", x=120, y=60, z=8)

  print(f"start: capped={tube.capped}")

  decap = DecapSkill(wc.arm, wc.world, "sample_tube",
                     coords=wc.coords_for("rack_a1"), cap_park=wc.coords_for("cap_park"))
  res = await decap.run()
  print(f"decap: ok={res.ok} ({res.message}); rotated {res.data['rotation_deg']:.0f} deg, "
        f"lifted {res.data['lift_mm']:.2f} mm; capped={tube.capped}")

  recap = RecapSkill(wc.arm, wc.world, "sample_tube", coords=wc.coords_for("rack_a1"))
  res = await recap.run()
  print(f"recap: ok={res.ok} ({res.message}); capped={tube.capped}")


if __name__ == "__main__":
  asyncio.run(main())
