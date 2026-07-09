import unittest

from plr_lr import Labware, Pose2D, Workcell, site_coords
from plr_lr.manipulation import DecapSkill, RecapSkill


class DecapTests(unittest.IsolatedAsyncioTestCase):
  async def _capped_tube_workcell(self):
    wc = Workcell.sim()
    await wc.setup()
    tube = Labware(
      name="tube",
      kind="microtube",
      capped=True,
      cap_diameter=9.0,
      thread_pitch_mm=1.5,
      cap_turns=2.5,
      grip_z=6.0,
    )
    wc.add_site("rack", x=120, y=0, z=8, occupant=tube)
    wc.add_site("cap_park", x=120, y=60, z=8)
    return wc, tube

  async def test_decap_removes_and_verifies(self):
    wc, tube = await self._capped_tube_workcell()
    skill = DecapSkill(
      wc.arm, wc.world, "tube",
      coords=wc.coords_for("rack"), cap_park=wc.coords_for("cap_park"),
    )
    result = await skill.run()
    self.assertTrue(result.ok, result.message)
    self.assertFalse(tube.capped)
    self.assertAlmostEqual(result.data["rotation_deg"], 2.5 * 360.0, places=6)
    self.assertAlmostEqual(result.data["lift_mm"], 2.5 * 1.5, places=6)
    self.assertIn("tube", wc.world.parking)

  async def test_precondition_blocks_open_vessel(self):
    wc, tube = await self._capped_tube_workcell()
    tube.capped = False
    skill = DecapSkill(
      wc.arm, wc.world, "tube",
      coords=wc.coords_for("rack"), cap_park=wc.coords_for("cap_park"),
    )
    result = await skill.run()
    self.assertFalse(result.ok)
    self.assertIn("already open", result.message)

  async def test_decap_then_recap_roundtrip(self):
    wc, tube = await self._capped_tube_workcell()
    await DecapSkill(
      wc.arm, wc.world, "tube",
      coords=wc.coords_for("rack"), cap_park=wc.coords_for("cap_park"),
    ).run()
    self.assertFalse(tube.capped)

    recap = await RecapSkill(wc.arm, wc.world, "tube", coords=wc.coords_for("rack")).run()
    self.assertTrue(recap.ok, recap.message)
    self.assertTrue(tube.capped)
    self.assertNotIn("tube", wc.world.parking)


if __name__ == "__main__":
  unittest.main()
