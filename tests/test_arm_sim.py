import unittest

from pylabrobot.arms import ExperimentalSCARA

from plr_lr import (
  ArmNotHomedError,
  GripperError,
  Labware,
  Pose2D,
  SimulationArmBackend,
  Workcell,
  Workspace,
  WorkspaceViolation,
  site_coords,
)


def coords(x, y, z=10.0):
  return site_coords(Pose2D(x, y, z))


class ArmSimTests(unittest.IsolatedAsyncioTestCase):
  async def test_home_interlock_blocks_motion(self):
    arm = ExperimentalSCARA(backend=SimulationArmBackend())
    await arm.setup()  # set up but not homed
    with self.assertRaises(ArmNotHomedError):
      await arm.pick_up_resource(coords(100, 100), plate_width=78.0)

  async def test_pick_then_drop_updates_state_and_trace(self):
    wc = Workcell.sim()
    await wc.setup()
    plate = Labware(name="sample_plate")
    wc.add_site("source", x=150, y=0, z=10, occupant=plate)
    wc.add_site("dest", x=-150, y=0, z=10)

    self.assertFalse(wc.backend.holding)
    moved = await wc.move_plate("source", "dest")
    self.assertEqual(moved, "sample_plate")
    self.assertFalse(wc.backend.holding)  # released at destination
    self.assertIsNone(wc.world.sites["source"].occupant)
    self.assertEqual(wc.world.sites["dest"].occupant, "sample_plate")

    actions = [e["action"] for e in wc.backend.trace]
    for expected in ("home", "open_gripper", "descend_pick", "close_gripper", "descend_drop"):
      self.assertIn(expected, actions)

  async def test_double_pick_raises(self):
    wc = Workcell.sim()
    await wc.setup()
    a = Labware(name="a")
    b = Labware(name="b")
    wc.add_site("s1", 120, 0, 10, occupant=a)
    wc.add_site("s2", 120, 80, 10, occupant=b)
    await wc.arm.pick_up_resource(wc.coords_for("s1"), plate_width=a.plate_width)
    with self.assertRaises(GripperError):
      await wc.arm.pick_up_resource(wc.coords_for("s2"), plate_width=b.plate_width)

  async def test_workspace_violation(self):
    wc = Workcell.sim(workspace=Workspace(x=(-200, 200), y=(-200, 200), z=(0, 300)))
    await wc.setup()
    wc.add_site("far", x=999, y=0, z=10)
    with self.assertRaises(WorkspaceViolation):
      await wc.arm.move_to(wc.coords_for("far"))


if __name__ == "__main__":
  unittest.main()
