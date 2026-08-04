"""Unit-boundary tests for the deliberately imperfect simulation backend."""

from __future__ import annotations

import math
import unittest

import numpy as np
from pylabrobot.arms import ExperimentalSCARA
from pylabrobot.arms.precise_flex.coords import PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from plr_lr.learn import MistrackingBackend, observe
from plr_lr.learn.synthetic import park


def coords(x: float, y: float, z: float, rz_rad: float) -> PreciseFlexCartesianCoords:
  """Build PyLabRobot coordinates from a learning-native rotation in radians."""
  return PreciseFlexCartesianCoords(
    location=Coordinate(x, y, z),
    rotation=Rotation(0.0, 0.0, math.degrees(rz_rad)),
    orientation=None,
  )


class MistrackingUnitBoundaryTests(unittest.IsolatedAsyncioTestCase):
  async def test_invalid_planted_parameters_are_refused_at_construction(self):
    for kwargs in (
      {"gain": float("nan")},
      {"bias": (0.0, 0.0, 0.0, float("inf"))},
      {"noise_mm": -0.1},
      {"noise_mm": float("nan")},
    ):
      with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
        MistrackingBackend(**kwargs)

  async def test_park_accepts_learning_radians_and_stores_pylabrobot_degrees(self):
    backend = MistrackingBackend()
    arm = ExperimentalSCARA(backend=backend)
    await arm.setup()
    await arm.home()

    park(backend, (10.0, -20.0, 30.0, 0.75))

    raw = await arm.get_cartesian_position()
    self.assertAlmostEqual(raw.rotation.z, math.degrees(0.75))
    recorded = await observe(arm)
    self.assertAlmostEqual(recorded.rz, 0.75)

  async def test_backend_applies_rotational_gain_and_bias_in_radians(self):
    gain = (1.0, 1.0, 1.0, 0.5)
    bias = (0.0, 0.0, 0.0, 0.1)
    backend = MistrackingBackend(gain=gain, bias=bias)
    arm = ExperimentalSCARA(backend=backend)
    await arm.setup()
    await arm.home()
    park(backend, (0.0, 0.0, 100.0, 0.5))

    await arm.move_to(coords(0.0, 0.0, 100.0, 1.0))

    # 0.5 + (1.0 - 0.5) * 0.5 + 0.1 = 0.85 radians.
    recorded = await observe(arm)
    self.assertAlmostEqual(recorded.rz, 0.85, places=12)
    raw = await arm.get_cartesian_position()
    self.assertAlmostEqual(raw.rotation.z, math.degrees(0.85), places=12)

    predicted = backend.true_dynamics().predict(
      np.array([0.0, 0.0, 100.0, 0.5]),
      np.array([0.0, 0.0, 0.0, 0.5]),
    )
    self.assertAlmostEqual(float(predicted[3]), recorded.rz, places=12)


if __name__ == "__main__":
  unittest.main()
