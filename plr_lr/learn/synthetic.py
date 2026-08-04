"""Arms that misbehave in known ways, so recovery of the known answer can be scored.

The simulation backend from phase 1 tracks perfectly: it arrives exactly where it
was told. That is the right default for testing control logic and the wrong one
for testing anything that learns, because a perfect arm leaves no residual to
learn and any fit against it is fitting noise.

:class:`MistrackingBackend` is the same backend with a deliberate, declared
defect: it reaches a fixed fraction of each commanded move, plus an offset, plus
optional noise. Because the defect is planted, it can be scored. Fit a
:class:`~plr_lr.learn.dynamics.ResidualDynamics` on episodes recorded from an arm
with ``gain=0.85`` and the fit either recovers something close to 0.85 or the
learning stack is broken. This is the same plant-and-recover validation the
vision phase uses for hand-eye calibration.

:func:`simulate_episode` skips the arm entirely and rolls actions through a
dynamics model directly, which is how the dynamics tests generate data with no
async and no backend at all.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Union

import numpy as np
from pylabrobot.arms.precise_flex.coords import PreciseFlexCartesianCoords
from pylabrobot.resources import Coordinate, Rotation

from ..arm.sim_backend import SimulationArmBackend
from .dynamics import DynamicsModel, KinematicDynamics
from .types import Action, Episode, Observation


class MistrackingBackend(SimulationArmBackend):
  """A simulated arm that does not go exactly where it was told.

  The achieved pose is ``current + (commanded - current) * gain + bias``, plus
  Gaussian noise of ``noise_mm`` on each translation axis when asked for. That
  is deliberately the same form as :class:`~plr_lr.learn.dynamics.
  KinematicDynamics`, so a residual fit on data from this arm is being asked to
  recover parameters that really are there.

  Args:
    gain: fraction of each commanded displacement actually achieved. A scalar,
      or four values for x, y, z, rz.
    bias: constant offset added to every move, in mm and rad.
    noise_mm: standard deviation of per-axis translation noise.
    seed: seed for the noise generator.
  """

  def __init__(
    self,
    gain: Union[float, Sequence[float]] = 1.0,
    bias: Optional[Sequence[float]] = None,
    noise_mm: float = 0.0,
    seed: Optional[int] = None,
    **kwargs,
  ) -> None:
    super().__init__(**kwargs)
    declared = KinematicDynamics(gain=gain, bias=bias)
    self.gain = declared.gain.copy()
    self.bias = declared.bias.copy()
    self.noise_mm = float(noise_mm)
    if not np.isfinite(self.noise_mm) or self.noise_mm < 0.0:
      raise ValueError("noise_mm must be finite and non-negative")
    self._noise_rng = np.random.default_rng(seed)

  def true_dynamics(self) -> KinematicDynamics:
    """The model a perfect fit would recover. For scoring, not for planning."""
    return KinematicDynamics(gain=self.gain, bias=self.bias)

  async def _move_cartesian(self, coords: PreciseFlexCartesianCoords, tag: str) -> None:
    current = np.array(
      [
        self._cartesian.location.x,
        self._cartesian.location.y,
        self._cartesian.location.z,
        math.radians(self._cartesian.rotation.z),
      ],
      float,
    )
    commanded = np.array(
      [
        coords.location.x,
        coords.location.y,
        coords.location.z,
        math.radians(coords.rotation.z),
      ],
      float,
    )
    achieved = current + (commanded - current) * self.gain + self.bias
    if self.noise_mm:
      achieved[:3] = achieved[:3] + self._noise_rng.normal(0.0, self.noise_mm, size=3)

    await super()._move_cartesian(
      PreciseFlexCartesianCoords(
        location=Coordinate(float(achieved[0]), float(achieved[1]), float(achieved[2])),
        rotation=Rotation(
          coords.rotation.x,
          coords.rotation.y,
          math.degrees(float(achieved[3])),
        ),
        orientation=coords.orientation,
      ),
      tag=tag,
    )


def park(backend: SimulationArmBackend, pose: Sequence[float]) -> None:
  """Place a simulated arm at a pose without driving it there.

  A fixture needs the arm to start where it says it starts. Commanding the move
  with ``move_to`` runs it through whatever defect the backend models, so a
  :class:`MistrackingBackend` lands somewhere else and the declared start pose
  becomes a fiction that every reported distance is measured from. Worse, the
  displacement is a function of the defect, so two backends under comparison
  start in different places and the comparison is confounded by the thing being
  compared.

  Writing the simulated state is something you can only do to a simulation, and
  is exactly what a fixture should do.
  """
  p = np.asarray(pose, float).reshape(4)
  current = backend._cartesian
  backend._cartesian = PreciseFlexCartesianCoords(
    location=Coordinate(float(p[0]), float(p[1]), float(p[2])),
    rotation=Rotation(current.rotation.x, current.rotation.y, math.degrees(float(p[3]))),
    orientation=current.orientation,
  )


def simulate_episode(
  dynamics: DynamicsModel,
  actions: Sequence[Action],
  start: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
  task: str = "synthetic",
  dt: float = 0.1,
) -> Episode:
  """Roll ``actions`` through ``dynamics`` and return the resulting episode.

  No arm, no async, no backend. Used to generate training data from a known
  model so that a fit can be checked against the answer it should have found.
  """
  pose = np.asarray(start, float).reshape(4)
  ep = Episode(task=task, metadata={"source": "synthetic"})
  for i, action in enumerate(actions):
    obs = Observation(
      t=i * dt, x=float(pose[0]), y=float(pose[1]), z=float(pose[2]), rz=float(pose[3])
    )
    ep.append(obs, action)
    pose = np.asarray(dynamics.predict(pose, action.as_delta(pose)), float).reshape(4)
  ep.final_observation = Observation(
    t=len(actions) * dt,
    x=float(pose[0]),
    y=float(pose[1]),
    z=float(pose[2]),
    rz=float(pose[3]),
  )
  return ep


def random_walk_actions(
  n: int,
  step_mm: float = 20.0,
  step_rad: float = 0.0,
  seed: int = 0,
) -> List[Action]:
  """A sequence of bounded random delta actions, for exciting a dynamics fit.

  Random exploration is a poor controller and good training data: it visits
  directions a goal-directed policy never would, which is what keeps a fitted
  model from being valid only along the path the demonstrations happened to take.
  """
  rng = np.random.default_rng(seed)
  out: List[Action] = []
  for _ in range(n):
    d = rng.uniform(-step_mm, step_mm, size=3)
    drz = float(rng.uniform(-step_rad, step_rad)) if step_rad else 0.0
    out.append(Action.delta(float(d[0]), float(d[1]), float(d[2]), drz))
  return out
