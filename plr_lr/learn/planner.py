"""Turn a dynamics model into a controller by searching over what it predicts.

A world model on its own does not move anything. It predicts. The planner is
what makes a prediction actionable: sample candidate action sequences, roll each
one forward through the model, keep the sequence whose predicted end state is
closest to the goal, execute only its first action, then replan from what
actually happened. That last part is what makes this robust, and it is also what
makes model quality easy to overstate: replanning every step hides moderate
model error in the success rate, and surfaces it in the number of steps taken.
Score both.

    planner = SamplingPlanner(goal=Goal.at(200, 0, 40), dynamics=model)
    episode = await rollout(arm, planner, envelope=envelope)

The search is random shooting with a fraction of samples aimed straight at the
goal. It is deliberately simple: it is vectorized numpy, it needs no gradients,
and it works against any object satisfying :class:`~plr_lr.learn.dynamics.
DynamicsModel`, learned or analytic.

The goal-biased fraction is one trajectory repeated, not a fraction of distinct
candidates, so ``n_samples`` overstates the search width. ``n_samples=256`` at
the default ``goal_bias=0.25`` searches 193 distinct sequences. Read
:attr:`SamplingPlanner.last_n_distinct` for the number that may be quoted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np

from .dynamics import DynamicsModel, KinematicDynamics
from .types import Action, Observation


def _signed_rz_error(target: float, actual: float) -> float:
  """Shortest signed rotation from ``actual`` to ``target``, in radians."""
  return float(np.arctan2(np.sin(target - actual), np.cos(target - actual)))


def n_distinct_sequences(sequences: np.ndarray) -> int:
  """How many of the candidate action sequences are actually different.

  Drawing ``n`` candidates and evaluating ``n`` rows is not the same as searching
  ``n`` candidates, and only the second one is a search width. The goal-biased
  rows of :meth:`SamplingPlanner._sample` are all overwritten with the *same*
  aimed trajectory, so a planner configured for 256 samples at a goal bias of
  0.25 evaluates 193 distinct sequences and 64 copies of one of them. Reporting
  the 256 overstates the search by a third, and it overstates it silently,
  because every downstream number is an argmin over rows and an argmin does not
  care that most of its input was duplicated.

  Compared bit for bit rather than within a tolerance. The duplicates are literal
  copies of one array, so exact comparison finds every one of them, and two
  independently drawn candidates that agreed to the last bit would be the same
  motion anyway.
  """
  flat = np.asarray(sequences, float)
  return int(np.unique(flat.reshape(flat.shape[0], -1), axis=0).shape[0])


@dataclass(frozen=True)
class Goal:
  """Where the tool should end up, and how close counts as arrived."""

  pose: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
  tolerance_mm: float = 2.0
  tolerance_rad: float = np.pi  # by default rotation is unconstrained
  rz_weight: float = 0.0  # mm of cost per radian of rz error during search

  def __post_init__(self) -> None:
    """Canonicalize the goal and reject values that cannot define a real target."""
    try:
      pose = tuple(float(value) for value in self.pose)
    except (TypeError, ValueError, OverflowError) as exc:
      raise ValueError("goal pose must contain exactly 4 finite numbers") from exc
    if len(pose) != 4 or not np.all(np.isfinite(pose)):
      raise ValueError("goal pose must contain exactly 4 finite numbers")

    try:
      tolerance_mm = float(self.tolerance_mm)
      tolerance_rad = float(self.tolerance_rad)
      rz_weight = float(self.rz_weight)
    except (TypeError, ValueError, OverflowError) as exc:
      raise ValueError("goal tolerances and rz_weight must be finite numbers") from exc
    if not np.isfinite(tolerance_mm) or tolerance_mm < 0.0:
      raise ValueError("tolerance_mm must be finite and non-negative")
    if not np.isfinite(tolerance_rad) or not 0.0 <= tolerance_rad <= np.pi:
      raise ValueError("tolerance_rad must be finite and between 0 and pi")
    if not np.isfinite(rz_weight) or rz_weight < 0.0:
      raise ValueError("rz_weight must be finite and non-negative")

    object.__setattr__(self, "pose", pose)
    object.__setattr__(self, "tolerance_mm", tolerance_mm)
    object.__setattr__(self, "tolerance_rad", tolerance_rad)
    object.__setattr__(self, "rz_weight", rz_weight)

  @classmethod
  def at(
    cls,
    x: float,
    y: float,
    z: float,
    rz: float = 0.0,
    tolerance_mm: float = 2.0,
    tolerance_rad: float = np.pi,
    rz_weight: float = 0.0,
  ) -> "Goal":
    return cls((x, y, z, rz), tolerance_mm, tolerance_rad, rz_weight)

  @property
  def vector(self) -> np.ndarray:
    return np.array(self.pose, float)

  def translation_error(self, pose: Sequence[float]) -> np.ndarray:
    """Distance in mm from the goal, for one pose or a batch of them."""
    p = np.atleast_2d(np.asarray(pose, float))
    d = np.linalg.norm(p[:, :3] - self.vector[:3], axis=1)
    return d if np.asarray(pose, float).ndim > 1 else d[0]

  def rz_error(self, pose: Sequence[float]) -> np.ndarray:
    """Absolute rz error wrapped to [0, pi], for one pose or a batch."""
    p = np.atleast_2d(np.asarray(pose, float))
    d = np.abs(np.arctan2(np.sin(p[:, 3] - self.pose[3]), np.cos(p[:, 3] - self.pose[3])))
    return d if np.asarray(pose, float).ndim > 1 else d[0]

  def cost(self, pose: Sequence[float]) -> np.ndarray:
    """Search objective: millimetres, with rz folded in at ``rz_weight``."""
    return self.translation_error(pose) + self.rz_weight * self.rz_error(pose)

  def reached(self, pose: Sequence[float]) -> bool:
    return bool(
      self.translation_error(pose) <= self.tolerance_mm
      and self.rz_error(pose) <= self.tolerance_rad
    )


class SamplingPlanner:
  """Model-predictive control by random shooting. Satisfies ``Policy``.

  Args:
    goal: where to drive the tool.
    dynamics: the model to plan against. Defaults to
      :class:`~plr_lr.learn.dynamics.KinematicDynamics`, which assumes the arm
      tracks perfectly.
    horizon: how many steps ahead each candidate sequence is rolled.
    n_samples: candidate sequences drawn per step. Not the search width: the
      goal-biased fraction is one trajectory repeated, so ``last_n_distinct``
      after a :meth:`plan` call is the number that was actually searched and is
      the only one of the two that may be quoted as such.
    step_mm: largest displacement any single sampled step may command.
    goal_bias: fraction of candidates that aim at the goal rather than being
      drawn at random. Pure random shooting converges slowly in four dimensions;
      this keeps the search usable at small sample counts. Those candidates are
      identical to each other, so raising this lowers ``last_n_distinct``.
    refine_iters: how many times a goal-directed candidate is corrected through
      the dynamics model before being proposed. This is the planner asking the
      model the useful question. Scoring only asks "where would this go", which
      a mediocre model answers well enough to rank candidates. Refining asks
      "what should I command to arrive there", and a model that is wrong about
      the arm's gain or offset gets that visibly wrong. Set to 0 to aim
      geometrically and let scoring do all the work.
    action_cost: penalty per mm of commanded motion, which discourages
      thrashing when several paths reach the goal.
  """

  def __init__(
    self,
    goal: Goal,
    dynamics: Optional[DynamicsModel] = None,
    horizon: int = 4,
    n_samples: int = 256,
    step_mm: float = 25.0,
    step_rad: float = 0.0,
    goal_bias: float = 0.25,
    refine_iters: int = 3,
    action_cost: float = 0.0,
    seed: int = 0,
  ) -> None:
    if isinstance(horizon, bool) or not isinstance(horizon, (int, np.integer)) or horizon < 1:
      raise ValueError("horizon must be at least 1")
    if (
      isinstance(n_samples, bool)
      or not isinstance(n_samples, (int, np.integer))
      or n_samples < 1
    ):
      raise ValueError("n_samples must be at least 1")
    if (
      isinstance(refine_iters, bool)
      or not isinstance(refine_iters, (int, np.integer))
      or refine_iters < 0
    ):
      raise ValueError("refine_iters must be a non-negative integer")

    step_mm = float(step_mm)
    step_rad = float(step_rad)
    goal_bias = float(goal_bias)
    action_cost = float(action_cost)
    if not np.isfinite(step_mm) or step_mm <= 0.0:
      raise ValueError("step_mm must be positive and finite")
    if not np.isfinite(step_rad) or step_rad < 0.0:
      raise ValueError("step_rad must be non-negative and finite")
    if not np.isfinite(goal_bias) or not 0.0 <= goal_bias <= 1.0:
      raise ValueError("goal_bias must be between 0 and 1")
    if not np.isfinite(action_cost) or action_cost < 0.0:
      raise ValueError("action_cost must be non-negative and finite")
    rotation_matters = goal.tolerance_rad < np.pi or goal.rz_weight > 0.0
    if rotation_matters and step_rad == 0.0:
      raise ValueError(
        "step_rad must be positive when the goal constrains or scores wrist rotation"
      )
    self.goal = goal
    self.dynamics: DynamicsModel = dynamics or KinematicDynamics()
    self.horizon = int(horizon)
    self.n_samples = int(n_samples)
    self.step_mm = step_mm
    self.step_rad = step_rad
    self.goal_bias = goal_bias
    self.refine_iters = int(refine_iters)
    self.action_cost = action_cost
    self.seed = seed
    self.rng = np.random.default_rng(seed)
    self.last_predicted_cost: Optional[float] = None
    #: Distinct candidate sequences in the last :meth:`plan` call, which is not
    #: ``n_samples``. See :func:`n_distinct_sequences` for why the difference is
    #: worth carrying: ``n_samples`` is what was drawn, this is what was searched.
    self.last_n_distinct: Optional[int] = None

  def reset(self) -> None:
    self.rng = np.random.default_rng(self.seed)
    self.last_predicted_cost = None
    self.last_n_distinct = None

  # -- search ----------------------------------------------------------------

  def _sample(self, pose: np.ndarray) -> np.ndarray:
    """Candidate action sequences, shape ``(n_samples, horizon, 4)``.

    Translations are sampled inside the ball of radius ``step_mm``, not the cube,
    so that no candidate asks for more than a safety envelope configured to the
    same ``max_step_mm`` will allow. Sampling the cube would put a corner of
    ``step_mm * sqrt(3)`` on the table, the envelope would quietly scale every
    such step down, and the planner would be scoring motions the arm never made.

    Radii are drawn log-uniformly across four decades rather than all at
    ``step_mm``, so the coarse move that crosses the deck and the fine move that
    finishes a tight approach are both always on the table. Shrinking the whole
    sampling radius as the goal gets closer looks like the tidier answer and is
    wrong: an arm with a constant offset needs a command *larger* than the
    distance remaining in order to land, and a radius capped at that distance
    puts the goal permanently out of reach.

    ``round(n_samples * goal_bias)`` of the returned rows are overwritten with
    one identical aimed trajectory, so the array is ``n_samples`` rows wide and
    fewer than ``n_samples`` candidates deep. That is deliberate -- the aimed
    trajectory is the one the planner most wants scored, and duplicating it costs
    nothing but vectorised arithmetic -- and it is exactly why
    :attr:`last_n_distinct` exists. Do not deduplicate here to make the number
    match: the shape is load-bearing for the vectorised rollout in
    :meth:`plan`, and it is the reported width, not the sampling, that was wrong.
    """
    n, h = self.n_samples, self.horizon
    lo, hi = np.log(max(self.step_mm * 1e-4, 1e-12)), np.log(max(self.step_mm, 1e-12))
    radii = np.exp(self.rng.uniform(lo, hi, size=(n, h, 1)))
    dirs = self.rng.normal(size=(n, h, 3))
    dirs = dirs / np.maximum(np.linalg.norm(dirs, axis=2, keepdims=True), 1e-12)

    seqs = np.zeros((n, h, 4))
    seqs[:, :, :3] = dirs * radii
    if self.step_rad:
      seqs[:, :, 3] = self.rng.uniform(-self.step_rad, self.step_rad, size=(n, h))

    n_biased = int(round(n * self.goal_bias))
    translation_error = float(np.linalg.norm(self.goal.vector[:3] - pose[:3]))
    rotation_error = abs(_signed_rz_error(self.goal.pose[3], float(pose[3])))
    if n_biased and (translation_error > 0.0 or rotation_error > 0.0):
      # A goal-biased candidate has to be a goal-biased *trajectory*. Replacing
      # only its first action and leaving the rest random makes terminal scoring
      # punish the directed move for three random moves that follow it. Roll the
      # model forward after every inverse-aimed step so the entire horizon says
      # what the planner actually means: approach the goal, then stay there.
      predicted = pose.copy()
      for i in range(h):
        aimed = self._aim(predicted, self.goal.vector)
        seqs[:n_biased, i, :] = aimed
        predicted = np.asarray(self.dynamics.predict(predicted, aimed), float).reshape(4)
    return seqs

  def _clip_step(self, delta: np.ndarray) -> np.ndarray:
    """Bound translation and wrist rotation without changing their directions."""
    bounded = np.asarray(delta, float).reshape(4).copy()
    magnitude = float(np.linalg.norm(bounded[:3]))
    if magnitude > self.step_mm:
      bounded[:3] = bounded[:3] * (self.step_mm / magnitude)
    bounded[3] = float(np.clip(bounded[3], -self.step_rad, self.step_rad))
    return bounded

  def _aim(self, pose: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ask the model what to command in order to arrive at ``target``.

    Starts from the geometric guess, which is what you would command if the arm
    tracked perfectly, then corrects it by whatever the model says that guess
    would miss by. Each pass removes a further fraction of the error, so a few
    passes converge for any model that is roughly right about direction, and no
    passes at all reduces to aiming geometrically.
    """
    delta = np.zeros(4)
    delta[:3] = target[:3] - pose[:3]
    delta[3] = _signed_rz_error(float(target[3]), float(pose[3]))
    delta = self._clip_step(delta)
    for _ in range(self.refine_iters):
      predicted = np.asarray(self.dynamics.predict(pose, delta), float).reshape(4)
      delta = delta.copy()
      delta[:3] = delta[:3] + (target[:3] - predicted[:3])
      delta[3] = delta[3] + _signed_rz_error(float(target[3]), float(predicted[3]))
      delta = self._clip_step(delta)
    return delta

  def plan(self, pose: Sequence[float]) -> Tuple[np.ndarray, float]:
    """Return the best first displacement and its predicted terminal cost.

    Also records :attr:`last_n_distinct`, the number of candidates that were
    actually different from one another. A caller reporting search width has to
    report that number and not ``n_samples``.
    """
    p0 = np.asarray(pose, float).reshape(4)
    seqs = self._sample(p0)
    self.last_n_distinct = n_distinct_sequences(seqs)

    poses = np.repeat(p0[None, :], self.n_samples, axis=0)
    effort = np.zeros(self.n_samples)
    for h in range(self.horizon):
      step = seqs[:, h, :]
      poses = np.atleast_2d(self.dynamics.predict(poses, step))
      effort += np.linalg.norm(step[:, :3], axis=1)

    scores = self.goal.cost(poses) + self.action_cost * effort
    best = int(np.argmin(scores))
    return seqs[best, 0, :], float(scores[best])

  # -- Policy ----------------------------------------------------------------

  def act(self, observation: Observation) -> Action:
    pose = observation.pose_vector()
    if self.goal.reached(pose):
      self.last_predicted_cost = float(self.goal.cost(pose))
      # No search ran on this step, so the previous step's width must not be left
      # standing where a caller would read it as this step's.
      self.last_n_distinct = None
      return Action.hold(terminate=True)
    delta, cost = self.plan(pose)
    self.last_predicted_cost = cost
    return Action.delta(
      dx=float(delta[0]), dy=float(delta[1]), dz=float(delta[2]), drz=float(delta[3])
    )
