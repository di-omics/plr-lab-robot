"""Models of what the arm will do next, including one you can fit from your own data.

A planner is only as good as the model it plans against, so this module keeps
two things strictly apart:

* :class:`KinematicDynamics` is the assumption every open-loop script already
  makes: the arm goes exactly where it was told. It needs no data.
* :class:`ResidualDynamics` learns the difference between that assumption and
  what your arm actually did, from episodes you recorded. It is a linear least
  squares fit on ``[pose, delta, 1]``, which is small enough to run on a laptop
  CPU in milliseconds and large enough to capture the systematic errors that
  matter in practice: a consistent undershoot, an axis that lags, a delta-
  proportional loss from backlash or compliance.

The fit refuses to flatter itself. Every fit holds out a fraction of the
transitions, scores both the learned model and the unlearned baseline on that
holdout, and reports both numbers. ``DynamicsFit.improved`` is the only claim
worth making, and it is false whenever the learned model did not actually beat
doing nothing. Fitting fewer transitions than the model has parameters is
refused outright, because that fit will always look perfect on training data and
tell you nothing.

``improved`` is three-valued, not two. Recording one episode leaves nothing to
split by episode, so the holdout becomes neighbouring frames of the same motion
and the claim is not evaluable at all; that case returns ``None`` and prints
``SPLIT-UNSOUND`` rather than a verdict. ``None`` is not ``False``: the data did
not refute the correction, it was never in a position to test it.

    model, fit = ResidualDynamics.fit(dataset)
    print(fit.summary())
    if not fit.improved:
        ...  # keep the baseline; the data did not support a learned correction

That guard reads correctly for all three values, because ``not None`` is True:
an unevaluable fit keeps the baseline, which is the same thing a refuted fit
does. Only ``fit.improved is True`` may be read as evidence.
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Sequence, Tuple, Union, runtime_checkable

import numpy as np

from .dataset import Dataset, episodes_from
from .types import Episode

N_FEATURES = 9  # pose (4) + delta (4) + bias (1)
N_OUTPUTS = 4

#: Relative singular-value cutoff for the least squares solve. See the note in
#: :meth:`ResidualDynamics.fit` for why the numpy default is the wrong choice here.
RCOND = 1e-8

#: Names of the design-matrix columns, in order, for reporting which directions
#: the demonstrations left unexcited.
FEATURE_NAMES = ("x", "y", "z", "rz", "dx", "dy", "dz", "drz", "bias")


class InsufficientData(ValueError):
  """Raised when a fit has too few transitions to mean anything."""


class DynamicsAdapterError(ValueError):
  """Raised when an injected dynamics predictor breaks the adapter contract."""


@runtime_checkable
class DynamicsModel(Protocol):
  """Predicts the pose that follows a commanded displacement."""

  def predict(self, pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Map ``(pose, delta)`` to the next pose. Both accept a batch leading axis."""
    ...


def _wrap(angles: np.ndarray) -> np.ndarray:
  """Wrap radians to [-pi, pi] so an error of 2 pi reads as 0, not as 6.28."""
  return np.arctan2(np.sin(angles), np.cos(angles))


def _rmse(pred: np.ndarray, truth: np.ndarray) -> Tuple[float, float]:
  """Translation RMSE in mm and rotation RMSE in rad, kept separate on purpose."""
  translation = float(np.sqrt(np.mean(np.sum((pred[:, :3] - truth[:, :3]) ** 2, axis=1))))
  rotation = float(np.sqrt(np.mean(_wrap(pred[:, 3] - truth[:, 3]) ** 2)))
  return translation, rotation


def _fingerprint(poses: np.ndarray, deltas: np.ndarray) -> frozenset:
  """Identify the rows a model was fit on, so scoring them again can be caught."""
  rows = np.round(np.concatenate([poses, deltas], axis=1), 9)
  return frozenset(map(tuple, rows.tolist()))


def _pair(pose: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, bool]:
  pose_arr, delta_arr = np.asarray(pose, float), np.asarray(delta, float)
  if pose_arr.ndim > 2 or delta_arr.ndim > 2:
    raise ValueError(
      f"pose and delta must be a 4-vector or a batch of them, got shapes "
      f"{pose_arr.shape} and {delta_arr.shape}. A 3-D input would broadcast into a "
      "prediction of the wrong shape and fail somewhere further downstream."
    )
  p = np.atleast_2d(pose_arr)
  d = np.atleast_2d(delta_arr)
  single = pose_arr.ndim == 1 and delta_arr.ndim == 1
  if p.shape[0] == 1 and d.shape[0] > 1:
    p = np.repeat(p, d.shape[0], axis=0)
  if d.shape[0] == 1 and p.shape[0] > 1:
    d = np.repeat(d, p.shape[0], axis=0)
  if p.shape[1] != 4 or d.shape[1] != 4:
    raise ValueError(f"pose and delta must be 4-vectors, got {p.shape} and {d.shape}")
  return p, d, single


def _features(pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
  """Design matrix rows: ``[pose, delta, 1]``."""
  return np.concatenate([pose, delta, np.ones((pose.shape[0], 1))], axis=1)


class KinematicDynamics:
  """The arm goes where it was told, scaled by a per-axis tracking ``gain``.

  ``gain`` of 1.0 is perfect tracking. A gain below 1.0 on an axis says the arm
  consistently falls short of a commanded move on that axis, which is the single
  most common real deviation and the one worth being able to state explicitly
  without any data.
  """

  def __init__(
    self,
    gain: Union[float, Sequence[float]] = 1.0,
    bias: Optional[Sequence[float]] = None,
  ) -> None:
    try:
      gain_array = np.broadcast_to(np.asarray(gain, float), (4,)).astype(float).copy()
      bias_array = (
        np.zeros(4) if bias is None else np.asarray(bias, float).reshape(4).copy()
      )
    except (TypeError, ValueError) as exc:
      raise ValueError("gain and bias must be numeric scalars or 4-vectors") from exc
    if not np.all(np.isfinite(gain_array)) or not np.all(np.isfinite(bias_array)):
      raise ValueError("gain and bias must contain only finite values")
    self.gain = gain_array
    self.bias = bias_array

  def predict(self, pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    p, d, single = _pair(pose, delta)
    out = p + d * self.gain + self.bias
    return out[0] if single else out

  def __repr__(self) -> str:
    return f"KinematicDynamics(gain={np.round(self.gain, 4).tolist()})"


#: How much better than the baseline a fit has to be before it may say so. A bare
#: strict comparison calls a coin flip an improvement: with nothing to learn, the
#: fitted model lands on either side of the baseline about half the time, and half
#: of those runs print the word "improved".
IMPROVEMENT_MARGIN = 0.05

#: Printed in place of a verdict when the holdout cannot support one. A
#: transition-level split is the only way this fit produces one, and it produces
#: it silently: a single recorded episode has no second episode to hold out, so
#: the split falls back to picking rows at random out of one continuous motion.
SPLIT_UNSOUND = "SPLIT-UNSOUND"

#: The one split that makes ``improved`` answerable. Named rather than inlined so
#: that adding a third split strategy has to decide, explicitly, whether the new
#: strategy is sound enough to license a claim.
SOUND_SPLIT = "episode"

_EPS = 1e-12


@dataclass
class DynamicsFit:
  """What a fit is entitled to claim about itself.

  Translation and rotation error are kept apart. Summing them into one number
  adds radians to millimetres, and since a full radian of wrist error then counts
  for about as much as one millimetre of travel, the combined figure is close to
  blind to rotation while being labelled in mm.
  """

  n_train: int
  n_holdout: int
  train_rmse: float  # translation, mm
  holdout_rmse: float  # translation, mm
  baseline_holdout_rmse: float  # translation, mm
  train_rotation_rmse: float = 0.0  # rad
  holdout_rotation_rmse: float = 0.0  # rad
  baseline_holdout_rotation_rmse: float = 0.0  # rad
  # SOUND_SPLIT, or "transition" for a single-episode fit. Read by `improved`.
  split_by: str = SOUND_SPLIT
  rank: int = N_FEATURES
  unexcited: Tuple[str, ...] = ()  # inputs the demonstrations never varied

  @property
  def split_is_sound(self) -> bool:
    """Whether the holdout rows are independent enough to license a claim."""
    return self.split_by == SOUND_SPLIT

  @property
  def improved(self) -> Optional[bool]:
    """Did the learned model beat the unlearned baseline on held-out data?

    Three-valued. ``True`` and ``False`` require a margin on translation and no
    regression in rotation, scored on whole episodes the fit never touched. A
    baseline that is already exact leaves nothing to improve on, and says so.

    ``None`` means the question was not answerable. A single-episode fit splits
    over transitions, so frame 41 trains and frame 42 is held out: two samples of
    one motion a fortieth of a second apart. Measured on this codebase, that
    split calls pure encoder noise an improvement on about five percent of runs,
    which is exactly the sort of number that gets quoted. Returning ``False``
    there would be its own lie, because it reads as "the data refuted this" when
    the truth is that no test was performed. Record a second episode.
    """
    if not self.split_is_sound:
      return None
    if self.baseline_holdout_rmse <= _EPS:
      return False
    if self.holdout_rmse > self.baseline_holdout_rmse * (1.0 - IMPROVEMENT_MARGIN):
      return False
    return self.holdout_rotation_rmse <= self.baseline_holdout_rotation_rmse + _EPS

  @property
  def improvement_ratio(self) -> float:
    """Holdout error as a fraction of the baseline's. Below 1.0 is an improvement."""
    if self.baseline_holdout_rmse <= _EPS:
      return float("inf") if self.holdout_rmse > _EPS else 1.0
    return float(self.holdout_rmse / self.baseline_holdout_rmse)

  def summary(self) -> str:
    improved = self.improved
    if improved is None:
      verdict = f"{SPLIT_UNSOUND} (no verdict)"
    else:
      verdict = "improved" if improved else "NO IMPROVEMENT"
    lines = [
      f"dynamics fit: {verdict}",
      f"  transitions:  {self.n_train} train, {self.n_holdout} holdout "
      f"(split by {self.split_by}, design rank {self.rank}/{N_FEATURES})",
      f"  train:        {self.train_rmse:.4f} mm, {self.train_rotation_rmse:.5f} rad",
      f"  holdout:      {self.holdout_rmse:.4f} mm, {self.holdout_rotation_rmse:.5f} rad",
      f"  baseline:     {self.baseline_holdout_rmse:.4f} mm, "
      f"{self.baseline_holdout_rotation_rmse:.5f} rad "
      f"(translation ratio {self.improvement_ratio:.3f}, "
      f"must be below {1.0 - IMPROVEMENT_MARGIN:.2f} to count)",
    ]
    if self.unexcited:
      lines.append(
        f"  note: {', '.join(self.unexcited)} never varied in this data, so the model "
        "says nothing about them and their coefficients are pinned at zero."
      )
    if not self.split_is_sound:
      lines.append(
        f"  {SPLIT_UNSOUND}: one episode only, so train and holdout are neighbouring "
        "frames of the same motion. No improvement claim is made either way. Record "
        "a second episode and refit."
      )
    return "\n".join(lines)


class ResidualDynamics:
  """A base model plus a linear residual fit from recorded transitions."""

  #: Predictions further than this from the arm's own scale are refused. A
  #: residual model iterated by a planner can run away long before it reaches inf,
  #: and a finite but absurd pose is acted on silently where a NaN would not be.
  PLAUSIBLE_MM = 1e5

  _DIVERGENCE_HINT = (
    "A residual model iterated by a planner can diverge when it was fit on "
    "transitions that do not cover the states being planned through. Refit on data "
    "that reaches this part of the workspace, or plan against the base model."
  )

  def __init__(
    self,
    base: Optional[DynamicsModel] = None,
    weights: Optional[np.ndarray] = None,
  ) -> None:
    self.base: DynamicsModel = base or KinematicDynamics()
    self.weights = None if weights is None else np.asarray(weights, float)
    #: The transitions this model was fit on, set by :meth:`fit`. Used to catch
    #: a later attempt to score the model on its own training data.
    self.trained_on: Optional[frozenset] = None

  def predict(self, pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    p, d, single = _pair(pose, delta)
    out = np.atleast_2d(self.base.predict(p, d))
    if self.weights is not None:
      # ndarray.dot rather than the @ operator: on macOS/Accelerate builds of
      # numpy 2.0, matmul emits spurious divide-by-zero and overflow warnings
      # from its padding lanes even for finite inputs, and a planner calling
      # this thousands of times floods the console with them. Same result, and
      # genuine divergence is caught by the explicit check below rather than by
      # a platform-dependent warning.
      out = out + _features(p, d).dot(self.weights)
    if not np.all(np.isfinite(out)):
      raise FloatingPointError(
        "dynamics prediction is not finite. " + self._DIVERGENCE_HINT
      )
    if np.any(np.abs(out[:, :3]) > self.PLAUSIBLE_MM):
      raise FloatingPointError(
        f"dynamics prediction left the plausible range for a lab arm "
        f"(more than {self.PLAUSIBLE_MM:.0f} mm from the origin). A diverging model "
        f"passes through large finite nonsense long before it reaches inf, and the "
        f"planner would act on every step of the way. " + self._DIVERGENCE_HINT
      )
    return out[0] if single else out

  # -- fitting ---------------------------------------------------------------

  @staticmethod
  def _transitions(source: Union[Dataset, Sequence[Episode]]) -> Tuple[np.ndarray, ...]:
    """Every Cartesian transition, plus which episode each one came from."""
    poses: List[np.ndarray] = []
    deltas: List[np.ndarray] = []
    nexts: List[np.ndarray] = []
    groups: List[int] = []
    for group, ep in enumerate(episodes_from(source)):
      for obs, action, nxt in ep.transitions():
        pose = obs.pose_vector()
        try:
          delta = action.as_delta(pose)
        except ValueError:
          continue  # joint-space steps carry no Cartesian delta to learn from
        poses.append(pose)
        deltas.append(delta)
        nexts.append(nxt.pose_vector())
        groups.append(group)
    if not poses:
      raise InsufficientData(
        "no Cartesian transitions found. Episodes need at least two observations "
        "and Cartesian actions; a scripted recording of only endpoints will not do."
      )
    return np.array(poses), np.array(deltas), np.array(nexts), np.array(groups)

  @staticmethod
  def _split(groups: np.ndarray, holdout_fraction: float, seed: int) -> Tuple[np.ndarray, ...]:
    """Choose holdout rows, by whole episode when there is more than one.

    Splitting at random over transitions puts frame 41 in the training set and
    frame 42 in the holdout. Those two are the same motion a fortieth of a second
    apart, so a model that only works along the path the demonstrations happened
    to take still scores well on that holdout, which is precisely the failure the
    holdout exists to catch. Whole episodes go to one side or the other.

    One episode leaves no whole episode to hold out, so the fallback below splits
    over transitions and returns the label that makes
    :attr:`DynamicsFit.improved` refuse to answer. The label is the mechanism:
    the fallback still produces usable numbers, and it is the verdict built on
    them, not the numbers, that is withheld.
    """
    rng = np.random.default_rng(seed)
    unique = np.unique(groups)
    if unique.size > 1:
      shuffled = rng.permutation(unique)
      n_holdout_groups = max(1, int(round(unique.size * holdout_fraction)))
      # Never hold out everything: the fit needs something to learn from.
      n_holdout_groups = min(n_holdout_groups, unique.size - 1)
      holdout_groups = set(shuffled[:n_holdout_groups].tolist())
      mask = np.array([g in holdout_groups for g in groups])
      return np.where(~mask)[0], np.where(mask)[0], SOUND_SPLIT

    n = groups.size
    order = rng.permutation(n)
    n_holdout = max(1, int(round(n * holdout_fraction)))
    return order[n_holdout:], order[:n_holdout], "transition"

  @classmethod
  def fit(
    cls,
    source: Union[Dataset, Sequence[Episode]],
    base: Optional[DynamicsModel] = None,
    holdout_fraction: float = 0.25,
    seed: int = 0,
    min_transitions_per_feature: int = 2,
  ) -> Tuple["ResidualDynamics", DynamicsFit]:
    """Fit the residual and report honestly whether it helped.

    Returns the model and its fit report. The model is returned even when the
    fit did not improve on the baseline, so the caller can decide; check that
    ``fit.improved is True`` before planning against it. A single-episode source
    makes that ``None`` rather than ``True`` or ``False``, because one episode
    cannot be split by episode and the resulting holdout is not independent
    evidence.
    """
    if not 0.0 < holdout_fraction < 1.0:
      raise ValueError("holdout_fraction must be between 0 and 1")
    if min_transitions_per_feature < 1:
      raise ValueError(
        "min_transitions_per_feature must be at least 1. Setting it to 0 turns off "
        "the guard against an underdetermined fit, which is the one guard that "
        "stops this from returning a model that memorised its training rows."
      )

    base = base or KinematicDynamics()
    poses, deltas, nexts, groups = cls._transitions(source)

    tr, ho, split_by = cls._split(groups, holdout_fraction, seed)
    n_train, n_holdout = tr.size, ho.size
    needed = max(N_FEATURES * min_transitions_per_feature, N_FEATURES)
    if n_train < needed:
      raise InsufficientData(
        f"{n_train} training transitions for {N_FEATURES} features. A fit this "
        f"underdetermined reproduces its training data and predicts nothing; "
        f"record at least {needed + n_holdout} transitions."
      )

    design = _features(poses[tr], deltas[tr])
    rank = int(np.linalg.matrix_rank(design, tol=RCOND * float(np.linalg.norm(design, 2))))
    if rank == 0:
      raise InsufficientData(
        f"the {n_train} training transitions carry no usable variation at all. "
        "There is nothing here to fit."
      )

    base_train = np.atleast_2d(base.predict(poses[tr], deltas[tr]))
    # An explicit rcond, not the machine-precision default. Directions the
    # demonstrations never excited have tiny singular values, and the default
    # cutoff keeps them: least squares then divides the residual by that tiny
    # number and returns an order-1 coefficient built entirely out of encoder
    # jitter. Truncating instead pins those directions at zero, so an unexcited
    # axis produces a model that ignores it rather than one that invents it.
    weights, *_ = np.linalg.lstsq(design, nexts[tr] - base_train, rcond=RCOND)
    model = cls(base=base, weights=weights)
    model.trained_on = _fingerprint(poses[tr], deltas[tr])

    def score(m: DynamicsModel, rows: np.ndarray) -> Tuple[float, float]:
      return _rmse(np.atleast_2d(m.predict(poses[rows], deltas[rows])), nexts[rows])

    train_t, train_r = score(model, tr)
    hold_t, hold_r = score(model, ho)
    base_t, base_r = score(base, ho)
    spread = design.std(axis=0)
    unexcited = tuple(
      name
      for name, s in zip(FEATURE_NAMES, spread)
      if name != "bias" and s <= RCOND * max(1.0, float(np.abs(design).max()))
    )
    fit = DynamicsFit(
      n_train=n_train,
      n_holdout=n_holdout,
      split_by=split_by,
      rank=rank,
      unexcited=unexcited,
      train_rmse=train_t,
      holdout_rmse=hold_t,
      baseline_holdout_rmse=base_t,
      train_rotation_rmse=train_r,
      holdout_rotation_rmse=hold_r,
      baseline_holdout_rotation_rmse=base_r,
    )
    return model, fit

  def __repr__(self) -> str:
    state = "unfit" if self.weights is None else "fit"
    return f"ResidualDynamics({state}, base={self.base!r})"


class _InjectedDynamics:
  """Dependency-free boundary around a caller-supplied learned predictor.

  A checkpoint is metadata, not executable integration. The caller owns loading
  whatever model it trusts and supplies a synchronous callable that accepts two
  ``(N, 4)`` numpy arrays and returns the predicted next poses as ``(N, 4)``.
  Keeping that boundary here avoids pretending that installing a package or
  naming a checkpoint establishes compatible state, action, or unit semantics.
  """

  model = "world model"

  def __init__(
    self,
    predictor: Optional[Callable[[np.ndarray, np.ndarray], np.ndarray]] = None,
    checkpoint: Optional[Any] = None,
  ) -> None:
    if predictor is not None and not callable(predictor):
      raise TypeError("predictor must be callable or None")
    self.predictor = predictor
    self.checkpoint = checkpoint

  def predict(self, pose: np.ndarray, delta: np.ndarray) -> np.ndarray:
    p, d, single = _pair(pose, delta)
    if self.predictor is None:
      checkpoint_note = " Checkpoint metadata was recorded." if self.checkpoint is not None else ""
      raise NotImplementedError(
        f"the {self.model} adapter has no predictor callable.{checkpoint_note} A checkpoint "
        "or installed package alone is not an adapter: inject a synchronous "
        "predictor(pose_batch, delta_batch) -> next_pose_batch callable whose output "
        "has shape (N, 4), after mapping the checkpoint's state, action, and units."
      )

    raw = self.predictor(p, d)
    if inspect.isawaitable(raw):
      close = getattr(raw, "close", None)
      if callable(close):
        close()
      raise DynamicsAdapterError(
        f"{self.model} predictor returned an awaitable; predictors must be synchronous"
      )
    try:
      out = np.asarray(raw, float)
    except (TypeError, ValueError) as exc:
      raise DynamicsAdapterError(
        f"{self.model} predictor output must be numeric with shape ({p.shape[0]}, 4)"
      ) from exc

    expected = (p.shape[0], 4)
    if out.shape != expected:
      raise DynamicsAdapterError(
        f"{self.model} predictor returned shape {out.shape}; expected exactly {expected}. "
        "The callable contract is batched even when predict() receives one pose."
      )
    if not np.all(np.isfinite(out)):
      raise FloatingPointError(
        f"{self.model} prediction is not finite. " + ResidualDynamics._DIVERGENCE_HINT
      )
    if np.any(np.abs(out[:, :3]) > ResidualDynamics.PLAUSIBLE_MM):
      raise FloatingPointError(
        f"{self.model} prediction left the plausible range for a lab arm "
        f"(more than {ResidualDynamics.PLAUSIBLE_MM:.0f} mm from the origin). "
        + ResidualDynamics._DIVERGENCE_HINT
      )
    return out[0] if single else out


class VideoWorldModel(_InjectedDynamics):
  """Adapter for an injected video-world-model pose predictor.

  A video model usually predicts future frames rather than poses, so the
  injected callable must also perform the readout from predicted pixels back to
  tool pose. This package does not load a checkpoint or provide that mapping.
  Score the complete callable with
  :func:`plr_lr.learn.evaluate.evaluate_dynamics` against a held-out set before
  letting it plan.
  """

  model = "video world model"


class LearnedDynamics(_InjectedDynamics):
  """Adapter for an injected state-space dynamics predictor."""

  model = "learned state dynamics"
