"""Tests for plr_lr.learn.predict.

This module is the only thing standing between an outside model and a published
number about it, so the tests are organised around the claims it makes rather than
around its functions. Every claim below is one a later refactor could quietly
weaken while leaving the module importable and the happy path green.

The claims are: that a request which cannot be scored is refused at the point it is
built; that an adapter with no callable behind it raises rather than returning
zeros; that an async predictor is closed before being refused, because a dropped
coroutine surfaces as an unrelated warning much later; that the abstain flag and
the all-NaN row agree in both directions, so abstaining while banking a usable
prediction is impossible; that flagging with a finite prediction stays legal,
because refusing and warning are different acts; that an undeclared training set or
an undeclared fitted probe withholds the row; that a predictor which ignores its
action argument is refused rather than scored, demonstrated on two nulls whose
divergence is exactly zero and not merely small; that a policy raises on
construction instead of being ranked as a world model; and that satisfying the
Protocol is not evidence of being able to predict.

The two nulls are checked for exact zero divergence on purpose. A null that fails
the gate because its noise happened to land below the bar would pass on some other
seed, and a control that fails only most of the time is not a control.
"""

from __future__ import annotations

import unittest
import warnings

import numpy as np

from plr_lr.learn.predict import (
  ACTION_SPACES,
  ENVELOPE_STEP_MM,
  INTERVENTION_ALPHA,
  INTERVENTION_AXES,
  INTERVENTION_K,
  INTERVENTION_QUORUM,
  NO_TRAINING_DATA,
  PLAUSIBLE_MM,
  REFUSALS,
  ActionEmittingAdapter,
  AdapterOutputError,
  BlindPredictor,
  CosmosPolicyAdapter,
  FittedComponent,
  FrameRef,
  GR00TAdapter,
  InjectedPredictor,
  InterventionFailed,
  InterventionResult,
  NoisePredictor,
  NotAWorldModel,
  Pi0Adapter,
  ProvenanceUndeclared,
  RANDOM_PROBE,
  RolloutPrediction,
  RolloutPredictor,
  RolloutRequest,
  UndeclaredComponent,
  interventional_check,
  require_action_conditioned,
  validate,
)

STATE_FIELDS = ("x", "y", "z", "rz")
STATE_UNITS = ("mm", "mm", "mm", "rad")
TOLERANCE = (2.0, 2.0, 2.0, 0.05)
DECLARED = frozenset({"synthetic:S1:seeds-0-399"})


def a_frame(index: int = 0) -> FrameRef:
  return FrameRef(
    path=f"frames/{index:04d}.png",
    sha256=f"{index:064x}",
    t_capture_s=float(index) * 0.1,
    clock_source="rendered",
  )


def a_request(
  episode_id: str = "ep-0000",
  horizon: int = 2,
  step_mm: float = 5.0,
  state=(100.0, 0.0, 50.0, 0.0),
) -> RolloutRequest:
  """One scoreable request. Actions move along +x by ``step_mm`` on every step."""
  actions = np.zeros((horizon, 5), dtype=float)
  actions[:, 0] = step_mm
  return RolloutRequest(
    task_id="S1",
    episode_id=episode_id,
    state=np.array(state, dtype=float),
    state_fields=STATE_FIELDS,
    state_units=STATE_UNITS,
    frames=(a_frame(0), a_frame(1)),
    actions=actions,
    horizon=horizon,
    tolerance=np.array(TOLERANCE, dtype=float),
  )


def gate_requests(n: int = INTERVENTION_K) -> tuple:
  return tuple(a_request(episode_id=f"ep-{i:04d}") for i in range(n))


def a_prediction(request: RolloutRequest, **overrides) -> RolloutPrediction:
  """A well-formed prediction for ``request``, with fields overridable per test."""
  dim = int(request.state.size)
  state = np.tile(np.asarray(request.state, dtype=float), (int(request.horizon), 1))
  fields = dict(
    state=state,
    confidence=0.75,
    abstain=False,
    flag=False,
    readout="analytic",
    n_forward_passes=1,
    wall_clock_s=0.01,
  )
  fields.update(overrides)
  assert dim >= 1
  return RolloutPrediction(**fields)


class DeclaredPredictor:
  """A minimally honest adapter: declares provenance, tracks the commanded action.

  ``gain`` is how much of the commanded translation actually reaches the predicted
  terminal state. A gain of 1.0 is a perfect tracker; the small gains are used to
  show that the interventional bar is a real bar and not a formality.
  """

  action_space = "ee_delta_mm"
  license_text = "MIT, test fixture."

  def __init__(self, gain: float = 1.0, state_dim: int = 4, horizon: int = 2,
               confidence: float = 0.8, abstain: bool = False):
    self.name = "DeclaredPredictor"
    self.gain = float(gain)
    self.state_dim = int(state_dim)
    self.horizon = int(horizon)
    self.confidence = float(confidence)
    self.abstain = bool(abstain)
    self.training_fingerprints = DECLARED
    self.fitted_components = ()

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    if self.abstain:
      return RolloutPrediction.abstained(int(request.horizon), int(request.state.size))
    cumulative = np.cumsum(request.actions[:, :4], axis=0) * self.gain
    state = np.asarray(request.state, dtype=float)[None, :] + cumulative
    return RolloutPrediction(
      state=state,
      confidence=self.confidence,
      abstain=False,
      flag=False,
      readout="analytic",
      n_forward_passes=int(request.horizon),
      wall_clock_s=0.001,
    )


class AnisotropicPredictor(DeclaredPredictor):
  """Reads its action argument on one column and barely on the others.

  This is not a contrived shape. It is what every bench task looks like: S5's
  terminal engagement depth answers on commanded delta z and only marginally on the
  two lateral columns, and S4's answers on the two lateral columns and only
  marginally on z. Measured on the built-in 200-episode pack with each task's own
  declared floor predictor, at a bar of 0.2500, the worst episode of sixty responded
  0.1536 to a lateral push on S5 and 0.3500 to a vertical push on S4.

  A gate that draws one isotropic direction is therefore reading task anisotropy and
  reporting it as a property of the model. ``lateral`` is the leak rate onto the
  other two columns and defaults to zero so the artifact is exact rather than
  approximate.
  """

  def __init__(self, column: int = 2, gain: float = 1.0, lateral: float = 0.0, **kwargs):
    super().__init__(gain=gain, **kwargs)
    self.name = "AnisotropicPredictor"
    self.column = int(column)
    self.lateral = float(lateral)

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    weights = np.full(3, self.lateral, dtype=float)
    weights[self.column] = 1.0
    driven = np.cumsum(np.asarray(request.actions, dtype=float)[:, :3] @ weights)
    state = np.tile(np.asarray(request.state, dtype=float), (int(request.horizon), 1))
    state[:, 0] = state[:, 0] + driven * self.gain
    return RolloutPrediction(
      state=state,
      confidence=self.confidence,
      abstain=False,
      flag=False,
      readout="one-axis",
      n_forward_passes=int(request.horizon),
      wall_clock_s=0.001,
    )


class MinorityPredictor(DeclaredPredictor):
  """Reads its actions on the first ``n_live`` episodes of the pool and ignores them after.

  Exists to put a number on how much of the gate the quorum rule actually gives away.
  A model that responds on a minority of the requests it was checked on has not shown
  that it reads its action argument, and this is the double that proves the majority
  bar is a bar rather than a formality.
  """

  def __init__(self, n_live: int = 9, **kwargs):
    super().__init__(**kwargs)
    self.name = "MinorityPredictor"
    self.n_live = int(n_live)

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    if int(request.episode_id.rsplit("-", 1)[-1]) < self.n_live:
      return super().predict(request)
    state = np.tile(np.asarray(request.state, dtype=float), (int(request.horizon), 1))
    return RolloutPrediction(
      state=state,
      confidence=self.confidence,
      abstain=False,
      flag=False,
      readout="identity",
      n_forward_passes=1,
      wall_clock_s=0.0,
    )


class OutOfRangePredictor(DeclaredPredictor):
  """Answers most probes and refuses the one that pushes it outside its declared range.

  A real adapter does this: an intervention that walks the commanded action off the
  distribution the checkpoint was trained on is exactly the case a well-built model
  abstains on. An abstention on one probe is an absence of evidence about action
  conditioning, not evidence against it, and a gate that scored it as a failure would
  punish the adapters that refuse honestly.
  """

  def __init__(self, refuse_column: int = 0, limit_mm: float = 20.0, **kwargs):
    super().__init__(**kwargs)
    self.name = "OutOfRangePredictor"
    self.refuse_column = int(refuse_column)
    self.limit_mm = float(limit_mm)

  def predict(self, request: RolloutRequest) -> RolloutPrediction:
    commanded = float(np.abs(np.asarray(request.actions, dtype=float)[:, self.refuse_column]).max())
    if commanded > self.limit_mm:
      return RolloutPrediction.abstained(int(request.horizon), int(request.state.size))
    return super().predict(request)


class FakeAwaitable:
  """An awaitable whose close() is observable, so the test can prove it was called."""

  def __init__(self) -> None:
    self.closed = False

  def __await__(self):  # pragma: no cover - never awaited, by design
    yield

  def close(self) -> None:
    self.closed = True


class TestRequestRefusesTheUnscoreable(unittest.TestCase):
  """A request that cannot produce a meaningful score is refused where it is built."""

  def test_a_good_request_builds(self):
    request = a_request()
    self.assertEqual(request.horizon, 2)
    self.assertEqual(request.actions.shape, (2, 5))
    self.assertIsInstance(request.state, np.ndarray)

  def test_zero_tolerance_is_refused(self):
    with self.assertRaises(ValueError) as caught:
      RolloutRequest(
        task_id="S1",
        episode_id="ep-0000",
        state=np.zeros(4),
        state_fields=STATE_FIELDS,
        state_units=STATE_UNITS,
        frames=(),
        actions=np.zeros((1, 5)),
        horizon=1,
        tolerance=np.array([2.0, 2.0, 2.0, 0.0]),
      )
    self.assertIn("strictly positive", str(caught.exception))

  def test_horizon_must_equal_the_number_of_actions(self):
    with self.assertRaises(ValueError):
      RolloutRequest(
        task_id="S1",
        episode_id="ep-0000",
        state=np.zeros(4),
        state_fields=STATE_FIELDS,
        state_units=STATE_UNITS,
        frames=(),
        actions=np.zeros((2, 5)),
        horizon=4,
        tolerance=np.array(TOLERANCE),
      )

  def test_field_names_must_cover_every_state_column(self):
    with self.assertRaises(ValueError) as caught:
      RolloutRequest(
        task_id="S1",
        episode_id="ep-0000",
        state=np.zeros(4),
        state_fields=("x", "y"),
        state_units=STATE_UNITS,
        frames=(),
        actions=np.zeros((1, 5)),
        horizon=1,
        tolerance=np.array(TOLERANCE),
      )
    self.assertIn("implicit", str(caught.exception))

  def test_actions_must_use_the_five_column_layout(self):
    with self.assertRaises(ValueError):
      RolloutRequest(
        task_id="S1",
        episode_id="ep-0000",
        state=np.zeros(4),
        state_fields=STATE_FIELDS,
        state_units=STATE_UNITS,
        frames=(),
        actions=np.zeros((2, 4)),
        horizon=2,
        tolerance=np.array(TOLERANCE),
      )

  def test_an_unnamed_clock_source_is_refused(self):
    with self.assertRaises(ValueError) as caught:
      FrameRef(path="a.png", sha256="ab", t_capture_s=0.0, clock_source="camera")
    self.assertIn("clock_source", str(caught.exception))

  def test_a_frame_without_a_hash_is_refused(self):
    with self.assertRaises(ValueError) as caught:
      FrameRef(path="a.png", sha256="", t_capture_s=0.0, clock_source="rendered")
    self.assertIn("manifest", str(caught.exception))

  def test_newest_frame_is_the_latest_capture_not_the_last_element(self):
    request = a_request()
    self.assertEqual(request.newest_frame().t_capture_s, 0.1)
    self.assertIsNone(
      RolloutRequest(
        task_id="S1",
        episode_id="ep-0000",
        state=np.zeros(4),
        state_fields=STATE_FIELDS,
        state_units=STATE_UNITS,
        frames=(),
        actions=np.zeros((1, 5)),
        horizon=1,
        tolerance=np.array(TOLERANCE),
      ).newest_frame()
    )


class TestOutputContract(unittest.TestCase):
  """Shape, type and finiteness, each refused with the model and the signature named."""

  def setUp(self):
    self.request = a_request()
    self.predictor = DeclaredPredictor()

  def check(self, pred):
    return validate(pred, self.request, self.predictor)

  def test_a_good_prediction_validates_and_is_canonical(self):
    out = self.check(a_prediction(self.request, state=[[1.0, 2.0, 3.0, 0.1]] * 2))
    self.assertIsInstance(out.state, np.ndarray)
    self.assertEqual(out.state.dtype, np.dtype(float))
    self.assertIsInstance(out.confidence, float)

  def test_the_refusal_names_the_model_and_the_signature(self):
    with self.assertRaises(AdapterOutputError) as caught:
      self.check(a_prediction(self.request, state=np.zeros((5, 4))))
    message = str(caught.exception)
    self.assertIn("DeclaredPredictor", message)
    self.assertIn("predict(request: RolloutRequest) -> RolloutPrediction", message)

  def test_a_non_prediction_return_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check({"state": np.zeros((2, 4)), "confidence": 0.5})

  def test_wrong_horizon_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, state=np.zeros((3, 4))))

  def test_wrong_state_dim_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, state=np.zeros((2, 3))))

  def test_a_declared_state_dim_that_disagrees_with_the_task_is_refused(self):
    wrong = DeclaredPredictor(state_dim=7)
    with self.assertRaises(AdapterOutputError) as caught:
      validate(a_prediction(self.request), self.request, wrong)
    self.assertIn("state_dim", str(caught.exception))

  def test_non_numeric_state_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, state=[["a", "b", "c", "d"]] * 2))

  def test_infinite_state_outside_an_abstention_is_refused(self):
    state = np.zeros((2, 4))
    state[1, 2] = np.inf
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, state=state))

  def test_a_partly_nan_prediction_is_neither_a_refusal_nor_an_answer(self):
    state = np.zeros((2, 4))
    state[0, :] = np.nan
    with self.assertRaises(AdapterOutputError) as caught:
      self.check(a_prediction(self.request, state=state))
    self.assertIn("[0]", str(caught.exception))

  def test_confidence_above_one_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, confidence=1.5))

  def test_confidence_below_zero_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, confidence=-0.01))

  def test_a_boolean_confidence_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, confidence=True))

  def test_a_non_boolean_abstain_is_refused(self):
    with self.assertRaises(AdapterOutputError) as caught:
      self.check(a_prediction(self.request, abstain=1))
    self.assertIn("not a bool", str(caught.exception))

  def test_an_empty_readout_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, readout=""))

  def test_negative_declared_compute_is_refused(self):
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, n_forward_passes=-1))
    with self.assertRaises(AdapterOutputError):
      self.check(a_prediction(self.request, wall_clock_s=-0.5))

  def test_a_nonsense_position_raises_floating_point_error_not_an_output_error(self):
    state = np.zeros((2, 4))
    state[1, 0] = PLAUSIBLE_MM * 2.0
    with self.assertRaises(FloatingPointError) as caught:
      self.check(a_prediction(self.request, state=state))
    self.assertIn("finite nonsense", str(caught.exception))

  def test_a_large_value_in_a_non_millimetre_column_is_not_a_position(self):
    state = np.zeros((2, 4))
    state[1, 3] = PLAUSIBLE_MM * 2.0
    self.check(a_prediction(self.request, state=state))


class TestAwaitableIsClosedBeforeRefusal(unittest.TestCase):
  """An async predictor is refused, and the coroutine is closed on the way out."""

  def test_the_awaitable_is_closed(self):
    awaitable = FakeAwaitable()
    with self.assertRaises(AdapterOutputError) as caught:
      validate(awaitable, a_request(), DeclaredPredictor())
    self.assertTrue(awaitable.closed)
    self.assertIn("awaitable", str(caught.exception))

  def test_a_real_coroutine_leaves_no_never_awaited_warning(self):
    async def predict_async():  # pragma: no cover - never awaited, by design
      return None

    coroutine = predict_async()
    with warnings.catch_warnings(record=True) as caught:
      warnings.simplefilter("always")
      with self.assertRaises(AdapterOutputError):
        validate(coroutine, a_request(), DeclaredPredictor())
      del coroutine
    never_awaited = [w for w in caught if "never awaited" in str(w.message)]
    self.assertEqual(never_awaited, [])


class TestRowInvariant(unittest.TestCase):
  """abstain is True exactly when the state is all-NaN and the confidence is NaN.

  Checked in both directions. The gaming channel this closes is a row that abstains
  while still carrying a usable prediction, which can be read as a refusal when the
  refusal was right and as a prediction when the prediction was right.
  """

  def setUp(self):
    self.request = a_request()
    self.predictor = DeclaredPredictor()

  def test_a_correct_abstention_validates(self):
    out = validate(
      RolloutPrediction.abstained(2, 4), self.request, self.predictor
    )
    self.assertTrue(out.abstain)
    self.assertTrue(np.isnan(out.state).all())
    self.assertTrue(np.isnan(out.confidence))

  def test_abstain_with_a_finite_prediction_is_refused(self):
    with self.assertRaises(AdapterOutputError) as caught:
      validate(
        a_prediction(self.request, abstain=True, confidence=float("nan")),
        self.request,
        self.predictor,
      )
    self.assertIn("free lunch", str(caught.exception))

  def test_abstain_with_a_numeric_confidence_is_refused(self):
    pred = RolloutPrediction(
      state=np.full((2, 4), np.nan),
      confidence=0.5,
      abstain=True,
      flag=False,
      readout="abstained",
      n_forward_passes=0,
      wall_clock_s=0.0,
    )
    with self.assertRaises(AdapterOutputError) as caught:
      validate(pred, self.request, self.predictor)
    self.assertIn("carries no", str(caught.exception))

  def test_an_all_nan_state_without_the_abstain_flag_is_refused(self):
    pred = RolloutPrediction(
      state=np.full((2, 4), np.nan),
      confidence=0.5,
      abstain=False,
      flag=False,
      readout="analytic",
      n_forward_passes=1,
      wall_clock_s=0.0,
    )
    with self.assertRaises(AdapterOutputError):
      validate(pred, self.request, self.predictor)

  def test_a_nan_confidence_without_the_abstain_flag_is_refused(self):
    with self.assertRaises(AdapterOutputError) as caught:
      validate(
        a_prediction(self.request, confidence=float("nan")),
        self.request,
        self.predictor,
      )
    self.assertIn("row invariant", str(caught.exception))

  def test_abstaining_and_flagging_at_once_is_illegal(self):
    pred = RolloutPrediction(
      state=np.full((2, 4), np.nan),
      confidence=float("nan"),
      abstain=True,
      flag=True,
      readout="abstained",
      n_forward_passes=0,
      wall_clock_s=0.0,
    )
    with self.assertRaises(AdapterOutputError) as caught:
      validate(pred, self.request, self.predictor)
    self.assertIn("different acts", str(caught.exception))

  def test_flagging_a_finite_prediction_is_legal_and_is_an_act(self):
    out = validate(
      a_prediction(self.request, flag=True), self.request, self.predictor
    )
    self.assertTrue(out.flag)
    self.assertFalse(out.abstain)
    self.assertTrue(np.isfinite(out.state).all())

  def test_the_abstained_helper_refuses_a_degenerate_shape(self):
    with self.assertRaises(ValueError):
      RolloutPrediction.abstained(0, 4)


class TestProvenanceWithholdsTheRow(unittest.TestCase):
  """R7 and R8: nothing is scored for an adapter that will not say what it was fit on."""

  def setUp(self):
    self.request = a_request()

  def test_no_predictor_at_all_is_an_undeclared_provenance(self):
    with self.assertRaises(ProvenanceUndeclared) as caught:
      validate(a_prediction(self.request), self.request)
    self.assertIn("PROVENANCE: UNDECLARED", str(caught.exception))

  def test_missing_fingerprints_withhold_the_row(self):
    predictor = DeclaredPredictor()
    del predictor.training_fingerprints
    with self.assertRaises(ProvenanceUndeclared) as caught:
      validate(a_prediction(self.request), self.request, predictor)
    self.assertIn("PROVENANCE: UNDECLARED", str(caught.exception))

  def test_empty_fingerprints_withhold_the_row(self):
    predictor = DeclaredPredictor()
    predictor.training_fingerprints = frozenset()
    with self.assertRaises(ProvenanceUndeclared):
      validate(a_prediction(self.request), self.request, predictor)

  def test_a_literal_undeclared_fingerprint_withholds_the_row(self):
    predictor = DeclaredPredictor()
    predictor.training_fingerprints = frozenset({"UNDECLARED"})
    with self.assertRaises(ProvenanceUndeclared):
      validate(a_prediction(self.request), self.request, predictor)

  def test_declaring_that_nothing_was_fit_is_a_declaration(self):
    predictor = DeclaredPredictor()
    predictor.training_fingerprints = frozenset({NO_TRAINING_DATA})
    validate(a_prediction(self.request), self.request, predictor)

  def test_an_undeclared_fitted_component_withholds_the_row(self):
    predictor = DeclaredPredictor()
    predictor.fitted_components = (
      FittedComponent(
        name="latent_to_uL_linear_probe",
        fit_on_split="UNDECLARED",
        n_examples=200,
        description="latent to microlitres",
      ),
    )
    with self.assertRaises(UndeclaredComponent) as caught:
      validate(a_prediction(self.request), self.request, predictor)
    self.assertIn("R8", str(caught.exception))

  def test_a_component_fit_on_an_unrecognised_split_is_treated_as_undeclared(self):
    predictor = DeclaredPredictor()
    predictor.fitted_components = (
      FittedComponent("probe", "scoring", 200, "fit on the scored episodes"),
    )
    with self.assertRaises(UndeclaredComponent):
      validate(a_prediction(self.request), self.request, predictor)

  def test_a_component_fit_on_no_examples_was_not_fit(self):
    predictor = DeclaredPredictor()
    predictor.fitted_components = (FittedComponent("probe", "dev", 0, "empty"),)
    with self.assertRaises(UndeclaredComponent):
      validate(a_prediction(self.request), self.request, predictor)

  def test_a_properly_declared_component_passes(self):
    predictor = DeclaredPredictor()
    predictor.fitted_components = (
      FittedComponent("latent_to_uL_linear_probe", "dev", 240, "latent to microlitres"),
    )
    out = validate(a_prediction(self.request), self.request, predictor)
    self.assertFalse(out.abstain)

  def test_a_non_component_in_the_tuple_cannot_name_a_split(self):
    predictor = DeclaredPredictor()
    predictor.fitted_components = ("a probe, honest",)
    with self.assertRaises(UndeclaredComponent):
      validate(a_prediction(self.request), self.request, predictor)

  def test_the_output_contract_is_checked_before_the_paperwork(self):
    """A broken shape hears about the shape, not about its fingerprints."""
    with self.assertRaises(AdapterOutputError):
      validate(a_prediction(self.request, state=np.zeros((9, 4))), self.request)


class TestInjectedPredictor(unittest.TestCase):
  """A checkpoint is not an adapter, and satisfying the Protocol is not predicting."""

  def test_a_bare_adapter_satisfies_the_protocol_and_still_cannot_predict(self):
    adapter = InjectedPredictor()
    self.assertIsInstance(adapter, RolloutPredictor)
    with self.assertRaises(NotImplementedError):
      adapter.predict(a_request())

  def test_the_docstring_says_the_isinstance_check_is_not_evidence(self):
    doc = InjectedPredictor.__doc__ or ""
    self.assertIn("isinstance", doc)
    self.assertIn("cannot predict", doc)

  def test_the_refusal_names_the_model_and_the_expected_signature(self):
    adapter = InjectedPredictor(name="iVideoGPT", checkpoint="thuml/ivideogpt")
    with self.assertRaises(NotImplementedError) as caught:
      adapter.predict(a_request())
    message = str(caught.exception)
    self.assertIn("iVideoGPT", message)
    self.assertIn("Checkpoint metadata was recorded", message)
    self.assertIn("predict(request: RolloutRequest) -> RolloutPrediction", message)

  def test_a_non_callable_predictor_is_refused_at_construction(self):
    with self.assertRaises(TypeError):
      InjectedPredictor(predictor="a checkpoint path")

  def test_an_unnamed_action_space_is_refused_at_construction(self):
    with self.assertRaises(ValueError):
      InjectedPredictor(action_space="whatever the model uses")
    for space in ACTION_SPACES:
      InjectedPredictor(action_space=space)

  def test_an_injected_callable_is_held_to_the_output_contract(self):
    adapter = InjectedPredictor(
      predictor=lambda request: a_prediction(request, state=np.zeros((7, 4)))
    )
    with self.assertRaises(AdapterOutputError):
      adapter.predict(a_request())

  def test_a_working_injected_callable_returns_a_validated_prediction(self):
    adapter = InjectedPredictor(predictor=lambda request: a_prediction(request))
    out = adapter.predict(a_request())
    self.assertIsInstance(out, RolloutPrediction)
    self.assertEqual(out.state.shape, (2, 4))

  def test_an_async_injected_callable_is_refused(self):
    adapter = InjectedPredictor(predictor=lambda request: FakeAwaitable())
    with self.assertRaises(AdapterOutputError):
      adapter.predict(a_request())

  def test_predict_does_not_check_provenance_because_that_withholds_a_score(self):
    """The seam enforces the output contract; validate() withholds the row."""
    adapter = InjectedPredictor(predictor=lambda request: a_prediction(request))
    out = adapter.predict(a_request())
    self.assertFalse(adapter.training_fingerprints)
    with self.assertRaises(ProvenanceUndeclared):
      validate(out, a_request(), adapter)

  def test_repr_says_whether_anything_is_behind_it(self):
    self.assertIn("empty", repr(InjectedPredictor()))
    self.assertIn("injected", repr(InjectedPredictor(predictor=lambda r: None)))


class TestInterventionalGate(unittest.TestCase):
  """A predictor that ignores its action argument is refused, not ranked."""

  def test_the_constants_are_absolute(self):
    self.assertEqual(INTERVENTION_ALPHA, 0.25)
    self.assertEqual(INTERVENTION_K, 16)

  def test_a_tracking_predictor_passes(self):
    result = interventional_check(DeclaredPredictor(gain=1.0), gate_requests())
    self.assertTrue(result.passed, result.summary())
    self.assertEqual(result.n_checked, INTERVENTION_K)
    self.assertEqual(result.failed, ())
    self.assertIn("PASSED", result.summary())

  def test_blind_predictor_fails_the_gate(self):
    result = interventional_check(BlindPredictor(), gate_requests())
    self.assertFalse(result.passed)
    self.assertEqual(len(result.failed), INTERVENTION_K)

  def test_blind_divergence_is_exactly_zero_not_merely_small(self):
    result = interventional_check(BlindPredictor(), gate_requests())
    self.assertEqual(set(result.divergence), {0.0})

  def test_noise_predictor_fails_the_gate(self):
    result = interventional_check(NoisePredictor(sigma=5.0), gate_requests())
    self.assertFalse(result.passed)
    self.assertEqual(len(result.failed), INTERVENTION_K)

  def test_noise_divergence_is_exactly_zero_because_the_noise_ignores_actions(self):
    result = interventional_check(NoisePredictor(sigma=50.0), gate_requests())
    self.assertEqual(set(result.divergence), {0.0})

  def test_the_nulls_fail_on_every_seed_tried(self):
    for seed in range(5):
      for null in (BlindPredictor(), NoisePredictor(sigma=3.0)):
        result = interventional_check(null, gate_requests(), seed=seed)
        self.assertFalse(result.passed, f"{null.name} passed at seed {seed}")

  def test_a_failed_gate_is_a_refusal_and_not_a_low_score(self):
    result = interventional_check(BlindPredictor(), gate_requests())
    with self.assertRaises(InterventionFailed) as caught:
      result.raise_for_refusal()
    self.assertIn("no score is emitted", str(caught.exception))
    self.assertIn("REFUSED", str(caught.exception))

  def test_the_result_carries_no_rankable_scalar(self):
    """A figure of merit here would get sorted, and sorting is the thing to prevent."""
    result = interventional_check(BlindPredictor(), gate_requests())
    numeric = {"alpha", "step_bound_mm"}
    for name, value in vars(result).items():
      if name in numeric or isinstance(value, (str, bool, tuple)):
        continue
      self.assertIsInstance(value, int, f"{name} is a rankable float")

  def test_require_action_conditioned_raises_for_a_blind_model(self):
    with self.assertRaises(InterventionFailed):
      require_action_conditioned(BlindPredictor(), gate_requests())

  def test_require_action_conditioned_returns_the_result_for_a_real_one(self):
    result = require_action_conditioned(DeclaredPredictor(gain=1.0), gate_requests())
    self.assertIsInstance(result, InterventionResult)
    self.assertTrue(result.passed)

  def test_alpha_is_a_real_bar(self):
    """A model that moves one percent of the commanded distance does not clear it."""
    self.assertFalse(interventional_check(DeclaredPredictor(gain=0.01), gate_requests()).passed)
    self.assertTrue(interventional_check(DeclaredPredictor(gain=0.5), gate_requests()).passed)

  def test_an_abstaining_predictor_produces_no_evidence_and_does_not_pass(self):
    result = interventional_check(DeclaredPredictor(abstain=True), gate_requests())
    self.assertFalse(result.passed)
    self.assertEqual(result.n_abstained, INTERVENTION_K)
    self.assertIn("no evidence", result.summary())

  def test_too_few_requests_is_refused_rather_than_reported_as_a_gate_outcome(self):
    with self.assertRaises(ValueError) as caught:
      interventional_check(BlindPredictor(), gate_requests(3))
    self.assertIn("nothing was checked", str(caught.exception))

  def test_an_empty_request_pool_never_reads_as_a_pass(self):
    with self.assertRaises(ValueError):
      interventional_check(DeclaredPredictor(), ())

  def test_alpha_outside_the_unit_interval_is_refused(self):
    for bad in (0.0, 1.0, -0.5, 2.0):
      with self.assertRaises(ValueError):
        interventional_check(DeclaredPredictor(), gate_requests(), alpha=bad)

  def test_k_must_be_a_positive_integer(self):
    for bad in (0, -1, True):
      with self.assertRaises(ValueError):
        interventional_check(DeclaredPredictor(), gate_requests(), k=bad)

  def test_the_gate_uses_the_package_envelope_step_and_not_a_caller_supplied_one(self):
    result = interventional_check(DeclaredPredictor(), gate_requests())
    self.assertEqual(result.step_bound_mm, ENVELOPE_STEP_MM)
    self.assertEqual(ENVELOPE_STEP_MM, 25.0)

  def test_the_gate_is_reproducible_at_a_fixed_seed(self):
    first = interventional_check(DeclaredPredictor(gain=0.3), gate_requests(), seed=7)
    second = interventional_check(DeclaredPredictor(gain=0.3), gate_requests(), seed=7)
    self.assertEqual(first.divergence, second.divergence)

  def test_the_gate_leaves_the_requests_it_was_given_untouched(self):
    requests = gate_requests()
    before = [r.actions.copy() for r in requests]
    interventional_check(DeclaredPredictor(), requests)
    for original, request in zip(before, requests):
      np.testing.assert_array_equal(original, request.actions)


class TestTheGateIsNotDecidedByTheSeed(unittest.TestCase):
  """The verdict is a property of the model, not of a CLI default.

  The measured defect: the gate drew one isotropic 3-D direction per request and
  demanded that every request clear the bar. Run on the built-in 200-episode pack
  against each task's own declared floor predictor, that refused a genuine predictor
  in 8 of 100 task-seed cells, S5 on 5 of 20 gate seeds and S3 on 3 of 20. Because a
  gate refusal withholds the whole scorecard under rule R4, ``--gate-seed 2`` scored
  no rows at all and ``--gate-seed 3`` scored every row, for the same model on the
  same pack.

  Both halves of the repair are asserted here, because either one alone leaves the
  verdict decided by a draw.
  """

  def test_an_off_axis_predictor_passes_at_every_seed(self):
    """The exact artifact: a model that reads delta z and is pushed sideways."""
    predictor = AnisotropicPredictor(column=2)
    for seed in range(20):
      result = interventional_check(predictor, gate_requests(), seed=seed)
      self.assertTrue(result.passed, f"seed {seed}: {result.summary()}")

  def test_it_passes_while_two_of_its_three_probes_sit_below_the_bar(self):
    """Passing is not the whole claim. Passing *despite* a sub-bar probe is."""
    result = interventional_check(AnisotropicPredictor(column=2), gate_requests())
    self.assertTrue(result.passed, result.summary())
    weakest = result.weakest_probe()
    self.assertIsNotNone(weakest)
    self.assertLess(weakest[1], max(result.required))
    for row in result.probe_divergence:
      lateral = [row[i] for i in range(3) if i != 2]
      self.assertTrue(all(value <= max(result.required) for value in lateral), row)
      self.assertGreater(row[2], max(result.required), row)

  def test_a_genuine_predictor_passes_at_every_seed_on_every_axis_it_could_read(self):
    for column in range(3):
      predictor = AnisotropicPredictor(column=column)
      for seed in range(20):
        result = interventional_check(predictor, gate_requests(), seed=seed)
        self.assertTrue(result.passed, f"column {column} seed {seed}")

  def test_the_recorded_response_is_the_largest_probe_and_not_a_drawn_one(self):
    result = interventional_check(AnisotropicPredictor(column=1), gate_requests())
    self.assertEqual(len(result.probe_divergence), INTERVENTION_K)
    for observed, row in zip(result.divergence, result.probe_divergence):
      self.assertEqual(observed, max(row))

  def test_the_probes_are_a_complete_basis_plus_one_drawn_direction(self):
    result = interventional_check(DeclaredPredictor(), gate_requests())
    self.assertEqual(result.probes, ("p0", "p1", "p2", RANDOM_PROBE))
    self.assertEqual(len(result.probes), len(INTERVENTION_AXES) + 1)
    self.assertEqual(
      INTERVENTION_AXES, ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    )

  def test_the_drawn_probe_can_never_turn_a_pass_into_a_refusal(self):
    """It enters through a max, so it can only raise a response, never lower one."""
    predictor = AnisotropicPredictor(column=2)
    for seed in range(20):
      result = interventional_check(predictor, gate_requests(), seed=seed)
      for row in result.probe_divergence:
        self.assertGreaterEqual(max(row), max(row[: len(INTERVENTION_AXES)]))

  def test_the_verdict_does_not_move_with_the_size_of_the_request_pool(self):
    """probe --episodes 40 refusing what --episodes 60 passed was the second symptom."""
    predictor = AnisotropicPredictor(column=2)
    for pool in (INTERVENTION_K, 40, 60, 100, 200):
      for seed in range(5):
        result = interventional_check(predictor, gate_requests(pool), seed=seed)
        self.assertTrue(result.passed, f"pool {pool} seed {seed}")

  def test_the_nulls_still_fail_at_every_one_of_those_seeds(self):
    """Permissiveness bought nothing if it also let a non-world-model through."""
    for seed in range(20):
      for null in (BlindPredictor(), NoisePredictor(sigma=3.0), NoisePredictor(sigma=50.0)):
        result = interventional_check(null, gate_requests(), seed=seed)
        self.assertFalse(result.passed, f"{null.name} passed at seed {seed}")
        self.assertEqual(result.n_responded, 0)
        self.assertEqual(set(result.divergence), {0.0})


class TestTheQuorumIsABarAndNotAFormality(unittest.TestCase):
  """A strict majority of the checked requests has to respond, and it is measured."""

  def test_the_quorum_is_a_strict_majority(self):
    self.assertEqual(INTERVENTION_QUORUM, 0.5)
    result = interventional_check(DeclaredPredictor(), gate_requests())
    self.assertEqual(result.n_required, 9)
    self.assertEqual(result.n_checked, 16)

  def test_a_model_that_responds_on_a_minority_is_refused(self):
    result = interventional_check(MinorityPredictor(n_live=8), gate_requests())
    self.assertFalse(result.passed, result.summary())
    self.assertEqual(result.n_responded, 8)
    self.assertEqual(result.n_required, 9)

  def test_one_more_responding_request_is_the_whole_difference(self):
    result = interventional_check(MinorityPredictor(n_live=9), gate_requests())
    self.assertTrue(result.passed, result.summary())
    self.assertEqual(result.n_responded, 9)

  def test_the_boundary_does_not_move_with_the_seed(self):
    for seed in range(20):
      self.assertFalse(
        interventional_check(MinorityPredictor(n_live=8), gate_requests(), seed=seed).passed
      )
      self.assertTrue(
        interventional_check(MinorityPredictor(n_live=9), gate_requests(), seed=seed).passed
      )

  def test_a_blind_model_is_nine_requests_short_and_not_one(self):
    """The margin is what makes this a floor rather than a coin flip."""
    result = interventional_check(BlindPredictor(), gate_requests())
    self.assertEqual(result.n_responded, 0)
    self.assertEqual(len(result.failed), INTERVENTION_K)
    self.assertFalse(result.passed)

  def test_a_response_below_alpha_still_fails_however_many_requests_show_it(self):
    """The quorum counts requests that cleared the bar, never requests that moved."""
    result = interventional_check(AnisotropicPredictor(column=2, gain=0.001), gate_requests())
    self.assertFalse(result.passed, result.summary())
    self.assertEqual(result.n_responded, 0)
    self.assertTrue(all(0.0 < d < max(result.required) for d in result.divergence))


class TestAbstentionOnOneProbeIsNotEvidenceAgainst(unittest.TestCase):
  """A probe the adapter refuses contributes nothing, and does not condemn the request."""

  def test_a_request_is_decided_on_the_probes_it_answered(self):
    predictor = OutOfRangePredictor(refuse_column=0, limit_mm=20.0)
    result = interventional_check(predictor, gate_requests(), seed=0)
    self.assertTrue(result.passed, result.summary())
    self.assertEqual(result.n_abstained, 0)
    self.assertTrue(any(np.isnan(row[0]) for row in result.probe_divergence))

  def test_the_partial_abstention_is_reported_rather_than_absorbed(self):
    predictor = OutOfRangePredictor(refuse_column=0, limit_mm=20.0)
    result = interventional_check(predictor, gate_requests(), seed=0)
    self.assertTrue(
      any("abstained on some probes" in note for note in result.notes), result.notes
    )

  def test_a_request_whose_every_probe_abstains_produces_no_evidence(self):
    result = interventional_check(DeclaredPredictor(abstain=True), gate_requests())
    self.assertFalse(result.passed)
    self.assertEqual(result.n_abstained, INTERVENTION_K)
    self.assertEqual(result.n_responded, 0)
    self.assertTrue(all(np.isnan(d) for d in result.divergence))


class TestTheGateConstantsAreNeverDerivedFromData(unittest.TestCase):
  """The one repair this gate must never receive is a smaller bar.

  A seed-unstable gate has exactly two fixes available: change what it measures, or
  lower the number it measures against until the seed that refused stops refusing.
  The second one fits a published constant to the single submission it was about to
  reject, and it leaves no trace in the diff beyond one changed float. This test is
  the trace.
  """

  def test_the_values_have_not_moved(self):
    self.assertEqual(INTERVENTION_ALPHA, 0.25)
    self.assertEqual(INTERVENTION_K, 16)
    self.assertEqual(INTERVENTION_QUORUM, 0.5)

  def test_the_gate_defaults_are_the_constants_themselves(self):
    import inspect

    signature = inspect.signature(interventional_check)
    self.assertEqual(signature.parameters["alpha"].default, INTERVENTION_ALPHA)
    self.assertEqual(signature.parameters["k"].default, INTERVENTION_K)
    self.assertEqual(signature.parameters["seed"].default, 0)

  def test_each_constant_is_assigned_once_at_module_scope_from_a_literal(self):
    import ast
    import pathlib

    import plr_lr.learn.predict as module

    source = pathlib.Path(module.__file__).read_text(encoding="ascii")
    tree = ast.parse(source)
    watched = ("INTERVENTION_ALPHA", "INTERVENTION_K", "INTERVENTION_QUORUM")
    found = {name: [] for name in watched}
    for node in ast.walk(tree):
      targets = []
      if isinstance(node, ast.Assign):
        targets = list(node.targets)
      elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
        targets = [node.target]
      for target in targets:
        if isinstance(target, ast.Name) and target.id in found:
          found[target.id].append(node)

    for name in watched:
      nodes = found[name]
      self.assertEqual(
        len(nodes), 1, f"{name} is assigned {len(nodes)} times; it must be assigned once"
      )
      node = nodes[0]
      self.assertEqual(node.col_offset, 0, f"{name} is assigned inside a scope")
      self.assertIsInstance(node, ast.Assign, f"{name} is not a plain assignment")
      self.assertIsInstance(
        node.value, ast.Constant, f"{name} is computed rather than written down"
      )
      self.assertIsInstance(node.value.value, (int, float))

  def test_the_axes_are_written_down_and_not_generated(self):
    import ast
    import pathlib

    import plr_lr.learn.predict as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="ascii"))
    assigns = [
      node
      for node in ast.walk(tree)
      if isinstance(node, ast.AnnAssign)
      and isinstance(node.target, ast.Name)
      and node.target.id == "INTERVENTION_AXES"
    ]
    self.assertEqual(len(assigns), 1)
    value = assigns[0].value
    self.assertIsInstance(value, ast.Tuple)
    for element in value.elts:
      self.assertIsInstance(element, ast.Tuple)
      for number in element.elts:
        self.assertIsInstance(number, ast.Constant)

  def test_nothing_in_the_module_reads_a_submission_into_the_bar(self):
    """``alpha`` reaches the comparison from the signature and from nowhere else."""
    import inspect

    import plr_lr.learn.predict as module

    source = inspect.getsource(module.interventional_check)
    self.assertIn("float(alpha) * commanded_mm / ENVELOPE_STEP_MM", source)
    for forbidden in ("percentile", "quantile", "argmax", "getattr(p, "):
      self.assertNotIn(forbidden, source, f"the gate reads {forbidden!r}")


class TestPolicyIsNotAWorldModel(unittest.TestCase):
  """A taxonomy stated in prose is not a rule; a constructor error is."""

  def test_the_base_adapter_raises_on_construction(self):
    with self.assertRaises(NotAWorldModel):
      ActionEmittingAdapter()

  def test_every_named_policy_adapter_raises_on_construction(self):
    for adapter in (GR00TAdapter, Pi0Adapter, CosmosPolicyAdapter):
      with self.assertRaises(NotAWorldModel) as caught:
        adapter()
      self.assertIn(adapter.model, str(caught.exception))

  def test_the_message_routes_the_caller_to_the_policy_seam(self):
    with self.assertRaises(NotAWorldModel) as caught:
      GR00TAdapter()
    message = str(caught.exception)
    self.assertIn("plr_lr.learn.policy.VLAPolicy", message)
    for named in ("GR00T", "pi0", "Cosmos Policy"):
      self.assertIn(named, message)

  def test_it_does_not_even_define_predict(self):
    self.assertFalse(hasattr(ActionEmittingAdapter, "predict"))

  def test_constructor_arguments_do_not_get_it_through(self):
    with self.assertRaises(NotAWorldModel):
      GR00TAdapter(checkpoint="nvidia/GR00T-N1.7-3B", state_dim=4)


class TestReferenceNulls(unittest.TestCase):
  """The two nulls satisfy the contract in every respect except the one that matters."""

  def test_both_nulls_satisfy_the_protocol(self):
    for null in (BlindPredictor(), NoisePredictor()):
      self.assertIsInstance(null, RolloutPredictor)

  def test_both_nulls_produce_validating_predictions(self):
    request = a_request()
    for null in (BlindPredictor(), NoisePredictor()):
      out = validate(null.predict(request), request, null)
      self.assertFalse(out.abstain)

  def test_the_nulls_declare_that_they_were_fit_on_nothing(self):
    for null in (BlindPredictor(), NoisePredictor()):
      self.assertEqual(null.training_fingerprints, frozenset({NO_TRAINING_DATA}))

  def test_noise_is_reproducible_across_calls_and_ignores_the_actions(self):
    null = NoisePredictor(sigma=2.0)
    request = a_request()
    other = a_request(step_mm=999.0)
    np.testing.assert_array_equal(
      null.predict(request).state, null.predict(other).state
    )

  def test_noise_differs_between_episodes(self):
    null = NoisePredictor(sigma=2.0)
    first = null.predict(a_request(episode_id="ep-0000")).state
    second = null.predict(a_request(episode_id="ep-0001")).state
    self.assertFalse(np.allclose(first, second))

  def test_a_null_refuses_a_degenerate_declaration(self):
    with self.assertRaises(ValueError):
      BlindPredictor(state_dim=0)
    with self.assertRaises(ValueError):
      NoisePredictor(sigma=0.0)
    with self.assertRaises(ValueError):
      BlindPredictor(confidence=1.5)


class TestEveryRefusalIsReachable(unittest.TestCase):
  """Each refusal in the taxonomy is produced by some call, and none is decorative."""

  def reach(self):
    """One call per refusal class. Returns the class actually raised, per entry."""
    request = a_request()
    predictor = DeclaredPredictor()

    def no_predictor_injected():
      InjectedPredictor(name="empty").predict(request)

    def broken_output():
      validate(a_prediction(request, state=np.zeros((5, 4))), request, predictor)

    def absurd_position():
      state = np.zeros((2, 4))
      state[0, 0] = PLAUSIBLE_MM * 10.0
      validate(a_prediction(request, state=state), request, predictor)

    def undeclared_provenance():
      validate(a_prediction(request), request)

    def undeclared_component():
      other = DeclaredPredictor()
      other.fitted_components = (FittedComponent("probe", "UNDECLARED", 10, "probe"),)
      validate(a_prediction(request), request, other)

    def ignores_its_actions():
      require_action_conditioned(BlindPredictor(), gate_requests())

    def policy_registered_as_a_world_model():
      GR00TAdapter()

    return {
      NotImplementedError: no_predictor_injected,
      AdapterOutputError: broken_output,
      FloatingPointError: absurd_position,
      ProvenanceUndeclared: undeclared_provenance,
      UndeclaredComponent: undeclared_component,
      InterventionFailed: ignores_its_actions,
      NotAWorldModel: policy_registered_as_a_world_model,
    }


  def test_every_declared_refusal_is_raised_by_some_call(self):
    reachable = self.reach()
    self.assertEqual(set(reachable), set(REFUSALS))
    for error, call in reachable.items():
      with self.assertRaises(error):
        call()

  def test_the_refusals_are_typed_and_not_bare_exceptions(self):
    for error in REFUSALS:
      self.assertTrue(issubclass(error, Exception))
      self.assertNotIn(error, (Exception, BaseException, RuntimeError, ValueError))

  def test_every_exception_defined_in_the_module_is_registered(self):
    """A new refusal that nothing can reach is documentation, not a rule."""
    import plr_lr.learn.predict as module

    defined = [
      value
      for value in vars(module).values()
      if isinstance(value, type)
      and issubclass(value, Exception)
      and value.__module__ == module.__name__
    ]
    for error in defined:
      registered = error in REFUSALS or any(issubclass(r, error) for r in REFUSALS)
      self.assertTrue(registered, f"{error.__name__} is unreachable and unregistered")


if __name__ == "__main__":
  unittest.main()
