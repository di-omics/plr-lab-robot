# plr-lab-robot

PyLabRobot LabRobot: bring a robotic arm under PyLabRobot control, then give it eyes and hands.

> The Python package is `plr_lr` (short, for imports). The repository is named
> `plr-lab-robot` so its purpose is obvious at a glance; they are the same project.

PyLabRobot ships an arm front end (`ExperimentalSCARA`) and a hardware driver
(PreciseFlex), but no simulation backend, so the arm API cannot be exercised
without a physical robot. `plr-lr` closes that gap and builds three layers on top,
in four phases:

```
  arm  ->  vision  ->  manipulation  ->  learn
```

1. **arm** - a simulation backend for `ExperimentalSCARA`: joint and Cartesian
   state, gripper state, a home interlock, a workspace envelope, and a command
   trace. Stand the arm up and move plates with no hardware.
2. **vision** - an eye-in-hand (wrist-mounted) camera: hand-eye calibration,
   a detector seam, and a closed-loop guided pick that corrects a taught pose
   from what the camera sees.
3. **manipulation** - dexterous skills beyond plate handling, starting with
   uncapping and recapping threaded vessels. Every skill is gated: precondition,
   execute, verify.
4. **learn** - record what the arm did, fit a model of it, plan against the fit,
   and score the result against null controls that must fail. On top of that sits
   **DeferLab**, a benchmark that scores a world model on the predictions it
   declined to make, and the screen that carries the benchmark's own frozen
   threshold into the gate in front of a motion.

Everything runs simulation-first and CPU-only, and is validated by planting known
ground truth and scoring recovery. Swap the simulation backend for a hardware
backend and the same scripts drive a real arm.

## Install

```bash
pip install -e .
```

Runtime dependencies are `pylabrobot` and `numpy`. Real SOTA detectors
(RF-DETR, SAM2) are an optional `vision` extra; the simulation path needs
neither weights nor a GPU. The scoring core is lighter still: `plr_lr/learn/selective.py`
imports nothing outside the standard library, and `scripts/score_standalone.py` is a
single-file copy of it that runs on a bare `python:3.9-slim` with zero installs, so an
outside submitter can score a result file without installing this package at all. A CI
job runs exactly that, with no `pip install` line, to keep the claim honest.

The benchmark harness around it (`plr_lr/learn/bench/`) does use numpy in two modules,
`controls.py` and `runner.py`. The split is deliberate: arithmetic that produces a
published number stays dependency-free and separately checkable; the machinery that
drives episodes through a model does not have to be.

## Quickstart

### Phase 1: stand up the arm

```python
import asyncio
from plr_lr import Labware, Workcell

async def main():
    wc = Workcell.sim()          # ExperimentalSCARA + simulation backend
    await wc.setup()             # connect + home

    plate = Labware(name="assay_plate")
    wc.add_site("incubator_out", x=180, y=0, z=12, occupant=plate)
    wc.add_site("reader_in", x=-180, y=40, z=12)
    await wc.move_plate("incubator_out", "reader_in")

asyncio.run(main())
```

### Phase 2: vision-guided pick

Calibrate the wrist camera, detect the plate, and shift the taught pick to the
true position. A 7 mm placement error is corrected to sub-millimeter residual
even with pixel noise (see `examples/02_vision_guided_pick.py`).

```python
from plr_lr.vision import calibrate_eye_in_hand, estimate_offset, vision_guided_pick
```

### Phase 3: dexterous manipulation

```python
from plr_lr.manipulation import DecapSkill, RecapSkill

decap = DecapSkill(wc.arm, wc.world, "sample_tube",
                   coords=wc.coords_for("rack_a1"), cap_park=wc.coords_for("cap_park"))
result = await decap.run()       # rotates while lifting by thread pitch, then verifies
```

### Phase 4: learn from what the arm did

```python
from plr_lr.learn import ResidualDynamics, SamplingPlanner, evaluate

model, fit = ResidualDynamics.fit(episodes)   # fit.improved is True, False or None
report = await evaluate(lambda: SamplingPlanner(goal, dynamics=model), task, n_trials=5)
report.valid                                  # False unless null controls ran and failed
```

`examples/05_learn_from_demonstration.py` runs the whole loop against an arm with
a planted defect (gain 0.85, a fixed per-move offset) and recovers both to the
printed digits. The recovery is the least interesting part of it. The point is
the three answers a fit is allowed to give: `improved=True` on four episodes,
`NO IMPROVEMENT` on an arm that already tracks perfectly, and `improved=None` on
a single episode, because train and holdout are then neighbouring frames of the
same motion. `None` is not `False`. The data did not refute the correction; it
was never in a position to test it.

Run the worked examples end to end:

```bash
PYTHONPATH=. python examples/01_stand_up_arm.py
PYTHONPATH=. python examples/02_vision_guided_pick.py
PYTHONPATH=. python examples/03_decap_recap.py
PYTHONPATH=. python examples/05_learn_from_demonstration.py
```

## Kinematics: telling the arm where to go

Until now this repo could exercise an arm API and could not tell an arm where to go.
`_joints` and `_cartesian` were independent variables: `move_to` with joint angles never
updated the Cartesian pose, and `move_to` with a Cartesian pose never updated the joints.
The two could disagree without limit and nothing noticed.

`plr_lr.arm.kinematics` closes that with modified Craig D-H for the xArm 6 and 7, joint
limits taken from the SDK's own `x_config.py`, and a reach envelope derived from that
geometry rather than quoted.

```python
from plr_lr.arm import xarm6, forward, inverse

m = xarm6()
pose = forward(m, [0.3, -0.5, 0.4, 0.2, 0.6, 0.1])
result = inverse(m, pose)          # 8 solutions, not one
```

The asymmetry between the two directions is the whole design.

**Forward kinematics is exact.** Given joints and geometry, the pose is determined. Round
tripping a configuration through FK then IK recovers the pose to within 1e-6 mm in position
and 1e-8 rad in orientation, on every solution the solver returns. Those are the bounds the
suite asserts; the observed residuals are smaller, and the asserted number is the one quoted
here because a bound nothing tests is a number waiting to go quietly wrong.

**Inverse kinematics is not.** A 6-DOF arm reaches the same Cartesian point in several
distinct configurations, so `inverse` returns **all** of them and choosing is a named
policy rather than a hidden default. That matters physically: the wrong branch reaches the
identical point by swinging the elbow through the deck.

Four things it refuses rather than approximating:

- **A pose outside the envelope.** It reports the distance and the bound and offers no
  nearest-reachable alternative, because silently moving somewhere else is how a plate ends
  up next to the nest instead of in it. The bound is derived from the shipped D-H over the
  declared joint limits and ignores self-collision, so it is necessary and not sufficient.
- **Geometry it cannot source.** `unverified_model` carries provenance and refuses to
  compute a pose at all. An arm driven on invented link lengths moves confidently to the
  wrong place.
- **A solution outside a joint limit.** Returned flagged, never silently.
- **A near-singular pose**, where the joint velocities required for a bounded Cartesian
  velocity diverge.

And `reachable()` says nothing about whether a path exists. Conflating the two is how an arm
gets commanded into a move that is valid at both endpoints and passes through the bench.

### Repeatability is not accuracy

The distinction the datasheet invites you to get wrong, and the one that matters most for a
lab. A repeatability figure bounds returning to a **taught** pose. It does not bound the
error when moving to a **computed** one, which is what IK produces, because absolute
accuracy also carries link tolerance, joint offset, deflection and thermal error.

So a `Pose` knows its own origin. `PoseOrigin.TAUGHT` and `PoseOrigin.COMPUTED` are not
interchangeable for a placement that has to land inside a nest's lead-in chamfer, and
`stations` reports which of your stations are which.

## Stations: a plate move that is checked rather than assumed

`workcell.py` was a dict of taught poses and a `move_plate` that teleported occupancy, so
`incubator_out` and `reader_in` were string keys with no incubator and no reader behind
them. A workcell that cannot say what a station **is** cannot say whether a move between
two of them is legal.

`plr_lr.arm.stations` computes legality instead of asserting it. A move is refused when the
source holds nothing, the destination is full, either pose is missing, either pose is
unreachable, or the destination instrument is mid-cycle. That last one is not a queueing
problem: a plate placed into a reader that is still reading is a crash.

Handed two stations with nothing set, it returns four refusals rather than permission,
including the one worth quoting: *an untaught station is a name, not a place.*

`sequence()` checks a whole run, because a later move can be illegal only because an
earlier one left a station occupied, and reports the first that fails and why.

A completed move is also not a seated plate. `MoveOutcome` distinguishes a placement the
arm merely finished from one a sensor confirmed, the same way a provenance layer separates
software intent from an instrument's word.

## DeferLab: scoring the decision not to predict

Lives at `plr_lr/learn/bench/`. Run it with `python -m plr_lr.learn.bench run`.

Every world-model benchmark scores a rollout that was produced. This one scores
the decision not to produce one. It scores on bench tolerances rather than on
pixels, it credits a correct refusal exactly as much as a correct prediction, it
refuses to publish any score at all for a model that ignores its action argument,
and the threshold it freezes is the same object the screen carries into the gate
in front of a motion. None of the arithmetic below is new. The ordering is the
design.

The order of operations is enforced and is not a matter of taste:

```
  interventional gate  ->  null control suite  ->  per-task rows
```

The gate runs before any episode is scored and its failure is a **refusal, not a
low score**, because a low score still gets ranked and a ranked non-world-model
is the exact claim-versus-evidence mismatch this benchmark exists to indict. The
suite runs second and no scorecard is emitted unless it passes. Only then does a
single number exist. Reversing any two of those lets a number exist before the
thing that would have withheld it, and once a number exists somebody quotes it.

### The tasks

Five synthetic tasks, tier S, CPU-only, declared in `plr_lr/learn/bench/tasks.py`
as frozen dataclasses. A task there is a declaration, not a generator: it fixes
the state fields and their units, the per-field tolerance every downstream metric
divides by, the horizons that may be scored, and the episode count that buys a
rankable row.

| id | title | state | tolerance | horizons | n | renderer |
|---|---|---|---|---|---|---|
| S1 | `reach-regime` | x, y, z, rz | 2.0 mm and 0.05 rad | 1, 2, 4, 8 | 400 | none, state only |
| S2 | `residual-drawdown` | `residual_uL` | 0.30 uL | 1, 2, 3 | 400 | `tip_view` |
| S3 | `dispense-volume` | `volume_uL` | 0.50 uL | 1, 2, 3 | 400 | `column_view` |
| S4 | `plate-seating` | dx, dy, drz, seated | 0.5 mm **and** 0.5 deg jointly | 1, 2, 4 | 400 | `microplate` |
| S5 | `tip-engagement` | `z_engage_mm`, attached | 0.75 mm | 1, 2, 4 | 400 | `tip_view` |

Tolerance is per field and is never collapsed into one weight. Adding a
millimetre to a degree needs a weight, a weight chosen after seeing the errors is
a fitted parameter, and the resulting scalar is labelled in whichever unit was
written first. Every criterion is per-field error divided by that field's
tolerance and then a max over fields, which has no weight in it at all. That is
how S4's joint half-millimetre-and-half-degree criterion becomes one number.

**Unknowable episodes are interleaved inside every task and are never split out
into a task of their own.** A dedicated "ambiguous" task is won outright by
`if task_id == "T6": abstain`, which is four tokens long, needs no model, and
would top the abstention column of any leaderboard that shipped one. Episode
identifiers carry no knowability token, the pack refuses a contiguous block, and
the per-task unknowable fraction has to land inside `UNKNOWABLE_BAND = (0.05,
0.40)` or the abstention metrics are refused outright: outside that band they are
a statement about the base rate rather than about the model.

Unknowability is built by proven independence or by full occlusion, never by
render collision. Each task declares `hidden_latent` and `hidden_branches`, so
the ceiling a non-abstaining model faces on those episodes is `1 /
hidden_branches`, a number in the declaration rather than an adjective in a
docstring. A build-time gate in `packio.py` advances every unknowable episode
under every branch of its hidden latent and requires the terminal states to sit
more than two tolerances apart; if they do not, no manifest is emitted. That gate
exists because an earlier construction hid the S1 latent inside a drift that only
fired from the second step, which left all 80 unknowable horizon-1 episodes
byte-identical for every value of the latent while the docstring still said the
ceiling was chance.

The pack `python -m plr_lr.learn.bench run` builds by default is a self-contained
demonstration pack, not the scored fixture, and it says so in the `source` field
printed on every row built from it. It exists because a harness that cannot run
until an unrelated repository ships is a harness whose refusals have never been
executed. Scored fixtures arrive over the versioned scene-pack format in
`plr_lr/learn/bench/packio.py`, which is the entire interface to the vision side
of the build: no package import in either direction, and a test asserts it.

### The headline

```
hit_i  = 1 if max_d ( |s_hat_i[d] - s_i[d]| / tau[d] ) <= 1 else 0
AURC        = (1/N) * sum_{m=1..N} risk(m)          # risk-coverage, sorted by confidence
AURC_oracle = (1/N) * sum_{m=1..N} max(0, m - (N - e)) / m
AURC_random = e / N
E_AURC      = AURC - AURC_oracle                    # the headline, lower is better
```

`AURC`, `AURC_oracle` and `AURC_random` print in the same row as `E_AURC`,
always. The band a score sits in is never a second lookup. The correct
constant-confidence null band is `AURC_random - AURC_oracle` and **not** zero,
and reading the gap to zero as "close to perfect" is the exact mistake this
layout exists to prevent.

Selective risk at zero coverage prints `null`, never `0.0`: a risk of zero over
an empty set is the most flattering possible lie. Tolerance prediction error is a
median with an IQR; the mean is refused, and so is any percentile-derived
threshold and any max-over-frames selection.

The act/defer table is frozen and prints whole:

```
act_i = 1 if (c_i >= theta and not abstain_i) else 0
ADA   = ( |{act & hit}| + |{defer & !hit}| ) / N
```

`theta` is fitted by split conformal on a **disjoint calibration pack** drawn
from a different seed, then frozen and reprinted verbatim as `calibrated_on` on
every verdict it later decides. Fitting it on the pack being scored would be
fitting a decision rule to the answer key. The full 2x2 prints every time, with
`unsafe_act_rate` and `wasted_defer_rate` beside it, and `ADA` alone is never
printed.

### Abstention credit

The novel column, and the one that has to be read with its denominator:

```
credit_i = 1 if (unknowable_i and abstain_i)
         = 1 if (not unknowable_i and not abstain_i and hit_i)
ACR              = sum(credit) / N
RefusalRecall    = |{unknowable and abstain}| / |{unknowable}|
FAR              = |{knowable and abstain}|   / |{knowable}|
```

Abstaining on a knowable episode scores zero. There is no free lunch, and N7 is
the control that proves it. **`ACR` is never printed without `FAR`, and
`RefusalRecall` is never printed without `RefusalPrecision`.** That is not a
formatting convention: `AbstentionCredit` has no attribute named `acr`, and the
credit rate is reachable only through `credit_and_false_alarm()`, which hands
back the false-alarm rate in the same tuple. A model that abstains on everything
scores a respectable ACR, and quoting that alone is unfalsifiable.

Every rate in the package is a `Rate`, which cannot be built from a float. Each
one carries the counts it came from, so its Wilson interval is always available
and a naked `0.93` can never enter a report.

### The null controls

Ten of them, in `plr_lr/learn/bench/controls.py`. No scorecard is emitted unless
the suite passes, and "nothing was checked" and "the check passed" stay different
answers: an empty result list is not a pass.

| id | control | required outcome |
|---|---|---|
| N1 | `ConstantPredictor`, state unchanged | must fail tolerance |
| N2 | `ShuffledActionPredictor`, true dynamics with actions permuted | must fail; a paired CI including zero prints `NOT ACTION-CONDITIONED` and withholds the headline |
| N3 | `ConfidenceShuffle`, 1000 permutations | defines the no-information floor a model has to beat |
| N4 | `AlwaysConfident` | must land at `AURC_random - AURC_oracle` |
| N5 | `PixelPerfect` | must top the `NOT RANKED` pixel column and must lose the E-AURC ranking |
| N6 | `OracleLeak`, handed the true terminal state | **must pass at TolAcc 1.0.** If it fails the harness is broken, not the model |
| N7 | `AlwaysAbstain` | ACR equal to the unknowable base rate, `FAR = 1.0` |
| N8 | `SeedLeakPredictor`, fit on the published dev seeds | must not close the gap on the withheld half |
| N9 | `ClampOnlyScreen`, confidence is minus the clamp count | must not reproduce the arm correlation |
| N10 | `BlindPredictor`, `NoisePredictor` | must **fail the interventional gate**, producing a refusal rather than a low score |

N5 is the reason a pixel column exists at all. PSNR and SSIM appear under a
heading that reads literally `NOT RANKED`, so a control that tops them outright
can be watched losing the ranking in the same row. A citation does not convince a
video-model author; a demonstration does.

A control can also be satisfied without being evidence. `ClampOnlyScreen` on a
pack that carries no clamp column is trivially satisfied, and the report says so
in that sentence rather than counting it.

### The tiers, and what is never printed

| n | what prints |
|---|---|
| below 30 | `UNDERPOWERED n=<k>` and no number, under any flag |
| 30 to 199 | the rate with its Wilson interval, stamped `PROVISIONAL`, not rankable |
| 200 and up | rankable |

There is deliberately no parameter that turns the floor off, because a parameter
that turns it off is a parameter that gets used. Below 30 nothing is a
measurement. Between 30 and 200 something is visible but not comparable.

Sixteen refusal rules sit above that, enumerated as `REFUSAL_RULES` in
`plr_lr/learn/bench/report.py`, each reachable by a test. Among them: no
cross-task aggregate scalar ever (R10), because a composite index lets a strong
S1 hide a broken S3; no table mixing seed epochs, horizons or tolerances (R11);
no per-action latency claim off a host arrival clock (R12); and no hardware label
for a run with no device serial hash (R13).

### The command line

Six verbs, and not one of them takes a hardware flag. That absence is asserted by
an AST test rather than left to review: this package has exactly one hardware
permission gate and it lives on the rollout call, and a second gate is the
documented way the first one stops being the one that matters.

```bash
python -m plr_lr.learn.bench run          # gate, controls, then rows: the scorecard
python -m plr_lr.learn.bench controls     # the null suite on its own
python -m plr_lr.learn.bench probe        # the action-conditioning gate on its own
python -m plr_lr.learn.bench score        # the scorecard as receipt JSON
python -m plr_lr.learn.bench screen-fit   # fit the frozen threshold on a disjoint pack
python -m plr_lr.learn.bench receipt      # declarations, seeds and refusals, no rows
```

`scripts/score_standalone.py` is a single-file copy of the scoring core that
reads a result JSON and prints the same card. The duplication is the feature: a
submitter who wants to check a number before believing it should not have to
install this package, resolve its dependencies, and thereby run the scorer inside
the environment that produced the result. `tests/test_learn_selective.py` runs
the script against a fixture and asserts that both the printed card and the JSON
agree with the package implementation character for character, and CI runs it on
a stock `python:3.9-slim` image with `--network none` and no install line.

## The first scorecard

This is what `python -m plr_lr.learn.bench run` printed on 2026-08-03, on the
built-in demonstration pack, scoring the declared floor. It is not a good result
and it is not supposed to be. Long lines are wrapped to fit this page and a few
declaration strings are elided with `...`; every digit is as printed. Run the
command to get the unabridged card.

```
DeferLab v0.1 scorecard -- ResidualDynamics floor
  pack builtin-demo-20260803 | epoch demo-2026-08-03 | split dev | basis: simulated
  declared fitted_components: ['none declared']
  declared hardware: CPU only, no accelerator
  declared license_text: MIT, this package. No external checkpoint is involved.
  declared pack_separation: CHECKED: 1280 episode-horizons over 400 unknowable
    episodes, every branch pair more than 2 tolerances apart and every branch more
    than 1 from the nominal
  declared seeds: {'epoch': 'demo-2026-08-03', 'tie_break_seed': 0, 'gate_seed': 0, ...}
  declared tasks_scored: ['S1', 'S2', 'S3', 'S4', 'S5']
  declared threshold: calibration pack builtin-demo-20261780 ... n=792 | alpha=0.1000
    | rank=714/792 | theta=0.601822
  declared training_fingerprints: ['none:analytic-null-fit-on-no-data']
```

The interventional gate passed on all five tasks and all ten null controls came
out the way they had to. `N6 OracleLeak on S1 [must-pass]: TolAcc@1 1.0000,
exactly as required`. Five controls were satisfied trivially and say so on their
own lines, and the suite adds that all 55 control lines report no measured clamp
rate, because nothing in this run went through a safety envelope, so the absence
of a `THROTTLED` mark is an absence of measurement rather than a measurement of
zero. The measured trust horizons were S1 1, S2 3, S3 3, S4 4 and S5 4.

Then the rows: sixteen of them, five tasks by their declared horizons. Here is
the first, verbatim.

```
  -- S1 horizon 1 [epoch demo-2026-08-03, split dev, basis: simulated]
    tolerance: [2.0, 2.0, 2.0, 0.05]
    E-AURC 0.2148 (lower is better) | AURC 0.2407 | oracle 0.0259 | random 0.2175
      | no-information band 0.1916
    risk at zero coverage: null
    act/defer at theta=0.6018 calibrated_on='calibration pack builtin-demo-20261780 ...'
      act   & hit   313 | act   & miss    87
      defer & hit     0 | defer & miss     0
      ADA: 0.7825 [0.7395, 0.8201] 95% Wilson (313/400)
      unsafe_act_rate: 0.2175 [0.1799, 0.2605] 95% Wilson (87/400)
      wasted_defer_rate: 0.0000 [0.0000, 0.0095] 95% Wilson (0/400)
    ACR: 0.7750 [0.7316, 0.8132] 95% Wilson (310/400) | FAR: 0.0000 [0.0000, 0.0119]
      95% Wilson (0/320) | RefusalRecall: 0.0000 [0.0000, 0.0458] 95% Wilson (0/80)
      PROVISIONAL | RefusalPrecision: undefined, the model never abstained
    TPE median 0.3549 IQR (0.1397, 0.8476) in tolerance units over n=400
    SuccessGap@1: +0.2175
      LooseSuccess@1: 1.0000 [0.9905, 1.0000] 95% Wilson (400/400)
      TolAcc@1: 0.7825 [0.7395, 0.8201] 95% Wilson (313/400)
      delta +0.2175 [+0.1775, +0.2575] 95% paired bootstrap, 10000 resamples, seed 0
    U/N at w_fail=-1: +0.5650
    U/N at w_fail=-4: -0.0875 (headline, a stipulated preference, not a measurement)
    U/N at w_fail=-10: -1.3925
    declared compute (NOT RANKED): median 5 forward passes, median 0.0000 s wall clock
    pixel (NOT RANKED): not declared
    training overlap: none of 400 scored rows falls inside any declared training
      scope (corpus=deferlab split=dev epoch=demo-2026-08-03
      pack=builtin-demo-20260803 task=S1)
    NO INFORMATION: the confidence ordering did not beat its own shuffle
    rankable: False
```

and the end of the card:

```
  valid: False
  no cross-task aggregate is printed (R10): a composite index lets a strong task
  hide a broken one.
```

The command exits 1.

**Read what it says.** Fourteen of the sixteen rows print `NO INFORMATION`: the
floor's confidence ordering did not beat a shuffle of its own confidences over
1000 permutations, so its abstention claim is withheld rather than scored. Two
rows come out rankable, S2 horizon 2 and S3 horizon 2, and neither is a result
worth quoting. `RefusalRecall` is `0.0000` on fourteen of the sixteen rows,
`0.0125` on S3 horizon 3 and `0.0250` on S4 horizon 4. The floor essentially
never refuses. Its `ACR` of `0.7750` on the first row is not evidence of
judgement; it is what you collect for being right on the knowable episodes while
abstaining on nothing, and the `FAR: 0.0000` printed beside it is what makes that
legible at a glance. That pairing is the entire reason R5 exists.

The shape is the expected one. The declared floor is an analytic residual fit
with no uncertainty model attached to it, so it has nothing to say about which of
its own predictions to distrust, and the harness says exactly that instead of
ranking it anyway.

**A good-looking first row would have been the warning sign.** If a thirty-line
analytic fit had come back with high refusal recall and a strong E-AURC on its
first run, the conclusion would not have been that the floor is clever. It would
have been that the unknowable episodes were findable without a model, or that the
confidence column was correlated with the answer key by construction, or that the
tie-break ordering was doing the work. Every mechanism above, the interleaving
check, the separation gate, the confidence shuffle, the frozen threshold fitted
on a disjoint pack, exists to make that failure loud rather than flattering. The
first thing the finished harness did was refuse to say anything nice about its
author's own predictor, which is the only first result that would have been
believable.

Every row reads `basis: simulated` and will keep reading it until a hardware run
record with a device serial hash exists.

## The screen: the same threshold, in front of a motion

`plr_lr/learn/screen.py` is the only place in the package where a prediction
becomes permission to move. Admission requires **all** of five conditions, with
no weighting and no way to trade one against another: the adapter did not
abstain; the confidence is at least `theta`, the same threshold the benchmark
froze, reprinted verbatim as `calibrated_on` on every report it decides; the
horizon is inside the measured trust horizon; the readout says it measured
something; and the newest frame is younger than a declared staleness bound,
checked against `FrameRef.t_capture_s` rather than against when the prediction
happened to be computed.

Three properties are load-bearing and easy to undo by accident.

**Deferral is a terminal state, not a retry.** There is no fallback action, no
reduced-speed variant, no "try the safe version instead". The documented failure
mode of learned controllers is that they convert uncertainty into unwarranted
actionability, and a fallback action is precisely that conversion wearing a
cautious name: the run continues, something moves, and the abstention appears in
the log as though it had been honoured. At most two re-observations are allowed,
because looking again is not acting. After that the run stops safe and stays
stopped.

**A clamped action is not the action that was screened.** The safety envelope
rewrites a command after the screen has passed on it, so the ACT verdict belongs
to a command that is not the one about to be sent. Every clamped step is recorded
as `ABSTAIN(CLAMPED)`. That is not fastidiousness: the measured clamp rate of the
existing random null against the default envelope is close to half its steps, and
without this rule an arm study would report "abstention predicts failure" when
what it measured was "clamping predicts failure".

**Every abstention reason that applies is reported, not the first one hit.**
Repairing the staleness and re-running is otherwise how the low confidence gets
discovered, one round trip at a time.

Between the screen and any real arm sits `rollout(..., allow_hardware=True)`. The
check is `allow_hardware is not True`, an identity test, so truthy non-booleans
do not unlock it, and it is the only hardware permission gate in this package.
**This repository publishes no hardware backend.** Every pose in every example
came out of `SimulationArmBackend`. The refusal is the part that is checked
mechanically: it is exercised against a real PyLabRobot hardware driver class,
and it refuses a policy that does nothing at all just as firmly, because the gate
is about the backend and not about how dangerous the policy looks.

## The honest ceiling

A world model predicts the **visible half** forward: where the liquid will be,
where the labware will be, whether the plate will seat. It is structurally blind
to the **invisible half**: what the reagent actually is. A rollout never replaces
the orthogonal readout. It only decides how many actions you spend before you pay
for one.

Things a world model physically cannot resolve about a bench, at any scale:

- **Reagent identity and concentration in a correctly filled well.** Same volume,
  same meniscus, same look. That is why S3 carries an independence-constructed
  unknowable regime rather than a hard task.
- **Whether the contents match the label.** Provenance is a record, not a pixel.
- **Cross-contamination and carryover.** Sub-visual by definition.
- **Whether a tip is sealed rather than merely seated.** Engagement depth is
  visible; the pneumatic seal is not.
- **Temperature, sterility, and sample degradation.** No optical channel.
- **Whether an assay result is biologically real.** Task success is not
  experimental validity. This benchmark measures that gap; it does not close it.

Boundaries on the artifact itself:

- **No world model here runs usefully CPU-only.** V-JEPA 2-AC, Cosmos,
  RynnWorld-4D, Ctrl-World, Vid2World and iVideoGPT are all CUDA-dependent in
  practice. The harness is CPU-only; the models are not. Budget one 24 GB card or
  a rented GPU session. Torch adapters live in a separate repository for exactly
  this reason: torch, CUDA and per-checkpoint licenses cannot enter this package
  without permanently breaking its `pylabrobot` plus `numpy` invariant.
- **No world model here belongs in a servo loop.** V-JEPA 2-AC is 16 s per
  action, and NVIDIA's own world-action-model figure is 590 to 800 ms per action
  chunk. Against a 20 ms control period that is roughly 30 control periods for
  the chunk and 800 for the single action, and naive open-loop execution is
  already reported collapsing past a delay of about 13. The world model screens
  and plans; it never closes the loop. Predict, screen, commit a short chunk,
  execute open-loop under the envelope, re-observe.
- **The synthetic tier measures calibration under shift, not perception
  difficulty.** A thirty-line classical readout can beat a video world model on a
  200 px synthetic render. That readout ships as the declared floor row, and a
  world model that does not beat it on a task has produced no evidence of value
  on that task. The claim lives in the abstention and calibration columns under
  interleaved unknowable episodes, drift onset, glare and occlusion, not in the
  raw accuracy column.
- **A recorded fixture proves the contract, not the checkpoint.** A green CPU
  suite cannot detect that upstream weights changed, that a license changed, or
  that a model regressed.
- **Seed secrecy is a speed bump, not a guarantee.** The generator is public, so
  anyone can enumerate seeds, and presenting a withheld scoring range as a
  defence would be dishonest. The real defences are the interventional gate, N8,
  and declared training fingerprints.
- **`isinstance(m, RolloutPredictor)` is not evidence the thing can predict.** The
  Protocol is structural, so a bare `InjectedPredictor` with no callable injected
  satisfies it and then raises on the first call. No code path in this package
  treats an isinstance check as a capability check.
- **Every row reads `basis: simulated` until a hardware run record with a device
  serial hash exists.** The lab OT-2 would not power on on 2026-08-01, and every
  hardware path in that driver is simulation-only and is never called
  bench-validated. The same rule binds here.

## Layout

```
plr_lr/
  world.py            deck model: sites, labware, poses, capped/held state
  qc.py               readiness gate (setup, home, gripper, hand-eye rms)
  arm/
    sim_backend.py    SimulationArmBackend (implements SCARABackend)
    workcell.py       Workcell: taught sites + move_plate
    kinematics.py     modified Craig D-H, xArm 6 and 7; FK exact, IK all branches
    stations.py       a move is computed legal rather than asserted legal
  vision/
    transforms.py     rigid 4x4 transforms
    camera.py         pinhole intrinsics + eye-in-hand pose
    hand_eye.py       eye-in-hand calibration (Kabsch/Umeyama)
    detector.py       Detector seam + SimDetector + RF-DETR/SAM2 stubs
    guided_pick.py    ray-plane back-projection + closed-loop pick
  manipulation/
    skills.py         Skill base: precondition/execute/verify
    grasp.py          grasp planning for caps
    decap.py          DecapSkill / RecapSkill
  learn/
    types.py          Observation, Action, Step, Episode, field and unit tables
    record.py         capture: observe(), freedrive sessions, scripted runs
    dataset.py        on-disk episode store, plain JSON
    dynamics.py       DynamicsModel seam, KinematicDynamics, ResidualDynamics
    planner.py        SamplingPlanner: search over what a model predicts
    policy.py         the Policy seam, the nulls, injected-transport VLA shells
    rollout.py        SafetyEnvelope, and the one hardware permission gate
    evaluate.py       score a policy, with null controls that must fail
    synthetic.py      MistrackingBackend: arms that misbehave in known ways
    predict.py        the adapter contract, validate(), interventional_check()
    readout.py        pixels or latents back to a pose, components declared
    screen.py         Screen: the five conditions and a closed set of refusals
    selective.py      risk-coverage, E-AURC, ACR/FAR, Wilson, bootstrap. stdlib only
    bench/
      tasks.py        S1-S5 declarations, tolerances, hidden latents
      packio.py       the versioned scene-pack format and its build-time gates
      metrics.py      tolerance error, success gap, trust horizon, training overlap
      controls.py     N1-N10 and their bands
      report.py       BenchRow, BenchReport, the sixteen refusal rules
      runner.py       gate, then controls, then rows
      commands.py     six verbs, none of them with a hardware flag
      __main__.py     python -m plr_lr.learn.bench
examples/             one runnable script per phase
scripts/              the standalone scorer and the house-rule checker
tests/                plant-and-recover validation
docs/phases.md        what each phase is for
```

## Validation

```bash
python -m pytest -q
```

`1065 passed`, measured on 2026-08-03. That is the observed count and not a
rounded one, because a README reporting a suite it did not run is the first thing
in a repository to stop being true.

Use pytest rather than `python -m unittest discover -s tests -t .`. Both run, but
the unittest runner collects 1042 of the 1065 and then prints OK: the 23
house-rule tests are plain pytest functions, so unittest discovers the file, runs
none of it, and says nothing about having skipped it. A green run that quietly
checked 23 fewer things is exactly the failure mode the house-rule tests exist to
catch, so it is not the command this page recommends.

The suite plants ground truth and scores recovery: the hand-eye transform is
recovered to sub-micron RMS, the guided pick recovers a known offset, decap flips
and verifies the capped state, and the dynamics fit recovers a planted gain and
offset to the printed digits. The benchmark half is tested the other way round,
against things that must fail: hand-computed AURC and E-AURC on a six-episode
fixture, every must-fail control failing, N6 passing, and every one of the
sixteen refusals reachable by a call. The readiness gate in `qc.py` mirrors the
di-omics wet-lab QC shape so the same pass/fail reporting carries into hardware
bring-up.

CI runs the suite on Python 3.9 and 3.12 with no extras installed, asserts that
`torch`, `tensorflow`, `jax` and `cv2` are not importable, lints with
`ruff check --select E4,E7,E9,F --line-length 100`, runs every example start to
finish, and scores a generated fixture with `scripts/score_standalone.py` inside
`python:3.9-slim` with `--network none` and no install line of any kind.

## On hardware

The vision and manipulation layers are backend-agnostic: they talk to the arm
through the PyLabRobot `ExperimentalSCARA` API. To move to a real arm, construct
`ExperimentalSCARA` with a hardware backend (for example `PreciseFlexBackend`)
instead of `SimulationArmBackend`; teach the sites, calibrate the camera, and the
same code path applies. `rollout` refuses until `allow_hardware=True` is passed at
the call site, which is deliberate and is the only such gate here.

No arm has been run. Nothing in this repository has moved a physical robot, no
hardware backend is published here, and every number on this page came out of a
simulation with a planted defect or out of a synthetic scene pack.

## License

MIT.
