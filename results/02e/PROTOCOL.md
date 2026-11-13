# 02E protocol: the unreliable sense of upright

Declared after the Phase 0 gate and short feasibility probes, and before any
Phase 1 battery, pilot or training. Base: `main` at `f363976` (02D merged).
02B/02C/02D sources, checkpoints, results and outcome definitions stay fixed.
All 02E work is under `movement_lab/*02e*`, `tests/test_upright02e.py` and
`results/02e/`.

## Question and hypotheses

Can Microduck stay balanced when its estimate of upright is slightly wrong, and
can its own actions help resolve that uncertainty?

- **H1, mechanism.** The 02D residuals use absolute projected gravity as a
  disturbance cue. A constant IMU misalignment is then a sustained false cue,
  which they turn into a large standing residual and an unforced fall. The
  frozen 02B reflex only shifts its stance. Prediction: steady pre-event
  coefficient magnitude and residual grow with |offset|. Learned controllers
  fall in quiet trials beyond a small, sign-dependent offset; the reflex does
  not at ±2°.
- **H2, observability.** IMU gravity alone cannot separate misalignment from a
  floor slope or a changed stance. Leg kinematics on level ground make the
  misalignment observable, but they report a floor slope as a sensor offset.
  Gyro and kinematic rotation increments are slope-independent, but the
  misalignment enters only at second order (offset × probe rotation). It is
  readable only if the soles do not rotate during the probe.
- **H3, remedies.** Explicit level-ground calibration should restore learned
  controllers under misalignment and should fail on slopes. Learning with
  randomized misalignment may instead buy insensitivity without attributing
  anything. An exponential history makes high-pass gravity available, so it may
  matter more under calibration uncertainty than it did in 02D.

## Information boundary

Controllers keep 02D's inputs: five normalized IMU channels (plus history for
the history arm). A calibration *front-end* may additionally read noisy joint
encoders (upstream U(±0.001) rad noise; optional fixed offsets ~U(±0.015) rad,
the upstream encoder-bias range), time since start and its own probe command.
It never reads contact, root state, the true offset, floor slope or the
disturbance. During its window (0.3–1.3 s, plus a 0.3 s smooth blend) every
residual is held at zero and the frozen reflex stands on the raw IMU. The
estimated rotation is then applied to both gravity and gyro for every
controller. A `gate` control holds residuals off for the same 1.6 s without
correcting anything.

Magnitudes come from upstream: random-axis IMU mounting error up to 6° in
training, a reported ~5° systematic board pitch offset corrected in the runtime,
and ±0.015 rad encoder bias. Misalignment rotates gravity and gyro, as in 02D.
A floor slope rotates physical gravity relative to the floor with a correct IMU.
Drift ramps the misalignment linearly by ±2° over 0.5–4.5 s (0.5°/s).

## Feasibility already observed (see ATTEMPTS.md)

On level ground, static gravity-vs-kinematics recovered ±2° pitch/roll to within
0.2°. A 2° slope read as a 2.08° offset. A hip-yaw probe produced no clean trunk
yaw with planted feet. Early rocking-probe estimators (rotation increments, then
cumulative rotation vectors) gave 5–150° errors. The soles rolled and twisted
0.7–3.7°, and the false corrections caused falls. Those formulations also had
two flaws:
- they scored roll misalignment, which a roll probe cannot observe;
- cumulative angles add a second-order integration error of the same size as the
  signal.

The declared probe estimator therefore compares *angular velocities*: the gyro
against the trunk rate implied by encoder central differences (±40 ms) with the
soles assumed fixed. A roll-only probe uses a single-axis fit, and only its
pitch component is scored. The probe is **only a diagnostic** in Phase 1B, with
an oracle decomposition: clean gyro vs true trunk rate (must be exact), noisy
gyro vs true rate, and noisy gyro vs kinematic rate from true joint velocities.
It becomes a candidate front-end only if, across the whole 1B battery, median
|pitch error| ≤ 0.5°, the maximum ≤ 1°, and no probe trial falls.

## Phase 1A: characterization (no learning)

Nine frozen 02D-evaluated controllers: frozen 02B reflex, 02C onset-oracle, 02C
IMU trigger, history s1–s3, memoryless s1–s3. Settings: nominal; pitch −4, −2,
−1, −0.5, +0.5, +1, +2, +4°; roll ±2°; drift ±2°; slope ±2° (15). Kinds: quiet,
forward 0.20 and 0.32 m/s, backward 0.08, false gyro 0.75 rad/s × 40 ms (5).
Two fresh seeds per kind (7,000,000,000+) with onset U(1.8, 3.4) s, 6 s
horizon. Seeds are paired across settings and controllers: 150 trials per
controller, 1,350 total.

Metrics: strict 02D `recovered_200hz` and step audit (unchanged), survival,
failure time/cause and validity. Also steady true pitch/roll and measured
gravity, coefficients, residual RMS, minimum and pre-event COM support margins
(toe/heel), torque and target saturation, settling time, final tilt and root
drift.

## Phase 1B: what the sensors can tell (estimators)

Reflex only, 3 s quiet (4.5 s for probes), four seeds (7,100,000,000+), 14
settings:
- pitch −4/−2/0/+2/+4 and roll ±2;
- slope pitch ±2 and roll +2;
- +2 offset with +2 or −2 slope;
- encoder offsets with 0 and +2 offset.

Estimators:
- `prior`: IMU only, reading vs the nominal quiet reading from separate seeds;
- `static`;
- `rock` at 0.05 and 0.10 rad roll, 1 Hz, 2 s.

For rock, decompose the offline error with noise-free encoders and true trunk
rotation (an oracle reference, never eligible). Report errors and the probe's
balance cost.

## Phase 1C: calibration front-end pilot (no learning)

Development battery D (seeds 7,200,000,000+):
- settings: nominal, pitch ±2/±4, roll +2, slope ±2, and +2 offset with
  encoder offsets (9);
- the five kinds above, two seeds each: 90 trials per configuration.

Controllers: reflex and the six 02D learned controllers. Front-ends: `none`,
`gate`, `prior`, `static`.

Gate for confirming `static` (vs `none`):
- over the six learned controllers on the misalignment cells (pitch ±2/±4,
  roll +2, encoders: 360 paired trials), at least 30 net paired recoveries;
- at nominal, zero quiet and zero backward losses;
- the reflex loses no nominal recovery.

Slope cells are reported, not gated (H2 predicts failure). `prior` is judged
the same way.

## Phase 1D: robustness-learning pilot

`learn02e` must first reproduce 02D's published `history-s72` generation 1
bitwise with randomization off. Pilots use 02D's revised curriculum and CEM
seed 72, 8 × 16 × 12 = 1,536 episodes, with misalignment of U(0, 4°) about a
random horizontal axis on every stratum (separate RNG stream):
- randomized history pilot;
- randomized memoryless pilot;
- matched memoryless control without randomization (new).

The published 02D history-s72 is the history control. Gate per arm, randomized
vs its matched control on D with front-end `none`:
- at least 10 net paired recoveries on the 60 misalignment trials;
- no nominal quiet or backward loss;
- nominal recoveries at least the control's minus one.

## Phase 2: confirmation, only for what passed

Training (only for arms that passed 1D): seeds 1/2/3, 12 × 16 × 12 = 2,304
episodes each. The curriculum and CEM seeds equal 02D's runs, so each run pairs
exactly with its published 02D counterpart. At most 13,824 episodes (02D's
Phase B scale), local CPU, at most six low-priority workers, and a 600 s launch
cap per run invocation. Every generation is atomic and resumable; final =
last training winner; no held-out selection.

Held-out battery H (seeds 8,000,000,000+), declared now:
- settings: nominal; pitch −4/−2/−1/+1/+2/+4; roll ±2; drift ±2; slope ±2; and
  encoder offsets at 0 and +2 (15);
- kinds: quiet; forward .08/.20/.28/.32; backward .08; left .12; false gyro (8);
- onset U(2.2, 3.4) s, three seeds: 360 trials per configuration.

Configurations:
- reflex and the six 02D controllers with `none` and with any front-end that
  passed 1C;
- trained arms with `none`.

Report paired gains/losses against the 02D counterpart, nominal quiet and
backward reliability as explicit regression gates, slope behavior, and
continuous metrics. Pooled trials are shared conditions, not independent
replicates.

## Integrity

Unchanged 200 Hz audit, validity, ≥40 ms flight, >2 mm clearance, >5 mm
relocation, ≥100 ms loaded replant and whole-trial recovery for a step. No
sliding or brief upright instant counts. Every attempt, failed pilot and
revision is preserved, and any revision is declared here before use. If a gate
fails, stop that branch and report it. Simulation only; no hardware claim.
