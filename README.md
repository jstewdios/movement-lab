# Movement Lab 00A — Balance and Contact

A small independent probe of existing Microduck policies. No training, upstream
edits, checkpoint conversion, or services. Uses the installed upstream Python
and its registered task/runner so the checkpoint observation normalizer is kept.

From this directory:

```bash
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$PWD/.cache/matplotlib"
PY=/home/james/microduck_rl/.venv/bin/python
RUNS=/home/james/microduck_rl/logs/rsl_rl/velocity
$PY -m movement_lab.evaluate "$RUNS/2026-10-25_20-16-28_movement-lab-001/model_99.pt" --out results/99-balance
$PY -m movement_lab.evaluate "$RUNS/2026-10-25_20-38-15_movement-lab-001/model_348.pt" --out results/348-balance
$PY -m movement_lab.report results/99-balance results/348-balance --out results/comparison.html
$PY -m movement_lab.report results/348-balance --out results/348-inspect.html
$PY -m unittest discover -s tests -v
```

Open the HTML in a browser. Reports are self-contained, with summary overlays,
expandable plots for every joint, and provenance. `--vx 0.1` requests forward
motion; the default zero command probes balance. Use a new output directory for
each run; existing records are never overwritten. CUDA access is needed.
`--seconds 5 --seed 42` are defaults. Trials stop at the first termination or
timeout, without auto-reset. Requested durations round up to a policy step.

## Evaluation conditions

Task: `Mjlab-Velocity-Flat-MicroDuck`, current installed play configuration,
one environment, flat ground. Curricula, observation corruption, pushes,
encoder/IMU offsets, and environment randomization events are disabled.
BAM remains active: 7.35 V battery, voltage sag gain 0.1 V/Nm, fixed four-physics-step
command delay. Initial base offsets are x=y=yaw=0 and z=0.125 m relative to the
robot's configured default; joint offsets and velocities are zero. Head/body
pose command deltas are zero. These are controlled nominal conditions, not a
reconstruction of each checkpoint's randomized training environment.

`trial.npz` contains arrays with the same leading sample dimension, sampled at
every physics step (normally 200 Hz); policy actions update at 50 Hz:

| Array | Meaning |
| --- | --- |
| `time_s`, `control_index` | Actual simulator time and zero-based policy step |
| `action` | Raw actor output held for that policy step, before wrapper clipping |
| `command` | Twist (3), head delta (4), body delta (6), in upstream order |
| `joint_target_rad` | Scaled/offset position command before actuator delay |
| `joint_position_rad`, `joint_velocity_rad_s` | Actual generalized joint position/velocity |
| `joint_force_nm` | `qfrc_actuator`: actuator contribution in joint DoF space |
| `actuator_force` | Scalar transmission-space actuator output, upstream ordering |
| `body_pose_w` | Every body link's world position (m), quaternion **wxyz** |
| `body_velocity_w` | Every body link's world linear (m/s), angular velocity (rad/s) |
| `foot_found` | Netforce sensor match count, positive means contact |
| `foot_force_w_n` | Net contact force, world frame, left then right (N) |
| `qpos`, `qvel` | Full simulator generalized state for further analysis |

The plain walk model has 14 scalar actuated joints. Names are stored in metadata;
this implementation is intentionally limited to that task/model. `initial.npz`
saves the initial generalized state separately; summary statistics exclude t=0.
`metadata.json` includes checkpoint SHA256, upstream Git revision and dirty status,
package/GPU/Python versions, seed, timestep, names, stop status, summary, and config
and recorder hashes. The full resolved config and original run parameter YAMLs
are saved for inspection; YAML is copied as data, never executed/deserialized.

## Measurement semantics and limitations

mjlab's `mj_step` leaves derived kinematics one substep stale. The physics-rate
recorder calls `sim.forward()` after integration, then updates sensor caches,
before reading all measurements. Effort and contact forces are therefore
endpoint solver evaluations with the current held actuator control, not exact
interval-averaged force or impulse. Extra forward calls may affect solver warm
starts. `--control-rate` uses the normal stepping path and samples only after
each policy step, allowing a trajectory comparison to quantify that effect.
This mode uses a private sensor cache invalidation method; the primary mode
wraps the public scene update method. Both depend on the recorded mjlab version.

Contact coverage is specifically the two foot collision geoms against terrain.
It does not measure arbitrary body support, self-contact, center of pressure,
support polygons, or balance margin. The walk robot deliberately omits many
body collisions; apparent recovery or floor support is model-dependent. `found`
is a contact match, not a force-threshold detector. In these trials the raw world-z sensor support force is negative (opposite
to upward ground reaction on the robot); the plots and arrays preserve it.
Forces can include compliant
contact near the surface and solver artifacts. Peaks between sampled endpoints
are missed. No measured signal here is a hardware load-cell reading.

`qfrc_actuator` is not total joint load, BAM electrical power, current, thermal
stress, or battery consumption. It excludes constraint and passive force
contributions. Absolute mechanical work uses endpoint `sum(abs(tau*qdot))*dt`;
it is a sampling-based effort proxy and includes both positive and negative
work, without electrical efficiency or regeneration. RMS force pools all joints
and times; it can hide a single overloaded joint (inspect joint plots too).
Root tilt measures trunk z-axis alignment with world up; a low tilt alone does
not establish stable balance. Contact fraction is sampled occupancy, not a
validated gait event classifier. Reports warn about differing conditions, but
whole-trial summaries with different termination times are not matched-horizon
comparisons. One seed and nominal initial state do not establish robustness.

## Infrastructure inspected

- Upstream `README.md`, `AGENTS.md`, velocity task/robot/BAM configuration,
  saved run parameters, and actuator implementation.
- Installed mjlab 1.3.0 `scripts/play.py`, environment stepping/reset,
  entity data, contact sensors, vec-env wrapper, and recorder manager.
- Playback supports native MuJoCo or Viser and local checkpoint hot swapping.
  Its optional `VideoRecorder` writes under the checkpoint's run directory;
  use this independent recorder to keep upstream read-only.
- The built-in recorder manager has reset and policy-step hooks, but no
  physics-substep snapshot hook. A local scene-update wrapper avoids copying
  the environment stepping implementation or changing upstream.

For interactive policy playback, upstream's existing `play` command supports
`--checkpoint-file PATH --viewer viser --num-envs 1`. That is a fresh interactive
rollout, not replay of the recorded trial or these exact controlled conditions.
The saved arrays and offline plots are the first-slice inspection interface.
No ONNX export is needed. Upstream robot mesh licensing still applies to any
future rendering/export work; this repository does not copy robot assets.

See [first results and validation](results/VALIDATION.md) and the generated
[comparison report](results/comparison.html).

## Movement Lab 00B — Disturbance and Recovery

00A commands and records above remain the baseline. The optional 00B evaluator
adds an instantaneous **world-x root-link velocity increment**, after 2 s of
zero-command standing. This changes generalized velocity, not position, joint
velocity, or actuator targets. It represents a reproducible momentum disturbance,
not a force applied at a particular point. Do not infer a force in N from it.
The requested and actual times, full generalized velocities immediately before
and after, and first integrated velocity are saved in metadata. Timing is
quantized to a physics step (including floating-point clock error).

```bash
PY=/home/james/microduck_rl/.venv/bin/python
# New output root for each complete study; completed trials are resumable.
$PY -m movement_lab.study00b --run --out results/00b-new
$PY -m movement_lab.validate00b results/00b-new
$PY -m unittest discover -s tests -v
```

The matrix is 22 four-second trials: checkpoints 99 and 348 and a configured
BAM default-position controller, seeds 42/43, Δvx = 0, ±0.03, ±0.06 m/s
for policies; default-position controls only (they cannot establish standing).
The default-position controller supplies zero raw actions (upstream action
scaling/offset gives configured default joint targets); it is powered and has
BAM delay/friction, **not passive or zero torque**. Its checkpoint is loaded
only to retain the shared evaluator and environment initialization.

The post-settling window is [actual intervention time − 0.5 s, intervention).
Trials failing the explicit settled gate are retained but cannot be classified
as recovered. The report distinguishes upright survival, sustained recovery,
root drift, endpoint actuator effort/work, contact, ankle lift, foot shift,
and conservative step evidence. Recovery permits a new standing location.
A recovered flag is an exploratory threshold decision; inspect trajectories and
continuous metrics. Foot translations alone may be sliding, not stepping.
`summary.json` contains all metrics including both foot contact fractions and
peak lifts; `comparison.html` overlays synchronized trajectories and provenance.
`validate00b` checks identical pre-event trajectories against each matched
control and the sign of both velocity and displacement after integration.

See [00B validation and interpretation](results/00b/standing/VALIDATION.md) and
[00B comparison](results/00b/standing/comparison.html). The larger +0.15 m/s pilot
is stored separately and excluded from the comparison matrix.

### Tests with and without local 00B trajectories

From the repository root, with the Python dependencies installed:

```bash
python -m unittest discover -s tests -v
```

A fresh checkout runs the measurement tests and checks against the committed
00A trajectories. The optional `test_saved_interventions` checks each record in
`results/00b/standing`: metadata without `trial.npz` is explicitly reported as a
skipped subtest, not as validated evidence. Available trajectories still run all
saved-record assertions, even when other records are missing; corrupt or invalid
available records fail. The 00B raw trajectories are intentionally excluded from
Git and must be preserved locally or regenerated to validate those records.

To validate preserved records or a new study output directory, explicitly select
the directory containing the per-trial subdirectories (each with `metadata.json`
and `trial.npz`):

```bash
MOVEMENT_LAB_00B_FIXTURES=/absolute/path/to/00b-records \
  python -m unittest discover -s tests -v
# For example, after the study00b --out results/00b-new command above:
MOVEMENT_LAB_00B_FIXTURES=results/00b-new \
  python -m unittest discover -s tests -v
```

Explicit selection is strict: a nonexistent/empty directory or any discovered
metadata record lacking its trajectory fails instead of skipping. This checks
all discovered records, not completeness of the 22-trial matrix. The separate
`python -m movement_lab.validate00b /path/to/00b-records` command additionally
checks source/config provenance and matched-control causal comparisons; use it
with complete study output, including the saved sources and initial states.
Neither mode requires rerunning simulations when the original records exist.

Fixture-routing regression tests include a deliberately mismatched **real 00A**
trajectory and 00B metadata as a negative control: the existing sample-count
assertion must reject it. This tests validation behavior and is not evidence
that any missing 00B trajectory has been validated.

## Movement Lab 00C — The Boundary of Balance

The independent `study00c` runner reuses the corrected 00B intervention and
validator. It records whole-robot COM in addition to existing measurements via
an opt-in `--record-com` evaluator flag; ordinary 00A/00B evaluation is unchanged.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m movement_lab.study00c --phase pilot --out results/00c/boundary-new
# Inspect cost.jsonl, REPORT.md and validation.json before expanding.
$PY -m movement_lab.study00c --phase expand --out results/00c/boundary-new
$PY -m movement_lab.study00c --phase report --out results/00c/boundary-new
$PY -m unittest discover -s tests -v
```

Pilot: four 5 s trials, seed 42, both policies, zero and +0.06 m/s. Expansion
matches both policies and seeds 42/43 at every selected signed amplitude, with
controls. Starting at ±0.06, two rounds double amplitude if every matched cell
survives or refine below the smallest observed failure. One additional adaptive
probe and one small interior audit per direction check the sampled region.
There are at most 44 trials, amplitude capped at 0.48 m/s. These are sampled
outcomes without a monotonicity assumption or a continuous safe-region claim.
`adaptive_plan.json` preserves choices across resumes. Completed trials are
reused; incomplete directories are retained and cause an error, not overwritten.

`REPORT.md` describes conditions, outcome counts, region uncertainty and limits;
`comparison.html` overlays event-aligned policy/control trajectories and
`summary.json` includes continuous joint/contact/COM metrics. `divergence.json`
compares policies only through their common recorded horizon. `manifest.json`
hashes records, local raw trajectories and saved analysis sources. The new
return-to-origin flag supplements the unchanged 00B recovery criterion.
Raw 00C trajectories and evaluation logs are ignored by Git; all remain local.

### Published 00C artifacts and reproduction limits

The completed [scientific report](results/00c/boundary/REPORT.md) and
[self-contained comparison](results/00c/boundary/comparison.html) accompany
44 retained trial records, small initial states, resolved configurations,
checkpoint/source hashes, simulator versions, summary/divergence metrics,
validation results, adaptive decisions, and measured subprocess costs. Only
the two PNGs referenced by the report are published; the comparison embeds
all trajectory plots and representative joint plots.

Raw trajectories, evaluation logs, caches, other PNG exports, and archived
trial directories remain local and are ignored by Git. The
[exclusion notes](results/00c/excluded/README.md) and archived-artifact hash
manifest document the pre-event mismatch, repeat, and interruption. Nothing
was excluded because it fell, and no acceptance tolerance was relaxed.
`boundary/manifest.json` describes the complete local evidence set, including
unpublished files; a fresh checkout cannot verify those missing files or rerun
raw-record validation without restoring or regenerating the trajectories.
The published report and visualization can be read without them.

Regeneration requires the separately installed upstream environment, matching
simulator versions, CUDA, and the two checkpoint files at the recorded paths
(or equivalent relocated paths in the baseline metadata). Checkpoints and
upstream robot assets are not bundled. Seeds do not guarantee bitwise replay
across runs or machines; the observed pre-event mismatch remains unexplained.
The measured 1846.9 s cost includes completed attempts and the repeat, excludes
the untimed interrupted attempt, and is not GPU kernel time. Publication does
not add simulation trials or change the scientific results.

## Movement Lab 01A — Free Movement Discovery

A separate open-loop experiment uses the upstream **curated groundcontact** model
and the existing BAM XL330 M6 position-target interface. It loads no policy or
checkpoint and changes no 00A–00C code or upstream files. This model differs from
00A–00C's feet-only walk collision model; their physical outcomes are not directly
comparable.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 $PY -m movement_lab.discover01a --out results/01a/new
$PY -m movement_lab.discovery_metrics results/01a/new
$PY -m movement_lab.validate01a results/01a/new
$PY -m movement_lab.replay01a results/01a/new/head_turn
$PY -m unittest discover -s tests -v
```

Use a new output directory. `--only zero` runs a single baseline smoke test;
full-pilot validation deliberately requires all eight sequences and the repeat.
The fixed pilot is nine 3 s trials (27 simulated seconds / 5,400 physics steps),
including powered zero action and its repeat. Commands are named-joint offsets
with a cosine rise/hold/return, bounded to 0.25 rad and checked against joint limits
with 0.02 rad command margin. The action-manager mapping determines joint order;
position targets, measured joints, actuator output, generalized states, link poses,
whole-body COM, and foot sensors are recorded at 200 Hz. Commands update at 50 Hz.
The last 1.2 s holds default targets. Four-physics-step actuator delay, fixed 7.35 V
supply, 0.1 V/Nm sag, and the upstream actuator torque/friction constraints remain
active. Zero action is powered default-position control, not zero torque.

The fixed descriptive archive groups observed tilt, horizontal COM excursion,
measured joint range, and endpoint class. It retains every record in every bin;
there is no reward optimization, elite replacement, or adaptive search. CPU
MuJoCo geometry checks at recorded states supplement the GPU foot sensors,
recording all modeled terrain contacts and penetration, including nonfoot support.
These checks do not reconstruct GPU contact forces. Stop flags cover nonfinite
states, penetration over 10 mm, generalized speed over 100, joint-limit overshoot
over 0.03 rad, and foot force over 100 N. These are diagnostic thresholds, not
hardware safety ratings; stop decisions take effect at the next control boundary
(up to 20 ms). Native NaN termination remains active; walking/fall termination is
removed so the falling motion can be observed.

Open `replay.html` for a portable measured-link-center replay with joint targets
versus actual positions, or use `replay01a` for full geometry in a desktop MuJoCo
viewer. Both replay saved states without reintegrating dynamics. The local compiled
MJB includes upstream geometry and requires a compatible MuJoCo version; it is
ignored by Git. Raw NPZ trajectories are retained, and model/config/source hashes
are recorded. Regeneration requires the installed upstream environment and GPU.
The HTML uses 50 Hz decimated samples played at 12.5 frames/s (four times slower
than recorded time); local NPZ retains every physics sample and is excluded from Git.

See [01A report](results/01a/atlas-v2/REPORT.md),
[atlas](results/01a/atlas-v2/atlas.png), and
[recorded-state replay](results/01a/atlas-v2/replay.html).
The [attempt log](results/01a/ATTEMPTS.md) documents preflight fixes and excluded
diagnostic outputs. No roll, jump, or recovery success is inferred from a fall,
feet-off interval, low endpoint speed, or contact alone.

## Movement Lab 01B — Stable Baseline and Controlled Departures

Completed bounded study: a validated 1 mm-clearance starting condition supports
small head-yaw articulation with partial return under checkpoint 99. Bilateral
flexion leaves a stable offset; lateral and unilateral commands miss the declared
response thresholds. No unloading, stepping or falling is demonstrated.

Read the [controlled-departure report](results/01b/departures/REPORT.md) and
[comparison with measured replays](results/01b/departures/comparison.html).
Eight six-second departure-study GPU runs completed with the unchanged 3 mm
penetration limit. The [initial failed baseline](results/01b/REPORT.md) and
[landing diagnosis](results/01b/landing-diagnosis/REPORT.md) preserve the progression:
the original 3.16 mm overlap also occurs in 00A, while a separate 1 mm-clearance
initialization passes the same gate twice before departure evaluation.

The [publication and reproduction notes](results/01b/PUBLICATION.md) distinguish
published reports/replays/configurations/source/provenance from local raw data,
compiled robot geometry, logs and redundant exports. No prior study was changed.
Full saved-record validation requires those local artifacts and the separately
installed upstream environment/checkpoint; a fresh checkout cannot verify missing
raw evidence. Portable replays use saved link centers without integrating dynamics.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m unittest discover -s tests -v
$PY -m movement_lab.validate01b results/01b/gpu-gate
$PY -m movement_lab.validate01b results/01b/landing-diagnosis/clearance-1mm
$PY -m movement_lab.validate01b results/01b/landing-diagnosis/clearance-1mm-repeat
$PY -m movement_lab.validate_departures01b
```
