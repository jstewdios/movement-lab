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
