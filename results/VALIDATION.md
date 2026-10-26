# First evaluation — 2026-10-25

Five-second, seed-42, zero-command trials on RTX 2070 SUPER. Both checkpoints
completed without termination or timeout. Config hashes match. No training.

| Measurement | model_99 | model_348 |
| --- | ---: | ---: |
| Maximum trunk tilt (deg) | 5.906 | 7.710 |
| Minimum root height (m) | 0.11457 | 0.11473 |
| RMS actuator joint force (Nm) | 0.04871 | 0.07100 |
| Peak absolute actuator joint force (Nm) | 0.24704 | 0.31628 |
| Absolute mechanical work proxy (J) | 0.07423 | 0.22637 |
| Contact fraction, left/right | 0.993 / 0.993 | 0.993 / 0.993 |

These rollouts show two-foot standing under a zero command. The later checkpoint
uses more effort in this particular condition. This does not rank their walking
ability or robustness. Each has a brief initial landing transient; no warm-up is
excluded. The simulation starts above settled standing height. Inspect the plots
before interpreting peak effort or contact forces as steady behavior.

Validation:

- Formula fixtures cover upright/90-degree tilt, known torque and velocity
  integration, contact occupancy, force magnitude, and invalid records.
- Saved-record integration checks cover synchronization, 200 Hz sample spacing,
  four samples per held actor output, zero commands, root/joint state mapping,
  unit-transmission actuator force mapping, and zero force without contact.
- `99-repeat` reproduces **every saved array bit-for-bit** from `99-balance` on
  this machine. This is evidence for local repeatability, not a guarantee across
  devices, driver versions, or future simulation releases.
- Independent gravity sanity check: the walk XML inertial masses sum to
  0.73724318 kg, giving 7.2323556 N at 9.81 m/s². Over t > 1 s in `99-balance`,
  the sum of raw foot world-z forces averages -7.2317433 N (0.0085% difference
  in magnitude). Thus the raw netforce sensor sign here is opposite to upward
  ground reaction on the robot. Records retain the raw sensor convention.
- Physics-rate versus ordinary stepping (`99-control-rate`), comparing only
  matching policy endpoints: maximum root xyz difference 0.0003073 m, maximum
  joint position difference 0.0038355 rad, and maximum joint actuator-force
  difference 0.0076148 Nm. Extra forward solves measurably affect the trajectory;
  the primary records must be interpreted as instrumented evaluations. Their
  effect is small in this trial, with no general bound for other behaviors.
- Policy-rate sampling also misses an effort peak: 0.18681 Nm recorded peak
  versus 0.24704 Nm in the physics-rate trial. Sampling and trajectory changes
  both contribute; do not treat this as a clean estimate of aliasing alone.

To repeat the two validation trials:

```bash
PY=/home/james/microduck_rl/.venv/bin/python
CKPT=/home/james/microduck_rl/logs/rsl_rl/velocity/2026-10-25_20-16-28_movement-lab-001/model_99.pt
$PY -m movement_lab.evaluate "$CKPT" --out results/99-repeat-new
$PY -m movement_lab.evaluate "$CKPT" --out results/99-control-rate-new --control-rate
```

The HTML comparison renders offline and includes all 14 joint command/actual
and effort traces for both checkpoints. No visual animation was recorded.
