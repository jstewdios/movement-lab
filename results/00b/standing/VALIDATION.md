# Movement Lab 00B — evidence and validation

The corrected small study does **not demonstrate better disturbance recovery for checkpoint 348**. Both policies handle the tested mild disturbances, while 348 uses more RMS actuator joint force. This is consistent with greater effort maintaining posture under these conditions; it does not establish greater stiffness, energy consumption, or superior balance.

## Comparison

22 GPU trials on RTX 2070 SUPER: checkpoints 99 and 348, seeds 42/43, zero command, 4 s duration, intervention after 2 s, Δvx = 0, ±0.03, ±0.06 m/s. Two additional default-position controls are included in that count. Each policy has eight disturbed trials and two controls. All policy trials first pass the half-second settled-standing gate. All default-position controls terminate before intervention and are not recovery comparisons.

| Policy | Disturbed recoveries | Mean post-event RMS force Nm (all 10 conditions) | Mean sampled absolute work J | Disturbed peak root shift range mm |
|---|---:|---:|---:|---:|
| 99 | 8/8 | 0.04617 | 0.000872 | 0.470–1.624 |
| 348 | 8/8 | 0.06490 | 0.001520 | 0.647–4.430 |

Recovery time reaches the earliest eligible 0.020 s window for both policies in these trials, followed by a continuous 0.5 s stable window and stable final 0.5 s. This is a threshold floor, not evidence that their dynamics are identical. Both maintain foot contact throughout the post-event interval; no conservative step events are detected. Measured ankle translations are sub-step repositioning, rocking, or slip, not demonstrated stepping. Absolute trunk tilt depends on each policy’s baseline posture. Inspect per-condition trajectories rather than ranking peak tilt alone.

## Validation

- Eight unit/integration tests pass, including all original 00A tests, recovery-versus-upright fixtures, sliding-versus-step fixtures, and saved disturbance state/sampling checks.
- `validation.json` checks all 22 corrected records: finite synchronized samples, recorded command arrays identically zero, exact pre-event array equality against matched controls, checkpoint-pair config/version/seed/initial-state agreement, and recorded source/config SHA256 hashes.
- Each applied intervention has the requested signed generalized root velocity change within 2e-7 m/s; all other generalized velocities remain unchanged within 2e-7. First integrated root velocity AND displacement differ from the matched control in the requested direction. Intervention timing is approximately 2.005 s (one 5 ms step late due to floating-point clock comparison); actual times are recorded and used for analysis and plots.
- Generalized state/body-pose mapping, unit actuator transmission mapping, physics-step spacing, held action cadence, contact absence/zero force, and quaternion validity are checked by the original and added saved-record tests. Forces and work preserve 00A endpoint-solver and SI conventions.
- Seeds 42/43 give different policy trajectories despite identical initial generalized states. Inherited observation delay buffers remain enabled (base angular velocity and gravity delays range 0–1 policy frame with periodic updates); corruption/noise and randomization events are disabled. These are two seeded instrumented nominal trials, not a statistically powered robustness population.
- A separate +0.15 m/s pilot on model 99 verified injection (+0.150000006 m/s) and subsequent physical motion/rotation; it fell at 3.40 s. That asymmetric pilot is excluded from policy ranking. It motivated the modest comparison range.
- The first matrix exposed an upstream command sampler override: `rel_forward_envs=0.2` could command vx=0.3 even when velocity ranges were zero. That interrupted matrix is retained under `../matrix/` and explicitly excluded. Corrected 00B alone sets that bucket to zero. 00A behavior and results are preserved.

## Reproduction and provenance

Run `python -m movement_lab.study00b --run --out NEW_ROOT` with the upstream interpreter, then `python -m movement_lab.validate00b NEW_ROOT` and `python -m unittest discover -s tests -v`. Completed trials can resume; existing trial directories are never overwritten. Regenerate this report with `python -m movement_lab.study00b --out results/00b/standing`. See repository README for interpreter/checkpoint paths.

Per-trial metadata includes checkpoint SHA256, upstream revision/status, package/GPU/Python versions, seed, timings, termination, full event before/after/integrated velocities, body/joint names, config and recorder hashes. Initial states, full training YAML data, resolved configs, and recorder sources are retained. `study_manifest.json` hashes final analysis sources and record artifacts; `analysis_sources/` preserves final report/validation/assessment code.

## Limits

This is an instantaneous world-x root velocity increment, not a force pulse at a measured trunk point. It changes the robot’s translational momentum through a state intervention; contact constraints immediately redistribute it. Nominal forward/backward means world ±x at initial yaw zero. No finite-force magnitude, duration, force application point, or hardware equivalence is claimed.

The zero increment control uses the same velocity API round trip, which can introduce angular-coordinate floating-point roundoff. It is not claimed bitwise equivalent to 00A after intervention time (observed 4 s control versus 00A maxima: root/joint qpos differences <0.00044 and joint force differences <0.004 Nm). All matched pre-event arrays are identical. The ordinary 00A path never installs the intervention hook.

Recovery thresholds are exploratory, explicitly encoded in `disturbance.assess`, and allow a new standing location. These mild probes often remain within the stability envelope; they do not establish recovery from a large loss of balance or a recovery boundary. Follow-up should bracket that boundary for both checkpoints with matched conditions rather than extrapolate from the single large pilot.

Foot positions are ankle body origins, not contact patch centers. Conservative stepping requires sampled contact loss, >5 mm lift, then >5 mm displaced recontact. No detected steps does not exclude tiny motions or missed substep events. Contact covers only foot-ground geoms. Feet-only robot collisions, solver contact sign, endpoint force sampling, and extra forward solves retain all 00A limitations. No support polygon, center of pressure, electrical power, direct stiffness, or hardware safety measurement is provided.

No training, upstream edits, 00A result overwrites, or merge were performed.
