# Movement Lab 01C — Controlled Transitions

**Bounded pilot: 5 validated ten-second trials. Flexion established a stable changed configuration; opposite flexion and signed head yaw showed partial return. No exact return, symmetric crouch, meaningful height extension, unloading or step was established. Coverage omitted by the launch cap: rock-forward, rock-backward, final-control.**

[Compact atlas and synchronized recorded-state replay](comparison.html) · [Independent validation](independent-validation.json) · [Per-joint command/response analysis](joint-analysis.json)

## Evidence and smallest credible experiment

01B used the existing checkpoint-99 standing actor with additive joint-position residuals. All 14 scalar position targets are addressable through the action manager; this is not independent control of actual joint angles, body pose, force, or COM. HOME + actor + residual remains the action semantics, with unit scale, unmodified observation pipeline and checkpoint normalizer. BAM retains four physics steps of delay, voltage/sag, friction and torque constraints. Feedback can oppose commands; modeled limits do not guarantee tracking.

The 01B flex command did not produce a symmetric crouch: one knee barely moved and the principal displacement was forward. After the residual ended, actual offsets and actor targets remained altered. The final left-ankle offset was −0.01353 rad while its actor-target difference was +0.01745 rad; the final COM difference was predominantly forward (1.511 mm). This is evidence of a persistent closed-loop/plant state, not a still-active residual. Friction, compliance and controller adaptation cannot be separated causally without a further intervention.

Positive head yaw in 01B fell from a 0.04330 rad peak to 0.02065 rad at six seconds. Late actor-target difference was −0.01381 rad, and the actual joint was moving only about −0.000285 rad/s. Negative yaw retained −0.00495 rad. The late near-plateau favors controller/plant persistence over a fast return cut off by the evaluation window, but the old window alone cannot rule out slow recovery. This pilot changes the hold/return timing and observes longer; it cannot independently isolate timing from controller behavior. [Recomputed 01B windows](../evidence01b.json).

The 1 mm sole-clearance reset remains appropriate for this nominal controlled study: it reduces the unsupported landing transient without pinning the free root. Root reset z remains 0.1181823627 m, HOME joints and zero generalized velocity. Curated groundcontact, solver and unchanged **3 mm penetration limit** remain authoritative. It is not evidence that this initialization is optimal or robust.

## Protocol and interpretation

Preregistered bounded pilot: baseline, signed flex/extension, signed head yaw, signed sagittal rocking, final zero repeat. Eight ten-second trials, one seed (42), at most 80 simulated seconds. The driver stops before movements if the baseline or matched-history audit fails and stops launching after 510 wall seconds, reserving 90 seconds for the last trial. No training. Amplitude signs and command families form a small interpretable exploration; no scalar reward or displacement optimization and no outcome replacement. CLI exposes amplitude scale, rise, hold and return duration; the pilot fixes their magnitudes rather than running a factorial timing sweep.

Every pulse starts at 2 s, rises over 0.6 s, holds 1.0 s, and returns over 1.4 s; residual is zero from 5 s through 10 s. This extends the original 01B return and observation windows. Head peaks are ±0.08 rad. Flex peaks are hip pitches L/R −/+0.03, knees +/−0.05, ankles −/+0.02 rad; extension reverses them. Rock peaks are hip pitches L/R −/+0.02 and ankles +/−0.02 rad; reversing signs probes the opposite direction. Names describe commands, not guaranteed achieved motion. Asymmetric leg trials were omitted because 01B showed no threshold-level response or unloading.

The same independent physical gate checks the whole trajectory. Pre-command [1.5,2) s and final [9.5,10] s use 01B stability criteria: both feet supported, no nonfoot contact, tilt below 12°, COM speed ≤0.03 m/s, root angular speed ≤0.3 rad/s and horizontal per-axis COM range ≤5 mm. Late hold uses its final half-second. Actual matched-control response must reach 0.005 rad at a joint or 0.5 mm COM distance during the pulse. A stable changed configuration is an accepted transition result even if its offset persists. Partial return separately requires each detected response channel to contract below half its peak. Absolute final return additionally applies unchanged 01B pose tolerances; it is not exact recovery. Quiet survival alone is never evidence of return or an achieved movement.

## Achieved configurations and transitions

| Command | Peak / final joint response (rad) | Peak / final COM response (mm) | Hold / final stable | Outcome |
|---|---:|---:|---|---|
| baseline | 0.00000 / 0.00000 | 0.000 / 0.000 | True / True | zero / below response threshold |
| extend | 0.00435 / 0.00150 | 0.771 / 0.245 | True / True | stable transition with partial return |
| flex | 0.01675 / 0.01356 | 1.936 / 1.616 | True / True | stable changed configuration |
| head-negative | 0.02760 / 0.00513 | 0.109 / 0.057 | True / True | stable transition with partial return |
| head-positive | 0.04342 / 0.01976 | 0.479 / 0.236 | True / True | stable transition with partial return |

**Budget-limited coverage:** planned conditions not executed: rock-forward, rock-backward, final-control. These are untested, not negative movement results. The independent historical six-second baseline prefix was reproduced; a new ten-second final-control repeat was not obtained if listed above.

**flex:** dominant measured response is left_ankle (-0.01675 to 0.00000 rad relative to control), with final offset -0.01356 rad and final target offset 0.01769 rad. Absolute final 01B pose tolerances pass: True. This broad tolerance pass does not establish exact return.

**extend:** dominant measured response is right_hip_pitch (-0.00435 to 0.00007 rad relative to control), with final offset -0.00095 rad and final target offset 0.00264 rad. Absolute final 01B pose tolerances pass: True. This broad tolerance pass does not establish exact return.

**head-positive:** dominant measured response is head_yaw (0.00000 to 0.04342 rad relative to control), with final offset 0.01976 rad and final target offset -0.01295 rad. Absolute final 01B pose tolerances pass: True. This broad tolerance pass does not establish exact return.

**head-negative:** dominant measured response is head_yaw (-0.02760 to 0.00000 rad relative to control), with final offset -0.00513 rad and final target offset 0.00337 rad. Absolute final 01B pose tolerances pass: True. This broad tolerance pass does not establish exact return.

All differences are against the zero-residual control at identical sample indices. Peak is measured during the pulse, final is the final half-second mean. Per-joint data retain signed actual extrema, command amplitude, final target offsets and velocities. The atlas plots measured COM x/z and maximum joint response; synchronized replays use the same saved sample for selected trial and control, with raw joint targets, actual positions, COM and foot forces. These are measured link origins, not collision meshes. Full-geometry desktop replay remains available through `movement_lab.replay01a` on each trial folder without reintegrating dynamics.

## Contact, effort, validation and budget

Retained 5 completed records. Driver wall time: **510.80 s**, including subprocess startup, recording and per-trial validation; not GPU kernel time. First sandbox attempt failed CUDA initialization before producing samples and is excluded from movement outcomes. See [attempt log](../ATTEMPTS.md).

Worst penetration across validated records: 1.156 mm. Maximum sampled whole-trial absolute joint work: 0.0869 J. Work uses endpoint sum(abs(tau*qdot))*dt, with no electrical/thermal interpretation.

Every record passes hash/checkpoint checks, finite synchronized arrays, CPU saved-state geometry/COM/contact audits, joint/state mapping, actuator transmission mapping and 0.9601 Nm effort bound, target limits with 0.02 rad margin, measured joint overshoot ≤0.03 rad, exact initial and pre-command matching, residual-envelope timing and HOME + actor + residual composition. Any final control is required to match every recorded array exactly; missing planned controls are explicitly listed above. The new baseline must match all available historical 01B arrays over its six-second prefix. Model/configuration hashes are equal across conditions. Upstream source hashes are rechecked without editing upstream. **36 tests pass**, including persistent-offset acceptance without return, partial return, invalid contact and pulse timing.

Executed driver bookkeeping count: 6; authoritative validated trial count: 5. The original driver includes a budget-stop marker in its count when the cap is reached. The final driver source corrects this bookkeeping; the executed source snapshot and original cost file remain unchanged.

Raw sensor Fz has the opposite sign to upward ground reaction. Continuous foot contact and nonzero load indicate loading modulation, not stepping. Small foot-origin translation can be compliance or sliding. No lifted-foot unloading or step is claimed. COM position is not center of pressure; a quiet final half-second is a finite-horizon observation, not asymptotic stability or robustness.

## Limits and reproduction

Curated groundcontact remains an incomplete body collision model. CPU contact audits validate saved-state geometry, not GPU contact-force solver equivalence. Physics-endpoint forward/sense refreshes may affect solver warm starts. One controller, seed, reset and amplitude magnitude per command family. Signs are conditions, not replications. The longer/slower pulse and longer observation window are changed together: this is an exploratory transition atlas, not a causal estimate of a return time constant or isolated friction effect. Actual head/leg response must be read from the per-joint analysis; commanded flexion/extension and rocking do not automatically establish crouching, height extension, or reversible whole-body rocking. Stable reconfiguration is distinct from an origin return and from disturbance recovery.

No upstream edits, long training, commits, pushes or merges. Earlier studies and contact criteria are unchanged. Per-trial source/model/config/checkpoint/software/upstream hashes and initial states accompany local full-rate arrays. `manifest.json` hashes local data too; those hashes cannot validate missing trajectories or geometry after publication. The executed study snapshot and final offline sources are retained separately.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 $PY -m movement_lab.study01c --out results/01c/new-pilot
$PY -m movement_lab.report01c results/01c/new-pilot
$PY -m movement_lab.write_report01c results/01c/new-pilot
$PY -m unittest discover -s tests -v
```
