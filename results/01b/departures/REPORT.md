# Movement Lab 01B — Controlled departures from the revised baseline

**Result: stable standing with small head-yaw articulation followed by partial return was demonstrated. Bilateral flexion produced a persistent, stable configuration change. Lateral and unilateral commands produced smaller responses that did not meet the declared success thresholds. No unloading, stepping, falling or physically invalid trajectory was observed.**

[Interactive comparison and measured replays](comparison.html) · [Joint tracking](joint-tracking.png) · [Continuous measurements](comparison.png) · [Full numerical summary](summary.json)

## Baseline and bounded method

The candidate initialization remains HOME joints, zero generalized velocity, and root z reset 0.1181823627 m (approximately 1 mm sole clearance). Checkpoint 99, its upstream observation pipeline/normalizer, curated **groundcontact** collision model and BAM constraints remain active. There is no post-reset state teleportation, root pinning, retraining or upstream change.

An independent six-second zero-residual run in the departure recorder passed the original baseline gate. Every recorded array matches the preceding landing-diagnosis baseline exactly. A final independent zero-residual run again matches all arrays exactly. Each intervening trial has exactly the same initial qpos/qvel and 400 pre-command physics samples as the new zero control. Resolved physics configuration and compiled model hashes match across all eight trials. Thus departures begin from the same measured settling trajectory, rather than merely the same nominal joint targets.

The declared budget was **eight six-second runs**, seed 42: zero control, signed head yaw, bilateral flexion, signed lateral hip-roll residuals, unilateral left flexion, final zero control. All eight completed: 48 simulated seconds / 9,600 physics samples. Simulator float accumulation gives 6.00009 s per record. The seven post-baseline subprocesses consumed 294.48 s including setup/recording/reporting, excluding subsequent independent validation. Baseline trial recording took 24.46 s; its full subprocess time was not captured, so total known time is a **318.93 s lower bound**, plus baseline startup and offline analysis. This is not GPU kernel time. No extra GPU run, amplitude search or continuation was performed.

Commands are smooth cosine residuals: start at 2 s, rise over 0.6 s, hold for 0.4 s, return over 0.6 s, then observe for 2.4 s. The actual composed target is **HOME + actor output + residual**, scale 1 rad/unit; actor feedback and BAM's four-substep delay, voltage/sag/friction/torque constraints remain active. Targets retain a 0.02 rad hard-limit margin. Raw actor, residual, composed target, actual joints and effort are separately saved. Returning the residual to zero does not force feedback targets or the robot back to the undisturbed trajectory.

| Residual | Named-joint offsets at peak |
|---|---|
| Head yaw ± | head_yaw ±0.08 rad |
| Bilateral flex | hip pitches L/R −/+0.03; knees +/−0.05; ankles −/+0.02 rad |
| Lateral ± | both hip_roll joints ±0.025 rad |
| Unilateral left flex | left hip_pitch −0.02, knee +0.035, ankle −0.015 rad |

## What counted as controlled

The [fixed plan](PLAN.md) retains the **3 mm penetration limit**, nonfinite/nonfoot-contact stops and physical constraint checks. Each pre-command [1.5,2) s window must be settled. Whole-trial tilt must stay below 12°. The final half-second must have both feet contacting, no nonfoot support, COM speed ≤0.03 m/s, root angular speed ≤0.3 rad/s and horizontal per-axis COM range ≤5 mm.

An absolute return window additionally requires horizontal COM offset ≤5 mm, vertical offset ≤2 mm, orientation change ≤3° and every joint offset ≤0.03 rad relative to the pre-command means. A continuous 0.5 s stable/returned window must occur after the residual ends. **These broad tolerances alone do not count as successful movement.** A controlled response must also have actual matched-control joint motion ≥0.005 rad or COM motion ≥0.5 mm during the pulse, and each detected response channel must contract to less than half its peak by the final mean. All criteria were implemented before departure evaluation.

## Measured outcomes

Differences below use the zero-residual trajectory at identical sample indices. Joint peak is the largest absolute single-joint response during [2,3.6) s; final is the largest absolute final-half-second mean offset. COM is full 3D distance, not commanded displacement.

| Trial | Actual joint peak / final (rad) | COM peak / final (mm) | Peak effort after 2 s (Nm) | Interpretation |
|---|---:|---:|---:|---|
| Zero control | 0 / 0 | 0 / 0 | 0.1034 | Stable standing |
| Head − | 0.02715 / 0.00495 | 0.104 / 0.026 | 0.1040 | Articulation with partial return |
| Head + | 0.04330 / 0.02065 | 0.479 / 0.153 | 0.1164 | Articulation with partial return; marginal contraction criterion |
| Bilateral flex | 0.01588 / 0.01353 | 1.709 / 1.547 | 0.1070 | Stable changed configuration; not controlled return |
| Lateral − | 0.00149 / 0.00046 | 0.114 / 0.027 | 0.1037 | Below declared response thresholds |
| Lateral + | 0.00113 / 0.00062 | 0.032 / 0.062 | 0.1034 | Below declared response thresholds |
| Unilateral | 0.00492 / 0.00289 | 0.252 / 0.157 | 0.1037 | Small articulation; below thresholds |
| Final zero | 0 / 0 | 0 / 0 | 0.1034 | Exact repeat of initial control |

**Head:** the measured yaw responses have the commanded signs. After residual removal, the negative response shrinks by approximately 82%; positive shrinks by approximately 52%, leaving a 0.02065 rad yaw offset. Positive therefore only narrowly meets the exploratory half-peak rule. Neither establishes exact return to the original head pose. Full-body tilt maxima are 6.05° and 6.23°, near the control's 6.08°. These are controlled small articulations with partial return, not whole-body turning.

**Bilateral flex:** the largest responses are left ankle (−0.01588 rad) and right knee (−0.01299 rad); left knee moves only about +0.00094 rad relative to control. The command does not establish a symmetric crouch. COM shifts predominantly forward, and about 91% of its peak matched displacement remains at the end. The final pose is quiet and within the broad absolute tolerances, but fails the contraction rule. This explicitly demonstrates why survival or motionlessness is insufficient. Feedback targets also remain altered after residual removal; the residual itself is exactly zero.

**Lateral and unilateral:** small actual articulation and load modulation are retained as observations, not rounded up to successes. Hip-roll changes are far smaller than the 0.025 rad residuals. Unilateral motion peaks at 0.004924 rad, just below the fixed 0.005 rad criterion; that cutoff was not changed. No leg lift or meaningful lateral COM transfer is established.

All absolute stable/returned windows began at the first sampled time after residual removal and lasted at least 0.5 s. The numerical onset delay is approximately 0.000035 s due to clock accumulation; this **does not measure a near-zero recovery time**. Confirmation requires the following half-second, and the positive-head and flex results retain visible offsets. Inspect the final means and full curves rather than treating this onset metric as a settling time constant.

## Contact, loading and physical credibility

Every run's worst whole-trial penetration is **1.156 mm**, during the shared landing. Contact is audited at every 5 ms sample. There are no nonfoot contacts, no post-settling foot-contact transitions, and no post-command interval with foot |Fz| below 0.1 N. That diagnostic force threshold is not a calibrated unloading detector. Continuous two-foot contact gives no evidence of stepping or lifted-foot unloading. Small ankle-origin shifts can be compliance, articulation or sliding and are not classified as steps.

Raw sensor Fz is negative for upward reaction on the robot. During the late hold [2.8,3) s, left load magnitude changes relative to control by about +0.108 N (head −), +0.210 N (head +), +0.330 N (flex), −0.076 N (lateral −), +0.129 N (lateral +) and −0.006 N (unilateral), with roughly opposite right-foot changes. These show load modulation without loss of support. The lateral traces do not meet the declared actual-motion thresholds for controlled weight shifting; sensor load fraction is not center of pressure or a support-margin estimate.

The maximum trunk tilt across the study is 6.92° in flex. The shared landing peak actuator effort is 0.2470 Nm; the table isolates later effort so that common initialization does not dominate the comparison. Endpoint forces and work are sampling-based mechanical estimates, not motor electrical energy, impulse or thermal validation.

## Validation, provenance and limits

All **31 tests pass**, including actual-motion/return criteria, a persistent-offset negative control, quiet-fallen/contact-invalid records, insufficient sustained return and failed pre-command settling. Independent saved-record validation checks every sample's CPU geometry/COM/contact distance, synchronized finite arrays/quaternions, actuator-to-joint transmission mapping, residual schedule/slew/return, composed-target limits, source/checkpoint/model/config hashes, exact initial states and pre-command causal matching. Foot occupancy agrees between native geometry and GPU sensors. CPU replay audits geometry; it does not independently validate GPU force-solver equivalence.

Matched outcomes in `summary.json`, `execution.jsonl` and the comparison apply the explicit zero-control subtraction. Per-trial `departure_metrics` were computed without a matched record; their `controlled_response=false` means that paired success was not evaluated there. The paired analysis is authoritative for controlled-response classification. Generic inherited 01A `summary.valid` uses its older 10 mm descriptive convention; the stricter 01B physical gate remains authoritative.

The baseline completed physics but its report finalization failed due to function-name shadowing; offline recovery also exposed a NumPy-boolean JSON serialization issue. Both reporting issues were fixed and tested before departures. Original raw arrays, executed source snapshots, pre-finalization metadata and failure log are preserved. Finalized analysis sources are separately hashed. No dynamics were rerun to repair reporting. Provenance for that baseline was reconstructed offline by checking checkpoint, upstream hashes, software versions and resolved config against the prior landing record; unavailable total wall time is explicitly null. Later runs completed normal provenance capture.

Raw full-rate trajectories, compiled geometry and logs remain local and hashed. Portable replays show measured link origins and actual/target joints, not collision meshes; desktop full-geometry replay sets saved states without reintegration. The [summary](summary.json) contains per-joint extrema, COM, foot-loading/contact, effort and return metrics. Source snapshots and the study manifest allow local revalidation; raw trajectories/models are not bundled for publication.

These results use one policy, seed, initialization, pulse shape and small amplitude per group. Signed head trials are different conditions, not replications. Exact zero controls demonstrate local repeatability, not robustness. Thresholds are exploratory; the positive-head contraction and unilateral response lie close to their cutoffs. Instrumentation refreshes forward solves and may affect warm starts. Curated groundcontact remains an incomplete body collision model. No controller-removal test identifies whether friction, feedback or another mechanism causes persistent offsets.

**Scientific transition demonstrated:** stable two-foot standing → small head articulation → partial return while retaining support. Flex additionally demonstrates stable stance reconfiguration without sufficient return. Reversible whole-body weight transfer, unloading and stepping remain unestablished. A useful next bounded study would repeat the head conditions and distinguish persistent leg offsets from slower return; none was launched.

The original failed baseline, landing diagnosis and all earlier 00A–01A evidence are preserved. No upstream edits, retraining, commits, pushes or merges occurred.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m movement_lab.validate_departures01b
$PY -m unittest discover -s tests -v
# Example full-geometry saved-state replay:
$PY -m movement_lab.replay01a results/01b/departures/head-negative/head_-1
```
