# Movement Lab 00C — The Boundary of Balance

44 matched nominal GPU trials; 1846.9 s cumulative subprocess wall time (includes startup, compilation and recording; not GPU kernel time).

Zero velocity commands; 5 s horizon; signed world-x root velocity increments after 2 s. Each signed amplitude has checkpoints 99/348 × seeds 42/43 after expansion. Actual event times, full pre/post velocities, initial states, policy/config/source hashes and simulator versions are retained in per-trial records. Raw trajectories stay local.

## Scientific summary

Both controllers show contact-maintaining recovery at the smaller tested disturbances and falling-like loss of posture at larger forward disturbances. At this sampling resolution, the policies have the **same categorical forward outcomes**: both seeds recover at +0.12 m/s and terminate at +0.18 and +0.24 m/s. This establishes an observed transition span, not a continuous recoverability threshold.

Backward outcomes depend strongly on seed: both policies and seeds recover at −0.18 m/s; at −0.24 m/s both seed-42 trials recover and both seed-43 trials terminate. Thus the tested backward responses tolerate a larger velocity increment than the forward responses, but there is no single seed-independent cutoff. The two seeds change inherited observation-delay behavior under identical nominal initial generalized states; they are not independent randomized physical initial conditions.

No smaller-failure/larger-survival reversal was observed on this sampled grid within a matched policy/seed/direction. The experiment deliberately retains every point and includes independent ±0.03 m/s interior audits, but this sparse sampling cannot rule out unsampled recovery islands, timing sensitivity, or nonmonotonic responses elsewhere. Neither the shared categorical outcomes nor the continuous trajectory differences establish that checkpoint 348 is generally better or worse.

## Representative behavior

- **Backward −0.18 m/s:** all four trajectories recover without conservative step evidence. Recovery takes 0.065–0.075 s for 99 and 0.125–0.380 s for 348. These are threshold-window onset times, not settling time constants.
- **Backward −0.24 m/s, seed 42:** both retain sampled foot contacts. Checkpoint 348 recovers in 0.760 s; 99 in 1.540 s. Final mean root displacement is about 3.5 and 3.8 mm, respectively. The largest joint effort peaks are about 0.313 Nm for 348 (right ankle) and 0.134 Nm for 99 (right knee). This condition reverses the recovery-time ordering at −0.18; effort does not measure stiffness or prove why recovery occurred.
- **Return versus recovery:** 99's surviving −0.24 trial retains about 3.43° mean trunk-orientation change, including yaw, whereas 348 retains about 2.12°. The former misses the new 3° return tolerance despite passing the unchanged 00B recovery rule. This is a threshold-dependent distinction between nearby final postures, not proof of a different attractor or a superior controller.
- **Backward −0.24 m/s, seed 43:** 348 crosses 12° tilt at 0.485 s, first loses foot contact at 0.600 s, loses 4 cm of root height at 0.670 s, and terminates at 0.775 s after injection. For 99 those times are 0.665, 0.775, 0.845, and 0.935 s. Peak angular speeds are about 13.4 and 7.8 rad/s. These are different collapse trajectories, not a causal diagnosis of an actuator or contact failure.
- **Forward +0.18 m/s:** all four trials show tilt growth, contact loss, height loss and termination. Trunk tilt crosses 12° at 0.190–0.375 s; first foot contact loss occurs at 0.485–0.640 s; termination follows at 0.695–0.855 s. The order of termination between checkpoints changes with seed. The step-evidence detector flags foot lift and displaced recontact in these failed trials. These flags describe fall-associated foot motion, not demonstrated successful stepping recovery.

The report overlays each disturbed trajectory with its own undisturbed control. `divergence.json` also compares policy responses after subtracting each policy's control, using only their common recorded time horizon. This separates baseline posture differences from disturbance responses. Joint target/actual and effort plots include both seeds for amplitudes near the observed transition. Final root/COM/posture metrics are means over the last 0.5 s, even for failed trials; they are not impact endpoints or stable final states when recovery failed.

![Backward seed-42 survival at −0.24 m/s](trajectory-dv-0.24.png)

![Forward seed-42 failures at +0.18 m/s](trajectory-dv+0.18.png)

## Validation incident and practical limits

One completed +0.06 m/s record failed exact pre-event matching and was archived under `../excluded/`; its identical-setting repeat passed. One in-progress trial was interrupted during that investigation and preserved. No acceptance tolerance was relaxed, and no record was excluded for falling. The cause of the small pre-event discrepancy remains unresolved. Strict validation now runs after every new trial. See `../excluded/README.md` for component errors and provenance. The reported completed-subprocess cost includes the excluded completed attempt and repeat; the interrupted attempt was not timed, so that sum understates total wall cost.

Early termination exposed and prompted a regression-tested validator fix: integer indices now compare a short disturbed record with the corresponding pre-event samples of a longer control. All historical 00A/00B result files are preserved. The new return criterion was refined during 00C to distinguish displacement, joint posture, and full trunk orientation; it is exploratory, not preregistered or a revision of historical recovery claims.

The experiment establishes finite-horizon simulated responses to these signed root-velocity interventions under the recorded conditions. It does not establish hardware push tolerance, an impact-force limit, a continuous safe region, population failure probabilities, actuator saturation, a support polygon, or a causal failure mechanism. Finer probes in the transition spans, more seeds and disturbance timings, and ordinary-stepping sensitivity checks near failure would be the most useful next experiments; none were launched automatically.


## Tested outcomes

| Δvx m/s | 99 survival / recovery / return | 348 survival / recovery / return |
|---:|---:|---:|
| -0.24 | 1 / 1 / 0 (n=2) | 1 / 1 / 1 (n=2) |
| -0.18 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| -0.12 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| -0.06 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| -0.03 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| +0.00 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| +0.03 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| +0.06 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| +0.12 | 2 / 2 / 2 (n=2) | 2 / 2 / 2 (n=2) |
| +0.18 | 0 / 0 / 0 (n=2) | 0 / 0 / 0 (n=2) |
| +0.24 | 0 / 0 / 0 (n=2) | 0 / 0 / 0 (n=2) |

## Region and uncertainty

Every row is a sampled condition, not a continuous safe interval. The shared adaptive rule doubles amplitude when all four cells survive, otherwise probes midway below the smallest observed failure; a separate interior audit samples below the smallest tested magnitude. Failure information from either checkpoint determines shared probes. No monotone model or binary search guarantee is used. Maximum amplitude is 0.48 m/s and maximum scheduled trial count is 44; no expansion beyond this budget is automatic.

- 99 backward: largest sampled survival 0.24 m/s; smallest sampled failure 0.24 m/s; larger-survival/smaller-failure pairs: 0. These extrema can overlap and are not capability thresholds.
- 99 forward: largest sampled survival 0.12 m/s; smallest sampled failure 0.18 m/s; larger-survival/smaller-failure pairs: 0. These extrema can overlap and are not capability thresholds.
- 348 backward: largest sampled survival 0.24 m/s; smallest sampled failure 0.24 m/s; larger-survival/smaller-failure pairs: 0. These extrema can overlap and are not capability thresholds.
- 348 forward: largest sampled survival 0.12 m/s; smallest sampled failure 0.18 m/s; larger-survival/smaller-failure pairs: 0. These extrema can overlap and are not capability thresholds.

## Measurement decisions and limits

The unchanged 00B settled/recovery rules are reused. Survival requires tilt <45°, height loss <4 cm and no done signal throughout the observation. Its complement is called a failure envelope, not verified physical impact. Recovery requires a continuous 0.5 s stable window and stable final 0.5 s; it permits displacement. Return additionally requires final mean horizontal root displacement <5 mm and every final mean joint offset <0.1 rad, mean trunk orientation within 3° (including yaw), and mean root height within 2 cm of the pre-event 0.5 s. These new exploratory criteria do not revise historical 00B findings. A surviving unresolved trial may recover after the horizon.

Whole-robot COM is the mass-weighted sum of body inertial centers, with mass-weighted body COM linear velocity; root position is reported separately. Ankle positions are foot proxies. Contact loss plus >5 mm lift and displaced recontact supplies conservative step evidence, not a validated gait label. Foot translation can be rocking or sliding. Effort is qfrc_actuator; joint excursions and peak joint forces are in summary.json. No electrical consumption, actuator saturation, support margin or hardware capability is inferred. This is a generalized-velocity state intervention, not a measured force pulse at a trunk point. Work and whole-post-event summaries have different horizons after termination; use common-time traces for policy comparisons.

Seeds change inherited observation delays but share the same nominal physical initial state. Two seeds cannot support population confidence intervals. Timing, terrain, orientation, physical initial-state variation and longer-horizon recovery remain unresolved. Feet-only collision geometry and endpoint forward solves retain 00A/00B limitations; near-boundary solver sensitivity is not bounded.

Validation: unchanged 00B validator checks commands identically zero, signed velocity injection with other velocities unchanged, timing within one physics step, exact pre-event equality to controls, first-step velocity/displacement direction, policy-pair configurations/versions/initial-state equality, and source/config hashes. Additional checks verify COM shape, plausible root distance and total mass. See validation.json; unit tests exercise recovery distinctions and nonmonotone scheduling.

## Individual trajectories

| Trial | Outcome | Peak tilt ° | Root shift mm | Peak angular speed rad/s | Foot unload fraction L/R | Step evidence L/R |
|---|---|---:|---:|---:|---|---|
| 348-s42-dv+0.00 | returned standing | 5.93 | 0.04 | 0.01 | [0.0, 0.0] | [False, False] |
| 348-s42-dv+0.03 | returned standing | 6.13 | 0.14 | 0.11 | [0.0, 0.0] | [False, False] |
| 348-s42-dv+0.06 | returned standing | 6.57 | 0.39 | 0.25 | [0.0, 0.0] | [False, False] |
| 348-s42-dv+0.12 | returned standing | 8.57 | 1.26 | 0.55 | [0.0, 0.0] | [False, False] |
| 348-s42-dv+0.18 | failure envelope / termination | 71.15 | 75.52 | 7.14 | [0.1366906474820144, 0.1510791366906475] | [True, True] |
| 348-s42-dv+0.24 | failure envelope / termination | 73.09 | 74.68 | 6.81 | [0.17391304347826086, 0.19130434782608696] | [True, True] |
| 348-s42-dv-0.03 | returned standing | 5.92 | 0.26 | 0.12 | [0.0, 0.0] | [False, False] |
| 348-s42-dv-0.06 | returned standing | 5.96 | 0.51 | 0.25 | [0.0, 0.0] | [False, False] |
| 348-s42-dv-0.12 | returned standing | 6.53 | 0.83 | 0.52 | [0.0, 0.0] | [False, False] |
| 348-s42-dv-0.18 | returned standing | 7.64 | 1.59 | 0.81 | [0.0, 0.0] | [False, False] |
| 348-s42-dv-0.24 | returned standing | 8.23 | 3.52 | 1.27 | [0.0, 0.0] | [False, False] |
| 348-s43-dv+0.00 | returned standing | 3.46 | 0.74 | 0.01 | [0.0, 0.0] | [False, False] |
| 348-s43-dv+0.03 | returned standing | 4.16 | 1.17 | 0.14 | [0.0, 0.0] | [False, False] |
| 348-s43-dv+0.06 | returned standing | 4.81 | 1.24 | 0.29 | [0.0, 0.0] | [False, False] |
| 348-s43-dv+0.12 | returned standing | 6.75 | 2.03 | 0.60 | [0.0, 0.0] | [False, False] |
| 348-s43-dv+0.18 | failure envelope / termination | 75.83 | 85.95 | 9.12 | [0.11695906432748537, 0.13450292397660818] | [True, True] |
| 348-s43-dv+0.24 | failure envelope / termination | 77.22 | 86.41 | 8.31 | [0.16793893129770993, 0.1984732824427481] | [True, True] |
| 348-s43-dv-0.03 | returned standing | 3.42 | 0.33 | 0.11 | [0.0, 0.0] | [False, False] |
| 348-s43-dv-0.06 | returned standing | 3.64 | 0.57 | 0.23 | [0.0, 0.0] | [False, False] |
| 348-s43-dv-0.12 | returned standing | 3.97 | 1.00 | 0.48 | [0.0, 0.0] | [False, False] |
| 348-s43-dv-0.18 | returned standing | 4.97 | 2.31 | 0.78 | [0.0, 0.0] | [False, False] |
| 348-s43-dv-0.24 | failure envelope / termination | 75.75 | 71.57 | 13.41 | [0.15483870967741936, 0.12258064516129032] | [False, False] |
| 99-s42-dv+0.00 | returned standing | 5.91 | 0.27 | 0.02 | [0.0, 0.0] | [False, False] |
| 99-s42-dv+0.03 | returned standing | 5.98 | 0.50 | 0.13 | [0.0, 0.0] | [False, False] |
| 99-s42-dv+0.06 | returned standing | 6.40 | 1.05 | 0.28 | [0.0, 0.0] | [False, False] |
| 99-s42-dv+0.12 | returned standing | 8.21 | 1.70 | 0.57 | [0.0, 0.0] | [False, False] |
| 99-s42-dv+0.18 | failure envelope / termination | 76.60 | 71.37 | 7.87 | [0.09523809523809523, 0.25170068027210885] | [True, True] |
| 99-s42-dv+0.24 | failure envelope / termination | 77.68 | 69.98 | 8.34 | [0.12173913043478261, 0.3217391304347826] | [True, True] |
| 99-s42-dv-0.03 | returned standing | 6.00 | 0.19 | 0.10 | [0.0, 0.0] | [False, False] |
| 99-s42-dv-0.06 | returned standing | 6.23 | 0.30 | 0.20 | [0.0, 0.0] | [False, False] |
| 99-s42-dv-0.12 | returned standing | 6.57 | 0.25 | 0.46 | [0.0, 0.0] | [False, False] |
| 99-s42-dv-0.18 | returned standing | 6.53 | 0.48 | 0.74 | [0.0, 0.0] | [False, False] |
| 99-s42-dv-0.24 | recovered displaced/posture changed | 9.67 | 3.79 | 1.03 | [0.0, 0.0] | [False, False] |
| 99-s43-dv+0.00 | returned standing | 5.02 | 0.06 | 0.00 | [0.0, 0.0] | [False, False] |
| 99-s43-dv+0.03 | returned standing | 5.22 | 0.23 | 0.12 | [0.0, 0.0] | [False, False] |
| 99-s43-dv+0.06 | returned standing | 5.63 | 0.74 | 0.26 | [0.0, 0.0] | [False, False] |
| 99-s43-dv+0.12 | returned standing | 6.97 | 1.18 | 0.57 | [0.0, 0.0] | [False, False] |
| 99-s43-dv+0.18 | failure envelope / termination | 70.06 | 68.10 | 7.33 | [0.0736196319018405, 0.22085889570552147] | [True, True] |
| 99-s43-dv+0.24 | failure envelope / termination | 72.93 | 69.21 | 7.34 | [0.1092436974789916, 0.31092436974789917] | [True, True] |
| 99-s43-dv-0.03 | returned standing | 4.93 | 0.35 | 0.11 | [0.0, 0.0] | [False, False] |
| 99-s43-dv-0.06 | returned standing | 4.93 | 1.03 | 0.22 | [0.0, 0.0] | [False, False] |
| 99-s43-dv-0.12 | returned standing | 4.93 | 1.58 | 0.48 | [0.0, 0.0] | [False, False] |
| 99-s43-dv-0.18 | returned standing | 5.88 | 1.82 | 0.77 | [0.0, 0.0] | [False, False] |
| 99-s43-dv-0.24 | failure envelope / termination | 70.22 | 70.92 | 7.78 | [0.0481283422459893, 0.06417112299465241] | [True, False] |
