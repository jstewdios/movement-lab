# 01B landing/contact diagnosis

**The original failure is a shared initial-landing transient, not evidence of a groundcontact-specific policy failure. A separate 1 mm-clearance initialization passes the unchanged baseline gate in two independent runs. No departures ran.**

## Exact failure and historical comparison

The violating pair is `terrain` (geom 0) / `robot/right_foot_collision` (geom 79), on `robot/ankle_right`. CPU forward geometry at saved GPU states gives:

| Event | Time | Signed contact distance | Contact position, world xyz (mm) | Normal, terrain → sole |
|---|---:|---:|---|---|
| First gate violation | 0.050 s | −3.1261 mm | (−12.0680, −42.1039, −1.5631) | (0, 0, 1) |
| Deepest penetration | 0.055 s | −3.1600 mm | (−12.0473, −42.1239, −1.5800) | (0, 0, 1) |

The reported contact point is MuJoCo's contact position, not a measured surface point; its z coordinate lies midway across the overlap. Geometry checks do not reconstruct GPU solver force or impulse. [Full contact records](original/audit.json) include every contact through the recorded 60 ms, normal, position, solref, solimp and friction.

Original root z is 0.125 m; both sole mesh minima start **7.81764 mm above the plane**, with zero generalized velocity. First sampled contact occurs at 40 ms, when root vertical velocity is −0.39620 m/s. Peak overlap occurs as the root arrests its descent: vertical velocity is −0.00407 m/s at 55 ms, then +0.03005 m/s at the 60 ms stop.

Saved 00A checkpoint-99 data have **exactly identical** time, actor commands, joint targets/positions/velocities/efforts, full qpos/qvel, foot occupancy and foot forces through all 12 original 01B samples. Applying the groundcontact geometry audit to the historical 00A trajectory reproduces the same penetration peak; the offending sole geometry itself is identical to walk. Thus prior 00A standing validation did not certify this stricter landing-contact criterion. It did not test that criterion.

## Commands, geometry and solver

The original 01B record ends at 60 ms; it provides no observations from 60–100 ms. The existing 00A record supplies a clearly labeled historical 100 ms comparison, and both new conditions record their own full first 100 ms. [Original](original/first100ms.csv), [historical 00A](historical00a/first100ms.csv) and [revised initialization](clearance-1mm-audit/first100ms.csv) CSVs contain all 14 named joints' raw actor commands, targets, actual positions/velocities and efforts, plus root z/vertical velocity.

Original absolute actor maxima are 0.27976, 0.31905 and 0.31719 rad for the first three 20 ms updates. Actual right-ankle range is 0.06396 rad over the recorded 60 ms, versus 0.07301 rad over historical 00A's first 100 ms. Targets are HOME + actor output, before the retained 20 ms BAM delay; these are powered commands, not zero torque or direct motor-voltage measurements. Peak endpoint joint effort is 0.24704 Nm. No residual command is applied.

Walk and groundcontact sole mesh vertices, local transforms, collision masks, condim, friction, margin/gap and contact solref/solimp match exactly. Groundcontact additionally permits battery/hip/leg/head/jaw floor contact; none occurs in the original trace or either revised run. The prior [model audit](../model-audit.json) documents the remaining collision differences and equal inertia/joint/transmission parameters.

The saved simulation configurations match exactly: 5 ms timestep, decimation 4, implicitfast integration, Newton solver with 10 iterations, tolerance 1e-8, line-search 20 iterations/tolerance 0.01, pyramidal cone, impratio 1, gravity −9.81 m/s², nconmax 35 and njmax 1500. Actual offending contacts have dim 3, solref (0.02, 1), solimp (0.9, 0.95, 0.001, 0.5, 2), and friction (1, 1, 0.005, 0.0001, 0.0001). The original audit finds at most two simultaneous native contacts. There is no observed geometry-capacity pressure; GPU iteration residuals/capacity counters were not recorded, so solver convergence is not independently established.

**Most likely explanation:** an approximately 8 mm unsupported drop with downward momentum produces a short compliant-contact compression at finite timestep. Controller-induced articulation contributes to the landing configuration, but is identical in the matched historical record. The evidence does not implicate the added collision surfaces. No solver-parameter sweep was performed, so the respective contributions of contact compliance, timestep and convergence cannot be isolated or declared a solver bug.

## Separate initialization and bounded evaluation

A geometry-derived reset reduces root z to **0.1181823627 m**, leaving both soles 1 mm above the plane, HOME joints and zero generalized velocity. The actual float32 reset z is 0.1181823611 m. Only initial root z changes. The free root remains unconstrained; no state is teleported or pinned after reset. Checkpoint 99, observation pipeline, BAM, collision model, solver and **all original acceptance criteria** remain unchanged.

Budget: at most two six-second baseline-only GPU runs. After the first passed, the second was used for an independent-process repeat of the same condition rather than another height. Both retain 1,200 physics samples (6.00009 s simulator clock, from float accumulation). Runner wall time totals **81.01 s**, including startup/recording, excluding offline diagnosis; this is not GPU kernel time. No 0.2 mm-clearance condition, training, or departure trials ran.

| Measurement | 1 mm condition | Independent repeat |
|---|---:|---:|
| Original baseline gate | pass | pass |
| Worst penetration, whole run | 1.156 mm | identical |
| Maximum trunk tilt, whole run | 6.079° | identical |
| Maximum COM speed, t ≥ 1.5 s | 2.158 mm/s | identical |
| Maximum root angular speed, t ≥ 1.5 s | 0.03721 rad/s | identical |
| COM x / y range, t ≥ 1.5 s | 0.717 / 0.204 mm | identical |
| Both-foot contact occupancy, t ≥ 1.5 s | 100% | identical |
| Nonfoot-contact samples | 0 | 0 |
| Mean total raw foot Fz, t ≥ 1.5 s | −7.23215 N | identical |

The raw foot-force convention is opposite to upward reaction. Its mean magnitude matches 0.73724318 kg × 9.81 m/s² within approximately 0.003%. Every recorded array matches bit-for-bit across the independent processes. This demonstrates a reproducible finite-horizon, contact-supported standing baseline for this condition and seed, not broad robustness or hardware fidelity. Peak tracking error remains 0.3491 rad during the transient, and peak effort 0.2470 Nm; passing is not perfect tracking.

## Validation and next action

All **27 tests pass**. Original and revised records pass source/checkpoint/manifest checks, finite synchronized arrays, quaternion/state mapping, default+actor+residual composition, gate recomputation and independent CPU saved-state geometry/COM/contact checks at every sample. New tests cover mesh-based world clearance under reset translation and the diagnostic runner's zero-only schedule. [Comparison data](comparison.json) verify only initial root z changed and repeats match. Full-rate NPZ, MJB and logs remain local and hashed; [replay](clearance-1mm/replay.html) replays measured link centers without dynamics. Desktop geometry replay: `python -m movement_lab.replay01a results/01b/landing-diagnosis/clearance-1mm/zero`.

**Recommended next action:** use the named 1 mm-clearance condition as the explicit starting condition for a separately bounded 01B departure slice, with the same settling and contact checks and stabilizing controller retained. Preserve the unsupported-start failure as its own condition. This diagnosis establishes standing after a gentler landing; it demonstrates no controlled articulation, weight transfer, unloading, stepping or recovery maneuver.

Original failed evidence, report and criteria are unchanged. Upstream is unchanged. No commit, push, merge or continuation into departures was performed.
