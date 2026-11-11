# 02D protocol, declared before the pilot

Base: merged 02C, `e2c28b8`. Historical files and outcome definitions stay fixed.
The 02B reflex, model, BAM parameters and 02C seed-1 two-waypoint option are frozen.
Seed 1 is chosen from the already published 02C result (36/60, best established
overall recovery), before any 02D result. Seed 2 remains an overcorrection check.

## Representation and information

Learn two signed, tanh-bounded coefficients of the existing 14-joint waypoints.
Recompute them at 50 Hz from five normalized noisy IMU channels (gravity x/y,
gyro xyz), their exponential history (80 ms), and bias: 22 parameters. This is
a small stateful controller with a fixed memory time constant, not a learned
gait or a replayed option. It can reverse, mix or stop its inherited coordination
patterns at any update. Motor vocabulary is restricted; no claim about the
capacity of unrestricted joint policies follows. Memoryless control: same
basis, nonlinear output, optimizer and trial budget, 12 parameters. Both start
at zero residual. No onset, elapsed trial time, push magnitude, world velocity,
position or solver contact enters either controller. IMU projected gravity is
the same ideal orientation-estimator approximation used in 02B, not a tested
hardware estimator. Contact is measured for evaluation only.

Physics: 200 Hz; observations/actions: 50 Hz; nominal command delay: 20 ms.
No extra artificial sensor latency nominally. Legacy gyro-lag and additional
whole-IMU delay are explicit evaluation interventions. The residual and combined
offset retain 02C's ±0.8 rad clamp, joint margin 0.02 rad and BAM torque envelope.

## Budget and decision rule

Phase A: restore exact model; reproduce the published 02C grid for frozen reflex,
seed 1, seed 2, fixed IMU trigger and 80 ms delayed seed 1 at quiet, .20, .26 and
.32 m/s (six cases each). Check deterministic replay, observation/actuation
timing, passive recording, and zero-residual equivalence. One pilot, seed 71:
8 generations × 24 candidates × 8 conditions = 1,536 episodes, at most two
low-priority workers and a 600 s launch budget per invocation. A generation
already launched may finish. Preserve all attempts; resume only exact sources.

Pilot development validation: disjoint seed range, 24 six-second conditions
(quiet, .08, .20, .28, .32, .36 forward impulses, four each). Proceed if the final
pilot winner gains at least two paired recoveries over zero residual, loses no
quiet cases, and has a higher mean training score than zero evaluated on the
last batch. This small screen permits a confirmation experiment; it is not
confirmation itself. If it fails, stop or make at most one evidence-based
revision, declaring it before another pilot. Do not automatically extend search.

Phase B if gate passes: history and memoryless arms, seeds 1/2/3, each 12 × 24 ×
8 = 2,304 episodes (13,824 total), 600 s launch cap per run invocation. Every
generation stores all candidates, outcomes, conditions, optimizer and RNG state
atomically. Use the last training winner for each run; never choose a seed or
checkpoint from the final held-out results. Independent seeds, paired condition
seeds across arms. No massive training campaign or GPU training is planned.

## Experience and objective

Each training batch stratifies quiet standing, small forward disturbance,
moderate forward disturbance, two near-boundary forward disturbances,
a force pulse of varied duration, a lateral or backward disturbance, and a
false alarm (brief injected IMU rotation-rate transient without physical push).
Onsets vary from 1.2–1.8 s; episodes last 4 s, resets use ±.01 rad joint noise.
The physical pulses apply force at trunk COM; velocity impulses retain 02C
semantics. Pulse strength is calibrated in the pilot and saved in conditions.
No movement target, foot lift, displacement, contact switch or stepping reward.
Fitness retains survival/tilt and final-second quiet recovery, with a small
residual-magnitude and change penalty. Quiet-standing motion is measured too.

## Confirmation, auditing and visibility

New held-out seeds, later onsets, 6 s horizon. Compare frozen 02B, 02C seed-1
oracle, its fixed IMU trigger, all three memoryless and all three history runs.
Include quiet, forward magnitudes, backward/lateral impulses, force durations,
false alarms, and repeated pushes. Limited paired shifts: friction .6/1.4,
6.5 V, 30 ms actuation delay, 40/80 ms observation delay, +2° pitch calibration.
Report all wins AND regressions. Paired conditions are shared observations,
not independent samples when pooled over controllers or shifts.

Use unchanged 02C 200 Hz audit plus supported final-second and joint-limit
checks. Check torque/target bounds and self-collisions. Never erase a prior fall.
Use unchanged ≥40 ms flight, >2 mm minimum-sole clearance, >5 mm relocation,
≥100 ms loaded replant AND whole-trial recovery for a recovery step. Separate
contact-point sliding from ankle-frame displacement. No locomotion claim from
passive movement. No hardware claim.

Replay: deterministic first-replicate examples, synchronized measured states,
quiet/small/large/failure/false-alarm and first versus final training checkpoint.
Publish representative full-rate records, compressed per-trial tables, progress,
provenance, tests, compact report, and a portable browser replay in a PR.

## Diagnostic plan added while the first pilot validation was running

The main grid is 68 conditions per controller (17 cells × four replicates);
the shift battery is 48 (eight settings × three magnitudes × two replicates).
Both contain all three final history and memoryless seeds and the three legacy
controls, totaling 1,044 confirmation episodes. Replays always show the first
replicate of named cells, with seed 1 and generation 1 fixed in advance.

After training, run a separate causal diagnostic on forward .08/.20/.32/.36:
hold the learned residual at its last pre-event value for the rest of the trial,
leaving the frozen reflex active. This deliberately uses the oracle event time
only to REMOVE feedback, not to improve the proposed controller. It preserves
the learned pre-event stance and tests whether recovery comes from that stance
or from changing the residual after the push. Also replace the history input
with the current observation, without retraining. These are diagnostic controls,
never candidates for selection. Limit to two new seeds per magnitude, three
history controllers and their intact counterparts (72 episodes).

## One evidence-based revision, declared after the first pilot failed

Pilot 71 recovered 21/24 development conditions versus the reflex's 16/24,
with six paired gains and one loss. That loss was an unforced fall at 2.52 s;
the no-quiet-regression gate therefore failed. No Phase B run was launched.
The last batch winner also lost a small-disturbance case, consistent with a
mean-score tradeoff against easy cases. All eight original checkpoints remain.

Revision: retain architecture, optimizer, action/validity rules and objective;
increase explicit quiet coverage from one to four reset/noise seeds per batch,
plus one benign .01–.03 m/s disturbance. These five low-disturbance strata last
6 s. The other seven strata remain exactly the original curriculum. Reduce
population from 24 to 16 (elite 6 to 4), giving 12 × 16 = the SAME 192 episodes
per generation. This trades candidate breadth for standing-reliability evidence;
it is not a claim that the two changes have separately identified effects.

One revised pilot, seed 72, eight generations = 1,536 episodes, then the same
gate on FRESH development seeds starting at 5,100,000,000. If it fails, stop.
If it passes, both Phase B arms use this revised curriculum: 12 × 16 × 12 =
2,304 episodes per seed, unchanged from the originally declared total. No
checkpoint, seed or final held-out condition will be selected to hide a failure.
The original pilot's numerical sources are archived under `pilot/source-v1`.
During this revision the evaluation resume identity was also normalized through
JSON so tuple-valued conditions compare correctly after reload; this affects
reuse checks only, not any simulation or outcome.

## Display additions after main-grid results began arriving

All three history controllers recovered all four 0.32 m/s cases, but history
seed 1 lost all four backward pushes and seed 3 fell in two quiet trials. Add
the first backward replicate and the first failed seed-3 quiet trial, paired
with the frozen reflex, to the recorded replay. These additions expose
regressions; they change no training, checkpoints, selection rule, conditions,
evaluation counts or success criteria. The original six named first-replicate
comparisons remain intact.
