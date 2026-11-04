# 02C attempts and decisions

## Preflight

Initial sandbox `nvidia-smi` and torch CUDA checks reported no access. A check
outside the sandbox found a working RTX 2070 SUPER; CUDA reduction succeeded.
The 64-world upstream benchmark completed with finite states. This distinction
matters: the original error was sandbox visibility, not absent hardware.

Native model loads and verifies the exact published 02B MJB hash. Upstream
revision and working tree were inspected; no upstream source or dependency was
changed. Two low-priority workers were selected for the small episodic search.

## Pilot 1: four knots / 0.8 s, seed 0

1,920 physical training trials, 20 generations, 166.50 s. Training successes
appeared by generation 5, but the final random batch had zero successes.
The score varied with the two sampled push magnitudes; harder batches often
contained no successful candidate and rewarded merely delaying the fall.

Seven development probes each of generation 8 and the final winner were run
for six seconds, later onset 2.5 s, seeds 2,500,000,000+ / 2,500,000,100+.
Both checkpoints recover the 0.28 and 0.30 m/s examples, and fail 0.32–0.40.
The final 0.28 / 0.30 responses move foot origins roughly 3–7 mm, but sole
clearance stays below 1.5 mm. **No verified recovery step.** Forced execution
with zero push falls backward; this is a push-specific action, not a standing
improvement. Post-fall foot lifts in failed trials are explicitly rejected.

The original source versions are archived in `pilot/source-v1` because training
conditions were subsequently revised; their hashes match the original config.
Raw development records are retained locally and hashed by publication.

## Revision before confirmation

Shorten the free residual to 0.4 s / two knots (28 parameters, both legs and
head still independently controllable). The original response spends much of
the approximately 0.6 s fall horizon traversing four waypoints; it need not
have that many opportunities to make a useful correction. This changes timing
and dimensionality together, so a better result cannot isolate either factor.

Stratify four training magnitudes over 0.28–0.36 m/s in every generation. This
retains easier and harder evidence in each update instead of allowing two
random hard pushes to replace all evidence of successful recovery. Pilot seed
0, 20 generations / 3,840 trials maximum, 420 s launch cap. This is a new
representation/experience test, not continuation of the first optimizer.

The revised pilot completed 3,840 training trials in 356.62 s. By generation 20,
83/192 candidate-condition trials recovered, compared with 0/96 in the original
pilot's final batch. This is **not a fair performance comparison**: magnitudes,
batch size and representation changed. Development generation-18 probes still
recover 0.28/0.30 and fail 0.32–0.40, without a recovery step. Stratification
improves the optimizer's evidence stream; it has not demonstrated a new physical
boundary. The final winner is also probed and retained; neither probe selects
the main checkpoints.

## Confirmation protocol freeze

Proceed with the simpler two-knot / 0.4 s representation. Six independently
initialized runs: CEM and matched random search, optimizer seeds 1/2/3. Each
gets 20 generations × 48 candidates × 4 stratified conditions = **3,840 physical
trials**, 23,040 total. This is the revised pilot's budget, not an extension.
The last paired training winner is evaluated, even if it is worse than an
earlier checkpoint. Maximum two low-priority workers across the whole driver;
600 s per-run generation-launch cap, with exact resumability. Held-out data
have not been consulted at this point.

## Contact attribution diagnostic (development data)

For the revised pilot's paired +0.30 m/s example, zero residual falls at 3.26 s;
the learned residual quietly recovers. Each sole has exactly the same material
contact-point travel in the first 40 ms: 1.74 / 1.69 mm. The option cannot yet
affect motion because its first knot starts at zero and the actuator has 20 ms
delay. **Initial displacement is passive push-induced slip.** The successful
response then has another 5.56 / 5.69 mm of integrated tangential contact-point
travel over the rest of the first second, with uninterrupted sole contact.
This includes compliant contact and is path length, not net travel.

`contact02c` computes `J_point(q) @ qvel` at geometric sole-floor contacts,
following [MuJoCo's point Jacobian definition](https://mujoco.readthedocs.io/en/3.10.0/APIreference/APIfunctions.html#mj-jac).
It avoids calling every movement of the elevated ankle frame a slide. The
diagnostic is descriptive: it neither proves that slip is the cause of recovery
nor licenses a stepping claim. The matched residual-removal comparison shows
that the controller changes the recovery outcome; both passive and active
effects contribute to the observed displacement.

Evaluation uses the actual, 20 ms-quantized disturbance onset for the independent
physics-rate precondition and support audit. The requested onset is retained in
the condition. Training and 02B physics are unchanged by this reporting detail.

The development filmstrip shows a brief whole-body bend rather than a swing
foot. Before any confirmation evaluation, add head-only and legs-only residual
ablations for every learned seed. They retain the entire frozen standing reflex
and remove only the indicated part of the learned option. This tests whether
upper-body motion, which 02B did not need for standing, contributes to recovery.

Over the first 0.5 s of that same development example, peak commanded motor
torque is 0.234 Nm with the option removed and 0.297 Nm with it active, against
the nominal 0.957 Nm actuator force limit. Absolute joint mechanical work is
0.090 versus 0.453 J (not electrical energy). The measured joints remain at
least 0.190 rad inside the command-margin limits. These observations do not
prove that another recovery is feasible, but do rule out simple torque clipping
in this specific successful maneuver. `actuator-diagnostics.json` preserves the
numbers; its `min_joint_command_margin_rad` field is measured-position distance
to the command-safe limits, not a target-command measurement.

## Capacity control: a changed interpretation

After the 24-case GPU cross-check confirmed the small gain, a review of the
action-bound change exposed a missing control. Five new development conditions
(seeds 2,500,000,400+) run the **old 02B weights unchanged**, but widen the
reflex's own clip from 0.35 to 0.8 rad. It recovers the 0.28 and **0.30 m/s**
examples and fails 0.32–0.36. No option, timing cue, retraining or new parameters
are involved. `pilot/wide-reflex.json` preserves those results.

This exposes a potential capacity confound, but the new conditions use different
initialization/noise seeds from the earlier probe, so they cannot establish
that wider clipping caused a gain. Before the main held-out grid, add this
expanded-reflex control and clip each learned option back to 0.35 rad. The main
searches, their budgets and checkpoint selection remain frozen. The initial GPU
cross-check covers the originally declared frozen reflex and learned options
only; its 24 outcomes are unchanged.

**The paired check corrects that first impression.** On the earlier failing
condition (seed 2,500,000,302), both original and widened reflexes fail. On the
new successful condition (2,500,000,401), both recover. Thus the two examples
show seed sensitivity, not a benefit from widening the old reflex. A separate
six-case GPU check agrees exactly with native outcomes: the widened reflex
still fails noise-free 0.30 m/s, whereas all three learned options recover it.
The broader held-out capacity ablation is retained; this is exactly why equal
push magnitude alone is not an adequate matched control.

## Continuity with the published 02B evidence

The full held-out grid includes seeds at which the old reflex already recovers
0.30 m/s. To distinguish new sampling from a changed simulator, replay ten
published 02B forward-push conditions (0.20–0.40 m/s, both old replicates) using
02B's unchanged rollout function. All ten survival outcomes agree with the
published Windows results; most fitness values agree to about 1e-12, with a
maximum difference of 0.00140 in one falling trial. This is outcome continuity,
not a claim of cross-platform bit identity. The old 0.30 m/s pair itself has
one recovery and one failure. `baseline-portability.json` preserves the check.

The Linux and Windows bundled BAM XL330 M6 JSON files have different byte hashes
but identical parsed parameter values. The Linux raw and canonical hashes were
added to `preflight.json`. All 123 files in the prior 02B publication manifest
also verify. These checks reinforce using matched current trials instead of
treating the approximate 0.28 m/s capture estimate as an exact failure threshold.

## Confirmation outcome and stopping decision

All six runs completed their declared 3,840-trial budgets. All 2,190 held-out
evaluations completed; no checkpoint or training seed was replaced after seeing
these results. Learned seeds 1/2/3 recover 36/21/35 main-grid cases, versus
30/13/19 for matched random search and 29 for the frozen reflex. At 0.32 m/s,
the learned controllers recover 16/18 controller-condition pairs, versus zero
for either reflex control and zero for the three random-search winners.
Reversal, 80 ms delay, and head/leg removal largely or completely remove this
gain. The full [report](REPORT.md) preserves all denominators and regressions.

No verified recovery step appears. In successful main-grid pushed CEM trials the minimum
sole point never clears the floor. The fixed response overcorrects small pushes
in seed 2; forcing it without a push causes 15/18 falls. The final conclusion
is a learned transient coordination benefit with continuous support, not
emergent locomotion. No extra generations were launched to chase a favorable
step. The next useful changes concern state-conditioned response and access to
valid contact transitions, rather than a larger budget for this same option.
