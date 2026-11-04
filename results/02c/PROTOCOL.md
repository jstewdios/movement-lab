# Study 02C — protocol and decision record

Design record written during development, before the main training runs.
Pilot seed 0 is development data. Later evaluation amendments and their timing
are explicit in ATTEMPTS.md; this is an exploratory investigation, not a
registered statistical trial. No held-out result selects a training checkpoint.

## Question and alternatives

Can balance experience select a useful change of support? Study 02B's frozen
`ars-push-s1` reflex supplies standing. Its own-stance forward capture estimate
is approximately 0.28 m/s. This is a pendulum approximation, not an impossibility
theorem: angular momentum, compliance and contact motion can violate it.

Three approaches were considered:

1. Recurrent neural policy with PPO: most general, but jointly searches sensing,
   memory and motor coordination and would make a bounded negative result hard
   to diagnose.
2. Event-cued finite joint residual, optimized across complete episodes: chosen
   first because it isolates temporal motor coordination. The event cue is an
   oracle. **Only the motion is learned; disturbance detection is not.**
3. Expanded memoryless reflex: useful control, but 02B already tried both harder
   experience and a proprioceptive residual without changing its boundary.

No prescribed footfall, inverse kinematics, alternating legs, target distance,
foot-lift bonus or imitation trajectory. Four free 14-joint knots, smooth
interpolation from and back to zero in 0.8 s, added to the frozen reflex.
Residuals and combined offsets are bounded at 0.8 rad, with the existing joint
limit margin. Widening 02B's 0.35 rad envelope is explicit; zero residual retains
the exact old reflex. Motor physics/limits are unchanged.

## Compute and model

The local `main` is `bafc45e1f03492d6cfa0d56273a158318f8f9e9e` (merged PR #7),
also verified against GitHub `main`. Upstream is read-only at
`b02fdea4902c2b93b7d34a83a6f05da4ececb7d3`, clean before work.

The WSL sandbox hides CUDA. Outside it, a CUDA reduction and 64-world MuJoCo-Warp
benchmark both succeed on an RTX 2070 SUPER. `preflight.json` records versions
and times. Native MuJoCo completes eight early-terminated 4 s trials in 1.22 s
on one thread; GPU completes 64 × 2 simulated seconds in 4.12 s (plus 4.84 s
construction). These workloads differ; this is feasibility, not a rigorous
speed contest. The small episodic search uses **two low-priority CPU workers**,
one BLAS thread each, with early exit, preserving native 02B physics exactly.
This does not saturate the six-core host. GPU is reserved for a fidelity check;
there is no need to install or modify upstream dependencies.

Every generation is an atomic resumable checkpoint with candidates, scores,
conditions, distribution, winner and RNG. Wall budgets stop launches at generation
boundaries; an in-flight generation may finish. Full-model files remain local;
weights, numerical trajectories and compressed result tables will be published.

## Pilot and planned revision rule

Initial pilot: seed 0, 20 generations, 48 candidates × 2 conditions, 1,920
episodes maximum; 360 s launch cap. Forward increments 0.29–0.36 m/s, random
onset 1.2–1.8 s, 4 s episodes, initial joints ±0.01 rad, the same IMU noise and
20 ms actuator delay as 02B.

Inspect actual trajectories and failure causes before a main experiment. A
failure should change the representation or experience on the basis of those
measurements, not merely extend training. Candidate revisions include a shorter
response, reduced search dimension, or a curriculum from recoverable pushes.
Any revision is recorded in ATTEMPTS.md before new held-out tests.

## Training objective

Alive fraction minus 0.25 × normalized tilt-squared integral, plus 1 for a full
final second of quiet recovery, minus 0.002 × mean squared command increments.
The outcome bonus requires quiet pre-push standing too. Quiet means tilt <5°,
angular speed <0.5 rad/s and horizontal root speed <0.05 m/s. There is **no
distance or support-change reward**. Falls, nonfoot contact, >3 mm penetration,
nonfinite states and >4 cm root height loss latch failure. Training samples
these gates at 50 Hz. Independent recorded evaluation checks at 200 Hz.

CEM uses the best 8/48 candidates to update a diagonal Gaussian (75% new,
25% old moments), initial std 0.25 rad, floor 0.04 rad. Incumbent and distribution
mean occupy two candidate slots and are rescored on each new paired batch.
The final generation's winner is the selected checkpoint. A nonadaptive random
control uses the same budget and initial proposal distribution, keeping a
rescored incumbent but never moving its sampling distribution.

## Confirmation and attribution

Main runs will use independent optimizer seeds 1–3 and matched random search.
Training seeds start at 2,000,000,000; development probes at 2,500,000,000;
held-out seeds at 3,000,000,000. Evaluation is six seconds, with later push onset,
unseen magnitudes and paired initial states/noise. Each final checkpoint is
evaluated, with no held-out checkpoint selection.

Controls: frozen 02B reflex / residual removal, matched random search, reversed
knot order, replacing knots by their time-average, no push, forced response
without a push, and delayed / IMU-triggered response. Causal comparisons use
identical pre-event states. A learned recovery beyond a matched baseline's
failure is evidence of control benefit; beyond a capture approximation alone
is insufficient.

A support relocation requires no foot contact for ≥40 ms, minimum sole clearance
reaching >2 mm, horizontal recontact displacement >5 mm, and ≥100 ms sustained
recontact with mean normal force >0.5 N. It counts as a recovery step only in
an otherwise valid, upright, quietly recovered episode. Slides, falls, short
contact flicker and post-fall replants remain visible but cannot pass this gate.
These thresholds are descriptive; inspect continuous trajectories and replay.
The independent evaluation also requires load-bearing foot contact and total
3-D root speed <0.05 m/s throughout the final second. It cannot count a quiet
looking flight phase or vertical bounce as a recovered stance. This stricter
endpoint guard was added during evaluation-code review, before the full grid;
training scores and checkpoints are unchanged.
Recorded evaluation also rejects measured joint-limit overshoot above 0.03 rad,
the earlier studies' numerical gate; commanded positions always retain the
0.02 rad margin. These are simulation validation thresholds, not hardware ratings.

## Background

- [Mannor, Rubinstein and Gat, The Cross Entropy Method for Fast Policy Search (2003)](https://cdn.aaai.org/ICML/2003/ICML03-068.pdf): episodic policy search by adapting a sampling distribution.
- [Pratt et al., Capture Point: A Step toward Humanoid Push Recovery (2006)](https://www.cs.cmu.edu/~cga/legs/Pratt_Goswami_Humanoids2006.pdf): physical motivation for changing support and the role of angular momentum.
- [02B report](../02b/REPORT.md) and its implementation supply the actual local baseline and model evidence.
