# 02B development attempts (development data, not evidence)

Everything below happened before the main matched matrix was launched, and it
shaped the protocol. These pilots used training/probe seeds from the same
generators as the main runs (seed 0, which the main matrix does not use) and no
held-out evaluation seeds. Their outputs are kept under `pilot/` (local raw
files are excluded from Git where noted in `.gitignore`).

## Environment decision

- The upstream venv on this Windows machine has `torch 2.9.1+cpu` (PyPI's Windows
  wheel), so mjlab's CUDA path cannot run here even though warp sees the RTX 2070
  Super. Swapping torch inside the upstream environment was rejected (it would
  change the shared upstream install to make the study run).
- mjlab does construct and step on `device='cpu'` (warp CPU kernels). That is used
  (a) once, to compile the exact 01A-02A model (`model02b`), and (b) for the
  fidelity cross-check (`crosscheck02b`). It is far too slow for learning.
- Learning and evaluation use native CPU MuJoCo with `bam.mujoco.MujocoController`,
  upstream's own deployment-rehearsal actuator path, with mjlab's step ordering
  and physics-step command delay reproduced. 11 worker processes give about 60
  simulated seconds per wall second.

## Physics checks before any learning

- The geometry-derived 1 mm-clearance reset reproduces 01B's root height
  (0.1181823622 m vs 0.1181823627 m).
- Fixed HOME targets topple forward from that reset with no disturbance: tilt
  crosses 45 deg at 1.12 s, both soles stay in contact, peak actuator torque is
  only 0.098 Nm (no saturation). This matches 01A's "zero action also falls"
  under a gentler start, and made the study's first question "can it learn to
  stand at all?" rather than "can it improve an existing stance?".

## Pilot 1 (`pilot/ars-push-s0`, 30 ARS iterations)

Standing was found within ~8 iterations. The first probe set (24 random
conditions) was too coarse: its score sat at exactly 0.792 for 20 iterations,
hiding any change. Replaced by the structured probe set used in the main runs
(8 unpushed + 8 directions x 4 magnitudes).

## Pilot 2 (`pilot/v2-*-s0`, three arms, 30 iterations each)

- ARS with pushes: pushed-probe survival 0.56 by iteration 5, flat afterwards.
- Random search with pushes: standing by iteration 10; pushed probes 0.31-0.44,
  with the selected "best" occasionally regressing (single-batch selection noise).
  This motivated the final re-scoring of the top 10 random candidates on 16 fresh
  training conditions (counted in its budget).
- ARS standing-only: pushed-probe survival 0.59 at iteration 5 without ever
  being pushed.

Decisions taken from the pilots: 60 iterations per main run (plateau by ~10),
dense early probing (iterations 0-6, 8, 10, 12, then every 5), three seeds per arm.

## Capture-point benchmark (added after pilot 2, before main results)

Pilot failures were direction-dependent (backward 0.15-0.2 m/s fell, forward
0.2 m/s survived). The soles extend 31 mm ahead of but only 17 mm behind the HOME
COM projection, giving a no-step limit of about 0.26 m/s forward, 0.15 m/s
backward and 0.50 m/s sideways. `physics02b` computes this bound; it is a
geometric inverted-pendulum estimate, not a measured stability margin.

## Mid-run code changes (verified not to affect running experiments)

- `sim02b.rollout` gained step/support bookkeeping, multi-push support and an
  optional `policy` hook while runs were in progress. Before/after bitwise
  comparisons of fitness and final root position on fixed conditions were
  identical; the default code path and random stream are unchanged.

## After the main matrix (analysis-stage notes)

- **Cross-check development run** (`pilot/crosscheck-dev`, pilot reflex): 24/25
  outcome agreement. It exposed a reporting bug (initial-state difference of 4.0
  came from mjlab's per-env grid origins in x/y); the check now compares only
  height, orientation and joints. The final cross-check ran afterwards.
- **Discarded interpretation:** translating reflex gains into a free-floating
  "COM shift relative to the feet" was rejected: with soles planted, commanded
  offsets act as torques through compliant servos, so that kinematic number is
  not a COM displacement. The reported quantity (commanded foot pitch relative
  to the trunk per unit tilt) is an orientation, not a position.
- **Interpretation corrected by analysis:** step-like foot lift-offs first looked
  like survivals beyond the *HOME-stance* no-step limit. Computing the limit of
  each reflex's *own* stance (added after evaluation; exploratory, not
  preregistered) showed every such survival lies inside its own limit and that
  push experience had shifted stances to gain heel margin. The report states the
  corrected reading; the HOME-stance split is retained in `tables.json`.
- **Performance fix:** the first pure-Python 2-D hull took ~1 s per call on the
  sole meshes (4,967 of 7,896 vertices lie on the hull); replaced by SciPy/Qhull.
  Capture-point bounds are identical to the third decimal (unit-tested).
- **Validator fix:** saved JSON conditions hold lists where in-memory conditions
  hold tuples; the validator now compares both through JSON.
- **Pre-existing on this Windows machine (not touched):** four 01C tests hardcode
  `dir='/tmp'` and fail here; 02A's review test needs `PYTHONUTF8=1`. All other
  tests pass or skip on missing local fixtures, as designed.
- Mesh renders (local only, `renders/` ignored) were used to check the motion by
  eye: HOME topples forward over the toes, the reflex stands nearly still after a
  0.2 m/s forward push, and a 0.2 m/s backward push rotates it about the heel edge.
