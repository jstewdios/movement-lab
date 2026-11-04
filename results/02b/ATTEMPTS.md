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

## After review (Codex review of PR #7)

Three findings, all fixed. Simulated trials, outcomes and every quoted number are
unchanged.

- **Recorded sole forces ignored the actuators.** Recording copied only
  positions and velocities into the measurement `MjData`, so its contact solve
  held up an unactuated robot and never saw sustained pulses. The recorded total
  sole force in quiet stance was 7.00 N against a 7.23 N weight; a 1 N downward
  trunk pulse showed as +0.10 N. Recording now copies the full state
  (`mj_copyData`, so ctrl, `xfrc_applied` and warm start are included): 7.23 N, and
  +1.01 N for the pulse. Every other recorded field (states, contact flags,
  COM, tilt) and every summary is bit-identical before and after, checked on the
  step-event trial and a pulse trial. Only the sole-force panels of
  `response.png` and `step_event.png` changed. No table, statistic or report
  statement used these forces (step events come from contact presence in the
  simulated state). A regression test checks weight support and pulse loading.
- **Model regeneration on a fresh checkout.** `model02b --out results/02b/model`
  refused to run because the directory holds the published metadata. It now
  compiles into a temporary directory and installs `model.mjb` only if it is the
  published model (hash and recorded properties). It never overwrites
  `model.json` or `resolved_config.txt`.
- **Manifest hashes depended on checkout line endings.** Windows text mode wrote
  CRLF, and `core.autocrlf` kept CRLF in this working tree, so the manifest held
  CRLF hashes while Git stores LF. A Linux checkout could not verify them.
  `.gitattributes` now pins LF for the 02B files, the 02B writers emit LF, and the
  manifest refuses CRLF in published text. Verification also checks the source
  hashes and names CRLF checkouts explicitly. The three published `.jsonl.gz`
  tables held CRLF rows inside; they were rebuilt with LF rows (identical
  records, new bytes).
- **Found while checking the regeneration:** two loops iterated over Python sets,
  whose order follows per-process hash randomization. One reordered keys in
  `tables.json`, the other in future `summary.json` files. Both are now sorted.
  Regenerated outputs are byte-identical across processes, and `tables.json`
  holds the same values as before.
- **Found by the fresh-checkout model check:** the published
  `resolved_config.txt` was stale. It was written by the first compile, before
  `env_cfg` gained the cross-check settings: observation corruption off, the
  fixed 1 mm-clearance reset, and terminations limited to `nan_state` and
  `time_out`. It therefore showed upstream's randomized reset and extra
  terminations, which no 02B run used. The compiled model is unaffected (the
  same hash; these are runtime settings), and the native study never reads
  them. The cross-check did run with the current settings: its initial states
  agree to 1.2e-8, which a randomized reset could not produce. The file now
  holds the current resolution, and a test keeps it equal to `env_cfg()`.
