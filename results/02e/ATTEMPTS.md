# 02E execution notes

## Phase 0: reproducibility before anything new

- **Published evidence verified.** 02B (103 files; 20 local-only files absent and
  reported as unverifiable), 02C and 02D (287 hashes) pass their own verifiers.
- **02D evidence audit recomputed from saved data:** 2,208 checks. 02D's
  `validate02d` CLI rewrites `results/02d/validation.json`, so its pure `audit()`
  was called directly; nothing under `results/02d` changed.
- **Fresh numerical reproduction.** 02D's unchanged preflight reproduced all 120
  historical 02C outcomes, plus zero-residual exactness, determinism, passive
  recording and BAM parameters. All 1,044 02D held-out rows (612 grid, 432
  shift) were re-simulated, and every row equals its published row exactly (6
  workers, 922 s).

### The compiled-model hash mismatch

- The historical `model.mjb` (`ebb2ce…`) was compiled on **Windows**: MSVC
  Python, upstream at `C:\Users\James\microduck_rl`. Recompiling there with the
  same packages reproduces `ebb2ce…` exactly. Recompiling on Linux/WSL gives
  `cba1a6…`, the hash 02D reported. Each platform is deterministic.
- `model.json` records upstream XML/constants hashes that Linux appears to
  contradict. Those are the CRLF hashes of a `core.autocrlf=true` Windows
  checkout: converting the Linux LF files to CRLF reproduces them byte for byte.
  Upstream content is identical at `b02fdea`, clean.
- Field by field, the two binaries agree in 584 of 590 numeric model fields:
  masses, inertias, geometry sizes, solver options, mesh vertices, faces and
  normals, names, asset paths. All six differing fields are MuJoCo
  convex-hull/polygon arrays (`mesh_graph`, `mesh_poly*`). In 8 of 38 meshes
  their bytes differ, including both soles.
- Every hull has the same vertex set. Six have identical face and polygon sets
  and differ only in ordering. `jaw` triangulates coplanar facets differently;
  `power_support` has a different polygon set.
- A minimal identical MJCF around a single upstream STL (`platform02e`) gives the
  same vertices but different hull/polygon bytes on Windows vs Linux MuJoCo
  3.10.0. The cause is platform-dependent hull processing in MuJoCo's compiler,
  not upstream drift, line endings or mjlab.
- Dynamics: on the fresh Linux binary, 1,009 of 1,044 02D trials are bitwise
  identical and **all 1,044 outcomes are identical**. Every one of the 35
  divergences starts after a fall, non-foot contact or self-contact (earliest
  1.67 s), when ordering-dependent contacts on the shells begin.
- **Decision:** the hash guard was not weakened. 02E simulates only with the
  hash-matching historical binary. The Linux binary stays local evidence; it
  embeds upstream meshes. `fields_digest_excluding_hull` is a diagnostic identity,
  not a replacement guard.

## Feasibility before the protocol (exploratory; results in `feasibility/`)

- `sim02e` with no 02E intervention is bit-identical to `sim02d`, with and
  without a +2° offset.
- Static gravity-vs-leg-kinematics, level floor, reflex standing: estimates of
  +1.93/−2.01/+2.17° for +2/−2 pitch and +2 roll; 0.11/−0.06° at nominal. A 2°
  pitch slope read as 2.08°, and +2 offset on a +2 slope as 3.89°. This is the
  predicted slope confound.
- Hip-yaw probe (both hip yaws, either sign, 0.05–0.2 rad): with both feet
  planted the trunk does not yaw cleanly; the feet twist or slip (up to 5 mm at
  0.2 rad, 10.6° tilt). Estimates were noise.
- Rocking probe, first formulation (per-step rotation increments): signal
  ~0.0025 rad per step vs comparable encoder-difference noise; errors 4–150°.
- Second formulation (cumulative rotation vectors): still 5–40°, and the large
  false corrections then toppled the reflex. A noise-free diagnosis showed the
  gyro integrates trunk rotation to ≤0.002 rad. The leg-kinematic trunk rotation
  was wrong by 0.007–0.043 rad because the soles rotated 0.7–3.7°. The signal
  (offset × probe rotation) is second order, about 0.001 rad.
- Before declaring the protocol, the probe estimator was corrected to compare
  angular velocities, with a single-axis fit for a single-axis probe. The
  cumulative formulation had scored the unobservable roll component and carried
  a second-order integration error. A smoke test of the corrected estimator:
  clean-gyro oracle exact at nominal; noisy-gyro oracle 0.03–1.0°;
  kinematic-from-true-joints 0.7–14° (rate error 35–89% from sole rotation);
  the 0.10 rad probe fell in all three smoke trials.

## Phase 1 execution

- **Training reproduction before any 02E learning.** `learn02e` with
  randomization off reproduced 02D's published `history-s72` generation 1
  exactly: candidates, outcomes, scores, CEM mean/std/best, RNG state, and every
  02D condition field (`learning/reproduce.json`).
- **Characterization stopped once, for speed only.** The first 1A launch spent
  4.5 of 6.6 s per trial in Python-loop polygon tests for support margins. It
  was stopped after the reflex's 150 trials. `support_margins` was vectorized
  with the same element-wise operations and tolerances; 900/900 recorded states
  gave bitwise-identical margins. The partial run, its identity and the original
  `sim02e.py` source are kept in `diagnostic/characterize-interrupted-v1/`. The
  validator requires its reflex rows to equal the rerun byte for byte. No result
  was seen or used to make this change; the reflex summary line printed the
  same 110/150 both times.
- **Erratum, documentation only.** The `sim02e` module docstring says a
  positive pitch slope puts "+x uphill". The implemented convention, tested in
  `test_floor_slope_and_drift_conventions`: positive pitch slope means the
  floor descends toward +x, so the robot faces downhill. A correct IMU on a
  trunk perpendicular to that floor reads the same sign as a positive pitch
  misalignment. The file was not edited after the batteries began, because the
  evaluation identities pin its hash.

### Worker memory: the run was stopped by host memory exhaustion

- **What happened.** 1A `characterize` (6 workers) ran at the same time as the
  three 1D pilots (3 × 2 `learn02e` workers). After `history-s3` (18:27) the six
  battery workers had each grown to about 1.6 GB. WSL (7.9 GB RAM, 2 GB swap)
  thrashed and was terminated, which also ended the operator session.
- **Nothing partial was kept, and nothing valid was lost.** 1A had six complete
  configurations (`reflex` … `history-s3`, 150 rows each, recovered counts as
  logged). `memoryless-s1..s3` were never written, and there were no `.tmp` files.
  Each pilot had three complete, atomic generations (576 episodes). Every pinned
  source and model hash still matches its identity, and `reflex` still equals
  the interrupted-v1 rows byte for byte.
- **Cause: a NumPy 2.4.1 leak, not retained simulator state and not expected
  footprint.** A fresh worker is about 170 MB: the MjModel buffer is 77 MB and
  each MjData arena is 14 MB. One in-process worker then grew 11–12 MB per 1A
  trial. Neither `gc.collect()` nor `malloc_trim` released it, and the count of
  GC-tracked objects did not change. Each path was tested separately, one
  process per variant:
  - learning rollouts: no growth;
  - 200 Hz recording, `mj_copyData`/`mj_forward`, and contact access: no growth;
  - `margins=True`: 11.5 MB per trial;
  - `physics02b.support_polygon` alone: 11.4 MB per 300 calls. Within it,
    `ConvexHull` (Qhull) does not grow; `np.unique` does.

  `np.unique` retains one buffer the size of its input on every call
  (retained/input = 1.0 for 200–6,000 rows, float and int, `axis=0` and 1-D).
  tracemalloc shows the buffers as live NumPy allocations. Sorting and lexsort
  do not leak. `diag02e.evaluate` records margins on every control step, so a
  6 s trial makes 300 calls. Over about 150 trials per worker that predicts
  1.7–1.9 GB per worker, which matches what was observed.
- **Remedy, outside every pinned source.** The pinned evaluation and learning
  sources are unchanged: `diag02e`, `sim02e`, `physics02b` and `learn02e`, and
  the NumPy version (a package change would alter the Phase 0 identity).
  `pool02e` replaces the battery pools at run time:
  - every batch of about `workers × 20` tasks runs in a freshly forked pool,
    which is then shut down;
  - a guard is polled from the main thread. It stops the run and terminates
    every worker if any worker exceeds 900 MB, the study's process tree exceeds
    2,000 MB, or WSL `MemAvailable` falls below 1,500 MB;
  - workers die with their driver (`PR_SET_PDEATHSIG`) and keep plain SIGTERM;
  - SIGTERM or SIGINT to the driver terminates and reaps the whole tree;
  - at most four workers.

  `diag02e` battery steps now run through `python -m movement_lab.pool02e`.
  `study02e` uses the same pools for its batteries, and for training it runs at
  most `--concurrent-runs` jobs at once under a process-tree guard.
- **Verified with one worker before any concurrency.**
  - Through the bounded pool, 30 1A `history-s1` tasks (all 15 settings × all 5
    kinds) reproduced the published rows byte for byte.
  - Over 15 tasks per lifetime, worker RSS went from 136–154 MB to a peak of
    311 MB (about 11.7 MB per task). It returned to baseline at each
    replacement, with no orphaned processes. The rate was 1.83 s per task, the
    same as unbounded.
  - `learn02e.worker` in one long-lived process stayed at 147→149 MB over 132
    evaluations.
  - Unit tests cover task order, bounded lifetimes, guard breaches, task
    exceptions and the worker signal settings.
- **Execution budget on this machine.** The host is a 6-core/12-thread Ryzen 5
  2600 with 16 GB. WSL gets 7.9 GB by default, and the Windows side had about
  3.5 GB free during these measurements. The budget:
  - batteries at most 4 workers × 20 tasks, about 0.4 GB peak per worker and
    about 1.7 GB total;
  - training at most 4 workers in total (one run × 2 workers to start);
  - batteries and training never run together above 4 workers;
  - concurrency is raised only after a short run's memory log stays within its
    expected peak.
- **1A completed with 2 bounded workers.** The run was
  `pool02e --workers 2 diag02e characterize`, 457 s. It retained the six
  completed configurations under the unchanged identity and added
  `memoryless-s1..s3`: 91, 84 and 91 of 150 recovered.
  - Memory: peak worker 388 MB and peak driver plus workers 822 MB. WSL
    `MemAvailable` stayed at or above 6,554 MB, with no guard breach and no
    swap.
  - Peak worker RSS per batch was 292–381 MB over 12 fresh-worker batches, with
    no trend across batches.
  - The validator's battery audit passes (957 checks).
  - Stdout is in `diagnostic/characterize-resume.log` and memory samples are in
    `diag02e-characterize-memory.log`.

## Interruptions and recovery (2026-11-13)

- **The multi-hour stall was the first 1A run, already diagnosed above.**
  - Six unbounded `diag02e characterize` workers ran alongside three
    2-worker pilots.
  - From about 18:27 on 2026-11-10, WSL thrashed with swap full, load
    about 22 and workers at 1.2–1.3 GB RSS plus about 0.3 GB of swap each.
  - It was terminated at 22:51. The cause was the NumPy `np.unique` leak, and
    the remedy is `pool02e`.
- **The 1C front-end pilot was stopped on request, not stalled.**
  - The run was `pool02e --workers 4 diag02e pilot`, started 02:53 on
    2026-11-11. It received SIGTERM at 03:05:19 after 12 of 28 configurations:
    `none` for all seven controllers, plus `gate` for reflex, history s1–s3 and
    memoryless s1.
  - The handler terminated the workers and left no `.tmp` or partial file.
  - Memory peaked at 372 MB per worker and 1,512 MB in total, with no guard
    breach.
- **Recovery check (read-only, no simulation).**
  - WSL had restarted, and no study process was running.
  - Every battery identity still matches the current sources and model. Every
    saved configuration has full rows in the declared order. The
    interrupted-v1 reflex rows still equal the rerun byte for byte.
  - Each 1D pilot's three generations reconstruct exactly from its seed:
    candidates, curriculum, scores, CEM mean/std/best and RNG state.
  - `learning/reproduce-history-s72-g1/config.json` pins the pre-vectorization
    `sim02e.py` (`c3fbb7…`). That is Phase 0's committed source and the
    interrupted-v1 snapshot. Learning rollouts never compute support margins,
    so the vectorization cannot affect that check.
  - All 20 tests in `tests.test_upright02e` pass.
- **State at recovery.**
  - 1A and 1B are complete.
  - 1C has 12 of 28 configurations.
  - 1D has 3 of 8 generations per pilot (576 of 1,536 episodes), and no final
    controllers.
  - No gate has been computed, and Phase 2 has not started.
  - `report02e`'s replay step imports `replay_select02e`, which was never
    written.
  - `PUBLICATION.md` describes the intended final layout, not this state.
- **Resume with the unchanged declared commands.** Completed files and
  generations are retained, and identities are re-checked:
  `pool02e --workers 4 diag02e pilot` → `diag02e gate` →
  `study02e pilot-learning --workers 2 --concurrent-runs 1` →
  `study02e evaluate-pilot --workers 4` → `study02e gate-learning`. Phase 2
  runs only for an arm or front-end that passes its gate.

## Phase 1 completion (2026-11-13)

- **1C resumed.** The run was `pool02e --workers 4 diag02e pilot`, 1,096 s. It
  retained the 12 saved configurations and ran the other 16.
  - Memory: peak worker 375 MB, peak total 1,512 MB, WSL `MemAvailable` at or
    above 6,058 MB, no breach.
- **1C gate (declared): no front-end passes.**
  - `static` gained a net +75 paired recoveries on misaligned cells (85 gains,
    10 losses), above the required 30. It also lost 4 nominal quiet trials:
    both quiet seeds of history s1 and of history s3. With a correct IMU its
    nominal estimates were 0.13–0.30° roll and −0.10 to −0.16° pitch.
  - `prior` netted −3.
  - The `gate` control, which corrects nothing, netted +9 and lost 1 nominal
    quiet trial (history s3). Part of the nominal cost is the calibration
    window itself.
  - Neither front-end proceeds to Phase 2.
- **1D resumed.** The run was `study02e pilot-learning --workers 2
  --concurrent-runs 1`. It took each pilot from generation 3 to 8 at about 28 s
  per generation.
  - Memory: peak worker 138 MB, peak total 358 MB.
- **1D evaluation.** `study02e evaluate-pilot --workers 4` on D, front-end
  `none`, recovered of 90:

  | Controller | Recovered |
  |---|---|
  | `history-s72` | 38 |
  | `history-rand-s72` | 62 |
  | `memoryless-rand-s72` | 61 |
  | `memoryless-s72` | 38 |

  - Memory: peak worker 376 MB.
- **1D gate (declared): both randomized arms pass.**
  - History arm: net +16 (18 gains, 2 losses).
  - Memoryless arm: net +15 (18 gains, 3 losses).
  - Both had zero nominal quiet or backward losses, and nominal recoveries
    equal to the control (8 and 8).
  - Slope cells, reported and not gated: 8 gains and 0 losses for each arm.
- **Phase 2 applies to both learning arms only.** That is
  `study02e train` (2 arms × 3 seeds × 2,304 episodes) and `study02e confirm`
  (battery H: 7 controllers with `none` plus the 6 trained runs, × 360 trials).

## Phase 2 and publication (2026-11-13)

- **The first Phase 2 launch ran nothing.**
  - It was an inline `wsl.exe … bash -c '…'` command. `wsl.exe` expands `$VAR`
    and `$?` before bash runs, so `$PY` became empty. Both steps failed at once
    with "command not found" and wrote no files.
  - For the same reason, the `exit=` lines that inline wrappers appended to
    `pilot/frontend.log`, `learning/pilot-learning.log` and
    `pilot/learning.log` show the wrapper's status, not the run's. That
    includes the earlier `stopped by signal 15` followed by `exit=0`.
  - Completion is established by each run's final memory-summary line, which
    is printed only after a step completes, and by the validator. Phase 2 was
    rerun from a script file.
- **Training.** The run was `study02e train --workers 4 --concurrent-runs 2`:
  6 runs × 12 generations, 2,304 episodes each (13,824 in total), 358–369 s
  per run.
  - Memory: peak worker 138 MB, peak total 675 MB.
- **History randomized s1 ended exactly on the frozen reflex.** The all-zero
  candidate (population index 2) won its last generation, so its final
  parameters are zero.
  - Final = last training winner, as declared; it was not replaced.
  - All 360 of its held-out rows equal the reflex's.
- **Confirmation.** The run was `study02e confirm --workers 4`: 13
  configurations × 360 trials, since no front-end passed 1C. It took 2,656 s.
  - Memory: peak worker 412 MB, peak total 1,668 MB, WSL `MemAvailable` at or
    above 5,897 MB.
- **Report.** `replay_select02e`, which `report02e` imports, was written after
  confirmation.
  - Each of its 26 recordings is the first replicate of a battery cell, and
    each reproduces its published row exactly (`report/replay-checks.json`).
  - Presets 05 (static's nominal quiet regression) and 09 (the held-out
    0.32 m/s trade-off) were chosen after their results were known. Their
    captions say so.
  - The raw 200 Hz `records/*.npz` (20 MB) stay local and hashed. The replay
    page is 17.7 MB (02D's is 30 MB).
  - Headless Chrome 157 check: 26 recordings, 9 presets, 108 snapshots.
  - The figure `confirmation.png` was added to `report02e`.
- **Validation.**
  - 21 of 21 tests pass.
  - `validate02e --audit --resimulate 12`: 6,444 checks and 12 exact held-out
    resimulations.
  - `--publish`/`--verify`: all published hashes verify.

## Review corrections (2026-11-13)

The automated review of PR #11 (Codex) raised two P2 findings. Both are fixed.
No trial outcome, gate decision, training run or conclusion changed.

- **Support margins were projected against gravity on sloped floors.**
  - `sim02e.support_margins` placed the COM's ground point at
    `com − g_xy · com_z / (−g_z)`. The gravity line through the COM meets the
    floor at `com + g_xy · com_z / (−g_z)`.
  - On a ±2° slope, the stored point was 2 · com_z · tan 2° ≈ 9.4 mm uphill of
    the true one (COM height about 0.135 m).
  - On a level floor, gravity's horizontal component is exactly +0.0, so both
    forms give bit-identical margins.
  - Margins are diagnostic only. No controller, front-end, failure test,
    score, gate or training rollout reads them (`learn02e` never requests
    them). Only six margin fields of sloped rows could be wrong:
    `min_support_margin_m`, `min_toe_margin_m`, `min_heel_margin_m`,
    `steady_toe_margin_m`, `steady_heel_margin_m` and
    `com_outside_support_s`.
- **Fix.** The sign was corrected. The `sim02e` docstring erratum recorded
  under Phase 1 ("+x uphill") was corrected in the same edit, since the file's
  hash was changing anyway.
- **Regeneration, `correct02e margins --workers 4`.**
  - It re-simulated exactly the 1,444 sloped battery rows: 180 (1A), 560
    (1C), 80 (1D) and 624 (H).
  - It also re-simulated the three sloped replay recordings. Their 200 Hz
    arrays came back identical, so their local `.npz` files were kept and only
    the margin fields of their summaries changed.
  - Every non-margin field of every regenerated row came back identical. All
    1,444 rows changed margins.
  - It took 17 minutes. Memory: peak worker 323 MB, peak total 1,330 MB, WSL
    `MemAvailable` at or above 6,249 MB.
- **Provenance.**
  - Battery identities now pin the corrected `sim02e.py`. Training
    configurations keep the hash they ran with.
  - The source that produced the original rows is archived in
    `corrections/margin-projection/source/`. The validator checks that it
    differs from the current source only by the declared edit.
  - `corrections/margin-projection/rows.jsonl.gz` keeps each regenerated row's
    old and new margins. The validator rebuilds every original file's rows
    byte for byte from it. The interrupted-v1 1A rows predate the correction,
    are unchanged, and are compared against those rebuilt rows.
- **What changed in the published evidence.**
  - The margin fields of the sloped rows, and the summaries of the three
    sloped replay records.
  - In `report/tables.json`, 66 values: the steady toe and heel margins of the
    slope settings in the 1A and held-out quiet-mechanism tables. On the +2°
    slope, the steady toe margin fell by 9.4 mm and the heel margin rose by
    9.4 mm.
  - The −2° slope changed more (mean −40 mm toe, −53 mm heel). There, the
    corrected ground point leaves the support polygon during some steady
    windows, and those means include the −1 m outside-support sentinel, so
    they are not distances. COM time outside support rose by 0.09 s on
    average.
  - `REPORT.md` never quoted a margin. No count, gate decision, figure,
    replay frame or `replay.html` byte changed.
- **Training was not restarted.** Learning rollouts never compute margins, so
  no candidate score, CEM update or final controller could change.
- **The learning-gate audit ignored `--root`.**
  - `validate02e --audit` recomputed the 1D gate with
    `study02e.gate_learning()`, which read and wrote the hard-coded
    `results/02e`.
  - For any other root, the saved decision was compared with itself, so it was
    never checked against that root's rows. `results/02e/pilot/gate-learning.json`
    was rewritten as a side effect.
  - Every published audit used the default root, where the read and the write
    hit the audited tree, so the published check was valid.
  - Fix: `gate_learning(root)`. The validator now recomputes both gates on a
    scratch copy of their input rows and compares the result with the saved
    decision.
  - The audit writes nothing under the audited root except `validation.json`,
    and nothing anywhere else. The old version wrote each gate file and then
    restored its saved bytes.
- **Regression tests.**
  - `test_support_margins_project_the_com_along_gravity`: pitch and roll
    slopes against a known polygon, plus a COM whose gravity line leaves past
    the toe. It fails on the archived source.
  - `test_margin_correction_is_exactly_the_declared_edit_and_spares_training`.
  - `GateAuditTests`: gates on a synthetic alternate root are recomputed from
    that root's rows, and a tampered decision is caught. Neither the alternate
    root nor `results/02e` is written. It fails on the old `gate_learning()`.
