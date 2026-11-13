# 02E evidence and reproduction

02E adds `movement_lab/*02e*` and `tests/test_upright02e.py`. It changes no
02B/02C/02D source, checkpoint or result; the validator pins every reused
helper and historical input by SHA256. All simulation used the hash-matching
historical model through the unchanged 02B guard.

## Published evidence

- `phase0/`: the reproducibility gate.
  - identities: model, upstream revision and its CRLF/LF file hashes, BAM
    parameters, packages, checkpoints and sources;
  - published-hash verification and the 02D evidence audit;
  - 02D's unchanged preflight rerun (120 historical 02C outcomes);
  - per-row resimulation of all 1,044 02D held-out trials, compared both with
    the published rows and with a freshly compiled Linux model;
  - a field-by-field and per-mesh model comparison, plus single-mesh platform
    probes from Windows and Linux.
- `feasibility/`: the pre-protocol probes, reproduced by command.
- `diagnostic/`:
  - the 1A characterization (9 controllers × 150 trials);
  - the 1B estimator diagnostic (224 trials plus 8 nominal calibration trials
    for the IMU-only prior), with error decomposition and a summary;
  - the interrupted first 1A launch with its original source.
- `pilot/`: the 1C front-end battery (7 controllers × 4 front-ends × 90), the 1D
  learning-pilot battery, and both gate decisions.
- `learning/`: the training-reproduction check and the three 1,536-episode
  pilots. Every generation stores candidates, outcomes, conditions, search
  distribution and RNG state.
- `runs/`: Phase 2 training for the two arms that passed 1D, 3 seeds × 2,304
  episodes each. No front-end passed 1C, so none was confirmed.
- `confirm/`: held-out battery H, 13 configurations × 360 trials: the reflex
  and six 02D controllers with front-end `none`, plus the six trained runs.
- `report/`:
  - `tables.json` and the figures;
  - `replay.html`, the portable replay of 26 declared recordings;
  - `replay-checks.json`, listing the battery rows each recording reproduced
    exactly;
  - `browser-check.json`, a headless Chrome check of every preset, camera and
    control (`movement_lab/browsercheck02e.js` appended to a QA copy), and
    `browser-preview.png`.
- `records/`: metadata and summaries of the replayed trials.
- `validation.json`, `manifest.json`, `tests.txt`.

These stay local and are hashed in `manifest.json` (`local_only`):
- The fresh Linux `model.mjb` (`phase0/linux-compile`) embeds upstream meshes.
  It is also hashed in `phase0/model-comparison.json`, and its `model.json`
  and resolved configuration are published.
- The replay's raw 200 Hz `records/*.npz` (about 20 MB). `report02e`
  regenerates them exactly; each must reproduce its published battery row.

## Commands

From the repository root with the separately installed upstream environment and
the hash-matching `results/02b/model/model.mjb` (see 02D notes):

```bash
PY=/home/james/microduck_rl/.venv/bin/python
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Phase 0
$PY -m movement_lab.repro02e identity
$PY -m movement_lab.repro02e artifacts
$PY -m movement_lab.repro02e compile        # fresh model into phase0/linux-compile (never installed)
$PY -m movement_lab.platform02e --meshes ~/microduck_rl/src/mjlab_microduck/robot/microduck --out results/02e/phase0/platform-linux.json
#   (repeat platform02e with the Windows venv to platform-windows.json)
$PY -m movement_lab.repro02e compare
$PY -m movement_lab.repro02e resimulate --alternate results/02e/phase0/linux-compile --workers 6
$PY -m movement_lab.repro02e preflight
$PY -m movement_lab.repro02e gate
# Phase 1
$PY -m movement_lab.diag02e feasibility
$PY -m movement_lab.diag02e estimators
# Battery steps go through bounded, memory-guarded pools: NumPy 2.4.1's np.unique
# leaks ~11.7 MB per margin-recording trial (ATTEMPTS.md, "Worker memory").
$PY -m movement_lab.pool02e --workers 4 diag02e characterize
$PY -m movement_lab.pool02e --workers 4 diag02e pilot
$PY -m movement_lab.diag02e gate
$PY -m movement_lab.study02e reproduce-training
$PY -m movement_lab.study02e pilot-learning --workers 2 --concurrent-runs 1
$PY -m movement_lab.study02e evaluate-pilot --workers 4
$PY -m movement_lab.study02e gate-learning
# Phase 2 (each refuses to run unless its gate passed)
$PY -m movement_lab.study02e train --workers 4 --concurrent-runs 2
$PY -m movement_lab.study02e confirm --workers 4
# Report, audit, publication
$PY -m movement_lab.report02e                # re-simulates the 26 replay records unless present
#   browser check: append movement_lab/browsercheck02e.js to a copy of report/replay.html, open headless
$PY -m unittest tests.test_upright02e -v > results/02e/tests.txt 2>&1
$PY -m movement_lab.validate02e --audit --resimulate 12
$PY -m movement_lab.validate02e --publish
python -m movement_lab.validate02e --verify   # portable: hashes only, no model needed
```

Use a new root, for example by copying `PROTOCOL.md` into an empty
`results/02e-reproduction`, to regenerate rather than verify. Completed
evaluation files are reused only when their identity matches: sources, model,
controller weights, front-end settings and exact conditions. Training resumes
only complete generations with identical configuration and sources.

## Limits of reproduction

Hash verification is portable. Physics checks need the Windows-compiled
binary. Phase 0 shows that a Linux compile reproduces every 02D outcome and is
bitwise identical until a fall or non-foot/self-contact. Even so, it is not a
drop-in replacement: its hash differs, and post-contact trajectories can
diverge. Bitwise replay across machines, operating systems or MuJoCo builds is
not claimed. Everything here is simulation; there is no hardware claim.
