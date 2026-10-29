# Completed 01B study: publication and reproduction

The study is complete at its declared bounds. It demonstrates stable two-foot
standing followed by small head-yaw articulation with partial return. Bilateral
flexion leaves a stable offset; lateral and unilateral commands miss the fixed
response thresholds. No reversible whole-body weight transfer, foot unloading,
stepping or recovery maneuver is established. The positive-head response only
narrowly meets the exploratory contraction rule. No threshold was relaxed.

## Published evidence

- [Initial baseline failure](REPORT.md), with [measured replay](gpu-gate/replay.html).
- [Landing/contact diagnosis](landing-diagnosis/REPORT.md), its contact locations,
  normals, penetration, first-100-ms CSVs, model/solver comparison and two revised
  initialization records.
- [Controlled-departure report](departures/REPORT.md), [comparison and replay index](departures/comparison.html),
  [joint tracking](departures/joint-tracking.png), [continuous plots](departures/comparison.png),
  fixed plan, paired numerical summary and measured subprocess costs.
- All eleven measured trials' metadata and small initial states, resolved
  configurations, source snapshots, checkpoint/upstream/model hashes, validation
  outputs, and original full-evidence manifests. The failed CUDA initialization
  attempt is represented by its original manifest and the reports; its duplicate
  preflight directory remains local.
- Evaluator/analysis/replay/validation source, the reporting-repair script and
  original pre-finalization metadata, all 31 scientific tests and two publication-scope regression tests, plus fresh
  [test](publication/tests.txt) and [saved-record validation](publication/saved-record-validation.txt) output.

The publication allowlist includes the exact historical source path `movement_lab/landing_diagnostics.py`; it does not allow other diagnostic filenames by wildcard. Two regression tests cover legitimate 01B sources and reject unrelated or near-matching paths. File-size and local-only artifact guards remain unchanged.

The reports remain chronological: the original failure report describes that
attempt, not the final status of the whole study. Later results do not overwrite
its failed gate. Original source snapshots include a baseline postprocessing
failure; the repaired analysis and reconstructed provenance are separately
identified in the departure report. The repaired baseline did not consume an
extra physics rollout. Total baseline subprocess wall time is unavailable;
the departure report states a lower bound, not an invented total cost.

## What remains local

Nothing was deleted. `.gitignore` excludes:

- Full-rate `results/01b/**/trial.npz` trajectories and all `replay_model.mjb`
  compiled models. The latter embed upstream robot geometry and are about 78 MiB
  each; upstream mesh licensing still applies. Checkpoints/assets are not bundled.
- Evaluation/driver logs, the zero-physics-step failed CUDA preflight directory
  `results/01b/gate/`, and landing preflight text/script copies.
- Redundant per-trial `atlas.png` exports (except the original failure report's
  linked plot), an intermediate test log, and ordinary caches.

[Publication inventory](publication/manifest.json) lists published files and
SHA256/size entries for these local-only evidence files. Caches are not evidence
and are excluded from that inventory. The manifest excludes itself. Earlier
manifests are preserved byte-for-byte and intentionally describe the **full local**
evidence set: their references to omitted files do not mean those files are
available or validated in a fresh checkout.

The standalone comparison embeds its charts. Portable HTML replays embed
50 Hz decimated measured link centers and actual/target joints; they play at
12.5 frames/s (four times slower). They are not collision renderings and cannot
replace 200 Hz raw trajectories for revalidation. Desktop full-geometry replay
needs the omitted compatible MJB and trajectory and sets saved states without
integrating dynamics.

## Verification available in a checkout

From the repository root, with the Python dependencies available:

```bash
python -m movement_lab.publication01b
python -m unittest discover -s tests -v
```

The first checks every published-file hash and size, without pretending to check
missing raw data. The full local run passed all 33 tests (31 scientific and two publication-scope regressions) without skips. A fresh
checkout may explicitly skip the local compiled-geometry test and optional older
raw-fixture subtests; skips are not evidence of their validation. Formula and
available-record tests still run. Saved-record validators require the omitted
artifacts and fail when they are absent.

## Full local validation and reproduction

For full validation, restore the local-only files to the exact paths in the
inventory, the separately installed upstream Microduck environment and checkpoint
at the paths in provenance, and the compatible MuJoCo version for MJB loading:

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m movement_lab.publication01b --full
$PY -m movement_lab.validate01b results/01b/gpu-gate
$PY -m movement_lab.validate01b results/01b/landing-diagnosis/clearance-1mm
$PY -m movement_lab.validate01b results/01b/landing-diagnosis/clearance-1mm-repeat
$PY -m movement_lab.validate_departures01b
$PY -m unittest discover -s tests -v
```

Reintegration requires CUDA, checkpoint 99 at the recorded default path,
upstream revision and asset hashes, and software versions from `provenance.json`
(mjlab, MuJoCo/Warp, Torch and BAM). Training files/checkpoints remain in the
separately installed upstream workspace. The scripts expect its registered task
and policy-normalizer/history pipeline. The recorded hardware was an RTX 2070
SUPER. Seed 42 and identical settings do not guarantee bitwise reproduction on
other devices, drivers or simulator versions.

Use a **new empty output directory**, never the published record paths. The
single-condition runner accepts `--out`, `--sequence` and the explicit reset height:

```bash
$PY -m movement_lab.departure01b --out /new/path/baseline --sequence zero --root-height 0.1181823627
$PY -m movement_lab.validate01b /new/path/baseline
```

For a fresh bounded matrix, run from a new empty working directory with this
repository on `PYTHONPATH`: generate its `results/01b/departures/baseline` using
the preceding command with that relative output path, validate it, then run
`python -m movement_lab.run_departures01b`. That driver has a fixed relative
output root, validates the baseline and each new record, and refuses existing
output directories. It runs at most seven additional six-second trials and
stops after a physical/recovery failure. It must not be run from the existing
publication checkout expecting it to overwrite or resume completed evidence.
Use `report_departures01b` and `validate_departures01b` in the same new working
directory to generate and validate the matched comparison. This publication
performed **no additional GPU trials**.

Regenerated records are new evidence: source/config/run metadata and hashes may
differ. Keep them separate rather than substituting them into the original
manifests. Archived `study01b` documents the original failed unsupported-start
attempt; `finalize_baseline01b` documents a one-time reporting repair and is not
a normal regeneration step. Existing report and preserved source references to
that repair must not be interpreted as instructions to rerun it on finalized data.

Earlier 00A–01A results and upstream files remain unchanged. Publication includes
no policy training or automatic continuation, and the PR is not merged by this workflow.
