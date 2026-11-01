# 02A publication and reproduction notes

The original pilot report, archives, proposal history, executed source snapshots,
measurements, configurations and manifests are preserved byte for byte. The report's
statement that no commit or publication was performed describes the experimental
run; this later publication is separately recorded here. No new GPU trials were run.

Five ten-second GPU trials completed in 364.6149 wall seconds: one standing baseline,
two adaptive proposals and two non-adaptive proposals. All passed physical validation.
Adaptive discovered one distinct outcome within its archive; control discovered two.
After observing negative head yaw, the adaptive explorer chose negative head yaw
plus a half-unit negative leg coefficient. This was a new command, but its normalized
measured descriptor distance was 0.424923 < 1, so the archive retained it as a redundant
valid outcome. Search adaptation changed the next proposal; the standing motor-control
policy was unchanged. No discovery-efficiency advantage was established. One search
seed and two proposals per arm do not establish that adaptation is generally inferior.
No newly discovered movement was independently confirmed and no motor policy was trained.

`pilot-gpu/REPORT.md`, `archive.json`, `attempts.json`, `proposals.json`, `metrics.json`,
`PLAN.json`, `cost.json`, outcome plots and representative full-rate trajectory plots
form the main result. Per-trial replay HTML files retain portable measured-link
replays, target/actual measurements and contact plots; they are not dynamics reruns.
The offline development archive is historical 01C evidence, not new 02A exploration.

The read-only `movement_lab.validate02a` audit recomputes physical measurements,
archive novelty/admission, both proposal histories, and the matched historical
baseline. It checks immutable original manifests, checkpoint and upstream hashes,
CPU saved-state geometry/COM/contact counts, actuator mapping/effort, complete trial
allocation, and original report conclusions against actual audited records. The
publication manifest separately inventories current source, test and validation
artifacts. Added tests reject missing arms/conditions, unexpected publication scope,
and counting different commands as distinct outcomes. Existing incomplete-baseline
and safety-stop regression tests remain active. No reports or original records are
regenerated as part of publication; missing evidence causes validation to fail.

Original source hashes freeze experimental code and shared documentation at the
end of the pilot. Later `.gitignore` rules only append exclusions; the original
hash remains verified against its exact byte prefix. Earlier 00A–01C tracked files,
all available original manifest entries, and upstream Microduck RL are unchanged.

## Published and local inventory

Published: source and tests; original and final source snapshots where they differ;
scientific reports and notes; both archives; full allocation and proposal history;
small initial states and per-trial metadata/configurations/provenance/validation;
original manifests; main outcome/trajectory PNGs; self-contained measured-state
replays; and publication validation/test evidence.

Excluded but preserved locally and hashed: five full-rate `trial.npz` trajectories,
five compiled `replay_model.mjb` models containing upstream geometry, ten runtime and
validation logs, two redundant publication console transcripts, and five redundant per-trial `atlas.png` exports. Caches and Python
bytecode are also excluded, but are not scientific evidence. Original manifests
continue to describe full local records. Publication's local-only inventory records
the excluded evidence without deleting or rewriting any file.

Full reproduction requires the separately installed compatible Microduck/mjlab,
MuJoCo and BAM environment, checkpoint 99 at its recorded SHA256, upstream robot
assets, and CUDA for simulation. Full saved-record audit requires restoring the
excluded local raw trajectories/models plus the earlier raw study evidence. A fresh
checkout can verify published hashes and run available-data tests, but cannot infer
validation of absent raw data. Absolute local paths and simulator/GPU versions in
provenance are intentional reproduction metadata, not credentials.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
PYTHONDONTWRITEBYTECODE=1 $PY -m unittest discover -s tests -v
$PY -m movement_lab.validate02a
$PY -m movement_lab.validate01c
$PY -m movement_lab.publication02a --full
```

See `publication/tests.txt`, `publication/saved-record-validation.json`,
`publication/earlier-record-validation.txt` and `publication/manifest.json`.
No merge or Movement Lab 02B is part of publication.
