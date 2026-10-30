# Movement Lab 01C — publication and reproduction

The bounded pilot is complete. Five **ten-second** trials retained 2,000 samples
per trial at 5 ms: zero-residual baseline, flexion, opposite flexion (extension
command), positive head yaw, and negative head yaw. The publication request's
“six-second trials” conflicts with the plan, metadata, arrays, report and runner.
Six seconds is the historical 01B prefix reproduced exactly by the new baseline;
it is not the 01C duration. No records were shortened or relabeled.

The measured evaluation-driver wall budget was 510.8048 s (8.51 min), including
startup, recording and per-trial validation, not GPU kernel time. A separate
failed sandbox CUDA initialization produced no simulation samples. Publication
adds no GPU trials and trains no policy.

## Findings and limits

Flexion established a stable changed configuration, with a final matched COM
difference of 1.616 mm and a maximum final joint offset of 0.01356 rad. It did
not establish a symmetric crouch. Opposite flexion and signed head yaw showed
partial return. Positive head yaw retained 0.01976 rad and negative yaw retained
−0.00513 rad after the longer return/observation window. Broad absolute pose
tolerances passing is not exact return. These are observed responses under an
existing controller, not newly learned skills.

No meaningful height extension, full return, foot unloading or stepping was
established. Rocking in both directions and the final zero-residual repeat were
not reached before the launch cap. No outcomes are assigned to those conditions.
The new ten-second repeat remains missing; reproducibility evidence includes the
exact six-second historical trajectory prefix and identical model/configuration,
plus exact initial states and 400 pre-command samples across the five records.
One policy, seed and initialization do not establish robustness.

Worst observed penetration was 1.156 mm, below the unchanged 3 mm limit. The
validated 1 mm-clearance initialization, groundcontact model, standing-policy
semantics, BAM constraints and acceptance criteria remain unchanged. Friction,
compliance and feedback contributions to persistent offsets remain unresolved.

## Published evidence

- [Scientific report](pilot-gpu/REPORT.md), [atlas and synchronized comparison
  replay](pilot-gpu/comparison.html), and the atlas image it references.
- Five per-condition measured-state replays, metadata, small initial states,
  resolved configurations, source snapshots, checkpoint/upstream/model hashes,
  validation results, command plan, paired summary, per-joint/contact/effort
  analysis and measured costs.
- Original complete-local manifests, [01B window diagnosis](evidence01b.json)
  and its input hashes, [attempt notes](ATTEMPTS.md), experiment/analysis source,
  read-only validation source, scientific and publication-scope tests.
- Fresh [original 36-test run](publication/tests.txt), [38-test run including two
  publication guards](publication/tests-final.txt), and
  [saved-record audit](publication/saved-record-validation-passed.txt).
- [Publication inventory](publication/manifest.json), with byte sizes and SHA256
  for every published file except itself, and every excluded local evidence file.

The first publication-validator attempt expected a persisted budget-stop key in
`summary.json`; inspection of the executed source showed that key was only added
in memory after the last summary write. The failed audit log is retained at
`publication/saved-record-validation.txt`. The read-only validator now checks the
actual five-record schema and documented cost count; no scientific gate changed.

The original cost file's `completed: 6` counts the budget-stop marker. The five
validated trial directories and paired summary are authoritative. The final
runner corrects this bookkeeping count; the executed runner snapshot and original
cost file are preserved. No dynamics or historical analyses were rerun to fix it.

## Local artifacts and evidence integrity

Nothing was deleted or overwritten. Full-rate `trial.npz` trajectories, compiled
`replay_model.mjb` geometry, runtime logs, the zero-sample failed CUDA directory
`pilot/`, redundant per-condition `atlas.png` exports, and ordinary caches stay
local through `.gitignore`. The inventory records excluded evidence sizes and
hashes; caches are not evidence and are not inventoried. Compiled models embed
upstream robot meshes and remain subject to upstream asset licensing.

All original manifests, CSVs, raw records, reports and source snapshots retain
their original bytes. Original manifests describe the full local evidence and
therefore reference omitted artifacts; they cannot validate missing raw data in
a fresh checkout. Portable replays show measured link origins rather than
collision geometry, decimated to 50 Hz and played four times slower. They do not
replace the full-rate trajectories or full-geometry desktop replay.

Earlier 00A–01B tracked evidence and source remain unchanged from publication base
`813d32698f2e0a7e73eb2ef5e4cd51ff81f6123c`. Upstream Microduck RL is clean at the
recorded commit and every recorded upstream source/asset hash matches. No
upstream modification, training, automatic merge or Movement Lab 02 work occurs.

The original 01B publication command rejects the intentionally extended shared
`.gitignore`/`README.md` because their old whole-file hashes are frozen. Its failed
log is retained at `publication/earlier-evidence-validation.txt`. The current
publication audit verifies their original inventoried bytes as unchanged prefixes,
and verifies every other 01B published/local-only file exactly, together with all
three original 01B evidence manifests. See the
[passing earlier-evidence audit](publication/earlier-evidence-audit-passed.txt).
No earlier inventory or diagnostic CSV was modified.

## Validation and regeneration

A checkout with dependencies can verify published hashes and run available tests:

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m movement_lab.publication01c
$PY -m unittest discover -s tests -v
```

All original 36 tests passed locally; two new publication-scope regressions bring
the full suite to 38 passing tests. Fresh checkouts may explicitly skip older
optional raw-data/compiled-model tests. Skips do not validate missing records.

For complete saved-record verification, restore all inventoried local artifacts,
the separately installed upstream environment/checkpoint, and compatible MuJoCo:

```bash
$PY -m movement_lab.publication01c --full
$PY -m movement_lab.validate01c
```

The read-only validator rechecks original hashes, all five records' CPU geometry,
COM, terrain/nonfoot/foot contacts, actuator transmission/effort bounds, joint
limits, residual timing and composition, matched prefixes and outcomes. It never
regenerates reports, changes manifests or integrates dynamics.

Regeneration requires CUDA, recorded software versions, checkpoint 99 and the
upstream robot assets. Use a new output directory:

```bash
PYTHONDONTWRITEBYTECODE=1 $PY -m movement_lab.study01c --out results/01c/new-pilot
$PY -m movement_lab.report01c results/01c/new-pilot
$PY -m movement_lab.write_report01c results/01c/new-pilot
```

Do not run report generators against preserved original evidence. They write
analysis artifacts. Hardware-dependent wall time can change which conditions
fit the launch cap; record actual coverage rather than assuming eight trials
completed. Seeds do not guarantee bitwise reproduction across hardware/software.
