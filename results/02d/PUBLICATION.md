# 02D evidence and reproduction

The experiment preserves 02B/02C code and results. Its executable sources are
the `movement_lab/*02d*` files. The frozen model and standing-reflex checkpoint,
and the 02C seed-1 motor vocabulary, are identified by SHA256 in `preflight.json`
and each training configuration. `provenance.json` additionally records the
unchanged upstream revision and BAM parameters. `dependencies.json` pins reused
optimizer, model, rendering and measurement helpers for the staged driver.

## Published evidence

- `pilot/`: both 1,536-episode attempts, every completed generation, both gate
  decisions, and source snapshots for the original trainer/evaluators.
- `runs/`: all six 2,304-episode runs. Each generation stores candidate weights,
  per-condition outcomes, scores, selected winner, search distribution, exact
  conditions, RNG state and measured elapsed time. Final checkpoints are always
  the last training winners.
- `eval/`: deterministic gzip tables for both development screens, the 68-case
  main grid and 48-case shift battery for all nine controllers, and 72 separate
  causal-diagnostic trials. Identity files pin evaluation inputs and code.
- `records/`: 200 Hz arrays and metadata for the predeclared first-replicate
  replay comparisons, plus the first pilot's explicitly selected quiet failure
  and the final seed-1 controller on that same development condition. Those two
  records are not new held-out confirmation cases. The first backward replicate
  and first seed-3 quiet failure were added after evaluation to expose regressions;
  their selection is disclosed in the replay and protocol.
- `report/`: a portable self-contained HTML replay, learning curves, filmstrip,
  recomputed tables and contact-point diagnostics. The schematic shows measured
  body frames and geometry-derived sole silhouettes; it contains no upstream
  meshes and runs no browser physics.
- `tests.txt`, `validation.json`, `manifest.json`: regression results, artifact
  and physical auditing, selected exact resimulations and publication hashes.

Full-rate trajectories are published for the replay subset. Every other
evaluation row includes its complete condition and can be regenerated with the
frozen controller/model. Episode counts in the report describe training or
evaluation separately; replay generation, unit-test rollouts, restoration and
source validation are additional work, not independent statistical evidence.

## Record units and semantics

`time_s`, `qpos`, `qvel`, `xpos`, contact forces and tilt preserve measured
simulation samples. Joint order and model properties are in 02B's `model.json`;
the floating root quaternion is wxyz. `offset` is the combined joint-target
offset from HOME; `residual` is only the additional 02D or fixed-option component.
`target_cmd` is clipped before the command-delay buffer; `target_applied` is what
the BAM controller receives. `torque_nm` is the BAM torque command, not total
joint loading or electrical energy. The work field is an endpoint mechanical
work proxy, following 02C.

`features` holds the controller's five normalized IMU channels: projected
gravity x/y divided by 0.1, then gyro xyz in rad/s. It includes configured noise,
calibration error and observation delay. `coefficients` contains the two tanh
outputs for learned controllers. Contact, root velocity, COM, position and
disturbance metadata are recorder/evaluator fields, never learner inputs.

`foot_found` is geometric contact, not a hardware load-cell signal.
`clearance_m` uses full 3-D sole-hull geometry. Replay hull projections are
sampled to at most 24 points and rounded to 0.1 mm for display; they do not decide
validity. Contact-point diagnostics use the sole material-point Jacobian and
separate sliding from movement of an ankle frame. They are sampled geometric
diagnostics, not force-weighted contact impulses.

## Running and checking

Use the [repository commands](../../README.md#movement-lab-02d--learning-when-and-how-to-react)
from the repository root with the matching upstream environment. The staged
driver rejects a changed dependency inventory; individual runs additionally
reject changed configurations and numerical sources. Interrupted staging NPZs
are ignored until atomically promoted to complete generations. Completed jobs
are retained. The source snapshots explain why the original pilot's hashes
differ from the revised trainer and final evaluator.

`python -m movement_lab.validate02d --root results/02d --verify` verifies
published hashes without a model. The full audit and `--resimulate 9` require
the matching local compiled model and native BAM/MuJoCo environment. Do not
interpret a skipped or unavailable simulation check as validation.

The compiled model embeds upstream assets and is not bundled. On this machine,
fresh compilation failed the historical binary hash check and installed nothing;
the exact preserved binary was restored from the previous checkout. The cause
of the alternate compile hash was not established. A fresh machine therefore
needs that hash-matching model or a verified regeneration, plus the separately
installed upstream dependencies. No cross-platform bitwise-trajectory guarantee,
new GPU-backend equivalence claim, or hardware demonstration is made.

## Browser packaging and verification

The [hosted replay](https://movement-lab-02d-jstewdios.jstewdios.chatgpt.site)
is owner-private. `deployment.json` preserves the successful deployment/version
IDs and exact source commit, plus the failed first attempt. The portable HTML
in this repository remains available to reviewers without Site access.

The portable replay is 30,459,214 bytes. Static hosting rejected the first
publication because one file exceeded its 25 MiB limit; no version went live.
`movement_lab.package02d` losslessly gzip/base64-encodes the embedded JSON,
producing a self-contained 4,409,361-byte hosted page. Native browser
`DecompressionStream` restores the exact same payload. No samples are removed
and no numeric values are rounded by packaging. `report/hosting-package.json`
records both file hashes and the uncompressed payload hash.

```bash
python -m movement_lab.package02d --input results/02d/report/replay.html \
  --output /tmp/02d-hosted.html --metadata /tmp/02d-hosted-hashes.json
```

`browsercheck02d.js` was appended to QA copies after the application code
(inside the hosted copy's module scope). Both pages passed in Chrome 156:
46 manual recording choices, 13 presets × three cameras × three timeline
positions, and playback/measurement/outcome controls. The two browser-check
JSONs record the result; `report/browser-preview.png` is a visual check of the
first comparison at +0.30 s. These are interface checks, not extra trials.
