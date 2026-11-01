# Movement Lab 02A — Learning to Explore

The explorer and physical archive are implemented. This bounded run is a feasibility result; it does not establish that adaptation improves discovery efficiency. No motor policy was trained and no discovery has independent confirmation.

Descriptive counts (no uncertainty estimate at this sample size): `{"adaptive": {"evaluated": 2, "discoveries": 1, "physically_invalid": 0, "valid_unproductive": 1, "discovery_curve": [1, 1]}, "control": {"evaluated": 2, "discoveries": 2, "physically_invalid": 0, "valid_unproductive": 0, "discovery_curve": [1, 2]}}`.

Adaptive retained 1 distinct outcome(s), versus 2 for uniform control. Adaptive second-proposal novelty: 0.42492323875427246; distinctness threshold: 1. Per-proposal discovery indicators: adaptive [1, 0], control [1, 1]. These two trials per arm and one paired search seed cannot establish an algorithm-level advantage. In the initial pilot, the adaptive second command was valid but redundant; experience changed the proposal without improving discovery.

Execution/validation wall time: 364.61 s. Stop status: **complete**. Ten-minute hard subprocess deadline; plotting and hashing occur after GPU work ends. 01C measured 510.8 s for five ten-second trials, motivating a shared baseline plus two proposals per arm. Missing or interrupted trials are retained as attempts and never counted as successes. Unequal completed budgets preclude a matched comparison.

| Trial | Head/flex coefficients | Measured outcome | Discovery in own archive | 01C return classification |
|---|---|---|---|---|
| adaptive-0 | [-1.0, 0.0] | measurable stable outcome | True | stable transition with partial return |
| control-0 | [-1.0, 0.0] | measurable stable outcome | True | stable transition with partial return |
| control-1 | [0.5, 1.0] | measurable stable outcome | True | stable changed configuration |
| adaptive-1 | [-1.0, -0.5] | measurable stable outcome | False | stable transition with partial return |

## Action, measurement, and adaptation

The action space has 24 coefficient pairs: {-1, -.5, 0, .5, 1}² except (0,0). Head yaw is at most ±.08 rad; bilateral hip/knee/ankle residuals at most .03/.05/.02 rad with 01C signs. Both coefficients may be nonzero. Timing is fixed: start 2 s, cosine rise .6 s, hold 1 s, cosine return 1.4 s; zero residual from 5–10 s. Maximum residual slew is .08π/(2×.6) = .210 rad/s. HOME + unchanged checkpoint-99 actor + residual uses existing BAM voltage, sag, torque, friction and four-substep delay. Composed targets retain .02 rad joint-limit margin and wrapper action bounds. Existing audits bound measured actuator effort at .9601 Nm and measured joint-limit overshoot at .03 rad; safety stops include >100 N foot force, >12° root tilt, nonfoot contact, >3 mm penetration and generalized speed >100. Stops take effect at the next control boundary, up to 20 ms; these are simulator diagnostic limits, not hardware ratings. This adds no within-trial learning.

Descriptor: signed phase means over hold [2.6,3.6), return [4,5), late [7,8), final [9.5,10.01) of all 14 actual joint positions, whole-body COM xyz, and three root rotation-matrix entries, each relative to the matched baseline. Normalization scales are .005 rad, .5 mm and sin(.5°). Max normalized coordinate difference ≥1 marks a distinct physical outcome. A response of that size versus standing, whole-record physical validity, and stable hold/final windows are mandatory before discovery acceptance. Contact trajectories gate acceptance rather than add novelty. No command-space bin is counted as physical diversity. These thresholds are exploratory and lack an independent noise/repeat calibration; phase averages suppress high-frequency artifacts but do not establish robustness.

The adaptive explorer uses its own earlier records only: 25% uniform exploration; otherwise nearest-observation transfer scores eligible neighborhoods by capped novelty minus distance from a half-unit coefficient mutation. Failed neighborhoods score distance minus two, encouraging escape. Ties follow declared action order. This deliberately small heuristic has no calibrated predictive model. The control samples uniformly without replacement and never uses validity or descriptors to choose. Each arm uses search seed 42, simulator seed 42, identical action space, controller, baseline and gates. The first proposal is matched. Search adaptation is proposal selection between trials, not policy learning.

Discovery counts are relative to each initially empty arm archive, not claims of behavior never seen in earlier studies. The matched first proposal revisits negative head yaw already measured in 01C. Archive JSON retains every classified attempt, including unsuccessful physical outcomes, descriptors, novelty, discovery flags and raw hashes. Attempts JSON separately retains launch/crash/timeout evidence. Stable reconfiguration and partial return use the independent 01C classifier; neither implies full return or reversible behavior. A candidate is never marked independently reproduced.

## What experience changed

Actual proposals and their decision paths: `[{"id": "baseline", "action": [0, 0], "proposal_reason": "mandatory standing control", "status": "validated record"}, {"id": "adaptive-0", "action": [-1.0, 0.0], "proposal_reason": "uniform without replacement", "status": "validated record"}, {"id": "control-0", "action": [-1.0, 0.0], "proposal_reason": "uniform without replacement", "status": "validated record"}, {"id": "control-1", "action": [0.5, 1.0], "proposal_reason": "uniform without replacement", "status": "validated record"}, {"id": "adaptive-1", "action": [-1.0, -0.5], "proposal_reason": "nearest valid neighborhood mutation; failed-region distance; novelty tie score", "status": "validated record"}]`. Only proposals explicitly labeled nearest valid neighborhood use outcome experience. Uniform proposals do not. The frozen standing actor, action envelope, timing, validity criteria and descriptor thresholds never update. A changed proposal demonstrates adaptation mechanics only; it cannot establish improved discovery without a sufficiently large matched control comparison. No within-trial learning was added.

## Evaluation design and leakage control

Primary endpoint: cumulative distinct validated outcomes at matched proposal counts. Secondary: nearest-member novelty, physically invalid fraction, valid but unproductive fraction, and per-proposal discovery yield in early versus late budget halves. Report proportions descriptively at this size. A subsequent adequately funded experiment should preregister ≥5 paired search seeds, a fixed equal trial count per arm, and independently evaluate the first two discoveries per arm with three fresh simulator seeds, requiring validity and descriptor distance <1 from the discovery. Confirmation outcomes must be written to a separate directory and never passed to Explorer.observe. No confirmation allocation or hidden confirmation data exists in this pilot. Historical records informed action/measurement design and are development data, not held-out evidence.

Offline proof revalidated four existing 01C records and exercised archive acceptance; it is not an adaptive-versus-control result. Unit tests demonstrate changed proposals under changed experience, outcome-independent control proposals, duplicate-outcome rejection, failure retention and incomplete-baseline rejection. This proves mechanics, not useful learning. There is no claim of walking, crouching, jumping, unloading or any predetermined skill.

## Evidence and reproduction

[Physical diversity](outcomes.png) · [Representative full-rate trajectories and discovery curves](trajectories.png) · [Archive](archive.json) · [Attempts](attempts.json) · [Plan](PLAN.json) · [Cost](cost.json).

The baseline is compared exactly to the historical 01C arrays, model and resolved configuration by report02a. Per-trial 01C checks retain hash/checkpoint, CPU geometry/COM/contact, controller composition, exact initial/precommand prefix, pulse, target and actuator mapping audits. The 3 mm penetration threshold, groundcontact and 1 mm sole-clearance reset remain unchanged. Baseline incompleteness or integrity failures halt the study; physical safety stops remain unsuccessful observations. The executed adapter is preserved in this output root. After the pilot its CLI was restricted to out/coefficients/seed so timing and amplitude overrides cannot bypass the declared envelope; the pilot already used those fixed defaults. The final driver also performs the historical baseline comparison before departures and returns a nonzero status on hard failure; the executed snapshot is preserved, and the pilot historical match was independently checked during the run. Sources are snapshotted and full local files hashed; missing raw NPZ/MJB files cannot be validated from hashes alone. Prior studies and upstream Microduck RL remain unchanged.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m unittest discover -s tests -v
$PY -m movement_lab.explore02a --recorded results/01c/pilot-gpu --out results/02a/offline-new
$PY -m movement_lab.study02a --out results/02a/pilot-new --budget 600
$PY -m movement_lab.report02a results/02a/pilot-new
```

Use new output directories. CUDA and the separately installed upstream environment/checkpoint are required for simulation. No long training, upstream changes, commits or publication operations are performed.
