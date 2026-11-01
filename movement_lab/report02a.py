"""Offline 02A report and measured trajectory visualizations."""
import argparse
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha
from movement_lab.explore02a import ACTIONS, plot
from movement_lab.historical02a import record_historical, describe_historical
from movement_lab.study01c import check


def main():
    p = argparse.ArgumentParser(); p.add_argument('root', type=Path); args = p.parse_args(); root = args.root
    attempts = json.loads((root/'attempts.json').read_text())
    archives = json.loads((root/'archive.json').read_text())
    cost = json.loads((root/'cost.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    control = root/'baseline/zero/trial.npz'
    rows = []
    baseline_validation = check(root/'baseline', root/'baseline')
    assert baseline_validation['physical_valid'], 'invalid new 02A baseline'
    c = dict(np.load(control))
    baseline_audit = record_historical(root/'baseline', root/'historical-baseline-audit.json')
    historical_note = describe_historical(baseline_audit)
    if control.exists():
        for r in attempts:
            if not r.get('arm') or 'outcome' not in r: continue
            checked = check(root/r['id'], root/'baseline')
            assert checked == r['validation'], ('new 02A validation mismatch', r['id'])
            meta = next((root/r['id']).glob('*/metadata.json')); m = json.loads(meta.read_text())
            h, f = r['action']; assert (h, f) in ACTIONS
            expected = {'head_yaw': .08*h, 'left_hip_pitch': -.03*f, 'right_hip_pitch': .03*f,
                        'left_knee': .05*f, 'right_knee': -.05*f, 'left_ankle': -.02*f, 'right_ankle': .02*f}
            assert m['amplitudes_rad'] == expected, 'adapter action mismatch'
            a = dict(np.load(meta.parent/'trial.npz')); n = len(a['time_s']); t = a['time_s']
            delta = a['joint_position_rad']-c['joint_position_rad'][:n]
            idx = m['joint_names'].index('head_yaw')
            axes[0,0].plot(t, delta[:,idx], label=r['id'])
            axes[0,1].plot(t, (a['com_position_w_m'][:,0]-c['com_position_w_m'][:n,0])*1000, label=r['id'])
            axes[1,0].plot(t, a['native_min_contact_distance_m']*1000, label=r['id'])
            row = r['outcome']; rows.append(f"| {r['id']} | {r['action']} | {row['classification']} | {row['discovery']} | {r['validation']['classification']} |")
    for name, history in archives.items():
        axes[1,1].step(range(1,len(history)+1), np.cumsum([r['discovery'] for r in history]), where='post', label=name)
    axes[0,0].set(xlabel='Time (s)', ylabel='Actual head yaw minus standing (rad)')
    axes[0,1].set(xlabel='Time (s)', ylabel='Actual COM x minus standing (mm)')
    axes[1,0].set(xlabel='Time (s)', ylabel='Minimum contact distance (mm)')
    axes[1,0].axhline(-3, color='red', linestyle='--')
    axes[1,1].set(xlabel='Evaluated proposal count', ylabel='Distinct validated outcomes', xticks=[1,2])
    for ax in axes.flat:
        if ax.lines: ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'trajectories.png'); plt.close(fig)
    plot([r for v in archives.values() for r in v], root)
    statistics = {}
    for name, history in archives.items():
        n = len(history)
        statistics[name] = dict(evaluated=n, discoveries=sum(r['discovery'] for r in history), physically_invalid=sum(not r['physical_valid'] for r in history), valid_unproductive=sum(r['physical_valid'] and not r['discovery'] for r in history), discovery_curve=np.cumsum([r['discovery'] for r in history]).tolist())
    (root/'metrics.json').write_text(json.dumps(statistics, indent=2))
    proposals = [{k:r[k] for k in ('id','action','proposal_reason','status')} for r in attempts]
    (root/'proposals.json').write_text(json.dumps(proposals, indent=2))
    result_note = 'Completed budgets are insufficient for the planned matched comparison.'
    if all(len(v) == 2 for v in archives.values()):
        adaptive = archives['adaptive']; control_rows = archives['control']
        result_note = (f"Adaptive retained {statistics['adaptive']['discoveries']} distinct outcome(s), versus "
                       f"{statistics['control']['discoveries']} for uniform control. "
                       f"Adaptive second-proposal novelty: {adaptive[1]['novelty']}; distinctness threshold: 1. "
                       f"Per-proposal discovery indicators: adaptive {[int(r['discovery']) for r in adaptive]}, "
                       f"control {[int(r['discovery']) for r in control_rows]}. "
                       "These two trials per arm and one paired search seed cannot establish an algorithm-level advantage. "
                       "No general discovery-efficiency conclusion is supported by this sample.")
    (root/'report-validation.json').write_text(json.dumps(dict(baseline=baseline_validation, historical=baseline_audit, new_records='strict per-trial 01C validation against new 02A baseline'), indent=2)+'\n')
    text = f'''# Movement Lab 02A — Learning to Explore

The explorer and physical archive are implemented. This bounded run is a feasibility result; it does not establish that adaptation improves discovery efficiency. No motor policy was trained and no discovery has independent confirmation.

Descriptive counts (no uncertainty estimate at this sample size): `{json.dumps(statistics)}`.

{result_note}

{historical_note}

Execution/validation wall time: {cost['wall_s']:.2f} s. Stop status: **{cost['status']}**. Ten-minute hard subprocess deadline; plotting and hashing occur after GPU work ends. 01C measured 510.8 s for five ten-second trials, motivating a shared baseline plus two proposals per arm. Missing or interrupted trials are retained as attempts and never counted as successes. Unequal completed budgets preclude a matched comparison.

| Trial | Head/flex coefficients | Measured outcome | Discovery in own archive | 01C return classification |
|---|---|---|---|---|
'''+'\n'.join(rows)+'''

## Action, measurement, and adaptation

The action space has 24 coefficient pairs: {-1, -.5, 0, .5, 1}² except (0,0). Head yaw is at most ±.08 rad; bilateral hip/knee/ankle residuals at most .03/.05/.02 rad with 01C signs. Both coefficients may be nonzero. Timing is fixed: start 2 s, cosine rise .6 s, hold 1 s, cosine return 1.4 s; zero residual from 5–10 s. Maximum residual slew is .08π/(2×.6) = .210 rad/s. HOME + unchanged checkpoint-99 actor + residual uses existing BAM voltage, sag, torque, friction and four-substep delay. Composed targets retain .02 rad joint-limit margin and wrapper action bounds. Existing audits bound measured actuator effort at .9601 Nm and measured joint-limit overshoot at .03 rad; safety stops include >100 N foot force, >12° root tilt, nonfoot contact, >3 mm penetration and generalized speed >100. Stops take effect at the next control boundary, up to 20 ms; these are simulator diagnostic limits, not hardware ratings. This adds no within-trial learning.

Descriptor: signed phase means over hold [2.6,3.6), return [4,5), late [7,8), final [9.5,10.01) of all 14 actual joint positions, whole-body COM xyz, and three root rotation-matrix entries, each relative to the matched baseline. Normalization scales are .005 rad, .5 mm and sin(.5°). Max normalized coordinate difference ≥1 marks a distinct physical outcome. A response of that size versus standing, whole-record physical validity, and stable hold/final windows are mandatory before discovery acceptance. Contact trajectories gate acceptance rather than add novelty. No command-space bin is counted as physical diversity. These thresholds are exploratory and lack an independent noise/repeat calibration; phase averages suppress high-frequency artifacts but do not establish robustness.

The adaptive explorer uses its own earlier records only: 25% uniform exploration; otherwise nearest-observation transfer scores eligible neighborhoods by capped novelty minus distance from a half-unit coefficient mutation. Failed neighborhoods score distance minus two, encouraging escape. Ties follow declared action order. This deliberately small heuristic has no calibrated predictive model. The control samples uniformly without replacement and never uses validity or descriptors to choose. Each arm uses search seed 42, simulator seed 42, identical action space, controller, baseline and gates. The first proposal is matched. Search adaptation is proposal selection between trials, not policy learning.

Discovery counts are relative to each initially empty arm archive, not claims of behavior never seen in earlier studies. The matched first proposal revisits negative head yaw already measured in 01C. Archive JSON retains every classified attempt, including unsuccessful physical outcomes, descriptors, novelty, discovery flags and raw hashes. Attempts JSON separately retains launch/crash/timeout evidence. Stable reconfiguration and partial return use the independent 01C classifier; neither implies full return or reversible behavior. A candidate is never marked independently reproduced.

## What experience changed

Actual proposals and their decision paths: `PROPOSALS_PLACEHOLDER`. Only proposals explicitly labeled nearest valid neighborhood use outcome experience. Uniform proposals do not. The frozen standing actor, action envelope, timing, validity criteria and descriptor thresholds never update. A changed proposal demonstrates adaptation mechanics only; it cannot establish improved discovery without a sufficiently large matched control comparison. No within-trial learning was added.

## Evaluation design and leakage control

Primary endpoint: cumulative distinct validated outcomes at matched proposal counts. Secondary: nearest-member novelty, physically invalid fraction, valid but unproductive fraction, and per-proposal discovery yield in early versus late budget halves. Report proportions descriptively at this size. A subsequent adequately funded experiment should preregister ≥5 paired search seeds, a fixed equal trial count per arm, and independently evaluate the first two discoveries per arm with three fresh simulator seeds, requiring validity and descriptor distance <1 from the discovery. Confirmation outcomes must be written to a separate directory and never passed to Explorer.observe. No confirmation allocation or hidden confirmation data exists in this pilot. Historical records informed action/measurement design and are development data, not held-out evidence.

Offline proof revalidated four existing 01C records and exercised archive acceptance; it is not an adaptive-versus-control result. Unit tests demonstrate changed proposals under changed experience, outcome-independent control proposals, duplicate-outcome rejection, failure retention and incomplete-baseline rejection. This proves mechanics, not useful learning. There is no claim of walking, crouching, jumping, unloading or any predetermined skill.

## Evidence and reproduction

[Physical diversity](outcomes.png) · [Representative full-rate trajectories and discovery curves](trajectories.png) · [Archive](archive.json) · [Attempts](attempts.json) · [Plan](PLAN.json) · [Cost](cost.json).

Historical evidence status and available comparisons are recorded above and in historical-baseline-audit.json and report-validation.json. Missing or partial history is never counted as historical validation; every available artifact must match. New 02A records are checked strictly against their own baseline regardless of historical availability. Per-trial 01C checks retain hash/checkpoint, CPU geometry/COM/contact, controller composition, exact initial/precommand prefix, pulse, target and actuator mapping audits. The 3 mm penetration threshold, groundcontact and 1 mm sole-clearance reset remain unchanged. Baseline incompleteness or integrity failures halt the study; physical safety stops remain unsuccessful observations. The executed adapter is preserved in this output root. After the pilot its CLI was restricted to out/coefficients/seed so timing and amplitude overrides cannot bypass the declared envelope; the pilot already used those fixed defaults. The final driver also performs the historical baseline comparison before departures and returns a nonzero status on hard failure; the executed snapshot is preserved, and the pilot historical match was independently checked during the run. Sources are snapshotted and full local files hashed; missing raw NPZ/MJB files cannot be validated from hashes alone. Prior studies and upstream Microduck RL remain unchanged.

```bash
PY=/home/james/microduck_rl/.venv/bin/python
$PY -m unittest discover -s tests -v
$PY -m movement_lab.explore02a --recorded results/01c/pilot-gpu --out results/02a/offline-new
$PY -m movement_lab.study02a --out results/02a/pilot-new --budget 600
$PY -m movement_lab.report02a results/02a/pilot-new
```

Use new output directories. CUDA and the separately installed upstream environment/checkpoint are required for simulation. No long training, upstream changes, commits or publication operations are performed.
'''
    text = text.replace('PROPOSALS_PLACEHOLDER', json.dumps(proposals))
    (root/'REPORT.md').write_text(text)
    (root/'report02a.py').write_bytes(Path(__file__).read_bytes())
    (root/'manifest.json').write_text(json.dumps({str(f.relative_to(root)):sha(f) for f in root.rglob('*') if f.is_file() and f.name!='manifest.json'}, indent=2))

if __name__ == '__main__': main()
