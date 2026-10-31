"""Bounded feasibility pilot. New output roots only; immutable earlier evidence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from movement_lab.explore02a import Explorer, measure, plot
from movement_lab.evaluate import sha


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--budget', type=float, default=600.)
    args = p.parse_args()
    if not 0 < args.budget <= 600: p.error('budget must be in (0,600]')
    out = args.out; out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic(); deadline = start+args.budget
    plan = dict(budget_wall_s=args.budget, trials_per_arm=2, search_seed=42, simulator_seed=42,
                order=['baseline', 'adaptive-0', 'control-0', 'control-1', 'adaptive-1'],
                independent_confirmation='not allocated; no reproducibility claim',
                action_space='24 head/flex coefficient pairs from {-1,-.5,0,.5,1} squared except zero',
                initialization='checkpoint99, groundcontact, root z .1181823627, 1mm clearance',
                validity='unchanged 01C/01B, 3mm penetration, stable hold and final',
                timing='2s start; .6s rise, 1s hold, 1.4s return, 10s horizon')
    (out/'PLAN.json').write_text(json.dumps(plan, indent=2))
    for name in ('study02a.py', 'explore02a.py', 'residual02a.py'):
        (out/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    arms = {name: Explorer(42, name == 'adaptive') for name in ('adaptive', 'control')}
    attempts = []; status = 'complete'
    def save():
        (out/'attempts.json').write_text(json.dumps(attempts, indent=2))
        (out/'archive.json').write_text(json.dumps({k: v.history for k, v in arms.items()}, indent=2))
    try:
        for name in plan['order']:
            if deadline-time.monotonic() < 100:
                status = 'budget: insufficient launch reserve'; break
            arm = None if name == 'baseline' else name.split('-')[0]
            action, reason = ((0, 0), 'mandatory standing control') if arm is None else arms[arm].propose()
            row = dict(id=name, arm=arm, action=list(action), proposal_reason=reason, status='launched')
            attempts.append(row); save()
            trial = out/name
            cmd = [sys.executable, '-m', 'movement_lab.transition01c', '--out', str(trial), '--sequence', 'zero'] if arm is None else [sys.executable, '-m', 'movement_lab.residual02a', '--out', str(trial), '--coefficients', *map(str, action)]
            tick = time.monotonic()
            with (out/(name+'.log')).open('w') as log:
                result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, timeout=max(.01, deadline-time.monotonic()))
            row.update(returncode=result.returncode, execution_wall_s=time.monotonic()-tick)
            if result.returncode: raise RuntimeError('execution failed: '+name)
            # The existing full audit runs in a deadline-bounded subprocess.
            code = "import json; from pathlib import Path; from movement_lab.study01c import check; print(json.dumps(check(Path(%r),Path(%r))))" % (str(trial), str(out/'baseline'))
            audit = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True,
                                   timeout=max(.01, deadline-time.monotonic()))
            (out/(name+'-validation.log')).write_text(audit.stdout+audit.stderr)
            if audit.returncode: raise RuntimeError('hard validation failure: '+name)
            row['validation'] = json.loads(audit.stdout.splitlines()[-1]); row['status'] = 'validated record'
            meta = next(trial.glob('*/metadata.json')); m = json.loads(meta.read_text())
            a = dict(np.load(meta.parent/'trial.npz'))
            baseline = dict(np.load(out/'baseline/zero/trial.npz'))
            if arm is None:
                assert row['validation']['physical_valid'], 'invalid baseline'
            else:
                r = measure(a, baseline, m)
                row['outcome'] = arms[arm].observe(dict(id=name, action=action, trajectory_sha256=sha(meta.parent/'trial.npz'), **r))
            save()
    except subprocess.TimeoutExpired:
        status = 'hard deadline reached; incomplete attempt retained'
        if attempts: attempts[-1]['status'] = 'deadline interrupted'
    except Exception as exc:
        status = 'hard failure: '+repr(exc)
        if attempts: attempts[-1]['status'] = status
    finally:
        save()
        elapsed = time.monotonic()-start
        (out/'cost.json').write_text(json.dumps(dict(wall_s=elapsed, status=status, budget_s=args.budget), indent=2))
        plot([r for e in arms.values() for r in e.history], out)
        (out/'manifest.json').write_text(json.dumps({str(f.relative_to(out)): sha(f) for f in out.rglob('*') if f.is_file() and f.name != 'manifest.json'}, indent=2))
        print(status, elapsed, flush=True)

if __name__ == '__main__': main()
