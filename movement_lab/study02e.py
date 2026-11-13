"""Staged 02E driver for the learning branch and confirmation (see results/02e/PROTOCOL.md).

    python -m movement_lab.study02e reproduce-training   # learn02e == 02D history-s72 generation 1
    python -m movement_lab.study02e pilot-learning       # three 1,536-episode pilots (1D)
    python -m movement_lab.study02e evaluate-pilot       # battery D, front-end none
    python -m movement_lab.study02e gate-learning
    python -m movement_lab.study02e train                # only arms that passed 1D
    python -m movement_lab.study02e confirm              # held-out battery H
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from movement_lab.control02d import Design
from movement_lab import pool02e

ROOT = Path('results/02e')
PILOTS = (('history-rand-s72', 'history', 72, 4.0), ('memoryless-rand-s72', 'memoryless', 72, 4.0),
          ('memoryless-s72', 'memoryless', 72, 0.0))
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')


def learn(out, arm, seed, iterations, randomize, workers=2, wait=True):
    args = [sys.executable, '-m', 'movement_lab.learn02e', '--out', str(out), '--arm', arm, '--seed', str(seed),
            '--iterations', str(iterations), '--randomize-deg', str(randomize), '--workers', str(workers), '--budget', '600']
    if out.exists():
        if (out / 'final.json').exists():
            cfg = json.loads((out / 'config.json').read_text())
            if (cfg['arm'], cfg['seed'], cfg['iterations'], cfg['randomize_deg']) != (arm, seed, iterations, randomize):
                raise SystemExit(f'completed run configuration mismatch: {out}')
            print('retained completed run', out, flush=True)
            return None
        args.append('--resume')
    proc = subprocess.Popen(args, env=ENV)
    if wait:
        proc.wait()
    return proc


def run_until_complete(jobs, workers=2, concurrent=1, guard=None):
    """Run at most ``concurrent`` training jobs at a time, ``workers`` each; relaunch (resume) any
    that stopped at the 600 s launch cap. The guard watches the whole process tree; on a breach or
    a stop signal every descendant is terminated (generations are atomic, so only work in progress is lost)."""
    guard = guard or pool02e.Guard()
    while True:
        pending = [j for j in jobs if not (j[0] / 'final.json').exists()]
        if not pending:
            return
        for i in range(0, len(pending), concurrent):
            group = pending[i:i + concurrent]
            procs = [p for p in (learn(*j, workers=workers, wait=False) for j in group) if p is not None]
            try:
                while any(p.poll() is None for p in procs):
                    guard.check(pool02e.descendants(os.getpid()), runs=[j[0].name for j in group])
                    time.sleep(guard.poll_s)
            except BaseException:
                pool02e.terminate_tree(pool02e.descendants(os.getpid()))
                raise
            if any(p.returncode != 0 for p in procs):
                raise SystemExit('training subprocess failed')


def load_controller(path):
    cfg = json.loads((path / 'config.json').read_text())
    vec = np.load(path / 'final.npz')['best']
    return (path.name, vec.tolist(), cfg['design'], 'learned')


def reproduce_training(workers=1):
    from movement_lab.learn02e import main as learn_main
    out = ROOT / 'learning/reproduce-history-s72-g1'
    if not out.exists():
        learn_main(['--out', str(out), '--arm', 'history', '--seed', '72', '--iterations', '1', '--workers', str(workers)])
    a, b = np.load(out / 'generation-001.npz'), np.load(Path('results/02d/pilot/history-s72/generation-001.npz'))
    result = {k: bool(np.array_equal(a[k], b[k])) for k in ('candidates', 'scores', 'outcomes', 'best', 'mean', 'std')}
    sa, sb = json.loads(str(a['state'])), json.loads(str(b['state']))
    result['rng_state'] = sa['rng'] == sb['rng']
    ca, cb = json.loads(str(a['conditions'])), json.loads(str(b['conditions']))
    result['conditions_02d_fields'] = all({k: x[k] for k in y} == y for x, y in zip(ca, cb)) and len(ca) == len(cb)
    result['passed'] = all(result.values())
    (ROOT / 'learning/reproduce.json').write_text(json.dumps(result, indent=1) + '\n', newline='\n')
    print(json.dumps(result, indent=1))
    if not result['passed']:
        raise SystemExit('learn02e does not reproduce learn02d')


def pilot_controllers():
    cs = [load_controller(Path('results/02d/pilot/history-s72'))]
    cs[0] = ('history-s72',) + cs[0][1:]
    cs += [load_controller(ROOT / 'learning' / name) for name, *_ in PILOTS]
    return cs


def gate_learning():
    from movement_lab.diag02e import paired, MISALIGNED_D
    from movement_lab.eval02c import read_rows
    dest = ROOT / 'pilot/learning'
    rows = {p.name[:-9]: read_rows(p) for p in dest.glob('*.jsonl.gz')}
    result = {}
    for arm, control in (('history-rand-s72', 'history-s72'), ('memoryless-rand-s72', 'memoryless-s72')):
        base, new = rows[control], rows[arm]
        g, l = paired(base, new, MISALIGNED_D)
        nominal = lambda kind: paired([r for r in base if r['set'] == kind], [r for r in new if r['set'] == kind], ('nominal',))
        rec = lambda rs: sum(r['recovered_200hz'] for r in rs if r['setting'] == 'nominal')
        quiet_l, back_l = nominal('quiet')[1], nominal('back-.08')[1]
        passed = g - l >= 10 and quiet_l == 0 and back_l == 0 and rec(new) >= rec(base) - 1
        result[arm] = dict(passed=passed, control=control, misaligned_gains=g, misaligned_losses=l, misaligned_net=g - l,
                           nominal_quiet_losses=quiet_l, nominal_backward_losses=back_l,
                           nominal_recovered=[rec(base), rec(new)], slope=paired(base, new, ('slope-2', 'slope+2')),
                           misaligned_recovered=[sum(r['recovered_200hz'] for r in rs if r['setting'] in MISALIGNED_D) for rs in (base, new)])
    (ROOT / 'pilot/gate-learning.json').write_text(json.dumps(result, indent=1) + '\n', newline='\n')
    print(json.dumps(result, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('phase', choices=('reproduce-training', 'pilot-learning', 'evaluate-pilot', 'gate-learning', 'train', 'confirm'))
    pool02e.add_arguments(p)   # --workers (1..4), --tasks-per-worker, --max-worker-mb, --min-available-mb, --memory-log
    p.add_argument('--concurrent-runs', type=int, default=1, help='training runs at once; --workers is split between them')
    a = p.parse_args(argv)
    guard = pool02e.guard_from(a, a.phase)
    if not 1 <= a.concurrent_runs <= a.workers:
        p.error('--concurrent-runs must be 1..--workers')
    from movement_lab import diag02e as D
    with pool02e.cleanup_on_signals():
        run_phase(a, D, guard)
    print(json.dumps(dict(memory=guard.summary())), flush=True)


def run_phase(a, D, guard):
    per_run = a.workers // a.concurrent_runs
    battery = lambda dest, configs, conds: _bounded_battery(D, guard, a.tasks_per_worker, dest, configs, conds, a.workers)
    if a.phase == 'reproduce-training':
        reproduce_training(a.workers)
    elif a.phase == 'pilot-learning':
        if not json.loads((ROOT / 'learning/reproduce.json').read_text())['passed']:
            raise SystemExit('training reproduction gate did not pass')
        run_until_complete([(ROOT / 'learning' / name, arm, seed, 8, rnd) for name, arm, seed, rnd in PILOTS],
                           per_run, a.concurrent_runs, guard)
    elif a.phase == 'evaluate-pilot':
        conds = D.battery(D.SETTINGS_D, D.KINDS, 7_200_000_000)
        battery(ROOT / 'pilot/learning', [(c[0], c, None) for c in pilot_controllers()], conds)
    elif a.phase == 'gate-learning':
        gate_learning()
    elif a.phase == 'train':
        gate = json.loads((ROOT / 'pilot/gate-learning.json').read_text())
        arms = [arm for arm, name in (('history', 'history-rand-s72'), ('memoryless', 'memoryless-rand-s72')) if gate[name]['passed']]
        if not arms:
            raise SystemExit('no arm passed the declared 1D gate; no confirmation training')
        run_until_complete([(ROOT / 'runs' / f'{arm}-rand-s{seed}', arm, seed, 12, 4.0) for arm in arms for seed in (1, 2, 3)],
                           per_run, a.concurrent_runs, guard)
    else:
        conds = D.battery(D.SETTINGS_H, D.KINDS_H, 8_000_000_000, reps=3, onset=(2.2, 3.4))
        cs = D.controllers(['reflex', *[f'{x}-s{s}' for x in ('history', 'memoryless') for s in (1, 2, 3)]])
        front = json.loads((ROOT / 'pilot/gate-frontend.json').read_text())['decision']
        from movement_lab.estimate02e import Settings
        prior = json.loads((ROOT / 'diagnostic/prior.json').read_text())['prior_gravity']
        fes = dict(static=Settings(kind='static').to_dict(), prior=Settings(kind='prior', prior_gravity=tuple(prior)).to_dict())
        configs = [(f'{c[0]}__none', c, None) for c in cs]
        configs += [(f'{c[0]}__{f}', c, fes[f]) for f, ok in front.items() if ok for c in cs]
        trained = sorted((ROOT / 'runs').glob('*-rand-s*')) if (ROOT / 'runs').exists() else []
        configs += [(f'{path.name}__none', load_controller(path), None) for path in trained if (path / 'final.json').exists()]
        battery(ROOT / 'confirm', configs, conds)


def _bounded_battery(D, guard, tasks_per_worker, dest, configs, conds, workers):
    """diag02e.run_battery (unchanged identity and tasks) with bounded, guarded worker pools."""
    with pool02e.bounded(D, tasks_per_worker, guard):
        D.run_battery(dest, configs, conds, workers)


if __name__ == '__main__':
    main()
