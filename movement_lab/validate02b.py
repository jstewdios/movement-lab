"""02B read-only validation of saved study outputs.

    python -m movement_lab.validate02b --root results/02b [--resimulate 24]

Checks (hard failures raise; nothing is rewritten except validation.json):
  * model.json hash vs the local compiled model (when present);
  * every main run: configuration, exact rollout budget, probe schedule, and that
    the saved final reflex equals the last snapshot;
  * held-out evaluation: complete controller x condition table, seeds disjoint
    from training and probe seeds, saved controllers identical to the runs'
    final reflexes, and summary aggregates recomputed from per-trial rows;
  * optional re-simulation of a random sample of held-out trials in fresh
    processes, requiring bitwise-identical outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from movement_lab import eval02b, learn02b, sim02b
from movement_lab.sim02b import Condition


def check_model(root, report):
    info = json.loads((root / 'model/model.json').read_text())
    mjb = root / 'model/model.mjb'
    if mjb.exists():
        assert hashlib.sha256(mjb.read_bytes()).hexdigest() == info['mjb_sha256'], 'model hash mismatch'
        report['model'] = 'hash verified'
    else:
        report['model'] = 'local model.mjb absent: hash not verifiable (regenerate with model02b)'
    return info


def check_runs(root, info, report):
    runs = {}
    for d in sorted((root / 'runs').iterdir()):
        if not d.is_dir():
            continue
        assert (d / 'final.json').exists(), f'incomplete run {d.name}'
        cfg = json.loads((d / 'config.json').read_text())
        rows = [json.loads(l) for l in (d / 'progress.jsonl').read_text().splitlines()]
        final = json.loads((d / 'final.json').read_text())
        z = np.load(d / 'thetas.npz')
        assert cfg['model_sha256'] == info['mjb_sha256'], f'{d.name}: model mismatch'
        assert f"{cfg['arm']}-{cfg['disturb']}-s{cfg['seed']}" == d.name, d.name
        train = [r for r in rows if r['kind'] == 'train']
        assert [r['it'] for r in train] == list(range(1, cfg['iterations'] + 1)), f'{d.name}: missing iterations'
        per_it = 2 * cfg['directions'] * cfg['conditions']
        expect = cfg['iterations'] * per_it + (160 if cfg['arm'] == 'random' else 0)
        assert final['rollouts'] == expect, f"{d.name}: budget {final['rollouts']} != {expect}"
        probes = [r['it'] for r in rows if r['kind'] == 'probe']
        want = [0] + [i for i in range(1, cfg['iterations'] + 1) if learn02b.probe_due(i) or i == cfg['iterations']]
        assert probes == want, f'{d.name}: probe schedule'
        assert np.array_equal(z['thetas'][-1], z['final']), f'{d.name}: final != last snapshot'
        assert len(z['thetas']) == cfg['iterations'] + 1 + (1 if cfg['arm'] == 'random' else 0)
        runs[d.name] = z['final']
    report['runs'] = {k: 'ok' for k in runs}
    return runs


def check_eval(root, runs, report, resimulate, model_dir):
    path = root / 'eval/final'
    rows = eval02b.read_results(path)
    summary = json.loads((path / 'summary.json').read_text())
    conds = [(s, c) for b in summary['batteries'] for s, c in eval02b.BATTERIES[b]()]
    ctrls = list(summary['controllers'])
    assert len(rows) == len(conds) * len(ctrls), 'incomplete evaluation table'
    for i, name in enumerate(ctrls):
        mine = rows[i * len(conds):(i + 1) * len(conds)]
        assert all(r['controller'] == name for r in mine)
        want = json.loads(json.dumps([(s, c.to_dict()) for s, c in conds]))  # tuples -> lists, as saved
        assert [[r['set'], r['condition']] for r in mine] == want, f'{name}: conditions'
        th = np.array(summary['thetas'][name])
        if name in runs:
            assert np.array_equal(th, runs[name]), f'{name}: evaluated reflex differs from run final'
        elif name == 'home':
            assert not th.any()
    seeds = {r['condition']['seed'] for r in rows}
    assert min(seeds) >= eval02b.EVAL_SEED_BASE, 'evaluation seed below held-out namespace'
    probe = {c.seed for c in learn02b.probe_conditions()}
    assert not (seeds & probe)
    re = eval02b.aggregate(rows)
    assert json.loads(json.dumps(re)) == summary['results'], 'summary aggregates do not recompute'
    report['eval'] = dict(rows=len(rows), controllers=len(ctrls), conditions=len(conds), aggregates='recomputed equal')
    if resimulate:
        sim = sim02b.Sim(model_dir)
        rnd = random.Random(2026)
        sample = rnd.sample(rows, resimulate)
        bad = []
        for r in sample:
            th = np.array(summary['thetas'][r['controller']])
            s, _ = sim02b.rollout(sim, th, Condition(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in r['condition'].items()}),
                                  terminate=True, grace_s=0.0)
            if (s['survived'], s['fall_time_s'], s['fitness']) != (r['survived'], r['fall_time_s'], r['fitness']):
                bad.append((r['controller'], r['set'], r['condition']['seed']))
        assert not bad, f'non-reproducible held-out trials: {bad}'
        report['eval']['resimulated_identical'] = len(sample)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('results/02b'))
    p.add_argument('--resimulate', type=int, default=0)
    args = p.parse_args(argv)
    report = {}
    info = check_model(args.root, report)
    runs = check_runs(args.root, info, report)
    if eval02b.read_results(args.root / 'eval/final') is not None:
        check_eval(args.root, runs, report, args.resimulate, args.root / 'model')
    (args.root / 'validation.json').write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    print('PASS')


if __name__ == '__main__':
    main()
