"""Read-only artifact audit, plus optional deterministic held-out resimulation."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from movement_lab.eval02c import controllers, grid, shifts, read_rows, init_worker, evaluate
from movement_lab.learn02c import completed_checkpoints, conditions, update_distribution
from movement_lab.sim02b import Condition, DEFAULT_MODEL
from movement_lab.sim02c import Design


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def validate(root, resimulate=0):
    result = dict(checks=[], errors=[], resimulated=0)
    def check(ok, message):
        result['checks' if ok else 'errors'].append(message)
    all_train = set()
    expected_runs = {f'{arm}-s{s}' for arm in ('cem', 'random') for s in (1, 2, 3)}
    check({p.name for p in (root / 'runs').iterdir() if p.is_dir()} == expected_runs, 'all six confirmation runs')
    model_info = json.loads((DEFAULT_MODEL / 'model.json').read_text())
    if (DEFAULT_MODEL / 'model.mjb').exists():
        check(sha(DEFAULT_MODEL / 'model.mjb') == model_info['mjb_sha256'], 'local compiled model hash')
    for run in sorted((root / 'runs').iterdir()):
        cfg = json.loads((run / 'config.json').read_text())
        check(cfg['model_sha256'] == model_info['mjb_sha256'], run.name + ': published model identity')
        check(sha(Path(cfg['base'])) == cfg['base_sha256'], run.name + ': frozen base checkpoint identity')
        check(cfg['iterations'] * cfg['population'] * cfg['conditions'] == 3840, run.name + ': declared budget')
        snaps = completed_checkpoints(run)
        check(len(snaps) == cfg['iterations'], f'{run.name}: generation count')
        for name, expected in cfg['sources'].items():
            # Published runs retain the exact source they used, even when later
            # maintenance changes checkpoint handling. Never rewrite run hashes.
            sources = (Path('movement_lab') / name, root / 'source-training' / name)
            check(any(p.is_file() and sha(p) == expected for p in sources), f'{run.name}: {name} source')
        prev_mean = np.zeros(cfg['knots'] * 14)
        prev_std = np.full(cfg['knots'] * 14, cfg['std'])
        for i, path in enumerate(snaps):
            z = np.load(path)
            cs = json.loads(str(z['conditions']))
            expected = [c.to_dict() for c in conditions(cfg['seed'], i, cfg['conditions'], cfg['lo'], cfg['hi'], cfg['stratified'])]
            check(json.dumps(cs, sort_keys=True) == json.dumps(expected, sort_keys=True), f'{run.name}/{i+1}: training conditions')
            all_train.update(c['seed'] for c in cs)
            state = json.loads(str(z['state']))
            check(state['trials'] == (i + 1) * cfg['population'] * cfg['conditions'], f'{run.name}/{i+1}: exact trial budget')
            check(np.array_equal(z['best'], z['candidates'][np.argmax(z['scores'])]), f'{run.name}/{i+1}: winner')
            scores = z['outcomes'][:, 0].reshape(cfg['population'], cfg['conditions']).mean(axis=1)
            check(np.array_equal(scores, z['scores']), f'{run.name}/{i+1}: scores from outcomes')
            if cfg['arm'] == 'cem':
                mean, std = update_distribution(prev_mean, prev_std, z['candidates'], scores, cfg['elite'], cfg['floor'])
                check(np.array_equal(mean, z['mean']) and np.array_equal(std, z['std']), f'{run.name}/{i+1}: adaptive update')
                prev_mean, prev_std = z['mean'], z['std']
            else:
                check(np.array_equal(z['mean'], prev_mean) and np.array_equal(z['std'], prev_std), f'{run.name}/{i+1}: nonadaptive distribution')
        check(np.array_equal(np.load(run / 'final.npz')['best'], np.load(snaps[-1])['best']), f'{run.name}: final selection')
    all_rows = []
    cs = {name: (vec, design) for name, vec, design in controllers(root)}
    for battery in ('grid', 'shifts', 'unforced'):
        expected = grid() if battery == 'grid' else shifts()
        names = set(cs) if battery == 'grid' else {name for name in cs if name.count('-') <= 1}
        if battery == 'unforced':
            names = {f'cem-s{s}' for s in (1, 2, 3)}
            expected = [('forced-no-push', Condition(seed=3_200_000_000 + i, duration_s=6, joint_noise_rad=.01)) for i in range(6)]
        paths = sorted((root / 'eval' / battery).glob('*.gz'))
        check({p.name.removesuffix('.jsonl.gz') for p in paths} == names, battery + ': all controllers')
        for path in paths:
            rows = read_rows(path)
            check(len(rows) == len(expected), f'{path.name}: condition count')
            check(json.dumps([(r['set'], r['condition']) for r in rows], sort_keys=True) == json.dumps([(label, c.to_dict()) for label, c in expected], sort_keys=True), f'{path.name}: exact held-out conditions')
            check(not all_train & {r['condition']['seed'] for r in rows}, f'{path.name}: seed separation')
            all_rows.extend(rows)
    grid_rows = [r for r in all_rows if r['set'].startswith('forward')]
    base = {r['condition']['seed']: r['pre_qpos'] for r in grid_rows if r['controller'] == 'reflex02b'}
    for r in grid_rows:
        # Sensor-triggered options can fire spuriously before the true push;
        # that is measured separately, not asserted to share the starting stance.
        if r['design']['trigger'] == 'event' and r['controller'] != 'wide-reflex':
            check(r['pre_qpos'] == base[r['condition']['seed']], r['controller'] + ': matched pre-push state')
    from movement_lab.analysis02c import tables
    check(tables(root) == json.loads((root / 'report/tables.json').read_text()), 'all tables independently recomputed')
    if resimulate:
        init_worker()
        ids = np.random.default_rng(2026).choice(len(all_rows), min(resimulate, len(all_rows)), replace=False)
        for i in ids:
            row = all_rows[i]
            vec, _ = cs[row['controller']]
            actual = evaluate((row['controller'], vec, Design(**row['design']), row['set'], Condition(**row['condition'])))
            check(actual == row, f'resimulation: {row["controller"]}/{row["condition"]["seed"]}')
            result['resimulated'] += 1
    result['passed'] = not result['errors']
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    p.add_argument('--resimulate', type=int, default=0)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    r = validate(a.root, a.resimulate)
    if a.out:
        a.out.write_text(json.dumps(r, indent=2) + '\n')
    print(json.dumps(dict(passed=r['passed'], checks=len(r['checks']), errors=r['errors'], resimulated=r['resimulated']), indent=2))
    if not r['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
