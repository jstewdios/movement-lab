"""02B study driver: matched learning matrix, then held-out evaluation.

    python -m movement_lab.study02b train --root results/02b
    python -m movement_lab.study02b evaluate --root results/02b
    python -m movement_lab.study02b ablate --root results/02b
    python -m movement_lab.study02b crosscheck --root results/02b
    python -m movement_lab.study02b revise --root results/02b --seeds 1

``train`` runs ARS-with-pushes, random-search-with-pushes and ARS-standing-only
for seeds 1-3 (interleaved by seed so partial results stay balanced). Completed
runs (``final.json`` present) are reused; an incomplete run directory is an
error, never overwritten. Each run's wall time is appended to ``cost.jsonl``.
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ARMS = (('ars', 'push'), ('random', 'push'), ('ars', 'none'))
SEEDS = (1, 2, 3)


def run_name(arm, disturb, seed):
    return f'{arm}-{disturb}-s{seed}'


def train(root, iterations, seeds):
    runs = root / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for arm, disturb in ARMS:
            out = runs / run_name(arm, disturb, seed)
            if (out / 'final.json').exists():
                print('reuse', out.name, flush=True)
                continue
            if out.exists():
                raise SystemExit(f'incomplete run directory {out}; inspect and move it aside')
            started = time.perf_counter()
            log = root / 'runs' / f'{out.name}.log'
            with open(log, 'w') as f:
                code = subprocess.call([sys.executable, '-m', 'movement_lab.learn02b', '--arm', arm, '--disturb', disturb,
                                        '--seed', str(seed), '--iterations', str(iterations), '--out', str(out)],
                                       stdout=f, stderr=subprocess.STDOUT)
            wall = time.perf_counter() - started
            with open(root / 'cost.jsonl', 'a', newline='\n') as f:
                f.write(json.dumps(dict(step='train', run=out.name, exit=code, wall_s=wall)) + '\n')
            print(out.name, 'exit', code, f'{wall:.0f}s', flush=True)
            if code:
                raise SystemExit(code)


def evaluate(root, seeds):
    ctrls = ['home=zero'] + [f'{run_name(a, d, s)}={root}/runs/{run_name(a, d, s)}/thetas.npz' for s in seeds for a, d in ARMS]
    started = time.perf_counter()
    cmd = [sys.executable, '-m', 'movement_lab.eval02b', '--out', str(root / 'eval/final')]
    for c in ctrls:
        cmd += ['--controller', c]
    code = subprocess.call(cmd)
    with open(root / 'cost.jsonl', 'a', newline='\n') as f:
        f.write(json.dumps(dict(step='evaluate', exit=code, wall_s=time.perf_counter() - started)) + '\n')
    raise SystemExit(code)


def _call(root, step, cmd):
    started = time.perf_counter()
    code = subprocess.call([sys.executable, '-m', *cmd])
    with open(root / 'cost.jsonl', 'a', newline='\n') as f:
        f.write(json.dumps(dict(step=step, exit=code, wall_s=time.perf_counter() - started)) + '\n')
    if code:
        raise SystemExit(code)


def ablate(root, seeds):
    cmd = ['movement_lab.eval02b', '--out', str(root / 'eval/ablation'), '--battery', 'nominal', '--battery', 'stand_long']
    for s in seeds:
        cmd += ['--ablate', f'ars-push-s{s}={root}/runs/ars-push-s{s}/thetas.npz']
    _call(root, 'ablate', cmd)


def crosscheck(root):
    cmd = ['movement_lab.crosscheck02b', '--out', str(root / 'crosscheck'), '--controller', 'home=zero']
    for run in ('ars-push-s1', 'random-push-s1', 'ars-none-s1'):
        cmd += ['--controller', f'{run}={root}/runs/{run}/thetas.npz']
    _call(root, 'crosscheck', cmd)


def revise(root, seeds, iterations):
    for s in seeds:
        for arm in ('linear', 'residual'):
            out = root / 'revision' / f'{arm}-s{s}'
            if (out / 'final.json').exists():
                continue
            _call(root, f'revise:{out.name}', ['movement_lab.revise02b', '--arm', arm, '--seed', str(s), '--iterations', str(iterations),
                                               '--init', f'{root}/runs/ars-push-s{s}/thetas.npz', '--out', str(out)])


def evaluate_revision(root, seeds):
    cmd = ['movement_lab.eval02b', '--out', str(root / 'eval/revision'), '--battery', 'push_grid', '--battery', 'nominal']
    for s in seeds:
        cmd += ['--controller', f'warm-start-s{s}={root}/runs/ars-push-s{s}/thetas.npz',
                '--controller', f'linear-s{s}=linear:{root}/revision/linear-s{s}/snapshots.npz',
                '--controller', f'residual-s{s}=residual:{root}/revision/residual-s{s}/snapshots.npz',
                '--controller', f'residual-s{s}-removed=linear:{root}/revision/residual-s{s}/snapshots.npz']
    _call(root, 'evaluate-revision', cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('train', 'evaluate', 'ablate', 'crosscheck', 'revise', 'evaluate-revision'))
    p.add_argument('--root', type=Path, default=Path('results/02b'))
    p.add_argument('--iterations', type=int, default=None, help='train: 60; revise: 25')
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    args = p.parse_args()
    if args.step == 'train':
        train(args.root, args.iterations or 60, args.seeds)
    elif args.step == 'evaluate':
        evaluate(args.root, args.seeds)
    elif args.step == 'ablate':
        ablate(args.root, args.seeds)
    elif args.step == 'crosscheck':
        crosscheck(args.root)
    elif args.step == 'evaluate-revision':
        evaluate_revision(args.root, args.seeds)
    else:
        revise(args.root, args.seeds, args.iterations or 25)


if __name__ == '__main__':
    main()
