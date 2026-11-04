"""Bounded sequential orchestration; at most two low-priority physics workers.

Run from the repository root with the separately installed upstream Python.
Completed jobs are reused; interrupted training resumes from a generation.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('train', 'evaluate', 'report'))
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    a = p.parse_args()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
               MPLCONFIGDIR=str((a.root / 'matplotlib').resolve()))
    if a.phase == 'train':
        for arm in ('cem', 'random'):
            for seed in (1, 2, 3):
                out = a.root / 'runs' / f'{arm}-s{seed}'
                if (out / 'final.json').exists():
                    print('complete:', out, flush=True)
                    continue
                args = ['--out', str(out), '--seed', str(seed), '--arm', arm,
                        '--iterations', '20', '--conditions', '4', '--knots', '2', '--horizon', '.4',
                        '--lo', '.28', '--hi', '.36', '--stratified', '--budget', '600']
                if out.exists():
                    args.append('--resume')
                subprocess.run([sys.executable, '-m', 'movement_lab.learn02c', *args], env=env, check=True)
                if not (out / 'final.json').exists():
                    raise SystemExit(f'launch cap reached: resume {out} by rerunning this command')
    elif a.phase == 'evaluate':
        for battery in ('grid', 'shifts', 'unforced'):
            subprocess.run([sys.executable, '-m', 'movement_lab.eval02c', '--root', str(a.root), '--battery', battery], env=env, check=True)
    else:
        for module in ('analysis02c', 'replay02c'):
            subprocess.run([sys.executable, '-m', 'movement_lab.' + module, '--root', str(a.root)], env=env, check=True)


if __name__ == '__main__':
    main()
