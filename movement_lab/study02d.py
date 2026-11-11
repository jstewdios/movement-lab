"""Staged 02D driver. Pilot gates learning; held-out evaluation never selects."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from movement_lab.learn02c import digest

DEPENDENCIES=('learn02c.py','model02b.py','contact02c.py','replay02c.py','physics02b.py')


def require_dependencies(root):
    """Guard reused numerical helpers as well as each run's own source hashes."""
    saved=json.loads((root/'dependencies.json').read_text())
    if set(saved)!=set(DEPENDENCIES):
        raise SystemExit('incomplete frozen dependency inventory')
    for name,sha in saved.items():
        if digest(Path('movement_lab')/name)!=sha:
            raise SystemExit('frozen dependency mismatch: '+name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=('preflight', 'pilot', 'pilot-revised', 'train', 'evaluate', 'report'))
    p.add_argument('--root', type=Path, default=Path('results/02d'))
    a = p.parse_args()
    if a.phase!='preflight':
        require_dependencies(a.root)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OMP_NUM_THREADS='1',
               OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR=str(Path('.cache/matplotlib').resolve()))
    def run(module, *args):
        subprocess.run([sys.executable, '-m', 'movement_lab.' + module, *map(str,args)], env=env, check=True)
    def train(out, arm, seed, iterations, revision=False):
        if (out / 'final.json').exists():
            # Completed evidence is immutable under the orchestration command.
            from movement_lab.learn02c import digest
            cfg=json.loads((out/'config.json').read_text())
            if cfg['seed']!=seed or cfg['arm']!=arm or cfg['iterations']!=iterations or cfg.get('revision',False)!=revision:
                raise SystemExit('completed run configuration mismatch')
            for name,sha in cfg['sources'].items():
                if digest(Path('movement_lab')/name)!=sha:
                    raise SystemExit('completed run source mismatch: '+name)
            print('retained completed run:',out,flush=True)
            return
        args = ['--out', out, '--arm', arm, '--seed', seed, '--iterations', iterations, '--budget', 600]
        if revision:
            args += ['--revision', '--population', 16, '--elite', 4]
        if out.exists():
            args.append('--resume')
        run('learn02d', *args)
        if not (out / 'final.json').exists():
            raise SystemExit('launch budget reached; rerun this phase to resume the declared episode budget')
    if a.phase == 'preflight':
        a.root.mkdir(parents=True, exist_ok=True)
        run('preflight02d', '--root', a.root)
    elif a.phase in ('pilot', 'pilot-revised'):
        if not json.loads((a.root / 'preflight.json').read_text())['passed']:
            raise SystemExit('preflight gate did not pass')
        revised = a.phase == 'pilot-revised'
        if revised and not (a.root / 'pilot/gate.json').exists():
            raise SystemExit('revision requires completed original pilot')
        seed = 72 if revised else 71
        train(a.root / f'pilot/history-s{seed}', 'history', seed, 8, revised)
        run('eval02d', '--root', a.root, '--battery', a.phase)
    elif a.phase == 'train':
        if not json.loads((a.root / 'pilot/gate-revised.json').read_text())['passed']:
            raise SystemExit('pilot did not justify Phase B')
        for arm in ('history', 'memoryless'):
            for seed in (1,2,3):
                train(a.root / 'runs' / f'{arm}-s{seed}', arm, seed, 12, True)
    elif a.phase == 'evaluate':
        for battery in ('grid', 'shifts'):
            run('eval02d', '--root', a.root, '--battery', battery)
    else:
        run('report02d', '--root', a.root)


if __name__ == '__main__':
    main()
