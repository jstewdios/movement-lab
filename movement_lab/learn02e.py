"""02E robustness learning: 02D's bounded CEM with randomized IMU misalignment.

Forked from learn02d (unchanged). Every training condition is 02D's revised
curriculum condition with the same seed; ``--randomize-deg X`` adds a constant
IMU mounting misalignment of U(0, X) degrees about a uniformly random horizontal
axis, drawn from a separate RNG stream ([condition seed, 302]), on every stratum.
With ``--randomize-deg 0`` the conditions, candidates, outcomes and search
distribution reproduce learn02d exactly (tested against 02D's published
history-s72 generation 1). Generations are atomic, resumable, and identity-checked.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, fields
import json
import math
import os
from pathlib import Path
import time
import numpy as np
from movement_lab.sim02b import Sim, DEFAULT_MODEL
from movement_lab.sim02c import BASE
from movement_lab.sim02d import Condition as Condition02d
from movement_lab.sim02e import Condition, rollout
from movement_lab.control02d import Design, OPTION, load_basis
from movement_lab.learn02c import digest, completed_checkpoints, update_distribution
from movement_lab.learn02d import conditions as conditions02d

SIM = THETA = BASIS = None
SOURCES = ('learn02e.py', 'sim02e.py', 'control02d.py', 'sim02d.py', 'sim02b.py', 'sim02c.py', 'eval02c.py',
           'learn02d.py', 'learn02c.py', 'physics02b.py')


def init_worker():
    global SIM, THETA, BASIS
    SIM = Sim(); THETA = np.load(BASE)['final']; BASIS = load_basis()


def worker(task):
    vec, c, design = task
    s, _ = rollout(SIM, THETA, vec, c, design, basis=BASIS)
    return [s['fitness'], float(s['recovered']), float(s['survived']), s['residual_rms_rad']]


def misalignment(seed, max_deg):
    """(roll, pitch, 0) in degrees: U(0, max) about a uniformly random horizontal axis."""
    if not max_deg:
        return (0., 0., 0.)
    rng = np.random.default_rng([seed, 302])
    angle, phi = float(rng.uniform(0, max_deg)), float(rng.uniform(0, 2 * math.pi))
    return (angle * math.cos(phi), angle * math.sin(phi), 0.)


def conditions(seed, generation, revision=True, randomize_deg=0.0):
    out = []
    for c in conditions02d(seed, generation, revision):
        kw = {f.name: getattr(c, f.name) for f in fields(Condition02d)}
        kw['imu_rpy_deg'] = misalignment(c.seed, randomize_deg)
        out.append(Condition(**kw))
    return out


def config(a, design):
    return dict(seed=a.seed, arm=a.arm, iterations=a.iterations, population=a.population, elite=a.elite,
                conditions_per_generation=12 if a.revision else 8, revision=a.revision, initial_std=a.std, floor=.025,
                randomize_deg=a.randomize_deg, design=asdict(design), model_sha256=digest(DEFAULT_MODEL / 'model.mjb'),
                base_sha256=digest(BASE), option_sha256=digest(OPTION),
                sources={n: digest(Path('movement_lab') / n) for n in SOURCES})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--arm', choices=('history', 'memoryless'), default='history')
    p.add_argument('--seed', type=int, default=72)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--population', type=int, default=16)
    p.add_argument('--elite', type=int, default=4)
    p.add_argument('--std', type=float, default=.2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--budget', type=float, default=600)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--no-revision', dest='revision', action='store_false', help="02D's original 8-stratum curriculum")
    p.add_argument('--randomize-deg', type=float, default=0.0)
    a = p.parse_args(argv)
    if not (0 <= a.seed < 1000 and 3 <= a.population and 1 <= a.elite < a.population and 1 <= a.workers <= 6 and 0 <= a.randomize_deg <= 6):
        p.error('invalid bounded configuration')
    design = Design(memory=a.arm == 'history')
    cfg = config(a, design)
    if a.resume:
        if json.loads((a.out / 'config.json').read_text()) != cfg:
            raise SystemExit('resume source/configuration mismatch')
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n', newline='\n')
    rng = np.random.default_rng(a.seed)
    mean = np.zeros(design.dim); std = np.full(design.dim, a.std)
    std.reshape(2, -1)[:, -1] = a.std / 4
    best = mean.copy(); generation = trials = 0; elapsed = 0.
    checkpoints = completed_checkpoints(a.out)
    if a.resume and checkpoints:
        z = np.load(checkpoints[-1]); state = json.loads(str(z['state']))
        mean, std, best = z['mean'], z['std'], z['best']
        rng.bit_generator.state = state['rng']
        generation, trials, elapsed = state['generation'], state['trials'], state['wall_s']
    if hasattr(os, 'nice'):
        os.nice(10)
    started = time.perf_counter()
    with ProcessPoolExecutor(a.workers, initializer=init_worker) as pool:
        while generation < a.iterations and time.perf_counter() - started < a.budget:
            cands = np.clip(mean + std * rng.standard_normal((a.population, design.dim)), -2, 2)
            cands[0], cands[1], cands[2] = best, mean, 0.
            cs = conditions(a.seed, generation, a.revision, a.randomize_deg)
            outcomes = np.asarray(list(pool.map(worker, [(v, c, design) for v in cands for c in cs], chunksize=4))).reshape(a.population, len(cs), 4)
            scores = outcomes[:, :, 0].mean(axis=1)
            best = cands[np.argmax(scores)].copy()
            mean, std = update_distribution(mean, std, cands, scores, a.elite, .025)
            generation += 1; trials += a.population * len(cs)
            state = dict(generation=generation, trials=trials, rng=rng.bit_generator.state,
                         wall_s=elapsed + time.perf_counter() - started)
            row = dict(generation=generation, trials=trials, wall_s=state['wall_s'],
                       winner_score=float(scores.max()), zero_score=float(scores[2]), mean_score=float(scores.mean()),
                       winner_recoveries=int(outcomes[np.argmax(scores), :, 1].sum()),
                       zero_recoveries=int(outcomes[2, :, 1].sum()))
            path = a.out / f'generation-{generation:03d}.npz'
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, candidates=cands, scores=scores, outcomes=outcomes, best=best, mean=mean, std=std,
                                state=json.dumps(state), row=json.dumps(row), conditions=json.dumps([asdict(c) for c in cs]))
            tmp.replace(path)
            print(json.dumps(row), flush=True)
    if generation == a.iterations:
        tmp = a.out / 'final.tmp.npz'
        np.savez_compressed(tmp, best=best)
        tmp.replace(a.out / 'final.npz')
        (a.out / 'final.json').write_text(json.dumps(state, indent=2) + '\n', newline='\n')


if __name__ == '__main__':
    main()
