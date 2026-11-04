"""Resumable diagonal cross-entropy search with a matched nonadaptive control.

Each completed generation is an atomic checkpoint containing parameters, search
distribution, RNG state, scores and selected training conditions. Independent
conditions rescore the incumbent and every new candidate each generation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np

from movement_lab.sim02b import Sim, Condition, DEFAULT_MODEL
from movement_lab.sim02c import BASE, Design, rollout

SIM = THETA = None


def init_worker(model, base):
    global SIM, THETA
    SIM = Sim(model)
    THETA = np.load(base)['final']


def worker(task):
    vec, cond, design = task
    s, _ = rollout(SIM, THETA, vec, cond, design)
    return {k: s[k] for k in ('fitness', 'survived', 'recovered', 'failure_cause', 'simulated_s')}


def conditions(seed, generation, count, lo, hi):
    out = []
    for j in range(count):
        s = 2_000_000_000 + seed * 100_000 + generation * 100 + j
        rng = np.random.default_rng(s)
        out.append(Condition(seed=s, duration_s=4, joint_noise_rad=.01,
                             push_time_s=float(rng.uniform(1.2, 1.8)),
                             push_dv=(float(rng.uniform(lo, hi)), 0.0)))
    return out


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def update_distribution(mean, std, candidates, scores, elite, floor):
    ids = np.argsort(-scores, kind='stable')[:elite]
    selected = candidates[ids]
    return .25 * mean + .75 * selected.mean(axis=0), np.maximum(floor, .25 * std + .75 * selected.std(axis=0))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--arm', choices=('cem', 'random'), default='cem')
    p.add_argument('--iterations', type=int, default=25)
    p.add_argument('--population', type=int, default=48)
    p.add_argument('--conditions', type=int, default=3)
    p.add_argument('--elite', type=int, default=8)
    p.add_argument('--std', type=float, default=.25)
    p.add_argument('--floor', type=float, default=.04)
    p.add_argument('--lo', type=float, default=.29)
    p.add_argument('--hi', type=float, default=.36)
    p.add_argument('--knots', type=int, default=4)
    p.add_argument('--horizon', type=float, default=.8)
    p.add_argument('--bound', type=float, default=.8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--budget', type=float, default=900, help='wall-second launch cap; resumable at generations')
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--base', type=Path, default=BASE)
    p.add_argument('--init', type=Path)
    p.add_argument('--resume', action='store_true')
    a = p.parse_args(argv)
    if not (1 <= a.elite < a.population and 1 <= a.workers <= 2 and a.conditions < 100 and 0 <= a.seed < 1000):
        p.error('invalid elite/population, workers (maximum 2), conditions or seed')
    design = Design(knots=a.knots, horizon=a.horizon, bound=a.bound)
    cfg = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items() if k not in ('resume', 'budget')}
    cfg.update(design=asdict(design), base_sha256=digest(a.base), model_sha256=digest(a.model / 'model.mjb'),
               sources={n: digest(Path('movement_lab') / n) for n in ('learn02c.py', 'sim02c.py', 'sim02b.py', 'model02b.py')})
    if a.resume:
        saved = json.loads((a.out / 'config.json').read_text())
        if saved != cfg:
            raise SystemExit('resume configuration/source mismatch')
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    rng = np.random.default_rng(a.seed)
    dim = a.knots * 14
    mean, std = np.zeros(dim), np.full(dim, a.std)
    if a.init:
        mean = np.load(a.init)['best'].copy()
    best = mean.copy()
    generation, elapsed, trials = 0, 0., 0
    checkpoints = sorted(a.out.glob('generation-*.npz'))
    if a.resume and checkpoints:
        z = np.load(checkpoints[-1])
        mean, std, best = z['mean'], z['std'], z['best']
        state = json.loads(str(z['state']))
        rng.bit_generator.state = state['rng']
        generation, elapsed, trials = state['generation'], state['wall_s'], state['trials']
    start = time.perf_counter()
    # Lower priority and bounded worker count keep the desktop responsive.
    if hasattr(os, 'nice'):
        os.nice(10)
    with ProcessPoolExecutor(a.workers, initializer=init_worker, initargs=(a.model, a.base)) as pool:
        while generation < a.iterations and time.perf_counter() - start < a.budget:
            center = mean if a.arm == 'cem' else np.zeros(dim)
            scale = std if a.arm == 'cem' else np.full(dim, a.std)
            cands = np.clip(center + scale * rng.standard_normal((a.population, dim)), -a.bound, a.bound)
            cands[0], cands[1] = best, center
            conds = conditions(a.seed, generation, a.conditions, a.lo, a.hi)
            res = list(pool.map(worker, [(v, c, design) for v in cands for c in conds], chunksize=4))
            scores = np.array([r['fitness'] for r in res]).reshape(a.population, a.conditions).mean(axis=1)
            best = cands[np.argmax(scores)].copy()
            if a.arm == 'cem':
                mean, std = update_distribution(mean, std, cands, scores, a.elite, a.floor)
            generation += 1
            trials += len(res)
            state = dict(generation=generation, trials=trials, rng=rng.bit_generator.state,
                         wall_s=elapsed + time.perf_counter() - start)
            row = dict(generation=generation, trials=trials, wall_s=state['wall_s'], best_score=float(scores.max()),
                       mean_score=float(scores.mean()), recovered=sum(r['recovered'] for r in res),
                       survived=sum(r['survived'] for r in res), std_mean=float(std.mean()),
                       failure_counts={k: sum(r['failure_cause'] == k for r in res) for k in
                                       ('tilt', 'height', 'nonfoot', 'penetration', 'nonfinite', None)})
            path = a.out / f'generation-{generation:03d}.npz'
            tmp = path.with_suffix('.tmp.npz')
            np.savez_compressed(tmp, mean=mean, std=std, best=best, candidates=cands, scores=scores,
                                outcomes=np.array([[r['fitness'], r['survived'], r['recovered']] for r in res]),
                                state=json.dumps(state), row=json.dumps(row),
                                conditions=json.dumps([asdict(c) for c in conds]))
            tmp.replace(path)
            print(json.dumps(row), flush=True)
    # The last generation's paired winner is the predeclared final checkpoint;
    # no held-out observations are used for choosing it.
    if generation == a.iterations:
        np.savez_compressed(a.out / 'final.npz', best=best, mean=mean)
        (a.out / 'final.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
