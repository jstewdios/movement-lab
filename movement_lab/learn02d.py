"""Bounded CEM with complete atomic generations and strict resume identity."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np
from movement_lab.sim02b import Sim, DEFAULT_MODEL
from movement_lab.sim02c import BASE
from movement_lab.sim02d import Condition, rollout
from movement_lab.control02d import Design, OPTION, load_basis
from movement_lab.learn02c import digest, completed_checkpoints, update_distribution

SIM = THETA = BASIS = None
SOURCES = ('control02d.py', 'sim02d.py', 'learn02d.py', 'sim02b.py', 'sim02c.py', 'eval02c.py')


def init_worker():
    global SIM, THETA, BASIS
    SIM = Sim(); THETA = np.load(BASE)['final']; BASIS = load_basis()


def worker(task):
    vec, c, design = task
    s, _ = rollout(SIM, THETA, vec, c, design, basis=BASIS)
    return [s['fitness'], float(s['recovered']), float(s['survived']), s['residual_rms_rad']]


def conditions(seed, generation, revision=False):
    """Eight strata, exactly paired across arms. Separate from all evaluation."""
    out = []
    for j in range(8):
        s = 4_000_000_000 + seed * 100_000 + generation * 100 + j
        rng = np.random.default_rng(s)
        t = float(rng.uniform(1.2, 1.8))
        c = Condition(seed=s, duration_s=4, joint_noise_rad=.01)
        if j in (1, 2, 3, 4):
            lo, hi = ((.04, .16), (.18, .28), (.29, .33), (.33, .37))[j - 1]
            c.push_time_s, c.push_dv = t, (float(rng.uniform(lo, hi)), 0.)
        elif j == 5:
            duration = float(rng.choice([.04, .12, .24]))
            # Nominal free-body impulse equivalent, NOT actual measured delta-v.
            force = .73724318 * float(rng.uniform(.08, .28)) / duration
            c.pulse_time_s = round(t / .02) * .02
            c.pulse_duration_s, c.pulse_force = duration, (force, 0., 0.)
        elif j == 6:
            c.push_time_s = t
            c.push_dv = ((-.08, 0.), (0., .12), (0., -.12))[generation % 3]
        elif j == 7:
            c.false_gyro_time_s = round(t / .02) * .02
            c.false_gyro_rad_s = float(rng.choice([-1., 1.]) * rng.uniform(.55, .9))
        out.append(c)
    if revision:
        out[0].duration_s = 6
        for j in range(8, 12):
            s = 4_000_000_000 + seed * 100_000 + generation * 100 + j
            c = Condition(seed=s, duration_s=6, joint_noise_rad=.01)
            if j == 11:
                rng = np.random.default_rng(s)
                c.push_time_s, c.push_dv = float(rng.uniform(1.2,1.8)), (float(rng.uniform(.01,.03)),0.)
            out.append(c)
    return out


def config(a, design):
    return dict(seed=a.seed, arm=a.arm, iterations=a.iterations, population=a.population,
                elite=a.elite, conditions_per_generation=12 if a.revision else 8, revision=a.revision, initial_std=a.std, floor=.025,
                design=asdict(design), model_sha256=digest(DEFAULT_MODEL / 'model.mjb'),
                base_sha256=digest(BASE), option_sha256=digest(OPTION),
                sources={n:digest(Path('movement_lab') / n) for n in SOURCES})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--arm', choices=('history', 'memoryless'), default='history')
    p.add_argument('--seed', type=int, default=71)
    p.add_argument('--iterations', type=int, default=8)
    p.add_argument('--population', type=int, default=24)
    p.add_argument('--elite', type=int, default=6)
    p.add_argument('--std', type=float, default=.2)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--budget', type=float, default=600)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--revision', action='store_true', help='four quiet strata plus benign motion; same episode budget with population 16')
    a = p.parse_args(argv)
    if not (0 <= a.seed < 1000 and 3 <= a.population and 1 <= a.elite < a.population and 1 <= a.workers <= 2):
        p.error('invalid bounded configuration')
    design = Design(memory=a.arm == 'history')
    cfg = config(a, design)
    if a.resume:
        if json.loads((a.out / 'config.json').read_text()) != cfg:
            raise SystemExit('resume source/configuration mismatch')
    else:
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / 'config.json').write_text(json.dumps(cfg, indent=2) + '\n')
    rng = np.random.default_rng(a.seed)
    mean = np.zeros(design.dim); std = np.full(design.dim, a.std)
    std.reshape(2, -1)[:, -1] = a.std / 4  # posture bias initially smaller
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
            cs = conditions(a.seed, generation, a.revision)
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
        (a.out / 'final.json').write_text(json.dumps(state, indent=2) + '\n')


if __name__ == '__main__':
    main()
