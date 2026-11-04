"""02B targeted revision: can experience beyond the no-step limit move the boundary?

The main study's reflexes plateau within a few updates near the capture-point
limit set by the soles. Both revision arms warm-start from a learned main-study
reflex and train ONLY on harder experience (pushes 0.15-0.45 m/s, mostly beyond
that limit), with identical ARS settings, seeds and rollout budgets:

* ``linear``   -- the same 14x6 linear IMU reflex (more targeted experience only).
* ``residual`` -- the same reflex plus a residual network that also senses joint
                  positions/velocities (with upstream encoder noise):
                  a = W s + b + V tanh(U z + c), V initialised to zero, so the
                  arm starts exactly at the warm-start reflex and any new
                  behaviour (e.g. stepping) is attributable to added capacity.

    python -m movement_lab.revise02b --arm residual --init results/02b/runs/ars-push-s1/thetas.npz \
        --seed 1 --out results/02b/revision/residual-s1
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from movement_lab import sim02b
from movement_lab.learn02b import TRAIN_SEED_BASE, PUSH_WINDOW, TRAIN_DURATION, JOINT_NOISE, ars_update, probe_conditions, probe_due
from movement_lab.sim02b import Condition, PARAM_SHAPE, N_JOINTS, N_FEAT

HARD_RANGE = (0.15, 0.45)
REVISION_SEED_OFFSET = 300   # training seeds: TRAIN_SEED_BASE + (300 + seed) * 1e6 + counter (< probe base)
HIDDEN = 16
Z_DIM = N_FEAT + 2 * N_JOINTS
Q_SCALE, QD_SCALE = 0.1, 2.0
N_LIN = int(np.prod(PARAM_SHAPE))
N_RES = HIDDEN * Z_DIM + HIDDEN + N_JOINTS * HIDDEN


class Residual:
    """Linear IMU reflex + residual tanh network over IMU and joint state."""

    def __init__(self, vec):
        vec = np.asarray(vec, float)
        self.theta = vec[:N_LIN].reshape(PARAM_SHAPE)
        r = vec[N_LIN:]
        self.U = r[:HIDDEN * Z_DIM].reshape(HIDDEN, Z_DIM)
        self.c = r[HIDDEN * Z_DIM:HIDDEN * Z_DIM + HIDDEN]
        self.V = r[HIDDEN * Z_DIM + HIDDEN:].reshape(N_JOINTS, HIDDEN)

    def __call__(self, feat, q_dev, qd):
        z = np.concatenate([feat, q_dev / Q_SCALE, qd / QD_SCALE])
        return self.theta[:, :N_FEAT] @ feat + self.theta[:, N_FEAT] + self.V @ np.tanh(self.U @ z + self.c)

    @staticmethod
    def initial(theta, rng):
        U = rng.standard_normal((HIDDEN, Z_DIM)) / math.sqrt(Z_DIM)
        return np.concatenate([np.asarray(theta, float).ravel(), U.ravel(), np.zeros(HIDDEN), np.zeros(N_JOINTS * HIDDEN)])


def hard_condition(seed):
    rng = np.random.default_rng(seed)
    angle = rng.uniform(0, 2 * math.pi)
    mag = rng.uniform(*HARD_RANGE)
    t = rng.uniform(*PUSH_WINDOW)
    return Condition(seed=int(seed), duration_s=TRAIN_DURATION, joint_noise_rad=JOINT_NOISE, push_time_s=float(t),
                     push_dv=(float(mag * math.cos(angle)), float(mag * math.sin(angle))))


def hard_conditions(seed, it, k):
    return [hard_condition(TRAIN_SEED_BASE + (REVISION_SEED_OFFSET + seed) * 1_000_000 + it * k + j) for j in range(k)]


_SIM = None


def _init(model_dir):
    global _SIM
    _SIM = sim02b.Sim(model_dir)


def _run(task):
    arm, vec, cond = task
    if arm == 'linear':
        s, _ = sim02b.rollout(_SIM, np.asarray(vec).reshape(PARAM_SHAPE), cond)
    else:
        s, _ = sim02b.rollout(_SIM, np.zeros(PARAM_SHAPE), cond, policy=Residual(vec))
    return dict(fitness=s['fitness'], survived=s['survived'], n_steps=s['n_steps'], valid=s['valid'],
                sim_s=s['steps_alive'] * sim02b.CONTROL_DT, alive_frac=s['alive_frac'])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('linear', 'residual'), required=True)
    p.add_argument('--init', type=Path, required=True, help='main-study thetas.npz to warm-start from')
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=25)
    p.add_argument('--directions', type=int, default=32)
    p.add_argument('--top', type=int, default=16)
    p.add_argument('--conditions', type=int, default=8)
    p.add_argument('--step', type=float, default=0.02)
    p.add_argument('--noise', type=float, default=0.03)
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--model', type=Path, default=sim02b.DEFAULT_MODEL)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(1000 + args.seed)
    theta0 = np.load(args.init)['final']
    vec = theta0.ravel().copy() if args.arm == 'linear' else Residual.initial(theta0, rng)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(hard_range=HARD_RANGE, hidden=HIDDEN, z_dim=Z_DIM, n_params=int(vec.size),
               model_sha256=json.loads((args.model / 'model.json').read_text())['mjb_sha256'])
    (args.out / 'config.json').write_text(json.dumps(cfg, indent=1))
    probes = probe_conditions()
    log = open(args.out / 'progress.jsonl', 'w')
    snaps, rollouts, sim_s = [vec.copy()], 0, 0.0
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        def evaluate(tasks):
            return list(pool.map(_run, tasks, chunksize=max(1, len(tasks) // (4 * args.workers))))

        def probe(it):
            res = evaluate([(args.arm, vec, c) for c in probes])
            pushed = [(r, c) for r, c in zip(res, probes) if c.push_time_s is not None]
            row = dict(kind='probe', it=it, survival_none=float(np.mean([r['survived'] for r, c in zip(res, probes) if c.push_time_s is None])),
                       survival_push=float(np.mean([r['survived'] for r, _ in pushed])),
                       survival_by_mag={str(m): float(np.mean([r['survived'] for r, c in pushed if abs(math.hypot(*c.push_dv) - m) < 1e-9]))
                                        for m in (0.1, 0.2, 0.3, 0.4)},
                       steps=int(sum(r['n_steps'] for r in res)), outcomes=[int(r['survived']) for r in res])
            log.write(json.dumps(row) + '\n'); log.flush()
            print('  probe', json.dumps({k: v for k, v in row.items() if k != 'outcomes'}), flush=True)

        probe(0)
        for it in range(args.iterations):
            conds = hard_conditions(args.seed, it, args.conditions)
            n, k = args.directions, args.conditions
            delta = rng.standard_normal((n, vec.size))
            cands = [vec + args.noise * delta[i] for i in range(n)] + [vec - args.noise * delta[i] for i in range(n)]
            res = evaluate([(args.arm, c, cond) for c in cands for cond in conds])
            rollouts += len(res)
            sim_s += sum(r['sim_s'] for r in res)
            f = np.array([r['fitness'] for r in res]).reshape(2 * n, k).mean(axis=1)
            surv = np.array([r['survived'] for r in res]).reshape(2 * n, k).mean(axis=1)
            vec, step, sigma = ars_update(vec, delta, f[:n], f[n:], args.top, args.step)
            snaps.append(vec.copy())
            row = dict(kind='train', it=it + 1, rollouts=rollouts, sim_s=sim_s, wall_s=time.perf_counter() - started,
                       cand_fitness_mean=float(f.mean()), cand_survival_mean=float(surv.mean()),
                       steps=int(sum(r['n_steps'] for r in res)), step_norm=float(np.linalg.norm(step)), reward_sigma=sigma,
                       residual_out_norm=float(np.linalg.norm(vec[N_LIN + HIDDEN * Z_DIM + HIDDEN:])) if args.arm == 'residual' else None)
            log.write(json.dumps(row) + '\n'); log.flush()
            print(json.dumps({k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in row.items()}), flush=True)
            if probe_due(it + 1) or it + 1 == args.iterations:
                probe(it + 1)
    log.close()
    np.savez_compressed(args.out / 'snapshots.npz', snapshots=np.array(snaps), final=vec)
    final = dict(arm=args.arm, seed=args.seed, init=str(args.init), rollouts=rollouts, sim_s=sim_s, wall_s=time.perf_counter() - started)
    (args.out / 'final.json').write_text(json.dumps(final, indent=1))
    print(json.dumps(final))


if __name__ == '__main__':
    main()
