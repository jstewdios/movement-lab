"""02B learning: Augmented Random Search on a linear IMU reflex, plus controls.

Arms (identical action space, physics, starting point theta=0, rollout budget):

* ``ars``    -- ARS V1-t (Mania et al. 2018): antithetic perturbations of the
               current reflex, scored on shared conditions; the update moves the
               reflex toward perturbations that physically did better. This is
               the only arm whose next experiment depends on earlier outcomes.
* ``random`` -- pure random exploration: every candidate is drawn independently
               around theta=0 (log-uniform scale), evaluated on the same
               conditions, and the best is kept. Experience is used only to pick
               the final winner, never to choose where to sample next.

``--disturb`` selects the training experience: ``push`` (random horizontal root
velocity increments, 00B/00C semantics) or ``none`` (standing only, identical
seeds and initial noise). Probe conditions are evaluated every few iterations
for observability; they never enter an update or a selection.

    python -m movement_lab.learn02b --arm ars --disturb push --seed 1 --out results/02b/runs/ars-push-s1
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
from movement_lab.sim02b import Condition, PARAM_SHAPE

TRAIN_SEED_BASE = 10_000_000      # training conditions: run_seed * 1e6 + counter
PROBE_SEED_BASE = 500_000_000     # probe conditions (never used for updates)
# Held-out evaluation uses >= 900_000_000 (eval02b); disjoint by construction.
PUSH_MAX = 0.4                    # m/s, horizontal root velocity increment
PUSH_WINDOW = (1.0, 2.0)          # s
TRAIN_DURATION = 4.0
JOINT_NOISE = 0.02                # rad, uniform initial joint offsets

_SIM = None


def _init(model_dir):
    global _SIM
    _SIM = sim02b.Sim(model_dir)


def _run(task):
    theta, cond = task
    s, _ = sim02b.rollout(_SIM, theta, cond)
    keep = ('fitness', 'survived', 'fall_time_s', 'fall_cause', 'alive_frac', 'max_tilt_deg', 'max_tilt_post_deg',
            'valid', 'max_pen_alive_m', 'sat_frac', 'peak_torque_nm', 'work_j', 'action_rms', 'rate_rms_rad_per_step',
            'final_root_disp_m', 'final_tilt_deg', 'min_root_z', 'foot_unloaded', 'steps_alive')
    out = {k: s[k] for k in keep}
    out['sim_s'] = s['steps_alive'] * sim02b.CONTROL_DT
    return out


def condition(seed, disturb, rng=None):
    """One training/probe condition. The push draw is made even when unused so
    that ``push`` and ``none`` share identical seeds and initial noise."""
    rng = np.random.default_rng(seed) if rng is None else rng
    angle = rng.uniform(0, 2 * math.pi)
    mag = rng.uniform(0, PUSH_MAX)
    t = rng.uniform(*PUSH_WINDOW)
    if disturb == 'none':
        mag = 0.0
    return Condition(seed=int(seed), duration_s=TRAIN_DURATION, joint_noise_rad=JOINT_NOISE,
                     push_time_s=float(t) if mag > 0 else None,
                     push_dv=(float(mag * math.cos(angle)), float(mag * math.sin(angle))))


def train_conditions(run_seed, it, k, disturb):
    return [condition(TRAIN_SEED_BASE + run_seed * 1_000_000 + it * k + j, disturb) for j in range(k)]


PROBE_MAGNITUDES = (0.1, 0.2, 0.3, 0.4)
PROBE_DIRECTIONS = 8


def probe_conditions():
    """Fixed structured probe set (identical for every arm): 8 unpushed trials and
    8 directions x 4 magnitudes pushed at 1.5 s. Probe seeds are disjoint from
    training seeds and probe outcomes never enter an update or selection."""
    conds = [Condition(seed=PROBE_SEED_BASE + j, duration_s=TRAIN_DURATION, joint_noise_rad=JOINT_NOISE) for j in range(8)]
    for i in range(PROBE_DIRECTIONS):
        a = 2 * math.pi * i / PROBE_DIRECTIONS
        for j, mag in enumerate(PROBE_MAGNITUDES):
            conds.append(Condition(seed=PROBE_SEED_BASE + 100 + 10 * i + j, duration_s=TRAIN_DURATION,
                                   joint_noise_rad=JOINT_NOISE, push_time_s=1.5,
                                   push_dv=(mag * math.cos(a), mag * math.sin(a))))
    return conds


def ars_update(theta, delta, f_plus, f_minus, top, step):
    """ARS V1-t update. Keeps the ``top`` directions ranked by max(f+, f-), and
    moves theta along sum_i (f+_i - f-_i) delta_i, scaled by step / (top * sigma_R)."""
    order = np.argsort(-np.maximum(f_plus, f_minus), kind='stable')[:top]
    sigma = float(np.concatenate([f_plus[order], f_minus[order]]).std())
    scale = step / (top * (sigma if sigma > 1e-8 else 1.0))
    update = scale * np.tensordot(f_plus[order] - f_minus[order], delta[order], axes=1)
    return theta + update, update, sigma


def probe_due(it):
    """Dense early probing captures the fast initial rise; sparse later."""
    return it <= 6 or it in (8, 10, 12) or it % 5 == 0


def summarize(results):
    f = np.array([r['fitness'] for r in results])
    return dict(fitness=float(f.mean()), survival=float(np.mean([r['survived'] for r in results])),
                alive_frac=float(np.mean([r['alive_frac'] for r in results])),
                valid=float(np.mean([r['valid'] for r in results])),
                sat_frac=float(np.mean([r['sat_frac'] for r in results])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--arm', choices=('ars', 'random'), required=True)
    p.add_argument('--disturb', choices=('push', 'none'), required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--iterations', type=int, default=60)
    p.add_argument('--directions', type=int, default=16)
    p.add_argument('--top', type=int, default=8)
    p.add_argument('--conditions', type=int, default=4)
    p.add_argument('--step', type=float, default=0.02)
    p.add_argument('--noise', type=float, default=0.03)
    p.add_argument('--random-scale', type=float, nargs=2, default=(0.01, 0.3),
                   help='log-uniform candidate scale range for the random arm')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--model', type=Path, default=sim02b.DEFAULT_MODEL)
    args = p.parse_args(argv)
    out = args.out
    out.mkdir(parents=True, exist_ok=False)
    cfg = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    cfg.update(push_max=PUSH_MAX, push_window=PUSH_WINDOW, train_duration=TRAIN_DURATION, joint_noise=JOINT_NOISE,
               fitness_weights=dict(tilt=sim02b.W_TILT, action=sim02b.W_ACTION, rate=sim02b.W_RATE),
               features=sim02b.FEATURES, feature_scale=sim02b.FEATURE_SCALE.tolist(), action_bound=sim02b.ACTION_BOUND,
               model_sha256=json.loads((args.model / 'model.json').read_text())['mjb_sha256'])
    (out / 'config.json').write_text(json.dumps(cfg, indent=1))
    rng = np.random.default_rng(args.seed)
    theta = np.zeros(PARAM_SHAPE)
    probes = probe_conditions()
    log = open(out / 'progress.jsonl', 'w')
    thetas, rollouts, sim_s = [theta.copy()], 0, 0.0
    best = dict(fitness=-np.inf, theta=theta.copy(), it=-1)
    candidates = []  # random arm: (fitness, it, theta)
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        def evaluate(pairs):
            res = list(pool.map(_run, pairs, chunksize=max(1, len(pairs) // (4 * args.workers))))
            return res

        def probe(it):
            res = evaluate([(theta if args.arm == 'ars' else best['theta'], c) for c in probes])
            pushed = [(r, c) for r, c in zip(res, probes) if c.push_time_s is not None]
            row = dict(kind='probe', it=it, **summarize(res),
                       survival_none=float(np.mean([r['survived'] for r, c in zip(res, probes) if c.push_time_s is None])),
                       survival_push=float(np.mean([r['survived'] for r, _ in pushed])),
                       survival_by_mag={str(m): float(np.mean([r['survived'] for r, c in pushed
                                                              if abs(math.hypot(*c.push_dv) - m) < 1e-9]))
                                        for m in PROBE_MAGNITUDES},
                       outcomes=[int(r['survived']) for r in res])
            log.write(json.dumps(row) + '\n'); log.flush()
            return row

        probe(0)
        for it in range(args.iterations):
            conds = train_conditions(args.seed, it, args.conditions, args.disturb)
            n, k = args.directions, args.conditions
            if args.arm == 'ars':
                delta = rng.standard_normal((n, *PARAM_SHAPE))
                cands = [theta + args.noise * delta[i] for i in range(n)] + [theta - args.noise * delta[i] for i in range(n)]
            else:
                scales = np.exp(rng.uniform(*np.log(args.random_scale), 2 * n))
                cands = [scales[i] * rng.standard_normal(PARAM_SHAPE) for i in range(2 * n)]
            tasks = [(c, cond) for c in cands for cond in conds]
            res = evaluate(tasks)
            rollouts += len(res)
            sim_s += sum(r['sim_s'] for r in res)
            f = np.array([r['fitness'] for r in res]).reshape(2 * n, k).mean(axis=1)
            surv = np.array([r['survived'] for r in res]).reshape(2 * n, k).mean(axis=1)
            row = dict(kind='train', it=it + 1, rollouts=rollouts, sim_s=sim_s, wall_s=time.perf_counter() - started,
                       cand_fitness_mean=float(f.mean()), cand_fitness_max=float(f.max()), cand_survival_mean=float(surv.mean()),
                       valid_frac=float(np.mean([r['valid'] for r in res])))
            if args.arm == 'ars':
                fp, fm = f[:n], f[n:]
                theta, step, sigma = ars_update(theta, delta, fp, fm, args.top, args.step)
                row.update(step_norm=float(np.linalg.norm(step)), theta_norm=float(np.linalg.norm(theta)),
                           reward_sigma=sigma, center_estimate=float(0.5 * (fp + fm).mean()))
            else:
                for i in range(2 * n):
                    candidates.append((float(f[i]), it, cands[i]))
                i = int(np.argmax(f))
                if f[i] > best['fitness']:
                    best = dict(fitness=float(f[i]), theta=cands[i].copy(), it=it)
                row.update(best_fitness=best['fitness'], best_it=best['it'], best_norm=float(np.linalg.norm(best['theta'])))
            thetas.append(theta.copy() if args.arm == 'ars' else best['theta'].copy())
            log.write(json.dumps(row) + '\n'); log.flush()
            print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}), flush=True)
            if probe_due(it + 1) or it + 1 == args.iterations:
                pr = probe(it + 1)
                print('  probe', json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in pr.items()}), flush=True)
        if args.arm == 'random':
            # Fair final selection: re-score the 10 best candidates on 16 fresh
            # training-distribution conditions (counted in the budget).
            top = sorted(candidates, key=lambda c: -c[0])[:10]
            sel = train_conditions(args.seed, args.iterations, 16, args.disturb)
            res = evaluate([(c[2], cond) for c in top for cond in sel])
            rollouts += len(res)
            sim_s += sum(r['sim_s'] for r in res)
            scores = np.array([r['fitness'] for r in res]).reshape(len(top), len(sel)).mean(axis=1)
            j = int(np.argmax(scores))
            theta = top[j][2]
            log.write(json.dumps(dict(kind='selection', rescored=scores.tolist(), chosen=j, chosen_it=top[j][1], rollouts=rollouts)) + '\n')
            thetas.append(theta.copy())
        final = dict(arm=args.arm, disturb=args.disturb, seed=args.seed, rollouts=rollouts, sim_s=sim_s,
                     wall_s=time.perf_counter() - started, theta_norm=float(np.linalg.norm(theta)))
        log.close()
    np.savez_compressed(out / 'thetas.npz', thetas=np.array(thetas), final=theta)
    (out / 'final.json').write_text(json.dumps(final, indent=1))
    print(json.dumps(final))


if __name__ == '__main__':
    main()
