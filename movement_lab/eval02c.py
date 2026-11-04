"""Paired, physics-rate held-out evaluation and causal controls for Study 02C."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import gzip
import json
import math
import os
from pathlib import Path
from unittest.mock import patch
import numpy as np

from movement_lab.sim02b import Sim, Condition, DEFAULT_MODEL
from movement_lab.sim02c import BASE, Design, rollout, audit
from movement_lab.physics02b import capture_bounds

SIM = THETA = None


def init_worker():
    global SIM, THETA
    SIM = Sim()
    THETA = np.load(BASE)['final']


def grid():
    """Training uses 4 s and onset 1.2–1.8 s; here 6 s and onset 2.2–3.2 s."""
    out = []
    for j, mag in enumerate((0., .20, .26, .28, .30, .32, .34, .36, .40, .45)):
        for rep in range(6):
            s = 3_000_000_000 + j * 10 + rep
            rng = np.random.default_rng(s)
            out.append((f'forward-{mag:.2f}', Condition(seed=s, duration_s=6, joint_noise_rad=.01,
                        push_time_s=float(rng.uniform(2.2, 3.2)) if mag else None, push_dv=(mag, 0.))))
    return out


def shifts():
    out = []
    for i, (name, kw) in enumerate((('friction-.6', {'foot_friction': .6}),
                                  ('friction-1.4', {'foot_friction': 1.4}),
                                  ('lag-6', {'lag_min': 6, 'lag_max': 6}),
                                  ('voltage-6.5', {'vin': 6.5}),
                                  ('imu-pitch-2', {'imu_rpy_deg': (0., 2., 0.)}))):
        for j, mag in enumerate((.28, .32, .36)):
            for rep in range(3):
                # Same random seeds across parameter shifts, paired with nominal
                # in this battery (separate from the main grid).
                seed = 3_100_000_000 + j * 10 + rep
                out.append((name, Condition(seed=seed, duration_s=6, joint_noise_rad=.01,
                            push_time_s=2.5, push_dv=(mag, 0.), **kw)))
    for j, mag in enumerate((.28, .32, .36)):
        for rep in range(3):
            out.append(('nominal', Condition(seed=3_100_000_000 + j * 10 + rep, duration_s=6,
                        joint_noise_rad=.01, push_time_s=2.5, push_dv=(mag, 0.))))
    return out


def load_run(path):
    cfg = json.loads((path / 'config.json').read_text())
    vec = np.load(path / 'final.npz')['best']
    return vec, Design(**cfg['design'])


def controllers(root, controls=True):
    out = [('reflex02b', np.zeros(56), Design()), ('wide-reflex', np.zeros(56), Design())]
    for run in sorted((root / 'runs').iterdir()):
        if not (run / 'final.npz').exists():
            continue
        vec, design = load_run(run)
        out.append((run.name, vec, design))
        if controls and run.name.startswith('cem'):
            knots = vec.reshape(design.knots, 14)
            legs, head = knots.copy(), np.zeros_like(knots)
            legs[:, 5:9] = 0
            head[:, 5:9] = knots[:, 5:9]
            out.extend([(run.name + '-reverse', knots[::-1].copy().ravel(), design),
                        (run.name + '-constant', np.tile(knots.mean(axis=0), (design.knots, 1)).ravel(), design),
                        (run.name + '-legs', legs.ravel(), design),
                        (run.name + '-head', head.ravel(), design),
                        (run.name + '-bound35', vec, replace(design, bound=.35)),
                        (run.name + '-delay80', vec, replace(design, delay=.08)),
                        (run.name + '-imu', vec, replace(design, trigger='imu'))])
    return out


def simulate(sim, theta, name, vec, cond, design):
    # A scoped intervention on this study's imported action function only.
    # Never changes sim02b or upstream; each worker runs one trial at a time.
    # Identical old weights and observations, only its clipping range widens.
    if name == 'wide-reflex':
        with patch('movement_lab.sim02c.action', wide_action):
            return rollout(sim, theta, vec, cond, design, record=True, terminate=False)
    return rollout(sim, theta, vec, cond, design, record=True, terminate=False)


def audit_record(rec, cond, z0, sim=None):
    """Evaluation additionally rejects vertical motion or unsupported endpoints."""
    out = audit(rec, cond, z0)
    final = rec['time_s'] > cond.duration_s - 1 + 1e-8
    supported = rec['foot_found'].any(axis=1) & (rec['foot_force_n'].sum(axis=1) > .5)
    stopped = np.linalg.norm(rec['qvel'][:, :3], axis=1) < .05
    out['quiet_supported_final_second'] = bool(final.any() and np.all(supported[final] & stopped[final]))
    if sim is not None:
        q = rec['qpos'][:, sim.qadr]
        overshoot = max(0., float(np.max(sim.lo - .02 - q)), float(np.max(q - (sim.hi + .02))))
        out['joint_limit_overshoot_rad'] = overshoot
        out['physical_valid'] = out['physical_valid'] and overshoot <= .03
        out['recovered_200hz'] = out['recovered_200hz'] and out['physical_valid']
    out['recovered_200hz'] = out['recovered_200hz'] and out['quiet_supported_final_second']
    out['recovery_with_step'] = out['recovery_with_step'] and out['recovered_200hz']
    return out


def evaluate(task):
    name, vec, design, label, cond = task
    s, rec = simulate(SIM, THETA, name, vec, cond, design)
    actual_push = None if cond.push_time_s is None else math.ceil(cond.push_time_s / .02 - 1e-9) * .02
    s.update(audit_record(rec, replace(cond, push_time_s=actual_push), SIM.z0, sim=SIM))
    s['actual_push_time_s'] = actual_push
    if s['pre_qpos'] is not None:
        aux = SIM.aux
        aux.qpos[:] = s['pre_qpos']
        SIM.mujoco.mj_kinematics(SIM.m, aux)
        SIM.mujoco.mj_comPos(SIM.m, aux)
        bound = capture_bounds(SIM, aux, [0.])['bounds'][0]
        s['capture_bound_m_s'] = bound
        s['beyond_capture'] = bool(cond.push_dv[0] > bound + .005)
    else:
        s['capture_bound_m_s'], s['beyond_capture'] = None, False
    time = rec['time_s']
    post = time > (actual_push if actual_push is not None else 0)
    ref_idx = max(0, int(np.searchsorted(time, actual_push or .5)) - 1)
    feet = rec['foot_pos'][:, :, :2]
    loaded = rec['foot_found'][1:] & rec['foot_found'][:-1] & post[1:, None]
    s['loaded_path_m'] = (np.linalg.norm(np.diff(feet, axis=0), axis=2) * loaded).sum(axis=0).tolist()
    s['peak_clearance_m'] = rec['clearance_m'][post].max(axis=0).tolist()
    s['post_foot_delta_xy'] = (feet[-1] - feet[ref_idx]).tolist()
    s['post_root_delta_xy'] = (rec['qpos'][-1, :2] - rec['qpos'][ref_idx, :2]).tolist()
    s.update(controller=name, set=label, condition=asdict(cond), design=asdict(design),
             base_action_bound_rad=.8 if name == 'wide-reflex' else .35)
    return s


def wide_action(theta, feat):
    return np.clip(theta[:, :5] @ feat + theta[:, 5], -.8, .8)


def write_gzip(path, rows):
    data = ''.join(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n' for r in rows).encode()
    path.write_bytes(gzip.compress(data, mtime=0))


def read_rows(path):
    return [json.loads(l) for l in gzip.decompress(Path(path).read_bytes()).splitlines()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    p.add_argument('--battery', choices=('grid', 'shifts', 'unforced'), default='grid')
    a = p.parse_args()
    out = a.root / 'eval' / a.battery
    out.mkdir(parents=True, exist_ok=True)
    cs = controllers(a.root, controls=a.battery != 'shifts')
    conds = grid() if a.battery == 'grid' else shifts()
    if a.battery == 'unforced':
        cs = [(name, vec, replace(design, trigger='clock')) for name, vec, design in cs if name.startswith('cem') and '-' not in name[4:]]
        conds = [('forced-no-push', Condition(seed=3_200_000_000 + i, duration_s=6, joint_noise_rad=.01)) for i in range(6)]
    if hasattr(os, 'nice'):
        os.nice(10)
    with ProcessPoolExecutor(2, initializer=init_worker) as pool:
        for name, vec, design in cs:
            dest = out / f'{name}.jsonl.gz'
            if dest.exists():
                raise SystemExit(f'refusing existing result: {dest}')
            rows = list(pool.map(evaluate, [(name, vec, design, label, cond) for label, cond in conds], chunksize=3))
            write_gzip(dest, rows)
            print(name, len(rows), 'recovery', sum(r['recovered_200hz'] for r in rows),
                  'step recovery', sum(r['recovery_with_step'] for r in rows), flush=True)


if __name__ == '__main__':
    main()
