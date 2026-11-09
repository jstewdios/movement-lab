"""Disjoint pilot and confirmation batteries. Every outcome is audited at 200 Hz."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import numpy as np
from movement_lab.sim02b import Sim, DEFAULT_MODEL
from movement_lab.sim02c import BASE
from movement_lab.sim02d import Condition, rollout, event_time
from movement_lab.control02d import Design, load_basis, OPTION
from movement_lab.eval02c import write_gzip, read_rows
from movement_lab.learn02c import digest

SIM = THETA = BASIS = None


def init_worker():
    global SIM, THETA, BASIS
    SIM = Sim(); THETA = np.load(BASE)['final']; BASIS = load_basis()


def battery(pilot=False, revised=False):
    out = []
    base = (5_100_000_000 if revised else 5_000_000_000) if pilot else 6_000_000_000
    kinds = ['quiet', 'fwd-.08', 'fwd-.20', 'fwd-.28', 'fwd-.32', 'fwd-.36']
    if not pilot:
        kinds += ['fwd-.26', 'fwd-.30', 'fwd-.34', 'back-.08', 'left-.12', 'right-.12',
                  'pulse-.06', 'pulse-.18', 'pulse-.30', 'false-gyro', 'double-push']
    for j, kind in enumerate(kinds):
        for rep in range(4):
            seed = base + j * 10 + rep
            rng = np.random.default_rng(seed)
            t = float(rng.uniform(2.2, 3.0))
            c = Condition(seed=seed, duration_s=6, joint_noise_rad=.01)
            if kind.startswith('fwd-'):
                c.push_time_s, c.push_dv = t, (float(kind[4:]), 0.)
            elif kind.startswith('pulse-'):
                c.pulse_duration_s = float(kind[6:])
                c.pulse_time_s = round(t / .02) * .02
                c.pulse_force = (.73724318 * .24 / c.pulse_duration_s, 0., 0.)
            elif kind == 'false-gyro':
                c.false_gyro_time_s = round(t / .02) * .02
                c.false_gyro_rad_s = (-1 if rep % 2 else 1) * .75
            elif kind == 'double-push':
                c.push_time_s, c.push_dv = t, (.24, 0.)
                c.extra_pushes = ((t + .3, .12, 0.),)
            elif kind != 'quiet':
                c.push_time_s = t
                c.push_dv = {'back-.08':(-.08, 0.), 'left-.12':(0., .12), 'right-.12':(0., -.12)}[kind]
            out.append((kind, c))
    return out


def shifts():
    out = []
    changes = [('nominal', {}), ('friction-.6', dict(foot_friction=.6)), ('friction-1.4', dict(foot_friction=1.4)),
               ('voltage-6.5', dict(vin=6.5)), ('actuation-30ms', dict(lag_min=6, lag_max=6)),
               ('sensor-40ms', dict(observation_delay=2)), ('sensor-80ms', dict(observation_delay=4)),
               ('pitch-2deg', dict(imu_rpy_deg=(0., 2., 0.)))]
    for name, kw in changes:
        for j, mag in enumerate((0., .20, .32)):
            for rep in range(2):
                c = Condition(seed=6_100_000_000 + j * 10 + rep, duration_s=6, joint_noise_rad=.01,
                              push_time_s=2.5 if mag else None, push_dv=(mag, 0.), **kw)
                out.append((name, c))
    return out


def load(path):
    cfg = json.loads((path / 'config.json').read_text())
    return np.load(path / 'final.npz')['best'], Design(**cfg['design'])


def controllers(root, pilot=False, revised=False):
    cs = [(name, np.zeros(22), Design(), name) for name in ('reflex', 'oracle02c', 'imu02c')]
    if pilot:
        vec, design = load(root / ('pilot/history-s72' if revised else 'pilot/history-s71'))
        cs.append(('pilot-history', vec, design, 'learned'))
    else:
        for arm in ('history', 'memoryless'):
            for seed in (1, 2, 3):
                name = f'{arm}-s{seed}'
                vec, design = load(root / 'runs' / name)
                cs.append((name, vec, design, 'learned'))
    return cs


def metrics(s, r, c):
    t = r['time_s']; onset = event_time(c)
    quiet = (t > .5) & (t <= (onset if onset is not None else c.duration_s))
    post = t > (onset if onset is not None else .5)
    strength = np.sqrt(np.mean(r['residual'] ** 2, axis=1))
    s.update(quiet_residual_rms_rad=float(np.sqrt(np.mean(r['residual'][quiet] ** 2))),
             quiet_root_path_m=float(np.linalg.norm(np.diff(r['qpos'][quiet, :2], axis=0), axis=1).sum()),
             peak_response_rad=float(strength[post].max()), response_active_s=float(np.count_nonzero(strength[post] > .02) * .005),
             peak_clearance_m=r['clearance_m'][post].max(axis=0).tolist(),
             foot_unloaded_s=((~r['foot_found'][post]).sum(axis=0) * .005).tolist(),
             post_root_delta_xy=(r['qpos'][-1, :2] - r['qpos'][np.flatnonzero(post)[0], :2]).tolist())
    return s


def effective_mode(mode, cond):
    """A true onset oracle cannot be falsely cued by a sensor-only artifact.

    Keep the event time for the physical audit, but omit the old motor option
    when no physical disturbance occurred. The standing reflex still receives
    the same corrupted IMU observations as every other controller.
    """
    if mode in ('oracle02c','delay02c') and cond.push_time_s is None and cond.pulse_time_s is None:
        return 'reflex'
    return mode


def evaluate(task):
    name, vec, design, mode, label, c = task
    s, r = rollout(SIM, THETA, vec, c, design, record=True, terminate=False, mode=effective_mode(mode,c), basis=BASIS)
    metrics(s, r, c)
    s.update(controller=name, set=label, condition=asdict(c), design=asdict(design), mode=mode)
    return s


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02d'))
    p.add_argument('--battery', choices=('pilot', 'pilot-revised', 'grid', 'shifts'), default='pilot')
    a = p.parse_args()
    pilot = a.battery.startswith('pilot')
    revised = a.battery == 'pilot-revised'
    cs = controllers(a.root, pilot, revised)
    conds = shifts() if a.battery == 'shifts' else battery(pilot, revised)
    dest = a.root / 'eval' / a.battery
    dest.mkdir(parents=True, exist_ok=True)
    identity = dict(model_sha256=digest(DEFAULT_MODEL / 'model.mjb'), base_sha256=digest(BASE), option_sha256=digest(OPTION),
                    controllers=[dict(name=n, weights=v.tolist(), design=asdict(d), mode=m) for n,v,d,m in cs],
                    conditions=[dict(label=l, condition=asdict(c)) for l,c in conds],
                    sources={n:digest(Path('movement_lab') / n) for n in
                             ('eval02d.py', 'sim02d.py', 'control02d.py', 'sim02b.py', 'sim02c.py', 'eval02c.py')})
    identity_file = dest / 'identity.json'
    identity = json.loads(json.dumps(identity))  # JSON normalizes tuple-valued conditions to lists.
    if identity_file.exists():
        if json.loads(identity_file.read_text()) != identity:
            raise SystemExit('evaluation identity mismatch; use a new output root')
    else:
        identity_file.write_text(json.dumps(identity, indent=2) + '\n')
    if hasattr(os, 'nice'):
        os.nice(10)
    with ProcessPoolExecutor(2, initializer=init_worker) as pool:
        for name, vec, design, mode in cs:
            path = dest / (name + '.jsonl.gz')
            if path.exists():
                print('retained existing', path, flush=True)
                continue
            rows = list(pool.map(evaluate, [(name, vec, design, mode, label, c) for label,c in conds], chunksize=2))
            tmp = path.with_suffix('.tmp')
            write_gzip(tmp, rows); tmp.replace(path)
            print(name, 'recoveries', sum(r['recovered_200hz'] for r in rows), '/', len(rows), flush=True)
    if pilot:
        old = read_rows(dest / 'reflex.jsonl.gz'); new = read_rows(dest / 'pilot-history.jsonl.gz')
        win = sum(y['recovered_200hz'] and not x['recovered_200hz'] for x,y in zip(old,new))
        lost = sum(x['recovered_200hz'] and not y['recovered_200hz'] for x,y in zip(old,new))
        quiet_lost = sum(x['set'] == 'quiet' and x['recovered_200hz'] and not y['recovered_200hz'] for x,y in zip(old,new))
        checkpoint = sorted((a.root / ('pilot/history-s72' if revised else 'pilot/history-s71')).glob('generation-[0-9][0-9][0-9].npz'))[-1]
        row = json.loads(str(np.load(checkpoint)['row']))
        passed = win - lost >= 2 and quiet_lost == 0 and row['winner_score'] > row['zero_score']
        gate = dict(passed=passed, paired_wins=win, paired_losses=lost, quiet_losses=quiet_lost, training=row)
        (a.root / ('pilot/gate-revised.json' if revised else 'pilot/gate.json')).write_text(json.dumps(gate, indent=2) + '\n')
        print(json.dumps(gate), flush=True)


if __name__ == '__main__':
    main()
