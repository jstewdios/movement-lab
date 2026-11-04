"""02B independent evaluation of reflex controllers on held-out conditions.

Every evaluation seed is >= 900,000,000 (training < 500M, probes 500M+); no
evaluation outcome is ever returned to a learner. All controllers in one call
see exactly the same conditions, so outcomes are paired per condition.

Batteries
  push_grid   12 directions x 10 magnitudes x 2 seeds, push at 3.0 s (training
              pushed at 1.0-2.0 s and never above 0.4 m/s), 7 s trials.
  stand_long  20 s unpushed with 2.5x larger initial joint noise.
  pulse       sustained horizontal forces (trunk 0.1 s / 0.3 s, head 0.1 s),
              equal impulse to a 0.1/0.2/0.3 m/s whole-body velocity change.
  repeated    three 0.15 m/s pushes 2 s apart in different directions.
  shift:*     physics/sensor changes never seen in training, each on the same
              8-direction x {0, .1, .2, .3} m/s grid as shift:nominal.

    python -m movement_lab.eval02b --out results/02b/eval/final \
        --controller home=zero --controller ars-push-s1=results/02b/runs/ars-push-s1/thetas.npz
"""
from __future__ import annotations

import argparse
import dataclasses
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
from movement_lab.sim02b import Condition

EVAL_SEED_BASE = 900_000_000
NOISE = 0.02
TOTAL_MASS = 0.73724318  # compiled model mass; checked against model.json at run time


def _dirs(n):
    return [2 * math.pi * i / n for i in range(n)]


def push_grid():
    out, s = [], EVAL_SEED_BASE
    for rep in range(2):
        for i, a in enumerate(_dirs(12)):
            for j, mag in enumerate(np.round(np.arange(0.05, 0.501, 0.05), 3)):
                s += 1
                out.append(('push_grid', Condition(seed=s, duration_s=7.0, joint_noise_rad=NOISE, push_time_s=3.0,
                                                   push_dv=(float(mag * math.cos(a)), float(mag * math.sin(a))))))
    return out


def stand_long():
    return [('stand_long', Condition(seed=EVAL_SEED_BASE + 10_000 + i, duration_s=20.0, joint_noise_rad=0.05)) for i in range(8)]


def pulses():
    out, s = [], EVAL_SEED_BASE + 20_000
    for kind, body, dur in (('trunk_0.1s', 'robot/trunk_base', 0.1), ('trunk_0.3s', 'robot/trunk_base', 0.3), ('head_0.1s', 'robot/jaw_soft', 0.1)):
        for a in _dirs(8):
            for dv in (0.1, 0.2, 0.3):
                s += 1
                f = dv * TOTAL_MASS / dur
                out.append((f'pulse:{kind}', Condition(seed=s, duration_s=7.0, joint_noise_rad=NOISE, pulse_body=body,
                                                       pulse_force=(f * math.cos(a), f * math.sin(a), 0.0),
                                                       pulse_time_s=3.0, pulse_duration_s=dur)))
    return out


def repeated():
    out = []
    for i, a in enumerate(_dirs(8)):
        dirs = [a, a + 2.3, a + 4.1]
        pv = [(0.15 * math.cos(b), 0.15 * math.sin(b)) for b in dirs]
        out.append(('repeated', Condition(seed=EVAL_SEED_BASE + 30_000 + i, duration_s=9.0, joint_noise_rad=NOISE,
                                          push_time_s=2.0, push_dv=pv[0],
                                          extra_pushes=((4.0, *pv[1]), (6.0, *pv[2])))))
    return out


SHIFTS = {
    'nominal': {},
    'vin_6.5V': dict(vin=6.5), 'vin_8.2V': dict(vin=8.2), 'sag_0.2': dict(vin_drop_gain=0.2),
    'lag_3to6': dict(lag_min=3, lag_max=6), 'gyro_lag_1': dict(gyro_lag=1),
    'imu_pitch_+5deg': dict(imu_rpy_deg=(0.0, 5.0, 0.0)), 'imu_roll_+5deg': dict(imu_rpy_deg=(5.0, 0.0, 0.0)),
    'foot_friction_0.6': dict(foot_friction=0.6), 'head_mass_x1.2': dict(head_mass_scale=1.2),
    'sensor_noise_x3': dict(gyro_noise=0.09, gravity_noise=0.03),
}


def shifts():
    out = []
    for name, kw in SHIFTS.items():
        s = EVAL_SEED_BASE + 40_000  # same seeds for every shift: paired against nominal
        for i, a in enumerate(_dirs(8)):
            for mag in (0.0, 0.1, 0.2, 0.3):
                s += 1
                push = dict(push_time_s=3.0, push_dv=(mag * math.cos(a), mag * math.sin(a))) if mag else {}
                out.append((f'shift:{name}', Condition(seed=s, duration_s=7.0, joint_noise_rad=NOISE, **push, **kw)))
    return out


def nominal():
    """Only the nominal shift grid (same seeds as shift:nominal) for ablations."""
    return [c for c in shifts() if c[0] == 'shift:nominal']


BATTERIES = dict(push_grid=push_grid, stand_long=stand_long, pulse=pulses, repeated=repeated, shift=shifts, nominal=nominal)
DEFAULT_BATTERIES = ('push_grid', 'stand_long', 'pulse', 'repeated', 'shift')
KEEP = ('survived', 'recovered', 'settle_time_s', 'fall_time_s', 'fall_cause', 'valid', 'max_tilt_deg', 'max_tilt_post_deg',
        'final_tilt_deg', 'final_root_disp_m', 'max_pen_alive_m', 'sat_frac', 'peak_torque_nm', 'work_j', 'action_rms',
        'rate_rms_rad_per_step', 'max_ang_speed', 'min_root_z', 'n_steps', 'transitions', 'foot_slide_m', 'fitness', 'pre_push_tilt_deg')

_SIM = None


def _init(model_dir):
    global _SIM
    _SIM = sim02b.Sim(model_dir)


def _run(task):
    name, theta, set_name, cond = task
    if isinstance(theta, tuple):  # ('residual', parameter vector) from revise02b
        from movement_lab.revise02b import Residual
        s, _ = sim02b.rollout(_SIM, sim02b.zero_theta(), cond, terminate=True, grace_s=0.0, policy=Residual(theta[1]))
    else:
        s, _ = sim02b.rollout(_SIM, theta, cond, terminate=True, grace_s=0.0)
    row = {k: s[k] for k in KEEP}
    row['steps'] = [st for st in s['steps'] if st['is_step']]
    return dict(controller=name, set=set_name, condition=cond.to_dict(), **row)


def wilson(k, n, z=1.96):
    if n == 0:
        return (None, None)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (c - h, c + h)


def aggregate(rows):
    groups = {}
    for r in rows:
        groups.setdefault((r['controller'], r['set']), []).append(r)
    out = {}
    for (c, s), rs in sorted(groups.items()):
        n, k = len(rs), sum(r['survived'] for r in rs)
        rec = sum(r['recovered'] for r in rs)
        surv = [r for r in rs if r['survived']]
        out.setdefault(c, {})[s] = dict(
            n=n, survived=k, survival=k / n, survival_ci95=wilson(k, n), recovered=rec, recovery=rec / n,
            valid=sum(r['valid'] for r in rs), steps=sum(r['n_steps'] for r in rs),
            median_settle_s=float(np.median([r['settle_time_s'] for r in surv if r['settle_time_s'] is not None])) if any(r['settle_time_s'] is not None for r in surv) else None,
            mean_max_tilt_survivors=float(np.mean([r['max_tilt_deg'] for r in surv])) if surv else None,
            mean_root_disp_survivors_mm=float(1000 * np.mean([r['final_root_disp_m'] for r in surv])) if surv else None,
            max_sat_frac=float(max(r['sat_frac'] for r in rs)), max_peak_torque_nm=float(max(r['peak_torque_nm'] for r in rs)),
            mean_work_j_survivors=float(np.mean([r['work_j'] for r in surv])) if surv else None,
            fall_causes={c2: sum(r['fall_cause'] == c2 for r in rs) for c2 in sorted({r['fall_cause'] for r in rs if r['fall_cause']})})
    return out


# Left-right reflection (upstream tasks/symmetry.py): joint permutation/signs and
# IMU feature signs (gravity_y, gyro_x/roll, gyro_z/yaw flip; bias column stays).
JOINT_PERM = [9, 10, 11, 12, 13, 5, 6, 7, 8, 0, 1, 2, 3, 4]
JOINT_SIGN = np.array([-1, -1, -1, -1, -1, 1, 1, -1, -1, -1, -1, -1, -1, -1], float)
FEATURE_SIGN = np.array([1, -1, -1, 1, -1, 1], float)
LEGS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
NECK_HEAD = [5, 6, 7, 8]


def mirror(theta):
    """The reflex a left-right reflected robot would run: theta'[i,f] = s_i theta[p_i,f] s_f."""
    return JOINT_SIGN[:, None] * np.asarray(theta)[JOINT_PERM] * FEATURE_SIGN[None]


def symmetry_fraction(theta):
    """Share of the reflex's squared norm that is mirror-symmetric (0.5 = unstructured)."""
    sym = 0.5 * (theta + mirror(theta))
    return float((sym ** 2).sum() / max(1e-12, (np.asarray(theta) ** 2).sum()))


def ablations(theta):
    """Mechanism probes: remove one ingredient of a learned reflex at a time."""
    th = np.asarray(theta, float)
    cols = lambda keep: th * np.array([*(1.0 if i in keep else 0.0 for i in range(5)), 1.0])[None]
    rows = lambda keep: th * np.array([1.0 if j in keep else 0.0 for j in range(14)])[:, None]
    out = {
        'full': th,
        'posture_only': cols(()),                       # learned bias, no feedback
        'feedback_only': np.concatenate([th[:, :5], np.zeros((14, 1))], 1),
        'tilt_feedback_only': cols((0, 1)),             # gravity x/y + bias
        'rate_feedback_only': cols((2, 3, 4)),          # gyro x/y/z + bias
        'sagittal_only': cols((0, 3)),                  # pitch tilt + pitch rate + bias
        'lateral_only': cols((1, 2)),                   # roll tilt + roll rate + bias
        'legs_only': rows(LEGS),
        'neck_head_only': rows(NECK_HEAD),
        'mirror_symmetrized': 0.5 * (th + mirror(th)),
        'mirrored': mirror(th),
    }
    # Drop one joint's whole reflex row (feedback and bias): which joints matter?
    for j, name in enumerate(sim02b.JOINTS):
        out[f'drop:{name}'] = rows([i for i in range(14) if i != j])
    return out


def read_results(path):
    """Per-trial rows from results.jsonl, or the published results.jsonl.gz copy."""
    import gzip
    path = Path(path)
    if (path / 'results.jsonl').exists():
        text = (path / 'results.jsonl').read_text()
    elif (path / 'results.jsonl.gz').exists():
        text = gzip.decompress((path / 'results.jsonl.gz').read_bytes()).decode()
    else:
        return None
    return [json.loads(l) for l in text.splitlines()]


def publish_results(path):
    """Write a deterministic gzip copy (no timestamp) of results.jsonl for Git."""
    import gzip
    path = Path(path)
    # LF content whichever platform wrote the local file (Windows text mode
    # once wrote CRLF), so the published bytes do not depend on the platform.
    raw = (path / 'results.jsonl').read_bytes().replace(b'\r\n', b'\n')
    (path / 'results.jsonl.gz').write_bytes(gzip.compress(raw, mtime=0))


def load_theta(spec):
    """'zero' | path/thetas.npz[:key] | residual:path/snapshots.npz | linear:path/snapshots.npz"""
    if spec == 'zero':
        return sim02b.zero_theta()
    kind, sep, rest = spec.partition(':')
    if sep and kind in ('residual', 'linear'):
        vec = np.load(rest)['final']
        # linear: on a residual snapshot keeps only its linear reflex (residual removed).
        n_lin = int(np.prod(sim02b.PARAM_SHAPE))
        return ('residual', vec) if kind == 'residual' else vec[:n_lin].reshape(sim02b.PARAM_SHAPE)
    path, _, key = spec.partition(':')
    z = np.load(path)
    return z[key or 'final']


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--controller', action='append', default=[], help='name=zero | name=path/thetas.npz[:key]')
    p.add_argument('--ablate', action='append', default=[], help='name=path/thetas.npz: evaluate every ablation variant')
    p.add_argument('--battery', action='append', choices=sorted(BATTERIES), help='default: all')
    p.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument('--model', type=Path, default=sim02b.DEFAULT_MODEL)
    args = p.parse_args(argv)
    info = json.loads((args.model / 'model.json').read_text())
    assert abs(info['total_mass'] - TOTAL_MASS) < 1e-6
    args.out.mkdir(parents=True, exist_ok=False)
    ctrls = {}
    for spec in args.controller:
        name, _, src = spec.partition('=')
        ctrls[name] = (src, load_theta(src))
    for spec in args.ablate:
        name, _, src = spec.partition('=')
        for var, th in ablations(load_theta(src)).items():
            ctrls[f'{name}/{var}'] = (f'{src} [{var}]', th)
    assert ctrls, 'no controllers'
    batteries = args.battery or list(DEFAULT_BATTERIES)
    conds = [c for b in batteries for c in BATTERIES[b]()]
    tasks = [(name, th, s, c) for name, (_, th) in ctrls.items() for s, c in conds]
    started = time.perf_counter()
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(str(args.model),)) as pool:
        rows = list(pool.map(_run, tasks, chunksize=4))
    wall = time.perf_counter() - started
    with open(args.out / 'results.jsonl', 'w', newline='\n') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    publish_results(args.out)
    summary = dict(controllers={k: v[0] for k, v in ctrls.items()},
                   thetas={k: (v[1][1].tolist() if isinstance(v[1], tuple) else v[1].tolist()) for k, v in ctrls.items()},
                   model_sha256=info['mjb_sha256'], n_conditions=len(conds), wall_s=wall,
                   batteries=batteries, results=aggregate(rows))
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=1), newline='\n')
    for c, sets in summary['results'].items():
        print(c, {s: f"{v['survived']}/{v['n']}" for s, v in sets.items()})
    print(f'wall {wall:.1f}s for {len(rows)} rollouts')


if __name__ == '__main__':
    main()
