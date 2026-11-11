"""Reproduce historical outcomes without changing the historical evidence."""
import argparse
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import platform
import hashlib
import sys
import time
import numpy as np
from movement_lab.sim02b import Sim, DEFAULT_MODEL
from movement_lab.sim02c import BASE, rollout as old_rollout, Design as FixedDesign
from movement_lab.eval02c import grid, read_rows, audit_record
from movement_lab.control02d import Design, load_basis, Controller, OPTION
from movement_lab.sim02d import rollout, Condition
from movement_lab.learn02c import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, default=Path('results/02d'))
    a = p.parse_args()
    dest = a.root / 'preflight.json'
    if dest.exists():
        raise SystemExit('refusing to overwrite preflight')
    sim = Sim(); theta = np.load(BASE)['final']; basis = load_basis()
    import bam
    motor=Path(bam.__file__).parent/'params/xl330/m6.json'
    motor_canonical=hashlib.sha256(json.dumps(json.loads(motor.read_text()),sort_keys=True,separators=(',',':')).encode()).hexdigest()
    historical_motor=json.loads(Path('results/02c/preflight.json').read_text())['bam_parameter_canonical_sha256']
    if motor_canonical != historical_motor:
        raise SystemExit('BAM parameters differ from 02C')
    rows = []
    start = time.perf_counter()
    names = ('reflex02b', 'cem-s1', 'cem-s2', 'cem-s1-imu', 'cem-s1-delay80')
    for name in names:
        source = {r['condition']['seed']: r for r in read_rows(Path('results/02c/eval/grid') / (name + '.jsonl.gz'))}
        b = np.zeros(28) if name == 'reflex02b' else np.load(Path('results/02c/runs') / ('cem-s2' if name == 'cem-s2' else 'cem-s1') / 'final.npz')['best']
        design = FixedDesign(knots=2, horizon=.4, trigger='imu' if name.endswith('imu') else 'event', delay=.08 if name.endswith('80') else 0.)
        for label, c in grid():
            if c.push_dv[0] not in (0, .20, .26, .32):
                continue
            s, r = old_rollout(sim, theta, b, c, design, record=True, terminate=False)
            actual = None if c.push_time_s is None else np.ceil(c.push_time_s / .02 - 1e-9) * .02
            s.update(audit_record(r, replace(c, push_time_s=actual), sim.z0, sim))
            prior = source[c.seed]
            rows.append(dict(controller=name, seed=c.seed, magnitude=c.push_dv[0],
                             recovered=s['recovered_200hz'], historical_recovered=prior['recovered_200hz'],
                             survived=s['survived'], historical_survived=prior['survived']))
        print(name, 'reproduced', flush=True)
    c = Condition(seed=42, duration_s=4, push_time_s=1.5, push_dv=(.32, 0), joint_noise_rad=.01)
    s, r = rollout(sim, theta, np.zeros(22), c, record=True, terminate=False)
    _, old = old_rollout(sim, theta, np.zeros(28), c, FixedDesign(knots=2, horizon=.4), record=True, terminate=False)
    again, r2 = rollout(sim, theta, np.zeros(22), c, record=True, terminate=False)
    plain, _ = rollout(sim, theta, np.zeros(22), c, terminate=False)
    exact = all(np.array_equal(r[k], old[k]) for k in ('qpos', 'qvel', 'torque_nm'))
    deterministic = all(np.array_equal(r[k], r2[k]) for k in r)
    passive = s['final_root_xy'] == plain['final_root_xy'] and s['fitness'] == plain['fitness']
    controller = Controller(np.zeros(22), basis)
    start_control = time.perf_counter()
    for _ in range(10000):
        controller(np.ones(5))
    latency = (time.perf_counter() - start_control) / 10000
    info = dict(platform=platform.platform(), packages={k: importlib.metadata.version(k) for k in
                ('numpy', 'mujoco', 'mjlab', 'mujoco-warp', 'better-actuator-models')},
                model_sha256=digest(DEFAULT_MODEL / 'model.mjb'), base_sha256=digest(BASE), option_sha256=digest(OPTION),
                physics_dt_s=.005, control_dt_s=.02, nominal_command_delay_s=.02,
                nominal_added_observation_delay_s=0, controller_mean_wall_s=latency,
                zero_residual_exact_02c=exact, deterministic=deterministic, recording_passive=passive,
                historical_rows=rows, outcome_matches=sum(r['recovered'] == r['historical_recovered'] and r['survived'] == r['historical_survived'] for r in rows),
                wall_s=time.perf_counter() - start)
    info.update(python=sys.version,bam_parameter_sha256=digest(motor),bam_parameter_canonical_sha256=motor_canonical,
                bam_parameters_match_02c=True)
    info['passed'] = bool(exact and deterministic and passive and info['outcome_matches'] == len(rows))
    dest.write_text(json.dumps(info, indent=2) + '\n')
    from movement_lab.study02d import DEPENDENCIES
    (a.root/'dependencies.json').write_text(json.dumps({n:digest(Path('movement_lab')/n) for n in DEPENDENCIES},indent=2)+'\n')
    print(json.dumps({k:v for k,v in info.items() if k != 'historical_rows'}, indent=2))
    if not info['passed']:
        raise SystemExit('compatibility gate failed')


if __name__ == '__main__':
    main()
