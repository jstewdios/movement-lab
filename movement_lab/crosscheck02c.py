"""Transfer final 02C options through upstream mjlab/BAM/MuJoCo-Warp on CUDA.

Matched noise-free starts, control-rate physical gates. This is a backend
fidelity check, not an additional source of checkpoint selection.
"""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
import numpy as np

from movement_lab.model02b import env_cfg
from movement_lab.sim02b import Sim, HOME, CONTROL_DT, FEATURE_SCALE, action, Condition
from movement_lab.sim02c import BASE, residual, rollout, quiet
from movement_lab.eval02c import controllers, wide_action, simulate
from movement_lab.crosscheck02b import quat_to_mat


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    p.add_argument('--wide-only', action='store_true', help='separate check of the added capacity control')
    a = p.parse_args()
    import torch
    import mujoco
    from mjlab.envs import ManagerBasedRlEnv
    torch.set_num_threads(1)
    out = a.root / ('crosscheck-wide' if a.wide_only else 'crosscheck')
    out.mkdir(parents=True, exist_ok=False)
    cs = [x for x in controllers(a.root, controls=False) if not x[0].startswith('random') and x[0] != 'wide-reflex']
    if a.wide_only:
        cs = [x for x in controllers(a.root, controls=False) if x[0] == 'wide-reflex']
    tasks = [(name, v, d, Condition(seed=0, duration_s=6, gyro_noise=0., gravity_noise=0.,
                                   push_time_s=2.5 if mag else None, push_dv=(mag, 0.)))
             for name, v, d in cs for mag in (0., .28, .30, .32, .34, .36)]
    sim = Sim()
    theta = np.load(BASE)['final']
    started = time.perf_counter()
    env = ManagerBasedRlEnv(env_cfg(num_envs=len(tasks)), device='cuda:0')
    fields = ('body_mass', 'body_inertia', 'geom_size', 'geom_friction', 'jnt_range', 'dof_armature', 'actuator_forcerange')
    same = {f: bool(np.array_equal(getattr(env.sim.mj_model, f), getattr(sim.m, f))) for f in fields}
    if not all(same.values()):
        raise RuntimeError(f'model mismatch: {same}')
    try:
        env.reset()
        data = env.sim.data
        qpos0 = data.qpos[:].cpu().numpy().copy()
        aux = mujoco.MjData(sim.m)
        traces = {k: [] for k in ('qpos', 'qvel', 'tilt', 'nonfoot', 'penetration')}
        for k in range(300):
            if k == 125:
                data.qvel[:, :2] += torch.tensor([c.push_dv for _, _, _, c in tasks], device=env.device)
                env.sim.forward()
            q = data.qpos[:].cpu().numpy()
            gyro = data.sensordata[:, sim.gyro_adr:sim.gyro_adr + 3].cpu().numpy()
            actions = []
            for i, (name, v, design, c) in enumerate(tasks):
                R = quat_to_mat(q[i, 3:7])
                g = R.T @ np.array([0., 0., -1.])
                feat = np.r_[g[:2], gyro[i]] / FEATURE_SCALE
                dt = None if c.push_time_s is None else k * CONTROL_DT - c.push_time_s
                base = wide_action(theta, feat) if name == 'wide-reflex' else action(theta, feat)
                off = np.clip(base + residual(v, dt, design), -design.bound, design.bound)
                actions.append(np.clip(HOME + off, sim.lo, sim.hi) - HOME)
            env.step(torch.tensor(np.array(actions), dtype=torch.float32, device=env.device))
            q = data.qpos[:].cpu().numpy().copy()
            qv = data.qvel[:].cpu().numpy().copy()
            tilt, nf, pen = [], [], []
            for i in range(len(tasks)):
                aux.qpos[:] = q[i]
                mujoco.mj_kinematics(sim.m, aux)
                mujoco.mj_collision(sim.m, aux)
                _, _, nfoot, penetration, _ = sim.contacts(aux)
                tilt.append(sim.tilt_deg(aux)); nf.append(nfoot); pen.append(penetration)
            for key, val in zip(traces, (q, qv, tilt, nf, pen)):
                traces[key].append(val)
        traces = {k: np.array(v) for k, v in traces.items()}
        np.savez_compressed(out / 'warp-traces.npz', initial=qpos0, **traces)
    finally:
        env.close()
    rows = []
    for i, (name, vec, design, cond) in enumerate(tasks):
        native, rec = simulate(sim, theta, name, vec, cond, design)
        t = traces['tilt'][:, i]; qv = traces['qvel'][:, i]; q = traces['qpos'][:, i]
        bad = ((t > 45) | (q[:, 2] < qpos0[i, 2] - .04) |
               (traces['nonfoot'][:, i] > 0) | (traces['penetration'][:, i] > .003))
        good = (t < 5) & (np.linalg.norm(qv[:, 3:6], axis=1) < .5) & (np.linalg.norm(qv[:, :2], axis=1) < .05)
        survived = not bad.any()
        recovered = survived and good[-50:].all() and (cond.push_time_s is None or good[100:125].all())
        rows.append(dict(controller=name, condition=asdict(cond), design=asdict(design),
                         native_recovered_50hz=native['recovered'], warp_recovered_50hz=bool(recovered),
                         native_survived=native['survived'], warp_survived=bool(survived),
                         max_initial_qpos_diff=float(np.abs(qpos0[i, 2:] - np.r_[sim.z0, 1, 0, 0, 0, HOME]).max()),
                         native_final_xy=rec['qpos'][-1, :2].tolist(),
                         warp_displacement_xy=(q[-1, :2] - qpos0[i, :2]).tolist(),
                         tilt_rms_difference=float(np.sqrt(np.mean((t - rec['tilt_deg'][3::4]) ** 2)))))
    report = dict(model_fields_equal=same, rows=rows, wall_s=time.perf_counter() - started,
                  outcome_agreement=float(np.mean([r['native_recovered_50hz'] == r['warp_recovered_50hz'] for r in rows])))
    (out / 'results.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
