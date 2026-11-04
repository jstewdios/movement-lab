"""02B fidelity cross-check: replay reflexes through upstream's mjlab/MuJoCo-Warp path.

Learning ran in native CPU MuJoCo with ``bam.mujoco`` (upstream's deployment
rehearsal actuator). This script runs the *same* reflex parameters inside the
upstream ``ManagerBasedRlEnv`` (warp physics + torch BAM actuator + mjlab's own
delay buffer) on matched, noise-free conditions, then compares trajectories and
outcomes against a native rollout of the same conditions. Different solvers are
not expected to agree bitwise; the question is whether physical outcomes and
trajectory shapes transfer between the two upstream simulation paths.

Non-foot floor contact is audited by copying warp states into a CPU MjData at
the control rate (the 01A geometry-audit convention); no state is written back.

    python -m movement_lab.crosscheck02b --out results/02b/crosscheck \
        --controller home=zero --controller ars=results/02b/runs/ars-push-s1/thetas.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from movement_lab import sim02b
from movement_lab.eval02b import load_theta
from movement_lab.model02b import env_cfg
from movement_lab.sim02b import Condition, HOME, CONTROL_DT, FEATURE_SCALE

DURATION = 6.0
PUSH_TIME = 2.0


def conditions():
    out = [('stand', Condition(seed=0, duration_s=DURATION, gyro_noise=0.0, gravity_noise=0.0))]
    for i in range(8):
        a = 2 * math.pi * i / 8
        for mag in (0.1, 0.2, 0.3):
            out.append((f'push{mag:.1f}@{int(round(math.degrees(a)))}',
                        Condition(seed=0, duration_s=DURATION, gyro_noise=0.0, gravity_noise=0.0, push_time_s=PUSH_TIME,
                                  push_dv=(mag * math.cos(a), mag * math.sin(a)))))
    return out


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def run_mjlab(env, sim, theta, conds):
    import mujoco
    import torch
    n = len(conds)
    m = env.sim.mj_model
    gyro_adr = int(m.sensor('robot/imu_ang_vel').adr[0])
    trunk = sim.trunk
    env.reset()
    data = env.sim.data
    qpos0 = data.qpos[:].detach().cpu().numpy().copy()
    n_ctrl = int(round(DURATION / CONTROL_DT))
    push_step = int(math.ceil(PUSH_TIME / CONTROL_DT - 1e-9))
    tilt = np.zeros((n_ctrl, n)); root = np.zeros((n_ctrl, n, 3)); nonfoot = np.zeros((n_ctrl, n), bool)
    aux = mujoco.MjData(sim.m)
    for k in range(n_ctrl):
        if k == push_step:
            dv = torch.tensor([c.push_dv for _, c in conds], dtype=data.qvel[:].dtype, device=data.qvel[:].device)
            data.qvel[:, 0:2] += dv
            env.sim.forward()
        qpos = data.qpos[:].detach().cpu().numpy()
        gyro = data.sensordata[:, gyro_adr:gyro_adr + 3].detach().cpu().numpy()
        actions = np.zeros((n, len(HOME)))
        for e in range(n):
            R = quat_to_mat(qpos[e, 3:7])
            g = R.T @ np.array([0.0, 0.0, -1.0])
            feat = np.array([g[0], g[1], *gyro[e]]) / FEATURE_SCALE
            target = np.clip(HOME + sim02b.action(theta, feat), sim.lo, sim.hi)
            actions[e] = target - HOME
        env.step(torch.tensor(actions, dtype=torch.float32, device=data.qpos[:].device))
        qpos = data.qpos[:].detach().cpu().numpy()
        for e in range(n):
            R = quat_to_mat(qpos[e, 3:7])
            tilt[k, e] = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
            root[k, e] = qpos[e, :3]
            aux.qpos[:] = qpos[e]
            mujoco.mj_kinematics(sim.m, aux)
            mujoco.mj_collision(sim.m, aux)
            nonfoot[k, e] = sim.contacts(aux)[2] > 0
    return qpos0, tilt, root, nonfoot


def run_native(sim, theta, conds):
    n_ctrl = int(round(DURATION / CONTROL_DT))
    tilt = np.full((n_ctrl, len(conds)), np.nan); root = np.full((n_ctrl, len(conds), 3), np.nan)
    summaries = []
    for e, (_, c) in enumerate(conds):
        s, rec = sim02b.rollout(sim, theta, c, record=True, terminate=False)
        # control-rate samples are every 4th physics sample (end of each period)
        tilt[:, e] = rec['tilt_deg'][3::4][:n_ctrl]
        root[:, e] = rec['qpos'][3::4, :3][:n_ctrl]
        summaries.append(s)
    return tilt, root, summaries


def outcome(tilt, root, nonfoot, z0):
    fell = (tilt > sim02b.FALL_TILT_DEG) | (root[..., 2] < z0 - sim02b.FALL_HEIGHT_LOSS)
    if nonfoot is not None:
        fell |= nonfoot
    return ~fell.any(axis=0)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--controller', action='append', required=True)
    p.add_argument('--device', default='cpu')
    p.add_argument('--model', type=Path, default=sim02b.DEFAULT_MODEL)
    args = p.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    from mjlab.envs import ManagerBasedRlEnv
    conds = conditions()
    sim = sim02b.Sim(args.model)
    started = time.perf_counter()
    env = ManagerBasedRlEnv(env_cfg(num_envs=len(conds)), device=args.device)
    report = dict(device=args.device, duration_s=DURATION, push_time_s=PUSH_TIME, conditions=[n for n, _ in conds],
                  model_sha256=sim.info['mjb_sha256'], controllers={})
    try:
        import mujoco
        warp_model = env.sim.mj_model
        same = all(np.array_equal(getattr(warp_model, f), getattr(sim.m, f)) for f in
                   ('body_mass', 'geom_size', 'jnt_range', 'dof_armature', 'geom_friction', 'actuator_forcerange'))
        report['model_fields_equal'] = bool(same)
        for spec in args.controller:
            name, _, src = spec.partition('=')
            theta = load_theta(src)
            t0 = time.perf_counter()
            qpos0, wt, wr, wnf = run_mjlab(env, sim, theta, conds)
            t_warp = time.perf_counter() - t0
            nt, nr, ns = run_native(sim, theta, conds)
            ws = outcome(wt, wr, wnf, sim.z0)
            nsurv = np.array([s['survived'] for s in ns])
            alive = ~np.isnan(nt)
            rms = np.sqrt(np.nanmean((wt - nt) ** 2, axis=0))
            disp_w = np.linalg.norm(wr[-1, :, :2] - wr[0, :, :2], axis=-1)
            disp_n = np.linalg.norm(nr[-1, :, :2] - nr[0, :, :2], axis=-1)
            report['controllers'][name] = dict(
                source=src, warp_wall_s=t_warp,
                # x/y excluded: mjlab places each env at its own grid origin.
                initial_qpos_max_abs_diff=float(np.abs(qpos0[:, 2:] - np.r_[sim.z0, 1, 0, 0, 0, HOME][None]).max()),
                warp_survived=ws.astype(int).tolist(), native_survived=nsurv.astype(int).tolist(),
                outcome_agreement=float(np.mean(ws == nsurv)),
                tilt_rms_diff_deg=rms.tolist(), warp_max_tilt=np.max(wt, axis=0).tolist(), native_max_tilt=np.nanmax(nt, axis=0).tolist(),
                warp_final_disp_mm=(1000 * disp_w).tolist(), native_final_disp_mm=(1000 * disp_n).tolist(),
                tilt_trace_warp=np.round(wt[::5], 3).tolist(), tilt_trace_native=np.round(nt[::5], 3).tolist())
            print(name, 'agreement', report['controllers'][name]['outcome_agreement'],
                  'warp', ws.astype(int).tolist(), 'native', nsurv.astype(int).tolist(), f'{t_warp:.0f}s')
    finally:
        env.close()
    report['wall_s'] = time.perf_counter() - started
    (args.out / 'crosscheck.json').write_text(json.dumps(report, indent=1), newline='\n')


if __name__ == '__main__':
    main()
