"""02C: learned finite-duration residuals on the frozen 02B standing reflex.

No leg trajectory, foot placement, gait phase or displacement reward is supplied.
The initial experiment is an explicitly *event-cued* recovery option: an oracle
provides disturbance onset, and the learner chooses every joint offset. Sensor
trigger tests are separate and are never called autonomous learned detection.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from pathlib import Path
import numpy as np

from movement_lab.sim02b import (Sim, Condition, HOME, CONTROL_DT, DT, DECIMATION,
                                FEATURE_SCALE, rpy_matrix, action)

BASE = Path('results/02b/runs/ars-push-s1/thetas.npz')


@dataclass(frozen=True)
class Design:
    knots: int = 4
    horizon: float = 0.8
    bound: float = 0.8
    trigger: str = 'event'
    delay: float = 0.0
    objective: str = 'balance'


def residual(vec, elapsed, design):
    """Smoothstep interpolation from zero, through learned knots, back to zero."""
    if elapsed is None or elapsed <= 0 or elapsed >= design.horizon:
        return np.zeros(14)
    knots = np.vstack((np.zeros(14), np.asarray(vec).reshape(design.knots, 14), np.zeros(14)))
    u = elapsed / design.horizon * (design.knots + 1)
    i = min(int(u), design.knots)
    f = u - i
    f = f * f * (3 - 2 * f)
    return np.clip(knots[i] * (1 - f) + knots[i + 1] * f, -design.bound, design.bound)


def quiet(tilt, angular, linear):
    return bool(tilt < 5.0 and angular < 0.5 and linear < 0.05)


def rollout(sim, theta, vec, cond, design=Design(), record=False, terminate=True):
    """Native BAM/MuJoCo ordering is shared with 02B; all gates are latched.

    Training measures at 50 Hz. Recorded evaluation additionally checks every
    200 Hz sample, so an inter-control fall/contact cannot become a success.
    Recording uses a full MjData copy and cannot affect the simulated state.
    """
    mj, m, d = sim.mujoco, sim.m, sim.d
    rng = np.random.default_rng(cond.seed)
    sim.reset(cond, rng)
    imu = rpy_matrix(cond.imu_rpy_deg)
    lag_hist = np.repeat(HOME[None], 8, axis=0)
    ptr = phys = 0
    gyro_hist = []
    push_step = None if cond.push_time_s is None else int(math.ceil(cond.push_time_s / CONTROL_DT - 1e-9))
    n = int(round(cond.duration_s / CONTROL_DT))
    trigger_time = None
    first_failure = None
    cause = None
    max_pen = 0.0
    alive_steps = 0
    tilt_cost = 0.0
    energy = 0.0
    prev_a = np.zeros(14)
    rate_cost = 0.0
    final_window = []
    pre_window = []
    onsets = []
    max_tilt = 0.0
    root0 = d.qpos[:3].copy()
    pre_state = None
    foot0 = None
    rec = {k: [] for k in ('time_s', 'qpos', 'qvel', 'offset', 'residual', 'target_applied',
                          'torque_nm', 'tilt_deg', 'foot_found', 'foot_force_n', 'foot_pos',
                          'nonfoot', 'penetration_m', 'com', 'xpos', 'clearance_m')} if record else None
    # Hull vertices suffice for exact sole minimum z, avoiding full-mesh work.
    if record and not hasattr(sim, 'sole_vertices02c'):
        from scipy.spatial import ConvexHull
        sim.sole_vertices02c = []
        for g in sim.feet:
            mesh = m.geom_dataid[g]
            a, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            v = m.mesh_vert[a:a + count]
            sim.sole_vertices02c.append(v[ConvexHull(v).vertices].copy())
    for k in range(n):
        t = k * CONTROL_DT
        if k == push_step:
            pre_state = d.qpos.copy()
            foot0 = d.xpos[list(sim.foot_bodies)].copy()
            d.qvel[:2] += cond.push_dv
            mj.mj_forward(m, d)
        R = d.xmat[sim.trunk].reshape(3, 3) @ imu
        grav = R.T @ np.array([0., 0., -1.]) + rng.uniform(-1, 1, 3) * cond.gravity_noise
        gyro = imu.T @ d.sensordata[sim.gyro_adr:sim.gyro_adr + 3] + rng.uniform(-1, 1, 3) * cond.gyro_noise
        gyro_hist.append(gyro)
        gyro_obs = gyro_hist[max(0, len(gyro_hist) - 1 - cond.gyro_lag)]
        feat = np.array([grav[0], grav[1], *gyro_obs]) / FEATURE_SCALE
        if trigger_time is None:
            if design.trigger == 'event' and push_step is not None and k >= push_step:
                trigger_time = push_step * CONTROL_DT + design.delay
            elif design.trigger == 'imu' and t >= 0.8 and abs(gyro_obs[1]) > 0.5:
                trigger_time = t + design.delay
            elif design.trigger == 'clock' and t >= 1.5:
                trigger_time = 1.5 + design.delay
        r = residual(vec, None if trigger_time is None else t - trigger_time, design)
        a = np.clip(action(theta, feat) + r, -design.bound, design.bound)
        target = np.clip(HOME + a, sim.lo, sim.hi)
        rate_cost += float(np.mean((a - prev_a) ** 2))
        prev_a = a
        for _ in range(DECIMATION):
            ptr = (ptr + 1) % len(lag_hist)
            lag_hist[ptr] = target
            lag = cond.lag_min if cond.lag_min == cond.lag_max else int(rng.integers(cond.lag_min, cond.lag_max + 1))
            applied = lag_hist[(ptr - min(lag, phys)) % len(lag_hist)]
            sim.ctrl.q_target = applied.copy()
            sim.ctrl.update()
            mj.mj_step(m, d)
            phys += 1
            energy += float(np.abs(d.ctrl * d.qvel[sim.vadr]).sum()) * DT
            if record:
                aux = sim.aux
                mj.mj_copyData(aux, m, d)
                mj.mj_forward(m, aux)
                lf, rf, nf, pen, force = sim.contacts(aux)
                clearance = [float((v @ aux.geom_xmat[g].reshape(3, 3).T + aux.geom_xpos[g])[:, 2].min())
                             for g, v in zip(sim.feet, sim.sole_vertices02c)]
                vals = dict(time_s=d.time, qpos=d.qpos.copy(), qvel=d.qvel.copy(), offset=a.copy(), residual=r.copy(),
                            target_applied=applied.copy(), torque_nm=d.ctrl.copy(), tilt_deg=sim.tilt_deg(aux),
                            foot_found=(lf, rf), foot_force_n=force, nonfoot=nf, penetration_m=pen,
                            foot_pos=aux.xpos[list(sim.foot_bodies)].copy(), com=aux.subtree_com[sim.trunk].copy(),
                            xpos=aux.xpos.copy(), clearance_m=clearance)
                for key in rec:
                    rec[key].append(vals[key])
        mj.mj_forward(m, d)
        tilt = sim.tilt_deg()
        _, _, nonfoot, pen, _ = sim.contacts()
        max_pen = max(max_pen, pen)
        max_tilt = max(max_tilt, tilt)
        finite = np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()
        bad = ('nonfinite' if not finite else 'nonfoot' if nonfoot else 'penetration' if pen > .003 else
               'tilt' if tilt > 45 else 'height' if d.qpos[2] < sim.z0 - .04 else None)
        if first_failure is None and bad:
            first_failure, cause = float(d.time), bad
        if first_failure is None:
            alive_steps += 1
            tilt_cost += min((tilt / 45) ** 2, 1.0)
        q = quiet(tilt, np.linalg.norm(d.qvel[3:6]), np.linalg.norm(d.qvel[:2]))
        if k >= n - 50:
            final_window.append(q)
        if push_step is not None and push_step - 25 <= k < push_step:
            pre_window.append(q)
        if first_failure is not None and terminate:
            break
    survived = first_failure is None and k == n - 1
    settled = len(pre_window) == 25 and all(pre_window)
    recovered = survived and len(final_window) == 50 and all(final_window) and (settled or push_step is None)
    # No movement, contact switch, foot lift or distance term in this objective.
    score = (alive_steps - .25 * tilt_cost) / n + float(recovered)
    score -= .002 * rate_cost / n
    summary = dict(fitness=float(score), survived=bool(survived), recovered=bool(recovered), settled=bool(settled),
                   failure_time_s=first_failure, failure_cause=cause, max_pen_m=float(max_pen),
                   max_tilt_deg=float(max_tilt), trigger_time_s=trigger_time, work_j=energy,
                   final_root_xy=d.qpos[:2].tolist(), final_root_z=float(d.qpos[2]),
                   final_tilt_deg=float(tilt), simulated_s=float(d.time),
                   pre_qpos=None if pre_state is None else pre_state.tolist(),
                   foot_shift_m=None if foot0 is None else np.linalg.norm(d.xpos[list(sim.foot_bodies), :2] - foot0[:, :2], axis=1).tolist())
    if record:
        rec = {k: np.asarray(v) for k, v in rec.items()}
        summary.update(audit(rec, cond, sim.z0))
    return summary, rec


def audit(rec, cond, z0):
    """Conservative, independent physics-rate recovery and airborne relocation audit."""
    t = rec['time_s']
    finite = all(np.isfinite(v).all() for v in rec.values())
    valid = finite and not np.any(rec['nonfoot']) and rec['penetration_m'].max() <= .003
    upright = bool(rec['tilt_deg'].max() < 45 and rec['qpos'][:, 2].min() > z0 - .04)
    done = t[-1] >= cond.duration_s - DT / 2
    good = (rec['tilt_deg'] < 5) & (np.linalg.norm(rec['qvel'][:, 3:6], axis=1) < .5) & (np.linalg.norm(rec['qvel'][:, :2], axis=1) < .05)
    pre = np.ones(len(t), bool) if cond.push_time_s is None else (t > cond.push_time_s - .5) & (t <= cond.push_time_s + 1e-8)
    settled = bool(good[pre].all()) if cond.push_time_s is not None else True
    recovered = bool(valid and upright and done and settled and good[t > cond.duration_s - 1 + 1e-8].all())
    events = []
    post = 0.0 if cond.push_time_s is None else cond.push_time_s
    for foot in range(2):
        off = None
        for i in range(1, len(t)):
            contact = rec['foot_found'][i, foot]
            if off is None and not contact and rec['foot_found'][i - 1, foot] and t[i] >= post:
                off = i
            if off is not None and contact:
                # Require ≥40 ms with no contact, ≥2 mm minimum-sole clearance,
                # ≥5 mm horizontal replant, and ≥100 ms sustained load/contact.
                air = (i - off) * DT
                shift = float(np.linalg.norm(rec['foot_pos'][i, foot, :2] - rec['foot_pos'][off - 1, foot, :2]))
                clearance = float(rec['clearance_m'][off:i, foot].max())
                end = i + 20
                stable = end <= len(t) and bool(rec['foot_found'][i:end, foot].all()) and float(rec['foot_force_n'][i:end, foot].mean()) > .5
                qualifies = air >= .04 - 1e-9 and shift > .005 and clearance > .002 and stable
                events.append(dict(foot='LR'[foot], off_s=float(t[off]), on_s=float(t[i]), air_s=float(air),
                                   shift_m=shift, clearance_m=clearance, stable_replant=bool(stable), qualifies=bool(qualifies)))
                off = None
    return dict(physical_valid=bool(valid), upright_200hz=upright, recovered_200hz=recovered,
                settled_200hz=settled, support_events=events,
                recovery_with_step=bool(recovered and any(e['qualifies'] for e in events)))
