"""02D rollout: unchanged BAM stepping and 02C validity, sensor-driven residual.

Forked locally from sim02c rather than changing any historical implementation.
All external interventions live in the environment; the controller receives only
five normalized IMU readings. Full-rate measurement is passive on a data copy.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from movement_lab.sim02b import (Sim, Condition as BaseCondition, HOME, CONTROL_DT,
    DT, DECIMATION, FEATURE_SCALE, rpy_matrix, action)
from movement_lab.sim02c import BASE, Design as FixedDesign, residual, audit
from movement_lab.control02d import Controller, Design, load_basis
from movement_lab.eval02c import audit_record


@dataclass
class Condition(BaseCondition):
    observation_delay: int = 0  # 20 ms samples; all IMU channels, also the reflex
    false_gyro_time_s: float | None = None
    false_gyro_duration_s: float = .04
    false_gyro_rad_s: float = 0.0


def event_time(cond):
    times = [t for t in (cond.push_time_s, cond.pulse_time_s, cond.false_gyro_time_s) if t is not None]
    return None if not times else math.ceil(min(times) / CONTROL_DT - 1e-9) * CONTROL_DT


def quiet(tilt, angular, linear):
    return bool(tilt < 5.0 and angular < 0.5 and linear < 0.05)


def rollout(sim, theta, vec, cond, design=Design(), record=False, terminate=True, mode="learned", basis=None):
    """Native BAM/MuJoCo ordering is shared with 02B; all gates are latched.

    Training measures at 50 Hz. Recorded evaluation additionally checks every
    200 Hz sample, so an inter-control fall/contact cannot become a success.
    Recording uses a full MjData copy and cannot affect the simulated state.
    """
    mj, m, d = sim.mujoco, sim.m, sim.d
    basis = load_basis() if basis is None else basis
    policy = Controller(vec, basis, design) if mode == 'learned' else None
    fixed = FixedDesign(knots=2, horizon=.4, trigger='imu' if mode == 'imu02c' else 'event',
                        delay=.08 if mode == 'delay02c' else 0.)
    actual_event = event_time(cond)
    observed = []
    residual_cost = 0.
    peak_torque = 0.
    pulse_body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, cond.pulse_body)
    assert pulse_body >= 0
    pushes = {math.ceil(t / CONTROL_DT - 1e-9): (x, y) for t, x, y in cond.extra_pushes}
    rng = np.random.default_rng(cond.seed)
    sim.reset(cond, rng)
    imu = rpy_matrix(cond.imu_rpy_deg)
    lag_hist = np.repeat(HOME[None], 8, axis=0)
    ptr = phys = 0
    gyro_hist = []
    push_step = None if cond.push_time_s is None else int(math.ceil(cond.push_time_s / CONTROL_DT - 1e-9))
    if push_step is not None:
        pushes[push_step] = cond.push_dv
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
                          'nonfoot', 'penetration_m', 'com', 'xpos', 'clearance_m',
                          'features', 'coefficients', 'target_cmd', 'self_contacts')} if record else None
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
        if actual_event is not None and abs(t - actual_event) < 1e-8:
            pre_state = d.qpos.copy()
            foot0 = d.xpos[list(sim.foot_bodies)].copy()
        if k in pushes:
            d.qvel[:2] += pushes[k]
            mj.mj_forward(m, d)
        R = d.xmat[sim.trunk].reshape(3, 3) @ imu
        grav = R.T @ np.array([0., 0., -1.]) + rng.uniform(-1, 1, 3) * cond.gravity_noise
        gyro = imu.T @ d.sensordata[sim.gyro_adr:sim.gyro_adr + 3] + rng.uniform(-1, 1, 3) * cond.gyro_noise
        if cond.false_gyro_time_s is not None and cond.false_gyro_time_s <= t < cond.false_gyro_time_s + cond.false_gyro_duration_s - 1e-9:
            gyro[1] += cond.false_gyro_rad_s
        gyro_hist.append(gyro)
        gyro_obs = gyro_hist[max(0, len(gyro_hist) - 1 - cond.gyro_lag)]
        feat = np.array([grav[0], grav[1], *gyro_obs]) / FEATURE_SCALE
        observed.append(feat)
        feat = observed[max(0, len(observed) - 1 - cond.observation_delay)]
        if policy is not None:
            r = policy(feat)
        elif mode == 'reflex':
            r = np.zeros(14)
        else:
            if trigger_time is None:
                if mode in ('oracle02c', 'delay02c') and actual_event is not None and t >= actual_event:
                    trigger_time = actual_event + fixed.delay
                elif mode == 'imu02c' and t >= .8 and abs(feat[3]) > .5:
                    trigger_time = t
                elif mode == 'clock02c' and t >= 1.5:
                    trigger_time = 1.5
            r = residual(np.asarray(basis).ravel(), None if trigger_time is None else t - trigger_time, fixed)
        residual_cost += float(np.mean((r / design.bound) ** 2))
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
            d.xfrc_applied[:] = 0
            if cond.pulse_time_s is not None:
                p0 = round(cond.pulse_time_s / DT)
                if p0 <= phys < p0 + round(cond.pulse_duration_s / DT):
                    d.xfrc_applied[pulse_body, :3] = cond.pulse_force
            sim.ctrl.update()
            mj.mj_step(m, d)
            phys += 1
            peak_torque = max(peak_torque, float(np.abs(d.ctrl).max()))
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
                            xpos=aux.xpos.copy(), clearance_m=clearance,
                            features=feat.copy(), coefficients=policy.coefficients.copy() if policy else np.zeros(2),
                            target_cmd=target.copy(), self_contacts=sum(c.geom1 != sim.floor and c.geom2 != sim.floor for c in aux.contact))
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
        if actual_event is not None and actual_event - .5 - 1e-8 <= t < actual_event - 1e-8:
            pre_window.append(q)
        if first_failure is not None and terminate:
            break
    survived = first_failure is None and k == n - 1
    settled = len(pre_window) == 25 and all(pre_window)
    recovered = survived and len(final_window) == 50 and all(final_window) and (settled or actual_event is None)
    # No movement, contact switch, foot lift or distance term in this objective.
    score = (alive_steps - .25 * tilt_cost) / n + float(recovered)
    score -= .002 * rate_cost / n + .02 * residual_cost / n
    summary = dict(fitness=float(score), survived=bool(survived), recovered=bool(recovered), settled=bool(settled),
                   failure_time_s=first_failure, failure_cause=cause, max_pen_m=float(max_pen),
                   max_tilt_deg=float(max_tilt), trigger_time_s=trigger_time, work_j=energy,
                   final_root_xy=d.qpos[:2].tolist(), final_root_z=float(d.qpos[2]),
                   residual_rms_rad=float(design.bound * np.sqrt(residual_cost / (k + 1))),
                   peak_torque_nm=peak_torque, final_tilt_deg=float(tilt), simulated_s=float(d.time),
                   pre_qpos=None if pre_state is None else pre_state.tolist(),
                   foot_shift_m=None if foot0 is None else np.linalg.norm(d.xpos[list(sim.foot_bodies), :2] - foot0[:, :2], axis=1).tolist())
    if record:
        rec = {k: np.asarray(v) for k, v in rec.items()}
        summary.update(audit_record(rec, replace(cond, push_time_s=actual_event), sim.z0, sim=sim))
        summary['self_contact_samples'] = int(np.count_nonzero(rec['self_contacts']))
        summary['torque_within_envelope'] = bool(peak_torque <= cond.vin * sim.kt_over_r + 1e-9)
        summary['targets_within_limits'] = bool(np.all(rec['target_cmd'] >= sim.lo - 1e-12) and np.all(rec['target_cmd'] <= sim.hi + 1e-12))
        summary['physical_valid'] &= summary['torque_within_envelope'] and summary['targets_within_limits'] and summary['self_contact_samples'] == 0
        summary['recovered_200hz'] &= summary['physical_valid']
        summary['recovery_with_step'] &= summary['recovered_200hz']
        summary['actual_event_time_s'] = actual_event
    return summary, rec

