"""02E rollout: 02D physics, sensing and audit, plus explicit calibration interventions.

Forked from sim02d (historical code is untouched). With every 02E field at its
default and no front-end/probe, the trial is bit-identical to ``sim02d.rollout``
(tested). Added environment interventions, never controller inputs:

* ``imu_rpy_deg`` (inherited) is a fixed IMU *mounting* misalignment: it rotates the
  reported gravity direction AND the gyro, as in 02D. ``drift_deg`` adds a slow
  linear change of that misalignment over ``drift_start_s .. +drift_duration_s``.
* ``slope_deg`` (roll, pitch) tilts gravity relative to the floor, which is
  physically equivalent to standing on an inclined plane. Positive pitch: the floor
  descends toward +x.
* Joint encoders (only read by an explicit front-end) carry upstream's uniform
  0.001 rad noise and optional fixed per-joint offsets ``encoder_offset_rad``,
  drawn from a separate RNG stream so the physical trial is unchanged.

A *front-end* sits between the IMU and every controller (standing reflex AND
residual). It may read raw IMU vectors, noisy encoders, its own commanded
probe and time; never contact, root state, the disturbance or the true offset.
A *probe* is a deliberate joint-target addition requested by the front-end.
"""
from __future__ import annotations
from dataclasses import dataclass, replace
import math
import numpy as np
from movement_lab.sim02b import HOME, CONTROL_DT, DT, DECIMATION, FEATURE_SCALE, rpy_matrix, action
from movement_lab.sim02c import Design as FixedDesign, residual
from movement_lab.sim02d import Condition as Condition02d, event_time, quiet
from movement_lab.control02d import Controller, Design, load_basis
from movement_lab.eval02c import audit_record
from movement_lab.physics02b import support_polygon

ENCODER_NOISE_RAD = 0.001   # upstream actor joint-position noise (uniform), as 02B revision


@dataclass
class Condition(Condition02d):
    drift_deg: tuple = (0.0, 0.0, 0.0)      # added to imu_rpy_deg, linearly over the drift window
    drift_start_s: float = 0.5
    drift_duration_s: float = 4.0
    slope_deg: tuple = (0.0, 0.0)           # (roll, pitch) of the floor relative to gravity
    encoder_offset_rad: float = 0.0         # per-joint fixed offsets ~ U(-x, x), separate RNG
    encoder_noise_rad: float = ENCODER_NOISE_RAD


def gravity_direction(slope_deg):
    """Unit gravity in the floor frame. Zero slope returns exactly (0, 0, -1)."""
    roll, pitch = slope_deg
    if roll == 0 and pitch == 0:
        return np.array([0., 0., -1.])
    # Floor rotated by R = Ry(pitch) Rx(roll) relative to gravity: g_floor = R^T (0,0,-1).
    return rpy_matrix((roll, pitch, 0.)).T @ np.array([0., 0., -1.])


def mounting(cond, t):
    if not any(cond.drift_deg):
        return rpy_matrix(cond.imu_rpy_deg)
    f = min(1.0, max(0.0, (t - cond.drift_start_s) / cond.drift_duration_s))
    return rpy_matrix(tuple(a + f * b for a, b in zip(cond.imu_rpy_deg, cond.drift_deg)))


def encoder_offsets(cond, n=14):
    if not cond.encoder_offset_rad:
        return np.zeros(n)
    return np.random.default_rng([cond.seed, 202]).uniform(-1, 1, n) * cond.encoder_offset_rad


MARGIN_DIRECTIONS = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, 2 * math.pi, 16, endpoint=False)]
                             + [[1., 0.], [-1., 0.]])


def ray_exits(poly, origin, dirs):
    """Vectorized physics02b.ray_exit for several directions; same element-wise operations and tolerances."""
    a = poly; e = np.roll(poly, -1, axis=0) - poly
    w = a - origin
    u0, u1 = dirs[:, :1], dirs[:, 1:]
    den = u0 * (-e[:, 1]) + u1 * e[:, 0]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (w[:, 0] * (-e[:, 1]) + w[:, 1] * e[:, 0]) / den
        s = (u0 * w[:, 1] - u1 * w[:, 0]) / den
    ok = (np.abs(den) >= 1e-12) & (t >= 0) & (s >= -1e-9) & (s <= 1 + 1e-9)
    return np.where(ok, t, np.inf).min(axis=1)


def inside_all(poly, p):
    """Vectorized physics02b.inside (2-D cross products against every CCW edge)."""
    e = np.roll(poly, -1, axis=0) - poly
    d = p - poly
    return bool(np.all(e[:, 0] * d[:, 1] - e[:, 1] * d[:, 0] >= -1e-12))


def support_margins(sim, data, g_hat):
    """Signed COM support margins (m) along gravity: min over 16 ray exits, and toe/heel along +x/-x."""
    poly = support_polygon(sim, data)
    com = data.subtree_com[sim.trunk]
    # Project the COM along gravity onto the floor plane z = 0: com + s g_hat with s = com_z / -g_z > 0.
    c = com[:2] + g_hat[:2] * (com[2] / -g_hat[2])
    ok = len(poly) >= 3 and inside_all(poly, c)
    if not ok:
        return -1.0, -1.0, -1.0
    t = ray_exits(poly, c, MARGIN_DIRECTIONS)
    return float(t[:16].min()), float(t[16]), float(t[17])


def rollout(sim, theta, vec, cond, design=Design(), record=False, terminate=True, mode="learned", basis=None,
            frontend=None, margins=False):
    """sim02d.rollout plus 02E interventions. ``frontend`` (optional) corrects IMU and may probe."""
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
    g_hat = gravity_direction(cond.slope_deg)
    gravity0 = m.opt.gravity.copy()
    m.opt.gravity[:] = 9.81 * g_hat if any(cond.slope_deg) else gravity0
    try:
        rng = np.random.default_rng(cond.seed)
        sim.reset(cond, rng)
        enc_rng = np.random.default_rng([cond.seed, 101])
        enc_off = encoder_offsets(cond)
        if frontend is not None:
            frontend.reset()
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
        max_tilt = 0.0
        pre_state = None
        foot0 = None
        probe_peak = 0.0
        margin_rows = []
        rec = {k: [] for k in ('time_s', 'qpos', 'qvel', 'offset', 'residual', 'target_applied',
                              'torque_nm', 'tilt_deg', 'foot_found', 'foot_force_n', 'foot_pos',
                              'nonfoot', 'penetration_m', 'com', 'xpos', 'clearance_m',
                              'features', 'coefficients', 'target_cmd', 'self_contacts',
                              'grav_true', 'grav_raw', 'gyro_raw', 'probe')} if record else None
        if record and not hasattr(sim, 'sole_vertices02c'):
            from scipy.spatial import ConvexHull
            sim.sole_vertices02c = []
            for g in sim.feet:
                mesh = m.geom_dataid[g]
                a, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                v = m.mesh_vert[a:a + count]
                sim.sole_vertices02c.append(v[ConvexHull(v).vertices].copy())
        probe = np.zeros(14)
        grav_raw = gyro_raw = None
        for k in range(n):
            t = k * CONTROL_DT
            if actual_event is not None and abs(t - actual_event) < 1e-8:
                pre_state = d.qpos.copy()
                foot0 = d.xpos[list(sim.foot_bodies)].copy()
            if k in pushes:
                d.qvel[:2] += pushes[k]
                mj.mj_forward(m, d)
            imu = mounting(cond, t)
            R = d.xmat[sim.trunk].reshape(3, 3) @ imu
            grav = R.T @ g_hat + rng.uniform(-1, 1, 3) * cond.gravity_noise
            gyro = imu.T @ d.sensordata[sim.gyro_adr:sim.gyro_adr + 3] + rng.uniform(-1, 1, 3) * cond.gyro_noise
            if cond.false_gyro_time_s is not None and cond.false_gyro_time_s <= t < cond.false_gyro_time_s + cond.false_gyro_duration_s - 1e-9:
                gyro[1] += cond.false_gyro_rad_s
            grav_raw, gyro_raw = grav.copy(), gyro.copy()
            enable = True
            if frontend is not None:
                q_enc = d.qpos[sim.qadr] + enc_off + enc_rng.uniform(-1, 1, 14) * cond.encoder_noise_rad
                grav, gyro, probe, enable = frontend(t, grav, gyro, q_enc)
                probe_peak = max(probe_peak, float(np.abs(probe).max()))
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
            if not enable:
                # Calibration phase: the front-end holds every residual at zero; the
                # frozen standing reflex keeps running on the (still raw) IMU.
                r = np.zeros(14)
            residual_cost += float(np.mean((r / design.bound) ** 2))
            a = np.clip(action(theta, feat) + r, -design.bound, design.bound)
            if frontend is not None:
                # The probe is a separate, bounded deliberate stance adjustment.
                a = np.clip(a + probe, -design.bound, design.bound)
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
                                target_cmd=target.copy(), self_contacts=sum(c.geom1 != sim.floor and c.geom2 != sim.floor for c in aux.contact),
                                grav_true=aux.xmat[sim.trunk].reshape(3, 3).T @ g_hat, grav_raw=grav_raw, gyro_raw=gyro_raw,
                                probe=probe.copy())
                    for key in rec:
                        rec[key].append(vals[key])
            mj.mj_forward(m, d)
            if margins:
                margin_rows.append((t + CONTROL_DT, *support_margins(sim, d, g_hat)))
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
        if frontend is not None:
            summary['frontend'] = frontend.report()
            summary['probe_peak_rad'] = probe_peak
        if margins:
            summary['margins'] = np.asarray(margin_rows)
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
    finally:
        m.opt.gravity[:] = gravity0
