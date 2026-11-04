"""02B rollout engine: native CPU MuJoCo + upstream BAM XL330 M6 controller.

Physics is the mjlab-compiled upstream model (``model02b``) stepped with
``bam.mujoco.MujocoController`` -- upstream's own CPU deployment-rehearsal
actuator path (``scripts/infer_policy.py``), mirrored to the warp actuator used
in training. Ordering mirrors mjlab ``ManagerBasedRlEnv.step``: per 20 ms
control period, four 5 ms substeps (command delay buffer -> BAM torque/friction
-> ``mj_step``), then one ``mj_forward`` before observations are read.

The controller under study is a *linear reflex*: joint-target offsets from the
upstream HOME pose are a linear function of what an onboard IMU reports
(projected gravity x/y and gyro x/y/z, both with sensor noise) plus a constant
posture bias. It never sees contact, positions, velocities or time.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from movement_lab.model02b import HOME as _HOME, JOINTS

HOME = np.array(_HOME)
DT = 0.005
DECIMATION = 4
CONTROL_DT = DT * DECIMATION
CLEARANCE_M = 0.001          # 01B's validated 1 mm sole-clearance reset
FEATURES = ('gravity_x', 'gravity_y', 'gyro_x', 'gyro_y', 'gyro_z')
FEATURE_SCALE = np.array([0.1, 0.1, 1.0, 1.0, 1.0])   # ~5.7 deg of tilt, 1 rad/s
N_JOINTS, N_FEAT = len(JOINTS), len(FEATURES)
PARAM_SHAPE = (N_JOINTS, N_FEAT + 1)                   # [W | posture bias]
ACTION_BOUND = 0.35          # rad, per-joint offset from HOME
LIMIT_MARGIN = 0.02          # rad from hard joint limits (01A-02A convention)
FALL_TILT_DEG = 45.0         # 00C survival rule: tilt < 45 deg ...
FALL_HEIGHT_LOSS = 0.04      # ... root height loss < 4 cm ...
PENETRATION_LIMIT = 0.003    # 01B/01C gate, applied to alive (standing) samples
DEFAULT_MODEL = Path(__file__).resolve().parents[1] / 'results/02b/model'


@dataclasses.dataclass
class Condition:
    """Everything that defines one physical trial except the controller."""
    seed: int = 0
    duration_s: float = 4.0
    joint_noise_rad: float = 0.0          # uniform initial joint offsets
    push_time_s: float | None = None
    push_dv: tuple = (0.0, 0.0)           # world-frame root velocity increment (00B/00C)
    extra_pushes: tuple = ()              # ((time_s, dvx, dvy), ...) after the first push
    pulse_body: str = 'robot/trunk_base'  # sustained force alternative
    pulse_force: tuple = (0.0, 0.0, 0.0)  # N, world frame at body COM
    pulse_time_s: float | None = None
    pulse_duration_s: float = 0.0
    vin: float = 7.35
    vin_drop_gain: float = 0.1
    lag_min: int = 4                       # physics-step command delay (mjlab semantics)
    lag_max: int = 4
    gyro_noise: float = 0.03               # upstream actor obs noise (uniform)
    gravity_noise: float = 0.01
    gyro_lag: int = 0                      # control-step observation delay
    imu_rpy_deg: tuple = (0.0, 0.0, 0.0)   # fixed IMU mounting misalignment
    foot_friction: float = 1.0
    head_mass_scale: float = 1.0

    def to_dict(self):
        return dataclasses.asdict(self)


def rpy_matrix(rpy_deg):
    r, p, y = np.radians(rpy_deg)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
                     [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
                     [-sp, cp*sr, cp*cr]])


def action(theta, features):
    """Joint offsets from HOME for normalized IMU features (bounded, linear)."""
    return np.clip(theta[:, :N_FEAT] @ features + theta[:, N_FEAT], -ACTION_BOUND, ACTION_BOUND)


class Sim:
    """One reusable CPU model + BAM controller. Not thread-safe; one per process."""

    def __init__(self, model_dir=DEFAULT_MODEL, verify=True):
        import mujoco
        from bam.model import load_model
        from bam.mujoco import MujocoController
        self.mujoco = mujoco
        model_dir = Path(model_dir)
        info = json.loads((model_dir / 'model.json').read_text())
        mjb = model_dir / 'model.mjb'
        if verify:
            digest = hashlib.sha256(mjb.read_bytes()).hexdigest()
            assert digest == info['mjb_sha256'], 'compiled model hash mismatch'
        self.info = info
        m = self.m = mujoco.MjModel.from_binary_path(str(mjb))
        self.d = mujoco.MjData(m)
        self.aux = mujoco.MjData(m)  # measurement-only kinematics; never stepped
        bm = load_model(motor_name='xl330', model='m6')
        bm.actuator.kp = info['nominal']['kp_fw']
        bm.actuator.max_current = None
        self.bam = bm
        names = ['robot/' + j for j in JOINTS]
        self.ctrl = MujocoController(bm, names, m, self.d, vin_drop_gain=0.1, vin_min=info['nominal']['vin_min'])
        self.kt_over_r = bm.kt.value / bm.R.value
        ids = lambda kind, n: mujoco.mj_name2id(m, kind, n)
        self.trunk = ids(mujoco.mjtObj.mjOBJ_BODY, 'robot/trunk_base')
        self.head = ids(mujoco.mjtObj.mjOBJ_BODY, 'robot/jaw_soft')
        self.floor = ids(mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
        self.feet = (ids(mujoco.mjtObj.mjOBJ_GEOM, 'robot/left_foot_collision'),
                     ids(mujoco.mjtObj.mjOBJ_GEOM, 'robot/right_foot_collision'))
        self.foot_bodies = tuple(int(m.geom_bodyid[g]) for g in self.feet)
        gyro = m.sensor('robot/imu_ang_vel')
        self.gyro_adr = int(gyro.adr[0])
        self.qadr = m.jnt_qposadr[self.ctrl.joint_indexes].copy()
        self.vadr = m.jnt_dofadr[self.ctrl.joint_indexes].copy()
        rng = m.jnt_range[self.ctrl.joint_indexes]
        self.lo, self.hi = rng[:, 0] + LIMIT_MARGIN, rng[:, 1] - LIMIT_MARGIN
        assert np.all(HOME > self.lo) and np.all(HOME < self.hi)
        self.collision_geoms = [g for g in range(m.ngeom) if g != self.floor and (m.geom_contype[g] or m.geom_conaffinity[g])]
        # Pristine copies of every field a condition may change (restore-then-apply).
        self._pristine = {k: getattr(m, k).copy() for k in
                          ('geom_friction', 'body_mass', 'body_inertia', 'actuator_forcerange', 'dof_frictionloss', 'dof_damping')}

    # -- setup --------------------------------------------------------------------
    def _lowest_point(self, data):
        m, low = self.m, np.inf
        for g in self.collision_geoms:
            mesh = m.geom_dataid[g]
            if mesh < 0:
                continue
            a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
            verts = m.mesh_vert[a:a + n] @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            low = min(low, verts[:, 2].min())
        return low

    def reset(self, cond: Condition, rng):
        mj, m, d = self.mujoco, self.m, self.d
        for k, v in self._pristine.items():
            getattr(m, k)[:] = v
        for g in self.feet:
            m.geom_friction[g, 0] = cond.foot_friction
        if cond.head_mass_scale != 1.0:
            m.body_mass[self.head] *= cond.head_mass_scale
            m.body_inertia[self.head] *= cond.head_mass_scale
        limit = cond.vin * self.kt_over_r
        m.actuator_forcerange[:] = [-limit, limit]
        mj.mj_setConst(m, d)
        mj.mj_resetData(m, d)
        q = HOME + rng.uniform(-1, 1, N_JOINTS) * cond.joint_noise_rad
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.qadr] = np.clip(q, self.lo, self.hi)
        d.qpos[2] = 0.2
        mj.mj_kinematics(m, d)
        d.qpos[2] += CLEARANCE_M - self._lowest_point(d)
        d.qvel[:] = 0
        mj.mj_forward(m, d)
        self.bam.actuator.vin = cond.vin
        self.ctrl.vin_drop_gain = cond.vin_drop_gain
        self.ctrl.reset(d.qpos)
        self.ctrl.last_ts = d.time
        self.z0 = float(d.qpos[2])

    # -- sensing --------------------------------------------------------------------
    def tilt_deg(self, data=None):
        data = self.d if data is None else data
        return math.degrees(math.acos(max(-1.0, min(1.0, data.xmat[self.trunk][8]))))

    def contacts(self, data=None):
        """(left found, right found, non-foot floor contacts, max penetration m, foot normal forces)."""
        mj, data = self.mujoco, (self.d if data is None else data)
        found, nonfoot, pen, force = [False, False], 0, 0.0, np.zeros(2)
        f6 = np.zeros(6)
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 != self.floor and c.geom2 != self.floor:
                continue
            other = c.geom2 if c.geom1 == self.floor else c.geom1
            pen = max(pen, -c.dist)
            if other in self.feet:
                k = self.feet.index(other)
                found[k] = True
                mj.mj_contactForce(self.m, data, i, f6)
                force[k] += f6[0]
            else:
                nonfoot += 1
        return found[0], found[1], nonfoot, pen, force


JOINT_POS_NOISE, JOINT_VEL_NOISE = 0.001, 0.25   # upstream actor obs noise (uniform)


def rollout(sim: Sim, theta, cond: Condition, record=False, terminate=True, grace_s=0.5, policy=None):
    """Run one trial. Returns (summary dict, record dict or None).

    ``terminate`` stops ``grace_s`` after the first fall (training/eval economy);
    the outcome is fixed at the fall, so later samples cannot change it.

    ``policy`` (optional, 02B revision only) replaces the linear IMU reflex with
    ``policy(imu_features, joint_pos_minus_home, joint_vel) -> offsets``; joint
    readings then carry upstream encoder noise. With ``policy=None`` the code
    path and random stream are exactly the linear-reflex study's.
    """
    mj, m, d = sim.mujoco, sim.m, sim.d
    rng = np.random.default_rng(cond.seed)
    theta = np.asarray(theta, dtype=float).reshape(PARAM_SHAPE)
    sim.reset(cond, rng)
    imu = rpy_matrix(cond.imu_rpy_deg)
    n_ctrl = int(round(cond.duration_s / CONTROL_DT))
    lag_hist = np.repeat(HOME[None], 8, axis=0)  # physics-step target history (backfilled)
    lag_ptr, lag = 0, cond.lag_min
    gyro_hist = []
    push_step = None if cond.push_time_s is None else int(math.ceil(cond.push_time_s / CONTROL_DT - 1e-9))
    pushes = {} if push_step is None else {push_step: cond.push_dv}
    for t_extra, dvx, dvy in cond.extra_pushes:
        pushes[int(math.ceil(t_extra / CONTROL_DT - 1e-9))] = (dvx, dvy)
    pulse = None
    if cond.pulse_time_s is not None and cond.pulse_duration_s > 0:
        body = mj.mj_name2id(m, mj.mjtObj.mjOBJ_BODY, cond.pulse_body)
        assert body >= 0, cond.pulse_body
        p0 = int(round(cond.pulse_time_s / DT))
        pulse = (body, p0, p0 + int(round(cond.pulse_duration_s / DT)), np.array(cond.pulse_force, float))
    limit = cond.vin * sim.kt_over_r
    s = dict(fall_time_s=None, fall_cause=None, max_tilt_deg=0.0, max_tilt_post_deg=0.0, max_pen_alive_m=0.0,
             nonfoot_alive=0, sat_frac=0.0, peak_torque_nm=0.0, work_j=0.0, action_sq=0.0, rate_sq=0.0,
             max_ang_speed=0.0, min_root_z=sim.z0, foot_unloaded=[0, 0], steps_alive=0, fitness=0.0)
    sat_count = 0
    phys = 0
    prev_a = np.zeros(N_JOINTS)
    rec = None
    if record:
        rec = {k: [] for k in ('time_s', 'qpos', 'qvel', 'torque_nm', 'target_cmd', 'target_applied', 'features',
                                'offset', 'com', 'foot_found', 'foot_force_n', 'nonfoot', 'penetration_m', 'tilt_deg', 'xpos')}
    tilt_acc, act_acc, rate_acc = 0.0, 0.0, 0.0
    fallen_at = None
    pre_push = []
    root0 = d.qpos[:2].copy()
    tilt_series, speed_series = [], []
    # Per-foot support bookkeeping: lift-off position, consecutive unloaded steps.
    foot_off = [None, None]
    foot_prev = [True, True]
    s.update(transitions=[0, 0], steps=[], foot_slide_m=[0.0, 0.0])
    foot_last = [d.xpos[b][:2].copy() for b in sim.foot_bodies]
    for k in range(n_ctrl):
        if k in pushes:
            d.qvel[0] += pushes[k][0]
            d.qvel[1] += pushes[k][1]
            mj.mj_forward(m, d)
        # --- observe (IMU only, noisy, optionally delayed / misaligned) ---
        R = d.xmat[sim.trunk].reshape(3, 3) @ imu
        grav = R.T @ np.array([0.0, 0.0, -1.0]) + rng.uniform(-1, 1, 3) * cond.gravity_noise
        gyro = imu.T @ d.sensordata[sim.gyro_adr:sim.gyro_adr + 3] + rng.uniform(-1, 1, 3) * cond.gyro_noise
        gyro_hist.append(gyro)
        gyro_obs = gyro_hist[max(0, len(gyro_hist) - 1 - cond.gyro_lag)]
        feat = np.array([grav[0], grav[1], *gyro_obs]) / FEATURE_SCALE
        if policy is None:
            a = action(theta, feat)
        else:
            q_dev = d.qpos[sim.qadr] - HOME + rng.uniform(-1, 1, N_JOINTS) * JOINT_POS_NOISE
            qd = d.qvel[sim.vadr] + rng.uniform(-1, 1, N_JOINTS) * JOINT_VEL_NOISE
            a = np.clip(policy(feat, q_dev, qd), -ACTION_BOUND, ACTION_BOUND)
        target = np.clip(HOME + a, sim.lo, sim.hi)
        alive = fallen_at is None
        if alive:
            act_acc += float(np.mean((a / ACTION_BOUND) ** 2))
            rate_acc += float(np.mean(((a - prev_a) / 0.1) ** 2)) if k else 0.0
        prev_a = a
        # --- act: four physics substeps with mjlab-style command delay ---
        for _ in range(DECIMATION):
            lag_ptr = (lag_ptr + 1) % len(lag_hist)
            lag_hist[lag_ptr] = target
            if cond.lag_max > cond.lag_min:
                lag = int(rng.integers(cond.lag_min, cond.lag_max + 1))
            applied = lag_hist[(lag_ptr - min(lag, phys)) % len(lag_hist)]
            sim.ctrl.q_target = applied.copy()
            if pulse is not None:
                d.xfrc_applied[pulse[0], :3] = pulse[3] if pulse[1] <= phys < pulse[2] else 0.0
            sim.ctrl.update()
            mj.mj_step(m, d)
            phys += 1
            tau = d.ctrl
            if alive:
                sat_count += int(np.any(np.abs(tau) >= 0.98 * limit))
                s['peak_torque_nm'] = max(s['peak_torque_nm'], float(np.abs(tau).max()))
                s['work_j'] += float(np.abs(tau * d.qvel[sim.vadr]).sum()) * DT
            if record:
                aux = sim.aux
                aux.qpos[:], aux.qvel[:], aux.time = d.qpos, d.qvel, d.time
                mj.mj_forward(m, aux)
                lf, rf, nf, pen, force = sim.contacts(aux)
                rec['time_s'].append(d.time); rec['qpos'].append(d.qpos.copy()); rec['qvel'].append(d.qvel.copy())
                rec['torque_nm'].append(tau.copy()); rec['target_cmd'].append(target.copy()); rec['target_applied'].append(applied.copy())
                rec['features'].append(feat * FEATURE_SCALE); rec['offset'].append(a.copy())
                rec['com'].append(aux.subtree_com[sim.trunk].copy()); rec['foot_found'].append((lf, rf))
                rec['foot_force_n'].append(force); rec['nonfoot'].append(nf); rec['penetration_m'].append(pen)
                rec['tilt_deg'].append(sim.tilt_deg(aux)); rec['xpos'].append(aux.xpos.copy())
        mj.mj_forward(m, d)
        if not np.all(np.isfinite(d.qpos)) or not np.all(np.isfinite(d.qvel)):
            s['fall_time_s'], s['fall_cause'] = d.time, 'nonfinite'
            break
        # --- measure at control rate ---
        tilt = sim.tilt_deg()
        lf, rf, nonfoot, pen, _ = sim.contacts()
        z = float(d.qpos[2])
        t = d.time
        if alive:
            s['steps_alive'] += 1
            tilt_acc += (tilt / FALL_TILT_DEG) ** 2
            s['max_tilt_deg'] = max(s['max_tilt_deg'], tilt)
            if push_step is not None and k >= push_step:
                s['max_tilt_post_deg'] = max(s['max_tilt_post_deg'], tilt)
            elif push_step is not None and k >= push_step - 25:
                pre_push.append(tilt)
            s['max_pen_alive_m'] = max(s['max_pen_alive_m'], pen)
            s['min_root_z'] = min(s['min_root_z'], z)
            s['max_ang_speed'] = max(s['max_ang_speed'], float(np.linalg.norm(d.qvel[3:6])))
            s['foot_unloaded'][0] += int(not lf)
            s['foot_unloaded'][1] += int(not rf)
            tilt_series.append(tilt)
            speed_series.append(float(np.linalg.norm(d.qvel[3:6])))
            for f, on in enumerate((lf, rf)):
                pos = d.xpos[sim.foot_bodies[f]][:2].copy()
                if on and foot_prev[f]:
                    s['foot_slide_m'][f] += float(np.linalg.norm(pos - foot_last[f]))
                if on != foot_prev[f]:
                    s['transitions'][f] += 1
                    if not on:
                        foot_off[f] = (t, pos)
                    elif foot_off[f] is not None:
                        # Conservative step evidence (00C): unloaded >= 40 ms and
                        # recontact > 5 mm from the lift-off point.
                        t_off, p_off = foot_off[f]
                        shift = float(np.linalg.norm(pos - p_off))
                        s['steps'].append(dict(foot='LR'[f], t_off=t_off, air_s=t - t_off, shift_m=shift,
                                               is_step=bool(t - t_off >= 0.04 - 1e-9 and shift > 0.005)))
                        foot_off[f] = None
                foot_prev[f] = on
                foot_last[f] = pos
            cause = ('tilt' if tilt > FALL_TILT_DEG else 'nonfoot_contact' if nonfoot else
                     'height' if z < sim.z0 - FALL_HEIGHT_LOSS else None)
            if cause:
                fallen_at = k
                s['fall_time_s'], s['fall_cause'] = t, cause
        if fallen_at is not None and terminate and (k - fallen_at) * CONTROL_DT >= grace_s:
            break
    alive_steps = s['steps_alive']
    s['survived'] = s['fall_time_s'] is None
    s['duration_s'] = cond.duration_s
    s['alive_frac'] = alive_steps / n_ctrl
    s['sat_frac'] = sat_count / max(1, alive_steps * DECIMATION)
    s['action_rms'] = math.sqrt(act_acc / max(1, alive_steps)) * ACTION_BOUND
    s['rate_rms_rad_per_step'] = math.sqrt(rate_acc / max(1, alive_steps)) * 0.1
    s['final_tilt_deg'] = sim.tilt_deg()
    s['final_root_disp_m'] = float(np.linalg.norm(d.qpos[:2] - root0))
    s['final_root_xy'] = d.qpos[:2].tolist()
    s['final_root_z'] = float(d.qpos[2])
    s['pre_push_tilt_deg'] = float(np.mean(pre_push)) if pre_push else None
    s['valid'] = s['max_pen_alive_m'] <= PENETRATION_LIMIT and s['fall_cause'] != 'nonfinite'
    s['n_steps'] = sum(st['is_step'] for st in s['steps'])
    s.update(recovery(tilt_series, speed_series, push_step, s['survived']))
    s['fitness'] = fitness(alive_steps, n_ctrl, tilt_acc, act_acc, rate_acc)
    if record:
        rec = {k: np.asarray(v) for k, v in rec.items()}
    return s, rec


# Training objective. Survival dominates; tilt shaping rewards actually being
# upright rather than slowly toppling; small terms discourage large or
# chattering commands. Falls end accumulation (the fallen tail scores zero).
W_TILT, W_ACTION, W_RATE = 0.5, 0.02, 0.05


def fitness(alive_steps, n_ctrl, tilt_acc, act_acc, rate_acc):
    return (alive_steps - W_TILT * tilt_acc - W_ACTION * act_acc - W_RATE * rate_acc) / n_ctrl


def zero_theta():
    return np.zeros(PARAM_SHAPE)


RECOVERED_TILT_DEG = 5.0      # quiet upright stance over the final second ...
RECOVERED_SPEED = 0.5         # ... with trunk angular speed below 0.5 rad/s
SETTLE_BAND_DEG = 1.0         # settling: back within 1 deg of pre-push tilt ...
SETTLE_HOLD_S = 0.25          # ... continuously for 0.25 s


def recovery(tilt, speed, push_step, survived):
    """Exploratory recovery descriptors from control-rate tilt/speed series."""
    out = dict(recovered=False, settle_time_s=None)
    n_final = int(round(1.0 / CONTROL_DT))
    if survived and len(tilt) >= n_final:
        out['recovered'] = bool(max(tilt[-n_final:]) < RECOVERED_TILT_DEG and max(speed[-n_final:]) < RECOVERED_SPEED)
    if survived and push_step is not None and push_step >= 25 and len(tilt) > push_step:
        ref = float(np.mean(tilt[push_step - 25:push_step]))
        hold = int(round(SETTLE_HOLD_S / CONTROL_DT))
        inside = np.abs(np.asarray(tilt[push_step:]) - ref) <= SETTLE_BAND_DEG
        run = 0
        for i, ok in enumerate(inside):
            run = run + 1 if ok else 0
            if run >= hold:
                out['settle_time_s'] = (i - hold + 1) * CONTROL_DT
                break
    return out
