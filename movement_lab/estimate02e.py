"""Calibration front-ends for 02E: what onboard signals can say about IMU misalignment.

All front-ends see only what Microduck could measure: the (possibly misaligned)
IMU gravity estimate and gyro, noisy joint encoders, time since start, and their
own probe commands. They never see contact, root state, the true offset, the
floor slope or the disturbance. Each one runs a calibration window at start-up
while the learned residual is held at zero and the frozen 02B reflex stands,
then applies its estimated sensor-to-trunk rotation smoothly to BOTH gravity and
gyro for every controller.

Information sources (the observability argument, tested empirically):

* ``prior``  -- IMU only. Compares the quiet gravity reading with the reading the
  same controller produced in nominal calibration. Needs level ground and the
  nominal stance; the closed loop moves the stance when the sensor is wrong.
* ``static`` -- gravity vs leg kinematics (encoders + model), assuming both soles
  are flat on LEVEL ground. Measures misalignment + floor slope + encoder error.
* ``yaw``    -- gyro vs leg kinematics during a small deliberate hip-yaw probe.
  Angular velocity is a free vector, so this compares two measurements of the
  same rotation in the sensor and trunk frames. Gravity and floor slope do not
  enter. Requires the soles not to rotate, and recovers only the misalignment
  components perpendicular to the probe's rotation axis (pitch/roll for yaw).
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field
import math
import numpy as np
from movement_lab.sim02b import HOME, CONTROL_DT

LEFT_YAW, RIGHT_YAW = 0, 9          # joint indices (model02b.JOINTS)
DIFF_STEPS = 2                       # encoder central difference over +/-40 ms (1 Hz probe: negligible bias)


def rot_between(a, b):
    """Minimal rotation R with R @ a_hat = b_hat."""
    a = a / np.linalg.norm(a); b = b / np.linalg.norm(b)
    v = np.cross(a, b); c = float(a @ b); s = float(np.linalg.norm(v))
    if s < 1e-15:
        return np.eye(3)
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    ang = math.atan2(s, c)
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def rotvec(R):
    ang = math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))
    if ang < 1e-12:
        return np.zeros(3)
    return ang / (2 * math.sin(ang)) * np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])


def expmap(v):
    ang = float(np.linalg.norm(v))
    if ang < 1e-15:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * K + (1 - math.cos(ang)) * K @ K


def rotation_fit(gyro_angles, reference_angles, single_axis=False):
    """Rotation R with R @ gyro ~ reference for paired rotation vectors (sensor vs trunk frame).

    Mean-centred (a constant offset in either series does not matter). With one
    dominant rotation axis only that axis's tilt is observable (minimal rotation);
    otherwise Kabsch/Wahba. Returns (R, singular values of the cross-covariance).
    """
    g = gyro_angles - gyro_angles.mean(axis=0)
    k = reference_angles - reference_angles.mean(axis=0)
    U, S, Vt = np.linalg.svd(g.T @ k)
    if single_axis or S[1] < 1e-3 * S[0]:
        a_gyro, a_kin = U[:, 0], Vt[0]
        return rot_between(a_gyro if a_gyro @ a_kin >= 0 else -a_gyro, a_kin), S.tolist()
    D = np.diag([1., 1., float(np.sign(np.linalg.det(Vt.T @ U.T)))])
    return Vt.T @ D @ U.T, S.tolist()


def tilt_angles_deg(R):
    """(roll, pitch) in degrees of the misalignment that R corrects, for small angles: R ~ Ry(p) Rx(r)."""
    v = rotvec(R)
    return float(np.degrees(v[0])), float(np.degrees(v[1]))


class Kinematics:
    """Trunk-to-sole kinematics from encoder angles on a private MjData (never the simulated one)."""

    def __init__(self, sim):
        import mujoco
        from scipy.spatial import ConvexHull
        self.mj, self.m, self.sim = mujoco, sim.m, sim
        self.data = mujoco.MjData(sim.m)
        self.feet = sim.foot_bodies
        # Sole-down normal of each foot body, from collision-mesh geometry alone:
        # the outward normal of the largest convex-hull facet facing down at HOME.
        self.data.qpos[:] = sim.m.qpos0
        self.data.qpos[3:7] = (1, 0, 0, 0)
        self.data.qpos[sim.qadr] = HOME
        mujoco.mj_kinematics(sim.m, self.data)
        self.normals = []
        for g, b in zip(sim.feet, self.feet):
            mesh = sim.m.geom_dataid[g]
            a, n = sim.m.mesh_vertadr[mesh], sim.m.mesh_vertnum[mesh]
            v_geom = sim.m.mesh_vert[a:a + n]
            Rg = self.data.geom_xmat[g].reshape(3, 3); Rb = self.data.xmat[b].reshape(3, 3)
            v_world = v_geom @ Rg.T
            hull = ConvexHull(v_world)
            # Group the hull's triangulated facets into planes; the sole is the
            # largest-area plane whose outward normal points down.
            planes = {}
            for eq, simplex in zip(hull.equations, hull.simplices):
                if eq[2] > -0.9:
                    continue
                p = v_world[simplex]
                area = .5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
                key = tuple(np.round(eq[:3], 3))
                acc = planes.setdefault(key, [0.0, np.zeros(3)])
                acc[0] += area; acc[1] += area * eq[:3]
            area, weighted = max(planes.values(), key=lambda x: x[0])
            n_world = weighted / np.linalg.norm(weighted)
            self.sole_area_m2 = getattr(self, 'sole_area_m2', []) + [float(area)]
            self.normals.append(Rb.T @ n_world)  # in foot-body frame

    def trunk_rotation_direction(self, axis, q=HOME):
        """Leg joint-target direction (rad per rad) that rotates the trunk about a trunk axis with both soles fixed.

        Linearized at ``q``: unknowns are root translation and the ten leg joints; the
        two feet's linear and angular velocities are constrained to zero. Head joints
        do not move. Minimum-norm least squares; the residual norm is returned.
        """
        mj, m, d = self.mj, self.m, self.data
        d.qpos[:] = m.qpos0
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.sim.qadr] = q
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        rows = []
        for b in self.feet:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mj.mj_jacBody(m, d, jp, jr, b)
            rows += [jp, jr]
        J = np.vstack(rows)                                   # 12 x nv
        legs = [i for i in range(14) if i not in (5, 6, 7, 8)]
        cols_q = [int(self.sim.vadr[i]) for i in legs]
        # Free-joint velocity: dofs 0-2 world linear, 3-5 body-frame angular velocity.
        A = J[:, [0, 1, 2] + cols_q]
        b = -J[:, 3:6] @ np.asarray(axis, float)
        x, *_ = np.linalg.lstsq(A, b, rcond=None)
        direction = np.zeros(14)
        direction[legs] = x[3:]
        return direction, float(np.linalg.norm(A @ x - b))

    def trunk_velocity(self, q, qd):
        """Trunk angular velocity (trunk frame) implied by joint velocities if both soles were fixed (mean of feet)."""
        mj, m, d = self.mj, self.m, self.data
        d.qpos[:] = m.qpos0
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.sim.qadr] = q
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d)
        cols = [int(v) for v in self.sim.vadr]
        out = []
        for b in self.feet:
            jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
            mj.mj_jacBody(m, d, jp, jr, b)
            out.append(-jr[:, cols] @ qd)      # root fixed: foot rate relative to trunk, negated
        return np.mean(out, axis=0)

    def trunk_foot(self, q):
        """Rotations of each foot body expressed in the trunk frame, for encoder angles q."""
        d = self.data
        d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[self.sim.qadr] = q
        self.mj.mj_kinematics(self.m, d)
        return [d.xmat[b].reshape(3, 3).copy() for b in self.feet]

    def predicted_gravity(self, q):
        """Gravity in the trunk frame if both soles were flat on level ground (mean of both feet)."""
        g = [R @ n for R, n in zip(self.trunk_foot(q), self.normals)]
        g = np.mean(g, axis=0)
        return g / np.linalg.norm(g)


@dataclass
class Settings:
    kind: str = 'none'              # none | gate | prior | static | yaw | rock
    start_s: float = 0.3            # calibration window start (after reset settling)
    end_s: float = 1.3              # estimate is formed at end_s
    blend_s: float = 0.3            # correction ramps in smoothly, then the residual is enabled
    probe_rad: float = 0.0          # yaw: hip-yaw amplitude; rock: trunk ROLL amplitude (rad of trunk rotation)
    probe_hz: float = 1.0
    probe_sign: float = 1.0         # yaw: +1 both hip yaws same sign; -1 mirrored
    pitch_rad: float = 0.0          # rock: trunk PITCH amplitude
    pitch_hz: float = 1.5
    prior_gravity: tuple = (0.0, 0.0, -1.0)   # 'prior': nominal quiet reading of the same controller

    def to_dict(self):
        return asdict(self)


class Frontend:
    """Callable ``(t, grav, gyro, q_enc) -> (grav, gyro, probe, residual_enabled)``."""

    ESTIMATING = ('prior', 'static', 'yaw', 'rock')

    def __init__(self, settings: Settings, kinematics: Kinematics | None = None):
        self.s = settings
        self.kin = kinematics
        if settings.kind == 'rock':
            self.dir_roll, self.res_roll = kinematics.trunk_rotation_direction((1., 0., 0.))
            self.dir_pitch, self.res_pitch = kinematics.trunk_rotation_direction((0., 1., 0.))
        self.reset()

    def reset(self):
        self.grav_sum = np.zeros(3); self.pred_sum = np.zeros(3); self.count = 0
        self.series = []
        self.estimate_time = None; self.rv = np.zeros(3); self.singular = None

    def enabled(self, t):
        if self.s.kind == 'none':
            return True
        return t >= self.s.end_s + self.s.blend_s - 1e-9

    def probe(self, t):
        s = self.s
        if s.kind not in ('yaw', 'rock') or not (s.start_s <= t < s.end_s):
            return np.zeros(14)
        u = (t - s.start_s) / (s.end_s - s.start_s)
        envelope = math.sin(math.pi * u) ** 2          # smooth on and off, zero at both window ends
        tau = t - s.start_s
        if s.kind == 'yaw':
            y = s.probe_rad * envelope * math.sin(2 * math.pi * s.probe_hz * tau)
            p = np.zeros(14); p[LEFT_YAW] = y; p[RIGHT_YAW] = s.probe_sign * y
            return p
        return envelope * (s.probe_rad * math.sin(2 * math.pi * s.probe_hz * tau) * self.dir_roll
                           + s.pitch_rad * math.sin(2 * math.pi * s.pitch_hz * tau) * self.dir_pitch)

    def estimate(self):
        s = self.s
        if s.kind == 'prior':
            return rot_between(self.grav_sum, np.asarray(s.prior_gravity, float))
        if s.kind == 'static':
            return rot_between(self.grav_sum, self.pred_sum)
        if len(self.series) < 2 * DIFF_STEPS + 3:
            return np.eye(3)
        gyro, ref = self.velocity_pairs()
        R, self.singular = rotation_fit(gyro, ref, single_axis=s.kind == 'yaw' or not s.pitch_rad)
        return R

    def velocity_pairs(self):
        """Gyro readings paired with the trunk rate implied by encoder central differences (+/- DIFF_STEPS)."""
        q = np.array([x[0] for x in self.series]); g = np.array([x[1] for x in self.series])
        h = DIFF_STEPS
        ref = [self.kin.trunk_velocity(q[i], (q[i + h] - q[i - h]) / (2 * h * CONTROL_DT)) for i in range(h, len(q) - h)]
        return g[h:len(q) - h], np.array(ref)

    def __call__(self, t, grav, gyro, q_enc):
        s = self.s
        if s.kind in self.ESTIMATING and s.start_s - 1e-9 <= t < s.end_s - 1e-9:
            self.count += 1
            self.grav_sum += grav
            if s.kind == 'static':
                self.pred_sum += self.kin.predicted_gravity(q_enc)
            if s.kind in ('yaw', 'rock'):
                # The same trunk rotation measured twice: by the gyro (sensor frame) and by leg
                # kinematics if the soles did not rotate (trunk frame). Angular velocity is a free
                # vector, so neither gravity nor the floor slope enters this comparison.
                self.series.append((np.asarray(q_enc, float).copy(), gyro.copy()))
        if self.estimate_time is None and s.kind in self.ESTIMATING and t >= s.end_s - 1e-9 and self.count:
            self.rv = rotvec(self.estimate()); self.estimate_time = t
        if self.estimate_time is not None:
            f = min(1.0, (t - self.estimate_time) / s.blend_s) if s.blend_s > 0 else 1.0
            R = expmap(f * self.rv)
            grav, gyro = R @ grav, R @ gyro
        return grav, gyro, self.probe(t), self.enabled(t)

    def report(self):
        out = dict(settings=self.s.to_dict(), samples=self.count, estimate_time_s=self.estimate_time,
                   correction_rotvec_rad=self.rv.tolist(), correction_roll_pitch_deg=[float(np.degrees(self.rv[0])), float(np.degrees(self.rv[1]))])
        if self.s.kind in ('yaw', 'rock'):
            rms = None
            if len(self.series) >= 2 * DIFF_STEPS + 3:
                rms = float(np.sqrt(np.mean(np.sum(self.velocity_pairs()[1] ** 2, axis=1))))
            out.update(kinematic_rate_rms_rad_s=rms, singular_values=self.singular,
                       observable='pitch/yaw tilt of the dominant (roll) axis only' if self.s.kind == 'rock' and not self.s.pitch_rad else None)
        return out
