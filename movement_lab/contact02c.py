"""Offline diagnostic separating ankle motion from material contact slip.

For each geometric sole-floor contact, J_point(q) @ qvel is the velocity of
the sole's material point at that world contact position. Its horizontal
component distinguishes sliding from rigid rotation around a stationary point.
No forces are reconstructed here; contacts are geometry matches. The measure
is a diagnostic, not a training reward or an independently observed sensor.
"""
import argparse
import json
from pathlib import Path
import numpy as np

from movement_lab.sim02b import Sim, DT


def analyze(sim, rec, push_time, failure_time=None):
    m, d, mj = sim.m, sim.aux, sim.mujoco
    jac = np.zeros((3, m.nv))
    tangential = np.zeros((len(rec['time_s']), 2))
    matched = np.zeros_like(tangential, bool)
    orientation = []
    for i, (q, v) in enumerate(zip(rec['qpos'], rec['qvel'])):
        d.qpos[:] = q; d.qvel[:] = v
        mj.mj_kinematics(m, d); mj.mj_comPos(m, d); mj.mj_collision(m, d)
        orientation.append(d.xmat[list(sim.foot_bodies)].copy())
        samples = [[], []]
        for c in d.contact:
            if c.geom1 != sim.floor and c.geom2 != sim.floor:
                continue
            g = c.geom2 if c.geom1 == sim.floor else c.geom1
            if g not in sim.feet:
                continue
            foot = sim.feet.index(g)
            mj.mj_jac(m, d, jac, None, c.pos, sim.foot_bodies[foot])
            samples[foot].append(float(np.linalg.norm((jac @ v)[:2])))
        for foot in range(2):
            if samples[foot]:
                matched[i, foot] = True
                tangential[i, foot] = np.mean(samples[foot])
    t = rec['time_s']
    start = .5 if push_time is None else np.ceil(push_time / .02 - 1e-9) * .02
    stop = t[-1] if failure_time is None else failure_time
    mask = (t > start) & (t < min(stop, start + 1.0))
    p = int(np.searchsorted(t, start))
    end = min(int(np.searchsorted(t, stop)), len(t) - 1)
    orient = np.asarray(orientation).reshape(-1, 2, 3, 3)
    diff = np.einsum('fij,fkj->fik', orient[end], orient[p])
    angles = np.degrees(np.arccos(np.clip((np.trace(diff, axis1=1, axis2=2) - 1) / 2, -1, 1)))
    return dict(window_s=[float(start), float(min(stop, start + 1))],
                contact_point_path_m=(tangential[mask] * DT).sum(axis=0).tolist(),
                first_40ms_contact_path_m=(tangential[(t > start) & (t <= start + .04 + 1e-9)] * DT).sum(axis=0).tolist(),
                later_contact_path_m=(tangential[mask & (t > start + .04 + 1e-9)] * DT).sum(axis=0).tolist(),
                peak_contact_point_speed_m_s=tangential[mask].max(axis=0).tolist(),
                contact_fraction=matched[mask].mean(axis=0).tolist(),
                sole_orientation_change_deg=angles.tolist())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    a = p.parse_args()
    sim = Sim(); rows = {}
    for path in sorted((a.root / 'records').glob('*.npz')):
        meta = json.loads(path.with_suffix('.json').read_text())
        r = np.load(path)
        rows[path.stem] = analyze(sim, r, meta['condition']['push_time_s'], meta['summary']['failure_time_s'])
    (a.root / 'report/contact-diagnostics.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
