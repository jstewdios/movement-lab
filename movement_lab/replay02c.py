"""Saved-state replay and physical diagnostics; no robot meshes are published."""
import argparse
from dataclasses import asdict, replace
import math
import json
from pathlib import Path
import numpy as np

from movement_lab.sim02b import Sim, Condition
from movement_lab.sim02c import BASE, rollout, audit
from movement_lab.eval02c import controllers, grid, simulate, audit_record
from movement_lab.replay02b import PAGE
from movement_lab.physics02b import convex_hull


def filmstrip(data, path, magnitude=.30):
    """First replicate at a declared magnitude; both compared outcomes shown."""
    import matplotlib.pyplot as plt
    selected = [d for d in data if d['label'].startswith((f'reflex02b, +{magnitude:.2f}', f'cem-s1, +{magnitude:.2f}'))]
    fig, axes = plt.subplots(2, 6, figsize=(13, 5), sharex=True, sharey=True, constrained_layout=True)
    for row, d in enumerate(selected):
        for col, delta in enumerate((0, .08, .16, .30, .50, 1.0)):
            f = min(d['frames'], key=lambda f: abs(f['t'] - d['push_time'] - delta))
            ax = axes[row, col]; positions = np.array(f['p'])
            for a, b in d['edges']:
                ax.plot(positions[[a, b], 0], positions[[a, b], 2], color='#2763aa', lw=2)
            for foot in f['soles_xz']:
                foot = np.array(foot)
                ax.fill(foot[:, 0], foot[:, 1], color='#2763aa', alpha=.6)
            ax.scatter(f['com'][0], f['com'][2], c='#e36c32', s=15)
            ax.scatter(f['head'][0], f['head'][2], facecolors='none', edgecolors='#2763aa', s=100)
            ax.axhline(0, c='gray', lw=.5)
            ax.set(title=f"{d['label'].split(',')[0]}: +{delta:.2f} s", aspect='equal', xlim=(-100, 220), ylim=(-10, 270), xlabel='x (mm)')
        axes[row, 0].set_ylabel('z (mm)')
    fig.suptitle(f'Held-out +{magnitude:.2f} m/s: frozen reflex (top), learned option (bottom); first replicate')
    fig.savefig(path, dpi=140)
    plt.close(fig)


def frames(sim, summary, rec, cond):
    """Render geometry-derived sole outlines and link origins from saved qpos."""
    from scipy.spatial import ConvexHull
    m, aux, mj = sim.m, sim.aux, sim.mujoco
    bodies = list(range(2, m.nbody))
    idx = {b: i for i, b in enumerate(bodies)}
    edges = [[idx[b], idx[int(m.body_parentid[b])]] for b in bodies if int(m.body_parentid[b]) in idx]
    verts = []
    for g in sim.feet:
        mesh = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[a:a + n]
        verts.append(v[ConvexHull(v).vertices])
    fs = []
    for i in range(3, len(rec['time_s']), 4):
        aux.qpos[:] = rec['qpos'][i]
        mj.mj_kinematics(m, aux); mj.mj_comPos(m, aux)
        soles, soles_xz, soles_yz = [], [], []
        for g, v in zip(sim.feet, verts):
            world = v @ aux.geom_xmat[g].reshape(3, 3).T + aux.geom_xpos[g]
            h = convex_hull(world[:, :2])
            h = h[np.linspace(0, len(h), min(24, len(h)), endpoint=False).astype(int)]
            soles.append(np.round(h * 1000, 1).tolist())
            for axes, dest in (([0, 2], soles_xz), ([1, 2], soles_yz)):
                h = convex_hull(world[:, axes])
                h = h[np.linspace(0, len(h), min(24, len(h)), endpoint=False).astype(int)]
                dest.append(np.round(h * 1000, 1).tolist())
        fs.append(dict(t=round(float(rec['time_s'][i]), 3), p=np.round(aux.xpos[bodies] * 1000, 1).tolist(),
                       head=np.round(aux.xipos[sim.head] * 1000, 1).tolist(), com=np.round(aux.subtree_com[sim.trunk] * 1000, 1).tolist(),
                       soles=soles, soles_xz=soles_xz, soles_yz=soles_yz,
                       contact=rec['foot_found'][i].tolist(), tilt=round(float(rec['tilt_deg'][i]), 2)))
    display = dict(survived=summary['survived'], fall_time_s=summary['failure_time_s'], fall_cause=summary['failure_cause'])
    return dict(edges=edges, frames=fs, summary=display, push_time=cond.push_time_s, push_dv=cond.push_dv)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    a = p.parse_args()
    out = a.root / 'report'; out.mkdir(exist_ok=True)
    records = a.root / 'records'; records.mkdir(exist_ok=True)
    sim = Sim(); theta = np.load(BASE)['final']
    cs = [x for x in controllers(a.root) if x[0] in ('reflex02b', 'wide-reflex', 'cem-s1', 'cem-s2', 'cem-s3', 'cem-s1-reverse', 'cem-s1-bound35')]
    # Fixed illustrative conditions: first held-out replicate at each magnitude.
    conditions = [c for label, c in grid() if label in ('forward-0.00', 'forward-0.20', 'forward-0.30', 'forward-0.32', 'forward-0.36') and c.seed % 10 == 0]
    data, panels = [], []
    for name, vec, design in cs:
        for c in conditions:
            key = f'{name}-dv{c.push_dv[0]:.2f}'
            s, rec = simulate(sim, theta, name, vec, c, design)
            actual_c = replace(c, push_time_s=None if c.push_time_s is None else math.ceil(c.push_time_s / .02 - 1e-9) * .02)
            s.update(audit_record(rec, actual_c, sim.z0, sim=sim))
            np.savez_compressed(records / f'{key}.npz', **rec)
            (records / f'{key}.json').write_text(json.dumps(dict(summary=s, condition=asdict(c), design=asdict(design)), indent=2) + '\n')
            d = frames(sim, s, rec, actual_c)
            d['label'] = f"{name}, +{c.push_dv[0]:.2f} m/s — {'quiet recovery' if s['recovered_200hz'] else 'failed recovery'}; {'verified step' if s['recovery_with_step'] else 'no verified recovery step'}"
            data.append(d)
            if name in ('reflex02b', 'cem-s1') and c.push_dv[0] in (.30, .32):
                panels.append((name, actual_c, rec))
    page = PAGE.replace('02B Reflex Replay', '02C Recovery Replay').replace('Microduck 02B: learned IMU reflex replay', 'Microduck 02C: learned recovery motions')
    page = page.replace('Recorded simulator states', 'The disturbance cue is supplied; joint motions were learned. Fixed first held-out replicates, including failures. Recorded simulator states')
    page = page.replace('Soles: floor footprint', 'Soles: measured silhouettes (top view: floor footprint)')
    page = page.replace('f.soles.forEach', "(id === 'side' ? f.soles_xz : id === 'front' ? f.soles_yz : f.soles).forEach")
    page = page.replace('P([q[0], q[1], 0])', "P(id === 'side' ? [q[0], 0, q[1]] : id === 'front' ? [0, q[0], q[1]] : [q[0], q[1], 0])")
    initial = next(i for i, d in enumerate(data) if d['label'].startswith('cem-s1, +0.32'))
    page = page.replace('setTrial(0); requestAnimationFrame(tick);', f'sel.value = {initial}; setTrial({initial}); requestAnimationFrame(tick);')
    (out / 'replay.html').write_text(page.replace('__DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf-8', newline='\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    filmstrip(data, out / 'filmstrip.png')
    filmstrip(data, out / 'filmstrip-boundary.png', .32)
    fig, axes = plt.subplots(4, len(panels), figsize=(14, 9), sharex='col', constrained_layout=True)
    for col, (name, c, r) in enumerate(panels):
        t = r['time_s'] - c.push_time_s
        axes[0, col].plot(t, r['tilt_deg']); axes[0, col].axhline(45, color='red', ls=':', lw=.8)
        axes[0, col].set(title=f'{name}, {c.push_dv[0]:.2f} m/s', ylabel='Trunk tilt (deg)', ylim=(0, 70))
        for foot in range(2):
            axes[1, col].plot(t, r['clearance_m'][:, foot] * 1000, label='LR'[foot])
            axes[2, col].plot(t, r['foot_force_n'][:, foot], label='LR'[foot])
        axes[1, col].set(ylabel='Lowest sole point (mm)', ylim=(-3, 15))
        axes[2, col].set(ylabel='Sole normal force (N)', ylim=(0, 15))
        axes[3, col].plot(t, r['residual'][:, [2, 3, 4, 11, 12, 13]])
        axes[3, col].set(ylabel='Learned leg offsets (rad)', xlabel='Time after push (s)', xlim=(-.3, 2), ylim=(-.8, .8))
        for ax in axes[:, col]:
            ax.axvline(0, color='gray', ls=':', lw=.8); ax.grid(alpha=.2)
    axes[1, 0].legend(fontsize=8)
    fig.savefig(out / 'physical-response.png', dpi=150)
    plt.close(fig)
    print(f'wrote {len(data)} recorded replays')


if __name__ == '__main__':
    main()
