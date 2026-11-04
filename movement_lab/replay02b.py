"""02B portable replay: recorded states as a self-contained HTML stick figure.

    python -m movement_lab.replay02b --root results/02b

Each trial is simulated once (named seed, recorded at 200 Hz) and decimated to
50 Hz. Drawn per frame from the recorded state only (no dynamics are re-run in
the browser): link frame origins joined along the kinematic tree, the head's
centre of mass, both sole footprints (convex hull of each sole mesh projected
to the floor; filled when the sole touches the floor), the whole-body COM and
its floor projection, and the reflex's commanded joint offsets. No mesh
geometry is embedded, so no upstream robot asset is redistributed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
from movement_lab import sim02b
from movement_lab.physics02b import convex_hull
from movement_lab.sim02b import Condition


def trials(root):
    th = lambda run: np.load(root / 'runs' / run / 'thetas.npz')['final']
    zero = sim02b.zero_theta()
    base = dict(duration_s=5.0, joint_noise_rad=0.0)
    return [
        ('Fixed HOME targets, no push: topples forward', zero, Condition(seed=930_000_001, **base)),
        ('Learned reflex (ars-push-s1), no push: stands', th('ars-push-s1'), Condition(seed=930_000_001, **base)),
        ('Learned reflex, forward 0.2 m/s push at 2 s', th('ars-push-s1'), Condition(seed=930_000_002, push_time_s=2.0, push_dv=(0.2, 0.0), **base)),
        ('Learned reflex, backward 0.2 m/s push at 2 s (beyond heel limit)', th('ars-push-s1'), Condition(seed=930_000_003, push_time_s=2.0, push_dv=(-0.2, 0.0), **base)),
        ('Learned reflex, left 0.3 m/s push at 2 s', th('ars-push-s1'), Condition(seed=930_000_004, push_time_s=2.0, push_dv=(0.0, 0.3), **base)),
        ('Random-search winner (random-push-s1), forward 0.2 m/s', th('random-push-s1'), Condition(seed=930_000_002, push_time_s=2.0, push_dv=(0.2, 0.0), **base)),
        # Held-out push_grid trial (seed 900000208): 0.4 m/s at 240 deg, beyond the ~0.30 m/s no-step limit.
        ('Push-trained reflex (ars-push-s3), 0.4 m/s at 240° (held-out): sole lifts, survives', th('ars-push-s3'),
         Condition(seed=900_000_208, duration_s=6.0, joint_noise_rad=0.02, push_time_s=3.0, push_dv=(0.4 * np.cos(np.radians(240)), 0.4 * np.sin(np.radians(240))))),
        ('Standing-only reflex (ars-none-s3), same trial: sole lifts, falls', th('ars-none-s3'),
         Condition(seed=900_000_208, duration_s=6.0, joint_noise_rad=0.02, push_time_s=3.0, push_dv=(0.4 * np.cos(np.radians(240)), 0.4 * np.sin(np.radians(240))))),
    ]


def capture(sim, theta, cond):
    import mujoco
    s, rec = sim02b.rollout(sim, theta, cond, record=True, terminate=False)
    m, aux = sim.m, sim.aux
    bodies = list(range(2, m.nbody))  # skip world and terrain
    parent = [int(m.body_parentid[b]) for b in bodies]
    # The floor projection of a mesh's hull = hull of its 3-D hull vertices (exact, cheaper).
    from scipy.spatial import ConvexHull
    sole_verts = []
    for g in sim.feet:
        mesh = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[a:a + n]
        sole_verts.append(v[ConvexHull(v).vertices])
    frames = []
    for i in range(3, len(rec['time_s']), 4):  # end of each 20 ms control period
        aux.qpos[:] = rec['qpos'][i]
        mujoco.mj_kinematics(m, aux)
        mujoco.mj_comPos(m, aux)
        soles = []
        for g, hv in zip(sim.feet, sole_verts):
            v = hv @ aux.geom_xmat[g].reshape(3, 3).T + aux.geom_xpos[g]
            h = convex_hull(v[:, :2])
            h = h[np.linspace(0, len(h), min(len(h), 24), endpoint=False).astype(int)]  # display outline only
            soles.append(np.round(1000 * h).astype(int).tolist())
        frames.append(dict(t=round(float(rec['time_s'][i]), 3),
                           p=np.round(1000 * aux.xpos[bodies]).astype(int).tolist(),
                           head=np.round(1000 * aux.xipos[sim.head]).astype(int).tolist(),
                           com=np.round(1000 * aux.subtree_com[sim.trunk], 1).tolist(),
                           soles=soles, contact=[bool(x) for x in rec['foot_found'][i]],
                           tilt=round(float(rec['tilt_deg'][i]), 1)))
    idx = {b: k for k, b in enumerate(bodies)}
    edges = [[idx[b], idx[p]] for b, p in zip(bodies, parent) if p in idx]
    summary = {k: s[k] for k in ('survived', 'fall_time_s', 'fall_cause', 'max_tilt_deg', 'n_steps')}
    return dict(edges=edges, frames=frames, summary=summary, push_time=cond.push_time_s, push_dv=list(cond.push_dv))


PAGE = r"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>02B Reflex Replay</title>
<style>
:root{--bg:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e4e3df;--robot:#2a78d6;--com:#eb6834;--sole:#86b6ef;--on:#1c5cab;--fall:#e34948}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#383835;--robot:#3987e5;--com:#d95926;--sole:#184f95;--on:#86b6ef;--fall:#e66767}}
:root[data-theme="dark"]{--bg:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#383835;--robot:#3987e5;--com:#d95926;--sole:#184f95;--on:#86b6ef;--fall:#e66767}
body{margin:0;padding:16px;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif;max-width:1100px}
h1{font-size:18px;margin:0 0 4px}p{color:var(--ink2);margin:4px 0 12px}
.row{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin:8px 0}
select,button,input{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--grid);border-radius:6px;padding:4px 8px}
.views{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}
canvas{width:100%;height:auto;border:1px solid var(--grid);border-radius:8px}
.cap{font-size:12px;color:var(--ink2)}#info{font-variant-numeric:tabular-nums}
</style></head><body>
<h1>Microduck 02B: learned IMU reflex replay</h1>
<p>Recorded simulator states (CPU MuJoCo + upstream BAM XL330 model), 50 Hz, played back without re-simulation. Blue: link frames along the kinematic tree, circle = head centre of mass. Soles: floor footprint, filled while touching the floor. Orange: whole-body COM and its floor projection. Stick figure only; no robot meshes are embedded.</p>
<div class="row"><select id="trial"></select><button id="play">Pause</button>
<label>speed <select id="speed"><option value="0.25">0.25×</option><option value="0.5" selected>0.5×</option><option value="1">1×</option></select></label>
<input id="scrub" type="range" min="0" max="0" value="0" style="flex:1;min-width:160px"></div>
<div id="info" class="cap"></div>
<div class="views"><div><canvas id="side" width="520" height="380"></canvas><div class="cap">Side view (x forward → right, z up)</div></div>
<div><canvas id="front" width="520" height="380"></canvas><div class="cap">Front view (y left → left, z up)</div></div>
<div><canvas id="top" width="520" height="380"></canvas><div class="cap">Top view: support footprints and COM projection (x up, y left)</div></div></div>
<script>
const DATA = __DATA__;
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const sel = document.getElementById('trial'), scrub = document.getElementById('scrub'), info = document.getElementById('info');
DATA.forEach((d, i) => { const o = document.createElement('option'); o.value = i; o.textContent = d.label; sel.appendChild(o); });
let cur = DATA[0], k = 0, playing = true, last = null, acc = 0;
function setTrial(i){ cur = DATA[i]; k = 0; scrub.max = cur.frames.length - 1; scrub.value = 0; draw(); }
sel.onchange = () => setTrial(+sel.value);
scrub.oninput = () => { k = +scrub.value; draw(); };
document.getElementById('play').onclick = e => { playing = !playing; e.target.textContent = playing ? 'Pause' : 'Play'; };
function view(id, ax, ay, flip){
  const c = document.getElementById(id), g = c.getContext('2d'), W = c.width, H = c.height;
  g.fillStyle = css('--bg'); g.fillRect(0, 0, W, H);
  // Fixed camera on the initial COM so drift and falls stay visible.
  const f = cur.frames[k], f0 = cur.frames[0], s = 1.3, st = 2.6, cx = W / 2, cy = id === 'top' ? H / 2 : H - 40;
  const P = (p) => id === 'top' ? [cx - st * (p[1] - f0.com[1]), cy - st * (p[0] - f0.com[0])] : [cx + flip * s * (p[ax] - f0.com[ax]), cy - s * p[ay]];
  g.strokeStyle = css('--grid'); g.lineWidth = 1;
  if (id !== 'top') { g.beginPath(); g.moveTo(0, cy); g.lineTo(W, cy); g.stroke(); }
  f.soles.forEach((poly, j) => { g.beginPath(); poly.forEach((q, i) => { const p = P([q[0], q[1], 0]); i ? g.lineTo(p[0], p[1]) : g.moveTo(p[0], p[1]); });
    g.closePath(); g.fillStyle = f.contact[j] ? css('--on') : 'transparent'; g.globalAlpha = id === 'top' ? 0.5 : 0.8; g.fill(); g.globalAlpha = 1;
    g.strokeStyle = css('--sole'); g.lineWidth = 2; g.stroke(); });
  if (id !== 'top') { g.strokeStyle = css('--robot'); g.lineWidth = 3; g.lineCap = 'round';
    cur.edges.forEach(([a, b]) => { const p = P(f.p[a]), q = P(f.p[b]); g.beginPath(); g.moveTo(p[0], p[1]); g.lineTo(q[0], q[1]); g.stroke(); });
    const h = P(f.head); g.beginPath(); g.arc(h[0], h[1], 18, 0, 7); g.stroke(); }
  const c3 = P(f.com), c0 = P([f.com[0], f.com[1], 0]); g.fillStyle = css('--com');
  g.beginPath(); g.arc(c3[0], c3[1], 5, 0, 7); g.fill(); g.beginPath(); g.arc(c0[0], c0[1], 4, 0, 7); g.fill();
  if (id !== 'top') { g.setLineDash([3, 3]); g.strokeStyle = css('--com'); g.lineWidth = 1; g.beginPath(); g.moveTo(c3[0], c3[1]); g.lineTo(c0[0], c0[1]); g.stroke(); g.setLineDash([]); }
}
function draw(){
  const f = cur.frames[k], pushed = cur.push_time !== null && f.t >= cur.push_time;
  view('side', 0, 2, 1); view('front', 1, 2, -1); view('top', 0, 1, 1);
  const s = cur.summary;
  info.innerHTML = `t = ${f.t.toFixed(2)} s · trunk tilt ${f.tilt.toFixed(1)}° · soles ${f.contact.map(c => c ? 'down' : 'up').join(' / ')} · ` +
    (cur.push_time !== null ? (pushed ? `<b>pushed</b> Δv = (${cur.push_dv.map(v => v.toFixed(2)).join(', ')}) m/s at ${cur.push_time} s · ` : `push at ${cur.push_time} s · `) : '') +
    (s.survived ? 'outcome: survived' : `outcome: <span style="color:var(--fall)">fell at ${s.fall_time_s.toFixed(2)} s (${s.fall_cause})</span>`);
  scrub.value = k;
}
function tick(ts){ if (last === null) last = ts; const dt = (ts - last) / 1000; last = ts;
  if (playing) { acc += dt * (+document.getElementById('speed').value); while (acc >= 0.02) { acc -= 0.02; k = (k + 1) % cur.frames.length; } draw(); }
  requestAnimationFrame(tick); }
setTrial(0); requestAnimationFrame(tick);
</script></body></html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02b'))
    args = p.parse_args(argv)
    sim = sim02b.Sim()
    data = []
    for label, theta, cond in trials(args.root):
        d = capture(sim, theta, cond)
        d['label'] = label
        data.append(d)
        print(label, d['summary'])
    out = args.root / 'report' / 'replay.html'
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(PAGE.replace('__DATA__', json.dumps(data, separators=(',', ':'))), encoding='utf-8')
    print('wrote', out, f'{out.stat().st_size / 1e6:.2f} MB')


if __name__ == '__main__':
    main()
