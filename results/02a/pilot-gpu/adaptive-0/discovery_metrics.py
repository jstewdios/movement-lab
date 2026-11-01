"""Conservative descriptive atlas, never a skill-success reward."""
import argparse
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha


def longest(mask, dt):
    best = run = 0.0
    for yes, duration in zip(mask, dt):
        run = run + duration if yes else 0.0
        best = max(best, run)
    return float(best)


def analyze(a, initial, limits):
    if not all(np.isfinite(v).all() for v in a.values()):
        return dict(valid=False, classification='invalid nonfinite simulation')
    t = a['time_s']
    dt = np.diff(t, prepend=0)
    q = a['body_pose_w'][:, 0, 3:7]
    tilt = np.rad2deg(np.arccos(np.clip(1-2*(q[:, 1]**2+q[:, 2]**2), -1, 1)))
    q0 = initial['body_pose_w'][0, 3:7]
    rotation = 2*np.rad2deg(np.arccos(np.clip(np.abs(q @ q0), 0, 1)))
    com = a['com_position_w_m']
    delta = com - initial['com_position_w_m']
    tail = t >= t[-1] - .5
    contact = a['foot_found'] > 0
    foot_air = ~contact.any(axis=1)
    air = foot_air & (a['native_ground_contacts'] == 0)
    actual = a['joint_position_rad']
    overshoot = np.maximum(limits[:, 0] - actual, actual - limits[:, 1])
    valid = bool(np.all(dt > 0) and np.max(np.abs(np.linalg.norm(a['body_pose_w'][..., 3:7], axis=-1)-1)) < 1e-4
                 and a['native_min_contact_distance_m'].min() >= -.01
                 and overshoot.max() <= .03 and np.max(np.abs(a['qvel'])) <= 100
                 and np.linalg.norm(a['foot_force_w_n'], axis=-1).max() <= 100)
    stable = bool(t[-1] >= 2.99 and tilt[tail].max() < 20
                  and np.linalg.norm(a['com_velocity_w_m_s'][tail], axis=-1).max() < .05
                  and np.linalg.norm(a['body_velocity_w'][tail, 0, 3:], axis=-1).max() < .5
                  and contact[tail].any(axis=1).all() and (a['native_nonfoot_contacts'][tail] == 0).all())
    fallen = bool(tilt.max() > 60)
    classification = ('invalid simulation/contact audit' if not valid else
                      'uncontrolled fall / body contact' if fallen else
                      'upright quiet endpoint' if stable else 'unsettled / nonupright endpoint')
    return dict(valid=valid, classification=classification, final_stability_gate=stable,
        duration_s=float(t[-1]), com_displacement_m=delta[-1].tolist(),
        max_horizontal_com_excursion_m=float(np.linalg.norm(delta[:, :2], axis=-1).max()),
        com_height_min_max_m=[float(com[:, 2].min()), float(com[:, 2].max())],
        max_com_height_gain_m=float(delta[:, 2].max()), max_tilt_deg=float(tilt.max()),
        max_orientation_change_deg=float(rotation.max()), final_tilt_deg=float(tilt[-1]),
        joint_range_rad=np.ptp(actual, axis=0).tolist(),
        peak_tracking_error_rad=float(np.abs(a['joint_target_rad']-actual).max()),
        peak_command_offset_rad=float(np.abs(a['action']).max()),
        foot_contact_fraction=contact.mean(0).tolist(),
        feet_off_total_s=float(dt[foot_air].sum()), feet_off_longest_s=longest(foot_air, dt),
        all_ground_off_total_s=float(dt[air].sum()), all_ground_off_longest_s=longest(air, dt),
        nonfoot_contact_total_s=float(dt[a['native_nonfoot_contacts'] > 0].sum()),
        min_native_contact_distance_m=float(a['native_min_contact_distance_m'].min()),
        max_joint_limit_overshoot_rad=float(max(0, overshoot.max())),
        peak_joint_effort_nm=float(np.abs(a['joint_force_nm']).max()),
        rms_joint_effort_nm=float(np.sqrt(np.mean(a['joint_force_nm']**2))),
        absolute_mechanical_work_j=float((np.abs(a['joint_force_nm']*a['joint_velocity_rad_s'])*dt[:, None]).sum()),
        final_com_speed_m_s=float(np.linalg.norm(a['com_velocity_w_m_s'][-1])),
        final_root_angular_speed_rad_s=float(np.linalg.norm(a['body_velocity_w'][-1, 0, 3:])))


def publish(out):
    out = Path(out)
    records, playback = {}, {}
    for f in sorted(out.glob('*/metadata.json')):
        m = json.loads(f.read_text())
        with np.load(f.parent / 'trial.npz') as z, np.load(f.parent / 'initial.npz') as ini:
            a = dict(z)
            recalculated = analyze(a, dict(ini), np.array(m['joint_limits_rad']))
            assert recalculated == m['summary'], f'recomputed metrics differ: {f}'
            assert all(len(v) == len(a['time_s']) for v in a.values())
            assert np.allclose(a['joint_target_rad'], np.array(m['target_offset_rad']) + a['action'], atol=1e-6)
            records[m['sequence']] = m
            playback[m['sequence']] = dict(t=a['time_s'][::4].tolist(),
                p=a['body_pose_w'][::4, :, :3].tolist(),
                q=a['joint_position_rad'][::4].tolist(), target=a['joint_target_rad'][::4].tolist())
    archive = {}
    for name, m in records.items():
        s = m['summary']
        if not s['valid']:
            key = 'invalid'
        else:
            # Fixed descriptive bins, keep EVERY member; no scalar optimization.
            key = str((s['classification'], int(s['max_tilt_deg']//15),
                       int(s['max_horizontal_com_excursion_m']//.025),
                       int(max(s['joint_range_rad'])//.1)))
        archive.setdefault(key, []).append(name)
    validation = dict(recomputed_metrics=True, synchronized_arrays=True, command_mapping=True)
    if 'zero' in records and 'zero_repeat' in records:
        with np.load(out/'zero/trial.npz') as a, np.load(out/'zero_repeat/trial.npz') as b:
            validation['baseline_repeat_max_qpos_difference'] = float(np.max(np.abs(a['qpos']-b['qpos']))) if a['qpos'].shape == b['qpos'].shape else None
        with np.load(out/'zero/initial.npz') as ref:
            validation['max_initial_qpos_difference'] = max(float(np.max(np.abs(np.load(out/n/'initial.npz')['qpos']-ref['qpos']))) for n in records)
    (out/'atlas.json').write_text(json.dumps(dict(records=records, archive=archive), indent=2)+'\n')
    (out/'validation.json').write_text(json.dumps(validation, indent=2)+'\n')
    # Portable state replay of measured link centers; no physics is reintegrated.
    html = '''<!doctype html><meta charset="utf-8"><title>01A recorded state replay</title>
<style>body{font:16px system-ui;max-width:1100px;margin:30px auto}canvas{border:1px solid #aaa}pre{white-space:pre-wrap}</style>
<h1>01A measured link-center replay</h1><p>Orthographic x–z and y–z views. Dots are link origins, not collision geometry. Slider replays recorded states, not a new simulation. Inspect joint targets versus actual positions below. Falls and feet-off intervals are not skills.</p>
<select id="trial"></select> <button id="play">Play / pause</button><input id="frame" type="range" min="0" value="0" style="width:50%"><span id="time"></span><br><canvas id="view" width="1000" height="400"></canvas><pre id="info"></pre>
<script>const data=DATA, meta=META; const sel=document.getElementById('trial'), slider=document.getElementById('frame'), c=document.getElementById('view'),ctx=c.getContext('2d');
for(const name of Object.keys(data)){sel.add(new Option(name,name))} let running=false;
function draw(){let d=data[sel.value],i=+slider.value;slider.max=d.t.length-1;i=Math.min(i,+slider.max);document.getElementById('time').textContent=d.t[i].toFixed(2)+' s';ctx.clearRect(0,0,1000,400);let ref=data[sel.value].p[0][0];for(let axis=0;axis<2;axis++){let mid=250+500*axis;ctx.strokeStyle='#999';ctx.beginPath();ctx.moveTo(mid-240,350);ctx.lineTo(mid+240,350);ctx.stroke();ctx.fillStyle='#333';ctx.fillText(axis?'y–z':'x–z',mid,20);for(let p of d.p[i]){ctx.fillStyle='#176b9a';ctx.beginPath();ctx.arc(mid+700*(p[axis]-ref[axis]),350-700*p[2],4,0,7);ctx.fill()}}
let m=meta[sel.value];document.getElementById('info').textContent=m.summary.classification+'\\n'+m.joint_names.map((n,j)=>n+': actual '+d.q[i][j].toFixed(3)+' / target '+d.target[i][j].toFixed(3)+' rad').join('\\n');}
sel.onchange=()=>{slider.value=0;draw()};slider.oninput=draw;document.getElementById('play').onclick=()=>running=!running;setInterval(()=>{if(running){slider.value=(+slider.value+1)%(+slider.max+1);draw()}},80);draw();</script>'''
    (out/'replay.html').write_text(html.replace('DATA', json.dumps(playback)).replace('META', json.dumps(records)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for name in records:
        with np.load(out/name/'trial.npz') as a:
            t=a['time_s']; com=a['com_position_w_m']
            axes[0,0].plot(t,com[:,2],label=name)
            axes[0,1].plot(com[:,0],com[:,1],label=name)
            q=a['body_pose_w'][:,0,3:7]
            axes[1,0].plot(t,np.rad2deg(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1))))
            axes[1,1].plot(t,np.linalg.norm(a['joint_target_rad']-a['joint_position_rad'],axis=1))
    for ax, title in zip(axes.flat, ['COM height (m) versus time (s)', 'COM x versus y (m)', 'Trunk tilt (degrees) versus time (s)', 'Joint tracking error L2 (rad) versus time (s)']):
        ax.set_title(title); ax.grid(alpha=.3)
    axes[0,0].legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'atlas.png');plt.close(fig)
    manifest={str(f.relative_to(out)):sha(f) for f in sorted(out.rglob('*')) if f.is_file() and f.name != 'manifest.json' and 'matplotlib' not in f.parts and f.suffix not in {'.log'}}
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('out',type=Path);publish(p.parse_args().out)
