"""Minimal matched matrix and offline comparison for 00B."""
import argparse
import html
import io
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from movement_lab.disturbance import assess, signals


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = []
    for path in sorted(root.glob('*/metadata.json')):
        m = json.loads(path.read_text())
        a = dict(np.load(path.parent/'trial.npz', allow_pickle=False))
        if 'disturbance' not in m:
            continue
        s = assess(a,m)
        records.append((path.parent.name,a,m,s))
    (root/'summary.json').write_text(json.dumps({n:s for n,a,m,s in records},indent=2)+'\n')
    parts = ['<!doctype html><meta charset="utf-8"><title>Movement Lab 00B</title>',
             '<style>body{font:15px system-ui;margin:30px}table{border-collapse:collapse}td,th{padding:6px;border:1px solid #aaa}svg{max-width:1100px;width:100%}</style>',
             '<h1>Movement Lab 00B — Disturbance and Recovery</h1>',
             '<p>Default-position trials terminate before standing; their recovery measurements are unavailable. A reported 0.02 s recovery is the earliest eligible window and does not resolve fast response differences.</p>',
             '<p>World-x root velocity increments at 2 s; zero twist; BAM and physics-rate recording as in 00A. Positive x is nominal forward at initial yaw zero. This is a state/momentum intervention, not a calibrated external force or local trunk impact. Two seeds test repeatability under nominal conditions, not a randomized robustness population.</p>',
             '<p>Settled: all samples in [1.5,2) s have speed &lt;0.04 m/s, angular speed &lt;0.5 rad/s, tilt &lt;12°, and both foot contacts. Recovery: a continuous 0.5 s window with those bounds, tilt within 3° of baseline and height within 2 cm, plus a stable final 0.5 s and no termination. New standing locations are permitted and reported. Thresholds are exploratory and fixed before comparison.</p>',
             '<table><tr><th>Trial</th><th>Settled / upright / recovered</th><th>Recovery s</th><th>Peak tilt °</th><th>Final root shift m</th><th>RMS Nm / work J</th><th>Foot shift m L/R</th><th>Step evidence L/R</th></tr>']
    def fmt(value):
        if value is None: return 'unavailable'
        if isinstance(value,list): return ' / '.join(fmt(v) for v in value)
        if isinstance(value,bool): return str(value)
        return f'{value:.4f}'
    for n,a,m,s in records:
        cells=[n, f"{s['settled']} / {s['upright']} / {s['recovered']}",fmt(s['recovery_s']),fmt(s['peak_tilt_deg']),fmt(s['final_root_shift_m']),fmt(s['rms_effort_nm'])+' / '+fmt(s['work_j']),fmt(s['foot_final_shift_m']),fmt(s['step_evidence'])]
        parts.append('<tr>'+''.join('<td>'+html.escape(c)+'</td>' for c in cells)+'</tr>')
    parts.append('</table>')
    parts.append('<h2>Policy comparison</h2><table><tr><th>Checkpoint</th><th>Settled / trials</th><th>Recovered disturbed / valid disturbed</th><th>Mean post-event RMS Nm</th><th>Mean post-event work J</th></tr>')
    for checkpoint in ('99','348'):
        group=[r for r in records if r[0].startswith(checkpoint+'-')]
        valid=[r for r in group if r[3]['settled']]
        disturbed=[r for r in valid if r[2]['disturbance']['requested_dv_m_s'] != 0]
        if valid:
            parts.append(f'<tr><td>{checkpoint}</td><td>{len(valid)} / {len(group)}</td><td>{sum(r[3]["recovered"] for r in disturbed)} / {len(disturbed)}</td><td>{np.mean([r[3]["rms_effort_nm"] for r in valid]):.5f}</td><td>{np.mean([r[3]["work_j"] for r in valid]):.6f}</td></tr>')
    parts.append('</table><p>Means above pool the fixed conditions equally. They are descriptive, not population estimates. Recovery at the earliest eligible window is a threshold floor; compare continuous trajectories. No direct stiffness measurement is made.</p>')
    notes=root/'VALIDATION.md'
    if notes.exists():
        parts.append('<details open><summary>Evidence, validation, and limitations</summary><pre style="white-space:pre-wrap">'+html.escape(notes.read_text())+'</pre></details>')
    for seed in sorted({m['seed'] for _,_,m,_ in records}):
        for dv in sorted({m['disturbance']['requested_dv_m_s'] for _,_,m,_ in records}):
            group=[r for r in records if r[2]['seed']==seed and r[2]['disturbance']['requested_dv_m_s']==dv]
            fig,axes=plt.subplots(9,1,figsize=(11,18),layout='constrained')
            for n,a,m,s in group:
                t=a['time_s']-m['disturbance'].get('actual_time_s',2); pre=(t>=-.5)&(t<0)
                if not pre.any(): pre=np.arange(len(t))<1
                axes[0].plot(t,signals(a),label=n)
                axes[1].plot(t,a['body_pose_w'][:,0,0]-a['body_pose_w'][pre,0,0].mean())
                axes[2].plot(t,a['body_velocity_w'][:,0,0])
                axes[3].plot(t,np.sqrt(np.mean(a['joint_force_nm']**2,axis=1)))
                axes[7].plot(t,a['body_pose_w'][:,0,2])
                axes[8].plot(t,np.linalg.norm(a['body_velocity_w'][:,0,3:],axis=1))
                for f,b in enumerate(('ankle_left','ankle_right')):
                    idx=m['body_names'].index(b)
                    axes[4].plot(t,a['body_pose_w'][:,idx,0]-a['body_pose_w'][pre,idx,0].mean(),label=n+f' foot {f}')
                    axes[5].plot(t,a['foot_force_w_n'][:,f,2])
                    axes[6].step(t,(a['foot_found'][:,f]>0).astype(float)+f*1.2,where='post')
            for ax,label in zip(axes,['Trunk tilt °','Root Δx m','Root vx m/s','Joint RMS Nm','Foot Δx m','Raw foot Fz N','Contact L / R+1.2','Root height m','Root angular speed rad/s']):
                ax.set_ylabel(label);ax.axvline(0,color='k',alpha=.3);ax.grid(alpha=.2)
            axes[0].legend(fontsize=7);axes[4].legend(fontsize=6);axes[-1].set_xlabel('Time since disturbance (s)')
            if seed == 43 and dv == .06:
                fig.savefig(root/'trajectory-preview.png',dpi=120)
            buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
            parts.extend([f'<h2>Seed {seed}, Δvx {dv} m/s</h2>',buf.getvalue()])
    parts.append('<p>Effort is qfrc_actuator and sampled absolute mechanical work, not electrical consumption or stiffness. Foot displacement may reflect sliding/rocking. Step evidence requires sampled loss of contact, &gt;5 mm ankle lift and &gt;5 mm displaced recontact; it is conservative, not a validated gait classifier. Foot force retains 00A’s negative support sign. Feet-only collision model and extra forward solves limit physical interpretation. See VALIDATION.md for checks and conclusions.</p>')
    for n,a,m,s in records:
        parts.append('<details><summary>'+html.escape(n)+' provenance</summary><pre>'+html.escape(json.dumps(m,indent=2))+'</pre></details>')
    (root/'comparison.html').write_text('\n'.join(parts))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/00b'))
    p.add_argument('--run',action='store_true')
    args=p.parse_args()
    if args.run:
        checkpoints=[json.loads(Path(f'results/{c}-balance/metadata.json').read_text())['checkpoint'] for c in ('99','348')]
        for seed in (42,43):
            for dv in (0.,-.03,.03,-.06,.06):
                for label,ckpt,default in [('99',checkpoints[0],False),('348',checkpoints[1],False),('default',checkpoints[0],True)]:
                    if default and dv != 0.:
                        continue  # default controller cannot establish standing; controls suffice
                    dest=args.out/f'{label}-s{seed}-dv{dv:+.2f}'
                    if (dest/'metadata.json').exists():
                        continue
                    cmd=[sys.executable,'-m','movement_lab.evaluate',ckpt,'--out',str(dest),'--seed',str(seed),'--seconds','4','--disturbance-dv',str(dv)]
                    if default:cmd.append('--default-position')
                    subprocess.run(cmd,check=True)
    report(args.out)

if __name__=='__main__':main()
