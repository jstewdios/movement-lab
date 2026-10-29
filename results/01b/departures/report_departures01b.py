"""Matched-control departure report, continuous plots and evidence table."""
import base64
import html
import json
from pathlib import Path
import numpy as np
from movement_lab.departure_metrics01b import metrics
from movement_lab.discovery_metrics import longest
from movement_lab.evaluate import sha

ROOT=Path('results/01b/departures')


def collect():
    records={}
    with np.load(ROOT/'baseline/zero/trial.npz') as z:control=dict(z)
    order=['baseline','head-negative','head-positive','flex','shift-negative','shift-positive','unilateral','final-control']
    for name in order:
        p=ROOT/name;files=list(p.glob('*/metadata.json'))
        if not files:continue
        m=json.loads(files[0].read_text())
        with np.load(files[0].parent/'trial.npz') as z:a=dict(z)
        result=metrics(a,control)
        t=a['time_s'];active=(t>=2)&(t<3.6);tail=t>=5.5;dt=np.diff(t,prepend=0)
        amps=np.array([m['amplitudes_rad'].get(n,0) for n in m['joint_names']]);dj=a['joint_position_rad']-control['joint_position_rad'][:len(t)]
        result['commanded_joint_signed_extrema_rad']={n:[float(dj[active,i].min()),float(dj[active,i].max())] for i,n in enumerate(m['joint_names']) if amps[i]!=0}
        result['residual_peak_rad']=float(abs(a['residual_rad']).max())
        result['max_tilt_deg']=m['summary']['max_tilt_deg'];result['worst_penetration_mm']=-1000*float(a['native_min_contact_distance_m'].min())
        result['foot_unloading_longest_s']=[longest((abs(a['foot_force_w_n'][:,i,2])<.1)&(t>=2),dt) for i in range(2)]
        result['foot_off_longest_s']=[longest((a['foot_found'][:,i]==0)&(t>=2),dt) for i in range(2)]
        load=abs(a['foot_force_w_n'][:,:,2]);share=load[:,0]/load.sum(1).clip(1e-8)
        cload=abs(control['foot_force_w_n'][:len(t),:,2]);cshare=cload[:,0]/cload.sum(1).clip(1e-8)
        result['matched_peak_left_load_share_change_pp']=100*float(abs(share[active]-cshare[active]).max())
        result['matched_hold_mean_foot_fz_change_n']=(a['foot_force_w_n'][(t>=2.8)&(t<3),:,2]-control['foot_force_w_n'][:len(t)][(t>=2.8)&(t<3),:,2]).mean(0).tolist()
        result['matched_final_foot_fz_change_n']=(a['foot_force_w_n'][tail,:,2]-control['foot_force_w_n'][:len(t)][tail,:,2]).mean(0).tolist()
        footids=[m['body_names'].index(n) for n in ('ankle_left','ankle_right')]
        dfoot=a['body_pose_w'][:,footids,:3]-control['body_pose_w'][:len(t),footids,:3]
        result['matched_peak_ankle_origin_shift_m']=np.linalg.norm(dfoot[t>=2],axis=2).max(0).tolist()
        result['postsettle_work_j']=float((abs(a['joint_force_nm']*a['joint_velocity_rad_s'])*dt[:,None])[t>=2].sum())
        records[name]=dict(sequence=m['sequence'],metrics=result,summary=m['summary'],joint_names=m['joint_names'],data=a,metadata=m)
    return records,control


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records,c=collect()
    summary={name:{k:v for k,v in r.items() if k not in ('data','metadata')} for name,r in records.items()}
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(3,2,figsize=(13,11),sharex=True)
    def tilt(a):
        q=a['body_pose_w'][:,0,3:7]
        return np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    for name,r in records.items():
        a=r['data'];t=a['time_s'];n=len(t)
        if name in ('baseline','final-control'):continue
        dj=a['joint_position_rad']-c['joint_position_rad'][:n];dc=a['com_position_w_m']-c['com_position_w_m'][:n]
        f=abs(a['foot_force_w_n'][:,:,2]);cf=abs(c['foot_force_w_n'][:n,:,2])
        curves=[abs(dj).max(1),1000*np.linalg.norm(dc,axis=1),1000*dc[:,1],
            100*(f[:,0]/f.sum(1).clip(1e-8)-cf[:,0]/cf.sum(1).clip(1e-8)),tilt(a)-tilt(c)[:n],abs(a['joint_force_nm']).max(1)]
        for ax,y in zip(axes.flat,curves):ax.plot(t,y,label=name,linewidth=1)
    titles=['Largest actual joint difference from control (rad)','COM distance from control (mm)',
        'COM lateral difference from control (mm)','Left fraction of |foot Fz|, difference (percentage points)',
        'Trunk tilt difference from control (degrees)','Largest absolute joint effort (Nm; endpoint estimate)']
    for ax,title in zip(axes.flat,titles):
        ax.set_title(title,fontsize=10);ax.axvspan(2,3.6,alpha=.1,color='gray');ax.grid(alpha=.2);ax.set_xlim(1.5,6)
    axes[0,0].legend(fontsize=8);axes[2,0].set_xlabel('Time (s)');axes[2,1].set_xlabel('Time (s)')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=140);plt.close(fig)
    moving=[(n,r) for n,r in records.items() if r['metadata']['amplitudes_rad']]
    fig,axes=plt.subplots(3,2,figsize=(13,12),sharex=True)
    for ax,(name,r) in zip(axes.flat,moving):
        a=r['data'];n=len(a['time_s'])
        for joint in r['metadata']['amplitudes_rad']:
            i=r['joint_names'].index(joint)
            line,=ax.plot(a['time_s'],a['joint_position_rad'][:,i]-c['joint_position_rad'][:n,i],label=joint)
            ax.plot(a['time_s'],a['joint_target_rad'][:,i]-c['joint_target_rad'][:n,i],color=line.get_color(),linestyle='--',alpha=.6)
        ax.set_title(name+' — actual solid / target dashed');ax.axvspan(2,3.6,color='gray',alpha=.1)
        ax.grid(alpha=.2);ax.set_xlim(1.5,6);ax.set_ylabel('Difference from zero control (rad)');ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'joint-tracking.png',dpi=140);plt.close(fig)
    rows=[]
    for name,r in records.items():
        m=r['metrics'];response=m.get('matched_peak_joint_response_rad',0)
        rows.append(f"<tr><td><a href='{name}/replay.html'>{html.escape(name)}</a></td><td>{response:.5f}</td><td>{1000*m.get('matched_peak_com_response_m',0):.3f}</td><td>{m['worst_penetration_mm']:.3f}</td><td>{html.escape(m['classification'])}</td></tr>")
    joint_image=base64.b64encode((ROOT/'joint-tracking.png').read_bytes()).decode()
    image=base64.b64encode((ROOT/'comparison.png').read_bytes()).decode()
    (ROOT/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><title>01B controlled departures</title><style>body{font:16px system-ui;max-width:1200px;margin:30px auto}td,th{padding:8px;border-bottom:1px solid #bbb}img{width:100%}</style><h1>01B measured departures</h1><p>Shading: residual active (2–3.6 s); stabilizing actor stays active. Differences use the independently validated zero-residual control. Links replay saved measured link centers, without reintegrating dynamics. Read REPORT.md for criteria, provenance and limitations.</p><table><tr><th>Trial / replay</th><th>Peak joint response rad</th><th>Peak COM response mm</th><th>Worst penetration mm</th><th>Outcome</th></tr>'+''.join(rows)+'</table><img src="data:image/png;base64,'+image+'"><h2>Commanded joints: composed target versus measured response</h2><img src="data:image/png;base64,'+joint_image+'">')
    print(json.dumps({n:r['metrics'] for n,r in records.items()},indent=2))


if __name__=='__main__':main()
