"""Offline 01C atlas, synchronized measured replay and independent audit."""
import argparse
import html
import json
from pathlib import Path
import numpy as np
from movement_lab.study01c import check
from movement_lab.evaluate import sha
from movement_lab.write_report01c import display

def generate_report(root):
    root=Path(root)
    records={};playback={};audit={};details={}
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    check(root/'baseline',root/'baseline')
    control=dict(np.load(root/'baseline/zero/trial.npz'))
    historical=Path('results/01b/departures/baseline')
    assert (root/'baseline/resolved_config.txt').read_bytes()==(historical/'resolved_config.txt').read_bytes()
    assert sha(root/'baseline/replay_model.mjb')==sha(historical/'replay_model.mjb')
    with np.load(historical/'zero/trial.npz') as old:
        for k in old: assert np.array_equal(control[k][:len(old[k])],old[k]),('historical prefix',k)
    fig,ax=plt.subplots(3,1,figsize=(10,10),sharex=True)
    for f in sorted(root.glob('*/*/metadata.json')):
        name=f.parent.parent.name;out=f.parent.parent;m=json.loads(f.read_text());a=dict(np.load(f.parent/'trial.npz'))
        audit[name]=check(out,root/'baseline')
        import mujoco
        model=mujoco.MjModel.from_binary_path(str(out/'replay_model.mjb'))
        assert (model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT).all()
        moment=np.zeros((model.nu,model.nv))
        for i,j in enumerate(model.actuator_trnid[:,0]):moment[i,model.jnt_dofadr[j]]=model.actuator_gear[i,0]
        joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in m['joint_names']]
        assert np.allclose(a['actuator_force']@moment[:,model.jnt_dofadr[joints]],a['joint_force_nm'],atol=1e-6)
        assert abs(a['joint_force_nm']).max()<=.9601
        prov=json.loads((out/'provenance.json').read_text())
        up=Path(prov['upstream'])
        for n,digest in prov['upstream_file_sha256'].items():assert sha(up/n)==digest
        audit[name]['actuator_mapping_and_effort']=True
        native=mujoco.MjData(model)
        terrain=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'terrain')
        for i in range(len(a['time_s'])):
            native.qpos[:]=a['qpos'][i];native.qvel[:]=a['qvel'][i];mujoco.mj_forward(model,native)
            count=0;nonfoot=0;feet=np.zeros(2,dtype=int)
            for contact in native.contact:
                if terrain in (contact.geom1,contact.geom2):
                    count+=1;g=contact.geom2 if contact.geom1==terrain else contact.geom1
                    gn=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
                    nonfoot+=int(not any(f in gn for f in ('left_foot_collision','right_foot_collision')))
                    for fi,foot in enumerate(('left_foot_collision','right_foot_collision')):feet[fi]+=int(foot in gn)
            assert count==a['native_ground_contacts'][i] and nonfoot==a['native_nonfoot_contacts'][i]
            assert np.array_equal(feet,a['native_foot_contacts'][i])
        audit[name]['independent_all_ground_contacts']=True
        t=a['time_s']; endpoint_available=bool(t[-1]>=9.99);late=(t>=9.5)&endpoint_available;pre=(t>=1.5)&(t<2)
        c={k:v[:len(t)] for k,v in control.items()}
        q=a['body_pose_w'][:,0,3:7];ref=q[pre].mean(0);ref/=np.linalg.norm(ref)
        orientation=2*np.degrees(np.arccos(np.clip(abs(q@ref),0,1)))
        dc=a['com_position_w_m']-a['com_position_w_m'][pre].mean(0)
        dj=a['joint_position_rad']-a['joint_position_rad'][pre].mean(0)
        absolute_return=bool((np.linalg.norm(dc[late,:2],axis=1)<=.005).all() and (abs(dc[late,2])<=.002).all() and (orientation[late]<=3).all() and (abs(dj[late])<=.03).all()) if endpoint_available else None
        audit[name]['absolute_final_return_01b_tolerances']=absolute_return
        audit[name]['maintained_hold_response']=bool(audit[name]['stable_hold'] and max(abs(np.array(audit[name]['hold_joint_delta_rad'])))>=.005) if audit[name]['stable_hold'] is not None else None
        audit[name]['maintained_final_joint_configuration']=bool(audit[name]['stable_final'] and audit[name]['final_joint_rad']>=.005) if endpoint_available else None
        perjoint=[]
        for j,n in enumerate(m['joint_names']):
            delta=a['joint_position_rad'][:,j]-c['joint_position_rad'][:,j]
            target_delta=a['joint_target_rad'][:,j]-c['joint_target_rad'][:,j]
            perjoint.append(dict(joint=n,command_peak_rad=m['amplitudes_rad'].get(n,0),actual_delta_min_rad=float(delta[t>=2].min()) if (t>=2).any() else None,actual_delta_max_rad=float(delta[t>=2].max()) if (t>=2).any() else None,final_delta_rad=float(delta[late].mean()) if endpoint_available else None,final_target_delta_rad=float(target_delta[late].mean()) if endpoint_available else None,velocity_peak_rad_s=float(abs(a['joint_velocity_rad_s'][:,j]).max())))
        details[name]=dict(joints=perjoint,peak_com_velocity_m_s=float(np.linalg.norm(a['com_velocity_w_m_s'],axis=1).max()),peak_root_angular_velocity_rad_s=float(np.linalg.norm(a['body_velocity_w'][:,0,3:],axis=1).max()),load_hold_delta_n=(a['foot_force_w_n'][(t>=3.1)&(t<3.6),:,2]-c['foot_force_w_n'][(t>=3.1)&(t<3.6),:,2]).mean(0).tolist() if audit[name]['stable_hold'] is not None else None)
        feet={}
        for foot in ('ankle_left','ankle_right'):
            bi=m['body_names'].index(foot)
            pos=a['body_pose_w'][:,bi,:3];delta=pos-c['body_pose_w'][:,bi,:3]
            feet[foot]=dict(peak_matched_horizontal_origin_shift_m=float(np.linalg.norm(delta[t>=2,:2],axis=1).max()) if (t>=2).any() else None,
                peak_origin_lift_from_pre_mean_m=float((pos[t>=2,2]-pos[pre,2].mean()).max()) if (t>=2).any() else None,
                final_matched_origin_delta_m=delta[late].mean(0).tolist() if endpoint_available else None)
        details[name]['foot_origin_measurements']=feet
        records[name]=m
        playback[name]=dict(t=t[::4].tolist(),p=a['body_pose_w'][::4,:,:3].tolist(),q=a['joint_position_rad'][::4].tolist(),target=a['joint_target_rad'][::4].tolist(),residual=a['residual_rad'][::4].tolist(),com=a['com_position_w_m'][::4].tolist(),load=a['foot_force_w_n'][::4,:,2].tolist())
        ax[0].plot(t,(a['com_position_w_m'][:,0]-c['com_position_w_m'][:,0])*1000,label=name)
        ax[1].plot(t,(a['com_position_w_m'][:,2]-c['com_position_w_m'][:,2])*1000)
        ax[2].plot(t,abs(a['joint_position_rad']-c['joint_position_rad']).max(1))
    for a,label in zip(ax,['Matched COM x (mm)','Matched COM z (mm)','Max joint response (rad)']):
        a.set_ylabel(label);a.axvspan(2,5,alpha=.12);a.grid(alpha=.2)
    ax[0].legend(ncol=2);ax[-1].set_xlabel('Recorded time (s)');fig.tight_layout();fig.savefig(root/'atlas.png');plt.close(fig)
    (root/'independent-validation.json').write_text(json.dumps(audit,indent=2))
    (root/'joint-analysis.json').write_text(json.dumps(details,indent=2))
    rows=[]
    for n,r in audit.items():
        rows.append(f"<tr><td>{html.escape(n)}</td><td>{display(r['peak_joint_rad'], '.5f')}</td><td>{display(r['final_joint_rad'], '.5f')}</td><td>{display(r['peak_com_mm'], '.3f')}</td><td>{display(r['final_com_mm'], '.3f')}</td><td>{html.escape(r['classification'])}</td></tr>")
    page='''<!doctype html><meta charset="utf-8"><title>01C transition atlas</title><style>body{font:16px system-ui;margin:30px auto;max-width:1150px}td{padding:8px;border-bottom:1px solid #ccc}canvas{border:1px solid #aaa}img{width:100%}pre{white-space:pre-wrap}</style><h1>01C controlled transitions</h1><p>Recorded link origins, without dynamics reintegration. Playback stops at each trial’s last recorded frame; unavailable endpoints are not extrapolated. Trial and control share the same frame and clock. Blue: selected trial, gray: zero control. Geometry is not rendered. Residual active 2–5 s. Playback is four times slower. See REPORT.md for interpretation.</p><table><tr><th>Trial</th><th>Peak joint rad</th><th>Final joint rad</th><th>Peak COM mm</th><th>Final COM mm</th><th>Outcome</th></tr>ROWS</table><img src="atlas.png"><h2>Synchronized measured replay</h2><select id="sel"></select><button id="play">Play / pause</button><input id="frame" type="range" min="0" max="499" value="0"><span id="clock"></span><canvas id="view" width="1100" height="400"></canvas><pre id="info"></pre><script>const data=DATA,meta=META;const sel=document.getElementById('sel'),slider=document.getElementById('frame'),ctx=document.getElementById('view').getContext('2d');for(let n of Object.keys(data))sel.add(new Option(n,n));let run=false;function draw(){let d=data[sel.value],b=data.baseline,i=Math.min(+slider.value,d.t.length-1);slider.max=d.t.length-1;slider.value=i;document.getElementById('clock').textContent=d.t[i].toFixed(3)+' s';ctx.clearRect(0,0,1100,400);for(let axis=0;axis<2;axis++){let mid=275+550*axis;ctx.fillStyle='#222';ctx.fillText(axis?'y–z':'x–z',mid,20);for(let [v,color] of [[b,'#bbb'],[d,'#1679b1']]){for(let p of v.p[i]){ctx.fillStyle=color;ctx.beginPath();ctx.arc(mid+1600*p[axis],370-1600*p[2],3,0,7);ctx.fill()}}}document.getElementById('info').textContent='COM m: '+d.com[i].map(x=>x.toFixed(5))+' | raw foot Fz N: '+d.load[i].map(x=>x.toFixed(3))+'\\n'+meta[sel.value].joint_names.map((n,j)=>n+': residual '+d.residual[i][j].toFixed(4)+' actual '+d.q[i][j].toFixed(4)+' target '+d.target[i][j].toFixed(4)).join('\\n')}sel.onchange=draw;slider.oninput=draw;document.getElementById('play').onclick=()=>run=!run;setInterval(()=>{if(run){slider.value=(+slider.value+1)%data[sel.value].t.length;draw()}},80);draw();</script>'''
    (root/'comparison.html').write_text(page.replace('ROWS',''.join(rows)).replace('DATA',json.dumps(playback)).replace('META',json.dumps(records)))
    for source in ('report01c.py','study01c.py','transition01c.py','write_report01c.py','diagnose01b_for01c.py'):
        (root/('final_'+source)).write_bytes(Path(__file__).with_name(source).read_bytes())
    files=[f for f in root.rglob('*') if f.is_file() and f.name!='manifest.json' and 'matplotlib' not in f.parts]
    (root/'manifest.json').write_text(json.dumps({str(f.relative_to(root)):sha(f) for f in files},indent=2))
    print('PASS independent audit and atlas:',len(audit),'records')

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    generate_report(parser.parse_args().root)


if __name__=='__main__':main()
