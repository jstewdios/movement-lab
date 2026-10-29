"""Offline landing audit. CPU forward geometry only; never integrates dynamics."""
import csv
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha


def foot_clearances(model, data):
    """Lowest transformed collision mesh vertex relative to world z=0 plane."""
    import mujoco
    result={}
    for name in ('left_foot_collision','right_foot_collision'):
        g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+name)
        assert g>=0 and model.geom_type[g]==mujoco.mjtGeom.mjGEOM_MESH
        mesh=model.geom_dataid[g];start=model.mesh_vertadr[mesh];count=model.mesh_vertnum[mesh]
        vertices=model.mesh_vert[start:start+count]
        world=vertices@data.geom_xmat[g].reshape(3,3).T+data.geom_xpos[g]
        result[name]=float(world[:,2].min())
    return result


def audit_record(path, model, out):
    import mujoco
    path=Path(path);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    a=np.load(path/'trial.npz');ini=np.load(path/'initial.npz');meta=json.loads((path/'metadata.json').read_text())
    d=mujoco.MjData(model);d.qpos[:]=ini['qpos'];d.qvel[:]=ini['qvel'];mujoco.mj_forward(model,d)
    initial=dict(root_qpos=ini['qpos'][:7].tolist(),root_qvel=ini['qvel'][:6].tolist(),sole_clearance_m=foot_clearances(model,d))
    events=[];worst=0.;ncontact=0
    for i,t in enumerate(a['time_s']):
        d.qpos[:]=a['qpos'][i];d.qvel[:]=a['qvel'][i];mujoco.mj_forward(model,d)
        ncontact=max(ncontact,d.ncon)
        for c in d.contact:
            worst=min(worst,float(c.dist))
            if t<=.10001:
                events.append(dict(time_s=float(t),geom_ids=[int(c.geom1),int(c.geom2)],
                    geom_names=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,int(g)) for g in (c.geom1,c.geom2)],
                    position_w_m=c.pos.tolist(),normal_geom1_to_geom2=c.frame[:3].tolist(),distance_m=float(c.dist),
                    solref=c.solref.tolist(),solimp=c.solimp.tolist(),friction=c.friction.tolist(),dim=int(c.dim)))
    names=meta['joint_names']
    header=['time_s','root_z_m','root_vz_m_s']
    arrays=[a['time_s'][:,None],a['qpos'][:,2,None],a['qvel'][:,2,None]]
    for field in ('action','joint_target_rad','joint_position_rad','joint_velocity_rad_s','joint_force_nm'):
        header += [field+':'+n for n in names];arrays.append(a[field])
    data=np.concatenate(arrays,axis=1);mask=a['time_s']<=.10001
    with (out/'first100ms.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(header);writer.writerows(data[mask])
    result=dict(initial=initial,samples=len(a['time_s']),duration_s=float(a['time_s'][-1]),
        first100ms_observed_through_s=float(a['time_s'][mask][-1]),minimum_contact_distance_m=worst,
        maximum_native_contacts=ncontact,contacts_first100ms=events,
        first100ms_joint_range_rad=np.ptp(a['joint_position_rad'][mask],axis=0).tolist(),joint_names=names,
        source_hashes={n:sha(path/n) for n in ('trial.npz','initial.npz','metadata.json')})
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    import mujoco
    root=Path('results/01b/landing-diagnosis')
    m=mujoco.MjModel.from_binary_path('results/01b/gpu-gate/replay_model.mjb')
    results={}
    for name,path in [('original','results/01b/gpu-gate/zero'),('historical00a','results/99-balance')]:
        results[name]=audit_record(path,m,root/name)
    with np.load('results/01b/gpu-gate/zero/trial.npz') as a,np.load('results/99-balance/trial.npz') as b:
        common=('time_s','action','joint_target_rad','joint_position_rad','joint_velocity_rad_s','joint_force_nm','qpos','qvel','foot_found','foot_force_w_n')
        results['historical_prefix_exact']={k:bool(np.array_equal(a[k],b[k][:len(a[k])])) for k in common}
    cfg=[]
    for p in ('results/99-balance/resolved_config.txt','results/01b/gpu-gate/resolved_config.txt'):
        text=Path(p).read_text();cfg.append(text.split('sim=SimulationCfg(',1)[1].split(', viewer=',1)[0])
    results['simulation_config_exact']=cfg[0]==cfg[1];results['simulation_config']=cfg[0]
    results['options']={k:float(getattr(m.opt,k)) for k in ('timestep','iterations','tolerance','ls_iterations','ls_tolerance','solver','integrator','cone','impratio','ccd_iterations','ccd_tolerance')}
    results['options']['gravity']=m.opt.gravity.tolist()
    for path in sorted(root.glob('clearance-*')):
        if not path.is_dir() or not (path/'zero/trial.npz').exists(): continue
        model=mujoco.MjModel.from_binary_path(str(path/'replay_model.mjb'))
        results[path.name]=audit_record(path/'zero',model,root/(path.name+'-audit'))
        results[path.name]['gate']=json.loads((path/'zero/metadata.json').read_text())['gate']
    compare(root, m)
    (root/'diagnosis.json').write_text(json.dumps(results,indent=2)+'\n')


def compare(root, model):
    import mujoco
    walk=mujoco.MjModel.from_xml_path('/home/james/microduck_rl/src/mjlab_microduck/robot/microduck/robot_walk.xml')
    result={}
    for name in ('left_foot_collision','right_foot_collision'):
        g=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'robot/'+name)
        w=mujoco.mj_name2id(walk,mujoco.mjtObj.mjOBJ_GEOM,name)
        fields=('geom_pos','geom_quat','geom_solref','geom_solimp','geom_friction','geom_margin','geom_gap','geom_condim','geom_contype','geom_conaffinity')
        result[name]={k:bool(np.array_equal(getattr(model,k)[g],getattr(walk,k)[w])) for k in fields}
        def verts(m,i):
            k=m.geom_dataid[i];start=m.mesh_vertadr[k]
            return m.mesh_vert[start:start+m.mesh_vertnum[k]]
        result[name]['mesh_vertices_exact']=bool(np.array_equal(verts(model,g),verts(walk,w)))
        assert all(result[name].values())
    result['conditions']={}
    for p in root.glob('clearance-*/zero/metadata.json'):
        meta=json.loads(p.read_text())
        with np.load(p.parent/'trial.npz') as a:
            t=a['time_s'];s=t>=1.5;q=a['body_pose_w'][:,0,3:7]
            tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
            result['conditions'][p.parent.parent.name]=dict(gate=meta['gate'],duration_s=float(t[-1]),
                min_distance_m=float(a['native_min_contact_distance_m'].min()),max_tilt_deg=float(tilt.max()),
                settled_max_com_speed_m_s=float(np.linalg.norm(a['com_velocity_w_m_s'][s],axis=1).max()),
                settled_max_angular_speed_rad_s=float(np.linalg.norm(a['body_velocity_w'][s,0,3:],axis=1).max()),
                settled_com_range_xyz_m=np.ptp(a['com_position_w_m'][s],axis=0).tolist(),
                settled_mean_foot_fz_n=float(a['foot_force_w_n'][s,:,2].sum(1).mean()),
                settled_contact_fraction=(a['foot_found'][s]>0).mean(0).tolist(),
                nonfoot_samples=int((a['native_nonfoot_contacts']>0).sum()),peak_effort_nm=float(abs(a['joint_force_nm']).max()),
                peak_tracking_rad=float(abs(a['joint_target_rad']-a['joint_position_rad']).max()),
                runner_wall_s=json.loads((p.parent.parent/'provenance.json').read_text())['wall_seconds'])
    paths=[root/'clearance-1mm',root/'clearance-1mm-repeat']
    if all((p/'zero/trial.npz').exists() for p in paths):
        with np.load(paths[0]/'zero/trial.npz') as a,np.load(paths[1]/'zero/trial.npz') as b:
            result['repeat_all_arrays_exact']=all(np.array_equal(a[k],b[k]) for k in a.files)
            assert result['repeat_all_arrays_exact']
        with np.load('results/01b/gpu-gate/zero/initial.npz') as original:
            for p in paths:
                with np.load(p/'zero/initial.npz') as ini:
                    diff=ini['qpos']-original['qpos'];diff[2]=0
                    assert np.array_equal(diff,np.zeros_like(diff))
                    assert np.array_equal(ini['qvel'],original['qvel'])
        result['only_initial_root_z_changed']=True
    (root/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
