"""Full saved-record validation for bounded 01B departures; no reintegration."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from movement_lab.departure01b import SEQUENCES,envelope
from movement_lab.departure_metrics01b import metrics
from movement_lab.evaluate import sha


def main():
    import mujoco
    root=Path('results/01b/departures');results={}
    with np.load(root/'baseline/zero/trial.npz') as z:control=dict(z)
    for p in sorted(root.glob('*/provenance.json')):
        out=p.parent;subprocess.run([sys.executable,'-m','movement_lab.validate01b',str(out)],check=True)
        prov=json.loads(p.read_text())
        for n,digest in prov.get('postprocess_source_sha256',{}).items():assert sha(out/n)==digest
        f=next(out.glob('*/metadata.json'));m=json.loads(f.read_text())
        with np.load(f.parent/'trial.npz') as z,np.load(f.parent/'initial.npz') as ini,np.load(root/'baseline/zero/initial.npz') as ref:
            a=dict(z);assert np.array_equal(ini['qpos'],ref['qpos']) and np.array_equal(ini['qvel'],ref['qvel'])
        limits=np.array(m['joint_limits_rad']);amp=np.array([SEQUENCES[m['sequence']].get(n,0) for n in m['joint_names']])
        expected=np.array([amp*envelope(i*m['control_dt']) for i in a['control_index']])
        assert np.allclose(expected,a['residual_rad'],atol=1e-7)
        assert np.abs(np.diff(a['residual_rad'],axis=0)).max()<=.08*np.pi/1.2*.02+1e-7
        assert (a['residual_rad'][a['time_s']>3.63]==0).all()
        assert (a['joint_target_rad']>limits[:,0]+.02-1e-6).all() and (a['joint_target_rad']<limits[:,1]-.02+1e-6).all()
        prefix=a['control_index']<100
        assert all(np.array_equal(a[k][prefix],v[:len(prefix)][prefix]) for k,v in control.items())
        assert (out/'resolved_config.txt').read_bytes()==(root/'baseline/resolved_config.txt').read_bytes()
        assert sha(out/'replay_model.mjb')==sha(root/'baseline/replay_model.mjb')
        model=mujoco.MjModel.from_binary_path(str(out/'replay_model.mjb'))
        assert (model.actuator_trntype==mujoco.mjtTrn.mjTRN_JOINT).all()
        moment=np.zeros((model.nu,model.nv))
        for i,j in enumerate(model.actuator_trnid[:,0]):moment[i,model.jnt_dofadr[j]]=model.actuator_gear[i,0]
        joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in m['joint_names']]
        assert np.allclose(a['actuator_force']@moment[:,model.jnt_dofadr[joints]],a['joint_force_nm'],atol=1e-6)
        assert abs(a['joint_force_nm']).max()<=.9601
        assert not m['terminated'] and not m['truncated'] and not m['stop_reasons']
        result=metrics(a,control)
        assert result['physical_gate']['pass'] and result['recovered']
        results[out.name]=dict(samples=len(a['time_s']),matched_prefix_exact=True,envelope=True,
            actuator_transmission_mapping=True,composed_target_limits=True,recovery_recomputed=True)
    (root/'independent-validation.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
