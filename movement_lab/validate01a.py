"""Validate preserved 01A records without CUDA or reintegration."""
import argparse
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha
from movement_lab.discover01a import SEQUENCES, envelope


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path)
    out=p.parse_args().out
    manifest=json.loads((out/'manifest.json').read_text())
    for name, digest in manifest.items():
        assert sha(out/name)==digest, name
    provenance=json.loads((out/'provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items():
        assert sha(out/name)==digest,name
    import mujoco
    model=mujoco.MjModel.from_binary_path(str(out/'replay_model.mjb'))
    native=mujoco.MjData(model)
    results={}
    for f in sorted(out.glob('*/metadata.json')):
        m=json.loads(f.read_text());name=m['sequence']
        names=m['joint_names'];amp=np.array([m['amplitudes_rad'].get(n,0) for n in names])
        assert m['amplitudes_rad']==SEQUENCES['zero' if name=='zero_repeat' else name]
        with np.load(f.parent/'trial.npz') as a, np.load(f.parent/'initial.npz') as ini:
            assert np.allclose(ini['qvel'],0),name
            assert 0 < len(a['time_s']) <= 600, name
            assert len(a['time_s']) == 600 or m['stop_reasons'] or m['terminated'] or m['truncated'], name
            expected=np.array([amp*envelope(int(i)*m['control_dt']) for i in a['control_index']])
            assert np.allclose(a['action'],expected,atol=1e-7),name
            assert np.max(np.abs(np.diff(a['action'],axis=0))) <= .25*np.pi/.6/2*m['control_dt']+1e-6
            assert (np.abs(a['action'][a['time_s']>1.83])<1e-7).all()
            body_ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot/'+n) for n in m['body_names']]
            assert min(body_ids)>=0
            pose_error=com_error=0.0
            mass=np.array(m['mass_kg'])
            for i in range(0,len(a['time_s']),20):
                native.qpos[:]=a['qpos'][i];native.qvel[:]=a['qvel'][i]
                mujoco.mj_forward(model,native)
                pose_error=max(pose_error,float(np.abs(native.xpos[body_ids]-a['body_pose_w'][i,:,:3]).max()))
                com=(native.xipos[body_ids]*mass[:,None]).sum(0)/mass.sum()
                com_error=max(com_error,float(np.abs(com-a['com_position_w_m'][i]).max()))
            assert pose_error<1e-5 and com_error<1e-5,(name,pose_error,com_error)
            results[name]=dict(native_link_position_max_error_m=pose_error,native_com_max_error_m=com_error,
                native_gpu_foot_occupancy_mismatch_fraction=float(np.mean((a['native_foot_contacts']>0)!=(a['foot_found']>0))),
                samples=len(a['time_s']), command_envelope_verified=True)
    assert set(results)==set(SEQUENCES)|{'zero_repeat'},'incomplete pilot'
    result=dict(manifest_hashes=True,source_hashes=True,trials=results)
    (out/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
