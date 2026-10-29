"""Validate 01B saved evidence without integrating dynamics or changing records."""
import argparse
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha
from movement_lab.controlled01b import gate
from movement_lab.measure import validate


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('out',type=Path);out=p.parse_args().out
    for name,digest in json.loads((out/'manifest.json').read_text()).items():
        assert sha(out/name)==digest,name
    provenance=json.loads((out/'provenance.json').read_text())
    for name,digest in provenance['source_sha256'].items(): assert sha(out/name)==digest,name
    assert sha(provenance['checkpoint'])==provenance['checkpoint_sha256']
    import mujoco
    model=mujoco.MjModel.from_binary_path(str(out/'replay_model.mjb'));d=mujoco.MjData(model)
    for f in out.glob('*/metadata.json'):
        m=json.loads(f.read_text())
        with np.load(f.parent/'trial.npz') as a,np.load(f.parent/'initial.npz') as ini:
            validate(a)
            assert (ini['qvel']==0).all()
            assert np.allclose(a['action'],a['actor_action']+a['residual_rad'],atol=1e-7)
            assert np.allclose(a['joint_target_rad'],a['action']+m['target_offset_rad'],atol=1e-6)
            assert gate(a,m['stop_reasons'],m['sequence'] in ('zero','zero_repeat'))==m['gate']
            ids=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot/'+n) for n in m['body_names']]
            joints=[mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_JOINT,'robot/'+n) for n in m['joint_names']]
            assert min(ids+joints)>=0
            assert np.allclose(a['qpos'][:,model.jnt_qposadr[joints]],a['joint_position_rad'],atol=1e-6)
            mass=np.array(m['mass_kg'])
            for i in range(len(a['time_s'])):
                d.qpos[:]=a['qpos'][i];d.qvel[:]=a['qvel'][i];mujoco.mj_forward(model,d)
                assert np.abs(d.xpos[ids]-a['body_pose_w'][i,:,:3]).max()<1e-5
                assert np.abs((d.xipos[ids]*mass[:,None]).sum(0)/mass.sum()-a['com_position_w_m'][i]).max()<1e-5
                assert abs(min([0.]+[float(c.dist) for c in d.contact])-a['native_min_contact_distance_m'][i])<1e-6
            assert np.array_equal(a['native_foot_contacts']>0,a['foot_found']>0)
    print('PASS: hashes, checkpoint, arrays, controller composition, mapping, gate, saved-state geometry/COM/contact audit')


if __name__=='__main__': main()
