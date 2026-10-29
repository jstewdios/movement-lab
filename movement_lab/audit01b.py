"""Read-only audit of historical configurations and current upstream XML models."""
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha


def audit(out):
    import mujoco
    base=Path('/home/james/microduck_rl/src/mjlab_microduck/robot/microduck')
    models={k:mujoco.MjModel.from_xml_path(str(base/f'robot_{k}.xml')) for k in ('walk','groundcontact')}
    a,b=models.values()
    common={}
    for field in ('body_mass','body_inertia','body_ipos','body_iquat','body_pos','body_quat','jnt_type','jnt_pos','jnt_axis','jnt_range','dof_damping','dof_armature','actuator_gear'):
        common[field]=bool(np.array_equal(getattr(a,field),getattr(b,field)))
    ini=np.load('results/01a/atlas-v2/zero/initial.npz')
    earlier=np.load('results/99-balance/initial.npz')
    result=dict(xml_hashes={k:sha(base/f'robot_{k}.xml') for k in models},
        identical_physical_arrays=common,initial_qpos_max_difference=float(np.abs(ini['qpos']-earlier['qpos']).max()),
        initial_qvel_max_difference=float(np.abs(ini['qvel']-earlier['qvel']).max()), models={})
    for name,m in models.items():
        coll=[]
        for i in range(m.ngeom):
            if m.geom_contype[i] or m.geom_conaffinity[i]:
                coll.append(dict(name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,i),body=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])),mesh=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_MESH,int(m.geom_dataid[i])) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_MESH else None,contype=int(m.geom_contype[i]),conaffinity=int(m.geom_conaffinity[i]),condim=int(m.geom_condim[i]),friction=m.geom_friction[i].tolist()))
        result['models'][name]=dict(nq=m.nq,nv=m.nv,nu=m.nu,mass=float(m.body_mass.sum()),collision_geoms=coll)
    # Validate 01A replay and COM without writing into its preserved artifacts.
    m=mujoco.MjModel.from_binary_path('results/01a/atlas-v2/replay_model.mjb'); d=mujoco.MjData(m)
    meta=json.loads(Path('results/01a/atlas-v2/zero/metadata.json').read_text())
    ids=[mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,'robot/'+n) for n in meta['body_names']]
    errors=[]
    with np.load('results/01a/atlas-v2/zero/trial.npz') as z:
        for i in range(0,len(z['time_s']),20):
            d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];mujoco.mj_forward(m,d)
            errors.append(float(np.abs(d.xpos[ids]-z['body_pose_w'][i,:,:3]).max()))
    result['historical_replay_max_link_error_m']=max(errors)
    Path(out).write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    import sys
    audit(sys.argv[1])
