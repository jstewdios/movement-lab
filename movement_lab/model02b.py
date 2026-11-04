"""02B: compile the upstream Microduck model once through mjlab and save it.

The compiled MjModel is produced by the *installed upstream* task/robot code,
exactly as 01A-02A configured it (velocity flat task, curated groundcontact
robot, BAM XL330 M6 spec edits, 7.35 V / 0.1 V/Nm, stiff friction, solver
options). No upstream file is modified. The saved MJB embeds upstream meshes, so
it stays local (ignored by Git); its hash and every checked property are
recorded in ``model.json`` so later rollouts can verify they use the same model.

    python -m movement_lab.model02b --out results/02b/model

On a fresh checkout ``results/02b/model`` already holds the published
``model.json`` and ``resolved_config.txt`` but not the MJB. The command then
compiles into a temporary directory and installs ``model.mjb`` only if its hash
equals the published one; the published files are never overwritten.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.dont_write_bytecode = True

JOINTS = ('left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
          'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
          'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle')
# Upstream HOME_FRAME (microduck_constants.py); verified against the compiled
# action offset below rather than trusted.
HOME = (0.0, -0.0873, -0.4579, -0.0049, 0.4530, 0.3491, 0.3491, 0.0, 0.0,
        0.0, 0.0873, 0.4579, 0.0049, -0.4530)
NOMINAL_VIN, NOMINAL_DROP_GAIN, NOMINAL_LAG = 7.35, 0.1, 4


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(upstream, *args):
    return subprocess.run(['git', '-C', str(upstream), *args], capture_output=True, text=True).stdout.strip()


TASK = 'Mjlab-Velocity-Flat-MicroDuck'
CLEARANCE_ROOT_Z = 0.1181823627  # 01B geometry-derived 1 mm sole clearance at HOME


def env_cfg(num_envs=1, seed=42):
    """The 01A-02A environment configuration (nominal BAM, no DR/pushes/curricula)."""
    import mjlab.tasks  # noqa: F401  (registers tasks)
    from copy import deepcopy
    from mjlab.tasks.registry import load_env_cfg
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
    cfg = load_env_cfg(TASK, play=True)
    cfg.scene.entities['robot'] = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)  # curated groundcontact
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.auto_reset = False
    cfg.curriculum = {}
    keep = {'expand_bam_friction_fields', 'reset_action_history', 'reset_base', 'reset_robot_joints'}
    cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
    cfg.events['reset_base'].params['pose_range'] = {'x': (0, 0), 'y': (0, 0), 'z': (CLEARANCE_ROOT_Z, CLEARANCE_ROOT_Z), 'yaw': (0, 0)}
    cfg.events['reset_base'].params['velocity_range'] = {}
    cfg.events['reset_robot_joints'].params.update(position_range=(0, 0), velocity_range=(0, 0))
    for group in cfg.observations.values():
        group.enable_corruption = False
    for act in cfg.scene.entities['robot'].articulation.actuators:
        act.vin_range = (NOMINAL_VIN, NOMINAL_VIN)
        act.vin_drop_gain_range = (NOMINAL_DROP_GAIN, NOMINAL_DROP_GAIN)
        act.delay_min_lag = act.delay_max_lag = NOMINAL_LAG
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in {'nan_state', 'time_out'}}
    return cfg


# Provenance of one compile (machine, interpreter, timing). Everything else in
# model.json describes the model itself and must match on regeneration.
PROVENANCE = ('wall_s', 'python', 'upstream', 'packages', 'device_for_compile')


def compile_model(out, upstream, device='cpu'):
    """Compile through mjlab into ``out``: model.mjb, model.json, resolved_config.txt."""
    out = Path(out)
    os.environ.setdefault('WANDB_MODE', 'disabled')
    started = time.perf_counter()
    import mujoco
    import numpy as np
    from mjlab.envs import ManagerBasedRlEnv

    task = TASK
    cfg = env_cfg()
    # LF on every platform: these files are published and hashed byte-for-byte.
    (out / 'resolved_config.txt').write_text(re.sub(r' at 0x[0-9a-fA-F]+', '', repr(cfg)), newline='\n')
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
        m = env.sim.mj_model
        term = env.action_manager.get_term('joint_pos')
        names = tuple(term.target_names)
        offset = term.offset[0].detach().cpu().numpy().astype(float)
        assert names == JOINTS, names
        assert term.scale == 1.0 and term.cfg.use_default_offset and term.cfg.clip is None
        assert np.allclose(offset, HOME, atol=1e-6), offset
        act_names = [m.actuator(i).name for i in range(m.nu)]
        assert act_names == ['robot/' + n for n in JOINTS], act_names
        joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + n) for n in JOINTS]
        assert min(joint_ids) >= 0
        assert list(m.actuator_trnid[:, 0]) == joint_ids
        assert (m.actuator_biastype == 0).all() and (m.actuator_gaintype == 0).all()
        dofs = m.jnt_dofadr[joint_ids]
        assert np.allclose(m.dof_frictionloss[dofs], 0) and np.allclose(m.dof_damping[dofs], 0)
        mjb = out / 'model.mjb'
        mujoco.mj_saveModel(m, str(mjb), None)
        geoms = []
        for i in range(m.ngeom):
            if m.geom_contype[i] or m.geom_conaffinity[i]:
                geoms.append(dict(id=i, name=m.geom(i).name, body=m.body(m.geom_bodyid[i]).name,
                                  condim=int(m.geom_condim[i]), priority=int(m.geom_priority[i]),
                                  friction=m.geom_friction[i].tolist()))
        upstream = Path(upstream).resolve()
        pkgs = {k: importlib.metadata.version(k) for k in ('mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'torch', 'better-actuator-models', 'numpy')}
        info = dict(
            task=task, robot='MICRODUCK_STANDUP_ROBOT_CFG (curated groundcontact)', device_for_compile=device,
            mjb_sha256=sha(mjb), joints=list(JOINTS), home=list(HOME), action_offset=offset.tolist(),
            joint_range=m.jnt_range[joint_ids].tolist(), forcerange=m.actuator_forcerange.tolist(),
            armature=m.dof_armature[dofs].tolist(), dof_solref=m.dof_solref[dofs].tolist(),
            opt=dict(timestep=m.opt.timestep, iterations=int(m.opt.iterations), ls_iterations=int(m.opt.ls_iterations),
                     solver=int(m.opt.solver), cone=int(m.opt.cone), integrator=int(m.opt.integrator), impratio=m.opt.impratio),
            decimation=int(cfg.decimation), total_mass=float(m.body_mass.sum()), collision_geoms=geoms,
            nominal=dict(vin=NOMINAL_VIN, vin_drop_gain=NOMINAL_DROP_GAIN, vin_min=6.0, kp_fw=200.0, lag_physics_steps=NOMINAL_LAG),
            upstream=dict(path=str(upstream), revision=git(upstream, 'rev-parse', 'HEAD'),
                          dirty=bool(git(upstream, 'status', '--porcelain')),
                          robot_xml_sha256=sha(upstream / 'src/mjlab_microduck/robot/microduck/robot_groundcontact.xml'),
                          constants_sha256=sha(upstream / 'src/mjlab_microduck/robot/microduck_constants.py')),
            packages=pkgs, python=sys.version, wall_s=time.perf_counter() - started)
        (out / 'model.json').write_text(json.dumps(info, indent=1), newline='\n')
        return info
    finally:
        env.close()


def model_differences(saved, fresh):
    """Fields describing the model (not its provenance) that differ."""
    keys = (set(saved) | set(fresh)) - set(PROVENANCE)
    canon = lambda v: json.loads(json.dumps(v))
    return sorted(k for k in keys if canon(saved.get(k)) != canon(fresh.get(k)))


def regenerate(out, compile_fn):
    """Restore the local MJB next to published metadata; never overwrite it.

    Returns the fresh info. Raises SystemExit, installing nothing, if the
    compiled model differs from the published one."""
    saved = json.loads((out / 'model.json').read_text())
    mjb = out / 'model.mjb'
    if mjb.exists():
        if sha(mjb) != saved['mjb_sha256']:
            raise SystemExit(f'{mjb} exists but does not match the published hash; move it away to regenerate')
        print(f'{mjb} already present and matches the published hash {saved["mjb_sha256"]}')
        return saved
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        fresh = compile_fn(tmp)
        diffs = model_differences(saved, fresh)
        if diffs:
            raise SystemExit(f'compiled model differs from the published one in {diffs} '
                             f'(published mjb {saved["mjb_sha256"]}, compiled {fresh["mjb_sha256"]}); nothing installed')
        if (tmp / 'resolved_config.txt').read_bytes() != (out / 'resolved_config.txt').read_bytes():
            print('WARNING: resolved_config.txt differs from the published copy. The compiled model is identical, '
                  'but the mjlab env settings (used by crosscheck02b) are not')
        notes = [k for k in PROVENANCE if json.dumps(saved.get(k)) != json.dumps(fresh.get(k)) and k != 'wall_s']
        shutil.move(str(tmp / 'model.mjb'), str(mjb))
    print(f'installed {mjb}: hash matches the published {saved["mjb_sha256"]}'
          + (f'; provenance differs only in {notes}' if notes else ''))
    return fresh


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--upstream', type=Path, default=Path(os.environ.get('MICRODUCK_RL', Path.home() / 'microduck_rl')))
    p.add_argument('--device', default='cpu', help='mjlab device used only to construct/compile the env')
    args = p.parse_args(argv)
    out = args.out.resolve()
    compile_fn = lambda d: compile_model(d, args.upstream, args.device)
    if (out / 'model.json').exists():
        info = regenerate(out, compile_fn)
    else:
        out.mkdir(parents=True, exist_ok=False)
        info = compile_fn(out)
    print(json.dumps({k: info[k] for k in ('mjb_sha256', 'opt', 'total_mass', 'wall_s')}, indent=1))


if __name__ == '__main__':
    main()
