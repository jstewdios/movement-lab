"""Run with upstream's existing interpreter; never writes into upstream."""
import argparse
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
import re
from pathlib import Path
import subprocess
import sys
sys.dont_write_bytecode = True

import numpy as np

from movement_lab.measure import summary


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--upstream', type=Path, default=Path('/home/james/microduck_rl'))
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--vx', type=float, default=0)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--control-rate', action='store_true', help='sample at policy rate without extra substep forward calls (instrumentation check)')
    p.add_argument('--disturbance-dv', type=float, default=None, help='00B: instantaneous world-x root velocity increment (m/s); zero is control')
    p.add_argument('--disturbance-time', type=float, default=2.0)
    p.add_argument('--default-position', action='store_true', help='00B: zero action, configured default joint targets through BAM')
    p.add_argument('--record-com', action='store_true', help='00C: record mass-weighted whole-robot center of mass')
    args = p.parse_args()
    if args.disturbance_dv is not None:
        if not np.isfinite([args.disturbance_dv, args.disturbance_time]).all() or not 1 <= args.disturbance_time < args.seconds - 1:
            p.error('00B requires finite disturbance and at least 1 s before and after')
        if args.control_rate:
            p.error('00B requires physics-rate measurement')
    if args.seconds <= 0 or not np.isfinite([args.seconds, args.vx]).all():
        p.error('seconds must be positive; seconds and vx must be finite')
    upstream = args.upstream.resolve()
    out = args.out.resolve()
    if out.is_relative_to(upstream):
        p.error('output must be outside upstream')
    out.mkdir(parents=True, exist_ok=False)
    for source in ('evaluate.py', 'measure.py'):
        (out / source).write_bytes(Path(__file__).with_name(source).read_bytes())
    # All incidental caches and runner outputs remain local.
    os.environ.setdefault('XDG_CACHE_HOME', str(Path.cwd() / '.cache'))
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'matplotlib'))
    os.environ.setdefault('TRITON_CACHE_DIR', str(out / 'cache' / 'triton'))
    import torch
    import mjlab.tasks  # registry plugins
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    task = 'Mjlab-Velocity-Flat-MicroDuck'
    cfg = load_env_cfg(task, play=True)
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.curriculum = {}
    keep = {'expand_bam_friction_fields', 'reset_action_history', 'reset_base', 'reset_robot_joints'}
    removed = sorted(set(cfg.events) - keep)
    cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
    cfg.events['reset_base'].params['pose_range'] = {'x': (0, 0), 'y': (0, 0), 'z': (.125, .125), 'yaw': (0, 0)}
    cfg.events['reset_base'].params['velocity_range'] = {}
    cfg.events['reset_robot_joints'].params.update(position_range=(0, 0), velocity_range=(0, 0))
    for group in cfg.observations.values():
        group.enable_corruption = False
    for act in cfg.scene.entities['robot'].articulation.actuators:
        act.vin_range = (7.35, 7.35)
        act.vin_drop_gain_range = (.1, .1)
        act.delay_min_lag = act.delay_max_lag = 4
    twist = cfg.commands['twist']
    twist.ranges.lin_vel_x = (args.vx, args.vx)
    twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0, 0)
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_standing_envs = 0
    twist.rel_turn_in_place_envs = 0
    if args.disturbance_dv is not None:
        twist.rel_forward_envs = 0  # custom forward bucket otherwise overrides zero ranges
    for name, cmd in cfg.commands.items():
        cmd.resampling_time_range = (1e9, 1e9)
        if name != 'twist':
            cmd.ranges = tuple((0, 0) for _ in cmd.ranges)
    # repr retains callables/config details without unsafe YAML deserialization.
    (out / 'resolved_config.txt').write_text(re.sub(r' at 0x[0-9a-fA-F]+', '', repr(cfg)))
    agent_cfg = load_rl_cfg(task)
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device=args.device)
        checkpoint = args.checkpoint.resolve()
        runner.load(str(checkpoint), load_cfg={'actor': True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        env.reset(seed=args.seed)
        obs = wrapped.get_observations()
        robot = env.scene['robot']
        if args.record_com:
            body_mass = robot.data.model.body_mass[:, robot.data.indexing.body_ids]
        assert robot.body_names[0] == 'trunk_base', robot.body_names
        rows = []
        current_action = None
        control_index = 0
        original_update = env.scene.update
        def array(t):
            return t[0].detach().cpu().numpy().copy()
        def record_update(dt):
            # mj_step leaves derived kinematics one substep stale. Explicitly
            # refresh before invalidating sensor caches and reading a snapshot.
            env.sim.forward()
            env.sim.sense()
            original_update(dt)
            snapshot()
        def snapshot():
            d = robot.data
            env.scene['feet_ground_contact']._invalidate_cache()
            feet = env.scene['feet_ground_contact'].data
            rows.append(dict(time_s=array(env.sim.data.time),
                control_index=np.array(control_index), action=array(current_action),
                joint_target_rad=array(d.joint_pos_target), joint_position_rad=array(d.joint_pos),
                joint_velocity_rad_s=array(d.joint_vel), joint_force_nm=array(d.qfrc_actuator),
                actuator_force=array(d.actuator_force), body_pose_w=array(d.body_link_pose_w),
                body_velocity_w=array(d.body_link_vel_w), foot_found=array(feet.found),
                foot_force_w_n=array(feet.force),
                command=np.concatenate([array(env.command_manager.get_command(n)) for n in ('twist','head_pose','body_pose')]),
                qpos=array(env.sim.data.qpos), qvel=array(env.sim.data.qvel)))
            if args.record_com:
                rows[-1]['com_position_w_m'] = array((d.body_com_pos_w * body_mass[..., None]).sum(dim=1) / body_mass.sum(dim=1)[:, None])
                rows[-1]['com_velocity_w_m_s'] = array((d.body_com_lin_vel_w * body_mass[..., None]).sum(dim=1) / body_mass.sum(dim=1)[:, None])
        if not args.control_rate:
            env.scene.update = record_update
        np.savez_compressed(out / 'initial.npz', qpos=array(env.sim.data.qpos), qvel=array(env.sim.data.qvel))
        disturbance = None
        if args.disturbance_dv is not None:
            from movement_lab.disturbance import install
            disturbance = install(env, robot, args.disturbance_time, args.disturbance_dv)
            (out / 'disturbance.py').write_bytes(Path(__file__).with_name('disturbance.py').read_bytes())
        done = False
        with torch.inference_mode():
            for control_index in range(int(np.ceil(args.seconds/env.step_dt))):
                current_action = torch.zeros_like(policy(obs)) if args.default_position else policy(obs)
                obs, _, dones, _ = wrapped.step(current_action)
                if args.control_rate:
                    snapshot()
                if bool(dones[0]):
                    done = True
                    break
        arrays = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
        stats = summary(arrays)
        np.savez_compressed(out / 'trial.npz', **arrays)
        git = lambda *a: subprocess.check_output(['git','--no-optional-locks','-C',str(upstream),*a], text=True).strip()
        metadata = dict(schema=1, task=task, checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
            upstream=str(upstream), upstream_commit=git('rev-parse','HEAD'), upstream_status=git('status','--porcelain'),
            seed=args.seed, vx=args.vx, requested_seconds=args.seconds, device=args.device,
            sample_mode='control' if args.control_rate else 'physics', physics_dt=env.physics_dt, control_dt=env.step_dt, terminated=bool(env.reset_terminated[0]),
            truncated=bool(env.reset_time_outs[0]), stopped_on_done=done,
            joint_names=robot.joint_names, body_names=robot.body_names,
            foot_order=['left_foot_collision','right_foot_collision'], removed_events=removed,
            python=sys.version, gpu=torch.cuda.get_device_name() if 'cuda' in args.device else None,
            versions={n: importlib.metadata.version(n) for n in ('mjlab','mujoco','mujoco-warp','warp-lang','torch','rsl-rl-lib','better-actuator-models')},
            measurement_code_sha256=sha(out/'evaluate.py'), analysis_code_sha256=sha(out/'measure.py'), config_sha256=sha(out/'resolved_config.txt'),
            summary=stats)
        if disturbance is not None:
            metadata['disturbance'] = disturbance
            metadata['controller'] = 'default-position' if args.default_position else 'policy'
            metadata['disturbance_code_sha256'] = sha(out/'disturbance.py')
        if args.record_com:
            metadata['body_mass_kg'] = array(body_mass).tolist()
            metadata['com_definition'] = 'mass-weighted body inertial centers and linear velocities, all robot bodies'
        for name in ('env.yaml','agent.yaml'):
            f = checkpoint.parent/'params'/name
            if f.exists():
                (out/f'training_{name}').write_bytes(f.read_bytes())
        (out/'metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
        print(json.dumps(stats, indent=2))
    finally:
        env.close()


if __name__ == '__main__':
    main()
