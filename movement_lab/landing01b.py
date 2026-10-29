"""01B landing diagnosis: baseline only; explicit independent root-height condition."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.dont_write_bytecode = True
import numpy as np
from movement_lab.evaluate import sha
from movement_lab.discovery_metrics import analyze
from movement_lab.controlled01b import gate, publish

SEQUENCES = {'zero': {}, 'zero_repeat': {}}
for name, values in {
    'head': {'head_yaw': .08},
    'flex': {'left_hip_pitch': -.03, 'right_hip_pitch': .03, 'left_knee': .05, 'right_knee': -.05, 'left_ankle': -.02, 'right_ankle': .02},
    'shift': {'left_hip_roll': .025, 'right_hip_roll': .025},
    'left': {'left_hip_pitch': -.02, 'left_knee': .035, 'left_ankle': -.015},
}.items():
    for sign in (-1, 1):
        SEQUENCES[f'{name}_{sign:+d}'] = {k: sign*v for k,v in values.items()}


def envelope(t):
    # Two seconds of feedback settling; rise .6, hold .4, return .6; recover 2.4 s.
    t -= 2
    if t < 0 or t >= 1.6: return 0.0
    if t < .6: return .5 - .5*np.cos(np.pi*t/.6)
    if t <= 1: return 1.0
    return .5 + .5*np.cos(np.pi*(t-1)/.6)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--upstream', type=Path, default=Path('/home/james/microduck_rl'))
    p.add_argument('--root-height', required=True, type=float, help='reset-only root z offset; no mid-trial state writes')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--checkpoint', type=Path, default=Path('/home/james/microduck_rl/logs/rsl_rl/velocity/2026-10-25_20-16-28_movement-lab-001/model_99.pt'))
    args = p.parse_args()
    if not np.isfinite(args.root_height) or not .117 <= args.root_height <= .125:
        p.error('diagnostic reset height must be within [0.117, 0.125] m')
    upstream, out = args.upstream.resolve(), args.out.resolve()
    if out.is_relative_to(upstream):
        p.error('output must be outside upstream')
    out.mkdir(parents=True, exist_ok=False)
    for name in ('landing01b.py', 'controlled01b.py', 'discovery_metrics.py', 'replay01a.py'):
        (out / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    os.environ.setdefault('XDG_CACHE_HOME', str(Path.cwd() / '.cache'))
    os.environ.setdefault('WANDB_MODE', 'disabled')
    os.environ.setdefault('MPLCONFIGDIR', str(out / 'matplotlib'))
    os.environ.setdefault('TRITON_CACHE_DIR', str(Path.cwd() / '.cache/triton'))
    started = time.perf_counter()
    import torch
    import mujoco
    import mjlab.tasks
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from dataclasses import asdict
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    task = 'Mjlab-Velocity-Flat-MicroDuck'
    cfg = load_env_cfg(task, play=True)
    from copy import deepcopy
    from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
    cfg.scene.entities['robot'] = deepcopy(MICRODUCK_STANDUP_ROBOT_CFG)
    cfg.seed = args.seed
    cfg.scene.num_envs = 1
    cfg.auto_reset = False
    cfg.curriculum = {}
    keep = {'expand_bam_friction_fields', 'reset_action_history', 'reset_base', 'reset_robot_joints'}
    removed = sorted(set(cfg.events) - keep)
    cfg.events = {k: v for k, v in cfg.events.items() if k in keep}
    cfg.events['reset_base'].params['pose_range'] = {'x': (0, 0), 'y': (0, 0), 'z': (args.root_height, args.root_height), 'yaw': (0, 0)}
    cfg.events['reset_base'].params['velocity_range'] = {}
    cfg.events['reset_robot_joints'].params.update(position_range=(0, 0), velocity_range=(0, 0))
    for group in cfg.observations.values():
        group.enable_corruption = False
    for act in cfg.scene.entities['robot'].articulation.actuators:
        act.vin_range = (7.35, 7.35)
        act.vin_drop_gain_range = (.1, .1)
        act.delay_min_lag = act.delay_max_lag = 4
    twist = cfg.commands['twist']
    twist.ranges.lin_vel_x = (0.0, 0.0)
    twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0, 0)
    twist.heading_command = False
    twist.ranges.heading = None
    twist.rel_standing_envs = 0
    twist.rel_turn_in_place_envs = 0
    twist.rel_forward_envs = 0
    for name, cmd in cfg.commands.items():
        cmd.resampling_time_range = (1e9, 1e9)
        if name != 'twist':
            cmd.ranges = tuple((0, 0) for _ in cmd.ranges)
    cfg.terminations = {k: v for k, v in cfg.terminations.items() if k in {'nan_state', 'time_out'}}
    # repr retains callables/config details without unsafe YAML deserialization.
    (out / 'resolved_config.txt').write_text(re.sub(r' at 0x[0-9a-fA-F]+', '', repr(cfg)))
    env = ManagerBasedRlEnv(cfg, device=args.device)
    try:
        agent_cfg = load_rl_cfg(task)
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner = (load_runner_cls(task) or MjlabOnPolicyRunner)(wrapped, asdict(agent_cfg), device=args.device)
        runner.load(str(args.checkpoint.resolve()), load_cfg={'actor': True}, strict=True, map_location=args.device)
        policy = runner.get_inference_policy(device=args.device)
        robot = env.scene['robot']
        term = env.action_manager.get_term('joint_pos')
        assert term.scale == 1.0 and term.cfg.use_default_offset and term.cfg.clip is None
        assert len(term.target_names) == 14
        joint_ids = term.target_ids
        joint_names = list(term.target_names)
        model = env.sim.mj_model
        native = mujoco.MjData(model)
        terrain_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
        assert terrain_id >= 0
        # Geometry-only CPU audit: never integrates or substitutes for GPU dynamics.
        mujoco.mj_saveModel(model, str(out / 'replay_model.mjb'), None)
        mass = robot.data.model.body_mass[:, robot.data.indexing.body_ids]
        def array(t):
            return t[0].detach().cpu().numpy().copy()
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/' + n) for n in joint_names]
        if min(ids) < 0:
            ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        assert min(ids) >= 0
        limits = model.jnt_range[ids].copy()
        offset = array(term.offset)
        plans = ['zero']  # Never schedule departures in this diagnostic runner.
        original_update = env.scene.update
        for name in plans:
            trial_start = time.perf_counter()
            trial = out / name
            trial.mkdir()
            env.scene.update = original_update
            # BAM lazily allocates delay buffers during inference-mode stepping.
            # Reset in the same mode so subsequent trials may clear those tensors.
            with torch.inference_mode():
                env.reset(seed=args.seed)
                obs = wrapped.get_observations()
                if hasattr(policy, 'reset'): policy.reset()
            env.sim.forward()
            env.sim.sense()
            original_update(0)
            initial = dict(qpos=array(env.sim.data.qpos), qvel=array(env.sim.data.qvel),
                           com_position_w_m=array((robot.data.body_com_pos_w * mass[..., None]).sum(1) / mass.sum(1)[:, None]),
                           body_pose_w=array(robot.data.body_link_pose_w), joint_position_rad=array(robot.data.joint_pos[:, joint_ids]))
            np.savez_compressed(trial / 'initial.npz', **initial)
            amplitudes = SEQUENCES['zero' if name == 'zero_repeat' else name]
            amp = np.array([amplitudes.get(n, 0.0) for n in term.target_names])
            assert np.max(np.abs(amp)) <= .25
            assert np.all(np.minimum(offset, offset + amp) > limits[:, 0] + .02)
            assert np.all(np.maximum(offset, offset + amp) < limits[:, 1] - .02)
            rows = []
            stop = []
            actor_action = torch.zeros((1, term.action_dim), device=args.device)
            residual = np.zeros(term.action_dim)
            current_action = torch.zeros((1, term.action_dim), device=args.device)
            control_index = 0
            def snapshot(dt):
                env.sim.forward()
                env.sim.sense()
                original_update(dt)
                d = robot.data
                env.scene['feet_ground_contact']._invalidate_cache()
                feet = env.scene['feet_ground_contact'].data
                row = dict(time_s=array(env.sim.data.time), control_index=np.array(control_index),
                    action=array(current_action), actor_action=array(actor_action), residual_rad=residual.copy(), joint_target_rad=array(d.joint_pos_target[:, joint_ids]),
                    joint_position_rad=array(d.joint_pos[:, joint_ids]), joint_velocity_rad_s=array(d.joint_vel[:, joint_ids]),
                    joint_force_nm=array(d.qfrc_actuator[:, joint_ids]), actuator_force=array(d.actuator_force),
                    body_pose_w=array(d.body_link_pose_w), body_velocity_w=array(d.body_link_vel_w),
                    foot_found=array(feet.found), foot_force_w_n=array(feet.force),
                    com_position_w_m=array((d.body_com_pos_w * mass[..., None]).sum(1) / mass.sum(1)[:, None]),
                    com_velocity_w_m_s=array((d.body_com_lin_vel_w * mass[..., None]).sum(1) / mass.sum(1)[:, None]),
                    qpos=array(env.sim.data.qpos), qvel=array(env.sim.data.qvel))
                finite = all(np.isfinite(v).all() for v in row.values())
                ground, nonfoot, min_dist = 0, 0, 0.0
                native_feet = np.zeros(2, dtype=int)
                if finite:
                    native.qpos[:] = row['qpos']
                    native.qvel[:] = row['qvel']
                    mujoco.mj_forward(model, native)
                    for c in native.contact:
                        min_dist = min(min_dist, float(c.dist))
                        # Explicit terrain geom: mjlab attaches it to a fixed child body.
                        if terrain_id in (c.geom1, c.geom2):
                            ground += 1
                            g = c.geom2 if c.geom1 == terrain_id else c.geom1
                            gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ''
                            nonfoot += int(not any(f in gn for f in ('left_foot_collision', 'right_foot_collision')))
                            for fi, foot in enumerate(('left_foot_collision', 'right_foot_collision')):
                                native_feet[fi] += int(foot in gn)
                row.update(native_foot_contacts=native_feet, native_ground_contacts=np.array(ground), native_nonfoot_contacts=np.array(nonfoot),
                           native_min_contact_distance_m=np.array(min_dist))
                rows.append(row)
                if not finite: stop.append('nonfinite')
                if min_dist < -.003: stop.append('penetration_over_3mm')
                if nonfoot: stop.append('nonfoot_contact')
                if finite and np.max(np.abs(row['qvel'])) > 100: stop.append('generalized_speed_over_100')
                if finite and np.max(np.maximum(limits[:, 0] - row['joint_position_rad'], row['joint_position_rad'] - limits[:, 1])) > .03:
                    stop.append('joint_limit_overshoot_over_0.03rad')
                if finite and np.linalg.norm(row['foot_force_w_n'], axis=-1).max() > 100:
                    stop.append('foot_force_over_100N')
            env.scene.update = snapshot
            with torch.inference_mode():
                for control_index in range(round(6.0 / env.step_dt)):
                    residual = amp * envelope(control_index * env.step_dt)
                    actor_action = policy(obs)
                    current_action = actor_action + torch.tensor(residual[None], dtype=torch.float32, device=args.device)
                    # No wrapper clipping may silently alter the composed command.
                    clip = agent_cfg.clip_actions
                    assert clip is None or float(current_action.abs().max()) <= clip
                    targets = offset + array(current_action)
                    if np.any(targets <= limits[:,0]+.02) or np.any(targets >= limits[:,1]-.02):
                        stop.append('composed_target_limit'); break
                    obs, _, _, _ = wrapped.step(current_action)
                    if stop or bool(env.reset_terminated[0]) or bool(env.reset_time_outs[0]):
                        break
            arrays = {k: np.stack([r[k] for r in rows]) for k in rows[0]}
            np.savez_compressed(trial / 'trial.npz', **arrays)
            metrics = analyze(arrays, initial, limits)
            metadata = dict(initialization_root_z_m=args.root_height, condition='independent landing initialization; zero residual', sequence=name, seed=args.seed, amplitudes_rad=amplitudes,
                physics_dt=env.physics_dt, control_dt=env.step_dt, requested_seconds=6,
                stop_reasons=sorted(set(stop)), terminated=bool(env.reset_terminated[0]),
                truncated=bool(env.reset_time_outs[0]), joint_names=joint_names, all_joint_names=robot.joint_names,
                body_names=robot.body_names, target_offset_rad=offset.tolist(), joint_limits_rad=limits.tolist(),
                mass_kg=array(mass).tolist(), summary=metrics, wall_seconds=time.perf_counter()-trial_start)
            (trial / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            decision = gate(arrays, stop, baseline=name in ('zero','zero_repeat'))
            metadata['gate'] = decision
            (trial / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
            print(name, json.dumps(decision), flush=True)
            if not decision['pass']:
                break
        env.scene.update = original_update
        git = lambda *a: subprocess.check_output(['git', '--no-optional-locks', '-C', str(upstream), *a], text=True).strip()
        assets = upstream / 'src/mjlab_microduck'
        files = sorted([*assets.rglob('*.py'), *assets.joinpath('robot/microduck').rglob('*.xml'), *assets.joinpath('robot/microduck').rglob('*.stl')])
        provenance = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=sha(args.checkpoint), schema=1, upstream=str(upstream), upstream_commit=git('rev-parse', 'HEAD'),
            upstream_status=git('status', '--porcelain'), task=task, model='curated groundcontact (not walk)',
            python=sys.version, gpu=torch.cuda.get_device_name() if 'cuda' in args.device else None,
            versions={n: importlib.metadata.version(n) for n in ('mjlab','mujoco','mujoco-warp','warp-lang','torch','better-actuator-models')},
            upstream_file_sha256={str(f.relative_to(upstream)): sha(f) for f in files},
            source_sha256={n: sha(out / n) for n in ('landing01b.py', 'controlled01b.py', 'discovery_metrics.py', 'resolved_config.txt', 'replay01a.py', 'replay_model.mjb')},
            wall_seconds=time.perf_counter()-started,
            measurement='physics endpoint forward/sense; CPU contact audit is geometry-only, not GPU force reconstruction')
        (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        publish(out)
    finally:
        env.close()


if __name__ == '__main__':
    main()
