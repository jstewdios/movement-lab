"""Bounded hardware / simulator benchmark. Writes only to the 02C output tree."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, default=Path('results/02c/preflight.json'))
    p.add_argument('--envs', type=int, default=64)
    a = p.parse_args()
    import numpy as np
    import torch
    from movement_lab.sim02b import Sim, Condition, rollout
    from movement_lab.model02b import env_cfg
    torch.set_num_threads(1)
    info = dict(platform=platform.platform(), cpu_count=os.cpu_count(),
                packages={k: importlib.metadata.version(k) for k in
                          ('torch', 'numpy', 'mujoco', 'mjlab', 'mujoco-warp', 'better-actuator-models')},
                cuda_available=torch.cuda.is_available())
    import bam
    motor = Path(bam.__file__).parent / 'params/xl330/m6.json'
    info['bam_parameter_sha256'] = hashlib.sha256(motor.read_bytes()).hexdigest()
    info['bam_parameter_canonical_sha256'] = hashlib.sha256(json.dumps(json.loads(motor.read_text()), sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    theta = np.load('results/02b/runs/ars-push-s1/thetas.npz')['final']
    sim = Sim()
    start = time.perf_counter()
    for i in range(8):
        rollout(sim, theta, Condition(seed=i, duration_s=4, push_time_s=1.5, push_dv=(.32, 0)))
    info['native_8_trials_wall_s'] = time.perf_counter() - start
    if info['cuda_available']:
        info['gpu'] = torch.cuda.get_device_name()
        from mjlab.envs import ManagerBasedRlEnv
        cfg = env_cfg(num_envs=a.envs)
        start = time.perf_counter()
        env = ManagerBasedRlEnv(cfg, device='cuda:0')
        try:
            info['gpu_compile_wall_s'] = time.perf_counter() - start
            env.reset()
            action = torch.zeros((a.envs, 14), device='cuda:0')
            for _ in range(5):
                env.step(action)
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(100):
                env.step(action)
            torch.cuda.synchronize()
            info.update(gpu_100_steps_wall_s=time.perf_counter() - start, gpu_envs=a.envs,
                        gpu_peak_bytes=torch.cuda.max_memory_allocated(),
                        gpu_finite=bool(torch.isfinite(env.sim.data.qpos[:]).all()))
        finally:
            env.close()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(info, indent=2) + '\n')
    print(json.dumps(info, indent=2))


if __name__ == '__main__':
    main()
