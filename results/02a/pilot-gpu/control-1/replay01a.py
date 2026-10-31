"""Replay saved generalized states with MuJoCo; does not rerun control/dynamics."""
import argparse
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trial', type=Path)
    p.add_argument('--speed', type=float, default=1)
    args = p.parse_args()
    if not np.isfinite(args.speed) or args.speed <= 0:
        p.error('speed must be finite and positive')
    import mujoco
    import mujoco.viewer
    model = mujoco.MjModel.from_binary_path(str(args.trial.parent / 'replay_model.mjb'))
    data = mujoco.MjData(model)
    with np.load(args.trial / 'trial.npz') as a, mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.monotonic()
        while viewer.is_running():
            t = ((time.monotonic() - start) * args.speed) % a['time_s'][-1]
            i = min(np.searchsorted(a['time_s'], t), len(a['time_s']) - 1)
            data.qpos[:] = a['qpos'][i]
            data.qvel[:] = a['qvel'][i]
            data.time = a['time_s'][i]
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(.01)


if __name__ == '__main__':
    main()
