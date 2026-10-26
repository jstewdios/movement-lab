"""Array-only measurements; SI units throughout."""
import numpy as np


def validate(a):
    n = len(a['time_s'])
    assert n > 0, 'empty trial'
    assert all(len(v) == n for v in a.values()), 'unsynchronized arrays'
    assert all(np.isfinite(v).all() for v in a.values()), 'nonfinite measurement'
    assert np.all(np.diff(a['time_s']) > 0), 'nonmonotonic time'
    q = a['body_pose_w'][..., 3:7]
    assert np.allclose(np.linalg.norm(q, axis=-1), 1, atol=1e-4), 'invalid quaternion'
    assert a['joint_velocity_rad_s'].shape == a['joint_force_nm'].shape


def summary(a):
    validate(a)
    # Quaternion wxyz: vertical alignment of the trunk local z axis.
    q = a['body_pose_w'][:, 0, 3:7]
    tilt = np.arccos(np.clip(1 - 2*(q[:, 1]**2 + q[:, 2]**2), -1, 1))
    power = a['joint_force_nm'] * a['joint_velocity_rad_s']
    dt = np.diff(a['time_s'], prepend=0)
    return dict(samples=len(dt), duration_s=float(a['time_s'][-1]),
                min_root_height_m=float(a['body_pose_w'][:, 0, 2].min()),
                max_root_tilt_deg=float(np.rad2deg(tilt).max()),
                rms_joint_force_nm=float(np.sqrt(np.mean(a['joint_force_nm']**2))),
                peak_abs_joint_force_nm=float(np.abs(a['joint_force_nm']).max()),
                absolute_mechanical_work_j=float(np.sum(np.abs(power)*dt[:, None])),
                foot_contact_fraction=np.mean(a['foot_found'] > 0, axis=0).tolist(),
                peak_foot_force_n=np.linalg.norm(a['foot_force_w_n'], axis=-1).max(axis=0).tolist())
