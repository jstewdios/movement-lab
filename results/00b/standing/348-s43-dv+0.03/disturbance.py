"""00B root-velocity intervention, immediately before a physics integration."""
import numpy as np
from movement_lab.measure import validate


def install(env, robot, at, dv):
    original = env.sim.step
    event = dict(mechanism='root link world-x velocity increment; no finite force',
                 requested_time_s=at, requested_dv_m_s=dv, applied=False)
    def step():
        t = float(env.sim.data.time[0])
        if not event['applied'] and t >= at - 1e-6:
            before = env.sim.data.qvel[0].detach().cpu().numpy().copy()
            velocity = robot.data.root_link_vel_w.clone()
            velocity[:, 0] += dv
            robot.write_root_link_velocity_to_sim(velocity)
            after = env.sim.data.qvel[0].detach().cpu().numpy().copy()
            event.update(applied=True, actual_time_s=t, qvel_before=before.tolist(),
                         qvel_after=after.tolist(), actual_dv_m_s=float(after[0]-before[0]))
            np.testing.assert_allclose(after[0]-before[0], dv, atol=2e-7)
            np.testing.assert_allclose(after[1:], before[1:], atol=2e-7)
        original()
        if event['applied'] and 'first_integrated_qvel' not in event:
            event['first_integrated_qvel'] = env.sim.data.qvel[0].detach().cpu().numpy().tolist()
    env.sim.step = step
    return event


def signals(a):
    q = a['body_pose_w'][:, 0, 3:7]
    return np.rad2deg(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2), -1, 1)))


def assess(a, m):
    validate(a)
    if not m['disturbance'].get('applied', True):
        return dict(settled=False, upright=False, recovered=False, recovery_s=None,
                    peak_tilt_deg=None, final_root_shift_m=None, peak_root_shift_m=None,
                    rms_effort_nm=None, work_j=None, contact_fraction=None,
                    foot_final_shift_m=None, foot_peak_lift_m=None, step_evidence=None,
                    limitation='terminated before scheduled disturbance; no recovery comparison')
    at = m['disturbance'].get('actual_time_s', m['disturbance']['requested_time_s'])
    t = a['time_s']; pre = (t >= at-.5) & (t < at-1e-5)
    post = t >= at; tail = t >= t[-1]-.5
    tilt = signals(a); xyz = a['body_pose_w'][:,0,:3]
    speed = np.linalg.norm(a['body_velocity_w'][:,0,:3], axis=1)
    angular = np.linalg.norm(a['body_velocity_w'][:,0,3:], axis=1)
    center = xyz[pre].mean(axis=0)
    settled = bool(np.max(speed[pre]) < .04 and np.max(angular[pre]) < .5 and np.max(tilt[pre]) < 12 and np.all(a['foot_found'][pre] > 0))
    # Stability permits a new standing location but explicitly reports displacement.
    stable = (tilt < 12) & (abs(tilt-tilt[pre].mean()) < 3) & (speed < .04) & (angular < .5) & (xyz[:,2] > center[2]-.02) & np.all(a['foot_found'] > 0, axis=1)
    dt = m['physics_dt']; hold = int(round(.5/dt)); recovery = None
    for i in np.flatnonzero(t >= at+.02):
        if i+hold <= len(t) and np.all(stable[i:i+hold]):
            recovery = float(t[i]-at); break
    upright = bool(np.max(tilt[post]) < 45 and np.min(xyz[post,2]) > center[2]-.04 and not m['stopped_on_done'])
    feet = [m['body_names'].index(n) for n in ('ankle_left','ankle_right')]
    pos = a['body_pose_w'][:,feet,:3]; ref = pos[pre].mean(axis=0)
    shift = np.linalg.norm(pos[tail,:,:2].mean(axis=0)-ref[:,:2], axis=1)
    lift = np.max(pos[post,:,2]-ref[:,2], axis=0)
    # Conservative sampled evidence: unload, lift, and displaced recontact.
    steps = []
    for f in range(2):
        airborne = (a['foot_found'][:,f] == 0) & (pos[:,f,2] > ref[f,2]+.005) & post
        evidence = False
        for i in np.flatnonzero(airborne):
            later = np.flatnonzero((np.arange(len(t)) > i) & (a['foot_found'][:,f] > 0))
            if len(later) and np.linalg.norm(pos[later[0],f,:2]-ref[f,:2]) > .005:
                evidence = True
        steps.append(evidence)
    power = np.abs(a['joint_force_nm']*a['joint_velocity_rad_s'])
    return dict(settled=settled, upright=upright, recovered=bool(settled and upright and recovery is not None and np.all(stable[tail])),
                recovery_s=recovery, peak_tilt_deg=float(tilt[post].max()),
                final_root_shift_m=float(np.linalg.norm(xyz[tail,:2].mean(axis=0)-center[:2])),
                peak_root_shift_m=float(np.linalg.norm(xyz[post,:2]-center[:2],axis=1).max()),
                rms_effort_nm=float(np.sqrt(np.mean(a['joint_force_nm'][post]**2))),
                work_j=float(power[post].sum()*dt), contact_fraction=np.mean(a['foot_found'][post]>0,axis=0).tolist(),
                foot_final_shift_m=shift.tolist(), foot_peak_lift_m=lift.tolist(), step_evidence=steps)
