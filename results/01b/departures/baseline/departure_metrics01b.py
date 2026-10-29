"""Fixed 01B departure criteria; array-only and independent of simulation."""
import numpy as np
from movement_lab.controlled01b import gate


def stable_window(a, mask):
    if not mask.any(): return False
    q=a['body_pose_w'][mask,0,3:7]
    tilt=np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2),-1,1)))
    return bool(tilt.max()<12 and (a['foot_found'][mask]>0).all()
        and not a['native_nonfoot_contacts'][mask].any()
        and np.linalg.norm(a['com_velocity_w_m_s'][mask],axis=1).max()<=.03
        and np.linalg.norm(a['body_velocity_w'][mask,0,3:],axis=1).max()<=.3
        and np.ptp(a['com_position_w_m'][mask,:2],axis=0).max()<=.005)


def metrics(a, control=None):
    t=a['time_s'];pre=(t>=1.5)&(t<2);end=t>=5.5;active=(t>=2)&(t<3.6)
    physical=gate(a,baseline=False)
    pre_ok=stable_window(a,pre);end_ok=stable_window(a,end) and t[-1]>=5.99
    result=dict(physical_gate=physical,settled_before_command=pre_ok,stable_final_half_second=end_ok)
    if not pre.any() or not end.any():
        return result|dict(recovered=False,controlled_response=False,classification='incomplete / invalid')
    com=a['com_position_w_m'];j=a['joint_position_rad'];q=a['body_pose_w'][:,0,3:7]
    refcom=com[pre].mean(0);refjoint=j[pre].mean(0);refq=q[pre].mean(0);refq/=np.linalg.norm(refq)
    orientation=2*np.degrees(np.arccos(np.clip(abs(q@refq),0,1)))
    delta=com-refcom
    return_mask=(t>=3.6)&(np.linalg.norm(delta[:,:2],axis=1)<=.005)&(abs(delta[:,2])<=.002)&(orientation<=3)&(abs(j-refjoint).max(1)<=.03)
    returned=bool(return_mask[end].all())
    # Require a continuous 0.5 s stable AND returned window after residual ends.
    recovery=None
    for start in np.flatnonzero(t>=3.6):
        w=(t>=t[start])&(t<t[start]+.5)
        if t[-1]+1e-6<t[start]+.495:break
        if return_mask[w].all() and stable_window(a,w): recovery=float(t[start]-3.6);break
    result.update(recovered=bool(physical['pass'] and pre_ok and end_ok and returned and recovery is not None),
        recovery_delay_s=recovery,returned_final=returned,
        peak_com_excursion_xyz_m=np.max(abs(delta[t>=2]),axis=0).tolist(),
        final_com_offset_xyz_m=(com[end].mean(0)-refcom).tolist(),
        final_max_joint_offset_rad=float(abs(j[end].mean(0)-refjoint).max()),
        peak_orientation_change_deg=float(orientation[t>=2].max()),
        foot_contact_transitions=np.count_nonzero(np.diff((a['foot_found'][t>=2]>0).astype(int),axis=0),axis=0).tolist(),
        min_foot_fz_magnitude_n=abs(a['foot_force_w_n'][t>=2,:,2]).min(0).tolist(),
        final_mean_foot_fz_n=a['foot_force_w_n'][end,:,2].mean(0).tolist(),
        peak_effort_postsettle_nm=float(abs(a['joint_force_nm'][t>=2]).max()))
    response=False
    if control is not None:
        assert len(control['time_s'])>=len(t)
        c={k:v[:len(t)] for k,v in control.items()}
        dj=j-c['joint_position_rad'];dc=com-c['com_position_w_m']
        peakjoint=float(abs(dj[active]).max()) if active.any() else 0.
        peakcom=float(np.linalg.norm(dc[active],axis=1).max()) if active.any() else 0.
        finaljoint=float(abs(dj[end].mean(0)).max());finalcom=float(np.linalg.norm(dc[end].mean(0)))
        response=peakjoint>=.005 or peakcom>=.0005
        result.update(matched_peak_joint_response_rad=peakjoint,matched_peak_com_response_m=peakcom,
            matched_peak_com_xyz_m=np.max(abs(dc[active]),axis=0).tolist(),matched_peak_joint_by_joint_rad=np.max(abs(dj[active]),axis=0).tolist(),
            matched_final_joint_offset_rad=finaljoint,matched_final_com_offset_m=finalcom,
            matched_peak_foot_fz_change_n=np.max(abs(a['foot_force_w_n'][active,:,2]-c['foot_force_w_n'][active,:,2]),axis=0).tolist(),
            response_contracts=bool((not peakjoint>=.005 or finaljoint<.5*peakjoint) and (not peakcom>=.0005 or finalcom<.5*peakcom)))
    result['controlled_response']=bool(response and result['recovered'] and result.get('response_contracts',False))
    result['classification']=('invalid/contact failure' if not physical['pass'] else 'unrecovered' if not result['recovered'] else
        'controlled articulation / return' if result['controlled_response'] else 'stable; measurable response or contraction not established')
    return result
