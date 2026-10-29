"""01B preregistered finite-horizon baseline/contact gate and publication."""
import json
from pathlib import Path
import numpy as np
from movement_lab.evaluate import sha


def gate(a, stop=(), baseline=True):
    t = a['time_s']
    q = a['body_pose_w'][:, 0, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2), -1, 1)))
    w = t >= 1.5
    reasons = list(stop)
    if not all(np.isfinite(v).all() for v in a.values()): reasons.append('nonfinite')
    if t[-1] < 5.99: reasons.append('incomplete_horizon')
    if a['native_min_contact_distance_m'].min() < -.003: reasons.append('penetration_over_3mm')
    if a['native_nonfoot_contacts'].any(): reasons.append('nonfoot_contact')
    if tilt.max() > 12: reasons.append('tilt_over_12deg')
    if not w.any(): reasons.append('no_settled_window')
    elif baseline:
        if not (a['foot_found'][w] > 0).all(): reasons.append('settled_foot_unloading')
        if np.linalg.norm(a['com_velocity_w_m_s'][w],axis=1).max() > .03: reasons.append('com_speed_over_0.03')
        if np.linalg.norm(a['body_velocity_w'][w,0,3:],axis=1).max() > .3: reasons.append('angular_speed_over_0.3')
        if np.ptp(a['com_position_w_m'][w,:2],axis=0).max() > .005: reasons.append('com_drift_over_5mm')
    return dict(pass_=not reasons, reasons=sorted(set(reasons))) | {'pass': not reasons}


def publish(out):
    # Reuse only the measurement replay/plots, retaining the original 01A metrics.
    from movement_lab.discovery_metrics import publish as descriptive_publish
    descriptive_publish(out)
    p=out/'replay.html'
    p.write_text(p.read_text().replace('01A','01B'))
    records={f.parent.name: json.loads(f.read_text()) for f in out.glob('*/metadata.json')}
    validation={}
    initial=None
    for name,m in records.items():
        with np.load(out/name/'trial.npz') as z, np.load(out/name/'initial.npz') as ini:
            a=dict(z)
            assert np.allclose(a['action'],a['actor_action']+a['residual_rad'],atol=1e-7)
            assert gate(a,m['stop_reasons'], name in ('zero','zero_repeat')) == m['gate']
            assert np.all(ini['qvel']==0)
            if initial is None: initial=ini['qpos'].copy()
            assert np.array_equal(initial,ini['qpos'])
            from movement_lab.study01b import SEQUENCES,envelope
            amp=np.array([SEQUENCES[name].get(n,0) for n in m['joint_names']])
            expected=np.array([amp*envelope(int(i)*m['control_dt']) for i in a['control_index']])
            assert np.allclose(expected,a['residual_rad'],atol=1e-7)
            validation[name]={'composition':True,'residual_schedule':True,'initial_state':True,'gate_recomputed':True}
    (out/'validation01b.json').write_text(json.dumps(validation,indent=2)+'\n')
    files=[p for p in out.rglob('*') if p.is_file() and p.name!='manifest.json' and 'matplotlib' not in p.parts]
    (out/'manifest.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in files},indent=2)+'\n')
