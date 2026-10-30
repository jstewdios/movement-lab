"""Bounded 01C pilot, with strict baseline gate and matched-record validation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from movement_lab.controlled01b import gate
from movement_lab.departure_metrics01b import stable_window
from movement_lab.evaluate import sha
from movement_lab import transition01c as command

PLAN=[('baseline','zero'),('flex','flex_+1'),('extend','flex_-1'),('head-positive','head_+1'),('head-negative','head_-1'),('rock-forward','rock_+1'),('rock-backward','rock_-1'),('final-control','zero_repeat')]

SAFETY_STOPS = {
    'penetration_over_3mm', 'nonfoot_contact', 'tilt_over_12deg',
    'generalized_speed_over_100', 'joint_limit_overshoot_over_0.03rad',
    'foot_force_over_100N', 'composed_target_limit', 'precommand_settling_failed',
}


def observed_safety_stops(a, limits):
    q = a['body_pose_w'][:, 0, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1-2*(q[:, 1]**2+q[:, 2]**2), -1, 1)))
    tests = {
        'penetration_over_3mm': a['native_min_contact_distance_m'].min() < -.003,
        'nonfoot_contact': a['native_nonfoot_contacts'].any(),
        'tilt_over_12deg': tilt.max() > 12,
        'generalized_speed_over_100': abs(a['qvel']).max() > 100,
        'joint_limit_overshoot_over_0.03rad': np.maximum(limits[:,0]-a['joint_position_rad'], a['joint_position_rad']-limits[:,1]).max() > .03,
        'foot_force_over_100N': np.linalg.norm(a['foot_force_w_n'], axis=-1).max() > 100,
    }
    return {name for name, failed in tests.items() if failed}


def measure(a,c,m):
    t=a['time_s']; pre=(t>=1.5)&(t<2); complete=bool(t[-1]>=9.99)
    end=(t>=9.5)&complete
    pulse=m['pulse']; finish=2+pulse['rise_s']+pulse['hold_s']+pulse['return_s']
    active=(t>=2)&(t<finish)
    hold_end=2+pulse['rise_s']+pulse['hold_s']
    hold=(t>=2+pulse['rise_s']+.5*pulse['hold_s'])&(t<hold_end)
    hold_available=bool(t[-1]>=hold_end-1e-4 and hold.any())
    dj=a['joint_position_rad']-c['joint_position_rad'][:len(t)]
    dc=a['com_position_w_m']-c['com_position_w_m'][:len(t)]
    pj=float(abs(dj[active]).max()) if active.any() else None
    pc=float(np.linalg.norm(dc[active],axis=1).max()) if active.any() else None
    fj=float(abs(dj[end].mean(0)).max()) if end.any() else None
    fc=float(np.linalg.norm(dc[end].mean(0))) if end.any() else None
    valid=gate(a,m['stop_reasons'],False)['pass'] and complete
    stable=bool(valid and stable_window(a,pre) and stable_window(a,end)) if complete else None
    response=bool((pj is not None and pj>=.005) or (pc is not None and pc>=.0005))
    contracts=bool(stable and (pj is None or pj<.005 or fj<.5*pj) and (pc is None or pc<.0005 or fc<.5*pc))
    post=t>=2
    result=dict(physical_valid=bool(valid),stable_final=stable,stable_hold=stable_window(a,hold) if hold_available else None,measurable_response=response,
        classification=('physical failure / incomplete' if not valid else 'unstable endpoint' if not stable else 'zero / below response threshold' if not response else 'stable transition with partial return' if contracts else 'stable changed configuration'),
        contracts_by_half=contracts,peak_joint_rad=pj,final_joint_rad=fj,peak_com_mm=None if pc is None else pc*1000,final_com_mm=None if fc is None else fc*1000,
        hold_joint_delta_rad=dj[hold].mean(0).tolist() if hold_available else None,final_joint_delta_rad=dj[end].mean(0).tolist() if end.any() else None,
        final_actor_delta_rad=(a['actor_action'][end]-c['actor_action'][:len(t)][end]).mean(0).tolist() if end.any() else None,
        final_com_delta_m=dc[end].mean(0).tolist() if end.any() else None,
        worst_penetration_mm=float(-1000*a['native_min_contact_distance_m'].min()),
        foot_contact_transitions=np.count_nonzero(np.diff((a['foot_found'][post]>0).astype(int),axis=0),axis=0).tolist() if post.any() else None,
        min_foot_load_n=abs(a['foot_force_w_n'][post,:,2]).min(0).tolist() if post.any() else None,
        peak_effort_nm=float(abs(a['joint_force_nm'][post]).max()) if post.any() else None,
        sampled_abs_work_j=float(np.sum(abs(a['joint_force_nm']*a['joint_velocity_rad_s']))*m['physics_dt']))
    if not complete:
        result.update(recorded_seconds=float(t[-1]),stop_reasons=m['stop_reasons'])
    return result

def check(out,baseline):
    subprocess.run([sys.executable,'-m','movement_lab.validate01b',str(out)],check=True)
    f=next(out.glob('*/metadata.json'));m=json.loads(f.read_text());a=dict(np.load(f.parent/'trial.npz'));c=dict(np.load(next(baseline.glob('*/trial.npz'))))
    from movement_lab.measure import validate
    validate(a); validate(c)
    assert len(c['time_s'])==2000 and c['time_s'][-1]>=9.99, 'incomplete baseline'
    assert gate(c,baseline=True)['pass'], 'invalid baseline'
    n=len(a['time_s'])
    assert 0<n<=2000 and n%4==0, 'invalid sample count'
    assert np.array_equal(a['time_s'],c['time_s'][:n]), 'missing or mismatched samples'
    assert set(a)==set(c), 'missing or unexpected arrays'
    declared=set(m['stop_reasons'])
    assert declared<=SAFETY_STOPS, ('unknown stop reasons',declared-SAFETY_STOPS)
    if n<2000:
        assert m['sequence'] not in ('zero','zero_repeat'), 'incomplete baseline'
        assert declared, 'unexplained incomplete movement'

    assert (out/'resolved_config.txt').read_bytes()==(baseline/'resolved_config.txt').read_bytes()
    assert sha(out/'replay_model.mjb')==sha(baseline/'replay_model.mjb')
    ini=dict(np.load(f.parent/'initial.npz')); ref=dict(np.load(next(baseline.glob('*/initial.npz'))))
    for k in ini: assert np.array_equal(ini[k],ref[k]),('initial',k)
    pre=a['control_index']<100
    for k in a: assert np.array_equal(a[k][pre],c[k][:n][pre]),('prefix',k)
    command.RISE=m['pulse']['rise_s'];command.HOLD=m['pulse']['hold_s'];command.RETURN=m['pulse']['return_s']
    amp=np.array([m['amplitudes_rad'].get(n,0) for n in m['joint_names']])
    expected=np.array([amp*command.envelope(int(i)*m['control_dt']) for i in a['control_index']])
    assert np.allclose(a['residual_rad'],expected,atol=1e-7)
    limits=np.array(m['joint_limits_rad'])
    assert (a['joint_target_rad']>limits[:,0]+.02).all() and (a['joint_target_rad']<limits[:,1]-.02).all()
    observed=observed_safety_stops(a,limits)
    measurable_stops=SAFETY_STOPS-{'composed_target_limit','precommand_settling_failed'}
    assert observed==declared&measurable_stops, ('inconsistent safety stop evidence',observed,declared)
    if 'precommand_settling_failed' in declared:
        assert not stable_window(a,(a['time_s']>=1.5)&(a['time_s']<2))

    if m['sequence'] in ('zero','zero_repeat'):
        assert gate(a,m['stop_reasons'],True)['pass']
        for k in a: assert np.array_equal(a[k],c[k]),('control repeat',k)
    result=measure(a,c,m); result['validation']=('hashes, CPU geometry, composition, exact prefix, pulse, limits passed' if not declared else 'integrity and safety-stop evidence verified; physical outcome failed')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();root=args.out
    root.mkdir(parents=True,exist_ok=False)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    (root/'PLAN.json').write_text(json.dumps(dict(trials=PLAN,seconds=10,launch_budget_wall_s=600,maximum_trials=8,policy='checkpoint99',initial_clearance_mm=1,penetration_limit_mm=3,pulse=dict(start=2,rise=.6,hold=1,return_duration=1.4),thresholds=dict(joint_response_rad=.005,com_response_m=.0005),interpretation='stable changed configurations accepted; return evaluated separately'),indent=2))
    (root/'study01c.py').write_bytes(Path(__file__).read_bytes())
    started=time.perf_counter();results={}
    for name,seq in PLAN:
        if time.perf_counter()-started>510:
            results['budget_stop']='less than 90 seconds remain';break
        out=root/name; tick=time.perf_counter()
        with (root/(name+'.log')).open('w') as log:
            proc=subprocess.run([sys.executable,'-m','movement_lab.transition01c','--out',str(out),'--sequence',seq],stdout=log,stderr=subprocess.STDOUT)
        record=dict(name=name,returncode=proc.returncode,wall_s=time.perf_counter()-tick)
        try:
            assert proc.returncode==0, 'simulation subprocess failed'
            record.update(check(out,root/'baseline'))
            if name=='baseline':
                historical=Path('results/01b/departures/baseline/zero/trial.npz')
                if historical.exists():
                    a=dict(np.load(out/'zero/trial.npz'));b=dict(np.load(historical))
                    record['historical_01b_prefix_exact']=all(np.array_equal(a[k][:len(b[k])],b[k]) for k in b)
                    assert record['historical_01b_prefix_exact']
        except Exception as e:
            record['failure']=repr(e);results[name]=record
            (root/'summary.json').write_text(json.dumps(results,indent=2));print(record,flush=True);break
        results[name]=record
        (root/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(record),flush=True)
        if not record['physical_valid']:break
    (root/'cost.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-started,completed=sum(isinstance(r,dict) and r.get('returncode')==0 for r in results.values())),indent=2))

if __name__=='__main__': main()
