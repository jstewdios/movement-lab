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

def measure(a,c,m):
    t=a['time_s']; pre=(t>=1.5)&(t<2); end=t>=9.5
    pulse=m['pulse']; finish=2+pulse['rise_s']+pulse['hold_s']+pulse['return_s']
    active=(t>=2)&(t<finish)
    hold=(t>=2+pulse['rise_s']+.5*pulse['hold_s'])&(t<2+pulse['rise_s']+pulse['hold_s'])
    dj=a['joint_position_rad']-c['joint_position_rad'][:len(t)]
    dc=a['com_position_w_m']-c['com_position_w_m'][:len(t)]
    pj=float(abs(dj[active]).max()); pc=float(np.linalg.norm(dc[active],axis=1).max())
    fj=float(abs(dj[end].mean(0)).max()) if end.any() else None
    fc=float(np.linalg.norm(dc[end].mean(0))) if end.any() else None
    valid=gate(a,m['stop_reasons'],False)['pass'] and t[-1]>=9.99
    stable=bool(valid and stable_window(a,pre) and stable_window(a,end))
    response=pj>=.005 or pc>=.0005
    contracts=bool(stable and (pj<.005 or fj<.5*pj) and (pc<.0005 or fc<.5*pc))
    dt=m['physics_dt']
    return dict(physical_valid=bool(valid),stable_final=stable,stable_hold=stable_window(a,hold),measurable_response=response,
        classification=('physical failure / incomplete' if not valid else 'unstable endpoint' if not stable else 'zero / below response threshold' if not response else 'stable transition with partial return' if contracts else 'stable changed configuration'),
        contracts_by_half=contracts,peak_joint_rad=pj,final_joint_rad=fj,peak_com_mm=pc*1000,final_com_mm=None if fc is None else fc*1000,
        hold_joint_delta_rad=dj[hold].mean(0).tolist(),final_joint_delta_rad=dj[end].mean(0).tolist() if end.any() else [],
        final_actor_delta_rad=(a['actor_action'][end]-c['actor_action'][:len(t)][end]).mean(0).tolist() if end.any() else [],
        final_com_delta_m=dc[end].mean(0).tolist() if end.any() else [],
        worst_penetration_mm=float(-1000*a['native_min_contact_distance_m'].min()),
        foot_contact_transitions=np.count_nonzero(np.diff((a['foot_found'][t>=2]>0).astype(int),axis=0),axis=0).tolist(),
        min_foot_load_n=abs(a['foot_force_w_n'][t>=2,:,2]).min(0).tolist(),
        peak_effort_nm=float(abs(a['joint_force_nm'][t>=2]).max()),
        sampled_abs_work_j=float(np.sum(abs(a['joint_force_nm']*a['joint_velocity_rad_s']))*dt))

def check(out,baseline):
    subprocess.run([sys.executable,'-m','movement_lab.validate01b',str(out)],check=True)
    f=next(out.glob('*/metadata.json'));m=json.loads(f.read_text());a=dict(np.load(f.parent/'trial.npz'));c=dict(np.load(next(baseline.glob('*/trial.npz'))))
    assert len(a['time_s'])==2000,'incomplete ten-second horizon'
    assert (out/'resolved_config.txt').read_bytes()==(baseline/'resolved_config.txt').read_bytes()
    assert sha(out/'replay_model.mjb')==sha(baseline/'replay_model.mjb')
    ini=dict(np.load(f.parent/'initial.npz')); ref=dict(np.load(next(baseline.glob('*/initial.npz'))))
    for k in ini: assert np.array_equal(ini[k],ref[k]),('initial',k)
    pre=a['control_index']<100
    for k in a: assert np.array_equal(a[k][pre],c[k][pre]),('prefix',k)
    command.RISE=m['pulse']['rise_s'];command.HOLD=m['pulse']['hold_s'];command.RETURN=m['pulse']['return_s']
    amp=np.array([m['amplitudes_rad'].get(n,0) for n in m['joint_names']])
    expected=np.array([amp*command.envelope(int(i)*m['control_dt']) for i in a['control_index']])
    assert np.allclose(a['residual_rad'],expected,atol=1e-7)
    limits=np.array(m['joint_limits_rad'])
    assert (a['joint_target_rad']>limits[:,0]+.02).all() and (a['joint_target_rad']<limits[:,1]-.02).all()
    assert np.max(np.maximum(limits[:,0]-a['joint_position_rad'],a['joint_position_rad']-limits[:,1]))<=.03
    if m['sequence'] in ('zero','zero_repeat'):
        assert gate(a,m['stop_reasons'],True)['pass']
        for k in a: assert np.array_equal(a[k],c[k]),('control repeat',k)
    result=measure(a,c,m); result['validation']='hashes, CPU geometry, composition, exact prefix, pulse, limits passed'
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
        if name=='baseline' and not record['physical_valid']:break
    (root/'cost.json').write_text(json.dumps(dict(wall_s=time.perf_counter()-started,completed=len(results)),indent=2))

if __name__=='__main__': main()
