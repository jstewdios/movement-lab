"""Saved-record causal and provenance checks; fails rather than interpreting bad data."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from movement_lab.measure import validate


def check(root):
    records={}
    for p in sorted(root.glob('*/metadata.json')):
        m=json.loads(p.read_text());a=dict(np.load(p.parent/'trial.npz'));validate(a)
        for source,key in [('evaluate.py','measurement_code_sha256'),('measure.py','analysis_code_sha256'),('disturbance.py','disturbance_code_sha256'),('resolved_config.txt','config_sha256')]:
            assert hashlib.sha256((p.parent/source).read_bytes()).hexdigest()==m[key]
        np.testing.assert_array_equal(a['command'], 0)
        e=m['disturbance']
        if not e['applied']:
            assert m['controller']=='default-position' and m['stopped_on_done'] and a['time_s'][-1]<e['requested_time_s']
            records[p.parent.name]=(a,m)
            continue
        np.testing.assert_allclose(np.array(e['qvel_after'])[0]-e['qvel_before'][0],e['requested_dv_m_s'],atol=2e-7)
        np.testing.assert_allclose(np.array(e['qvel_after'])[1:],e['qvel_before'][1:],atol=2e-7)
        assert abs(e['actual_time_s']-e['requested_time_s'])<=m['physics_dt']*1.01
        records[p.parent.name]=(a,m)
    causal=[]
    for name,(a,m) in records.items():
        dv=m['disturbance']['requested_dv_m_s']
        control_name=name.split('-dv')[0]+'-dv+0.00'
        c,cm=records[control_name]
        for key in ('config_sha256','upstream_commit','upstream_status','versions','physics_dt','control_dt','checkpoint_sha256','controller'):
            assert m[key]==cm[key],(name,key)
        if not m['disturbance']['applied']:
            continue
        onset=m['disturbance']['actual_time_s']
        # A boundary trial may terminate before its control. Integer indices
        # select the same pre-event samples without requiring equal horizons.
        pre=np.flatnonzero(a['time_s']<=onset+1e-6)
        # Identical states and all derived recordings up to injection.
        for key in a:
            np.testing.assert_array_equal(a[key][pre],c[key][pre],err_msg=name+':'+key)
        i=np.flatnonzero(a['time_s']>onset+1e-6)[0]
        delta=float(a['qvel'][i,0]-c['qvel'][i,0])
        dx=float(a['qpos'][i,0]-c['qpos'][i,0])
        if dv:
            assert delta*dv>0,(name,delta)
            assert dx*dv>0,(name,dx)
        causal.append(dict(trial=name,first_step_delta_vx=delta,first_step_delta_x=dx))
    for name,(a,m) in records.items():
        if name.startswith('99-'):
            peer='348-'+name[3:]
            if peer not in records: continue
            b,bm=records[peer]
            for key in ('config_sha256','seed','upstream_commit','upstream_status','versions','physics_dt','control_dt','sample_mode'):
                assert m[key]==bm[key], (name,peer,key)
            for key in ('qpos','qvel'):
                with np.load(root/name/'initial.npz') as initial, np.load(root/peer/'initial.npz') as other:
                    np.testing.assert_array_equal(initial[key],other[key])
    # Seeds are not independent physical perturbations in this nominal configuration.
    seed_differences={}
    for name,(a,m) in records.items():
        if '-s42-' in name:
            other=name.replace('-s42-','-s43-')
            if other in records:
                b,_=records[other]
                seed_differences[name]=bool(all(np.array_equal(a[k],b[k]) for k in a))
    result=dict(trials=len(records),causal_checks=causal,seed42_43_arrays_identical=seed_differences)
    (root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    print(json.dumps(check(p.parse_args().root),indent=2))
