"""Fixed seven-run continuation; validates each record and stops on first failure."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from movement_lab.departure_metrics01b import metrics
from movement_lab.evaluate import sha

PLAN=[('head-negative','head_-1'),('head-positive','head_+1'),('flex','flex_+1'),('shift-negative','shift_-1'),('shift-positive','shift_+1'),('unilateral','left_+1'),('final-control','zero')]
ROOT=Path('results/01b/departures')


def check(path):
    subprocess.run([sys.executable,'-m','movement_lab.validate01b',str(path)],check=True)
    f=next(path.glob('*/metadata.json'));m=json.loads(f.read_text())
    with np.load(f.parent/'trial.npz') as z,np.load(ROOT/'baseline/zero/trial.npz') as control:
        a=dict(z);c=dict(control)
        with np.load(f.parent/'initial.npz') as ini,np.load(ROOT/'baseline/zero/initial.npz') as ref:
            assert np.array_equal(ini['qpos'],ref['qpos']) and np.array_equal(ini['qvel'],ref['qvel'])
        prefix=a['control_index']<100
        for k in c:
            if k=='residual_rad':continue
            assert np.array_equal(a[k][prefix],c[k][:len(a['time_s'])][prefix]),(path,k,'precommand mismatch')
        assert np.array_equal(a['residual_rad'][prefix],np.zeros_like(a['residual_rad'][prefix]))
        assert (path/'resolved_config.txt').read_bytes()==(ROOT/'baseline/resolved_config.txt').read_bytes()
        assert sha(path/'replay_model.mjb')==sha(ROOT/'baseline/replay_model.mjb')
        result=metrics(a,c)
        result['exact_precommand_match']=True
        if m['sequence']=='zero':result['all_baseline_arrays_exact']=all(np.array_equal(a[k],c[k]) for k in c)
    assert m['gate']['pass'],m['gate']
    assert result['physical_gate']['pass'] and result['recovered'],result
    return result


def main():
    os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('MPLCONFIGDIR',str(Path.cwd()/'.cache/matplotlib'))
    check(ROOT/'baseline')
    for name,sequence in PLAN:
        out=ROOT/name
        if out.exists():raise RuntimeError(f'refuse to overwrite/resume {out}')
        start=time.perf_counter()
        with (ROOT/(name+'.log')).open('w') as log:
            p=subprocess.run([sys.executable,'-m','movement_lab.departure01b','--out',str(out),'--sequence',sequence],stdout=log,stderr=subprocess.STDOUT)
        record=dict(name=name,sequence=sequence,returncode=p.returncode,wall_s=time.perf_counter()-start)
        try:
            if p.returncode:raise RuntimeError(f'runner failed: {p.returncode}')
            record['metrics']=check(out)
        except Exception as e:
            record['failure']=repr(e)
            with (ROOT/'execution.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            return
        with (ROOT/'execution.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
