"""Predeclared post-training interventions, never used for selection."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
from movement_lab.control02d import Controller, Design
from movement_lab.sim02b import Sim
from movement_lab.sim02c import BASE
from movement_lab.sim02d import Condition, rollout, event_time
from movement_lab.eval02d import load, metrics
from movement_lab.eval02c import write_gzip


def intervention(kind, onset):
    class Intervened(Controller):
        def __init__(self,*a,**kw):
            super().__init__(*a,**kw)
            self.calls=0;self.last=np.zeros(14)

        def __call__(self, x):
            t=self.calls*.02;self.calls+=1
            if kind=='freeze-after-onset' and t >= onset-1e-9:
                return self.last.copy()
            if kind=='history-is-current':
                self.history=np.clip(np.asarray(x),-10,10).copy()
            self.last=super().__call__(x)
            return self.last.copy()
    return Intervened


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('results/02d'));a=p.parse_args()
    dest=a.root/'eval/diagnostic.jsonl.gz'
    if dest.exists():raise SystemExit('refusing to overwrite diagnostic evidence')
    sim=Sim();theta=np.load(BASE)['final'];rows=[];checks=[]
    for seed in (1,2,3):
        vec,design=load(a.root/'runs'/f'history-s{seed}')
        for j,mag in enumerate((.08,.20,.32,.36)):
            for rep in range(2):
                c=Condition(seed=6_200_000_000+j*10+rep,duration_s=6,joint_noise_rad=.01,push_time_s=2.5,push_dv=(mag,0.))
                original=None
                for kind in ('intact','freeze-after-onset','history-is-current'):
                    with patch('movement_lab.sim02d.Controller',intervention(kind,event_time(c))):
                        s,r=rollout(sim,theta,vec,c,design,record=True,terminate=False)
                    metrics(s,r,c)
                    s.update(controller=f'history-s{seed}',intervention=kind,condition=asdict(c))
                    rows.append(s)
                    if kind=='intact':original=r
                    if kind=='freeze-after-onset':
                        before=r['time_s']<=event_time(c)+1e-9
                        exact=np.array_equal(r['qpos'][before],original['qpos'][before])
                        checks.append(exact)
                        assert exact,'freeze changed the pre-event stance'
                        post=r['time_s']>event_time(c)+1e-9
                        assert np.all(r['residual'][post]==r['residual'][post][0])
        print('diagnosed',seed,flush=True)
    write_gzip(dest,rows)
    (a.root/'eval/diagnostic-checks.json').write_text(json.dumps(dict(pre_event_identity_checks=checks,passed=all(checks)),indent=2)+'\n')


if __name__=='__main__':main()
