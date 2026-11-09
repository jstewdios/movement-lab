"""Measured-state replay, progress figures and descriptive tables; no selection."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import numpy as np
from movement_lab.sim02b import Sim
from movement_lab.sim02c import BASE
from movement_lab.sim02d import rollout, event_time
from movement_lab.control02d import Design
from movement_lab.eval02d import controllers, battery, metrics, effective_mode
from movement_lab.eval02c import read_rows
from movement_lab.replay02c import frames as legacy_frames
from movement_lab.contact02c import analyze
from movement_lab.learn02c import completed_checkpoints


LABELS = dict(reflex='02B frozen reflex', oracle02c='02C fixed · onset oracle', imu02c='02C fixed · IMU trigger',
              **{'history-s1':'02D feedback + history · seed 1', 'memoryless-s1':'02D current IMU only · seed 1',
                 'early':'02D early · generation 1'})


def frames(sim, summary, record, condition):
    """Reuse 02C's measured-state renderer with a cheaper display-only hull.

    The 3-D hull already contains unique vertices. Qhull can project them
    directly without the old per-frame rounding/unique/sorting pass. Vertex
    ordering and the 24-point display sampling can differ; body states,
    contact measurements and full-geometry clearance audits are unchanged.
    """
    from scipy.spatial import ConvexHull
    from unittest.mock import patch
    def projection(points):
        pts=np.asarray(points,float)
        return pts[ConvexHull(pts).vertices]
    with patch('movement_lab.replay02c.convex_hull',projection):
        return legacy_frames(sim,summary,record,condition)


def tables(root):
    out = {}
    for name in ('grid', 'shifts'):
        data = {p.name[:-9]:read_rows(p) for p in sorted((root/'eval'/name).glob('*.jsonl.gz'))}
        ref = data['reflex']
        result = {}
        for ctrl, rows in data.items():
            groups = sorted({r['set'] for r in rows})
            result[ctrl] = dict(n=len(rows), recovered=sum(r['recovered_200hz'] for r in rows),
                wins=sum(r['recovered_200hz'] and not b['recovered_200hz'] for r,b in zip(rows,ref)),
                losses=sum(b['recovered_200hz'] and not r['recovered_200hz'] for r,b in zip(rows,ref)),
                steps=sum(r['recovery_with_step'] for r in rows),
                groups={g:dict(n=sum(r['set']==g for r in rows), recovered=sum(r['recovered_200hz'] for r in rows if r['set']==g)) for g in groups},
                quiet_residual_rms_rad=[r['quiet_residual_rms_rad'] for r in rows if r['set']=='quiet'],
                quiet_root_path_m=[r['quiet_root_path_m'] for r in rows if r['set']=='quiet'],
                physical_valid=sum(r['physical_valid'] for r in rows),
                self_contact_trials=sum(r['self_contact_samples']>0 for r in rows),
                recovered_peak_clearance_m=max([max(r['peak_clearance_m']) for r in rows if r['recovered_200hz']] or [0.]))
            result[ctrl]['paired_against']={name:dict(
                wins=sum(r['recovered_200hz'] and not b['recovered_200hz'] for r,b in zip(rows,base)),
                losses=sum(b['recovered_200hz'] and not r['recovered_200hz'] for r,b in zip(rows,base)))
                for name,base in data.items() if name in ('reflex','oracle02c','imu02c')}
        out[name] = result
    return out


def figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,2,figsize=(11,4), constrained_layout=True)
    progress = {}
    for arm, color in [('history','#287d65'),('memoryless','#b46729')]:
        for seed in (1,2,3):
            name=f'{arm}-s{seed}'
            rows=[json.loads(str(np.load(p)['row'])) for p in completed_checkpoints(root/'runs'/name)]
            progress[name]=rows
            x=[r['trials'] for r in rows]
            axes[0].plot(x,[r['mean_score'] for r in rows],color=color,alpha=.6,label=arm if seed==1 else None)
            axes[1].plot(x,[r['winner_score']-r['zero_score'] for r in rows],color=color,alpha=.6)
    axes[0].set(title='Candidate quality during learning',xlabel='Training episodes',ylabel='Mean batch fitness')
    axes[1].set(title='Finalist versus frozen reflex on each batch',xlabel='Training episodes',ylabel='Paired fitness difference')
    axes[1].axhline(0,color='gray',lw=.6)
    axes[0].legend()
    for ax in axes:ax.grid(alpha=.15)
    fig.savefig(root/'report/learning.png',dpi=150);plt.close(fig)
    (root/'report/progress.json').write_text(json.dumps(progress,indent=2)+'\n')


def replay(root, table):
    sim=Sim(); theta=np.load(BASE)['final']
    cs=[x for x in controllers(root) if x[0] in LABELS]
    first=np.load(root/'runs/history-s1/generation-001.npz')['best']
    cs.append(('early',first,Design(),'learned'))
    selected=[(label,c) for label,c in battery() if label in ('quiet','fwd-.08','fwd-.20','fwd-.32','fwd-.36','false-gyro','back-.08') and c.seed%10==0]
    records=root/'records';records.mkdir(exist_ok=True)
    data=[]; index={}; contacts={}
    for name,vec,design,mode in cs:
        for label,c in selected:
            key=f'{name}-{label}'
            npz=records/(key+'.npz'); meta=records/(key+'.json')
            if npz.exists():
                saved=json.loads(meta.read_text());r=dict(np.load(npz));s=saved['summary']
                assert saved['weights']==vec.tolist() and saved['condition']==json.loads(json.dumps(asdict(c)))
            else:
                s,r=rollout(sim,theta,vec,c,design,record=True,terminate=False,mode=effective_mode(mode,c))
                metrics(s,r,c)
                np.savez_compressed(npz,**r)
                meta.write_text(json.dumps(dict(controller=name,kind=label,weights=vec.tolist(),design=asdict(design),mode=effective_mode(mode,c),
                                               condition=asdict(c),z0=sim.z0,summary=s),indent=2)+'\n')
            d=frames(sim,s,r,replace(c,push_time_s=event_time(c)))
            for f,i in zip(d['frames'],range(3,len(r['time_s']),4)):
                f['strength']=round(float(np.sqrt(np.mean(r['residual'][i]**2))),5)
            d.update(label=f'{LABELS[name]} / {label}',kind=label,event_time=event_time(c),recovered=s['recovered_200hz'],
                     step=s['recovery_with_step'],failure_time=s['failure_time_s'])
            index[(name,label)]=len(data);data.append(d)
            if s['failure_time_s'] is None or s['failure_time_s'] > (event_time(c) or .5):
                contacts[key]=analyze(sim,r,event_time(c),s['failure_time_s'])
            else:
                contacts[key]=dict(skipped='failure before diagnostic window')
        print('recorded replay',name,flush=True)
    # Explicitly selected because it exposed the first pilot's quiet regression.
    # These are development diagnostics, not added held-out confirmation wins.
    c=next(c for label,c in battery(True) if label=='quiet' and c.seed==5_000_000_003)
    from movement_lab.eval02d import load
    for name,path in [('failed-pilot',root/'pilot/history-s71'),('final-on-development',root/'runs/history-s1')]:
        vec,design=load(path);key=name+'-quiet-development';npz=records/(key+'.npz');meta=npz.with_suffix('.json')
        if npz.exists():
            saved=json.loads(meta.read_text());r=dict(np.load(npz));s=saved['summary']
            assert saved['weights']==vec.tolist() and saved['condition']==canon_condition(c)
        else:
            s,r=rollout(sim,theta,vec,c,design,record=True,terminate=False);metrics(s,r,c)
            np.savez_compressed(npz,**r)
            meta.write_text(json.dumps(dict(controller=name,kind='quiet-development',weights=vec.tolist(),design=asdict(design),mode='learned',condition=asdict(c),z0=sim.z0,summary=s),indent=2)+'\n')
        d=frames(sim,s,r,c)
        for f,i in zip(d['frames'],range(3,len(r['time_s']),4)):f['strength']=round(float(np.sqrt(np.mean(r['residual'][i]**2))),5)
        d.update(label=('Failed first pilot' if name=='failed-pilot' else 'Final history seed 1')+' / prior quiet-development failure',
                 kind='quiet',event_time=None,recovered=s['recovered_200hz'],step=s['recovery_with_step'],failure_time=s['failure_time_s'])
        index[(name,'quiet')]=len(data);data.append(d)
    # Show the first measured quiet regression of history seed 3, with its
    # matched reflex. Display selection does not change any controller choice.
    failure=next(r for r in read_rows(root/'eval/grid/history-s3.jsonl.gz') if r['set']=='quiet' and not r['recovered_200hz'])
    from movement_lab.sim02d import Condition
    c=Condition(**failure['condition'])
    for name,mode in [('quiet-regression-reflex','reflex'),('quiet-regression-s3','learned')]:
        vec,design=load(root/'runs/history-s3') if mode=='learned' else (np.zeros(22),Design())
        npz=records/(name+'.npz');meta=npz.with_suffix('.json')
        if npz.exists():
            saved=json.loads(meta.read_text());r=dict(np.load(npz));s=saved['summary']
            assert saved['weights']==vec.tolist() and saved['condition']==canon_condition(c)
        else:
            s,r=rollout(sim,theta,vec,c,design,record=True,terminate=False,mode=mode);metrics(s,r,c)
            np.savez_compressed(npz,**r)
            meta.write_text(json.dumps(dict(controller=name,kind='quiet-regression',weights=vec.tolist(),design=asdict(design),mode=mode,condition=asdict(c),z0=sim.z0,summary=s),indent=2)+'\n')
        d=frames(sim,s,r,c)
        for f,i in zip(d['frames'],range(3,len(r['time_s']),4)):f['strength']=round(float(np.sqrt(np.mean(r['residual'][i]**2))),5)
        d.update(label=('Frozen reflex' if mode=='reflex' else 'History seed 3')+' / first held-out quiet regression',
                 kind='quiet',event_time=None,recovered=s['recovered_200hz'],step=s['recovery_with_step'],failure_time=s['failure_time_s'])
        index[(name,'quiet')]=len(data);data.append(d)
    presets=[]
    def pair(title,a,ka,b,kb,caption):
        presets.append(dict(title=title,left=index[(a,ka)],right=index[(b,kb)],caption=caption))
    pair('01 · Frozen reflex versus sensed feedback','reflex','fwd-.32','history-s1','fwd-.32',
         'Same held-out 0.32 m/s impulse and seed. The 02D controller receives IMU observations, with no onset cue.')
    pair('02 · Small versus large disturbance','history-s1','fwd-.08','history-s1','fwd-.32',
         'One controller, two magnitudes. These are the first replicates in their respective held-out cells; reset and noise seeds differ.')
    pair('03 · Early versus later learning','early','fwd-.32','history-s1','fwd-.32',
         'Generation 1 versus the preselected final generation 12, on exactly the same held-out condition. Neither checkpoint was chosen for this outcome.')
    pair('04 · The fixed IMU trigger versus learned feedback','imu02c','fwd-.32','history-s1','fwd-.32',
         'The fixed trigger runs a single 0.4-second sequence. The new controller continuously recomputes its two motor coefficients.')
    pair('05 · Quiet standing','reflex','quiet','history-s1','quiet',
         'No physical disturbance. The timeline uses 2.5 seconds as a visual reference only; neither controller receives it.')
    pair('06 · A false alarm','imu02c','false-gyro','history-s1','false-gyro',
         'A 40 ms pitch-gyro transient without a physical push. Both controllers, including their standing reflex, receive the same corrupted reading.')
    pair('07 · At and beyond the old recovery boundary','history-s1','fwd-.32','history-s1','fwd-.36',
         'Fixed first replicates at 0.32 and 0.36 m/s. Outcome labels report the full six-second audit, including any failure omitted by a single upright frame.')
    pair('08 · Does observation history help?','memoryless-s1','fwd-.32','history-s1','fwd-.32',
         'Matched training episode budgets. The memoryless controller uses current IMU only; the other also receives its 80 ms exponential history.')
    pair('09 · Oracle onset versus autonomous sensing','oracle02c','fwd-.32','history-s1','fwd-.32',
         'The 02C control is explicitly told the push onset. The 02D controller is never given that information.')
    pair('10 · Early versus later quiet behavior','early','quiet','history-s1','quiet',
         'A matched quiet trial shows whether learning adds unnecessary movement. Added target offset alone is not displacement or energy.')
    pair('11 · Why the first pilot was rejected','failed-pilot','quiet','final-on-development','quiet',
         'The original pilot fell without a push in this development trial. It is shown because it triggered the curriculum revision. This reused development case is not part of final held-out confirmation.')
    pair('12 · Regression: a backward disturbance','reflex','back-.08','history-s1','back-.08',
         'The first backward-push replicate exposes a regression: the frozen reflex recovers and history seed 1 fails. This case was added after evaluation to make the failure visible.')
    pair('13 · Regression: quiet standing in seed 3','quiet-regression-reflex','quiet','quiet-regression-s3','quiet',
         'The first failed quiet trial of history seed 3, paired with the frozen reflex. Selected to display a held-out regression; seed 3 remains in every result table.')
    learned=[table['grid'][f'history-s{s}']['recovered'] for s in (1,2,3)]
    simple=[table['grid'][f'memoryless-s{s}']['recovered'] for s in (1,2,3)]
    strong=[table['grid'][f'{arm}-s{seed}']['groups']['fwd-.32'] for arm in ('history','memoryless') for seed in (1,2,3)]
    n=table['grid']['reflex']['n'];ref=table['grid']['reflex']['recovered']
    payload=dict(trials=data,presets=presets,
                 finding=f'Without an onset cue, the six learned controllers recovered {sum(x["recovered"] for x in strong)}/{sum(x["n"] for x in strong)} nominal 0.32 m/s trials (four shared conditions per controller). Across {n} held-out conditions, history seeds recovered {" / ".join(map(str,learned))}; memoryless seeds {" / ".join(map(str,simple))}; the frozen reflex {ref}. Quiet-standing and backward-push regressions remain visible below.',
                 stats=[dict(value=f'{ref}/{n}',label='Frozen reflex · held-out recovery'),
                        dict(value=' / '.join(map(str,learned)),label=f'History seeds 1 / 2 / 3 · out of {n} each'),
                        dict(value=' / '.join(map(str,simple)),label=f'Memoryless seeds 1 / 2 / 3 · out of {n} each'),
                        dict(value=str(sum(table['grid'][f'history-s{s}']['steps'] for s in (1,2,3))),label='Verified recovery steps · history controllers')],
                 provenance='Evidence: results/02d; model SHA256 ebb2ce87…; full conditions, checkpoints, validity rules and limitations accompany the repository report.')
    template=Path('movement_lab/replay02d.html').read_text()
    (root/'report/replay.html').write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'))),encoding='utf-8')
    (root/'report/contact-diagnostics.json').write_text(json.dumps(contacts,indent=2)+'\n')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,6,figsize=(13,5),sharex=True,sharey=True,constrained_layout=True)
    for row,name in enumerate(('reflex','history-s1')):
        d=data[index[(name,'fwd-.32')]]
        for col,delta in enumerate((0,.08,.16,.30,.50,1.0)):
            f=min(d['frames'],key=lambda f:abs(f['t']-d['event_time']-delta));ax=axes[row,col];positions=np.asarray(f['p'])
            for b,parent in d['edges']:ax.plot(positions[[b,parent],0],positions[[b,parent],2],c='#287d65',lw=2)
            for sole in f['soles_xz']:
                sole=np.asarray(sole);ax.fill(sole[:,0],sole[:,1],color='#287d65',alpha=.5)
            ax.scatter(f['head'][0],f['head'][2],s=100,facecolors='none',edgecolors='#287d65')
            ax.scatter(f['com'][0],f['com'][2],s=15,c='#c87838')
            ax.axhline(0,color='gray',lw=.7)
            ax.set(aspect='equal',xlim=(-100,220),ylim=(-10,270),title=f'+{delta:.2f} s',xlabel='x (mm)')
        axes[row,0].set_ylabel(('Frozen reflex' if row==0 else '02D history seed 1')+'\nz (mm)')
    fig.suptitle('First held-out 0.32 m/s replicate · measured states, identical external condition')
    fig.savefig(root/'report/filmstrip.png',dpi=150);plt.close(fig)
    return payload


def canon_condition(c):
    return json.loads(json.dumps(asdict(c)))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/02d'));a=p.parse_args()
    (a.root/'report').mkdir(exist_ok=True)
    table=tables(a.root)
    (a.root/'report/tables.json').write_text(json.dumps(table,indent=2)+'\n')
    figures(a.root);replay(a.root,table)


if __name__=='__main__':main()
