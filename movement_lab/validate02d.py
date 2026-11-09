"""Read-only evidence audit, optional resimulation, and publication hashes."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import numpy as np
from movement_lab.learn02c import digest, completed_checkpoints, update_distribution
from movement_lab.learn02d import conditions
from movement_lab.eval02c import read_rows, audit_record
from movement_lab.control02d import Design, OPTION
from movement_lab.sim02d import Condition, event_time
from movement_lab.sim02c import BASE
from movement_lab.sim02b import DEFAULT_MODEL, Sim


def canon(x):return json.loads(json.dumps(x))


def audit(root, resimulate=0):
    count=0
    def check(ok,label):
        nonlocal count
        count+=1
        if not ok:raise AssertionError(label)
    pre=json.loads((root/'preflight.json').read_text())
    check(pre['passed'] and pre['outcome_matches']==120,'historical reproduction')
    check(pre['base_sha256']==digest(BASE),'frozen standing reflex')
    check(pre['option_sha256']==digest(OPTION),'frozen motor vocabulary')
    from movement_lab.study02d import DEPENDENCIES
    dependencies=json.loads((root/'dependencies.json').read_text())
    check(set(dependencies)==set(DEPENDENCIES),'complete frozen helper inventory')
    for name,sha in dependencies.items():
        check(digest(Path('movement_lab')/name)==sha,'frozen helper '+name)
    training=set();progress={}
    paths=sorted((root/'pilot').glob('history-s*'))+sorted((root/'runs').glob('*-s*'))
    expected_names={'history-s71','history-s72',*[f'{a}-s{s}' for a in ('history','memoryless') for s in (1,2,3)]}
    check({p.name for p in paths}==expected_names,'all eight declared learning runs, no extra run selection')
    for path in paths:
        cfg=json.loads((path/'config.json').read_text())
        expected=(8,24,8) if path.name=='history-s71' else (8,16,12) if path.name=='history-s72' else (12,16,12)
        check((cfg['iterations'],cfg['population'],cfg['conditions_per_generation'])==expected,'declared budget '+path.name)
        revision=cfg.get('revision',False)
        for name,sha in cfg['sources'].items():
            source=root/'pilot/source-v1'/name if path.name=='history-s71' and (root/'pilot/source-v1'/name).exists() else Path('movement_lab')/name
            check(digest(source)==sha,f'{path.name} source {name}')
        check(cfg['base_sha256']==digest(BASE) and cfg['option_sha256']==digest(OPTION),'checkpoint inputs')
        design=Design(**cfg['design']);mean=np.zeros(design.dim);std=np.full(design.dim,cfg['initial_std'])
        std.reshape(2,-1)[:,-1]=cfg['initial_std']/4;best=mean.copy();rng=np.random.default_rng(cfg['seed'])
        checkpoints=completed_checkpoints(path)
        check(len(checkpoints)==cfg['iterations'],'complete declared generations')
        for g,p in enumerate(checkpoints):
            z=np.load(p);state=json.loads(str(z['state']));row=json.loads(str(z['row']))
            cands=np.clip(mean+std*rng.standard_normal((cfg['population'],design.dim)),-2,2)
            cands[0],cands[1],cands[2]=best,mean,0.
            check(np.array_equal(cands,z['candidates']),f'{path.name}/{g} candidate RNG')
            cs=conditions(cfg['seed'],g,revision)
            check(json.loads(str(z['conditions']))==canon([asdict(c) for c in cs]),'curriculum reconstruction')
            training.update(c.seed for c in cs)
            scores=z['outcomes'][:,:,0].mean(axis=1)
            check(np.array_equal(scores,z['scores']),'scores from saved outcomes')
            best=cands[np.argmax(scores)].copy()
            mean,std=update_distribution(mean,std,cands,scores,cfg['elite'],cfg['floor'])
            for k,v in [('mean',mean),('std',std),('best',best)]:check(np.array_equal(v,z[k]),'CEM '+k)
            check(state['rng']==rng.bit_generator.state,'RNG state')
            check(state['trials']==(g+1)*cfg['population']*len(cs),'episode count')
        check(np.array_equal(np.load(path/'final.npz')['best'],best),'last training winner retained')
        progress[path.name]=state['trials']
    # Matched physical trial budgets and identical training conditions across arms.
    for seed in (1,2,3):
        for a,b in zip(completed_checkpoints(root/'runs'/f'history-s{seed}'),completed_checkpoints(root/'runs'/f'memoryless-s{seed}')):
            x,y=np.load(a),np.load(b)
            check(str(x['conditions'])==str(y['conditions']),'paired training conditions')
            check(x['outcomes'].shape[:2]==y['outcomes'].shape[:2],'matched episode budgets')
    development=set();held=set();allrows={}
    for group in ('pilot','pilot-revised','grid','shifts'):
        folder=root/'eval'/group
        if not folder.exists():continue
        identity=json.loads((folder/'identity.json').read_text())
        for name,sha in identity['sources'].items():
            archive='source-v1' if group=='pilot' else 'source-v2'
            archived=root/'pilot'/archive/name
            source=archived if group.startswith('pilot') and archived.exists() else Path('movement_lab')/name
            check(digest(source)==sha,f'{group} evaluation source {name}')
        ref=None
        for path in sorted(folder.glob('*.jsonl.gz')):
            rows=read_rows(path);allrows[(group,path.name[:-9])]=rows
            check(len(rows)==len(identity['conditions']),'complete evaluation rows')
            cells=[dict(label=r['set'],condition=r['condition']) for r in rows]
            check(cells==identity['conditions'],'exact declared evaluation conditions')
            for r in rows:
                (development if group.startswith('pilot') else held).add(r['condition']['seed'])
                if r['recovered_200hz']:
                    check(r['physical_valid'] and r['upright_200hz'] and r['quiet_supported_final_second'] and r['settled_200hz'],'valid sustained recovery')
                if r['recovery_with_step']:
                    check(r['recovered_200hz'] and any(e['qualifies'] for e in r['support_events']),'step requires valid whole trial')
    check(not(training&development or training&held or development&held),'seed separation')
    for g in ('grid','shifts'):
        if not(root/'eval'/g).exists():continue
        for name in ('reflex','oracle02c','imu02c',*[f'{a}-s{s}' for a in ('history','memoryless') for s in (1,2,3)]):
            check((g,name) in allrows,'required controller '+name)
    sim=None
    if (DEFAULT_MODEL/'model.mjb').exists():
        sim=Sim();check(digest(DEFAULT_MODEL/'model.mjb')==pre['model_sha256'],'model hash')
    if sim is None and any((root/'records').glob('*.npz')):
        raise SystemExit('Full physical record validation requires the matching local model; use --verify for portable hash verification.')
    for path in sorted((root/'records').glob('*.npz')):
        meta=json.loads(path.with_suffix('.json').read_text());r=dict(np.load(path));c=Condition(**meta['condition'])
        from dataclasses import replace
        result=audit_record(r,replace(c,push_time_s=event_time(c)),meta['z0'],sim)
        if sim is not None:
            result['physical_valid'] &= (not np.any(r['self_contacts']) and np.abs(r['torque_nm']).max()<=c.vin*sim.kt_over_r+1e-9
                and np.all(r['target_cmd']>=sim.lo-1e-12) and np.all(r['target_cmd']<=sim.hi+1e-12))
            result['recovered_200hz'] &= result['physical_valid']
            result['recovery_with_step'] &= result['recovered_200hz']
        for key in ('physical_valid','recovered_200hz','recovery_with_step','support_events'):
            check(result[key]==meta['summary'][key],path.stem+' '+key)
        check(np.allclose(np.diff(r['time_s']),.005,atol=1e-12),'200 Hz record')
        check(r['time_s'][-1]>=c.duration_s-.0025,'complete trajectory')
    resimulated=[]
    if resimulate:
        from movement_lab.eval02d import init_worker,evaluate,controllers
        init_worker()
        cs=controllers(root)
        for name,vec,design,mode in cs[:resimulate]:
            row=next(r for r in allrows[('grid',name)] if r['set']=='fwd-.32')
            fresh=evaluate((name,vec,design,mode,row['set'],Condition(**row['condition'])))
            check(canon(fresh)==row,'exact resimulation '+name);resimulated.append(name)
    return dict(passed=True,checks=count,training_episode_counts=progress,resimulated=resimulated,
                raw_records=len(list((root/'records').glob('*.npz'))),model_available=sim is not None)


def manifest(root):
    files={}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name=='manifest.json' or p.suffix=='.log' or any(x in p.parts for x in ('site-preview','__pycache__','matplotlib')):continue
        files[str(p)]=digest(p)
    for p in sorted(Path('movement_lab').glob('*02d*')):
        if p.is_file():files[str(p)]=digest(p)
    files['tests/test_adapt02d.py']=digest('tests/test_adapt02d.py')
    for p in (BASE,OPTION,Path('movement_lab/sim02b.py'),Path('movement_lab/sim02c.py'),Path('movement_lab/eval02c.py')):files[str(p)]=digest(p)
    for name in json.loads((root/'dependencies.json').read_text()):
        p=Path('movement_lab')/name;files[str(p)]=digest(p)
    return dict(files=files,local_model=dict(path=str(DEFAULT_MODEL/'model.mjb'),sha256=json.loads((root/'preflight.json').read_text())['model_sha256']))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('results/02d'))
    p.add_argument('--resimulate',type=int,default=0);p.add_argument('--publish',action='store_true');p.add_argument('--verify',action='store_true');a=p.parse_args()
    if a.verify:
        data=json.loads((a.root/'manifest.json').read_text())
        for path,sha in data['files'].items():
            if digest(path)!=sha:raise SystemExit('hash mismatch: '+path)
        print('verified',len(data['files']),'published hashes');return
    result=audit(a.root,a.resimulate)
    (a.root/'validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if a.publish:(a.root/'manifest.json').write_text(json.dumps(manifest(a.root),indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
