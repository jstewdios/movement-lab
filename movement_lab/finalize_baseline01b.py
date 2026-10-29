"""Recover the completed baseline's postprocessing failure; no simulation."""
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from movement_lab.controlled01b import gate,publish
from movement_lab.departure_metrics01b import metrics
from movement_lab.evaluate import sha


def main():
    out=Path('results/01b/departures/baseline');trial=out/'zero'
    old=trial/'metadata.json';archived=trial/'metadata-before-finalize.json'
    if archived.exists(): assert archived.read_bytes()==old.read_bytes()
    else: archived.write_bytes(old.read_bytes())
    m=json.loads(archived.read_text())
    with np.load(trial/'trial.npz') as z:
        a=dict(z);assert len(a['time_s'])==1200
        m['gate']=gate(a,m['stop_reasons'],True);m['departure_metrics']=metrics(a)
    assert m['gate']['pass'] and m['departure_metrics']['recovered']
    m['postprocessing_recovery']='Completed trajectory retained; metrics function shadowed by local dict at finalization. Metadata finalized offline without reintegration.'
    old.write_text(json.dumps(m,indent=2)+'\n')
    previous=json.loads(Path('results/01b/landing-diagnosis/clearance-1mm/provenance.json').read_text())
    upstream=Path(previous['upstream'])
    for f,digest in previous['upstream_file_sha256'].items():assert sha(upstream/f)==digest
    for n,v in previous['versions'].items():assert importlib.metadata.version(n)==v
    assert sha(previous['checkpoint'])==previous['checkpoint_sha256']
    assert (out/'resolved_config.txt').read_bytes()==Path('results/01b/landing-diagnosis/clearance-1mm/resolved_config.txt').read_bytes()
    provenance={k:previous[k] for k in ('checkpoint','checkpoint_sha256','schema','upstream','upstream_commit','upstream_status','task','model','versions','upstream_file_sha256')}
    git=lambda *args:subprocess.check_output(['git','--no-optional-locks','-C',str(upstream),*args],text=True).strip()
    assert git('rev-parse','HEAD')==previous['upstream_commit'] and git('status','--porcelain')==previous['upstream_status']
    provenance.update(python=sys.version,gpu='NVIDIA GeForce RTX 2070 SUPER (captured in evaluation.log)',
        source_sha256={n:sha(out/n) for n in ('departure01b.py','departure_metrics01b.py','controlled01b.py','discovery_metrics.py','resolved_config.txt','replay01a.py','replay_model.mjb')},
        wall_seconds=None,trial_wall_seconds=m['wall_seconds'],
        postprocessing_recovery='Reconstructed offline: upstream file hashes, checkpoint hash, config and installed versions verified against prior landing provenance. Total runner wall time was not saved; trial wall time is available. Original executed source snapshot is retained.',
        measurement='physics endpoint forward/sense; CPU contact audit is geometry-only')
    for name in ('finalize_baseline01b.py','departure_metrics01b.py'):
        dest='finalized_'+name
        (out/dest).write_bytes(Path(__file__).with_name(name).read_bytes())
    provenance['postprocess_source_sha256']={n:sha(out/n) for n in ('finalized_finalize_baseline01b.py','finalized_departure_metrics01b.py')}
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    publish(out)


if __name__=='__main__':main()
