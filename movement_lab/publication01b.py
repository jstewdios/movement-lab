"""Build or verify 01B publication hashes; --full requires preserved local evidence."""
import argparse
import json
from pathlib import Path
import subprocess
from movement_lab.evaluate import sha

ROOT=Path('results/01b')
MANIFEST=ROOT/'publication/manifest.json'
BASE='19cea89d66bb62594a0a4b51cdff640e23196f82'


def git(*args):
    return subprocess.check_output(['git','--no-optional-locks',*args],text=True).strip()


def full_audit():
    counts={}
    for p,base in ((ROOT/'evidence-manifest.json',Path('.')),
                   (ROOT/'landing-diagnosis/diagnostic-manifest.json',Path('.')),
                   (ROOT/'departures/study-manifest.json',ROOT/'departures')):
        entries=json.loads(p.read_text())
        for name,digest in entries.items():assert sha(base/name)==digest,str(base/name)
        counts[str(p)]=len(entries)
    prov=json.loads((ROOT/'gpu-gate/provenance.json').read_text());upstream=Path(prov['upstream'])
    for name,digest in prov['upstream_file_sha256'].items():assert sha(upstream/name)==digest,name
    assert git('-C',str(upstream),'rev-parse','HEAD')==prov['upstream_commit']
    assert git('-C',str(upstream),'status','--porcelain')==prov['upstream_status']==''
    changed=git('diff',BASE,'--name-only','--','results/00b','results/00c','results/01a').splitlines()
    assert not changed,changed
    return dict(historical_manifest_files=counts,upstream_files_unchanged=len(prov['upstream_file_sha256']))


def allowed_publication_path(name):
    # The historical 01B diagnostic module predates the *01b.py naming convention.
    return (name in ('.gitignore','README.md','movement_lab/landing_diagnostics.py')
        or name.startswith('results/01b/')
        or (name.startswith(('movement_lab/','tests/')) and name.endswith('01b.py')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',action='store_true');p.add_argument('--full',action='store_true');args=p.parse_args()
    if args.build:
        audited=full_audit()
        names=git('ls-files','--modified','--others','--exclude-standard').splitlines()
        files=sorted({n for n in names if n!=str(MANIFEST)})
        for name in files:
            assert allowed_publication_path(name),(name,'outside publication scope')
            assert Path(name).stat().st_size<5_000_000,(name,'unexpected large publication file')
            assert not name.endswith(('trial.npz','.mjb','.log')),(name,'local-only artifact')
        ignored=git('ls-files','--others','--ignored','--exclude-standard','--','results/01b').splitlines()
        ignored += [str(f) for f in Path('results').glob('01b*.log')]
        ignored=sorted({n for n in ignored if not any(k in Path(n).parts for k in ('matplotlib','cache','__pycache__'))})
        entry=lambda n:dict(bytes=Path(n).stat().st_size,sha256=sha(n))
        manifest=dict(schema=1,study='01B completed research study',base_commit=BASE,
            published_files={n:entry(n) for n in files},local_only_files={n:entry(n) for n in ignored},
            full_local_audit=audited,note='Manifest excludes itself. Historical manifests retain their original full-local evidence scope; missing local files are not validated in a fresh checkout.')
        MANIFEST.write_text(json.dumps(manifest,indent=2)+'\n')
    manifest=json.loads(MANIFEST.read_text())
    for name,entry in manifest['published_files'].items():
        assert Path(name).stat().st_size==entry['bytes'] and sha(name)==entry['sha256'],name
    if args.full:
        for name,entry in manifest['local_only_files'].items():
            assert Path(name).stat().st_size==entry['bytes'] and sha(name)==entry['sha256'],name
        full_audit()
    print(json.dumps(dict(published_files=len(manifest['published_files']),published_bytes=sum(v['bytes'] for v in manifest['published_files'].values()),
        local_files=len(manifest['local_only_files']),local_bytes=sum(v['bytes'] for v in manifest['local_only_files'].values()),full_local_evidence_verified=args.full or args.build),indent=2))


if __name__=='__main__':main()
