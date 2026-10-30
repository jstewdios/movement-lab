"""Build/verify the 01C publication inventory without changing original evidence."""
import argparse
import hashlib
import subprocess
import json
from pathlib import Path
from movement_lab.evaluate import sha
from movement_lab.validate01c import BASE, git

ROOT = Path('results/01c')
MANIFEST = ROOT/'publication/manifest.json'
SOURCES = {
    'movement_lab/diagnose01b_for01c.py', 'movement_lab/report01c.py',
    'movement_lab/study01c.py', 'movement_lab/transition01c.py',
    'movement_lab/write_report01c.py', 'movement_lab/validate01c.py',
    'movement_lab/publication01c.py', 'tests/test_transition01c.py',
    'tests/test_publication01c.py',
}


def allowed_publication_path(name):
    return name in SOURCES | {'.gitignore', 'README.md'} or name.startswith('results/01c/')


def full_audit():
    counts = {}
    for path in (ROOT/'manifest.json', ROOT/'pilot-gpu/manifest.json'):
        entries = json.loads(path.read_text())
        for name, digest in entries.items():
            assert sha(path.parent/name) == digest, name
        counts[str(path)] = len(entries)
    prov = json.loads((ROOT/'pilot-gpu/baseline/provenance.json').read_text())
    up = Path(prov['upstream'])
    assert git('-C', str(up), 'rev-parse', 'HEAD') == prov['upstream_commit']
    assert git('-C', str(up), 'status', '--porcelain') == prov['upstream_status'] == ''
    for name, digest in prov['upstream_file_sha256'].items():
        assert sha(up/name) == digest, name
    # 01B inventories also froze shared docs; verify their old bytes are intact
    # as prefixes, while every scientific and local-only file remains exact.
    old = json.loads(Path('results/01b/publication/manifest.json').read_text())
    for name, value in old['published_files'].items():
        if name in ('.gitignore', 'README.md'):
            previous = subprocess.check_output(['git', 'show', BASE+':'+name])
            assert hashlib.sha256(previous).hexdigest() == value['sha256']
            assert Path(name).read_bytes().startswith(previous), name
        else:
            assert entry(name) == value, name
    for name, value in old['local_only_files'].items():
        assert entry(name) == value, name
    from movement_lab.publication01b import full_audit as prior_full_audit
    prior_full_audit()
    # Include all tracked study inputs/results and existing code, not only result dirs.
    changed = git('diff', BASE, '--name-only').splitlines()
    for name in changed:
        assert allowed_publication_path(name), (name, 'earlier evidence or unrelated change')
    return dict(original_manifest_entries=counts, upstream_unchanged=True,
                earlier_tracked_evidence_unchanged=True)


def entry(name):
    return dict(bytes=Path(name).stat().st_size, sha256=sha(name))


def check_scope(name):
    assert allowed_publication_path(name), (name, 'outside 01C publication scope')
    assert Path(name).stat().st_size < 5_000_000, (name, 'unexpected large file')
    assert not name.endswith(('trial.npz', '.mjb', '.log')), (name, 'local-only artifact')
    assert not any(n in Path(name).parts for n in ('__pycache__', 'matplotlib', '.cache'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build', action='store_true')
    p.add_argument('--full', action='store_true')
    args = p.parse_args()
    if args.build:
        assert git('branch', '--show-current') == 'study/01c-controlled-transitions'
        audited = full_audit()
        names = git('diff', BASE, '--name-only').splitlines()
        names += git('ls-files', '--others', '--exclude-standard').splitlines()
        files = sorted(set(names) - {str(MANIFEST)})
        for name in files:
            check_scope(name)
        ignored = git('ls-files', '--others', '--ignored', '--exclude-standard', '--', str(ROOT)).splitlines()
        ignored = sorted(n for n in ignored if not any(k in Path(n).parts for k in ('matplotlib', 'cache', '__pycache__')))
        manifest = dict(schema=1, study='01C completed bounded pilot', base_commit=BASE,
            published_files={n: entry(n) for n in files},
            local_only_files={n: entry(n) for n in ignored}, full_local_audit=audited,
            note='Inventory excludes itself. Original manifests remain unchanged and cover full local evidence; missing data is not validated in a checkout.')
        MANIFEST.write_text(json.dumps(manifest, indent=2)+'\n')
    manifest = json.loads(MANIFEST.read_text())
    for name, value in manifest['published_files'].items():
        check_scope(name)
        assert entry(name) == value, name
    if args.full:
        for name, value in manifest['local_only_files'].items():
            assert entry(name) == value, name
        full_audit()
    print(json.dumps(dict(published_files_excluding_inventory=len(manifest['published_files']),
        published_bytes_excluding_inventory=sum(v['bytes'] for v in manifest['published_files'].values()),
        local_only_files=len(manifest['local_only_files']),
        local_only_bytes=sum(v['bytes'] for v in manifest['local_only_files'].values()),
        full_local_evidence_verified=args.full or args.build), indent=2))


if __name__ == '__main__':
    main()
