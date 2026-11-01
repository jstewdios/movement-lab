"""Explicit 02A publication scope, immutable evidence, and local-only inventory."""
import argparse
import json
from pathlib import Path
from movement_lab.evaluate import sha
from movement_lab.validate02a import BASE, git

ROOT = Path('results/02a')
MANIFEST = ROOT/'publication/manifest.json'
REVIEW_MANIFEST = ROOT/'review-fixes/manifest.json'
ORIGINAL_PUBLICATION = '31b5ea29741314c075c184fe27e106c402be50a3'
SOURCES = {f'movement_lab/{n}.py' for n in ('explore02a','residual02a','study02a','report02a','validate02a','publication02a','historical02a','process02a')}
SOURCES |= {'tests/test_explore02a.py', 'tests/test_publication02a.py', 'tests/test_review02a.py'}


def allowed_publication_path(name):
    return name in SOURCES | {'.gitignore', 'README.md'} or name.startswith('results/02a/')


def check_scope(name):
    assert allowed_publication_path(name), ('outside publication scope', name)
    p = Path(name)
    assert p.is_file() and not p.is_symlink(), name
    assert p.stat().st_size < 5_000_000, ('unexpected large file', name)
    assert not any(n in p.parts for n in ('__pycache__', '.cache', 'matplotlib')), name
    assert p.suffix in ('.py', '.md', '.json', '.txt', '.png', '.html', '.npz') or name == '.gitignore', name
    assert not name.endswith(('trial.npz', '.mjb', '.log', '/atlas.png')), ('local-only artifact', name)
    if p.suffix == '.npz': assert p.name == 'initial.npz', name


def entry(name):
    return dict(bytes=Path(name).stat().st_size, sha256=sha(name))


def inventory():
    changed = git('diff', BASE, '--name-only').splitlines()
    untracked = git('ls-files', '--others', '--exclude-standard').splitlines()
    return sorted(set(changed+untracked)-{str(MANIFEST), str(REVIEW_MANIFEST)})


def main():
    p = argparse.ArgumentParser(); p.add_argument('--build', action='store_true'); p.add_argument('--build-review', action='store_true'); p.add_argument('--full', action='store_true'); args = p.parse_args()
    if args.build:
        assert git('branch', '--show-current') == 'study/02a-learning-to-explore'
        assert git('remote', 'get-url', 'origin') == 'git@github.com:jstewdios/movement-lab.git'
        audit = json.loads((ROOT/'publication/saved-record-validation-final.json').read_text())
        assert audit['proposal_replay'] and audit['upstream_unchanged']
        assert len(audit['trials']) == 5 and all(r['physical_valid'] for r in audit['trials'].values())
        names = inventory()
        for n in names: check_scope(n)
        ignored = git('ls-files', '--others', '--ignored', '--exclude-standard', '--', str(ROOT)).splitlines()
        ignored = [n for n in ignored if not any(k in Path(n).parts for k in ('__pycache__','matplotlib','.cache'))]
        result = dict(schema=1, base_commit=BASE, published_files={n:entry(n) for n in names},
                      local_only_files={n:entry(n) for n in ignored},
                      note='Original evidence inventories are unchanged and include local raw data. This inventory excludes itself. Missing raw files cannot be validated from hashes alone.')
        with MANIFEST.open('x') as f: f.write(json.dumps(result, indent=2)+'\n')
    m = json.loads(MANIFEST.read_text())
    if args.build_review:
        audit = json.loads((ROOT/'review-fixes/saved-record-validation.json').read_text())
        assert audit['proposal_replay'] and audit['upstream_unchanged']
        assert len(audit['trials']) == 5 and all(r['physical_valid'] for r in audit['trials'].values())
        assert git('branch', '--show-current') == 'study/02a-learning-to-explore'
        names = inventory()
        for n in names: check_scope(n)
        updates = {n: entry(n) for n in names if n not in m['published_files'] or entry(n) != m['published_files'][n]}
        # Historical artifacts can never be replaced by a review amendment.
        assert all(n not in m['published_files'] or n in SOURCES | {'.gitignore', 'README.md'} for n in updates)
        with REVIEW_MANIFEST.open('x') as f:
            f.write(json.dumps(dict(schema=1, original_commit=ORIGINAL_PUBLICATION,
                original_inventory=entry(str(MANIFEST)), published_files=updates), indent=2)+'\n')
    updates = {}
    if args.build_review or REVIEW_MANIFEST.exists():
        import subprocess
        assert MANIFEST.read_bytes() == subprocess.check_output(['git', 'show', ORIGINAL_PUBLICATION+':'+str(MANIFEST)]), 'original publication inventory changed'
    if REVIEW_MANIFEST.exists():
        import hashlib
        import subprocess
        review = json.loads(REVIEW_MANIFEST.read_text())
        assert review['original_commit'] == ORIGINAL_PUBLICATION
        assert entry(str(MANIFEST)) == review['original_inventory']
        updates = review['published_files']
        for n, value in updates.items():
            check_scope(n); assert entry(n) == value, n
            if n in m['published_files']:
                assert n in SOURCES | {'.gitignore', 'README.md'}, ('historical evidence replacement', n)
                old = subprocess.check_output(['git', 'show', ORIGINAL_PUBLICATION+':'+n])
                assert dict(bytes=len(old), sha256=hashlib.sha256(old).hexdigest()) == m['published_files'][n], n
    assert set(inventory()) == set(m['published_files']) | set(updates), 'publication inventory drift'
    for n, value in m['published_files'].items():
        if n in updates: continue
        check_scope(n); assert entry(n) == value, n
    if args.full:
        for n, value in m['local_only_files'].items(): assert entry(n) == value, n
    combined = m['published_files'] | updates
    print(json.dumps(dict(published_files_including_inventory=len(combined)+1+int(REVIEW_MANIFEST.exists()),
        published_bytes_including_inventory=sum(v['bytes'] for v in combined.values())+MANIFEST.stat().st_size+(REVIEW_MANIFEST.stat().st_size if REVIEW_MANIFEST.exists() else 0),
        local_only_files=len(m['local_only_files']), local_only_bytes=sum(v['bytes'] for v in m['local_only_files'].values()),
        local_files_verified=args.full), indent=2))

if __name__ == '__main__': main()
