"""02B artifact manifest: hash every result file; mark published vs local-only.

    python -m movement_lab.publication02b --root results/02b          # write manifest.json
    python -m movement_lab.publication02b --root results/02b --verify # re-hash and compare

Local-only files (Git-ignored: compiled upstream model, logs) are hashed so a
later machine holding them can prove they are the originals; a fresh checkout
can verify only the published subset.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.dont_write_bytecode = True
SOURCES = ('model02b', 'sim02b', 'learn02b', 'eval02b', 'physics02b', 'revise02b', 'crosscheck02b',
           'analysis02b', 'replay02b', 'study02b', 'validate02b', 'publication02b')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ignored(paths):
    if not paths:
        return set()
    # POSIX separators: git does not match backslash paths against ignore rules.
    # Bytes, not text mode: Windows text pipes turn '\n' into '\r\n', which git
    # then treats as part of the path.
    r = subprocess.run(['git', 'check-ignore', '--stdin'], input=('\n'.join(Path(p).as_posix() for p in paths) + '\n').encode(),
                       capture_output=True)
    return {Path(line.strip()) for line in r.stdout.decode().splitlines() if line.strip()}


def inventory(root):
    files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'manifest.json')
    local = ignored(files)
    entries = {p.as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size, published=p not in local) for p in files}
    src = Path(__file__).parent
    sources = {f'movement_lab/{s}.py': sha(src / f'{s}.py') for s in SOURCES}
    sources['tests/test_react02b.py'] = sha(src.parent / 'tests/test_react02b.py')
    return dict(files=entries, sources=sources)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('results/02b'))
    p.add_argument('--verify', action='store_true')
    args = p.parse_args(argv)
    path = args.root / 'manifest.json'
    if not args.verify:
        inv = inventory(args.root)
        path.write_text(json.dumps(inv, indent=1))
        n_pub = sum(e['published'] for e in inv['files'].values())
        print(f"{len(inv['files'])} files ({n_pub} published, {len(inv['files']) - n_pub} local-only); {len(inv['sources'])} sources")
        return
    saved = json.loads(path.read_text())
    bad, missing = [], []
    for name, e in saved['files'].items():
        f = Path(name)
        if not f.exists():
            missing.append((name, e['published']))
        elif sha(f) != e['sha256']:
            bad.append(name)
    assert not bad, f'hash mismatch: {bad}'
    assert not [m for m, pub in missing if pub], f'missing published files: {missing}'
    print(f"verified {len(saved['files']) - len(missing)} files; {len(missing)} local-only files absent (not verifiable)")


if __name__ == '__main__':
    main()
