"""02B artifact manifest: hash every result file; mark published vs local-only.

    python -m movement_lab.publication02b --root results/02b          # write manifest.json
    python -m movement_lab.publication02b --root results/02b --verify # re-hash and compare

Run both from the repository root. Hashes are SHA-256 of the exact file bytes,
so ``sha256sum`` reproduces them. Published text is LF on every platform:
``.gitattributes`` pins ``eol=lf`` for the 02B files, the 02B writers emit LF,
and the manifest refuses to record published text containing CRLF (such a hash
would verify only on the machine that wrote it).

Local-only files (Git-ignored: compiled upstream model, logs, raw per-trial
tables) are hashed so a later machine holding them can prove they are the
originals; a fresh checkout can verify only the published subset.
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
TESTS = 'tests/test_react02b.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def is_text(data):
    """Git's heuristic: binary if a NUL byte appears in the first 8000 bytes."""
    return b'\0' not in data[:8000]


def crlf_text(paths):
    """Paths of text files that contain CRLF line endings."""
    out = []
    for p in paths:
        data = Path(p).read_bytes()
        if is_text(data) and b'\r\n' in data:
            out.append(Path(p))
    return out


def ignored(paths):
    if not paths:
        return set()
    # POSIX separators: git does not match backslash paths against ignore rules.
    # Bytes, not text mode: Windows text pipes turn '\n' into '\r\n', which git
    # then treats as part of the path.
    r = subprocess.run(['git', 'check-ignore', '--stdin'], input=('\n'.join(Path(p).as_posix() for p in paths) + '\n').encode(),
                       capture_output=True)
    return {Path(line.strip()) for line in r.stdout.decode().splitlines() if line.strip()}


def source_paths():
    src = Path(__file__).parent
    paths = {f'movement_lab/{s}.py': src / f'{s}.py' for s in SOURCES}
    paths[TESTS] = src.parent / TESTS
    return paths


def inventory(root):
    files = sorted(p for p in Path(root).rglob('*') if p.is_file() and p.name != 'manifest.json')
    local = ignored(files)
    sources = source_paths()
    bad = crlf_text([p for p in files if p not in local] + list(sources.values()))
    if bad:
        raise SystemExit('CRLF line endings in published text; their hashes would not verify on an LF checkout: '
                         + ', '.join(p.as_posix() for p in bad))
    entries = {p.as_posix(): dict(sha256=sha(p), bytes=p.stat().st_size, published=p not in local) for p in files}
    return dict(files=entries, sources={k: sha(v) for k, v in sources.items()})


def check(name, expected):
    """'ok' | 'missing' | 'crlf' (matches once CRLF -> LF: a CRLF checkout) | 'mismatch'."""
    f = Path(name)
    if not f.exists():
        return 'missing'
    data = f.read_bytes()
    if hashlib.sha256(data).hexdigest() == expected:
        return 'ok'
    if hashlib.sha256(data.replace(b'\r\n', b'\n')).hexdigest() == expected:
        return 'crlf'
    return 'mismatch'


def verify(saved):
    """Raise on any published/source mismatch; return (verified, local-only absent)."""
    status = {name: (check(name, e['sha256']), e['published']) for name, e in saved['files'].items()}
    status.update({name: (check(name, h), True) for name, h in saved['sources'].items()})
    crlf = sorted(n for n, (s, _) in status.items() if s == 'crlf')
    if crlf:
        raise SystemExit(f'{len(crlf)} files were checked out with CRLF line endings (a clone made before '
                         '.gitattributes pinned LF); delete them and run `git checkout -- <path>`: ' + ', '.join(crlf))
    bad = sorted(n for n, (s, _) in status.items() if s == 'mismatch')
    if bad:
        raise SystemExit(f'hash mismatch: {bad}')
    missing = sorted(n for n, (s, pub) in status.items() if s == 'missing' and pub)
    if missing:
        raise SystemExit(f'missing published files: {missing}')
    ok = sum(s == 'ok' for s, _ in status.values())
    return ok, sum(s == 'missing' for s, _ in status.values())


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('results/02b'))
    p.add_argument('--verify', action='store_true')
    args = p.parse_args(argv)
    path = args.root / 'manifest.json'
    if not args.verify:
        inv = inventory(args.root)
        path.write_text(json.dumps(inv, indent=1), newline='\n')
        n_pub = sum(e['published'] for e in inv['files'].values())
        print(f"{len(inv['files'])} files ({n_pub} published, {len(inv['files']) - n_pub} local-only); {len(inv['sources'])} sources")
        return
    ok, absent = verify(json.loads(path.read_text()))
    print(f"verified {ok} files (published results, local files present, sources); "
          f"{absent} local-only files absent (not verifiable)")


if __name__ == '__main__':
    main()
