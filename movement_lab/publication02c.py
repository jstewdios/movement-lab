"""Build or verify a byte-addressed publication manifest for Study 02C."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def source_paths():
    return sorted([*Path('movement_lab').glob('*02c.py'), Path('tests/test_move02c.py'),
                   Path('movement_lab/sim02b.py'), Path('movement_lab/model02b.py'),
                   Path('movement_lab/physics02b.py'), Path('movement_lab/replay02b.py'),
                   Path('movement_lab/crosscheck02b.py'), Path('results/02b/model/model.json'),
                   Path('results/02b/runs/ars-push-s1/thetas.npz')])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    p.add_argument('--verify', action='store_true')
    p.add_argument('--full', action='store_true', help='also require the local development trajectories')
    a = p.parse_args()
    dest = a.root / 'manifest.json'
    if a.verify:
        manifest = json.loads(dest.read_text())
        errors = []
        for section in ('sources', 'published', 'local') if a.full else ('sources', 'published'):
            for name, entry in manifest[section].items():
                path = Path(name)
                if not path.exists() or digest(path) != entry['sha256']:
                    errors.append(name)
        print(json.dumps(dict(passed=not errors, mismatches=errors, full=a.full), indent=2))
        if errors:
            raise SystemExit(1)
        return
    manifest = dict(sources={}, published={}, local={})
    for path in source_paths():
        manifest['sources'][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    for path in sorted(a.root.rglob('*')):
        if not path.is_file() or path == dest or 'matplotlib' in path.parts or '__pycache__' in path.parts:
            continue
        ignored = subprocess.run(['git', 'check-ignore', '-q', str(path)]).returncode == 0
        group = 'local' if ignored else 'published'
        manifest[group][str(path)] = dict(sha256=digest(path), bytes=path.stat().st_size)
    dest.write_text(json.dumps(manifest, indent=2) + '\n')
    print({k: len(v) for k, v in manifest.items()})


if __name__ == '__main__':
    main()
