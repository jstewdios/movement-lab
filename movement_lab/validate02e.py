"""02E evidence audit, exact resimulation and publication manifest. Read-only on all evidence.

The audit writes only ``validation.json`` under ``--root``; gates are recomputed on a scratch
copy of their input rows, so neither the audited tree nor any other evidence tree is written.

    python -m movement_lab.validate02e --root results/02e --audit [--resimulate N]   # writes validation.json
    python -m movement_lab.validate02e --root results/02e --publish                  # writes manifest.json
    python -m movement_lab.validate02e --root results/02e --verify                   # portable hash check
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import numpy as np
from movement_lab.learn02c import digest, completed_checkpoints, update_distribution
from movement_lab.eval02c import read_rows

canon = lambda x: json.loads(json.dumps(x))
REUSED = ('sim02b.py', 'sim02c.py', 'sim02d.py', 'control02d.py', 'eval02c.py', 'eval02d.py', 'physics02b.py',
          'learn02c.py', 'learn02d.py', 'model02b.py', 'validate02d.py', 'preflight02d.py', 'replay02c.py')
HISTORICAL_INPUTS = ('results/02b/model/model.json', 'results/02b/runs/ars-push-s1/thetas.npz', 'results/02c/runs/cem-s1/final.npz',
                     'results/02d/pilot/history-s72/final.npz', 'results/02d/pilot/history-s72/generation-001.npz',
                     *[f'results/02d/runs/{a}-s{s}/final.npz' for a in ('history', 'memoryless') for s in (1, 2, 3)])


class Audit:
    def __init__(self):
        self.count = 0

    def __call__(self, ok, label):
        self.count += 1
        if not ok:
            raise AssertionError(label)


def audit_margin_correction(root, check):
    """The declared sim02e margin-projection correction (correct02e), if present.

    Verifies the archived pre-correction source and the declared edit, that exactly the sloped
    rows of every margin-recording battery were regenerated, and rebuilds every corrected file's
    original rows byte for byte from the record. Returns the superseded sim02e.py hashes and the
    pre-correction rows per corrected file."""
    from movement_lab import correct02e as C
    path = root / C.RECORD / 'correction.json'
    if not path.exists():
        return set(), {}
    rec = json.loads(path.read_text())
    archived = root / rec['archived_source']
    check(digest(archived) == rec['before_sha256'] == C.BEFORE_SHA256, 'archived pre-correction sim02e.py')
    check([tuple(e) for e in rec['edits']] == list(C.EDITS), 'declared margin-projection edits')
    check(C.apply_edits(archived.read_text()) == C.SOURCE.read_text(), 'sim02e.py differs from the archived source only by the declared edits')
    check(digest(C.SOURCE) == rec['after_sha256'], 'corrected sim02e.py')
    by_file = {}
    for c in read_rows(root / C.RECORD / 'rows.jsonl.gz'):
        by_file.setdefault((c['group'], c['file']), []).append(c)
    check(sorted(rec['groups']) == sorted(C.GROUPS), 'every margin-recording battery corrected')
    before_rows = {}
    for group, entry in rec['groups'].items():
        ident = json.loads((root / group / 'identity.json').read_text())
        check(digest(root / group / 'identity.json') == entry['identity_after_sha256'] and ident['sources']['sim02e.py'] == rec['after_sha256'],
              f'{group} identity pins the corrected source')
        check(sorted(entry['files']) == sorted(c['file'] for c in ident['configurations']), f'{group}: every file corrected')
        for name, f in entry['files'].items():
            p = root / group / f'{name}.jsonl.gz'
            check(digest(p) == f['after_sha256'], f'{group}/{name} is the corrected file')
            rows = read_rows(p)
            cs = by_file.get((group, name), [])
            check([c['index'] for c in cs] == [i for i, r in enumerate(rows) if C.sloped(r)], f'{group}/{name}: exactly the sloped rows regenerated')
            check(len(cs) == f['regenerated'] and sum(c['before'] != c['after'] for c in cs) == f['changed'], f'{group}/{name} correction counts')
            old = list(rows)
            for c in cs:
                r = rows[c['index']]
                check((r['setting'], r['set'], r['condition']['seed']) == (c['setting'], c['set'], c['seed'])
                      and set(C.MARGIN_FIELDS) <= set(r) and C.margins_of(r) == c['after'], f'{group}/{name} row {c["index"]} corrected margins')
                old[c['index']] = {**r, **c['before']}
            check(C.sha(C.rows_text(rows)) == f['after_rows_sha256'], f'{group}/{name} corrected rows')
            check(C.sha(C.rows_text(old)) == f['before_rows_sha256'], f'{group}/{name}: original rows rebuilt byte for byte (only margins changed)')
            before_rows[(group, name)] = old
    sloped_records = sorted(p.stem for p in (root / 'records').glob('*.json') if C.sloped(json.loads(p.read_text()))) \
        if (root / 'records').exists() else []
    check(sorted(rec['replay_records']) == sloped_records, 'every sloped replay record corrected')
    for key, r in rec['replay_records'].items():
        meta = root / 'records' / f'{key}.json'
        check(digest(meta) == r['after_sha256'] and C.margins_of(json.loads(meta.read_text())['summary']) == by_file[('records', key)][0]['after'],
              f'replay record {key} corrected')
    check(sum(f['changed'] for g in rec['groups'].values() for f in g['files'].values()) > 0, 'the correction changed sloped margins')
    return {rec['before_sha256']}, before_rows


def audit_learning(root, check, superseded=frozenset()):
    """Training configurations may pin a superseded sim02e.py: learning rollouts never compute margins."""
    from movement_lab.learn02e import conditions
    from movement_lab.control02d import Design
    seeds, progress = set(), {}
    runs = sorted((root / 'learning').glob('*-s72')) + sorted((root / 'runs').glob('*-rand-s*') if (root / 'runs').exists() else [])
    for path in runs:
        cfg = json.loads((path / 'config.json').read_text())
        expected = (8, 16, 4, 12) if path.parent.name == 'learning' else (12, 16, 4, 12)
        check((cfg['iterations'], cfg['population'], cfg['elite'], cfg['conditions_per_generation']) == expected, 'declared budget ' + path.name)
        check(cfg['randomize_deg'] == (0. if path.name == 'memoryless-s72' else 4.), 'declared randomization ' + path.name)
        for name, sha in cfg['sources'].items():
            old = name == 'sim02e.py' and sha in superseded and 'margins' not in Path('movement_lab/learn02e.py').read_text()
            check(digest(Path('movement_lab') / name) == sha or old, f'{path.name} source {name}')
        design = Design(**cfg['design'])
        mean = np.zeros(design.dim); std = np.full(design.dim, cfg['initial_std']); std.reshape(2, -1)[:, -1] = cfg['initial_std'] / 4
        best = mean.copy(); rng = np.random.default_rng(cfg['seed'])
        cps = completed_checkpoints(path)
        check(len(cps) == cfg['iterations'], 'complete generations ' + path.name)
        for g, p in enumerate(cps):
            z = np.load(p); state = json.loads(str(z['state']))
            cands = np.clip(mean + std * rng.standard_normal((cfg['population'], design.dim)), -2, 2)
            cands[0], cands[1], cands[2] = best, mean, 0.
            check(np.array_equal(cands, z['candidates']), f'{path.name}/{g} candidate RNG')
            cs = conditions(cfg['seed'], g, cfg['revision'], cfg['randomize_deg'])
            check(json.loads(str(z['conditions'])) == canon([asdict(c) for c in cs]), 'curriculum reconstruction')
            seeds.update(c.seed for c in cs)
            scores = z['outcomes'][:, :, 0].mean(axis=1)
            check(np.array_equal(scores, z['scores']), 'scores from outcomes')
            best = cands[np.argmax(scores)].copy()
            mean, std = update_distribution(mean, std, cands, scores, cfg['elite'], cfg['floor'])
            for k, v in (('mean', mean), ('std', std), ('best', best)):
                check(np.array_equal(v, z[k]), 'CEM ' + k)
            check(state['rng'] == rng.bit_generator.state, 'RNG state')
            check(state['trials'] == (g + 1) * cfg['population'] * len(cs), 'episode count')
        check(np.array_equal(np.load(path / 'final.npz')['best'], best), 'last training winner retained ' + path.name)
        progress[path.name] = state['trials']
    return seeds, progress


def expected_battery(group):
    from movement_lab import diag02e as D
    if group.endswith('characterize') or group.endswith('characterize-interrupted-v1'):
        return D.battery(D.SETTINGS_1A, D.KINDS, 7_000_000_000)
    if group.startswith('pilot'):
        return D.battery(D.SETTINGS_D, D.KINDS, 7_200_000_000)
    return D.battery(D.SETTINGS_H, D.KINDS_H, 8_000_000_000, reps=3, onset=(2.2, 3.4))


def audit_batteries(root, check, before_rows=None):
    from movement_lab.diag02e import SOURCES
    from movement_lab.correct02e import rows_text
    seeds, rows_by = set(), {}
    groups = ['diagnostic/characterize', 'diagnostic/characterize-interrupted-v1', 'pilot/frontend', 'pilot/learning', 'confirm']
    for group in groups:
        folder = root / group
        if not folder.exists():
            continue
        identity = json.loads((folder / 'identity.json').read_text())
        snapshot = folder / 'source'
        for name, sha in identity['sources'].items():
            src = snapshot / name if (snapshot / name).exists() else Path('movement_lab') / name
            check(digest(src) == sha, f'{group} source {name}')
        conds = expected_battery(group)
        check(identity['conditions'] == canon([dict(setting=s, kind=k, condition=asdict(c)) for s, k, c in conds]), f'{group} declared conditions')
        for cfg in identity['configurations']:
            path = folder / f"{cfg['file']}.jsonl.gz"
            if not path.exists():
                check(group == 'diagnostic/characterize-interrupted-v1', f'{group} complete configurations')
                continue
            rows = read_rows(path)
            rows_by[(group, cfg['file'])] = rows
            check(len(rows) == len(conds), f'{group}/{cfg["file"]} complete rows')
            check([dict(setting=r['setting'], kind=r['set'], condition=r['condition']) for r in rows] == identity['conditions'], 'row order')
            for r in rows:
                seeds.add(r['condition']['seed'])
                if r['recovered_200hz']:
                    check(r['physical_valid'] and r['upright_200hz'] and r['quiet_supported_final_second'] and r['settled_200hz'], 'valid sustained recovery')
                if r['recovery_with_step']:
                    check(r['recovered_200hz'] and any(e['qualifies'] for e in r['support_events']), 'step requires valid whole trial')
    old = root / 'diagnostic/characterize-interrupted-v1'
    for p in sorted(old.glob('*.jsonl.gz')) if old.exists() else []:
        # The interrupted launch predates the margin-projection correction: compare with the rebuilt original rows.
        new = root / 'diagnostic/characterize' / p.name
        before = (before_rows or {}).get(('diagnostic/characterize', p.name[:-9]))
        expected = rows_text(before) if before is not None else gzip.decompress(new.read_bytes()) if new.exists() else None
        check(expected == gzip.decompress(p.read_bytes()), 'vectorized margins reproduce interrupted rows byte-for-byte: ' + p.name)
    return seeds, rows_by


def audit(root, resimulate=0):
    check = Audit()
    gate0 = json.loads((root / 'phase0/gate.json').read_text())
    check(gate0['passed'], 'phase 0 gate')
    from movement_lab.repro02e import summarize_resimulation
    res = read_rows(root / 'phase0/resimulation.jsonl.gz')
    saved = json.loads((root / 'phase0/resimulation-summary.json').read_text())
    check({k: v for k, v in summarize_resimulation(res).items()} == {k: v for k, v in saved.items() if k != 'run'}, 'resimulation summary from rows')
    check(json.loads((root / 'learning/reproduce.json').read_text())['passed'], 'learn02e reproduces learn02d')
    superseded, before_rows = audit_margin_correction(root, check)
    train_seeds, progress = audit_learning(root, check, superseded)
    eval_seeds, rows = audit_batteries(root, check, before_rows)
    est = read_rows(root / 'diagnostic/estimators.jsonl.gz')
    eval_seeds |= {r['seed'] for r in est}
    check(not (train_seeds & eval_seeds), 'training/evaluation seed separation')
    check(all(4_000_000_000 <= s < 5_000_000_000 for s in train_seeds), 'training seeds in the 02D curriculum range')
    check(all(7_000_000_000 <= s < 9_000_000_000 for s in eval_seeds), '02E evaluation seeds disjoint from 02C/02D ranges')
    audit_gates(root, check)
    from movement_lab import diag02e as D
    resimulated = []
    if resimulate:
        from movement_lab.sim02e import Condition
        D.init_worker()
        picked = []
        for (group, name), rs in sorted(rows.items()):
            if group in ('confirm', 'pilot/frontend'):
                picked.append((group, name, rs[len(rs) // 2]))
        picked = picked[:resimulate]
        # Plus one sloped row per margin-recording battery: corrected support margins must reproduce exactly.
        from movement_lab.correct02e import GROUPS, sloped
        for group in GROUPS:
            files = [(n, rs) for (g, n), rs in sorted(rows.items()) if g == group]
            if files:
                name, rs = files[len(files) // 2]
                sl = [r for r in rs if sloped(r)]
                picked.append((group, name, sl[len(sl) // 2]))
        for group, name, row in picked:
            ident = json.loads((root / group / 'identity.json').read_text())
            cfg = next(c for c in ident['configurations'] if c['file'] == name)
            c = Condition(**row['condition'])
            fresh = D.evaluate((cfg['controller'], cfg['weights'], cfg['design'], cfg['mode'], cfg['frontend'], row['setting'], row['set'], c))
            check(fresh == row, f'exact resimulation {group}/{name}')
            resimulated.append(f'{group}/{name}/{row["setting"]}/{row["set"]}/{row["condition"]["seed"]}')
    return dict(passed=True, checks=check.count, training_episode_counts=progress, resimulated=resimulated,
                evaluation_files=len(rows), estimator_rows=len(est), margin_corrected_files=len(before_rows))


def recompute_gate(root, rows_dir, decision, fn):
    """Run a gate function on a scratch copy of its input rows; True if it reproduces the saved decision.

    Gate functions write their decision under the root they are given. Running them on a copy
    means the audited tree, and every other evidence tree, is never written."""
    import contextlib, io, shutil, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        mirror = Path(tmp)
        shutil.copytree(root / rows_dir, mirror / rows_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            fn(mirror)
        return (mirror / decision).read_bytes() == (root / decision).read_bytes()


def audit_gates(root, check):
    """Both pilot gates recomputed from the rows under ``root`` (never from another tree)."""
    from movement_lab import diag02e as D
    from movement_lab.study02e import gate_learning
    if (root / 'pilot/gate-frontend.json').exists():
        check(recompute_gate(root, 'pilot/frontend', 'pilot/gate-frontend.json', D.gate), 'front-end gate recomputed from rows')
    if (root / 'pilot/gate-learning.json').exists():
        check(recompute_gate(root, 'pilot/learning', 'pilot/gate-learning.json', gate_learning), 'learning gate recomputed from rows')
    if (root / 'runs').exists():
        gate = json.loads((root / 'pilot/gate-learning.json').read_text())
        for run in (root / 'runs').glob('*-rand-s*'):
            check(gate[run.name.split('-rand')[0] + '-rand-s72']['passed'], 'confirmation training only for arms that passed')


def local_only(p):
    """Kept local and hashed: the Linux compile embeds upstream meshes; replay 200 Hz records regenerate exactly."""
    return p.suffix == '.mjb' or (p.parent.name == 'records' and p.suffix == '.npz')


def manifest(root):
    files = {}
    for p in sorted(root.rglob('*')):
        if not p.is_file() or p.name == 'manifest.json' or p.suffix in ('.log', '.tmp') or local_only(p) or '__pycache__' in p.parts:
            continue
        files[p.as_posix()] = digest(p)
    for p in sorted(Path('movement_lab').glob('*02e*')):
        files[p.as_posix()] = digest(p)
    files['tests/test_upright02e.py'] = digest('tests/test_upright02e.py')
    for name in REUSED:
        files[f'movement_lab/{name}'] = digest(Path('movement_lab') / name)
    for p in HISTORICAL_INPUTS:
        files[p] = digest(p)
    local = {p.as_posix(): digest(p) for p in sorted(root.rglob('*')) if p.is_file() and local_only(p)}
    return dict(files=files, local_only=local, model=dict(path='results/02b/model/model.mjb', sha256=json.loads(
        Path('results/02b/model/model.json').read_text())['mjb_sha256']))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--root', type=Path, default=Path('results/02e'))
    p.add_argument('--audit', action='store_true')
    p.add_argument('--resimulate', type=int, default=0)
    p.add_argument('--publish', action='store_true')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args(argv)
    if a.verify:
        data = json.loads((a.root / 'manifest.json').read_text())
        bad = [path for path, sha in data['files'].items() if not Path(path).exists() or digest(path) != sha]
        if bad:
            raise SystemExit('hash mismatch: ' + ', '.join(bad))
        print('verified', len(data['files']), 'published hashes;', len(data['local_only']), 'local-only files not required')
        return
    if a.audit:
        result = audit(a.root, a.resimulate)
        (a.root / 'validation.json').write_text(json.dumps(result, indent=1) + '\n', newline='\n')
        print(json.dumps(result, indent=1))
    if a.publish:
        (a.root / 'manifest.json').write_text(json.dumps(manifest(a.root), indent=1) + '\n', newline='\n')
        print('manifest written')


if __name__ == '__main__':
    main()
