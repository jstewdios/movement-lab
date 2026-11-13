"""02E corrections after review. Each step regenerates only the evidence its fix can affect.

    python -m movement_lab.correct02e margins --root results/02e --plan          # what would be re-simulated
    python -m movement_lab.correct02e margins --root results/02e --workers 4

``margins`` (PR #11 review, Codex P2): ``sim02e.support_margins`` projected the COM against
gravity's horizontal component instead of along it. The ground point of the gravity line
through the COM is ``com + s g`` with ``s = com_z / -g_z > 0``; the code subtracted. Support
margins are diagnostic only: no controller, front-end, failure test, score, gate or training
rollout reads them (``learn02e`` never requests them). On a level floor gravity's horizontal
component is exactly +0.0, so the projection is bit-identical either way. Only the six margin
fields of rows with a nonzero ``slope_deg`` can change.

The step re-simulates exactly those rows in every battery that records margins, requires
every other field to come back identical, and replaces the rows. The three sloped replay
records are re-simulated too: their 200 Hz arrays must be identical, and only the margin
fields of their published summaries change. The record in ``corrections/margin-projection``
keeps the source that produced the original rows, the declared edit, before/after hashes of
every file it touched, and before/after margins of every regenerated row, so the validator
can rebuild each original file's rows byte for byte. Completed files are skipped on a rerun.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import numpy as np
from movement_lab import pool02e
from movement_lab.eval02c import read_rows, write_gzip
from movement_lab.learn02c import digest

RECORD = 'corrections/margin-projection'
SOURCE = Path('movement_lab/sim02e.py')
BEFORE_SHA256 = '04fe0e8f24d61e8a11347e15347cdc181976f995947b7e3c359c43638fb3f531'   # pinned by every 02E identity before the fix
EDITS = (   # (before, after): the complete difference between the archived and the corrected sim02e.py
    ('  physically equivalent to standing on an inclined plane. Positive pitch: +x uphill.\n',
     '  physically equivalent to standing on an inclined plane. Positive pitch: the floor\n  descends toward +x.\n'),
    ('    # Project the COM along gravity onto the floor plane z = 0.\n'
     '    c = com[:2] - g_hat[:2] * (com[2] / -g_hat[2])\n',
     '    # Project the COM along gravity onto the floor plane z = 0: com + s g_hat with s = com_z / -g_z > 0.\n'
     '    c = com[:2] + g_hat[:2] * (com[2] / -g_hat[2])\n'),
)
MARGIN_FIELDS = ('min_support_margin_m', 'min_toe_margin_m', 'min_heel_margin_m', 'steady_toe_margin_m',
                 'steady_heel_margin_m', 'com_outside_support_s')
GROUPS = ('diagnostic/characterize', 'pilot/frontend', 'pilot/learning', 'confirm')   # every battery that records margins
FINDING = ('PR #11 review (Codex, P2): support_margins projected the COM opposite to gravity\'s horizontal component. '
           'For nonzero floor slopes this misplaced the COM ground point by 2 * com_z * tan(slope) along the slope, '
           'so the toe/heel/minimum margins and com_outside_support_s of sloped trials were wrong.')
UNAFFECTED = ('Trial outcomes and every non-margin row field (margins never feed back into the simulation); '
              'zero-slope rows (gravity horizontal component is exactly +0.0); gates (computed from recoveries); '
              'training and its pilots (learn02e rollouts never request margins); 1B estimators, feasibility probes '
              'and the Phase 0 resimulation (no margins); replay frames (no margin fields).')


def apply_edits(text, edits=EDITS):
    for before, after in edits:
        if text.count(before) != 1:
            raise ValueError('declared edit does not apply exactly once: ' + before.strip()[:60])
        text = text.replace(before, after)
    return text


def sloped(row):
    return any(row['condition']['slope_deg'])


def rows_text(rows):
    """The uncompressed content write_gzip stores (hash-comparable across zlib builds)."""
    return ''.join(json.dumps(r, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n' for r in rows).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def margins_of(row):
    return {k: row.get(k) for k in MARGIN_FIELDS}


def other_fields(row):
    return {k: v for k, v in row.items() if k not in MARGIN_FIELDS}


def verify_source(root):
    archived = root / RECORD / 'source/sim02e.py'
    if digest(archived) != BEFORE_SHA256:
        raise SystemExit(f'{archived} is not the pre-correction source')
    if apply_edits(archived.read_text()) != SOURCE.read_text():
        raise SystemExit('movement_lab/sim02e.py differs from the archived source by more than the declared edits')
    return digest(SOURCE)


def load_record(root, after_sha):
    path = root / RECORD / 'correction.json'
    if path.exists():
        rec = json.loads(path.read_text())
        changes = read_rows(root / RECORD / 'rows.jsonl.gz') if (root / RECORD / 'rows.jsonl.gz').exists() else []
        return rec, changes
    rec = dict(finding=FINDING, source=SOURCE.as_posix(), archived_source=f'{RECORD}/source/sim02e.py',
               before_sha256=BEFORE_SHA256, after_sha256=after_sha, edits=[list(e) for e in EDITS],
               fields=list(MARGIN_FIELDS), selection='rows whose condition has a nonzero slope_deg', unaffected=UNAFFECTED,
               groups={}, replay_records={})
    return rec, []


def save_record(root, rec, changes):
    folder = root / RECORD
    tmp = folder / 'rows.jsonl.tmp'; write_gzip(tmp, sorted(changes, key=lambda c: (c['group'], c['file'], c['index']))); tmp.replace(folder / 'rows.jsonl.gz')
    tmp = folder / 'correction.tmp'; tmp.write_text(json.dumps(rec, indent=1) + '\n', newline='\n'); tmp.replace(folder / 'correction.json')


def plan(root):
    out = {}
    for group in GROUPS:
        ident = json.loads((root / group / 'identity.json').read_text())
        n = sum(sloped(dict(condition=c['condition'])) for c in ident['conditions'])
        out[group] = dict(configurations=len(ident['configurations']), sloped_rows_per_configuration=n,
                          trials=n * len(ident['configurations']))
    from movement_lab.replay_select02e import TRIALS
    out['records'] = [t[0] for t in TRIALS if t[5].startswith('slope')]
    out['total_trials'] = sum(v['trials'] for k, v in out.items() if k != 'records') + len(out['records'])
    return out


def correct_batteries(root, ex, rec, changes, save):
    from movement_lab import diag02e as D
    from movement_lab.sim02e import Condition
    after_sha = rec['after_sha256']
    for group in GROUPS:
        ipath = root / group / 'identity.json'
        ident = json.loads(ipath.read_text())
        for name, s in ident['sources'].items():
            if name != 'sim02e.py' and digest(Path('movement_lab') / name) != s:
                raise SystemExit(f'{group}: source {name} changed since the battery ran; refusing')
        entry = rec['groups'].setdefault(group, dict(identity_before_sha256=digest(ipath), files={}))
        for cfg in ident['configurations']:
            path = root / group / f"{cfg['file']}.jsonl.gz"
            done = entry['files'].get(cfg['file'])
            if done and digest(path) == done['after_sha256']:
                continue
            if done and digest(path) != done['before_sha256']:
                raise SystemExit(f'{path} matches neither side of its correction record')
            if ident['sources']['sim02e.py'] != BEFORE_SHA256:
                raise SystemExit(f'{group} identity does not pin the pre-correction sim02e.py')
            before_bytes = path.read_bytes()
            rows = read_rows(path)
            idx = [i for i, r in enumerate(rows) if sloped(r)]
            tasks = [(cfg['controller'], cfg['weights'], cfg['design'], cfg['mode'], cfg['frontend'], rows[i]['setting'],
                      rows[i]['set'], Condition(**rows[i]['condition'])) for i in idx]
            fresh = list(ex.map(D.evaluate, tasks))
            new_rows = list(rows)
            file_changes = []
            for i, f in zip(idx, fresh):
                old = rows[i]
                if other_fields(f) != other_fields(old) or set(f) != set(old):
                    raise SystemExit(f'{group}/{cfg["file"]} row {i}: a non-margin field changed; the fix is not margin-only')
                new_rows[i] = f
                file_changes.append(dict(group=group, file=cfg['file'], index=i, setting=old['setting'], set=old['set'],
                                         seed=old['condition']['seed'], before=margins_of(old), after=margins_of(f)))
            new_bytes = gzip.compress(rows_text(new_rows), mtime=0)
            assert read_rows_bytes(new_bytes) == new_rows
            entry['files'][cfg['file']] = dict(before_sha256=sha(before_bytes), after_sha256=sha(new_bytes),
                                               before_rows_sha256=sha(rows_text(rows)), after_rows_sha256=sha(rows_text(new_rows)),
                                               rows=len(rows), regenerated=len(idx),
                                               changed=sum(c['before'] != c['after'] for c in file_changes))
            changes[:] = [c for c in changes if not (c['group'] == group and c['file'] == cfg['file'])] + file_changes
            save()                                     # the record first, so a stop can always be resumed or audited
            tmp = path.with_suffix('.tmp'); tmp.write_bytes(new_bytes); tmp.replace(path)
            print(f'{group}/{cfg["file"]}: {len(idx)} sloped rows re-simulated, {entry["files"][cfg["file"]]["changed"]} changed', flush=True)
        if ident['sources']['sim02e.py'] != after_sha:
            ident['sources']['sim02e.py'] = after_sha
            ipath.write_text(json.dumps(ident, indent=1) + '\n', newline='\n')
        entry['identity_after_sha256'] = digest(ipath)
        save()


def read_rows_bytes(data):
    return [json.loads(l) for l in gzip.decompress(data).splitlines()]


def correct_records(root, rec, changes, save):
    """The replay's sloped recordings: identical 200 Hz arrays, summaries with corrected margins."""
    from movement_lab.sim02b import Sim
    from movement_lab.sim02c import BASE
    from movement_lab.control02d import load_basis, Design
    from movement_lab.estimate02e import Kinematics
    from movement_lab.sim02e import Condition
    from movement_lab.report02e import record_trial
    from movement_lab.replay_select02e import TRIALS
    records = root / 'records'
    keys = [t[0] for t in TRIALS if (records / f'{t[0]}.json').exists() and sloped(json.loads((records / f'{t[0]}.json').read_text()))]
    pending = [k for k in keys if digest(records / f'{k}.json') != rec['replay_records'].get(k, {}).get('after_sha256')]
    if not pending:
        return
    sim = Sim(); theta = np.load(BASE)['final']; basis = load_basis(); kin = Kinematics(sim)
    for key in pending:
        meta_path, npz_path = records / f'{key}.json', records / f'{key}.npz'
        before = json.loads(meta_path.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            record_trial(sim, theta, basis, kin, before['controller'], before['weights'], before['design'], before['mode'],
                         before['frontend'], Condition(**before['condition']), tmp, key)
            after_bytes = (tmp / f'{key}.json').read_bytes()
            fresh, saved = np.load(tmp / f'{key}.npz'), np.load(npz_path)
            if set(fresh.files) != set(saved.files) or not all(np.array_equal(fresh[k], saved[k]) for k in fresh.files):
                raise SystemExit(f'replay record {key}: 200 Hz arrays differ; the fix is not margin-only')
        after = json.loads(after_bytes)
        if {k: v for k, v in after.items() if k != 'summary'} != {k: v for k, v in before.items() if k != 'summary'} \
                or other_fields(after['summary']) != other_fields(before['summary']):
            raise SystemExit(f'replay record {key}: a non-margin field changed')
        rec['replay_records'][key] = dict(before_sha256=digest(meta_path), after_sha256=sha(after_bytes),
                                          npz_sha256=digest(npz_path), npz_arrays_identical=True)
        changes[:] = [c for c in changes if not (c['group'] == 'records' and c['file'] == key)] + [
            dict(group='records', file=key, index=0, setting=None, set=None, seed=before['condition']['seed'],
                 before=margins_of(before['summary']), after=margins_of(after['summary']))]
        save()
        tmp = meta_path.with_suffix('.tmp'); tmp.write_bytes(after_bytes); tmp.replace(meta_path)
        print(f'records/{key}: summary margins corrected, arrays identical', flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('margins',))
    p.add_argument('--root', type=Path, default=Path('results/02e'))
    p.add_argument('--plan', action='store_true', help='print what would be re-simulated and stop')
    pool02e.add_arguments(p)
    a = p.parse_args(argv)
    if a.plan:
        print(json.dumps(plan(a.root), indent=1))
        return
    after_sha = verify_source(a.root)
    guard = pool02e.guard_from(a, 'correct-margins')
    rec, changes = load_record(a.root, after_sha)
    if rec['after_sha256'] != after_sha:
        raise SystemExit('sim02e.py changed since this correction started')
    (a.root / RECORD).mkdir(parents=True, exist_ok=True)
    save = lambda: save_record(a.root, rec, changes)
    from movement_lab import diag02e as D
    with pool02e.cleanup_on_signals():
        with pool02e.RecyclingExecutor(a.workers, D.init_worker, tasks_per_worker=a.tasks_per_worker, guard=guard) as ex:
            correct_batteries(a.root, ex, rec, changes, save)
        correct_records(a.root, rec, changes, save)
    rec['totals'] = dict(files=sum(len(g['files']) for g in rec['groups'].values()),
                         rows_regenerated=sum(f['regenerated'] for g in rec['groups'].values() for f in g['files'].values()),
                         rows_changed=sum(f['changed'] for g in rec['groups'].values() for f in g['files'].values()),
                         replay_records=len(rec['replay_records']))
    save()
    print(json.dumps(dict(totals=rec['totals'], memory=guard.summary())), flush=True)


if __name__ == '__main__':
    main()
