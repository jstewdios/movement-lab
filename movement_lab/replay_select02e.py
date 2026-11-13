"""02E recorded-state replay: a fixed, declared selection of trials (called by report02e).

Every replayed trial is the *first replicate* (rep 0) of a battery cell, chosen by controller,
front-end, setting and kind, never by a trial's outcome. Two presets were
chosen after their results were known and say so in their captions: 05 shows the nominal quiet
cell that made the static front-end fail its declared gate, and 09 shows the held-out strong-push
trade-off of randomized training.

Each recording re-simulates the trial and must reproduce the published battery row exactly
(every summary field), so the replay is also an exact-resimulation check. The 200 Hz
records stay local (regenerated deterministically); their JSON metadata and the replay
page are published.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

CANON = lambda x: json.loads(json.dumps(x))
CHAR, PILOT, LEARN, CONFIRM = 'diagnostic/characterize', 'pilot/frontend', 'pilot/learning', 'confirm'
LABELS = {'reflex': 'Frozen 02B reflex', 'history-s1': '02D history s1', 'history-s3': '02D history s3',
          'memoryless-s1': '02D memoryless s1', 'history-s72': '02D history s72 (control)',
          'history-rand-s72': 'History, randomized misalignment', 'memoryless-rand-s72': 'Memoryless, randomized misalignment',
          'history-rand-s1': 'History randomized s1', 'memoryless-rand-s1': 'Memoryless randomized s1'}
FE_LABELS = dict(none='raw IMU', gate='residual gate only', static='static calibration', prior='IMU-only prior')

# (key, battery group, row file, controller, front-end, setting, kind)
TRIALS = [
    ('h1-nominal-quiet', CHAR, 'history-s1', 'history-s1', 'none', 'nominal', 'quiet'),
    ('h1-pitch+1-quiet', CHAR, 'history-s1', 'history-s1', 'none', 'pitch+1', 'quiet'),
    ('h1-pitch+2-quiet', CHAR, 'history-s1', 'history-s1', 'none', 'pitch+2', 'quiet'),
    ('reflex-pitch+2-quiet', CHAR, 'reflex', 'reflex', 'none', 'pitch+2', 'quiet'),
    ('m1-pitch+2-quiet', CHAR, 'memoryless-s1', 'memoryless-s1', 'none', 'pitch+2', 'quiet'),
    ('D-h1-none-pitch+2-quiet', PILOT, 'history-s1__none', 'history-s1', 'none', 'pitch+2', 'quiet'),
    ('D-h1-gate-pitch+2-quiet', PILOT, 'history-s1__gate', 'history-s1', 'gate', 'pitch+2', 'quiet'),
    ('D-h1-static-pitch+2-quiet', PILOT, 'history-s1__static', 'history-s1', 'static', 'pitch+2', 'quiet'),
    ('D-h1-none-slope-2-quiet', PILOT, 'history-s1__none', 'history-s1', 'none', 'slope-2', 'quiet'),
    ('D-h1-static-slope-2-quiet', PILOT, 'history-s1__static', 'history-s1', 'static', 'slope-2', 'quiet'),
    ('D-h1-static-slope+2-quiet', PILOT, 'history-s1__static', 'history-s1', 'static', 'slope+2', 'quiet'),
    ('D-h3-none-nominal-quiet', PILOT, 'history-s3__none', 'history-s3', 'none', 'nominal', 'quiet'),
    ('D-h3-gate-nominal-quiet', PILOT, 'history-s3__gate', 'history-s3', 'gate', 'nominal', 'quiet'),
    ('D-h3-static-nominal-quiet', PILOT, 'history-s3__static', 'history-s3', 'static', 'nominal', 'quiet'),
    ('D-reflex-none-pitch+2-fwd', PILOT, 'reflex__none', 'reflex', 'none', 'pitch+2', 'fwd-.20'),
    ('D-h1-none-pitch+2-fwd', PILOT, 'history-s1__none', 'history-s1', 'none', 'pitch+2', 'fwd-.20'),
    ('D-h1-static-pitch+2-fwd', PILOT, 'history-s1__static', 'history-s1', 'static', 'pitch+2', 'fwd-.20'),
    ('L-control-pitch+2-quiet', LEARN, 'history-s72', 'history-s72', 'none', 'pitch+2', 'quiet'),
    ('L-hrand-pitch+2-quiet', LEARN, 'history-rand-s72', 'history-rand-s72', 'none', 'pitch+2', 'quiet'),
    ('L-mrand-pitch+2-quiet', LEARN, 'memoryless-rand-s72', 'memoryless-rand-s72', 'none', 'pitch+2', 'quiet'),
    ('H-h1-pitch+2-quiet', CONFIRM, 'history-s1__none', 'history-s1', 'none', 'pitch+2', 'quiet'),
    ('H-hrand1-pitch+2-quiet', CONFIRM, 'history-rand-s1__none', 'history-rand-s1', 'none', 'pitch+2', 'quiet'),
    ('H-mrand1-pitch+2-quiet', CONFIRM, 'memoryless-rand-s1__none', 'memoryless-rand-s1', 'none', 'pitch+2', 'quiet'),
    ('H-h1-nominal-fwd-.32', CONFIRM, 'history-s1__none', 'history-s1', 'none', 'nominal', 'fwd-.32'),
    ('H-hrand1-nominal-fwd-.32', CONFIRM, 'history-rand-s1__none', 'history-rand-s1', 'none', 'nominal', 'fwd-.32'),
    ('H-mrand1-nominal-fwd-.32', CONFIRM, 'memoryless-rand-s1__none', 'memoryless-rand-s1', 'none', 'nominal', 'fwd-.32'),
]

PRESETS = [
    ('01 · How much error can a learned controller absorb?', ['h1-nominal-quiet', 'h1-pitch+1-quiet', 'h1-pitch+2-quiet'],
     ['IMU correct', 'IMU pitched +1°', 'IMU pitched +2°'],
     'One 02D controller standing quietly with no push, under a growing constant IMU pitch misalignment (1A, first replicate). '
     'Compare the true trunk pitch with what the controller perceives.'),
    ('02 · The same 2° error, three controllers', ['reflex-pitch+2-quiet', 'h1-pitch+2-quiet', 'm1-pitch+2-quiet'],
     ['Frozen reflex', 'History s1', 'Memoryless s1'],
     'Identical quiet trial and +2° pitch misalignment (1A). The frozen reflex only shifts its stance; the learned residuals '
     'read the offset as a sustained disturbance cue.'),
    ('03 · Calibrating on level ground', ['D-h1-none-pitch+2-quiet', 'D-h1-gate-pitch+2-quiet', 'D-h1-static-pitch+2-quiet'],
     ['Raw IMU', 'Residual gate only', 'Static calibration'],
     'Development battery D, +2° pitch, quiet. The gate holds the residual off for 1.6 s without correcting anything; the static '
     'front-end also compares gravity with leg kinematics during 0.3–1.3 s and rotates the IMU readings afterwards.'),
    ('04 · Calibration on a slope', ['D-h1-none-slope-2-quiet', 'D-h1-static-slope-2-quiet', 'D-h1-static-slope+2-quiet'],
     ['Raw IMU · −2° slope', 'Static · −2° slope', 'Static · +2° slope'],
     'Correct IMU, tilted floor. Leg kinematics assume a level floor, so static calibration reports the slope as a sensor '
     'offset (H2). Slope cells were reported, not gated.'),
    ('05 · Regression: why static calibration failed its gate', ['D-h3-none-nominal-quiet', 'D-h3-gate-nominal-quiet', 'D-h3-static-nominal-quiet'],
     ['Raw IMU', 'Residual gate only', 'Static calibration'],
     'Selected to display a regression: the nominal quiet cell in which static calibration lost history s3 trials. The declared '
     'gate required zero nominal quiet losses. Nominal means a correct IMU, so any estimate is pure error.'),
    ('06 · A push under a 2° error', ['D-reflex-none-pitch+2-fwd', 'D-h1-none-pitch+2-fwd', 'D-h1-static-pitch+2-fwd'],
     ['Frozen reflex', 'History s1 · raw IMU', 'History s1 · static'],
     'Forward 0.20 m/s impulse at the same onset, with a +2° pitch misalignment (battery D, first replicate).'),
    ('07 · Learning with randomized misalignment', ['L-control-pitch+2-quiet', 'L-hrand-pitch+2-quiet', 'L-mrand-pitch+2-quiet'],
     ['02D control', 'History · randomized', 'Memoryless · randomized'],
     '1D pilots (1,536 episodes each) trained with U(0, 4°) misalignment about a random horizontal axis, against the matched '
     '02D control, on battery D at +2° pitch, quiet.'),
    ('08 · Held out: the same 2° error after confirmation training', ['H-h1-pitch+2-quiet', 'H-hrand1-pitch+2-quiet', 'H-mrand1-pitch+2-quiet'],
     ['02D history s1', 'History randomized s1', 'Memoryless randomized s1'],
     'Held-out battery H, +2° pitch, quiet (seed 1 of each arm, as everywhere in this replay). History randomized s1 ended '
     'on the all-zero candidate, so it is exactly the frozen reflex; it is shown as trained, not replaced.'),
    ('09 · Held out: what the robustness cost', ['H-h1-nominal-fwd-.32', 'H-hrand1-nominal-fwd-.32', 'H-mrand1-nominal-fwd-.32'],
     ['02D history s1', 'History randomized s1', 'Memoryless randomized s1'],
     'Correct IMU, held-out 0.32 m/s forward impulse: the strong push that 02D learned to recover. Randomized training kept '
     'the residual small, and most of that recovery went with it. Added after confirmation to display this trade-off '
     '(first replicate, seed 1 of each arm).'),
]


def condition_for(group, setting, kind):
    from movement_lab import diag02e as D
    conds = (D.battery(D.SETTINGS_1A, D.KINDS, 7_000_000_000) if group == CHAR else
             D.battery(D.SETTINGS_H, D.KINDS_H, 8_000_000_000, reps=3, onset=(2.2, 3.4)) if group == CONFIRM else
             D.battery(D.SETTINGS_D, D.KINDS, 7_200_000_000))
    return next(c for s, k, c in conds if s == setting and k == kind)   # first replicate (rep 0)


def configuration(root, group, file):
    """The exact controller weights, design, mode and front-end that produced the battery rows."""
    ident = json.loads((root / group / 'identity.json').read_text())
    return next(c for c in ident['configurations'] if c['file'] == file)


def build(root, t):
    from dataclasses import asdict
    from movement_lab.sim02b import Sim
    from movement_lab.sim02c import BASE
    from movement_lab.sim02d import event_time
    from movement_lab.control02d import load_basis
    from movement_lab.estimate02e import Kinematics
    from movement_lab.eval02c import read_rows
    from movement_lab.report02e import record_trial, frames_for
    sim = Sim(); theta = np.load(BASE)['final']; basis = load_basis(); kin = Kinematics(sim)
    records = root / 'records'; records.mkdir(exist_ok=True)
    available = [x for x in TRIALS if (root / x[1] / f'{x[2]}.jsonl.gz').exists()]
    data, index, checks = [], {}, []
    for key, group, file, name, fe_name, setting, kind in available:
        cond = condition_for(group, setting, kind)
        cfg = configuration(root, group, file)
        assert cfg['controller'] == name, (key, cfg['controller'])
        fe = cfg['frontend']
        s, r = record_trial(sim, theta, basis, kin, name, cfg['weights'], cfg['design'], cfg['mode'], fe, cond, records, key)
        row = next(x for x in read_rows(root / group / f'{file}.jsonl.gz')
                   if x['setting'] == setting and x['set'] == kind and x['condition'] == CANON(asdict(cond)))
        s = CANON(s)
        mismatch = sorted(k for k in s if s[k] != row.get(k))
        if mismatch:
            raise AssertionError(f'replay {key} does not reproduce {group}/{file}: {mismatch}')
        checks.append(f'{group}/{file}/{setting}/{kind}/{cond.seed}')
        d = frames_for(sim, s, r, cond)
        ev = event_time(cond)
        d.update(label=f'{LABELS[name]} · {FE_LABELS[fe_name]} · {setting} · {kind}', event_time=ev,
                 recovered=s['recovered_200hz'], step=s['recovery_with_step'], failure_time=s['failure_time_s'],
                 condition=f'{setting} · {kind}')
        if fe and fe['kind'] != 'none':
            d.update(phase_note='Calibration window: residual held off', phase_end=fe['end_s'] + fe['blend_s'])
        index[key] = len(data); data.append(d)
        print('recorded replay', key, flush=True)
    presets = [dict(title=title, panels=[index[k] for k in keys], roles=roles, caption=caption, start=-2.4)
               for title, keys, roles, caption in PRESETS if all(k in index for k in keys)]
    char = t['characterize']['cells']
    learned = [f'{a}-s{s}' for a in ('history', 'memoryless') for s in (1, 2, 3)]
    q = lambda n, s: char[n]['by_kind_setting'][s]['quiet']
    lq = [sum(q(n, s)[0] for n in learned) for s in ('nominal', 'pitch+2')]
    gate = t['frontend_pilot']['gate']['static']
    stats = [dict(value=f'{lq[0]}/12 → {lq[1]}/12', label='Six 02D controllers · quiet standing recovered, correct IMU → +2° pitch error (1A)'),
             dict(value=f'+{gate["misaligned_net"]} / {gate["nominal_quiet_losses"]}',
                  label='Static calibration · net misaligned gain / nominal quiet losses (gate: ≥30 and 0) → not confirmed')]
    finding = (f'A constant IMU misalignment of +2° is enough to make the learned 02D controllers fall while standing still '
               f'({lq[1]}/12 quiet trials recovered, against {lq[0]}/12 with a correct IMU); the frozen reflex is unaffected. '
               f'Static calibration against leg kinematics recovers much of the loss ({gate["misaligned_net"]:+d} net paired trials) '
               f'but adds nominal quiet failures, so it failed its declared gate.')
    if 'confirm' in t:
        cells = t['confirm']['cells']
        pooled = lambda arm, tag: sum(cells[f'{arm}{tag}-s{s}__none']['total'][0] for s in (1, 2, 3))
        strong = lambda tag: sum(cells[f'{a}{tag}-s{s}__none']['by_kind_setting']['nominal']['fwd-.32'][0]
                                 for a in ('history', 'memoryless') for s in (1, 2, 3))
        n = cells['reflex__none']['total'][1] * 3
        stats += [dict(value=f'{pooled("history", "")} → {pooled("history", "-rand")}', label=f'History arm, 3 seeds · held-out recoveries of {n}, 02D → randomized'),
                  dict(value=f'{pooled("memoryless", "")} → {pooled("memoryless", "-rand")}', label=f'Memoryless arm, 3 seeds · held-out recoveries of {n}, 02D → randomized'),
                  dict(value=f'{strong("")}/18 → {strong("-rand")}/18', label='Nominal held-out 0.32 m/s pushes recovered, six 02D runs → six randomized runs')]
        finding += (f' Training with randomized misalignment then passed its pilot gate and, on the held-out battery, recovered many '
                    f'more misaligned and sloped trials ({pooled("history", "")}→{pooled("history", "-rand")} and '
                    f'{pooled("memoryless", "")}→{pooled("memoryless", "-rand")} of {n} per arm). It did so by keeping the learned response '
                    f'small, close to the frozen reflex (one run ended exactly on it), and gave up most of the strong-push recovery that '
                    f'02D had learned ({strong("")}/18 → {strong("-rand")}/18 nominal 0.32 m/s pushes).')
    payload = dict(trials=data, presets=presets, finding=finding, stats=stats,
                   provenance='Evidence: results/02e; model SHA256 ebb2ce87…; every replayed trial reproduces its published battery row exactly.')
    template = Path('movement_lab/replay02e.html').read_text()
    (root / 'report/replay.html').write_text(template.replace('__DATA__', json.dumps(payload, separators=(',', ':'))), encoding='utf-8')
    (root / 'report/replay-checks.json').write_text(json.dumps(dict(reproduced_rows=checks), indent=1) + '\n', newline='\n')
    return payload
