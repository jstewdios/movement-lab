"""02E tables, figures and the recorded-state replay. Reads evidence; selects nothing by outcome.

    python -m movement_lab.report02e --root results/02e
"""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import numpy as np
from movement_lab.eval02c import read_rows

LEARNED = [f'{a}-s{s}' for a in ('history', 'memoryless') for s in (1, 2, 3)]
NAMES = dict(reflex='Frozen 02B reflex', oracle02c='02C fixed · onset oracle', imu02c='02C fixed · IMU trigger',
             **{f'{a}-s{s}': f'02D {a} s{s}' for a in ('history', 'memoryless') for s in (1, 2, 3)})


def load(folder):
    return {p.name[:-9]: read_rows(p) for p in sorted(Path(folder).glob('*.jsonl.gz'))}


def count(rows, **match):
    sel = [r for r in rows if all(r[k] == v for k, v in match.items())]
    return sum(r['recovered_200hz'] for r in sel), len(sel)


def paired(a, b, settings=None):
    A = {(r['setting'], r['set'], r['condition']['seed']): r for r in a if settings is None or r['setting'] in settings}
    B = {(r['setting'], r['set'], r['condition']['seed']): r for r in b if settings is None or r['setting'] in settings}
    return (sum(B[k]['recovered_200hz'] and not A[k]['recovered_200hz'] for k in A),
            sum(A[k]['recovered_200hz'] and not B[k]['recovered_200hz'] for k in A))


def cell_table(data):
    out = {}
    for name, rows in data.items():
        settings = list(dict.fromkeys(r['setting'] for r in rows)); kinds = list(dict.fromkeys(r['set'] for r in rows))
        out[name] = dict(total=count(rows), steps=sum(r['recovery_with_step'] for r in rows),
                         falls=sum(not r['survived'] for r in rows),
                         by_setting={s: count(rows, setting=s) for s in settings},
                         by_kind_setting={s: {k: count(rows, setting=s, set=k) for k in kinds} for s in settings})
    return out


def quiet_mechanism(rows):
    """Steady (pre-event) behaviour of quiet trials by setting: what the controller makes of the offset."""
    out = {}
    for s in dict.fromkeys(r['setting'] for r in rows):
        q = [r for r in rows if r['setting'] == s and r['set'] == 'quiet']
        val = lambda k: [r[k] for r in q if r[k] is not None]
        coef = [r['steady_coefficient_abs_max'] for r in q if r['steady_coefficient_abs_max'] is not None]
        out[s] = dict(recovered=sum(r['recovered_200hz'] for r in q), n=len(q),
                      fall_times=[r['failure_time_s'] for r in q],
                      steady_true_pitch_deg=float(np.mean(val('steady_true_pitch_deg'))) if val('steady_true_pitch_deg') else None,
                      steady_seen_pitch_deg=float(np.mean(val('steady_seen_pitch_deg'))) if val('steady_seen_pitch_deg') else None,
                      steady_residual_rms_rad=float(np.mean(val('steady_residual_rms_rad'))) if val('steady_residual_rms_rad') else None,
                      steady_coefficient_abs_max=float(np.max(coef)) if coef else None,
                      steady_toe_margin_mm=float(np.mean(val('steady_toe_margin_m')) * 1000) if val('steady_toe_margin_m') else None,
                      steady_heel_margin_mm=float(np.mean(val('steady_heel_margin_m')) * 1000) if val('steady_heel_margin_m') else None,
                      target_saturation_frac=float(np.mean(val('target_saturation_frac'))) if val('target_saturation_frac') else None,
                      torque_saturation_frac=float(np.mean(val('torque_saturation_frac'))) if val('torque_saturation_frac') else None)
    return out


def tables(root):
    t = {}
    char = load(root / 'diagnostic/characterize')
    t['characterize'] = dict(cells=cell_table(char), quiet_mechanism={n: quiet_mechanism(r) for n, r in char.items()})
    t['estimators'] = json.loads((root / 'diagnostic/estimators-summary.json').read_text())
    if (root / 'pilot/frontend').exists():
        fr = load(root / 'pilot/frontend')
        t['frontend_pilot'] = dict(cells=cell_table(fr), gate=json.loads((root / 'pilot/gate-frontend.json').read_text()))
    if (root / 'pilot/learning').exists():
        lr = load(root / 'pilot/learning')
        t['learning_pilot'] = dict(cells=cell_table(lr), gate=json.loads((root / 'pilot/gate-learning.json').read_text()))
    if (root / 'confirm').exists():
        cf = load(root / 'confirm')
        cells = cell_table(cf)
        comp = {}
        for name in cf:
            ctrl, fe = name.split('__')
            base = f'{ctrl}__none' if fe != 'none' else None
            if '-rand-' in ctrl:
                base = ctrl.replace('-rand', '') + '__none'
            if base and base in cf:
                comp[name] = dict(against=base, all=paired(cf[base], cf[name]),
                                  **{s: paired(cf[base], cf[name], (s,)) for s in dict.fromkeys(r['setting'] for r in cf[name])})
        nominal = {name: dict(quiet=count(rows, setting='nominal', set='quiet'), backward=count(rows, setting='nominal', set='back-.08'),
                              total=count(rows, setting='nominal')) for name, rows in cf.items()}
        t['confirm'] = dict(cells=cells, paired=comp, nominal_gates=nominal, quiet_mechanism={n: quiet_mechanism(r) for n, r in cf.items()})
    return t


def figures(root, t):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out = root / 'report'; out.mkdir(exist_ok=True)
    char = t['characterize']
    offsets = [-4, -2, -1, -.5, 0, .5, 1, 2, 4]
    key = lambda v: 'nominal' if v == 0 else f'pitch{v:+g}'
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    colors = dict(reflex='#222222', oracle02c='#8a8a8a', imu02c='#b0b0b0', **{f'history-s{s}': c for s, c in zip((1, 2, 3), ('#1b7f5e', '#3aa37f', '#7cc8a8'))},
                  **{f'memoryless-s{s}': c for s, c in zip((1, 2, 3), ('#b4561c', '#d07a3a', '#e8a777'))})
    for name, cells in char['cells'].items():
        frac = [cells['by_setting'][key(v)][0] / cells['by_setting'][key(v)][1] for v in offsets]
        quiet = [cells['by_kind_setting'][key(v)]['quiet'][0] / cells['by_kind_setting'][key(v)]['quiet'][1] for v in offsets]
        lw = 2.6 if name == 'reflex' else 1.6
        axes[0].plot(offsets, frac, 'o-', color=colors[name], lw=lw, ms=4, label=NAMES[name])
        axes[1].plot(offsets, quiet, 'o-', color=colors[name], lw=lw, ms=4)
    for name, mech in char['quiet_mechanism'].items():
        if name in LEARNED:
            axes[2].plot(offsets, [mech[key(v)]['steady_residual_rms_rad'] or np.nan for v in offsets], 'o-', color=colors[name], lw=1.6, ms=4)
    axes[0].set(title='All five disturbance kinds (10 trials per point)', xlabel='IMU pitch misalignment (deg)', ylabel='Strict recovery fraction')
    axes[1].set(title='Quiet standing only (2 trials per point)', xlabel='IMU pitch misalignment (deg)', ylabel='Strict recovery fraction')
    axes[2].set(title='Learned residual while standing quietly', xlabel='IMU pitch misalignment (deg)', ylabel='Residual RMS before any event (rad)')
    axes[0].legend(fontsize=7.5, ncol=2, frameon=False)
    for ax in axes:
        ax.grid(alpha=.2); ax.axvline(0, color='gray', lw=.6)
    fig.savefig(out / 'sensitivity.png', dpi=150); plt.close(fig)

    est = t['estimators']
    order = ['prior', 'static', 'rock-0.05', 'rock-0.1']
    settings = list(est['static']['cells'])
    fig, ax = plt.subplots(figsize=(14, 4.6), constrained_layout=True)
    width = .2
    for j, k in enumerate(order):
        vals = [abs(est[k]['cells'][s]['mean_error_roll_pitch_deg'][1]) if k.startswith('rock') else
                float(np.hypot(*est[k]['cells'][s]['mean_error_roll_pitch_deg'])) for s in settings]
        falls = [est[k]['cells'][s]['falls'] for s in settings]
        x = np.arange(len(settings)) + (j - 1.5) * width
        bars = ax.bar(x, np.minimum(vals, 20), width, label={'prior': 'IMU only (remembered reading)', 'static': 'Gravity vs leg kinematics (level floor)',
                                                            'rock-0.05': 'Active roll probe 0.05 rad (pitch only)', 'rock-0.1': 'Active roll probe 0.10 rad (pitch only)'}[k],
                      color=['#6b7b8c', '#1b7f5e', '#d07a3a', '#a2451c'][j])
        for xi, f in zip(x, falls):
            if f:
                ax.text(xi, 20.3, f'{f}✕', ha='center', fontsize=7, color='#a2451c')
    ax.set_xticks(np.arange(len(settings)), settings, rotation=30, ha='right')
    ax.set(ylabel='|mean estimation error| (deg, clipped at 20)', ylim=(0, 22),
           title='What the sensors can tell: estimation error by condition (4 trials each; ✕ = falls in those trials)')
    ax.legend(frameon=False, fontsize=8); ax.grid(alpha=.2, axis='y')
    fig.savefig(out / 'estimators.png', dpi=150); plt.close(fig)


def record_trial(sim, theta, basis, kin, name, vec, design, mode, fe, cond, records, key):
    """Record (or reload and verify) one replay trial."""
    from movement_lab.sim02e import rollout
    from movement_lab.eval02d import effective_mode
    from movement_lab.estimate02e import Frontend, Settings
    from movement_lab.control02d import Design
    from movement_lab import diag02e as D
    npz, meta = records / f'{key}.npz', records / f'{key}.json'
    canon = lambda x: json.loads(json.dumps(x))
    if npz.exists():
        saved = json.loads(meta.read_text())
        assert saved['weights'] == canon(vec) and saved['condition'] == canon(asdict(cond)) and saved['frontend'] == canon(fe)
        return saved['summary'], dict(np.load(npz))
    D.SIM = sim
    frontend = Frontend(Settings(**fe), kin) if fe else None
    s, r = rollout(sim, theta, np.asarray(vec), cond, Design(**design), record=True, terminate=False, mode=effective_mode(mode, cond),
                   basis=basis, frontend=frontend, margins=True)
    D.metrics(s, r, cond, Design(**design).bound)
    s['frontend_report'] = s.pop('frontend', None)
    keep = ('time_s', 'qpos', 'qvel', 'torque_nm', 'target_cmd', 'target_applied', 'offset', 'residual', 'features', 'coefficients',
            'grav_true', 'grav_raw', 'gyro_raw', 'probe', 'foot_found', 'foot_force_n', 'clearance_m', 'tilt_deg', 'nonfoot',
            'penetration_m', 'self_contacts', 'com', 'foot_pos')
    np.savez_compressed(npz, **{k: r[k] for k in keep})
    meta.write_text(json.dumps(canon(dict(controller=name, weights=vec, design=design, mode=mode, frontend=fe, condition=asdict(cond),
                                          z0=sim.z0, summary=s)), indent=1) + '\n', newline='\n')
    return canon(s), r


def frames_for(sim, s, r, cond):
    from movement_lab.report02d import frames
    from movement_lab.sim02d import event_time
    from movement_lab.diag02e import grav_deg
    rec = dict(r)
    d = frames(sim, s, rec, replace(cond, push_time_s=event_time(cond)))
    true_p, _ = grav_deg(r['grav_true']); seen_p, _ = grav_deg(r['features'][:, :2] * .1)
    strength = np.sqrt(np.mean((r['residual'] + r['probe']) ** 2, axis=1))
    for f, i in zip(d['frames'], range(3, len(r['time_s']), 4)):
        f.update(pitch_true=round(float(true_p[i]), 3), pitch_seen=round(float(seen_p[i]), 3), strength=round(float(strength[i]), 4))
    return d


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02e'))
    p.add_argument('--no-replay', action='store_true')
    a = p.parse_args(argv)
    (a.root / 'report').mkdir(exist_ok=True)
    t = tables(a.root)
    (a.root / 'report/tables.json').write_text(json.dumps(t, indent=1) + '\n', newline='\n')
    figures(a.root, t)
    if not a.no_replay:
        from movement_lab.replay_select02e import build
        build(a.root, t)


if __name__ == '__main__':
    main()
