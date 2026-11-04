"""02B analysis: figures and machine-readable tables from saved 02B outputs.

    python -m movement_lab.analysis02b --root results/02b

Reads runs/*/ (progress, reflex snapshots), eval/final and eval/ablation
(held-out results) and crosscheck/; re-simulates only a handful of named,
seeded trials for the response and evolution figures (no training, nothing
fed back to any learner). Writes report/*.png and report/tables.json.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from movement_lab import sim02b
from movement_lab.eval02b import SHIFTS, symmetry_fraction, wilson
from movement_lab.sim02b import Condition, FEATURES, JOINTS

# Reference palette (dataviz skill, light mode). Arms keep fixed slots.
SURFACE, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
ARM_COLORS = {'ars-push': '#2a78d6', 'random-push': '#eb6834', 'ars-none': '#1baf7a', 'home': '#8a8984'}
ARM_LABELS = {'ars-push': 'ARS, pushed experience', 'random-push': 'Random search, pushed',
              'ars-none': 'ARS, standing-only experience', 'home': 'Fixed HOME targets'}
SEQ = ['#cde2fb', '#b7d3f6', '#9ec5f4', '#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
SEQ_CMAP = LinearSegmentedColormap.from_list('seq', ['#f0efec'] + SEQ)
DIV_CMAP = LinearSegmentedColormap.from_list('div', ['#c0392b', '#e34948', '#f0b3ae', '#f0efec', '#9ec5f4', '#3987e5', '#1c5cab'])


def style():
    plt.rcParams.update({'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
                         'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
                         'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                         'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 9,
                         'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 2})


def arm_of(run):
    return run.rsplit('-s', 1)[0]


def load_runs(root):
    runs = {}
    for d in sorted((root / 'runs').iterdir()):
        if not (d / 'final.json').exists():
            continue
        rows = [json.loads(l) for l in (d / 'progress.jsonl').read_text().splitlines()]
        runs[d.name] = dict(config=json.loads((d / 'config.json').read_text()), rows=rows,
                            final=json.loads((d / 'final.json').read_text()), thetas=np.load(d / 'thetas.npz'))
    return runs


def load_eval(path):
    from movement_lab.eval02b import read_results
    rows = read_results(path)
    if rows is None:
        return None, None
    return rows, json.loads((path / 'summary.json').read_text())


# --------------------------------------------------------------------------- figures
def fig_learning(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, key, title in ((axes[0], 'survival_none', 'Unpushed probes (8)'), (axes[1], 'survival_push', 'Pushed probes (32; 0.1-0.4 m/s, 8 directions)')):
        ax.axhline(0, color=ARM_COLORS['home'], lw=2, ls='--')
        ax.text(1.0, 0.03, 'fixed HOME: falls in every probe', color=INK2, fontsize=8, transform=ax.get_yaxis_transform(), ha='right')
        for name, r in runs.items():
            arm = arm_of(name)
            per_it = r['config']['directions'] * 2 * r['config']['conditions']
            pr = [x for x in r['rows'] if x['kind'] == 'probe']
            ax.plot([p['it'] * per_it for p in pr], [p[key] for p in pr], color=ARM_COLORS[arm], lw=1.6, alpha=0.9,
                    marker='o', ms=3)
        ax.set_title(title, loc='left')
        ax.set_xlabel('training rollouts (each a 4 s physical trial)')
        ax.set_xscale('symlog', linthresh=500)
        ax.set_ylim(-0.05, 1.05)
    axes[0].set_ylabel('probe survival fraction (current reflex)')
    handles = [plt.Line2D([], [], color=ARM_COLORS[a], marker='o', ms=4) for a in ('ars-push', 'random-push', 'ars-none')]
    fig.legend(handles, [ARM_LABELS[a] + ' (3 seeds)' for a in ('ars-push', 'random-push', 'ars-none')],
               loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.04))
    axes[0].text(0.98, 0.9, 'ARS lines overlap at 1.0', transform=axes[0].transAxes, ha='right', fontsize=8, color=INK2)
    fig.suptitle('Learning curves: probe outcomes are measured, never used for updates', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / 'learning.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def grid_matrix(rows, ctrl_prefixes, set_name='push_grid'):
    dirs = np.round(np.degrees(np.arange(12) * 2 * np.pi / 12)).astype(int)
    mags = np.round(np.arange(0.05, 0.501, 0.05), 3)
    k = np.zeros((len(mags), len(dirs))); n = np.zeros_like(k)
    for r in rows:
        if r['set'] != set_name or not any(r['controller'].startswith(p) for p in ctrl_prefixes):
            continue
        dv = r['condition']['push_dv']
        mag = round(math.hypot(*dv), 3)
        ang = int(round(math.degrees(math.atan2(dv[1], dv[0])))) % 360
        i, j = int(np.argmin(np.abs(mags - mag))), int(np.where(dirs == ang)[0][0])
        k[i, j] += r['survived']; n[i, j] += 1
    return dirs, mags, k, n


def fig_push_grid(rows, bounds, out):
    groups = [('home', ('home',)), ('random-push', ('random-push',)), ('ars-none', ('ars-none',)), ('ars-push', ('ars-push',))]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.9), sharey=True)
    for ax, (g, pref) in zip(axes, groups):
        dirs, mags, k, n = grid_matrix(rows, pref)
        frac = np.where(n > 0, k / np.maximum(n, 1), np.nan)
        im = ax.imshow(frac, origin='lower', aspect='auto', cmap=SEQ_CMAP, vmin=0, vmax=1,
                       extent=(-15, 345, 0.025, 0.525))
        ax.plot(np.degrees(bounds['directions_rad']), bounds['bounds'], color=INK, lw=1.6, ls='--', drawstyle='default')
        ax.set_title(f"{ARM_LABELS[g]}\n(n={int(n.max())} per cell)", loc='left', fontsize=9)
        ax.set_xticks(dirs[::3]); ax.set_xticklabels(['0°\nfwd', '90°\nleft', '180°\nback', '270°\nright'])
        ax.set_xlabel('push direction (world)')
        ax.grid(False)
    axes[0].set_ylabel('velocity increment |Δv| (m/s)')
    axes[0].text(180, 0.40, 'dashed: no-step\ncapture-point limit\nof the HOME stance', fontsize=7.5, color=INK, ha='center')
    cb = fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01)
    cb.set_label('survival fraction', color=INK2)
    fig.suptitle('Held-out push grid (push at 3.0 s, 7 s trials; training never pushed after 2 s or above 0.4 m/s)', x=0.01, y=1.06, ha='left', fontweight='bold')
    fig.savefig(out / 'push_grid.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def pooled(rows, prefix, set_name, key='survived'):
    vals = [r[key] for r in rows if r['controller'].startswith(prefix) and r['set'] == set_name]
    return sum(vals), len(vals)


def fig_shifts(rows, out):
    names = list(SHIFTS)
    arms = ('home', 'random-push', 'ars-none', 'ars-push')
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for a_i, arm in enumerate(arms):
        ys, xs, lo, hi = [], [], [], []
        for i, s in enumerate(names):
            k, n = pooled(rows, arm, f'shift:{s}')
            if not n:
                continue
            l, h = wilson(k, n)
            ys.append(i + (a_i - 1.5) * 0.17); xs.append(k / n); lo.append(k / n - l); hi.append(h - k / n)
        ax.errorbar(xs, ys, xerr=[lo, hi], fmt='o', ms=6, color=ARM_COLORS[arm], ecolor=ARM_COLORS[arm], elinewidth=1.2,
                    capsize=0, label=ARM_LABELS[arm])
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names); ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.02)
    ax.set_xlabel('survival fraction, pooled over seeds (8 directions × {0, .1, .2, .3} m/s); bars: Wilson 95% CI')
    ax.set_title('Unseen physics and sensor changes (paired: identical seeds and pushes in every row)', loc='left')
    ax.legend(loc='lower center', bbox_to_anchor=(0.45, -0.38), ncol=2)
    fig.tight_layout()
    fig.savefig(out / 'shifts.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_reflex(runs, out):
    names = [n for n in runs if arm_of(n) == 'ars-push']
    thetas = [runs[n]['thetas']['final'] for n in names]
    vmax = max(np.abs(t).max() for t in thetas)
    panels = list(zip(names, thetas))
    if len(thetas) > 1:
        panels.append((f'mean of {len(thetas)} seeds', np.mean(thetas, axis=0)))
    agree = np.all(np.sign(thetas) == np.sign(thetas[0]), axis=0) if len(thetas) > 1 else None
    fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 1, 4.8), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (n, th) in zip(axes, panels):
        im = ax.imshow(th, cmap=DIV_CMAP, norm=TwoSlopeNorm(0, -vmax, vmax), aspect='auto')
        ax.set_xticks(range(6)); ax.set_xticklabels(['tilt fwd\n(g_x)', 'tilt side\n(g_y)', 'roll\nrate', 'pitch\nrate', 'yaw\nrate', 'posture\nbias'], fontsize=7.5)
        sub = (f'mirror-symmetric share {symmetry_fraction(th):.2f}' if not n.startswith('mean') else
               f'dot: same sign in all seeds ({int(agree.sum())}/84; chance ≈ {84 * 2 * 0.5 ** len(thetas):.0f})')
        ax.set_title(f'{n}\n{sub}', loc='left', fontsize=9)
        if n.startswith('mean'):
            yy, xx = np.nonzero(agree)
            ax.plot(xx, yy, 'o', ms=3, color=INK)
        ax.grid(False)
    axes[0].set_yticks(range(14)); axes[0].set_yticklabels(JOINTS, fontsize=8)
    cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.01)
    cb.set_label('rad per unit feature (tilt unit 0.1 ≈ 5.7°, rate unit 1 rad/s)', color=INK2)
    fig.suptitle('Learned reflexes (ARS, pushed experience): joint-target offset = W · IMU features + bias', x=0.01, y=1.03, ha='left', fontweight='bold')
    fig.savefig(out / 'reflex.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def fig_joint_importance(rows, out):
    seeds = sorted({r['controller'].split('/', 1)[0] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
    for ax, (title, pred) in zip(axes, (('Unpushed trials', lambda r: r['condition']['push_time_s'] is None),
                                        ('Pushed trials (0.1-0.3 m/s)', lambda r: r['condition']['push_time_s'] is not None))):
        for s_i, s in enumerate(seeds):
            full = np.mean([r['survived'] for r in rows if r['controller'] == f'{s}/full' and pred(r)])
            xs = [np.mean([r['survived'] for r in rows if r['controller'] == f'{s}/drop:{j}' and pred(r)]) - full for j in JOINTS]
            ax.plot(xs, np.arange(14) + (s_i - 1) * 0.15, 'o', ms=6, color=ARM_COLORS['ars-push'], alpha=[1.0, 0.7, 0.45][s_i % 3], label=s)
        ax.axvline(0, color=INK2, lw=1)
        ax.set_title(title, loc='left')
        ax.set_xlabel('survival change when that joint\'s reflex row is removed')
        ax.set_xlim(-1.05, 0.3)
    axes[0].set_yticks(range(14)); axes[0].set_yticklabels(JOINTS); axes[0].invert_yaxis()
    axes[1].legend(loc='lower left')
    fig.suptitle('Which joints carry the learned reflex? (drop one joint row at a time; held-out nominal conditions)', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / 'joint_importance.png', dpi=150)
    plt.close(fig)


def fig_ablation(rows, out):
    variants = []
    for r in rows:
        v = r['controller'].split('/', 1)[1]
        if v not in variants and not v.startswith('drop:'):
            variants.append(v)
    seeds = sorted({r['controller'].split('/', 1)[0] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (title, pred) in zip(axes, (('Unpushed (20 s stands + unpushed nominal grid)', lambda r: r['condition']['push_time_s'] is None),
                                        ('Pushed (nominal grid, 0.1-0.3 m/s)', lambda r: r['condition']['push_time_s'] is not None))):
        for s_i, s in enumerate(seeds):
            xs = []
            for v in variants:
                sel = [r['survived'] for r in rows if r['controller'] == f'{s}/{v}' and pred(r)]
                xs.append(np.mean(sel) if sel else np.nan)
            ax.plot(xs, np.arange(len(variants)) + (s_i - 1) * 0.15, 'o', ms=6, color=ARM_COLORS['ars-push'],
                    alpha=[1.0, 0.7, 0.45][s_i % 3], label=s)
        ax.set_title(title, loc='left')
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel('survival fraction')
    axes[0].set_yticks(range(len(variants))); axes[0].set_yticklabels(variants); axes[0].invert_yaxis()
    axes[1].legend(loc='lower right')
    fig.suptitle('Mechanism: remove one ingredient of each learned reflex', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / 'ablation.png', dpi=150)
    plt.close(fig)


def record(sim, theta, cond):
    s, rec = sim02b.rollout(sim, theta, cond, record=True, terminate=False)
    return s, rec


def fig_response(sim, runs, out, ref='ars-push-s1'):
    theta = runs[ref]['thetas']['final']
    zero = sim02b.zero_theta()
    base = dict(seed=910_000_001, duration_s=5.0, joint_noise_rad=0.0)
    cases = [('HOME, no push', zero, Condition(**base)),
             (f'{ref}, forward 0.2 m/s', theta, Condition(**base, push_time_s=2.0, push_dv=(0.2, 0.0))),
             (f'{ref}, backward 0.2 m/s', theta, Condition(**base, push_time_s=2.0, push_dv=(-0.2, 0.0))),
             (f'{ref}, left 0.3 m/s', theta, Condition(**base, push_time_s=2.0, push_dv=(0.0, 0.3)))]
    fig, axes = plt.subplots(4, len(cases), figsize=(13, 8.2), sharex=True)
    summaries = {}
    j_idx = {n: i for i, n in enumerate(JOINTS)}
    for c, (title, th, cond) in enumerate(cases):
        s, rec = record(sim, th, cond)
        summaries[title] = {k: s[k] for k in ('survived', 'fall_time_s', 'max_tilt_deg', 'n_steps', 'transitions', 'peak_torque_nm', 'sat_frac')}
        t = rec['time_s']
        alive = t <= (s['fall_time_s'] or 1e9) + 0.3
        ax = axes[0, c]
        ax.plot(t[alive], rec['tilt_deg'][alive], color=INK)
        ax.axhline(sim02b.FALL_TILT_DEG, color=GRID, lw=1)
        ax.set_title(title + ('\nsurvived' if s['survived'] else f"\nfell at {s['fall_time_s']:.2f} s"), loc='left', fontsize=9)
        ax.set_ylim(0, 60)
        com = rec['com'] - rec['com'][0]
        axes[1, c].plot(t[alive], 1000 * com[alive, 0], color=ARM_COLORS['ars-push'], label='COM x (forward)')
        axes[1, c].plot(t[alive], 1000 * com[alive, 1], color=ARM_COLORS['random-push'], label='COM y (left)')
        axes[1, c].set_ylim(-40, 40)
        off = rec['qpos'][:, 7:] - sim02b.HOME
        for jn, col in (('neck_pitch', '#6250d6'), ('left_hip_pitch', '#1baf7a'), ('left_ankle', '#eda100')):
            axes[2, c].plot(t[alive], off[alive, j_idx[jn]], color=col, lw=1.5, label=jn)
        axes[2, c].set_ylim(-0.25, 0.25)
        f = rec['foot_force_n']
        axes[3, c].plot(t[alive], f[alive, 0], color=ARM_COLORS['ars-push'], lw=1.5, label='left sole')
        axes[3, c].plot(t[alive], f[alive, 1], color=ARM_COLORS['random-push'], lw=1.5, label='right sole')
        axes[3, c].set_ylim(-0.5, 9)
        axes[3, c].set_xlabel('time (s)')
        if cond.push_time_s:
            for r_ in range(4):
                axes[r_, c].axvline(cond.push_time_s, color=INK2, lw=0.8, ls=':')
    axes[0, 0].set_ylabel('trunk tilt (deg)')
    axes[1, 0].set_ylabel('COM shift (mm)')
    axes[2, 0].set_ylabel('measured joint\nchange from HOME (rad)')
    axes[3, 0].set_ylabel('sole normal force (N)')
    for r_ in (1, 2, 3):
        axes[r_, -1].legend(loc='upper right', fontsize=7)
    fig.suptitle('What the robot does: fixed HOME falls; the learned reflex stands and absorbs some pushes (dotted line: push)', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / 'response.png', dpi=150)
    plt.close(fig)
    return summaries


def fig_evolution(sim, runs, out, ref='ars-push-s1'):
    th = runs[ref]['thetas']['thetas']
    its = [i for i in (0, 1, 2, 3, 4, 6, 10, 20, 60) if i < len(th)]
    conds = [('stand (no push)', Condition(seed=920_000_001, duration_s=6.0, joint_noise_rad=0.02)),
             ('forward 0.15 m/s at 3 s', Condition(seed=920_000_002, duration_s=6.0, joint_noise_rad=0.02, push_time_s=3.0, push_dv=(0.15, 0.0))),
             ('right 0.3 m/s at 3 s', Condition(seed=920_000_003, duration_s=6.0, joint_noise_rad=0.02, push_time_s=3.0, push_dv=(0.0, -0.3)))]
    cmap = LinearSegmentedColormap.from_list('it', ['#9ec5f4', '#0d366b'])
    fig, axes = plt.subplots(1, len(conds), figsize=(12, 3.6), sharey=True)
    table = {}
    for ax, (title, cond) in zip(axes, conds):
        for n_i, it in enumerate(its):
            s, rec = record(sim, th[it], cond)
            t = rec['time_s']
            alive = t <= (s['fall_time_s'] or 1e9)
            ax.plot(t[alive], rec['tilt_deg'][alive], color=cmap(n_i / max(1, len(its) - 1)), lw=1.6,
                    label=f'after {it} updates' if ax is axes[0] else None)
            if not s['survived']:
                ax.plot([t[alive][-1]], [rec['tilt_deg'][alive][-1]], 'x', color=cmap(n_i / max(1, len(its) - 1)), ms=7)
            table.setdefault(title, {})[it] = dict(survived=s['survived'], fall_time_s=s['fall_time_s'], max_tilt=s['max_tilt_deg'])
        ax.set_title(title, loc='left')
        ax.set_xlabel('time (s)')
        ax.set_ylim(0, 50)
        if cond.push_time_s:
            ax.axvline(cond.push_time_s, color=INK2, lw=0.8, ls=':')
    axes[0].set_ylabel('trunk tilt (deg); × = fall (45°)')
    axes[0].legend(fontsize=7, loc='upper right')
    fig.suptitle(f'Same held-out trials replayed with {ref} reflex snapshots: experience changes the movement', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / 'evolution.png', dpi=150)
    plt.close(fig)
    return table


STANCE_ITS = (1, 2, 3, 5, 10, 20, 40, 60)


def stance_quality(runs, out, workers=11):
    """Quiet-stance tilt of reflex snapshots on 8 held-out unpushed 6 s trials.
    Cached in report/stance.json (deterministic; delete the file to recompute)."""
    cache = out / 'stance.json'
    if cache.exists():
        return json.loads(cache.read_text())
    from concurrent.futures import ProcessPoolExecutor
    from movement_lab import eval02b
    conds = [Condition(seed=940_000_001 + i, duration_s=6.0, joint_noise_rad=0.02) for i in range(8)]
    tasks, keys = [], []
    for name, r in runs.items():
        th = r['thetas']['thetas']
        for it in STANCE_ITS:
            if it < len(th):
                for c in conds:
                    tasks.append((name, th[it], 'stance', c)); keys.append((name, it))
    with ProcessPoolExecutor(workers, initializer=eval02b._init, initargs=(str(sim02b.DEFAULT_MODEL),)) as pool:
        res = list(pool.map(eval02b._run, tasks, chunksize=4))
    table = {}
    for (name, it), r in zip(keys, res):
        d = table.setdefault(name, {}).setdefault(str(it), dict(final_tilt=[], survived=[]))
        d['final_tilt'].append(r['final_tilt_deg']); d['survived'].append(r['survived'])
    cache.write_text(json.dumps(table))
    return table


def restoring_gain(sim, theta):
    """Commanded foot pitch relative to the trunk (deg, mean of both soles) produced
    by the reflex's response to one feature unit of forward tilt (sin = 0.1, ~5.7 deg).
    Positive = with soles planted the command counter-rotates the trunk toward upright.
    This is a commanded kinematic quantity; compliant servos realise it as torque."""
    import mujoco
    m, d = sim.m, sim.aux

    def feet(offset):
        d.qpos[:] = 0; d.qpos[2] = 0.2; d.qpos[3:7] = (1, 0, 0, 0)
        d.qpos[sim.qadr] = sim02b.HOME + offset
        mujoco.mj_kinematics(m, d)
        rt = d.xmat[sim.trunk].reshape(3, 3)
        return [rt.T @ d.xmat[b].reshape(3, 3) for b in sim.foot_bodies]
    r0, r1 = feet(np.zeros(14)), feet(np.asarray(theta)[:, 0])
    return float(np.mean([math.degrees(math.atan2((a @ b.T)[0, 2], (a @ b.T)[0, 0])) for a, b in zip(r1, r0)]))


def fig_stance(table, runs, sim, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    gains = {}
    for name, d in table.items():
        its = sorted(int(k) for k in d)
        ys = [np.median([t for t, s in zip(d[str(i)]['final_tilt'], d[str(i)]['survived']) if s] or [np.nan]) for i in its]
        g = [restoring_gain(sim, runs[name]['thetas']['thetas'][i]) for i in its]
        gains[name] = dict(zip(map(str, its), g))
        axes[0].plot(its, g, marker='o', ms=3, lw=1.4, color=ARM_COLORS[arm_of(name)], alpha=0.85)
        axes[1].plot(its, ys, marker='o', ms=3, lw=1.4, color=ARM_COLORS[arm_of(name)], alpha=0.85)
    for ax in axes:
        ax.set_xscale('log')
        ax.set_xticks(STANCE_ITS); ax.set_xticklabels(STANCE_ITS)
        ax.set_xlabel('reflex after N updates (random search: best so far)')
    axes[0].axhline(0, color=INK2, lw=1)
    axes[0].set_ylabel('commanded foot counter-rotation\n(deg per 5.7° forward tilt)')
    axes[0].set_title('What experience changed: the sagittal restoring gain', loc='left', fontsize=9.5)
    axes[1].set_ylabel('median final trunk tilt (deg)')
    axes[1].set_ylim(0, None)
    axes[1].set_title('Stance tilt on 8 held-out unpushed 6 s trials (survivors)', loc='left', fontsize=9.5)
    handles = [plt.Line2D([], [], color=ARM_COLORS[a], marker='o', ms=4) for a in ('ars-push', 'random-push', 'ars-none')]
    axes[1].legend(handles, [ARM_LABELS[a] for a in ('ars-push', 'random-push', 'ars-none')], fontsize=8)
    fig.tight_layout()
    fig.savefig(out / 'stance.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return gains


def own_bounds(sim, theta, policy=None):
    """Capture-point no-step bounds (12 eval directions) of the stance a controller
    actually holds: final state of a noise-free 3 s unpushed trial, velocities zeroed."""
    import mujoco
    from movement_lab.physics02b import capture_bounds
    dirs = [2 * math.pi * i / 12 for i in range(12)]
    s, rec = sim02b.rollout(sim, theta, Condition(seed=3, duration_s=3.0, gyro_noise=0.0, gravity_noise=0.0),
                            record=True, terminate=False, policy=policy)
    d = sim.aux
    d.qpos[:] = rec['qpos'][-1]
    d.qvel[:] = 0
    mujoco.mj_forward(sim.m, d)
    b = capture_bounds(sim, d, dirs)
    return {int(round(math.degrees(a))) % 360: float(v) for a, v in zip(dirs, b['bounds'])}


def limit_split(rows, bounds_by_ctrl, home_bounds):
    """Survival inside/beyond the no-step limit, by HOME stance and by own stance."""
    out = {}
    for r in rows:
        if r['set'] != 'push_grid':
            continue
        c = r['controller']
        dv = r['condition']['push_dv']
        m = math.hypot(*dv)
        d = int(round(math.degrees(math.atan2(dv[1], dv[0])))) % 360
        o = out.setdefault(c, {k: [0, 0, 0] for k in ('home_inside', 'home_beyond', 'own_inside', 'own_beyond')})
        for ref, bnd in (('home', home_bounds), ('own', bounds_by_ctrl.get(c, home_bounds))):
            k = o[f"{ref}_{'beyond' if m > bnd[d] + 1e-9 else 'inside'}"]
            k[0] += 1; k[1] += r['survived']; k[2] += int(r['survived'] and r['n_steps'] > 0)
    return out


def fig_margins(own, ev_rows, out):
    """Where each learned reflex stands, and what that predicts."""
    names = [n for n in own if n != 'home']
    pooled_rate = lambda c, s: np.mean([r['survived'] for r in ev_rows if r['controller'] == c and r['set'] == s])
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    home = own.get('home')
    for n in names:
        col = ARM_COLORS[arm_of(n)]
        fwd, back = own[n]['0'], own[n]['180']
        axes[0].plot(back, fwd, 'o', ms=8, color=col)
        axes[0].annotate(n.rsplit('-', 1)[1], (back, fwd), textcoords='offset points', xytext=(5, 3), fontsize=7, color=INK2)
        axes[1].plot(back, pooled_rate(n, 'repeated'), 'o', ms=8, color=col)
        axes[2].plot(back, pooled_rate(n, 'shift:imu_pitch_+5deg'), 'o', ms=8, color=col)
    if home:
        axes[0].plot(home['180'], home['0'], 's', ms=8, color=ARM_COLORS['home'])
        axes[0].annotate('HOME stance', (home['180'], home['0']), textcoords='offset points', xytext=(5, -10), fontsize=7, color=INK2)
    axes[0].set_xlabel('own-stance backward no-step limit (m/s)'); axes[0].set_ylabel('own-stance forward limit (m/s)')
    axes[0].set_title('Learned stances trade toe for heel margin', loc='left', fontsize=9.5)
    axes[1].set_xlabel('own-stance backward limit (m/s)'); axes[1].set_ylabel('survival, 3 repeated 0.15 m/s pushes')
    axes[1].set_title('Heel margin predicts repeated-push survival', loc='left', fontsize=9.5)
    axes[2].set_xlabel('own-stance backward limit (m/s)'); axes[2].set_ylabel('survival with IMU pitched +5°')
    axes[2].set_title('…and tolerance of IMU pitch error', loc='left', fontsize=9.5)
    for ax in axes[1:]:
        ax.set_ylim(-0.05, 1.05)
    handles = [plt.Line2D([], [], color=ARM_COLORS[a], marker='o', ls='', ms=7) for a in ('ars-push', 'random-push', 'ars-none')]
    fig.legend(handles, [ARM_LABELS[a] for a in ('ars-push', 'random-push', 'ars-none')], loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle('Capture-point limits of the stance each reflex actually holds (noise-free 3 s stand, held-out outcomes)', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / 'margins.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


def support_trace(sim, rec, every=4):
    """Per control period: capture point, COM projection, and the convex hull of
    soles currently touching the floor (each sole's floor footprint)."""
    import mujoco
    from scipy.spatial import ConvexHull
    from movement_lab.physics02b import convex_hull, inside, G
    m, aux = sim.m, sim.aux
    hulls = []
    for g in sim.feet:
        mesh = m.geom_dataid[g]
        a, n = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
        v = m.mesh_vert[a:a + n]
        hulls.append(v[ConvexHull(v).vertices])
    com = rec['com']
    vel = np.gradient(com, sim02b.DT, axis=0)
    out = []
    for i in range(every - 1, len(rec['time_s']), every):
        aux.qpos[:] = rec['qpos'][i]
        mujoco.mj_kinematics(m, aux)
        pts, soles = [], []
        for f, (g, hv) in enumerate(zip(sim.feet, hulls)):
            w = hv @ aux.geom_xmat[g].reshape(3, 3).T + aux.geom_xpos[g]
            fp = convex_hull(w[:, :2])
            soles.append(fp)
            if rec['foot_found'][i][f]:
                pts.append(w[w[:, 2] <= w[:, 2].min() + 0.002, :2])
        poly = convex_hull(np.concatenate(pts)) if pts else np.zeros((0, 2))
        w0 = math.sqrt(G / max(com[i, 2], 1e-3))
        cp = com[i, :2] + vel[i, :2] / w0
        ok = len(poly) >= 3 and inside(poly, cp)
        out.append(dict(t=float(rec['time_s'][i]), cp=cp, com=com[i, :2].copy(), poly=poly, soles=soles, cp_inside=bool(ok),
                        contact=tuple(bool(x) for x in rec['foot_found'][i]), tilt=float(rec['tilt_deg'][i])))
    return out


def fig_step(sim, runs, out, step_run='ars-push-s3', ref_run='ars-none-s3', seed=900000208, angle_deg=240, mag=0.4):
    a = math.radians(angle_deg)
    cond = Condition(seed=seed, duration_s=5.0, joint_noise_rad=0.02, push_time_s=3.0, push_dv=(mag * math.cos(a), mag * math.sin(a)))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    table = {}
    for ax, run, col in ((axes[0], step_run, ARM_COLORS['ars-push']), (axes[1], ref_run, ARM_COLORS['ars-none'])):
        s, rec = record(sim, runs[run]['thetas']['final'], cond)
        tr = support_trace(sim, rec)
        win = [x for x in tr if 2.95 <= x['t'] <= (s['fall_time_s'] or 4.0)]
        for x, ls in ((win[0], '-'), (win[-1], '--')):
            for f, fp in enumerate(x['soles']):
                ax.fill(*np.r_[fp, fp[:1]].T * 1000, color=GRID if ls == '-' else 'none', ec=INK2, ls=ls, lw=1.2)
        ax.plot(*np.array([x['com'] for x in win]).T * 1000, color=INK2, lw=1.4, label='COM floor projection')
        ax.plot(*np.array([x['cp'] for x in win]).T * 1000, color=col, lw=2, label='capture point')
        outside = [x for x in win if not x['cp_inside']]
        if outside:
            ax.plot(*np.array([x['cp'] for x in outside]).T * 1000, 'o', ms=3, color=ARM_COLORS['random-push'], label='capture point outside support')
        ax.set_aspect('equal')
        ax.set_title(f"{run}: {'survived' if s['survived'] else 'fell at %.2f s' % s['fall_time_s']}; step events {s['n_steps']}", loc='left', fontsize=9)
        ax.set_xlabel('world x (mm)'); ax.set_ylabel('world y (mm)')
        ax.legend(fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
        t = rec['time_s']
        alive = t <= (s['fall_time_s'] or 1e9)
        axes[2].plot(t[alive], rec['foot_force_n'][alive, 0], color=col, lw=1.6, ls='-', label=f'{run} left sole')
        axes[2].plot(t[alive], rec['foot_force_n'][alive, 1], color=col, lw=1.2, ls=':', label=f'{run} right sole')
        table[run] = dict(survived=s['survived'], fall_time_s=s['fall_time_s'], steps=s['steps'],
                          cp_outside_s=sum(sim02b.CONTROL_DT for x in win if not x['cp_inside']))
    axes[2].set_xlim(2.9, 3.8)
    axes[2].set_xlabel('time (s)'); axes[2].set_ylabel('sole normal force (N)')
    axes[2].axvline(3.0, color=INK2, lw=0.8, ls=':')
    axes[2].set_title('Sole loading after the push (dotted: push)', loc='left', fontsize=9)
    axes[2].legend(fontsize=7)
    fig.suptitle(f'Held-out push {mag} m/s at {angle_deg}° (right and slightly back; no-step limit ≈ 0.30 m/s here). Same trial, two reflexes. '
                 'Grey: soles at the push; dashed: soles at the end of the window', x=0.01, ha='left', fontweight='bold', fontsize=9.5)
    fig.tight_layout()
    fig.savefig(out / 'step_event.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return table


REV_COLORS = {'warm start (ars-push)': '#8a8984', 'linear': '#2a78d6', 'residual': '#eb6834', 'residual removed': '#1baf7a'}


def fig_revision(root, out):
    """Revision learning curves and held-out push-grid outcomes, if present."""
    rev = root / 'revision'
    runs = sorted(d for d in rev.iterdir() if (d / 'final.json').exists()) if rev.exists() else []
    if not runs:
        return None
    ev_rows, _ = load_eval(root / 'eval/revision')
    fig, axes = plt.subplots(1, 2 if ev_rows else 1, figsize=(11, 3.8))
    axes = np.atleast_1d(axes)
    for d in runs:
        cfg = json.loads((d / 'config.json').read_text())
        rows = [json.loads(l) for l in (d / 'progress.jsonl').read_text().splitlines()]
        per_it = 2 * cfg['directions'] * cfg['conditions']
        pr = [r for r in rows if r['kind'] == 'probe']
        arm = cfg['arm']
        axes[0].plot([p['it'] * per_it for p in pr], [p['survival_push'] for p in pr], marker='o', ms=3,
                     color=REV_COLORS[arm], label=f"{d.name} ({cfg['n_params']} params)")
    axes[0].set_title('Pushed-probe survival during hard-push training', loc='left')
    axes[0].set_xlabel('revision rollouts (warm start at 0)')
    axes[0].set_ylabel('probe survival (32 pushed probes)')
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    table = {}
    if ev_rows:
        ctrls = []
        for r in ev_rows:
            if r['controller'] not in ctrls:
                ctrls.append(r['controller'])
        mags = np.round(np.arange(0.05, 0.501, 0.05), 3)
        for c in ctrls:
            sel = [r for r in ev_rows if r['controller'] == c and r['set'] == 'push_grid']
            ys = [np.mean([r['survived'] for r in sel if abs(math.hypot(*r['condition']['push_dv']) - m) < 1e-6]) for m in mags]
            key = ('warm start (ars-push)' if c.startswith('warm') else 'residual removed' if 'removed' in c
                   else 'residual' if c.startswith('residual') else 'linear')
            axes[1].plot(mags, ys, marker='o', ms=4, color=REV_COLORS[key], label=c)
            table[c] = dict(push_grid=float(np.mean([r['survived'] for r in sel])), by_mag=dict(zip(map(str, mags), map(float, ys))),
                            steps_in_survivors=int(sum(r['n_steps'] for r in sel if r['survived'])))
        axes[1].set_title('Held-out push grid by magnitude (12 directions × 2 seeds)', loc='left')
        axes[1].set_xlabel('|Δv| (m/s)')
        axes[1].set_ylabel('survival fraction')
        axes[1].set_ylim(-0.02, 1.02)
        axes[1].legend(fontsize=8)
    fig.suptitle('Targeted revision: experience beyond the no-step limit, with and without extra capacity', x=0.01, ha='left', fontweight='bold')
    fig.tight_layout()
    fig.savefig(out / 'revision.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    return table


# --------------------------------------------------------------------------- tables
def tables(runs, ev_rows, ab_rows):
    t = {}
    t['runs'] = {n: dict(r['final'], first_full_standing_probe_it=next((p['it'] for p in r['rows'] if p['kind'] == 'probe' and p['survival_none'] == 1.0), None),
                         final_probe=[p for p in r['rows'] if p['kind'] == 'probe'][-1] | {'outcomes': None},
                         symmetry=symmetry_fraction(r['thetas']['final']))
                 for n, r in runs.items()}
    if ev_rows:
        arms = ('home', 'random-push', 'ars-none', 'ars-push')
        sets = sorted({r['set'] for r in ev_rows})
        t['eval_pooled'] = {}
        for a in arms:
            for s in sets:
                k, n = pooled(ev_rows, a, s)
                if n:
                    rec, _ = pooled(ev_rows, a, s, 'recovered')
                    t['eval_pooled'].setdefault(a, {})[s] = dict(survived=k, n=n, survival=k / n, ci95=wilson(k, n), recovered=rec)
        # Paired comparison on identical held-out push-grid conditions.
        def outcomes(prefix):
            d = {}
            for r in ev_rows:
                if r['set'] == 'push_grid' and r['controller'].startswith(prefix):
                    d[(r['controller'].rsplit('-s', 1)[1], r['condition']['seed'])] = r['survived']
            return d
        pairs = {}
        for a, b in (('ars-push', 'random-push'), ('ars-push', 'ars-none'), ('ars-none', 'random-push')):
            oa, ob = outcomes(a), outcomes(b)
            keys = sorted(set(oa) & set(ob))
            pairs[f'{a} vs {b}'] = dict(n=len(keys), only_a=sum(oa[k] and not ob[k] for k in keys),
                                        only_b=sum(ob[k] and not oa[k] for k in keys), both=sum(oa[k] and ob[k] for k in keys))
        t['paired_push_grid'] = pairs
        per_seed = {}
        for r in ev_rows:
            if r['set'] == 'push_grid':
                per_seed.setdefault(r['controller'], []).append(r['survived'])
        t['push_grid_per_controller'] = {k: float(np.mean(v)) for k, v in per_seed.items()}
        steps = [r for r in ev_rows if r['n_steps'] > 0 and r['controller'] != 'home']
        t['step_events'] = dict(trials_with_step=len(steps), survived_with_step=sum(r['survived'] for r in steps),
                                n_trials=len([r for r in ev_rows if r['controller'] != 'home']))
        t['effort'] = {a: dict(max_sat_frac=max(r['sat_frac'] for r in ev_rows if r['controller'].startswith(a)),
                               max_peak_torque_nm=max(r['peak_torque_nm'] for r in ev_rows if r['controller'].startswith(a)),
                               max_pen_alive_mm=1000 * max(r['max_pen_alive_m'] for r in ev_rows if r['controller'].startswith(a)),
                               invalid=sum(not r['valid'] for r in ev_rows if r['controller'].startswith(a)))
                       for a in ('home', 'random-push', 'ars-none', 'ars-push')}
    if ab_rows:
        ab = {}
        for r in ab_rows:
            seed, var = r['controller'].split('/', 1)
            key = 'pushed' if r['condition']['push_time_s'] is not None else 'unpushed'
            ab.setdefault(var, {}).setdefault(key, []).append(r['survived'])
        t['ablation'] = {v: {k: dict(survival=float(np.mean(x)), n=len(x)) for k, x in d.items()} for v, d in ab.items()}
    return t


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02b'))
    p.add_argument('--skip-sim', action='store_true', help='skip re-simulated figures')
    args = p.parse_args(argv)
    style()
    out = args.root / 'report'
    out.mkdir(parents=True, exist_ok=True)
    runs = load_runs(args.root)
    ev_rows, _ = load_eval(args.root / 'eval/final')
    ab_rows, _ = load_eval(args.root / 'eval/ablation')
    fig_learning(runs, out)
    sim = sim02b.Sim()
    from movement_lab.physics02b import capture_bounds
    sim.reset(Condition(), np.random.default_rng(0))
    bounds = capture_bounds(sim, sim.d, list(np.arange(0, 2 * np.pi + 1e-9, np.pi / 36)))
    t = tables(runs, ev_rows, ab_rows)
    t['capture_bounds_home'] = bounds
    if ev_rows:
        fig_push_grid(ev_rows, bounds, out)
        fig_shifts(ev_rows, out)
    if ab_rows:
        fig_ablation(ab_rows, out)
        fig_joint_importance(ab_rows, out)
    fig_reflex(runs, out)
    t['revision'] = fig_revision(args.root, out)
    if not args.skip_sim:
        t['response'] = fig_response(sim, runs, out)
        t['evolution'] = fig_evolution(sim, runs, out)
        t['step_event'] = fig_step(sim, runs, out)
        home_b = {int(round(math.degrees(a))) % 360: float(v) for a, v in zip(bounds['directions_rad'], bounds['bounds'])
                  if int(round(math.degrees(a))) % 30 == 0}
        own = {name: own_bounds(sim, r['thetas']['final']) for name, r in runs.items()}
        own['home'] = home_b
        own = {k: {str(d): v for d, v in b.items()} for k, b in own.items()}
        t['own_stance_bounds'] = own
        if ev_rows:
            t['limit_split'] = limit_split(ev_rows, {k: {int(d): v for d, v in b.items()} for k, b in own.items()}, home_b)
            fig_margins(own, ev_rows, out)
            from scipy.stats import spearmanr
            names = [n for n in own if n != 'home']
            back = [own[n]['180'] for n in names]
            t['heel_margin_correlation'] = {
                s: dict(zip(('spearman_rho', 'p'), map(float, spearmanr(back, [np.mean([r['survived'] for r in ev_rows if r['controller'] == n and r['set'] == s]) for n in names]))))
                for s in ('push_grid', 'repeated', 'shift:imu_pitch_+5deg', 'shift:nominal')}
        rev_rows, _ = load_eval(args.root / 'eval/revision')
        if rev_rows:
            from movement_lab.eval02b import load_theta
            from movement_lab.revise02b import Residual
            rown = {}
            for c in {r['controller'] for r in rev_rows}:
                s = c.rsplit('-s', 1)[1].split('-')[0]
                spec = {'warm': f'{args.root}/runs/ars-push-s{s}/thetas.npz', 'linear': f'linear:{args.root}/revision/linear-s{s}/snapshots.npz',
                        'residual': f'residual:{args.root}/revision/residual-s{s}/snapshots.npz'}[c.split('-')[0]]
                if c.endswith('removed'):
                    spec = f'linear:{args.root}/revision/residual-s{s}/snapshots.npz'
                th = load_theta(spec)
                rown[c] = (own_bounds(sim, sim02b.zero_theta(), policy=Residual(th[1])) if isinstance(th, tuple) else own_bounds(sim, th))
            t['own_stance_bounds_revision'] = rown
            t['limit_split_revision'] = limit_split(rev_rows, rown, home_b)
        st = stance_quality(runs, out)
        t['restoring_gain_deg'] = fig_stance(st, runs, sim, out)
        th = np.array([runs[f'ars-push-s{s}']['thetas']['final'] for s in (1, 2, 3) if f'ars-push-s{s}' in runs])
        agree = np.all(np.sign(th) == np.sign(th[0]), axis=0)
        t['sign_agreement_ars_push'] = dict(total=int(agree.sum()), per_feature=dict(zip((*FEATURES, 'bias'), map(int, agree.sum(axis=0)))),
                                            tilt_fwd_joints=[JOINTS[j] for j in range(14) if agree[j, 0]],
                                            chance_total=84 * 2 * 0.5 ** len(th))
        t['stance'] = {n: {i: dict(median_final_tilt=float(np.median(v['final_tilt'])), survived=int(sum(v['survived'])), n=len(v['survived']))
                           for i, v in d.items()} for n, d in st.items()}
    (out / 'tables.json').write_text(json.dumps(t, indent=1, default=float))
    print('wrote', sorted(x.name for x in out.iterdir()))


if __name__ == '__main__':
    main()
