"""02E Phase 1 batteries (declared in results/02e/PROTOCOL.md). No learning here.

    python -m movement_lab.diag02e feasibility  --root results/02e   # pre-protocol probes, reproduced
    python -m movement_lab.diag02e characterize --root results/02e   # 1A: 9 controllers x 150
    python -m movement_lab.diag02e estimators   --root results/02e   # 1B: what the sensors can tell
    python -m movement_lab.diag02e pilot        --root results/02e   # 1C: front-ends on battery D
    python -m movement_lab.diag02e gate         --root results/02e   # 1C decision

Every trial is recorded at 200 Hz and audited with the unchanged 02D rules; only
per-trial summary rows are kept (deterministic gzip). Evaluation identities pin
sources, model, controllers and exact conditions; completed files are reused,
mismatches refused.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time
import numpy as np

from movement_lab.sim02b import Sim, DEFAULT_MODEL, CONTROL_DT, rpy_matrix
from movement_lab.sim02c import BASE
from movement_lab.sim02d import event_time
from movement_lab.sim02e import Condition, rollout, mounting
from movement_lab.control02d import Design, load_basis, OPTION
from movement_lab.estimate02e import Kinematics, Frontend, Settings, rotation_fit, rotvec
from movement_lab.eval02c import write_gzip, read_rows
from movement_lab.learn02c import digest

HIST = Path('results/02d')
KINDS = ('quiet', 'fwd-.20', 'fwd-.32', 'back-.08', 'false-gyro')
KINDS_H = ('quiet', 'fwd-.08', 'fwd-.20', 'fwd-.28', 'fwd-.32', 'back-.08', 'left-.12', 'false-gyro')
SOURCES = ('diag02e.py', 'sim02e.py', 'estimate02e.py', 'sim02d.py', 'sim02c.py', 'sim02b.py', 'control02d.py',
           'eval02c.py', 'eval02d.py', 'physics02b.py')


def pitch(v):
    return dict(imu_rpy_deg=(0., float(v), 0.))


SETTINGS_1A = [('nominal', {}), *[(f'pitch{v:+g}', pitch(v)) for v in (-4, -2, -1, -.5, .5, 1, 2, 4)],
               ('roll-2', dict(imu_rpy_deg=(-2., 0., 0.))), ('roll+2', dict(imu_rpy_deg=(2., 0., 0.))),
               ('drift+2', dict(drift_deg=(0., 2., 0.))), ('drift-2', dict(drift_deg=(0., -2., 0.))),
               ('slope+2', dict(slope_deg=(0., 2.))), ('slope-2', dict(slope_deg=(0., -2.)))]
SETTINGS_D = [('nominal', {}), ('pitch-4', pitch(-4)), ('pitch-2', pitch(-2)), ('pitch+2', pitch(2)), ('pitch+4', pitch(4)),
              ('roll+2', dict(imu_rpy_deg=(2., 0., 0.))), ('slope-2', dict(slope_deg=(0., -2.))),
              ('slope+2', dict(slope_deg=(0., 2.))), ('enc+pitch+2', dict(imu_rpy_deg=(0., 2., 0.), encoder_offset_rad=.015))]
SETTINGS_H = [('nominal', {}), *[(f'pitch{v:+g}', pitch(v)) for v in (-4, -2, -1, 1, 2, 4)],
              ('roll-2', dict(imu_rpy_deg=(-2., 0., 0.))), ('roll+2', dict(imu_rpy_deg=(2., 0., 0.))),
              ('drift+2', dict(drift_deg=(0., 2., 0.))), ('drift-2', dict(drift_deg=(0., -2., 0.))),
              ('slope+2', dict(slope_deg=(0., 2.))), ('slope-2', dict(slope_deg=(0., -2.))),
              ('enc', dict(encoder_offset_rad=.015)), ('enc+pitch+2', dict(imu_rpy_deg=(0., 2., 0.), encoder_offset_rad=.015))]
MISALIGNED_D = ('pitch-4', 'pitch-2', 'pitch+2', 'pitch+4', 'roll+2', 'enc+pitch+2')
SETTINGS_1B = [('pitch-4', pitch(-4)), ('pitch-2', pitch(-2)), ('nominal', {}), ('pitch+2', pitch(2)), ('pitch+4', pitch(4)),
               ('roll-2', dict(imu_rpy_deg=(-2., 0., 0.))), ('roll+2', dict(imu_rpy_deg=(2., 0., 0.))),
               ('slope-2', dict(slope_deg=(0., -2.))), ('slope+2', dict(slope_deg=(0., 2.))), ('slope-roll+2', dict(slope_deg=(2., 0.))),
               ('pitch+2,slope+2', dict(imu_rpy_deg=(0., 2., 0.), slope_deg=(0., 2.))),
               ('pitch+2,slope-2', dict(imu_rpy_deg=(0., 2., 0.), slope_deg=(0., -2.))),
               ('enc', dict(encoder_offset_rad=.015)), ('enc+pitch+2', dict(imu_rpy_deg=(0., 2., 0.), encoder_offset_rad=.015))]
FRONTENDS_1C = ('none', 'gate', 'prior', 'static')


def make_condition(kind, seed, rep, setting, onset=(1.8, 3.4), duration=6.0):
    rng = np.random.default_rng(seed)
    t = float(rng.uniform(*onset))
    c = Condition(seed=seed, duration_s=duration, joint_noise_rad=.01, **setting)
    if kind.startswith('fwd-'):
        c.push_time_s, c.push_dv = t, (float(kind[4:]), 0.)
    elif kind.startswith('back-'):
        c.push_time_s, c.push_dv = t, (-float(kind[5:]), 0.)
    elif kind.startswith('left-'):
        c.push_time_s, c.push_dv = t, (0., float(kind[5:]))
    elif kind == 'false-gyro':
        c.false_gyro_time_s = round(t / .02) * .02
        c.false_gyro_rad_s = (-1 if rep % 2 else 1) * .75
    else:
        assert kind == 'quiet', kind
    return c


def battery(settings, kinds, base, reps=2, onset=(1.8, 3.4)):
    """(setting, kind, condition); seeds paired across settings, controllers and front-ends."""
    out = []
    for name, kw in settings:
        for j, kind in enumerate(kinds):
            for rep in range(reps):
                out.append((name, kind, make_condition(kind, base + j * 10 + rep, rep, kw, onset)))
    return out


def controllers(names=None):
    from movement_lab.eval02d import controllers as hist
    cs = [(n, v.tolist(), asdict(d), m) for n, v, d, m in hist(HIST)]
    return [c for c in cs if names is None or c[0] in names]


# ----------------------------------------------------------------------------- workers
SIM = THETA = BASIS = KIN = None


def init_worker():
    global SIM, THETA, BASIS, KIN
    SIM = Sim(); THETA = np.load(BASE)['final']; BASIS = load_basis(); KIN = Kinematics(SIM)
    if hasattr(os, 'nice'):
        os.nice(10)


def grav_deg(g):
    """(pitch, roll) in degrees of a gravity direction expressed in the trunk/sensor frame."""
    g = np.asarray(g)
    return np.degrees(np.arcsin(np.clip(g[..., 0], -1, 1))), np.degrees(-np.arcsin(np.clip(g[..., 1], -1, 1)))


def metrics(s, r, c, design_bound=.8):
    from movement_lab.eval02d import metrics as metrics02d
    metrics02d(s, r, c)
    t = r['time_s']
    ev = event_time(c)
    fail = s['failure_time_s']
    alive = t <= (fail if fail is not None else np.inf)
    steady = (t >= 1.5) & (t < (ev if ev is not None else c.duration_s)) & alive
    true_p, true_r = grav_deg(r['grav_true'])
    seen_p, seen_r = grav_deg(r['features'][:, :2] * .1)
    raw_p, raw_r = grav_deg(r['grav_raw'])
    mean = lambda x, m: float(np.mean(x[m])) if m.any() else None
    s['steady_true_pitch_deg'], s['steady_true_roll_deg'] = mean(true_p, steady), mean(true_r, steady)
    s['steady_seen_pitch_deg'], s['steady_seen_roll_deg'] = mean(seen_p, steady), mean(seen_r, steady)
    s['steady_raw_pitch_deg'] = mean(raw_p, steady)
    s['steady_coefficients'] = r['coefficients'][steady].mean(axis=0).tolist() if steady.any() else None
    s['steady_coefficient_abs_max'] = float(np.abs(r['coefficients'][steady]).max()) if steady.any() else None
    s['steady_residual_rms_rad'] = float(np.sqrt(np.mean(r['residual'][steady] ** 2))) if steady.any() else None
    limit = c.vin * SIM.kt_over_r if SIM is not None else None
    if limit is not None:
        s['torque_saturation_frac'] = float(np.mean(np.any(np.abs(r['torque_nm'][alive]) >= .98 * limit, axis=1))) if alive.any() else None
        at_limit = (r['target_cmd'] <= SIM.lo + 1e-9) | (r['target_cmd'] >= SIM.hi - 1e-9)
        clipped = np.abs(r['offset']) >= design_bound - 1e-9
        s['target_saturation_frac'] = float(np.mean(np.any(at_limit | clipped, axis=1)[alive])) if alive.any() else None
    mg = s.pop('margins', None)
    if mg is not None and len(mg):
        ok = mg[:, 0] <= (fail if fail is not None else np.inf)
        m_ok = mg[ok]
        pre = ok & (mg[:, 0] >= 1.5) & (mg[:, 0] < (ev if ev is not None else c.duration_s))
        s['min_support_margin_m'] = float(m_ok[:, 1].min()) if len(m_ok) else None
        s['min_toe_margin_m'] = float(m_ok[:, 2].min()) if len(m_ok) else None
        s['min_heel_margin_m'] = float(m_ok[:, 3].min()) if len(m_ok) else None
        s['steady_toe_margin_m'] = float(mg[pre, 2].mean()) if pre.any() else None
        s['steady_heel_margin_m'] = float(mg[pre, 3].mean()) if pre.any() else None
        s['com_outside_support_s'] = float(np.count_nonzero(m_ok[:, 1] < 0) * CONTROL_DT)
    # Settling: after the event, back within 1 deg of the pre-event tilt with angular speed < 0.5 rad/s for 0.25 s.
    s['settle_time_s'] = None
    if ev is not None and fail is None:
        pre = (t >= ev - .5) & (t < ev)
        ref = float(r['tilt_deg'][pre].mean())
        ok = (np.abs(r['tilt_deg'] - ref) <= 1.) & (np.linalg.norm(r['qvel'][:, 3:6], axis=1) < .5) & (t >= ev)
        run, hold = 0, int(round(.25 / .005))
        for i in np.flatnonzero(t >= ev):
            run = run + 1 if ok[i] else 0
            if run >= hold:
                s['settle_time_s'] = float(t[i - hold + 1] - ev)
                break
    final = t > c.duration_s - 1 + 1e-8
    s['final_true_pitch_deg'] = float(true_p[final].mean())
    s['final_root_displacement_m'] = float(np.linalg.norm(r['qpos'][-1, :2] - r['qpos'][0, :2]))
    return s


def evaluate(task):
    from movement_lab.eval02d import effective_mode
    name, vec, design, mode, fe, setting, kind, c = task
    design = Design(**design)
    frontend = Frontend(Settings(**fe), KIN) if fe is not None else None
    s, r = rollout(SIM, THETA, np.asarray(vec), c, design, record=True, terminate=False, mode=effective_mode(mode, c),
                   basis=BASIS, frontend=frontend, margins=True)
    metrics(s, r, c, design.bound)
    s['frontend_report'] = s.pop('frontend', None)
    s.update(controller=name, frontend=(fe or {}).get('kind', 'none'), setting=setting, set=kind, condition=asdict(c), mode=mode)
    return json.loads(json.dumps(s))


def run_battery(dest, configs, conds, workers):
    """configs: [(file stem, controller tuple, frontend settings dict or None)]."""
    dest.mkdir(parents=True, exist_ok=True)
    identity = dict(model_sha256=digest(DEFAULT_MODEL / 'model.mjb'), base_sha256=digest(BASE), option_sha256=digest(OPTION),
                    configurations=[dict(file=f, controller=c[0], weights=c[1], design=c[2], mode=c[3], frontend=fe) for f, c, fe in configs],
                    conditions=[dict(setting=s, kind=k, condition=asdict(c)) for s, k, c in conds],
                    sources={n: digest(Path('movement_lab') / n) for n in SOURCES})
    identity = json.loads(json.dumps(identity))
    path = dest / 'identity.json'
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise SystemExit(f'evaluation identity mismatch in {dest}; use a new root')
    else:
        path.write_text(json.dumps(identity, indent=1) + '\n', newline='\n')
    started = time.perf_counter()
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        for stem, (name, vec, design, mode), fe in configs:
            out = dest / f'{stem}.jsonl.gz'
            if out.exists():
                print('retained', out, flush=True)
                continue
            tasks = [(name, vec, design, mode, fe, s, k, c) for s, k, c in conds]
            rows = list(pool.map(evaluate, tasks, chunksize=2))
            tmp = out.with_suffix('.tmp'); write_gzip(tmp, rows); tmp.replace(out)
            print(f'{stem}: recovered {sum(r["recovered_200hz"] for r in rows)}/{len(rows)} '
                  f'({time.perf_counter() - started:.0f}s)', flush=True)


# ----------------------------------------------------------------------------- 1B estimators
def true_correction_deg(c, t):
    """Roll/pitch (deg) of the rotation that maps the misaligned sensor frame back to the trunk frame."""
    v = rotvec(mounting(c, t))
    return [float(np.degrees(v[0])), float(np.degrees(v[1]))]


def decompose(r, c, fe):
    """Offline error decomposition of a probe estimate from the 200 Hz record (diagnostic only)."""
    import mujoco
    s = fe['settings']
    single = s['kind'] == 'yaw' or not s['pitch_rad']
    steps = [k for k in range(int(round(s['start_s'] / CONTROL_DT)), int(round(s['end_s'] / CONTROL_DT))) if k >= 1]
    i_state = [4 * k - 1 for k in steps]      # recorded state at the start of control step k
    i_read = [4 * k for k in steps]            # sensor reading taken at the start of step k
    q = r['qpos'][i_state]
    w_true = r['qvel'][i_state, 3:6]           # free-joint angular velocity: trunk body frame
    w_kin = np.array([KIN.trunk_velocity(qq[SIM.qadr], vv[SIM.vadr]) for qq, vv in zip(q, r['qvel'][i_state])])
    gyro_noisy = r['gyro_raw'][i_read]
    gyro_clean = np.array([mounting(c, k * CONTROL_DT).T @ w for k, w in zip(steps, w_true)])
    out = {}
    for label, g, ref in (('oracle_trunk_clean_gyro', gyro_clean, w_true), ('oracle_trunk_noisy_gyro', gyro_noisy, w_true),
                          ('kinematic_true_joints', gyro_noisy, w_kin)):
        R, sv = rotation_fit(g, ref, single_axis=single)
        v = np.degrees(rotvec(R))
        out[label] = dict(roll_pitch_deg=[float(v[0]), float(v[1])], singular_values=sv)
    rms = lambda x: float(np.sqrt(np.mean(np.sum(x ** 2, axis=1))))
    out['trunk_rate_rms_rad_s'] = rms(w_true)
    out['kinematic_rate_error_rms_rad_s'] = rms(w_kin - w_true)   # what the planted-sole assumption gets wrong
    # Sole rotation relative to the window start (ground truth), the violated planted-foot assumption.
    d = mujoco.MjData(SIM.m)
    rot = []
    for i in i_state:
        d.qpos[:] = r['qpos'][i]; mujoco.mj_kinematics(SIM.m, d)
        rot.append([d.xmat[b].reshape(3, 3).copy() for b in SIM.foot_bodies])
    out['max_sole_rotation_deg'] = [float(max(np.degrees(np.linalg.norm(rotvec(rot[0][f].T @ x[f]))) for x in rot)) for f in range(2)]
    R0 = np.zeros(9); mujoco.mju_quat2Mat(R0, q[0, 3:7]); R0 = R0.reshape(3, 3)
    peak = 0.
    for qq in q[:, 3:7]:
        M = np.zeros(9); mujoco.mju_quat2Mat(M, qq); peak = max(peak, float(np.linalg.norm(rotvec(R0.T @ M.reshape(3, 3)))))
    out['trunk_rotation_peak_deg'] = float(np.degrees(peak))
    out['observable_components'] = 'pitch (and yaw) only: roll probe' if single else 'roll and pitch'
    return out


def estimator_task(task):
    setting, seed, fe = task
    kw = dict(SETTINGS_1B)[setting]
    duration = 4.5 if fe['kind'] in ('rock', 'yaw') else 3.0
    c = Condition(seed=seed, duration_s=duration, joint_noise_rad=.01, **kw)
    frontend = Frontend(Settings(**fe), KIN)
    s, r = rollout(SIM, THETA, np.zeros(22), c, Design(), record=True, terminate=False, mode='reflex', basis=BASIS, frontend=frontend)
    rep = s['frontend']
    truth = true_correction_deg(c, rep['estimate_time_s'] or fe['end_s'])
    row = dict(setting=setting, seed=seed, estimator=fe['kind'], settings=fe, condition=asdict(c), frontend=rep,
               true_correction_roll_pitch_deg=truth,
               error_roll_pitch_deg=[e - tr for e, tr in zip(rep['correction_roll_pitch_deg'], truth)],
               recovered_200hz=s['recovered_200hz'], survived=s['survived'], failure_time_s=s['failure_time_s'],
               failure_cause=s['failure_cause'], physical_valid=s['physical_valid'])
    win = (r['time_s'] >= fe['start_s']) & (r['time_s'] < fe['end_s'])
    row['window_max_tilt_deg'] = float(r['tilt_deg'][win].max())
    feet = r['foot_pos'][win][:, :, :2]
    row['window_foot_travel_mm'] = (np.linalg.norm(feet - feet[0], axis=2).max(axis=0) * 1000).tolist()
    row['window_foot_unloaded_s'] = ((~r['foot_found'][win]).sum(axis=0) * .005).tolist()
    if fe['kind'] in ('rock', 'yaw'):
        row['decomposition'] = decompose(r, c, rep)
        for k, v in row['decomposition'].items():
            if isinstance(v, dict):
                v['error_roll_pitch_deg'] = [e - tr for e, tr in zip(v['roll_pitch_deg'], truth)]
    return json.loads(json.dumps(row))


def prior_task(seed):
    c = Condition(seed=seed, duration_s=1.5, joint_noise_rad=.01)
    fe = Frontend(Settings(kind='prior'), KIN)
    s, r = rollout(SIM, THETA, np.zeros(22), c, Design(), record=True, terminate=False, mode='reflex', basis=BASIS, frontend=fe)
    return (fe.grav_sum / fe.count).tolist()


def estimators(root, workers):
    dest = root / 'diagnostic'
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / 'estimators.jsonl.gz'
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        readings = list(pool.map(prior_task, [7_150_000_000 + i for i in range(8)]))
        g = np.mean(readings, axis=0)
        prior = dict(seeds=[7_150_000_000 + i for i in range(8)], readings=readings, prior_gravity=(g / np.linalg.norm(g)).tolist(),
                     note='mean raw quiet gravity reading of the frozen reflex in 0.3-1.3 s, nominal calibration, level floor')
        (dest / 'prior.json').write_text(json.dumps(prior, indent=1) + '\n', newline='\n')
        fes = [dict(Settings(kind='prior', prior_gravity=tuple(prior['prior_gravity'])).to_dict()),
               Settings(kind='static').to_dict(),
               Settings(kind='rock', probe_rad=.05, probe_hz=1., end_s=2.3).to_dict(),
               Settings(kind='rock', probe_rad=.10, probe_hz=1., end_s=2.3).to_dict()]
        tasks = [(name, 7_100_000_000 + i, fe) for fe in fes for name, _ in SETTINGS_1B for i in range(4)]
        rows = list(pool.map(estimator_task, tasks, chunksize=2))
    write_gzip(out, rows)
    summarize_estimators(root)


def summarize_estimators(root):
    rows = read_rows(root / 'diagnostic/estimators.jsonl.gz')
    groups = {}
    for r in rows:
        key = r['estimator'] + ('' if r['estimator'] not in ('rock', 'yaw') else f"-{r['settings']['probe_rad']:g}")
        groups.setdefault(key, {}).setdefault(r['setting'], []).append(r)
    summary = {}
    for key, by in groups.items():
        # A roll-only probe observes pitch (and yaw) misalignment only; score that component.
        comp = (lambda e: np.asarray(e)[..., 1:2]) if key.startswith(('rock', 'yaw')) else (lambda e: np.asarray(e))
        cells = {}
        for setting, rs in by.items():
            e = np.array([r['error_roll_pitch_deg'] for r in rs])
            cell = dict(mean_error_roll_pitch_deg=e.mean(axis=0).round(3).tolist(), max_abs_error_deg=float(np.abs(comp(e)).max().round(3)),
                        falls=sum(not r['survived'] for r in rs), recovered=sum(r['recovered_200hz'] for r in rs), n=len(rs),
                        window_max_tilt_deg=float(max(r['window_max_tilt_deg'] for r in rs)))
            if rs[0].get('decomposition'):
                for part in ('kinematic_true_joints', 'oracle_trunk_noisy_gyro', 'oracle_trunk_clean_gyro'):
                    pe = np.array([r['decomposition'][part]['error_roll_pitch_deg'] for r in rs])
                    cell[part + '_max_abs_error_deg'] = float(np.abs(comp(pe)).max().round(3))
                cell['max_sole_rotation_deg'] = float(max(max(r['decomposition']['max_sole_rotation_deg']) for r in rs))
                cell['trunk_rotation_peak_deg'] = float(np.mean([r['decomposition']['trunk_rotation_peak_deg'] for r in rs]))
                cell['kinematic_rate_error_fraction'] = float(np.mean([r['decomposition']['kinematic_rate_error_rms_rad_s'] /
                                                                      r['decomposition']['trunk_rate_rms_rad_s'] for r in rs]))
            cells[setting] = cell
        level = [s for s in by if 'slope' not in s]
        errs = np.abs(comp(np.array([r['error_roll_pitch_deg'] for s in level for r in by[s]])))
        every = np.abs(comp(np.array([r['error_roll_pitch_deg'] for rs in by.values() for r in rs])))
        summary[key] = dict(cells=cells, scored_components='pitch only (roll probe)' if key.startswith(('rock', 'yaw')) else 'roll and pitch',
                            level_floor_median_abs_error_deg=float(np.median(errs)), level_floor_max_abs_error_deg=float(errs.max()),
                            all_median_abs_error_deg=float(np.median(every)), all_max_abs_error_deg=float(every.max()),
                            falls=sum(not r['survived'] for rs in by.values() for r in rs))
        if key.startswith('rock'):
            summary[key]['meets_declared_candidate_criterion'] = bool(
                summary[key]['all_median_abs_error_deg'] <= .5 and summary[key]['all_max_abs_error_deg'] <= 1. and summary[key]['falls'] == 0)
    (root / 'diagnostic/estimators-summary.json').write_text(json.dumps(summary, indent=1) + '\n', newline='\n')
    for k, v in summary.items():
        print(k, {x: v[x] for x in v if x != 'cells'})


# ----------------------------------------------------------------------------- feasibility (pre-protocol)
def feasibility_task(task):
    kind, kw, fe = task
    c = Condition(seed=7, duration_s=3.5 if fe['kind'] != 'static' else 3.0, joint_noise_rad=.01, **kw)
    frontend = Frontend(Settings(**fe), KIN)
    s, r = rollout(SIM, THETA, np.zeros(22), c, Design(), record=True, terminate=False, mode='reflex', basis=BASIS, frontend=frontend)
    win = (r['time_s'] > fe['start_s']) & (r['time_s'] < fe['end_s'])
    feet = r['foot_pos'][win][:, :, :2]
    return dict(kind=kind, condition=kw, frontend=s['frontend'], true_correction_roll_pitch_deg=true_correction_deg(c, fe['end_s']),
                max_trunk_angular_speed=np.abs(r['qvel'][win, 3:6]).max(axis=0).tolist(),
                foot_travel_mm=(np.linalg.norm(feet - feet[0], axis=2).max(axis=0) * 1000).tolist(),
                window_max_tilt_deg=float(r['tilt_deg'][win].max()), recovered_200hz=s['recovered_200hz'], survived=s['survived'])


def feasibility(root, workers):
    dest = root / 'feasibility'
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / 'feasibility.json'
    if out.exists():
        raise SystemExit(f'refusing to overwrite {out}')
    tasks = []
    for kw in (dict(), pitch(2), pitch(-2), dict(imu_rpy_deg=(2., 0., 0.)), dict(slope_deg=(0., 2.)), dict(imu_rpy_deg=(0., 2., 0.), slope_deg=(0., 2.))):
        tasks.append(('static', kw, Settings(kind='static').to_dict()))
    for amp in (.05, .1, .2):
        for sign in (1., -1.):
            tasks.append(('yaw', pitch(2), Settings(kind='yaw', probe_rad=amp, probe_sign=sign).to_dict()))
    for amp in (.05, .1, .15):
        for kw in (pitch(2), dict(), dict(slope_deg=(0., 2.))):
            tasks.append(('rock', kw, Settings(kind='rock', probe_rad=amp, probe_hz=1., end_s=2.3).to_dict()))
    with ProcessPoolExecutor(workers, initializer=init_worker) as pool:
        rows = list(pool.map(feasibility_task, tasks))
    out.write_text(json.dumps(json.loads(json.dumps(rows)), indent=1) + '\n', newline='\n')
    for r in rows:
        print(r['kind'], r['condition'], r['frontend']['settings'].get('probe_rad'), 'est', np.round(r['frontend']['correction_roll_pitch_deg'], 2),
              'true', np.round(r['true_correction_roll_pitch_deg'], 2), 'survived', r['survived'])


# ----------------------------------------------------------------------------- 1C gate
def paired(a, b, cells=None):
    keys = lambda r: (r['setting'], r['set'], r['condition']['seed'])
    A = {keys(r): r for r in a if cells is None or r['setting'] in cells}
    B = {keys(r): r for r in b if cells is None or r['setting'] in cells}
    assert A.keys() == B.keys()
    gains = sum(B[k]['recovered_200hz'] and not A[k]['recovered_200hz'] for k in A)
    losses = sum(A[k]['recovered_200hz'] and not B[k]['recovered_200hz'] for k in A)
    return gains, losses


def gate(root):
    dest = root / 'pilot/frontend'
    rows = {p.name[:-9]: read_rows(p) for p in dest.glob('*.jsonl.gz')}
    learned = [f'{a}-s{s}' for a in ('history', 'memoryless') for s in (1, 2, 3)]
    result = {}
    for fe in ('static', 'prior', 'gate'):
        g = l = 0
        nominal_quiet_losses = nominal_back_losses = 0
        per = {}
        for name in learned:
            base, new = rows[f'{name}__none'], rows[f'{name}__{fe}']
            gi, li = paired(base, new, MISALIGNED_D)
            g += gi; l += li
            nominal = lambda kind: paired([r for r in base if r['set'] == kind], [r for r in new if r['set'] == kind], ('nominal',))[1]
            nominal_quiet_losses += nominal('quiet'); nominal_back_losses += nominal('back-.08')
            per[name] = dict(misaligned_gains=gi, misaligned_losses=li, slope=paired(base, new, ('slope-2', 'slope+2')),
                             nominal=paired(base, new, ('nominal',)))
        reflex_nominal_losses = paired(rows['reflex__none'], rows[f'reflex__{fe}'], ('nominal',))[1]
        passed = g - l >= 30 and nominal_quiet_losses == 0 and nominal_back_losses == 0 and reflex_nominal_losses == 0
        result[fe] = dict(passed=passed, misaligned_net=g - l, misaligned_gains=g, misaligned_losses=l,
                          nominal_quiet_losses=nominal_quiet_losses, nominal_backward_losses=nominal_back_losses,
                          reflex_nominal_losses=reflex_nominal_losses, per_controller=per)
    result['decision'] = {fe: result[fe]['passed'] for fe in ('static', 'prior')}
    (root / 'pilot/gate-frontend.json').write_text(json.dumps(result, indent=1) + '\n', newline='\n')
    print(json.dumps({k: {x: y for x, y in v.items() if x != 'per_controller'} if isinstance(v, dict) and 'per_controller' in v else v
                      for k, v in result.items()}, indent=1))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('step', choices=('feasibility', 'characterize', 'estimators', 'summarize-estimators', 'pilot', 'gate'))
    p.add_argument('--root', type=Path, default=Path('results/02e'))
    p.add_argument('--workers', type=int, default=6)
    a = p.parse_args(argv)
    if not 1 <= a.workers <= 6:
        p.error('at most six low-priority workers')
    if a.step == 'feasibility':
        feasibility(a.root, a.workers)
    elif a.step == 'characterize':
        conds = battery(SETTINGS_1A, KINDS, 7_000_000_000)
        configs = [(c[0], c, None) for c in controllers()]
        run_battery(a.root / 'diagnostic/characterize', configs, conds, a.workers)
    elif a.step == 'estimators':
        estimators(a.root, a.workers)
    elif a.step == 'summarize-estimators':
        summarize_estimators(a.root)
    elif a.step == 'pilot':
        prior = json.loads((a.root / 'diagnostic/prior.json').read_text())['prior_gravity']
        conds = battery(SETTINGS_D, KINDS, 7_200_000_000)
        cs = controllers(['reflex', *[f'{x}-s{s}' for x in ('history', 'memoryless') for s in (1, 2, 3)]])
        fes = dict(none=None, gate=Settings(kind='gate').to_dict(), prior=Settings(kind='prior', prior_gravity=tuple(prior)).to_dict(),
                   static=Settings(kind='static').to_dict())
        configs = [(f'{c[0]}__{f}', c, fes[f]) for f in FRONTENDS_1C for c in cs]
        run_battery(a.root / 'pilot/frontend', configs, conds, a.workers)
    else:
        gate(a.root)


if __name__ == '__main__':
    main()
