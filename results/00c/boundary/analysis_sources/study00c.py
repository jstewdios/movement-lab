"""Bounded matched disturbance exploration. No monotone success model is fitted."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from movement_lab.disturbance import assess, signals
from movement_lab.validate00b import check

SEEDS = (42, 43)
POLICIES = ('99', '348')


def name(policy, seed, dv):
    return f'{policy}-s{seed}-dv{dv:+.2f}'


def metrics(a, m):
    s = assess(a, m)
    at = m['disturbance']['actual_time_s']
    t = a['time_s']; pre = (t >= at-.5) & (t < at-1e-5)
    post = t > at+1e-6; tail = t >= t[-1]-.5
    xyz = a['body_pose_w'][:, 0, :3]
    q = a['joint_position_rad']; force = a['joint_force_nm']
    orientation = a['body_pose_w'][:, 0, 3:7]
    # Align quaternion signs before averaging; q and -q are the same rotation.
    def mean_orientation(mask):
        values = orientation[mask]
        aligned = values*np.where(values@values[0] < 0, -1., 1.)[:, None]
        mean = aligned.mean(0)
        return mean/np.linalg.norm(mean)
    orientation_offset = float(np.rad2deg(2*np.arccos(np.clip(abs(mean_orientation(pre)@mean_orientation(tail)), 0, 1))))
    s.update(
        # A failure envelope is not a verified physical ground impact.
        failure=bool(not s['upright']),
        horizon_complete=bool(t[-1] >= m['requested_seconds']-m['control_dt'] and not m['stopped_on_done']),
        returned_to_origin=bool(s['recovered'] and s['final_root_shift_m'] < .005
                                and np.max(np.abs(q[tail].mean(0)-q[pre].mean(0))) < .1
                                and orientation_offset < 3
                                and abs(xyz[tail,2].mean()-xyz[pre,2].mean()) < .02),
        final_orientation_offset_deg=orientation_offset,
        final_joint_offset_max_rad=float(np.max(np.abs(q[tail].mean(0)-q[pre].mean(0)))),
        peak_joint_speed_rad_s=float(np.abs(a['joint_velocity_rad_s'][post]).max()),
        peak_joint_effort_nm=float(np.abs(force[post]).max()),
        joint_peak_effort_nm=np.abs(force[post]).max(0).tolist(),
        joint_excursion_rad=np.ptp(q[post], axis=0).tolist(),
        peak_angular_speed_rad_s=float(np.linalg.norm(a['body_velocity_w'][post, 0, 3:], axis=1).max()),
        min_root_height_m=float(xyz[post, 2].min()),
        final_root_delta_xyz_m=(xyz[tail].mean(0)-xyz[pre].mean(0)).tolist(),
        peak_com_shift_m=float(np.linalg.norm(a['com_position_w_m'][post, :2]-a['com_position_w_m'][pre, :2].mean(0), axis=1).max()),
        final_com_delta_xyz_m=(a['com_position_w_m'][tail].mean(0)-a['com_position_w_m'][pre].mean(0)).tolist(),
        unloading_fraction=np.mean(a['foot_found'][post] == 0, axis=0).tolist(),
        baseline_tilt_deg=float(signals(a)[pre].mean()),
        peak_tilt_change_deg=float(np.max(np.abs(signals(a)[post]-signals(a)[pre].mean()))),
        final_tilt_deg=float(signals(a)[tail].mean()),
    )
    s['state'] = ('failure envelope / termination' if s['failure'] else
                  'returned standing' if s['returned_to_origin'] else
                  'recovered displaced/posture changed' if s['recovered'] else 'upright, recovery unresolved')
    def first(mask):
        indices = np.flatnonzero(post & mask)
        return float(t[indices[0]]-at) if len(indices) else None
    s['event_sequence_s'] = dict(
        tilt_over_12=first(signals(a) >= 12),
        tilt_over_45=first(signals(a) >= 45),
        height_loss_4cm=first(xyz[:, 2] <= xyz[pre, 2].mean()-.04),
        left_contact_loss=first(a['foot_found'][:, 0] == 0),
        right_contact_loss=first(a['foot_found'][:, 1] == 0),
        recorded_end=float(t[-1]-at),
    )
    return s


def load(root):
    records = {}
    for p in sorted(root.glob('*/metadata.json')):
        m = json.loads(p.read_text())
        with np.load(p.parent/'trial.npz', allow_pickle=False) as f:
            a = dict(f)
        records[p.parent.name] = (a, m, metrics(a, m))
    return records


def choose_probe(outcomes, audit=False):
    """Suggest only: each tested point remains evidence, including reversals."""
    levels = sorted(outcomes)
    if audit:
        # Interior point below the smallest tested nonzero amplitude; checks an
        # unsampled lower region even if larger disturbances survived.
        return round(levels[0]/2, 2)
    failures = [x for x in levels if outcomes[x]]
    if failures:
        upper = min(failures)
        lower = max([0.] + [x for x in levels if x < upper and not outcomes[x]])
        return round((lower+upper)/2, 2)
    return round(levels[-1]*2, 2)


def run_trial(root, policy, seed, dv):
    dest = root/name(policy, seed, dv)
    if (dest/'metadata.json').exists():
        return
    assert len(list(root.glob('*/metadata.json'))) < 44, '44-trial budget exhausted'
    ckpt = json.loads(Path(f'results/{policy}-balance/metadata.json').read_text())['checkpoint']
    start = time.monotonic()
    cmd = [sys.executable, '-m', 'movement_lab.evaluate', ckpt, '--out', str(dest),
           '--seed', str(seed), '--seconds', '5', '--disturbance-dv', str(dv), '--record-com']
    with (root/'evaluation.log').open('a') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    with (root/'cost.jsonl').open('a') as log:
        log.write(json.dumps(dict(trial=dest.name, wall_seconds=time.monotonic()-start))+'\n')
    # Validate immediately so a bad causal match cannot drive later selection.
    check(root)


def execute(root, phase):
    root.mkdir(parents=True, exist_ok=True)
    if phase == 'pilot':
        for dv in (0., .06):
            for policy in POLICIES:
                run_trial(root, policy, 42, dv)
        return
    assert all((root/name(p,42,dv)/'metadata.json').exists()
               for p in POLICIES for dv in (0., .06)), 'run pilot and review its cost before expansion'
    check(root)
    plan_path = root/'adaptive_plan.json'
    plan = json.loads(plan_path.read_text()) if plan_path.exists() else {}
    # Four cells for every amplitude, with same seeds and command settings.
    for dv in (0., .06):
        for policy in POLICIES:
            run_trial(root, policy, 43, dv)
    for sign in (-1, 1):
        for seed in SEEDS:
            for policy in POLICIES:
                run_trial(root, policy, seed, sign*.06)
        # Two exploratory rounds; a mixed outcome triggers refinement but is
        # never interpreted as a proof that all larger amplitudes fail.
        for round_index in range(2):
            records = load(root)
            outcomes = {}
            for a, m, s in records.values():
                dv = m['disturbance']['requested_dv_m_s']
                if dv*sign > 0:
                    outcomes[abs(dv)] = outcomes.get(abs(dv), False) or s['failure']
            dv = sign*min(.48, choose_probe(outcomes))
            key = f'{sign}:explore:{round_index}'
            dv = plan.setdefault(key, dv)
            plan_path.write_text(json.dumps(plan, indent=2)+'\n')
            for seed in SEEDS:
                for policy in POLICIES:
                    run_trial(root, policy, seed, dv)
        # One further transition probe and an independent small-magnitude audit.
        for audit in (False, True):
            records = load(root); outcomes = {}
            for a, m, s in records.values():
                dv = m['disturbance']['requested_dv_m_s']
                if dv*sign > 0:
                    outcomes[abs(dv)] = outcomes.get(abs(dv), False) or s['failure']
            dv = sign*min(.48, choose_probe(outcomes, audit))
            key = f'{sign}:refine:{audit}'
            dv = plan.setdefault(key, dv)
            plan_path.write_text(json.dumps(plan, indent=2)+'\n')
            for seed in SEEDS:
                for policy in POLICIES:
                    run_trial(root, policy, seed, dv)


def report(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    records = load(root)
    check(root)
    if len(records) > 4:
        for dv in {m['disturbance']['requested_dv_m_s'] for _,m,_ in records.values()}:
            assert all(name(p,seed,dv) in records for p in POLICIES for seed in SEEDS), ('incomplete matched condition', dv)
    # Additional COM and matching checks beyond unchanged 00B validator.
    for n, (a, m, s) in records.items():
        assert s['settled'], n
        mass = np.array(m['body_mass_kg'])
        assert np.all(mass >= 0) and abs(mass.sum()-.73724318) < 1e-6
        assert a['com_position_w_m'].shape == (len(a['time_s']), 3)
        assert np.max(np.linalg.norm(a['com_position_w_m']-a['body_pose_w'][:, 0, :3], axis=1)) < .2
        np.testing.assert_allclose(a['qpos'][:,:7], a['body_pose_w'][:,0,:], atol=2e-6)
        np.testing.assert_array_equal(a['qvel'][:,6:], a['joint_velocity_rad_s'])
        np.testing.assert_array_equal(a['actuator_force'], a['joint_force_nm'])
        np.testing.assert_allclose(np.diff(a['time_s']), m['physics_dt'], atol=2e-7)
        np.testing.assert_array_equal(a['foot_force_w_n'][a['foot_found']==0], 0)
        cadence = round(m['control_dt']/m['physics_dt'])
        assert len(a['time_s']) % cadence == 0
        for i in range(0, len(a['time_s']), cadence):
            np.testing.assert_array_equal(a['action'][i:i+cadence], np.broadcast_to(a['action'][i], a['action'][i:i+cadence].shape))
        assert not m['upstream_status'], 'upstream is dirty'
    (root/'summary.json').write_text(json.dumps({n:s for n, (_, _, s) in records.items()}, indent=2)+'\n')
    costs = [json.loads(x) for x in (root/'cost.jsonl').read_text().splitlines()]
    lines = ['# Movement Lab 00C — The Boundary of Balance', '',
             f'{len(records)} matched nominal GPU trials; {sum(x["wall_seconds"] for x in costs):.1f} s cumulative subprocess wall time (includes startup, compilation and recording; not GPU kernel time).', '',
             'Zero velocity commands; 5 s horizon; signed world-x root velocity increments after 2 s. Each signed amplitude has checkpoints 99/348 × seeds 42/43 after expansion. Actual event times, full pre/post velocities, initial states, policy/config/source hashes and simulator versions are retained in per-trial records. Raw trajectories stay local.', '',
             '## Tested outcomes', '',
             '| Δvx m/s | 99 survival / recovery / return | 348 survival / recovery / return |',
             '|---:|---:|---:|']
    for dv in sorted({m['disturbance']['requested_dv_m_s'] for _,m,_ in records.values()}):
        cells = []
        for p in POLICIES:
            group = [s for n, (_,m,s) in records.items() if n.startswith(p+'-') and m['disturbance']['requested_dv_m_s'] == dv]
            cells.append(' / '.join(str(sum(s[k] for s in group)) for k in ('upright','recovered','returned_to_origin'))+f' (n={len(group)})')
        lines.append(f'| {dv:+.2f} | '+ ' | '.join(cells)+' |')
    lines += ['', '## Region and uncertainty', '',
              'Every row is a sampled condition, not a continuous safe interval. The shared adaptive rule doubles amplitude when all four cells survive, otherwise probes midway below the smallest observed failure; a separate interior audit samples below the smallest tested magnitude. Failure information from either checkpoint determines shared probes. No monotone model or binary search guarantee is used. Maximum amplitude is 0.48 m/s and maximum scheduled trial count is 44; no expansion beyond this budget is automatic.', '']
    for p in POLICIES:
        for sign, direction in ((-1, 'backward'), (1, 'forward')):
            group = [(abs(m['disturbance']['requested_dv_m_s']),s) for n,(_,m,s) in records.items() if n.startswith(p+'-') and m['disturbance']['requested_dv_m_s']*sign > 0]
            survival = [x for x,s in group if s['upright']]; failures = [x for x,s in group if s['failure']]
            reversals = [(x,y) for x,s in group if s['failure'] for y,v in group if v['upright'] and y>x]
            lines.append(f'- {p} {direction}: largest sampled survival {max(survival) if survival else None} m/s; smallest sampled failure {min(failures) if failures else None} m/s; larger-survival/smaller-failure pairs: {len(set(reversals))}. These extrema can overlap and are not capability thresholds.')
    lines += ['', '## Measurement decisions and limits', '',
              'The unchanged 00B settled/recovery rules are reused. Survival requires tilt <45°, height loss <4 cm and no done signal throughout the observation. Its complement is called a failure envelope, not verified physical impact. Recovery requires a continuous 0.5 s stable window and stable final 0.5 s; it permits displacement. Return additionally requires final mean horizontal root displacement <5 mm and every final mean joint offset <0.1 rad, mean trunk orientation within 3° (including yaw), and mean root height within 2 cm of the pre-event 0.5 s. These new exploratory criteria do not revise historical 00B findings. A surviving unresolved trial may recover after the horizon.', '',
              'Whole-robot COM is the mass-weighted sum of body inertial centers, with mass-weighted body COM linear velocity; root position is reported separately. Ankle positions are foot proxies. Contact loss plus >5 mm lift and displaced recontact supplies conservative step evidence, not a validated gait label. Foot translation can be rocking or sliding. Effort is qfrc_actuator; joint excursions and peak joint forces are in summary.json. No electrical consumption, actuator saturation, support margin or hardware capability is inferred. This is a generalized-velocity state intervention, not a measured force pulse at a trunk point. Work and whole-post-event summaries have different horizons after termination; use common-time traces for policy comparisons.', '',
              'Seeds change inherited observation delays but share the same nominal physical initial state. Two seeds cannot support population confidence intervals. Timing, terrain, orientation, physical initial-state variation and longer-horizon recovery remain unresolved. Feet-only collision geometry and endpoint forward solves retain 00A/00B limitations; near-boundary solver sensitivity is not bounded.', '',
              'Validation: unchanged 00B validator checks commands identically zero, signed velocity injection with other velocities unchanged, timing within one physics step, exact pre-event equality to controls, first-step velocity/displacement direction, policy-pair configurations/versions/initial-state equality, and source/config hashes. Additional checks verify COM shape, plausible root distance and total mass. See validation.json; unit tests exercise recovery distinctions and nonmonotone scheduling.', '']
    lines += ['## Individual trajectories', '',
              '| Trial | Outcome | Peak tilt ° | Root shift mm | Peak angular speed rad/s | Foot unload fraction L/R | Step evidence L/R |',
              '|---|---|---:|---:|---:|---|---|']
    for n, (_, m, s) in records.items():
        lines.append(f'| {n} | {s["state"]} | {s["peak_tilt_deg"]:.2f} | {s["final_root_shift_m"]*1000:.2f} | {s["peak_angular_speed_rad_s"]:.2f} | {s["unloading_fraction"]} | {s["step_evidence"]} |')
    notes = root/'INTERPRETATION.md'
    if notes.exists():
        lines[6:6] = [notes.read_text(), '']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')
    parts = ['<!doctype html><meta charset="utf-8"><title>Movement Lab 00C</title>', '<style>body{font:15px system-ui;margin:30px}img{max-width:100%}pre{white-space:pre-wrap}</style>']
    import html
    parts.append('<pre>'+html.escape((root/'REPORT.md').read_text())+'</pre>')
    # All matched disturbed trajectories plus controls; time aligned at the event.
    divergence = {}
    for seed in SEEDS:
        for dv in sorted({m['disturbance']['requested_dv_m_s'] for _,m,_ in records.values()}):
            if not all(name(p,seed,dv) in records for p in POLICIES): continue
            fig, axes = plt.subplots(10, 1, figsize=(11, 20), layout='constrained')
            for p in POLICIES:
                for v in dict.fromkeys((dv, 0.)):
                    a,m,s = records[name(p,seed,v)]
                    t=a['time_s']-m['disturbance']['actual_time_s']; pre=(t>=-.5)&(t<-1e-5)
                    style = '-' if v==dv else ':'
                    label=f'{p} Δvx={v:+.2f}'
                    color = 'C0' if p=='99' else 'C1'
                    series=[signals(a),a['body_velocity_w'][:,0,4],a['body_pose_w'][:,0,0]-a['body_pose_w'][pre,0,0].mean(),a['com_position_w_m'][:,0]-a['com_position_w_m'][pre,0].mean(),a['body_pose_w'][:,0,2],np.sqrt(np.mean(a['joint_force_nm']**2,axis=1)),np.max(np.abs(a['joint_velocity_rad_s']),axis=1)]
                    for ax,y in zip(axes,series): ax.plot(t,y,style,color=color,label=label)
                    for f,b in enumerate(('ankle_left','ankle_right')):
                        idx=m['body_names'].index(b)
                        axes[7].plot(t,a['body_pose_w'][:,idx,0]-a['body_pose_w'][pre,idx,0].mean(),style,color=color,alpha=1-f*.45,label=label+f' foot {f}')
                        axes[8].step(t,(a['foot_found'][:,f]>0).astype(float)+f*1.2,style,color=color,alpha=1-f*.45)
                        axes[9].plot(t,a['foot_force_w_n'][:,f,2],style,color=color,alpha=1-f*.45)
            for ax,lab in zip(axes,['Trunk tilt °','Trunk ωy rad/s','Root Δx m','COM Δx m','Root height m','Joint RMS Nm','Peak |joint speed| rad/s','Ankle Δx m','Contact L/R+1.2','Raw foot Fz N']):
                ax.set_ylabel(lab); ax.axvline(0,color='k',alpha=.3); ax.grid(alpha=.2); ax.set_xlim(-.5,3)
            axes[0].legend(fontsize=8); axes[7].legend(fontsize=6); axes[-1].set_xlabel('Time since disturbance (s)')
            buf=io.StringIO();fig.savefig(buf,format='svg')
            if seed==42 and dv!=0: fig.savefig(root/f'trajectory-dv{dv:+.2f}.png',dpi=90)
            plt.close(fig); parts.extend([f'<h2>Seed {seed}, Δvx {dv:+.2f}</h2>',buf.getvalue()])
            a,am,_=records[name('99',seed,dv)]; b,bm,_=records[name('348',seed,dv)]
            n=min(len(a['time_s']),len(b['time_s'])); post=a['time_s'][:n] > am['disturbance']['actual_time_s']+1e-6
            pre=(a['time_s'][:n] >= am['disturbance']['actual_time_s']-.5) & ~post
            # Matched absolute times until the first termination, no padding.
            divergence[name('pair',seed,dv)] = dict(common_end_s=float(a['time_s'][n-1]),
                post_rms_root_separation_m=float(np.sqrt(np.mean(np.sum((a['body_pose_w'][:n,0,:3][post]-b['body_pose_w'][:n,0,:3][post])**2,axis=1)))),
                pre_rms_joint_separation_rad=float(np.sqrt(np.mean((a['joint_position_rad'][:n][pre]-b['joint_position_rad'][:n][pre])**2))),
                post_rms_joint_separation_rad=float(np.sqrt(np.mean((a['joint_position_rad'][:n][post]-b['joint_position_rad'][:n][post])**2))))
            ca=records[name('99',seed,0.)][0]; cb=records[name('348',seed,0.)][0]
            root_response=(a['body_pose_w'][:n,0,:3]-ca['body_pose_w'][:n,0,:3])-(b['body_pose_w'][:n,0,:3]-cb['body_pose_w'][:n,0,:3])
            tilt_response=(signals(a)[:n]-signals(ca)[:n])-(signals(b)[:n]-signals(cb)[:n])
            divergence[name('pair',seed,dv)].update(
                post_rms_root_response_difference_m=float(np.sqrt(np.mean(np.sum(root_response[post]**2,axis=1)))),
                post_rms_tilt_response_difference_deg=float(np.sqrt(np.mean(tilt_response[post]**2))))
    representatives = set()
    for sign in (-1, 1):
        failures = [abs(m['disturbance']['requested_dv_m_s']) for _,m,s in records.values()
                    if m['disturbance']['requested_dv_m_s']*sign > 0 and s['failure']]
        levels = sorted({abs(m['disturbance']['requested_dv_m_s']) for _,m,_ in records.values()
                         if m['disturbance']['requested_dv_m_s']*sign > 0})
        if not levels: continue
        upper = min(failures) if failures else max(levels)
        representatives.add(sign*upper)
        below = [x for x in levels if x < upper]
        if below: representatives.add(sign*max(below))
    for dv in sorted(representatives):
        fig, axes = plt.subplots(14, 2, figsize=(13, 27), layout='constrained')
        for seed in SEEDS:
            for p in POLICIES:
                if name(p,seed,dv) not in records: continue
                a,m,s = records[name(p,seed,dv)]
                t=a['time_s']-m['disturbance']['actual_time_s']
                color='C0' if p=='99' else 'C1'
                for j,joint in enumerate(m['joint_names']):
                    axes[j,0].plot(t,a['joint_position_rad'][:,j],color=color,ls='-' if seed==42 else '--',label=f'{p} s{seed} actual')
                    axes[j,0].plot(t,a['joint_target_rad'][:,j],':',color=color,label=f'{p} s{seed} target')
                    axes[j,0].set_ylabel(joint+' rad',fontsize=7)
                    axes[j,1].plot(t,a['joint_force_nm'][:,j],color=color,ls='-' if seed==42 else '--',label=f'{p} s{seed}')
                    axes[j,1].set_ylabel('Nm',fontsize=7)
        for ax in axes.flat:
            ax.axvline(0,color='k',alpha=.3); ax.grid(alpha=.2); ax.set_xlim(-.5,3)
        axes[0,0].legend(fontsize=6); axes[0,1].legend(fontsize=6)
        axes[-1,0].set_xlabel('Time since disturbance (s)'); axes[-1,1].set_xlabel('Time since disturbance (s)')
        buf=io.StringIO();fig.savefig(buf,format='svg');plt.close(fig)
        parts.extend([f'<details><summary>Representative joint behavior, seeds 42/43, Δvx {dv:+.2f}</summary>',buf.getvalue(),'</details>'])
    (root/'divergence.json').write_text(json.dumps(divergence,indent=2)+'\n')
    # Matplotlib SVG paths contain trailing spaces; normalize for publication.
    (root/'comparison.html').write_text('\n'.join(line.rstrip() for line in '\n'.join(parts).splitlines())+'\n')
    sources = root/'analysis_sources'; sources.mkdir(exist_ok=True)
    for file in ('study00c.py','validate00b.py','disturbance.py','measure.py'):
        (sources/file).write_bytes(Path(__file__).with_name(file).read_bytes())
    paths = [p for p in root.rglob('*') if p.is_file() and ('cache' not in p.parts and 'matplotlib' not in p.parts) and p.name!='manifest.json']
    (root/'manifest.json').write_text(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)},indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=Path('results/00c/boundary'))
    p.add_argument('--phase',choices=('pilot','expand','report'),default='report')
    args=p.parse_args()
    if args.phase!='report': execute(args.out,args.phase)
    report(args.out)

if __name__=='__main__': main()
