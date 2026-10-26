"""Offline HTML inspection of one trial or comparison of two trials."""
import argparse
import html
import io
import json
from pathlib import Path

import numpy as np
from movement_lab.measure import summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('trials', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    if not 1 <= len(args.trials) <= 2:
        p.error('supply one or two trial directories')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    loaded = []
    for path in args.trials:
        with np.load(path/'trial.npz', allow_pickle=False) as f:
            a = dict(f)
        m = json.loads((path/'metadata.json').read_text())
        loaded.append((a, m, summary(a)))
    labels = [f"{Path(m['checkpoint']).name} ({p.name})" for p, (_, m, _) in zip(args.trials, loaded)]
    fig, axes = plt.subplots(5, 1, figsize=(11, 13), layout='constrained')
    for (a, m, s), label in zip(loaded, labels):
        t = a['time_s']
        q = a['body_pose_w'][:, 0, 3:7]
        tilt = np.rad2deg(np.arccos(np.clip(1-2*(q[:,1]**2+q[:,2]**2), -1, 1)))
        axes[0].plot(t, a['body_pose_w'][:,0,2], label=label)
        axes[1].plot(t, tilt, label=label)
        axes[2].plot(t, np.sqrt(np.mean(a['joint_force_nm']**2, axis=1)), label=label)
        for foot, side in enumerate(('left','right')):
            axes[3].plot(t, a['foot_force_w_n'][:,foot,2], label=f'{label} {side}', alpha=.8)
            axes[4].step(t, (a['foot_found'][:,foot] > 0).astype(float) + 1.2*foot,
                         label=f'{label} {side}', where='post', alpha=.7)
    for ax, title in zip(axes, ('Root height (m)', 'Root tilt (deg)', 'RMS across joint forces (Nm)',
                               'Foot world z force (N; sensor sign)', 'Contact: left 0/1, right 1.2/2.2')):
        ax.set_ylabel(title)
        ax.grid(alpha=.25)
        ax.legend(fontsize=7)
    axes[-1].set_xlabel('Simulation time (s)')
    buf = io.StringIO()
    fig.savefig(buf, format='svg')
    plt.close(fig)
    parts = ['<!doctype html><meta charset="utf-8"><title>Movement Lab 00A</title>',
             '<style>body{font:16px system-ui;max-width:1150px;margin:32px auto}td,th{padding:8px;border-bottom:1px solid #ccc}svg{max-width:100%;height:auto}summary{cursor:pointer}</style>',
             '<h1>Movement Lab 00A — Balance and Contact</h1>',
             '<p>One nominal, seeded rollout per policy. These measurements do not establish robustness or hardware performance. Trials stop on termination; compare common time windows when durations differ.</p>']
    comparable = ('task','seed','vx','sample_mode','physics_dt','control_dt','config_sha256','upstream_commit','versions')
    if len(loaded) == 2:
        mismatches = [k for k in comparable if loaded[0][1][k] != loaded[1][1][k]]
        if mismatches:
            parts.append('<p><b>Conditions differ:</b> '+html.escape(', '.join(mismatches))+'</p>')
    parts.append('<table><tr><th>Measurement</th>'+''.join('<th>'+html.escape(l)+'</th>' for l in labels)+'</tr>')
    for k in loaded[0][2]:
        parts.append('<tr><td>'+html.escape(k)+'</td>'+''.join('<td>'+html.escape(str(s[k]))+'</td>' for _,_,s in loaded)+'</tr>')
    parts.extend(['</table>', buf.getvalue()])
    for (a,m,s), label in zip(loaded,labels):
        # All joint command/actual/effort traces, without a viewer dependency.
        fig, axes = plt.subplots(len(m['joint_names']), 2, figsize=(13, 2*len(m['joint_names'])), squeeze=False, layout='constrained')
        for j,name in enumerate(m['joint_names']):
            axes[j,0].plot(a['time_s'], a['joint_target_rad'][:,j], label='target')
            axes[j,0].plot(a['time_s'], a['joint_position_rad'][:,j], label='actual')
            axes[j,0].set_ylabel(name+' (rad)', fontsize=7)
            axes[j,0].legend(fontsize=6)
            axes[j,1].plot(a['time_s'], a['joint_force_nm'][:,j])
            axes[j,1].set_ylabel('actuator joint force (Nm)')
        buf = io.StringIO()
        fig.savefig(buf, format='svg')
        plt.close(fig)
        parts.append('<details><summary>'+html.escape(label)+' — all joints</summary>'+buf.getvalue()+'</details>')
        parts.append('<details><summary>Provenance</summary><pre>'+html.escape(json.dumps(m,indent=2))+'</pre></details>')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text('\n'.join(parts))
    print(args.out.resolve())


if __name__ == '__main__':
    main()
