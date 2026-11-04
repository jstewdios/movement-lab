"""Recompute 02C tables/figures from published numerical evidence."""
from collections import defaultdict
import argparse
import json
from pathlib import Path
import numpy as np

from movement_lab.eval02c import read_rows
from movement_lab.learn02c import completed_checkpoints


def summarize(rows):
    return dict(n=len(rows), recovered=sum(r['recovered_200hz'] for r in rows),
                pushed_n=sum(r['condition']['push_time_s'] is not None for r in rows),
                pushed_recovered=sum(r['recovered_200hz'] and r['condition']['push_time_s'] is not None for r in rows),
                trained_band_n=sum(.28 <= r['condition']['push_dv'][0] <= .36 for r in rows),
                trained_band_recovered=sum(r['recovered_200hz'] and .28 <= r['condition']['push_dv'][0] <= .36 for r in rows),
                step_recovery=sum(r['recovery_with_step'] for r in rows),
                survived=sum(r['survived'] for r in rows),
                physical_valid=sum(r['physical_valid'] for r in rows),
                beyond_capture=sum(r['beyond_capture'] for r in rows),
                recovered_beyond_capture=sum(r['recovered_200hz'] and r['beyond_capture'] for r in rows))


def tables(root):
    out = dict(batteries={}, learning={})
    for battery in sorted((root / 'eval').iterdir()):
        if not battery.is_dir():
            continue
        rows = [r for path in sorted(battery.glob('*.gz')) for r in read_rows(path)]
        by = defaultdict(list)
        groups = defaultdict(list)
        for r in rows:
            by[r['controller']].append(r)
            groups[(r['controller'], r['set'])].append(r)
        out['batteries'][battery.name] = dict(
            controllers={k: summarize(v) for k, v in sorted(by.items())},
            by_set={k: {label: summarize(v) for (ctrl, label), v in sorted(groups.items()) if ctrl == k} for k in sorted(by)})
        if battery.name == 'grid':
            base = {r['condition']['seed']: r for r in by['reflex02b']}
            out['batteries']['grid']['prestate_equal_to_reflex'] = {
                name: dict(n=sum(r['condition']['push_time_s'] is not None for r in rs),
                           identical=sum(r['condition']['push_time_s'] is not None and r['pre_qpos'] == base[r['condition']['seed']]['pre_qpos'] for r in rs))
                for name, rs in sorted(by.items())}
            comparisons = {}
            for name, rs in by.items():
                if not name.startswith('cem') or name.count('-') != 1:
                    continue
                cmp = {}
                for control in ('reflex02b', 'wide-reflex', name.replace('cem', 'random'), name + '-reverse', name + '-constant', name + '-legs', name + '-head', name + '-bound35', name + '-delay80', name + '-imu'):
                    if control not in by:
                        continue
                    rhs = {r['condition']['seed']: r for r in by[control]}
                    a_only = sum(r['recovered_200hz'] and not rhs[r['condition']['seed']]['recovered_200hz'] for r in rs)
                    b_only = sum(not r['recovered_200hz'] and rhs[r['condition']['seed']]['recovered_200hz'] for r in rs)
                    cmp[control] = dict(learned_only=a_only, control_only=b_only)
                comparisons[name] = cmp
            out['batteries']['grid']['paired'] = comparisons
            recovered = [r for r in rows if r['controller'].startswith('cem') and r['controller'].count('-') == 1 and r['recovered_200hz'] and r['condition']['push_time_s'] is not None]
            out['movement'] = dict(n_recovered=len(recovered),
                                  max_clearance_m=max((max(r['peak_clearance_m']) for r in recovered), default=0),
                                  max_foot_shift_m=max((max(r['foot_shift_m']) for r in recovered), default=0),
                                  max_loaded_path_m=max((max(r['loaded_path_m']) for r in recovered), default=0),
                                  step_recovery=sum(r['recovery_with_step'] for r in recovered),
                                  recovered_where_reflex_fails=sum(not base[r['condition']['seed']]['recovered_200hz'] for r in recovered))
    for run in sorted((root / 'runs').iterdir()):
        history = [json.loads(str(np.load(p)['row'])) for p in completed_checkpoints(run)]
        out['learning'][run.name] = history
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('results/02c'))
    a = p.parse_args()
    out = a.root / 'report'; out.mkdir(exist_ok=True)
    data = tables(a.root)
    (out / 'tables.json').write_text(json.dumps(data, indent=2) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for name, history in data['learning'].items():
        color = '#1f77b4' if name.startswith('cem') else '#bc6b24'
        style = ('-', '--', ':')[int(name[-1]) - 1]
        axes[0].plot([r['trials'] for r in history], [r['mean_score'] for r in history], color=color, linestyle=style, alpha=.8, label=name)
    axes[0].set(xlabel='Physical training trials', ylabel='Mean candidate balance score', title='Learning changes the quality of proposals')
    axes[0].legend(fontsize=7)
    sets = data['batteries']['grid']['by_set']
    for name, groups in sets.items():
        if name.count('-') > 1:
            continue
        pushed = [(float(label.split('-')[1]), r) for label, r in groups.items() if float(label.split('-')[1]) > 0]
        axes[1].plot([m for m, r in pushed], [r['recovered'] / r['n'] for m, r in pushed], marker='o', markersize=3, label=name)
    axes[1].set(xlabel='Forward root velocity increment (m/s)', ylabel='Quiet recovery fraction', xlim=(.19, .46), ylim=(-.03, 1.03), title='Held-out pushes; six conditions per magnitude')
    axes[1].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(out / 'learning-and-recovery.png', dpi=160)
    plt.close(fig)
    print(json.dumps({k: v for k, v in data.items() if k != 'learning'}, indent=2))


if __name__ == '__main__':
    main()
