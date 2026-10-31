"""Interpretable across-trial exploration; no actor training. NumPy-only core."""
import argparse
import json
from pathlib import Path
import numpy as np
from movement_lab.controlled01b import gate
from movement_lab.departure_metrics01b import stable_window
from movement_lab.evaluate import sha

ACTIONS = [(h, f) for h in (-1., -.5, 0., .5, 1.) for f in (-1., -.5, 0., .5, 1.) if (h, f) != (0., 0.)]
PHASES = ((2.6, 3.6), (4., 5.), (7., 8.), (9.5, 10.01))


def distance(a, b):
    # Maximum normalized difference of sustained states; no peak/noise reward.
    return float(np.max(np.abs(np.asarray(a)-np.asarray(b))))


class Explorer:
    def __init__(self, seed, adaptive):
        self.rng = np.random.default_rng(seed)
        self.adaptive = adaptive
        self.history = []
        self.members = []

    def propose(self):
        remaining = [a for a in ACTIONS if list(a) not in [r['action'] for r in self.history]]
        if not remaining:
            raise ValueError('action space exhausted')
        if not self.adaptive or not self.history or self.rng.random() < .25:
            return remaining[int(self.rng.integers(len(remaining)))], 'uniform without replacement'
        # A nearest-observation model transfers local outcome and validity evidence.
        # Valid neighborhoods favor half-unit mutations; failures favor distance; only valid outcomes supply novelty.
        scores = []
        for a in remaining:
            d = np.linalg.norm(np.asarray([r['action'] for r in self.history])-a, axis=1)
            i = int(np.argmin(d)); r = self.history[i]
            novelty = min([distance(r['descriptor'], m['descriptor']) for m in self.members
                           if m['id'] != r['id']], default=1.) if r['eligible'] else 0.
            scores.append(min(novelty, 3.) - abs(float(d[i])-.5) if r['eligible'] else float(d[i])-2.)
        return remaining[int(np.argmax(scores))], 'nearest valid neighborhood mutation; failed-region distance; novelty tie score'

    def observe(self, record):
        r = dict(record)
        r['action'] = list(r['action'])
        r['novelty'] = min((distance(r['descriptor'], m['descriptor']) for m in self.members), default=None) if r['eligible'] else None
        r['discovery'] = bool(r['eligible'] and (r['novelty'] is None or r['novelty'] >= 1.))
        self.history.append(r)
        if r['discovery']:
            self.members.append(r)
        return r


def measure(a, baseline, metadata):
    t = a['time_s']; n = len(t)
    assert len(baseline['time_s']) == 2000 and baseline['time_s'][-1] >= 9.99, 'incomplete baseline'
    assert gate(baseline, baseline=True)['pass'], 'invalid baseline'
    assert np.array_equal(t, baseline['time_s'][:n]), 'unmatched time samples'
    complete = n == 2000 and t[-1] >= 9.99
    valid = complete and gate(a, metadata['stop_reasons'], False)['pass']
    masks = [(t >= lo) & (t < hi) for lo, hi in PHASES]
    stable = bool(valid and all(stable_window(a, w) for w in (masks[0], masks[-1])))
    descriptor = None
    if complete and all(w.any() for w in masks):
        def features(x):
            q = x['body_pose_w'][:, 0, 3:7]
            # Rotation-matrix entries avoid quaternion sign discontinuities.
            w, u, v, z = q.T
            orientation = np.stack((2*(u*v-w*z), 2*(u*z+w*v), 2*(v*z-w*u)), axis=1)
            return np.concatenate((x['joint_position_rad']/.005,
                                   x['com_position_w_m']/.0005,
                                   orientation/np.sin(np.deg2rad(.5))), axis=1)
        delta = features(a)-features(baseline)
        descriptor = np.concatenate([delta[w].mean(0) for w in masks]).tolist()
    response = descriptor is not None and distance(descriptor, np.zeros(len(descriptor))) >= 1.
    return dict(physical_valid=bool(valid), stable=stable, eligible=bool(stable and response),
                descriptor=descriptor, stop_reasons=metadata['stop_reasons'], samples=n,
                classification='invalid / incomplete' if not valid else 'unstable hold or endpoint' if not stable else 'measurable stable outcome' if response else 'below response threshold')


def recorded_proof(root, out):
    """Existing records test mechanics, never imply a fair search comparison."""
    from movement_lab.study01c import check
    out.mkdir(parents=True, exist_ok=False)
    explorer = Explorer(42, True)
    baseline = dict(np.load(root/'baseline/zero/trial.npz'))
    rows = []
    for name, action in [('flex', (0, 1)), ('extend', (0, -1)), ('head-positive', (1, 0)), ('head-negative', (-1, 0))]:
        path = root/name
        integrity = check(path, root/'baseline')
        meta = next(path.glob('*/metadata.json'))
        a = dict(np.load(meta.parent/'trial.npz')); m = json.loads(meta.read_text())
        rows.append(explorer.observe(dict(id=name, action=action, integrity=integrity['validation'],
                                         trajectory_sha256=sha(meta.parent/'trial.npz'), **measure(a, baseline, m))))
    (out/'archive.json').write_text(json.dumps(rows, indent=2)+'\n')
    plot(rows, out)


def plot(rows, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    good = [r for r in rows if r.get('descriptor') is not None]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for r in good:
        d = np.asarray(r['descriptor']).reshape(4, -1)
        # Signed COM x/z in mm: 14 joints then COM xyz, then orientation.
        ax[0].scatter(d[0, 14]*.5, d[0, 16]*.5)
        ax[0].annotate(r['id'], (d[0, 14]*.5, d[0, 16]*.5), fontsize=7)
        ax[1].plot(range(4), np.max(abs(d[:, :14]), axis=1)*.005, marker='o', label=r['id'])
    ax[0].set(xlabel='Hold COM x difference (mm)', ylabel='Hold COM z difference (mm)')
    ax[1].set(xticks=range(4), xticklabels=['hold', 'return', 'late', 'final'], ylabel='Max absolute phase-mean joint difference (rad)')
    if good: ax[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(out/'outcomes.png'); plt.close(fig)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--recorded', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    a = p.parse_args(); recorded_proof(a.recorded, a.out)

if __name__ == '__main__': main()
