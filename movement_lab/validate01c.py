"""Read-only audit of preserved 01C records; never regenerates research artifacts."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from movement_lab.evaluate import sha
from movement_lab.study01c import check

ROOT = Path('results/01c')
BASE = '813d32698f2e0a7e73eb2ef5e4cd51ff81f6123c'
TRIALS = {'baseline', 'flex', 'extend', 'head-positive', 'head-negative'}


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], text=True).strip()


def audit(root=ROOT):
    import mujoco
    # Original manifests describe the full local evidence, including excluded files.
    for manifest in (root/'manifest.json', root/'pilot-gpu/manifest.json'):
        for name, digest in json.loads(manifest.read_text()).items():
            assert sha(manifest.parent/name) == digest, str(manifest.parent/name)
    for name, digest in json.loads((root/'evidence01b-provenance.json').read_text()).items():
        assert sha(name) == digest, name
    pilot = root/'pilot-gpu'
    baseline = pilot/'baseline'
    historical = Path('results/01b/departures/baseline')
    assert (baseline/'resolved_config.txt').read_bytes() == (historical/'resolved_config.txt').read_bytes()
    assert sha(baseline/'replay_model.mjb') == sha(historical/'replay_model.mjb')
    with np.load(baseline/'zero/trial.npz') as c, np.load(historical/'zero/trial.npz') as old:
        for key in old:
            assert np.array_equal(c[key][:len(old[key])], old[key]), ('historical prefix', key)
    saved = json.loads((pilot/'independent-validation.json').read_text())
    summary = json.loads((pilot/'summary.json').read_text())
    assert set(saved) == TRIALS
    assert set(summary) == TRIALS
    # The executed driver persists summary before adding its in-memory stop marker.
    cost = json.loads((pilot/'cost.json').read_text())
    assert cost['completed'] == len(TRIALS) + 1
    assert 510 <= cost['wall_s'] < 600
    assert not (pilot/'rock-forward').exists() and not (pilot/'rock-backward').exists()
    assert not (pilot/'final-control').exists()
    results = {}
    for name in sorted(TRIALS):
        out = pilot/name
        result = check(out, baseline)
        for key, value in result.items():
            assert saved[name][key] == value, (name, key, 'saved validation')
            assert summary[name][key] == value, (name, key, 'saved summary')
        assert result['physical_valid'] and result['stable_final'] and result['stable_hold']
        f = next(out.glob('*/metadata.json'))
        m = json.loads(f.read_text())
        assert m['requested_seconds'] == 10 and m['physics_dt'] == .005
        assert not m['terminated'] and not m['truncated'] and not m['stop_reasons']
        prov = json.loads((out/'provenance.json').read_text())
        up = Path(prov['upstream'])
        assert git('-C', str(up), 'rev-parse', 'HEAD') == prov['upstream_commit']
        assert git('-C', str(up), 'status', '--porcelain') == prov['upstream_status'] == ''
        for source, digest in prov['upstream_file_sha256'].items():
            assert sha(up/source) == digest, source
        model = mujoco.MjModel.from_binary_path(str(out/'replay_model.mjb'))
        data = mujoco.MjData(model)
        terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
        assert terrain >= 0
        assert (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT).all()
        moment = np.zeros((model.nu, model.nv))
        for i, j in enumerate(model.actuator_trnid[:, 0]):
            moment[i, model.jnt_dofadr[j]] = model.actuator_gear[i, 0]
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/'+n) for n in m['joint_names']]
        with np.load(f.parent/'trial.npz') as a:
            assert len(a['time_s']) == 2000 and 9.99 < a['time_s'][-1] < 10.01
            assert np.allclose(a['actuator_force'] @ moment[:, model.jnt_dofadr[joints]], a['joint_force_nm'], atol=1e-6)
            assert abs(a['joint_force_nm']).max() <= .9601
            for i in range(len(a['time_s'])):
                data.qpos[:] = a['qpos'][i]; data.qvel[:] = a['qvel'][i]
                mujoco.mj_forward(model, data)
                count = nonfoot = 0
                feet = np.zeros(2, dtype=int)
                for contact in data.contact:
                    if terrain not in (contact.geom1, contact.geom2):
                        continue
                    count += 1
                    geom = contact.geom2 if contact.geom1 == terrain else contact.geom1
                    geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
                    names = ('left_foot_collision', 'right_foot_collision')
                    nonfoot += int(not any(n in geom_name for n in names))
                    for fi, foot in enumerate(names):
                        feet[fi] += int(foot in geom_name)
                assert count == a['native_ground_contacts'][i]
                assert nonfoot == a['native_nonfoot_contacts'][i]
                assert np.array_equal(feet, a['native_foot_contacts'][i])
        results[name] = dict(samples=2000, requested_seconds=10, **result)
    assert results['flex']['classification'] == 'stable changed configuration'
    for name in ('extend', 'head-positive', 'head-negative'):
        assert results[name]['classification'] == 'stable transition with partial return'
    assert round(max(r['worst_penetration_mm'] for r in results.values()), 3) == 1.156
    assert not git('diff', BASE, '--name-only', '--', 'results/00b', 'results/00c', 'results/01a', 'results/01b')
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('root', type=Path, nargs='?', default=ROOT)
    args = p.parse_args()
    print(json.dumps(audit(args.root), indent=2))
    print('PASS: five ten-second trials; original hashes, matched controls, physical gates, contacts, actuator mapping, outcomes and unchanged upstream')


if __name__ == '__main__':
    main()
