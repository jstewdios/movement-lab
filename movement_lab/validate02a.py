"""Read-only 02A saved-record and proposal-history audit; no simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
from movement_lab.evaluate import sha
from movement_lab.explore02a import Explorer, ACTIONS, measure
from movement_lab.study01c import check

ROOT = Path('results/02a')
BASE = 'aa51dc70d6050491cc03b1aaf9f2ce6bf88bcb94'
ORDER = ['baseline', 'adaptive-0', 'control-0', 'control-1', 'adaptive-1']


def git(*args):
    return subprocess.check_output(['git', '--no-optional-locks', *args], text=True).strip()


def findings(attempts, archives):
    """Strict completed-pilot inventory; missing conditions cannot disappear in totals."""
    assert [r['id'] for r in attempts] == ORDER, 'missing, reordered or extra conditions'
    assert set(archives) == {'adaptive', 'control'}, 'missing or extra arm'
    for arm in archives:
        expected = [r for r in attempts if r['arm'] == arm]
        assert len(expected) == len(archives[arm]) == 2, 'incomplete arm'
        assert [r['outcome'] for r in expected] == archives[arm], 'attempt/archive mismatch'
    assert all(r['status'] == 'validated record' and r['returncode'] == 0 for r in attempts)
    assert all(r['validation']['physical_valid'] for r in attempts), 'physical validation failure'
    return {arm: dict(evaluated=len(rows), discoveries=sum(r['discovery'] for r in rows),
                     physically_invalid=sum(not r['physical_valid'] for r in rows),
                     valid_unproductive=sum(r['physical_valid'] and not r['discovery'] for r in rows),
                     discovery_curve=np.cumsum([r['discovery'] for r in rows]).tolist())
            for arm, rows in archives.items()}


def audit():
    root = ROOT; pilot = root/'pilot-gpu'
    # Original inventories are immutable; they include excluded local raw files.
    manifests = {}
    for p in root.rglob('manifest.json'):
        if 'publication' in p.parts or 'review-fixes' in p.parts: continue
        entries = json.loads(p.read_text())
        for n, digest in entries.items(): assert sha(p.parent/n) == digest, (str(p), n)
        manifests[str(p)] = len(entries)
    for n, digest in json.loads((root/'source-hashes.json').read_text()).items():
        if n in ('.gitignore', 'README.md'):
            data = Path(n).read_bytes()
            assert any(hashlib.sha256(data[:i+1]).hexdigest() == digest for i, value in enumerate(data) if value == 10), ('original shared-doc prefix', n)
        elif n in ('movement_lab/study02a.py', 'movement_lab/report02a.py'):
            # Experimental source hashes describe the original publication, not review fixes.
            from movement_lab.publication02a import ORIGINAL_PUBLICATION
            original = subprocess.check_output(['git', 'show', ORIGINAL_PUBLICATION+':'+n])
            assert hashlib.sha256(original).hexdigest() == digest, n
        else:
            assert sha(n) == digest, n
    changed = git('diff', BASE, '--name-only').splitlines()
    from movement_lab.publication02a import allowed_publication_path
    assert all(allowed_publication_path(n) for n in changed), ('earlier tracked changes', changed)
    for n in ('.gitignore', 'README.md'):
        old = subprocess.check_output(['git', 'show', BASE+':'+n])
        assert Path(n).read_bytes().startswith(old), ('earlier documentation rewritten', n)
    attempts = json.loads((pilot/'attempts.json').read_text())
    archives = json.loads((pilot/'archive.json').read_text())
    stats = findings(attempts, archives)
    assert stats == json.loads((pilot/'metrics.json').read_text())
    cost = json.loads((pilot/'cost.json').read_text())
    assert cost['status'] == 'complete' and round(cost['wall_s'], 1) == 364.6 and cost['budget_s'] == 600
    plan = json.loads((pilot/'PLAN.json').read_text())
    assert plan['order'] == ORDER and plan['trials_per_arm'] == 2
    assert plan['search_seed'] == plan['simulator_seed'] == 42
    assert json.loads((pilot/'proposals.json').read_text()) == [
        {k:r[k] for k in ('id', 'action', 'proposal_reason', 'status')} for r in attempts]
    baseline = dict(np.load(pilot/'baseline/zero/trial.npz'))
    historical = Path('results/01c/pilot-gpu/baseline')
    with np.load(historical/'zero/trial.npz') as old:
        assert set(old) == set(baseline) and all(np.array_equal(baseline[k], old[k]) for k in old)
    arms = {a: Explorer(42, a == 'adaptive') for a in archives}
    checks = {}
    for row in attempts:
        out = pilot/row['id']; arm = row['arm']
        checked = check(out, pilot/'baseline')
        assert checked == row['validation'], ('validation drift', row['id'])
        mpath = next(out.glob('*/metadata.json')); m = json.loads(mpath.read_text())
        a = dict(np.load(mpath.parent/'trial.npz'))
        assert len(a['time_s']) == 2000 and m['requested_seconds'] == 10
        assert m['seed'] == 42 and m['initialization_root_z_m'] == .1181823627
        assert not m['terminated'] and not m['truncated'] and not m['stop_reasons']
        assert m['pulse']['rise_s'] == .6 and m['pulse']['hold_s'] == 1 and m['pulse']['return_s'] == 1.4
        assert (out/'resolved_config.txt').read_bytes() == (historical/'resolved_config.txt').read_bytes()
        assert sha(out/'replay_model.mjb') == sha(historical/'replay_model.mjb')
        prov = json.loads((out/'provenance.json').read_text()); up = Path(prov['upstream'])
        assert git('-C', str(up), 'rev-parse', 'HEAD') == prov['upstream_commit']
        assert git('-C', str(up), 'status', '--porcelain') == prov['upstream_status'] == ''
        for n, digest in prov['upstream_file_sha256'].items(): assert sha(up/n) == digest, n
        # Validate all native contact counts and actuator force mapping, supplementing check().
        import mujoco
        model = mujoco.MjModel.from_binary_path(str(out/'replay_model.mjb')); d = mujoco.MjData(model)
        terrain = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'terrain')
        assert terrain >= 0
        joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'robot/'+n) for n in m['joint_names']]
        assert (model.actuator_trntype == mujoco.mjtTrn.mjTRN_JOINT).all()
        moment = np.zeros((model.nu, model.nv))
        for i, j in enumerate(model.actuator_trnid[:, 0]): moment[i, model.jnt_dofadr[j]] = model.actuator_gear[i, 0]
        assert np.allclose(a['actuator_force'] @ moment[:, model.jnt_dofadr[joints]], a['joint_force_nm'], atol=1e-6)
        assert abs(a['joint_force_nm']).max() <= .9601
        for i in range(len(a['time_s'])):
            d.qpos[:] = a['qpos'][i]; d.qvel[:] = a['qvel'][i]; mujoco.mj_forward(model, d)
            ground = nonfoot = 0; feet = np.zeros(2, dtype=int)
            for contact in d.contact:
                if terrain not in (contact.geom1, contact.geom2): continue
                ground += 1
                geom = contact.geom2 if contact.geom1 == terrain else contact.geom1
                name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ''
                labels = ('left_foot_collision', 'right_foot_collision')
                nonfoot += int(not any(n in name for n in labels))
                for fi, label in enumerate(labels): feet[fi] += int(label in name)
            assert ground == a['native_ground_contacts'][i] and nonfoot == a['native_nonfoot_contacts'][i]
            assert np.array_equal(feet, a['native_foot_contacts'][i])
        if arm:
            # Only this arm's previous outcomes enter its proposal RNG/model.
            action, reason = arms[arm].propose()
            assert list(action) == row['action'] and reason == row['proposal_reason'], ('proposal replay', row['id'])
            assert action in ACTIONS
            h, f = action
            expected = {'head_yaw': .08*h, 'left_hip_pitch': -.03*f, 'right_hip_pitch': .03*f,
                        'left_knee': .05*f, 'right_knee': -.05*f, 'left_ankle': -.02*f, 'right_ankle': .02*f}
            assert m['amplitudes_rad'] == expected
            observed = arms[arm].observe(dict(id=row['id'], action=action,
                trajectory_sha256=sha(mpath.parent/'trial.npz'), **measure(a, baseline, m)))
            assert observed == row['outcome'], ('measured archive mismatch', row['id'])
        checks[row['id']] = dict(samples=len(a['time_s']), **checked)
    assert {k: e.history for k, e in arms.items()} == archives
    # Recompute the offline development archive without regenerating its evidence.
    oldroot = Path('results/01c/pilot-gpu'); replay = Explorer(42, True); offline = []
    for name, action in [('flex',(0,1)), ('extend',(0,-1)), ('head-positive',(1,0)), ('head-negative',(-1,0))]:
        out = oldroot/name; checked = check(out, oldroot/'baseline')
        mp = next(out.glob('*/metadata.json')); m = json.loads(mp.read_text()); a = dict(np.load(mp.parent/'trial.npz'))
        offline.append(replay.observe(dict(id=name, action=action, integrity=checked['validation'],
            trajectory_sha256=sha(mp.parent/'trial.npz'), **measure(a, baseline, m))))
    assert offline == json.loads((root/'offline/archive.json').read_text())
    assert stats['adaptive']['discoveries'] == 1 and stats['control']['discoveries'] == 2
    second = archives['adaptive'][1]
    assert second['eligible'] and not second['discovery'] and 0 < second['novelty'] < 1
    # Claims below are checked against recomputed records, not inferred from command names.
    report = (pilot/'REPORT.md').read_text()
    for phrase in ('364.61 s', 'Adaptive retained 1 distinct outcome(s), versus 2',
                   'No motor policy was trained', 'no discovery has independent confirmation',
                   'cannot establish an algorithm-level advantage', 'valid but redundant'):
        assert phrase in report, ('missing or stale report conclusion', phrase)
    return dict(original_manifests_verified=manifests, trials=checks, metrics=stats,
                wall_s=cost['wall_s'], proposal_replay=True, offline_replay=True,
                upstream_unchanged=True, earlier_tracked_records_unchanged=True,
                independent_confirmation=False, motor_policy_trained=False)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--out', type=Path)
    args = p.parse_args(); result = audit(); payload = json.dumps(result, indent=2)+'\n'
    if args.out:
        with args.out.open('x') as f: f.write(payload)
    print(payload)

if __name__ == '__main__': main()
