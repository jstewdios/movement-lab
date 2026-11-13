"""02E: reproducibility tooling, information boundary, estimators and exact fork equivalence."""
import importlib.util
import json
import math
from dataclasses import asdict, fields
from pathlib import Path
import unittest
import numpy as np

from movement_lab.sim02b import DEFAULT_MODEL, rpy_matrix, HOME
from movement_lab.estimate02e import rot_between, rotvec, expmap, rotation_fit, Settings, Frontend
from movement_lab.sim02e import Condition, gravity_direction, mounting, encoder_offsets

PHYSICS = (DEFAULT_MODEL / 'model.mjb').exists() and importlib.util.find_spec('bam') is not None


class MathTests(unittest.TestCase):
    def test_rotation_helpers_round_trip(self):
        rng = np.random.default_rng(0)
        for _ in range(20):
            v = rng.normal(size=3) * .3
            np.testing.assert_allclose(rotvec(expmap(v)), v, atol=1e-12)
            a, b = rng.normal(size=3), rng.normal(size=3)
            R = rot_between(a, b)
            np.testing.assert_allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-12)
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)

    def test_rotation_fit_recovers_a_misalignment_and_only_what_is_observable(self):
        Rb = rpy_matrix((1.5, -2.0, .7))                 # sensor misalignment
        rng = np.random.default_rng(1)
        trunk = rng.normal(size=(200, 3))                 # rich rotation: fully observable
        R, _ = rotation_fit(trunk @ Rb, trunk)             # gyro = Rb^T w  (row form: w @ Rb)
        np.testing.assert_allclose(R, Rb, atol=1e-10)
        # Rotation about one axis only reveals that axis's tilt, never rotation about it.
        roll = np.outer(np.sin(np.linspace(0, 6, 200)), [1., 0., 0.])
        R1, _ = rotation_fit(roll @ rpy_matrix((0., 2., 0.)), roll, single_axis=True)
        self.assertAlmostEqual(math.degrees(rotvec(R1)[1]), 2.0, places=6)
        R2, _ = rotation_fit(roll @ rpy_matrix((2., 0., 0.)), roll, single_axis=True)
        self.assertAlmostEqual(float(np.linalg.norm(rotvec(R2))), 0.0, places=9)

    def test_floor_slope_and_drift_conventions(self):
        np.testing.assert_array_equal(gravity_direction((0., 0.)), [0., 0., -1.])
        g = gravity_direction((0., 2.))
        # Positive pitch slope: gravity has a +x component in the floor frame, i.e. the floor
        # descends toward +x and the robot faces downhill. A correct IMU on a robot standing
        # perpendicular to such a floor reads the same sign as a +2 deg pitch misalignment.
        self.assertGreater(g[0], 0); self.assertAlmostEqual(g[0], math.sin(math.radians(2)), places=12)
        np.testing.assert_allclose(rpy_matrix((0., 2., 0.)).T @ [0, 0, -1.], g, atol=1e-15)
        c = Condition(drift_deg=(0., 2., 0.), drift_start_s=.5, drift_duration_s=4.)
        np.testing.assert_array_equal(mounting(c, 0.), rpy_matrix((0., 0., 0.)))
        np.testing.assert_allclose(mounting(c, 2.5), rpy_matrix((0., 1., 0.)), atol=1e-15)
        np.testing.assert_allclose(mounting(c, 9.), rpy_matrix((0., 2., 0.)), atol=1e-15)

    def test_encoder_offsets_are_bounded_seeded_and_off_by_default(self):
        np.testing.assert_array_equal(encoder_offsets(Condition(seed=3)), np.zeros(14))
        a = encoder_offsets(Condition(seed=3, encoder_offset_rad=.015))
        np.testing.assert_array_equal(a, encoder_offsets(Condition(seed=3, encoder_offset_rad=.015)))
        self.assertLessEqual(np.abs(a).max(), .015); self.assertGreater(np.abs(a).max(), 0)

    def test_frontend_only_sees_declared_inputs_and_gates_residual(self):
        fe = Frontend(Settings(kind='gate'))
        g, w, probe, enabled = fe(.5, np.array([.1, 0, -1.]), np.ones(3), HOME)
        np.testing.assert_array_equal(g, [.1, 0, -1.]); np.testing.assert_array_equal(probe, np.zeros(14))
        self.assertFalse(enabled)
        self.assertTrue(fe(1.6, g, w, HOME)[3])
        self.assertTrue(Frontend(Settings(kind='none'))(0., g, w, HOME)[3])
        import inspect
        self.assertEqual(list(inspect.signature(Frontend.__call__).parameters), ['self', 't', 'grav', 'gyro', 'q_enc'])

    def test_prior_frontend_corrects_smoothly_after_its_window(self):
        Rb = rpy_matrix((0., 2., 0.))
        g_true = np.array([0., 0., -1.])
        fe = Frontend(Settings(kind='prior', prior_gravity=(0., 0., -1.)))
        seen = []
        for k in range(100):
            out = fe(k * .02, Rb.T @ g_true, np.zeros(3), HOME)
            seen.append(out[0])
        self.assertAlmostEqual(math.degrees(rotvec(expmap(fe.rv))[1]), 2., places=9)
        np.testing.assert_allclose(seen[10], Rb.T @ g_true)          # before estimate: raw
        np.testing.assert_allclose(seen[-1], g_true, atol=1e-12)      # after blend: corrected
        mid = seen[int(round(1.45 / .02))]
        self.assertTrue(0 < abs(mid[0]) < abs((Rb.T @ g_true)[0]))    # ramped, not stepped

    def test_support_margins_project_the_com_along_gravity(self):
        # Regression (PR #11 review): the COM's ground point is com + s g with s = com_z / -g_z > 0.
        # The original code subtracted, moving the point uphill instead of downhill.
        from types import SimpleNamespace
        from unittest import mock
        import movement_lab.sim02e as S
        poly = np.array([[-.04, -.03], [.06, -.03], [.06, .05], [-.04, .05]])   # CCW; toe +x, heel -x; asymmetric
        h = .2
        sim, data = SimpleNamespace(trunk=0), SimpleNamespace(subtree_com=np.array([[0., 0., h]]))
        with mock.patch.object(S, 'support_polygon', return_value=poly):
            np.testing.assert_allclose(S.support_margins(sim, data, gravity_direction((0., 0.))), (.03, .06, .04), atol=1e-12)
            shift = h * math.tan(math.radians(2))
            g = gravity_direction((0., 2.))                       # floor descends toward +x
            _, toe, heel = S.support_margins(sim, data, g)
            self.assertAlmostEqual(toe, .06 - shift, places=12)   # downhill: the gravity line lands nearer the toe
            self.assertAlmostEqual(heel, .04 + shift, places=12)
            low, _, _ = S.support_margins(sim, data, gravity_direction((2., 0.)))   # gravity has a -y component
            self.assertAlmostEqual(low, .03 - shift, places=12)
            data.subtree_com[0] = (.06 - shift / 2, 0., h)        # the gravity line leaves past the toe
            self.assertEqual(S.support_margins(sim, data, g), (-1., -1., -1.))
            self.assertGreater(S.support_margins(sim, data, gravity_direction((0., -2.)))[1], 0)   # uphill: inside

    def test_margin_correction_is_exactly_the_declared_edit_and_spares_training(self):
        from movement_lab.correct02e import apply_edits, EDITS, BEFORE_SHA256, RECORD, MARGIN_FIELDS
        from movement_lab.learn02c import digest
        archived = Path('results/02e') / RECORD / 'source/sim02e.py'
        if not archived.exists():
            self.skipTest('no published correction record')
        self.assertEqual(digest(archived), BEFORE_SHA256)
        self.assertEqual(apply_edits(archived.read_text(), EDITS), Path('movement_lab/sim02e.py').read_text())
        with self.assertRaises(ValueError):
            apply_edits(Path('movement_lab/sim02e.py').read_text(), EDITS)   # already applied: no silent double edit
        # Learning rollouts never compute margins, so training pinned to the archived source is unaffected.
        self.assertNotIn('margins', Path('movement_lab/learn02e.py').read_text())
        self.assertEqual(len(MARGIN_FIELDS), 6)


class GateAuditTests(unittest.TestCase):
    """Regression (PR #11 review): the gate audit recomputes from the requested root and writes nothing."""

    @staticmethod
    def _rows(rng):
        from movement_lab.diag02e import MISALIGNED_D
        settings = ['nominal', *MISALIGNED_D, 'slope-2', 'slope+2']
        return [dict(setting=s, set=k, condition=dict(seed=7_200_000_000 + 10 * i + j), recovered_200hz=bool(rng.random() < .6))
                for i, s in enumerate(settings) for j, k in enumerate(('quiet', 'back-.08', 'fwd-.20'))]

    @staticmethod
    def _snapshot(folder):
        return {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in sorted(folder.rglob('*')) if p.is_file()} if folder.exists() else {}

    def test_gates_are_recomputed_under_the_requested_root_without_writing(self):
        import contextlib, io, tempfile
        from movement_lab import diag02e as D
        from movement_lab.eval02c import write_gzip
        from movement_lab.study02e import gate_learning
        from movement_lab.validate02e import Audit, audit_gates
        published = Path('results/02e/pilot')
        before_published = self._snapshot(published)
        rng = np.random.default_rng(5)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'alternate'
            (root / 'pilot/learning').mkdir(parents=True); (root / 'pilot/frontend').mkdir()
            for n in ('history-s72', 'history-rand-s72', 'memoryless-s72', 'memoryless-rand-s72'):
                write_gzip(root / f'pilot/learning/{n}.jsonl.gz', self._rows(rng))
            for n in ('reflex', *[f'{a}-s{s}' for a in ('history', 'memoryless') for s in (1, 2, 3)]):
                for fe in ('none', 'static', 'prior', 'gate'):
                    write_gzip(root / f'pilot/frontend/{n}__{fe}.jsonl.gz', self._rows(rng))
            with contextlib.redirect_stdout(io.StringIO()):
                gate_learning(root); D.gate(root)
            self.assertEqual(self._snapshot(published), before_published)   # a gate for another root leaves published evidence alone
            mine = json.loads((root / 'pilot/gate-learning.json').read_text())
            if (published / 'gate-learning.json').exists():
                self.assertNotEqual(mine, json.loads((published / 'gate-learning.json').read_text()))
            before_alternate = self._snapshot(root)
            check = Audit()
            audit_gates(root, check)
            self.assertEqual(check.count, 2)
            self.assertEqual(self._snapshot(root), before_alternate)        # the audit writes nothing under the audited root
            # A decision that does not follow from this root's rows is caught (it used to be compared with itself).
            for name, key, field in (('gate-learning.json', 'history-rand-s72', 'misaligned_gains'), ('gate-frontend.json', 'static', 'misaligned_net')):
                path = root / 'pilot' / name
                saved = path.read_bytes()
                d = json.loads(saved); d[key][field] += 1
                path.write_text(json.dumps(d, indent=1) + '\n', newline='\n')
                with self.assertRaises(AssertionError):
                    audit_gates(root, Audit())
                self.assertNotEqual(path.read_bytes(), saved)               # the audit did not "restore" anything
                path.write_bytes(saved)
        self.assertEqual(self._snapshot(published), before_published)


class SeedAndCurriculumTests(unittest.TestCase):
    def test_battery_seeds_paired_and_disjoint_from_all_training_and_02d(self):
        from movement_lab import diag02e as D
        a = D.battery(D.SETTINGS_1A, D.KINDS, 7_000_000_000)
        d = D.battery(D.SETTINGS_D, D.KINDS, 7_200_000_000)
        h = D.battery(D.SETTINGS_H, D.KINDS_H, 8_000_000_000, reps=3, onset=(2.2, 3.4))
        self.assertEqual((len(a), len(d), len(h)), (150, 90, 360))
        by_setting = {}
        for s, k, c in a:
            by_setting.setdefault(s, []).append((k, c.seed, c.push_time_s, c.false_gyro_time_s))
        first = next(iter(by_setting.values()))
        self.assertTrue(all(v == first for v in by_setting.values()))   # paired across settings
        seeds = lambda b: {c.seed for _, _, c in b}
        groups = [seeds(a), seeds(d), seeds(h), set(range(7_100_000_000, 7_100_000_004)), set(range(7_150_000_000, 7_150_000_008))]
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                self.assertFalse(groups[i] & groups[j])
        for g in groups:   # training 4e9 range, 02C 2e9/3e9, 02D 5e9/6e9
            self.assertTrue(all(7_000_000_000 <= s < 9_000_000_000 for s in g))
        self.assertTrue(all(c.push_time_s is None or 2.2 <= c.push_time_s < 3.4 for _, _, c in h))

    def test_learn02e_conditions_reduce_to_02d_and_randomize_only_misalignment(self):
        from movement_lab.learn02e import conditions, misalignment
        from movement_lab.learn02d import conditions as old
        for g in (0, 5):
            new, ref = conditions(72, g, True, 0.), old(72, g, True)
            for x, y in zip(new, ref):
                self.assertEqual({f.name: getattr(x, f.name) for f in fields(y)}, asdict(y))
            rnd = conditions(72, g, True, 4.)
            for x, y in zip(rnd, ref):
                kw = asdict(x); kw.pop('imu_rpy_deg')
                ref_kw = asdict(y); ref_kw.pop('imu_rpy_deg')
                self.assertEqual({k: kw[k] for k in ref_kw}, ref_kw)
                r, p, yaw = x.imu_rpy_deg
                self.assertEqual(yaw, 0.); self.assertLessEqual(math.hypot(r, p), 4. + 1e-12)
        self.assertEqual(misalignment(5, 0.), (0., 0., 0.))
        self.assertEqual(misalignment(5, 4.), misalignment(5, 4.))


@unittest.skipUnless(PHYSICS, 'local compiled model / BAM unavailable')
class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from movement_lab.sim02b import Sim
        from movement_lab.sim02c import BASE
        from movement_lab.control02d import load_basis
        from movement_lab.estimate02e import Kinematics
        cls.sim = Sim(); cls.theta = np.load(BASE)['final']; cls.basis = load_basis(); cls.kin = Kinematics(cls.sim)
        cls.h1 = np.load('results/02d/runs/history-s1/final.npz')['best']

    def roll(self, vec, c, mode='learned', **kw):
        from movement_lab.sim02e import rollout
        from movement_lab.control02d import Design
        return rollout(self.sim, self.theta, vec, c, Design(), record=True, terminate=False, mode=mode, basis=self.basis, **kw)

    def test_fork_is_bit_identical_to_02d_without_interventions(self):
        from movement_lab import sim02d
        from movement_lab.control02d import Design
        cases = [(self.h1, 'learned', Condition(seed=11, duration_s=2.5, joint_noise_rad=.01, push_time_s=1.5, push_dv=(.32, 0.), imu_rpy_deg=(0., 2., 0.))),
                 (np.zeros(22), 'imu02c', Condition(seed=12, duration_s=2.5, pulse_time_s=1.5, pulse_duration_s=.12, pulse_force=(.4, 0., 0.))),
                 (np.zeros(22), 'reflex', Condition(seed=13, duration_s=2.5, false_gyro_time_s=1.5, false_gyro_rad_s=.75))]
        for vec, mode, c in cases:
            a, ra = sim02d.rollout(self.sim, self.theta, vec, c, Design(), record=True, terminate=False, mode=mode, basis=self.basis)
            b, rb = self.roll(vec, c, mode)
            for k in ra:
                np.testing.assert_array_equal(ra[k], rb[k])
            self.assertEqual(a, {k: v for k, v in b.items() if k in a})

    def test_passthrough_frontend_and_margins_do_not_change_physics(self):
        c = Condition(seed=14, duration_s=2., push_time_s=1., push_dv=(.2, 0.), encoder_offset_rad=.015)
        _, plain = self.roll(self.h1, c)
        s, fe = self.roll(self.h1, c, frontend=Frontend(Settings(kind='none'), self.kin), margins=True)
        np.testing.assert_array_equal(plain['qpos'], fe['qpos'])
        self.assertEqual(len(s['margins']), 100)

    def test_gate_holds_residual_off_during_calibration_only(self):
        c = Condition(seed=15, duration_s=2.2)
        _, r = self.roll(self.h1, c, frontend=Frontend(Settings(kind='gate'), self.kin))
        t = r['time_s']
        self.assertEqual(float(np.abs(r['residual'][t < 1.6 - 1e-9]).max()), 0.)
        self.assertGreater(float(np.abs(r['residual'][t > 1.62]).max()), 0.)

    def test_slope_changes_physical_gravity_and_restores_model(self):
        g0 = self.sim.m.opt.gravity.copy()
        _, r = self.roll(np.zeros(22), Condition(seed=16, duration_s=1., slope_deg=(0., 2.)), mode='reflex')
        np.testing.assert_array_equal(self.sim.m.opt.gravity, g0)
        # Projected gravity in the trunk frame is computed against the rotated gravity: at reset the
        # trunk is perpendicular to the floor, so it reads the slope (before the reflex responds).
        self.assertAlmostEqual(float(r['grav_true'][0, 0]), math.sin(math.radians(2)), delta=.003)

    def test_static_estimator_on_level_floor_and_slope_confound(self):
        c = Condition(seed=17, duration_s=1.7, imu_rpy_deg=(0., 2., 0.), joint_noise_rad=.01)
        s, _ = self.roll(np.zeros(22), c, mode='reflex', frontend=Frontend(Settings(kind='static'), self.kin))
        self.assertLess(abs(s['frontend']['correction_roll_pitch_deg'][1] - 2.), .5)
        c = Condition(seed=17, duration_s=1.7, slope_deg=(0., 2.), joint_noise_rad=.01)
        s, _ = self.roll(np.zeros(22), c, mode='reflex', frontend=Frontend(Settings(kind='static'), self.kin))
        self.assertGreater(s['frontend']['correction_roll_pitch_deg'][1], 1.5)   # the slope is misattributed

    def test_kinematic_trunk_rate_matches_truth_for_rigid_planted_feet(self):
        # Pure kinematics: with the root fixed, the rate implied for each sole is exact.
        import mujoco
        m = self.sim.m; d = mujoco.MjData(m)
        q = HOME + .05; qd = np.linspace(-.3, .3, 14)
        d.qpos[:] = m.qpos0; d.qpos[3:7] = (1, 0, 0, 0); d.qpos[self.sim.qadr] = q
        d.qvel[:] = 0; d.qvel[self.sim.vadr] = qd
        mujoco.mj_forward(m, d)
        w = []
        for b in self.sim.foot_bodies:
            v = np.zeros(6); mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b, v, 0)
            w.append(-v[:3])
        np.testing.assert_allclose(self.kin.trunk_velocity(q, qd), np.mean(w, axis=0), atol=1e-10)

    def test_learn02e_reproduces_published_02d_training_outcomes(self):
        from movement_lab.learn02e import conditions
        from movement_lab.control02d import Design
        from movement_lab.sim02e import rollout
        z = np.load('results/02d/pilot/history-s72/generation-001.npz')
        cs = conditions(72, 0, True, 0.)
        for i, j in ((3, 0), (7, 5)):
            s, _ = rollout(self.sim, self.theta, z['candidates'][i], cs[j], Design(), basis=self.basis)
            self.assertEqual([s['fitness'], float(s['recovered']), float(s['survived']), s['residual_rms_rad']], z['outcomes'][i, j].tolist())


class EvidenceTests(unittest.TestCase):
    def test_phase0_gate_and_exact_reproduction_record(self):
        gate = json.loads(Path('results/02e/phase0/gate.json').read_text())
        self.assertTrue(gate['passed'])
        res = json.loads(Path('results/02e/phase0/resimulation-summary.json').read_text())
        self.assertEqual((res['rows'], res['exact_historical']), (1044, 1044))
        self.assertEqual(res['alternate']['outcomes_equal'], 1044)
        self.assertEqual(res['alternate']['divergence_before_any_such_event'], [])
        cmp_ = json.loads(Path('results/02e/phase0/model-comparison.json').read_text())
        self.assertTrue(cmp_['differences_only_in_hull_fields'])
        self.assertEqual(cmp_['model_a']['fields_digest_excluding_hull'], cmp_['model_b']['fields_digest_excluding_hull'])
        self.assertEqual(json.loads(Path('results/02e/learning/reproduce.json').read_text())['passed'], True)

    def test_replay_selection_is_declared_and_only_raw_records_stay_local(self):
        from movement_lab.replay_select02e import TRIALS, PRESETS
        from movement_lab.validate02e import local_only
        keys = [t[0] for t in TRIALS]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertTrue(all(len(p[1]) == 3 and set(p[1]) <= set(keys) for p in PRESETS))
        self.assertTrue(local_only(Path('results/02e/records/x.npz')))
        self.assertTrue(local_only(Path('results/02e/phase0/linux-compile/model.mjb')))
        self.assertFalse(local_only(Path('results/02e/records/x.json')))
        self.assertFalse(local_only(Path('results/02e/runs/history-rand-s1/generation-001.npz')))


# ----------------------------------------------------------------------------- bounded worker pools
def _pool_probe(i):
    import ctypes, os, signal
    sig = ctypes.c_int()
    ctypes.CDLL('libc.so.6').prctl(2, ctypes.byref(sig))          # PR_GET_PDEATHSIG
    return i, os.getpid(), sig.value, signal.getsignal(signal.SIGTERM) == signal.SIG_DFL, signal.getsignal(signal.SIGINT) == signal.SIG_IGN


def _pool_sleep(i):
    import time
    time.sleep(.3)
    return i


def _pool_fail(i):
    if i == 3:
        raise ValueError('task 3')
    return i


class WorkerPoolTests(unittest.TestCase):
    """pool02e: operational safety only; it must not change what a task computes or the result order."""

    def test_results_in_order_and_worker_lifetime_bounded(self):
        import signal
        from movement_lab.pool02e import RecyclingExecutor
        with RecyclingExecutor(2, tasks_per_worker=3) as pool:
            out = list(pool.map(_pool_probe, range(13), chunksize=2))
            self.assertEqual(pool.batches, 3)                     # ceil(13 / (2 x 3))
        self.assertEqual([o[0] for o in out], list(range(13)))
        pids = [o[1] for o in out]
        self.assertTrue(set(pids[:6]).isdisjoint(pids[6:12]) and set(pids[6:12]).isdisjoint(pids[12:]))
        self.assertTrue(all(o[2] == signal.SIGTERM and o[3] and o[4] for o in out))   # die with driver; plain SIGTERM

    def test_guard_breach_stops_and_leaves_no_workers(self):
        import os
        from movement_lab.pool02e import RecyclingExecutor, Guard, MemoryGuardError, descendants
        for guard in (Guard(max_worker_mb=1, poll_s=.05), Guard(max_total_mb=1, poll_s=.05)):
            with self.assertRaises(MemoryGuardError):
                with RecyclingExecutor(2, tasks_per_worker=4, guard=guard) as pool:
                    list(pool.map(_pool_sleep, range(8)))
            self.assertEqual(descendants(os.getpid()), [])

    def test_task_exception_propagates_and_leaves_no_workers(self):
        import os
        from movement_lab.pool02e import RecyclingExecutor, descendants
        with self.assertRaises(ValueError):
            with RecyclingExecutor(2, tasks_per_worker=2) as pool:
                list(pool.map(_pool_fail, range(8)))
        self.assertEqual(descendants(os.getpid()), [])

    def test_worker_cap(self):
        from movement_lab.pool02e import RecyclingExecutor, MAX_WORKERS
        with self.assertRaises(ValueError):
            RecyclingExecutor(MAX_WORKERS + 1)


if __name__ == '__main__':
    unittest.main()
