"""Scientific validity and reproducibility checks for 02C."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

from movement_lab.sim02c import Design, residual, audit, rollout, BASE
from movement_lab.sim02b import Condition, Sim, DEFAULT_MODEL
from movement_lab.learn02c import conditions, update_distribution

PHYSICS = (DEFAULT_MODEL / 'model.mjb').exists() and importlib.util.find_spec('bam') is not None


def synthetic_record():
    n = 1200
    q = np.zeros((n, 21)); q[:, 2] = .118
    return dict(time_s=np.arange(1, n + 1) * .005, qpos=q, qvel=np.zeros((n, 20)),
                nonfoot=np.zeros(n), penetration_m=np.zeros(n), tilt_deg=np.zeros(n),
                foot_found=np.ones((n, 2), bool), foot_pos=np.zeros((n, 2, 3)),
                foot_force_n=np.full((n, 2), 3.6), clearance_m=np.zeros((n, 2)))


class MeasurementTests(unittest.TestCase):
    def test_quiet_flight_or_vertical_motion_is_not_a_recovered_stance(self):
        from movement_lab.eval02c import audit_record
        c = Condition(duration_s=6, push_time_s=2)
        r = synthetic_record()
        self.assertTrue(audit_record(r, c, .118)['recovered_200hz'])
        r['qvel'][-1, 2] = .1
        self.assertFalse(audit_record(r, c, .118)['recovered_200hz'])
        r['qvel'][:] = 0
        r['foot_found'][-1] = False
        self.assertFalse(audit_record(r, c, .118)['recovered_200hz'])
        r['foot_found'][:] = True
        r['foot_force_n'][-1] = 0
        self.assertFalse(audit_record(r, c, .118)['recovered_200hz'])

    def test_measured_joint_limit_exploit_is_rejected(self):
        from types import SimpleNamespace
        from movement_lab.eval02c import audit_record
        fake = SimpleNamespace(qadr=np.array([7]), lo=np.array([-.98]), hi=np.array([.98]))
        r = synthetic_record(); c = Condition(duration_s=6, push_time_s=2)
        self.assertTrue(audit_record(r, c, .118, sim=fake)['recovered_200hz'])
        r['qpos'][500, 7] = 1.04
        self.assertFalse(audit_record(r, c, .118, sim=fake)['physical_valid'])
        self.assertFalse(audit_record(r, c, .118, sim=fake)['recovered_200hz'])

    def test_a_real_relocation_requires_air_clearance_and_sustained_replant(self):
        r = synthetic_record()
        r['foot_found'][500:512, 0] = False
        r['clearance_m'][500:512, 0] = .004
        r['foot_pos'][512:, 0, 0] = .01
        c = Condition(duration_s=6, push_time_s=2)
        s = audit(r, c, .118)
        self.assertTrue(s['recovery_with_step'])
        r['clearance_m'][:] = .001
        self.assertFalse(audit(r, c, .118)['recovery_with_step'])
        r['clearance_m'][500:512, 0] = .004
        r['foot_force_n'][512:532, 0] = 0
        self.assertFalse(audit(r, c, .118)['recovery_with_step'])

    def test_sliding_and_contact_flicker_are_not_steps(self):
        r = synthetic_record(); c = Condition(duration_s=6, push_time_s=2)
        r['foot_pos'][:, :, 0] = np.arange(1200)[:, None] * .00001
        self.assertFalse(audit(r, c, .118)['recovery_with_step'])
        r['foot_found'][500:502, 0] = False
        r['clearance_m'][500:502, 0] = .005
        r['foot_pos'][502:, 0, 0] += .01
        self.assertFalse(audit(r, c, .118)['recovery_with_step'])

    def test_fall_replant_low_height_penetration_and_motion_cannot_win(self):
        c = Condition(duration_s=6, push_time_s=2)
        for field, idx, value in [('nonfoot', 500, 1), ('tilt_deg', 500, 50),
                                  ('penetration_m', 500, .004), ('qpos', (500, 2), .04),
                                  ('qvel', (-1, 0), .1), ('qvel', (-1, 4), 1.)]:
            r = synthetic_record(); r[field][idx] = value
            with self.subTest(field=field, idx=idx):
                self.assertFalse(audit(r, c, .118)['recovered_200hz'])
        r = synthetic_record(); r['tilt_deg'][350] = 10
        self.assertFalse(audit(r, c, .118)['settled_200hz'])

    def test_untriggered_and_finished_options_are_exactly_zero(self):
        d = Design(); v = np.ones(56)
        for t in (None, -.1, 0, .8, 4):
            np.testing.assert_array_equal(residual(v, t, d), np.zeros(14))
        for t in np.linspace(0, .8, 100):
            self.assertLessEqual(abs(residual(v, t, d)).max(), d.bound)

    def test_all_seed_namespaces_are_disjoint_and_bins_are_covered(self):
        from movement_lab.eval02c import grid, shifts
        train = [c for seed in range(4) for g in range(30) for c in conditions(seed, g, 4, .28, .36, True)]
        ts = {c.seed for c in train}
        self.assertEqual(len(ts), len(train))
        self.assertFalse(ts & {c.seed for _, c in grid() + shifts()})
        for j, c in enumerate(conditions(1, 2, 4, .28, .36, True)):
            self.assertGreaterEqual(c.push_dv[0], .28 + j * .02)
            self.assertLess(c.push_dv[0], .28 + (j + 1) * .02)

    def test_cem_responds_to_experience_not_iteration_number(self):
        mean = np.zeros(2); std = np.ones(2)
        c = np.array([[2., 1.], [-2., -1.], [0., 0.]])
        a, _ = update_distribution(mean, std, c, np.array([3, 1, 0]), 1, .04)
        b, _ = update_distribution(mean, std, c, np.array([1, 3, 0]), 1, .04)
        np.testing.assert_array_equal(a, -b)


class CheckpointTests(unittest.TestCase):
    def test_checkpoint_discovery_only_lists_promoted_generations_in_numeric_order(self):
        from movement_lab.learn02c import completed_checkpoints
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name in ('generation-001.npz', 'generation-999.npz', 'generation-1000.npz',
                         'generation-1001.tmp.npz', 'generation-latest.npz', 'generation-001.backup.npz'):
                (root / name).touch()
            (root / 'generation-1002.npz').mkdir()
            self.assertEqual([p.name for p in completed_checkpoints(root)],
                             ['generation-001.npz', 'generation-999.npz', 'generation-1000.npz'])

    def test_resume_ignores_interrupted_checkpoint_write(self):
        from movement_lab import learn02c

        # Keep the real search, RNG and disk I/O; replace only expensive physics.
        def outcomes(_worker, tasks, **kwargs):
            for vec, cond, _design in tasks:
                yield dict(fitness=float(-np.mean((vec - .1) ** 2) + cond.seed % 17 * .001),
                           survived=True, recovered=True, failure_cause=None, simulated_s=4.)

        save = np.savez_compressed
        with tempfile.TemporaryDirectory() as td, \
                patch.object(learn02c, 'ProcessPoolExecutor') as pool, \
                patch.object(learn02c.os, 'nice', create=True):
            physics = pool.return_value.__enter__.return_value.map
            physics.side_effect = outcomes
            root = Path(td)
            model = root / 'model'; model.mkdir()
            (model / 'model.mjb').write_bytes(b'model hash fixture; physics is stubbed')
            base = root / 'base.npz'; base.write_bytes(b'baseline hash fixture')
            common = ['--iterations', '3', '--population', '4', '--elite', '2',
                      '--conditions', '2', '--workers', '1', '--model', str(model), '--base', str(base)]
            uninterrupted = root / 'uninterrupted'
            learn02c.main(['--out', str(uninterrupted), *common])

            for interrupted_generation in (1, 2):
                with self.subTest(interrupted_generation=interrupted_generation):
                    run = root / f'interrupted-{interrupted_generation}'
                    staging = run / f'generation-{interrupted_generation:03d}.tmp.npz'

                    def interrupt_write(path, *args, **kwargs):
                        save(path, *args, **kwargs)
                        if Path(path) == staging:
                            data = staging.read_bytes()
                            staging.write_bytes(data[:len(data) // 2])
                            raise OSError('simulated interrupted checkpoint write')

                    with patch.object(learn02c.np, 'savez_compressed', side_effect=interrupt_write):
                        with self.assertRaisesRegex(OSError, 'simulated interrupted checkpoint write'):
                            learn02c.main(['--out', str(run), *common])
                    self.assertTrue(staging.exists())
                    self.assertFalse((run / f'generation-{interrupted_generation:03d}.npz').exists())
                    self.assertFalse((run / 'final.json').exists())

                    committed = {p.name: p.read_bytes() for p in learn02c.completed_checkpoints(run)}
                    physics.reset_mock()
                    learn02c.main(['--out', str(run), '--resume', *common])
                    self.assertEqual(physics.call_count, 4 - interrupted_generation)
                    for name, content in committed.items():
                        self.assertEqual((run / name).read_bytes(), content)
                    self.assertFalse(staging.exists())
                    for generation in (1, 2, 3):
                        name = f'generation-{generation:03d}.npz'
                        with np.load(uninterrupted / name) as expected, np.load(run / name) as actual:
                            for key in ('candidates', 'scores', 'outcomes', 'mean', 'std', 'best', 'conditions'):
                                np.testing.assert_array_equal(actual[key], expected[key], err_msg=f'{name}: {key}')
                            left, right = json.loads(str(actual['state'])), json.loads(str(expected['state']))
                            for key in ('generation', 'trials', 'rng'):
                                self.assertEqual(left[key], right[key], f'{name}: {key}')
                    with np.load(uninterrupted / 'final.npz') as expected, np.load(run / 'final.npz') as actual:
                        for key in ('best', 'mean'):
                            np.testing.assert_array_equal(actual[key], expected[key])
                    self.assertEqual(json.loads((run / 'final.json').read_text())['trials'], 24)


@unittest.skipUnless(PHYSICS, 'local compiled model / BAM unavailable')
class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Sim()
        cls.theta = np.load(BASE)['final']

    def test_zero_option_exactly_reproduces_02b_and_recording_is_passive(self):
        from movement_lab import sim02b
        c = Condition(seed=6, duration_s=3, push_time_s=1.5, push_dv=(.32, 0))
        s, r = rollout(self.sim, self.theta, np.zeros(56), c, record=True, terminate=False)
        old, rold = sim02b.rollout(self.sim, self.theta, c, record=True, terminate=False)
        np.testing.assert_array_equal(r['qpos'], rold['qpos'])
        np.testing.assert_array_equal(r['qvel'], rold['qvel'])
        self.assertEqual(s['failure_time_s'], old['fall_time_s'])
        plain, _ = rollout(self.sim, self.theta, np.zeros(56), c, terminate=False)
        self.assertEqual(plain['final_root_xy'], s['final_root_xy'])
        self.assertEqual(plain['fitness'], s['fitness'])

    def test_sequences_cannot_anticipate_the_event(self):
        c = Condition(seed=7, duration_s=2, push_time_s=1.5, push_dv=(.3, 0))
        _, a = rollout(self.sim, self.theta, np.zeros(56), c, record=True, terminate=False)
        _, b = rollout(self.sim, self.theta, np.full(56, .4), c, record=True, terminate=False)
        pre = a['time_s'] <= 1.5 + 1e-9
        np.testing.assert_array_equal(a['qpos'][pre], b['qpos'][pre])
        # First command is zero at the cue, then the 20 ms command delay:
        # the initial 40 ms of displacement is causally shared with the control.
        delayed = a['time_s'] <= 1.54 + 1e-9
        np.testing.assert_array_equal(a['qpos'][delayed], b['qpos'][delayed])
        self.assertGreater(abs(a['qpos'][-1] - b['qpos'][-1]).max(), .01)

    def test_material_contact_point_jacobian_preserves_root_translation(self):
        s = self.sim
        s.reset(Condition(), np.random.default_rng(1))
        s.mujoco.mj_comPos(s.m, s.d)
        jac = np.zeros((3, s.m.nv))
        v = np.zeros(s.m.nv); v[0] = .2
        for body in s.foot_bodies:
            point = s.d.xpos[body].copy(); point[2] = 0
            s.mujoco.mj_jac(s.m, s.d, jac, None, point, body)
            np.testing.assert_allclose(jac @ v, [.2, 0, 0], atol=1e-12)

    def test_resume_restores_distribution_rng_and_selected_winner(self):
        from movement_lab.learn02c import main
        with tempfile.TemporaryDirectory() as td:
            a, b = Path(td, 'a'), Path(td, 'b')
            common = ['--iterations', '2', '--population', '3', '--elite', '1', '--conditions', '1', '--workers', '1']
            main(['--out', str(a), *common])
            main(['--out', str(b), '--budget', '.001', *common])
            self.assertEqual(len(list(b.glob('generation-*.npz'))), 1)
            main(['--out', str(b), '--resume', *common])
            for gen in (1, 2):
                x, y = np.load(a / f'generation-{gen:03d}.npz'), np.load(b / f'generation-{gen:03d}.npz')
                for key in ('candidates', 'scores', 'outcomes', 'mean', 'std', 'best'):
                    np.testing.assert_array_equal(x[key], y[key])


if __name__ == '__main__':
    unittest.main()
