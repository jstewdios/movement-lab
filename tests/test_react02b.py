"""02B tests. Pure-function tests always run; physics tests need the local
compiled model (``python -m movement_lab.model02b --out results/02b/model``)
plus upstream ``mujoco``/``bam`` and are skipped, explicitly, without them."""
import importlib.util
import math
import unittest

import numpy as np

from movement_lab import sim02b
from movement_lab.sim02b import ACTION_BOUND, Condition, PARAM_SHAPE, action, fitness, recovery

HAVE_PHYSICS = (sim02b.DEFAULT_MODEL / 'model.mjb').exists() and all(
    importlib.util.find_spec(m) is not None for m in ('mujoco', 'bam'))


class ReflexTests(unittest.TestCase):
    def test_zero_reflex_is_home(self):
        self.assertTrue(np.array_equal(action(np.zeros(PARAM_SHAPE), np.ones(5)), np.zeros(14)))

    def test_action_is_bounded_and_linear_inside_bound(self):
        th = np.zeros(PARAM_SHAPE)
        th[4, 0] = 0.1        # left ankle on gravity_x
        th[4, 5] = 0.02       # posture bias
        a = action(th, np.array([1.0, 0, 0, 0, 0]))
        self.assertAlmostEqual(a[4], 0.12)
        self.assertEqual(action(th, np.array([100.0, 0, 0, 0, 0]))[4], ACTION_BOUND)
        self.assertEqual(action(th, np.array([-100.0, 0, 0, 0, 0]))[4], -ACTION_BOUND)

    def test_imu_misalignment_matrix(self):
        R = sim02b.rpy_matrix((0, 5, 0))
        self.assertTrue(np.allclose(R @ R.T, np.eye(3)))
        self.assertAlmostEqual(math.degrees(math.acos(R[2, 2])), 5.0, places=6)


class ObjectiveTests(unittest.TestCase):
    def test_survival_dominates_and_falls_score_lower(self):
        full = fitness(200, 200, 0.0, 0.0, 0.0)
        fall = fitness(50, 200, 0.0, 0.0, 0.0)
        self.assertEqual(full, 1.0)
        self.assertLess(fall, full)
        # Maximal tilt shaping on every alive step cannot make falling better.
        self.assertGreater(fitness(200, 200, 200.0, 0.0, 0.0), fall)

    def test_penalties_reduce_fitness(self):
        self.assertLess(fitness(200, 200, 0.0, 10.0, 10.0), fitness(200, 200, 0.0, 0.0, 0.0))


class LearnerTests(unittest.TestCase):
    def test_ars_update_climbs_a_linear_objective(self):
        from movement_lab.learn02b import ars_update
        rng = np.random.default_rng(0)
        g = rng.normal(size=PARAM_SHAPE)
        theta = np.zeros(PARAM_SHAPE)
        for _ in range(20):
            delta = rng.standard_normal((16, *PARAM_SHAPE))
            fp = np.array([(g * (theta + 0.03 * d)).sum() for d in delta])
            fm = np.array([(g * (theta - 0.03 * d)).sum() for d in delta])
            theta, step, _ = ars_update(theta, delta, fp, fm, 8, 0.02)
            self.assertGreater((g * step).sum(), 0)
        cos = (g * theta).sum() / np.linalg.norm(g) / np.linalg.norm(theta)
        self.assertGreater(cos, 0.5)

    def test_ars_update_ignores_unranked_directions(self):
        from movement_lab.learn02b import ars_update
        delta = np.zeros((4, *PARAM_SHAPE))
        for i in range(4):
            delta[i, 0, i] = 1.0
        fp = np.array([1.0, 0.0, 0.0, 0.5])
        fm = np.array([0.0, 0.0, 0.0, 0.0])
        _, step, _ = ars_update(np.zeros(PARAM_SHAPE), delta, fp, fm, 2, 0.1)
        self.assertGreater(step[0, 0], 0)
        self.assertGreater(step[0, 3], 0)
        self.assertEqual(step[0, 1], 0)
        self.assertEqual(step[0, 2], 0)

    def test_probe_schedule_is_dense_early(self):
        from movement_lab.learn02b import probe_due
        self.assertTrue(all(probe_due(i) for i in range(7)))
        self.assertFalse(probe_due(7))
        self.assertTrue(probe_due(60))


class RevisionTests(unittest.TestCase):
    def test_zero_residual_reproduces_the_warm_start_reflex(self):
        from movement_lab.revise02b import Residual
        rng = np.random.default_rng(3)
        th = rng.normal(0, 0.05, PARAM_SHAPE)
        pol = Residual(Residual.initial(th, rng))
        feat = rng.normal(size=5)
        a = pol(feat, rng.normal(0, 0.1, 14), rng.normal(0, 1, 14))
        self.assertTrue(np.allclose(a, th[:, :5] @ feat + th[:, 5]))

    def test_hard_revision_seeds_are_disjoint_and_beyond_the_easy_range(self):
        from movement_lab import learn02b, revise02b
        hard = [c for s in (1, 2, 3) for it in range(40) for c in revise02b.hard_conditions(s, it, 8)]
        main = {c.seed for s in range(10) for it in range(200) for c in learn02b.train_conditions(s, it, 4, 'push')}
        self.assertFalse({c.seed for c in hard} & main)
        self.assertLess(max(c.seed for c in hard), learn02b.PROBE_SEED_BASE)
        mags = [math.hypot(*c.push_dv) for c in hard]
        self.assertGreaterEqual(min(mags), revise02b.HARD_RANGE[0])
        self.assertLessEqual(max(mags), revise02b.HARD_RANGE[1])


class CapturePointTests(unittest.TestCase):
    def test_hull_and_ray_exit_on_a_rectangle(self):
        from movement_lab.physics02b import convex_hull, inside, ray_exit
        pts = np.array([[-0.02, -0.06], [0.03, -0.06], [0.03, 0.06], [-0.02, 0.06], [0.0, 0.0], [0.01, 0.02]])
        hull = convex_hull(pts)
        self.assertEqual(len(hull), 4)
        o = np.zeros(2)
        self.assertTrue(inside(hull, o))
        self.assertFalse(inside(hull, np.array([0.05, 0.0])))
        self.assertAlmostEqual(ray_exit(hull, o, np.array([1.0, 0.0])), 0.03)
        self.assertAlmostEqual(ray_exit(hull, o, np.array([-1.0, 0.0])), 0.02)
        self.assertAlmostEqual(ray_exit(hull, o, np.array([0.0, 1.0])), 0.06)

    @unittest.skipUnless(HAVE_PHYSICS, 'compiled 02B model unavailable')
    def test_home_stance_bound_is_weakest_backward(self):
        from movement_lab.physics02b import capture_bounds
        sim = sim02b.Sim()
        sim.reset(Condition(), np.random.default_rng(0))
        b = capture_bounds(sim, sim.d, [0.0, math.pi / 2, math.pi])
        self.assertTrue(b['com_inside'])
        fwd, left, back = b['bounds']
        self.assertLess(back, fwd)
        self.assertLess(fwd, left)
        self.assertAlmostEqual(back, 0.149, delta=0.01)


class RecoveryTests(unittest.TestCase):
    def test_quiet_stance_recovers_and_settles(self):
        n = 300
        tilt = [1.0] * 100 + [8.0] * 10 + [1.2] * (n - 110)
        r = recovery(tilt, [0.01] * n, 100, True)
        self.assertTrue(r['recovered'])
        self.assertAlmostEqual(r['settle_time_s'], 10 * sim02b.CONTROL_DT)

    def test_fall_or_sustained_lean_is_not_recovery(self):
        n = 300
        self.assertFalse(recovery([1.0] * n, [0.0] * n, 100, False)['recovered'])
        lean = recovery([1.0] * 100 + [7.0] * 200, [0.0] * n, 100, True)
        self.assertFalse(lean['recovered'])
        self.assertIsNone(lean['settle_time_s'])

    def test_oscillation_is_not_recovery(self):
        n = 300
        self.assertFalse(recovery([1.0] * n, [0.1] * 250 + [2.0] * 50, 100, True)['recovered'])


class SeedSeparationTests(unittest.TestCase):
    def test_training_probe_and_evaluation_seeds_are_disjoint(self):
        from movement_lab import eval02b, learn02b
        train = {c.seed for s in range(10) for it in range(200) for c in learn02b.train_conditions(s, it, 4, 'push')}
        train |= {c.seed for s in range(10) for c in learn02b.train_conditions(s, 200, 16, 'push')}
        probe = {c.seed for c in learn02b.probe_conditions()}
        ev = {c.seed for b in eval02b.BATTERIES.values() for _, c in b()}
        self.assertFalse(train & probe)
        self.assertFalse(train & ev)
        self.assertFalse(probe & ev)
        self.assertLess(max(train), learn02b.PROBE_SEED_BASE)
        self.assertGreaterEqual(min(ev), eval02b.EVAL_SEED_BASE)

    def test_push_and_none_arms_share_seeds_and_initial_noise(self):
        from movement_lab import learn02b
        a = learn02b.train_conditions(3, 7, 4, 'push')
        b = learn02b.train_conditions(3, 7, 4, 'none')
        self.assertEqual([c.seed for c in a], [c.seed for c in b])
        self.assertTrue(all(c.push_time_s is None and c.push_dv == (0.0, 0.0) for c in b))
        self.assertTrue(all(c.joint_noise_rad == learn02b.JOINT_NOISE for c in a + b))
        self.assertTrue(all(math.hypot(*c.push_dv) <= learn02b.PUSH_MAX for c in a))

    def test_evaluation_extends_beyond_training(self):
        from movement_lab import eval02b, learn02b
        grid = eval02b.push_grid()
        self.assertGreater(max(math.hypot(*c.push_dv) for _, c in grid), learn02b.PUSH_MAX)
        self.assertTrue(all(c.push_time_s > learn02b.PUSH_WINDOW[1] for _, c in grid))
        shift_sets = {s for s, _ in eval02b.shifts()}
        self.assertEqual(len(shift_sets), len(eval02b.SHIFTS))
        # Every shift reuses the nominal seeds/pushes so comparisons are paired.
        by = {}
        for s, c in eval02b.shifts():
            by.setdefault(s, []).append((c.seed, c.push_dv))
        self.assertTrue(all(v == by['shift:nominal'] for v in by.values()))

    def test_mirror_is_an_involution_and_matches_upstream_home(self):
        from movement_lab.eval02b import mirror, symmetry_fraction, ablations
        th = np.random.default_rng(0).normal(size=PARAM_SHAPE)
        self.assertTrue(np.allclose(mirror(mirror(th)), th))
        # HOME itself must be mirror-symmetric under the upstream joint convention.
        home = np.zeros(PARAM_SHAPE)
        home[:, 5] = sim02b.HOME
        self.assertTrue(np.allclose(mirror(home), home, atol=1e-12))
        sym = 0.5 * (th + mirror(th))
        self.assertAlmostEqual(symmetry_fraction(sym), 1.0)
        v = ablations(th)
        self.assertTrue(np.array_equal(v['posture_only'][:, :5], np.zeros((14, 5))))
        self.assertTrue(np.array_equal(v['posture_only'][:, 5], th[:, 5]))
        self.assertTrue(np.array_equal(v['neck_head_only'][[0, 4, 9, 13]], np.zeros((4, 6))))

    def test_mirrored_reflex_mirrors_its_response(self):
        from movement_lab.eval02b import mirror, FEATURE_SIGN, JOINT_PERM, JOINT_SIGN
        th = np.random.default_rng(1).normal(0, 0.05, PARAM_SHAPE)
        feat = np.array([0.3, -0.2, 0.1, 0.4, -0.5])
        a = action(th, feat)
        a_m = action(mirror(th), feat * FEATURE_SIGN[:5])
        self.assertTrue(np.allclose(a_m, JOINT_SIGN * a[JOINT_PERM]))

    def test_published_gzip_results_round_trip(self):
        import json
        import tempfile
        from pathlib import Path
        from movement_lab.eval02b import publish_results, read_results
        with tempfile.TemporaryDirectory() as d:
            rows = [dict(controller='a', set='s', survived=True), dict(controller='b', set='s', survived=False)]
            (Path(d) / 'results.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            publish_results(d)
            first = (Path(d) / 'results.jsonl.gz').read_bytes()
            publish_results(d)
            self.assertEqual(first, (Path(d) / 'results.jsonl.gz').read_bytes())  # deterministic
            (Path(d) / 'results.jsonl').unlink()
            self.assertEqual(read_results(d), rows)
            self.assertIsNone(read_results(Path(d) / 'missing'))

    def test_manifest_classifies_every_ignored_path(self):
        import shutil
        from pathlib import Path
        from movement_lab.publication02b import ignored
        if shutil.which('git') is None:
            self.skipTest('git unavailable')
        paths = [Path('results/02b/model/model.mjb'), Path('results/02b/runs/x.log'),
                 Path('results/02b/eval/final/results.jsonl'), Path('results/02b/REPORT.md'),
                 Path('results/02b/eval/final/results.jsonl.gz')]
        # Regression: several paths at once (Windows text pipes once hid all but the last).
        self.assertEqual(ignored(paths), set(paths[:3]))

    def test_wilson_interval(self):
        from movement_lab.eval02b import wilson
        lo, hi = wilson(10, 10)
        self.assertGreater(lo, 0.69)
        self.assertAlmostEqual(hi, 1.0)
        lo, hi = wilson(0, 10)
        self.assertAlmostEqual(lo, 0.0)


@unittest.skipUnless(HAVE_PHYSICS, 'compiled 02B model or upstream mujoco/bam unavailable')
class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = sim02b.Sim()

    def test_clearance_reset_matches_01b(self):
        self.sim.reset(Condition(), np.random.default_rng(0))
        self.assertAlmostEqual(self.sim.z0, 0.1181823627, delta=1e-8)
        self.assertAlmostEqual(self.sim._lowest_point(self.sim.d), sim02b.CLEARANCE_M, delta=1e-9)

    def test_fixed_home_falls_forward_without_disturbance(self):
        s, _ = sim02b.rollout(self.sim, sim02b.zero_theta(), Condition(seed=1, duration_s=3.0))
        self.assertFalse(s['survived'])
        self.assertEqual(s['fall_cause'], 'tilt')
        self.assertGreater(s['final_root_xy'][0], 0.02)

    def test_rollouts_are_deterministic(self):
        th = np.random.default_rng(5).normal(0, 0.05, PARAM_SHAPE)
        c = Condition(seed=11, duration_s=2.0, joint_noise_rad=0.02, push_time_s=1.0, push_dv=(0.1, -0.05))
        a, _ = sim02b.rollout(self.sim, th, c)
        b, _ = sim02b.rollout(self.sim, th, c)
        self.assertEqual(a['fitness'], b['fitness'])
        self.assertEqual(a['final_root_xy'], b['final_root_xy'])

    def test_condition_changes_do_not_accumulate(self):
        base = Condition(seed=2, duration_s=1.0)
        ref, _ = sim02b.rollout(self.sim, sim02b.zero_theta(), base)
        for kw in (dict(vin=6.5), dict(foot_friction=0.5), dict(head_mass_scale=1.3)):
            sim02b.rollout(self.sim, sim02b.zero_theta(), Condition(seed=2, duration_s=1.0, **kw))
        again, _ = sim02b.rollout(self.sim, sim02b.zero_theta(), base)
        self.assertEqual(ref['fitness'], again['fitness'])
        for k, v in self.sim._pristine.items():
            if k not in ('dof_frictionloss', 'dof_damping', 'actuator_forcerange'):
                self.assertTrue(np.array_equal(getattr(self.sim.m, k), v), k)

    def test_command_delay_is_four_physics_steps(self):
        th = np.zeros(PARAM_SHAPE)
        th[7, 5] = 0.2  # constant head-yaw bias, commanded from the first control step
        _, rec = sim02b.rollout(self.sim, th, Condition(seed=0, duration_s=0.1, gyro_noise=0, gravity_noise=0), record=True)
        applied = rec['target_applied'][:, 7]
        self.assertTrue(np.allclose(rec['target_cmd'][:, 7], 0.2))
        # mjlab DelayBuffer backfills the first command, so even the first
        # substeps apply it; with constant commands applied == commanded.
        self.assertTrue(np.allclose(applied, 0.2))
        th2 = np.zeros(PARAM_SHAPE)
        th2[7, 2] = 1.0  # head yaw follows gyro_x: commands change every control step
        _, rec = sim02b.rollout(self.sim, th2, Condition(seed=0, duration_s=0.2, gyro_noise=0.05, gravity_noise=0), record=True)
        cmd, app = rec['target_cmd'][:, 7], rec['target_applied'][:, 7]
        self.assertTrue(np.allclose(app[8:], cmd[4:-4]))

    def test_push_changes_only_root_linear_velocity(self):
        c = Condition(seed=0, duration_s=0.2, gyro_noise=0, gravity_noise=0, push_time_s=0.1, push_dv=(0.2, 0.0))
        _, rec = sim02b.rollout(self.sim, sim02b.zero_theta(), c, record=True)
        _, ref = sim02b.rollout(self.sim, sim02b.zero_theta(), Condition(seed=0, duration_s=0.2, gyro_noise=0, gravity_noise=0), record=True)
        k = int(round(0.1 / sim02b.DT))
        self.assertTrue(np.array_equal(rec['qpos'][:k], ref['qpos'][:k]))
        self.assertGreater(rec['qvel'][k, 0] - ref['qvel'][k, 0], 0.1)


if __name__ == '__main__':
    unittest.main()
