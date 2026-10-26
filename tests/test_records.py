"""Integration checks against saved real rollouts (skip before evaluation)."""
import json
from pathlib import Path
import unittest
import numpy as np
from movement_lab.measure import validate


class RecordedTrials(unittest.TestCase):
    def test_saved_rollouts(self):
        paths = sorted(Path('results').glob('*balance/trial.npz'))
        if not paths:
            self.skipTest('run checkpoint evaluation first')
        for path in paths:
            with self.subTest(trial=str(path)):
                a = dict(np.load(path, allow_pickle=False))
                m = json.loads((path.parent/'metadata.json').read_text())
                validate(a)
                self.assertEqual(a['joint_position_rad'].shape[1], 14)
                self.assertEqual(a['foot_found'].shape[1], 2)
                np.testing.assert_allclose(a['qpos'][:,:7], a['body_pose_w'][:,0,:], atol=2e-6)
                np.testing.assert_array_equal(a['qpos'][:,7:], a['joint_position_rad'])
                np.testing.assert_array_equal(a['qvel'][:,6:], a['joint_velocity_rad_s'])
                # This task has unit joint transmissions, independently exposing
                # the same actuator contribution in both coordinate spaces.
                np.testing.assert_array_equal(a['actuator_force'], a['joint_force_nm'])
                np.testing.assert_array_equal(a['command'], np.zeros_like(a['command']))
                np.testing.assert_allclose(np.diff(a['time_s']), m['physics_dt'], atol=2e-7)
                self.assertEqual(len(a['time_s']), 4*(int(a['control_index'][-1])+1))
                np.testing.assert_array_equal(a['control_index'], np.repeat(np.arange(len(a['time_s'])//4),4))
                np.testing.assert_array_equal(a['action'][::4], a['action'][3::4])
                self.assertTrue((a['foot_found'] > 0).any())
                absent = a['foot_found'] == 0
                np.testing.assert_array_equal(a['foot_force_w_n'][absent], 0)

    def test_repeatability(self):
        paths = [Path('results')/name/'trial.npz' for name in ('99-balance','99-repeat')]
        if not all(p.exists() for p in paths):
            self.skipTest('run repeatability check first')
        a,b = [dict(np.load(p, allow_pickle=False)) for p in paths]
        for key in a:
            np.testing.assert_array_equal(a[key],b[key],err_msg=key)
