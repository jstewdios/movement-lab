import unittest
import numpy as np
from movement_lab.discover01a import envelope, SEQUENCES
from movement_lab.discovery_metrics import longest


class DiscoveryTests(unittest.TestCase):
    def test_pulse_boundaries_and_slew(self):
        t = np.linspace(0, 3, 3001)
        y = np.array([envelope(x) for x in t])
        self.assertTrue(((y >= 0) & (y <= 1)).all())
        self.assertTrue((y[t >= 1.8] == 0).all())
        self.assertLess(np.max(np.abs(np.diff(y))) / .001 * .25, .66)
        for boundary in [.2, .8, 1.2, 1.8]:
            self.assertLess(abs(envelope(boundary+1e-6)-envelope(boundary-1e-6)), 1e-5)
        self.assertEqual(SEQUENCES['zero'], {})
        self.assertLessEqual(max(abs(v) for seq in SEQUENCES.values() for v in seq.values()), .25)

    def test_airtime_requires_consecutive_samples(self):
        self.assertAlmostEqual(longest([True, True, False, True], [.1]*4), .2)

class ClassificationTests(unittest.TestCase):
    def make_trial(self):
        n = 600
        pose = np.zeros((n, 1, 7)); pose[:, :, 2] = .15; pose[:, :, 3] = 1
        a = dict(time_s=np.arange(1, n+1)*.005, body_pose_w=pose,
            com_position_w_m=np.tile([0,0,.15], (n,1)), com_velocity_w_m_s=np.zeros((n,3)),
            body_velocity_w=np.zeros((n,1,6)), joint_position_rad=np.zeros((n,2)),
            joint_target_rad=np.zeros((n,2)), action=np.zeros((n,2)), qvel=np.zeros((n,8)),
            joint_force_nm=np.zeros((n,2)), joint_velocity_rad_s=np.zeros((n,2)),
            foot_found=np.ones((n,2)), foot_force_w_n=np.zeros((n,2,3)),
            native_ground_contacts=np.ones(n), native_nonfoot_contacts=np.zeros(n),
            native_min_contact_distance_m=np.zeros(n))
        initial=dict(body_pose_w=pose[0].copy(), com_position_w_m=np.array([0,0,.15]))
        return a, initial, np.array([[-1,1],[-1,1]])

    def test_body_support_is_not_airborne_or_stable(self):
        from movement_lab.discovery_metrics import analyze
        a, i, limits = self.make_trial()
        a['foot_found'][:] = 0
        a['native_nonfoot_contacts'][:] = 1
        s = analyze(a,i,limits)
        self.assertEqual(s['feet_off_total_s'], 3)
        self.assertEqual(s['all_ground_off_total_s'], 0)
        self.assertFalse(s['final_stability_gate'])

    def test_quiet_fall_is_not_success(self):
        from movement_lab.discovery_metrics import analyze
        a, i, limits = self.make_trial()
        self.assertTrue(analyze(a,i,limits)['final_stability_gate'])
        a['body_pose_w'][:,0,3] = np.cos(np.pi/4)
        a['body_pose_w'][:,0,4] = np.sin(np.pi/4)
        s = analyze(a,i,limits)
        self.assertFalse(s['final_stability_gate'])
        self.assertIn('fall',s['classification'])
        a['native_min_contact_distance_m'][:] = -.02
        self.assertFalse(analyze(a,i,limits)['valid'])
        a['qvel'][0,0] = np.nan
        self.assertFalse(analyze(a,i,limits)['valid'])
