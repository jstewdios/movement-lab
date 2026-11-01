import unittest
import numpy as np
from movement_lab.explore02a import Explorer, ACTIONS, distance, measure

class ExplorationTests(unittest.TestCase):
    def record(self, action, eligible=True):
        return dict(id=str(action), action=action, eligible=eligible, descriptor=[float(action[0]), float(action[1])] if eligible else None)

    def test_invalid_retained_and_duplicate_outcome_not_discovered(self):
        e = Explorer(42, True)
        self.assertFalse(e.observe(self.record(ACTIONS[0], False))['discovery'])
        self.assertTrue(e.observe(self.record(ACTIONS[1]))['discovery'])
        r = self.record(ACTIONS[2]); r['descriptor'] = e.members[0]['descriptor']
        self.assertFalse(e.observe(r)['discovery'])
        self.assertEqual(len(e.history), 3)

    def test_reproducible_and_history_influences_proposal(self):
        a, b = Explorer(42, True), Explorer(42, True)
        first, _ = a.propose(); self.assertEqual(first, b.propose()[0])
        a.observe(self.record(first)); b.observe(self.record(first, False))
        self.assertNotEqual(a.propose()[0], b.propose()[0])

    def test_control_ignores_outcome(self):
        a, b = Explorer(3, False), Explorer(3, False)
        for _ in range(10):
            x, _ = a.propose(); self.assertEqual(x, b.propose()[0])
            a.observe(self.record(x)); b.observe(self.record(x, False))

    def test_missing_baseline_hard_failure(self):
        with self.assertRaisesRegex(AssertionError, 'incomplete baseline'):
            measure({'time_s': np.arange(5)}, {'time_s': np.arange(5)}, {})

class PhysicalDescriptorTests(unittest.TestCase):
    def arrays(self):
        n = 2000
        a = dict(time_s=np.arange(1,n+1)*.005,
                 body_pose_w=np.zeros((n,1,7)), joint_position_rad=np.zeros((n,14)),
                 com_position_w_m=np.zeros((n,3)), com_velocity_w_m_s=np.zeros((n,3)),
                 body_velocity_w=np.zeros((n,1,6)), foot_found=np.ones((n,2)),
                 native_nonfoot_contacts=np.zeros(n), native_min_contact_distance_m=np.zeros(n))
        a['body_pose_w'][:,0,3] = 1
        return a

    def test_phase_noise_rejected_and_contact_failure_retained(self):
        c = self.arrays(); a = {k:v.copy() for k,v in c.items()}
        # Large alternating noise averages to zero; peaks do not earn novelty.
        a['joint_position_rad'][:,0] = np.tile([.02,-.02],1000)
        self.assertFalse(measure(a,c,{'stop_reasons':[]})['eligible'])
        a['joint_position_rad'][:,0] = .01
        self.assertTrue(measure(a,c,{'stop_reasons':[]})['eligible'])
        a['native_min_contact_distance_m'][1000] = -.0031
        r = measure(a,c,{'stop_reasons':['penetration_over_3mm']})
        self.assertFalse(r['eligible']); self.assertFalse(r['physical_valid'])

class AdapterEnvelopeTests(unittest.TestCase):
    def test_fixed_execution_arguments_and_combination(self):
        from unittest.mock import patch
        from movement_lab import residual02a, transition01c
        import sys
        with patch.object(sys, 'argv', ['adapter', '--out', '/tmp/unused02a', '--coefficients', '-1', '-.5']):
            with patch.object(transition01c, 'main') as run:
                residual02a.main()
                self.assertEqual(sys.argv[1:], ['--out', '/tmp/unused02a', '--seed', '42', '--sequence', 'explore02a'])
                self.assertEqual(transition01c.SEQUENCES['explore02a']['head_yaw'], -.08)
                self.assertEqual(transition01c.SEQUENCES['explore02a']['left_knee'], -.025)
                run.assert_called_once()

if __name__ == '__main__': unittest.main()
