import unittest
import numpy as np
from movement_lab.controlled01b import gate
from movement_lab.study01b import envelope, SEQUENCES


class ControlledGateTests(unittest.TestCase):
    def standing(self):
        t=np.arange(.005,6.001,.005)
        p=np.zeros((len(t),1,7));p[:,:,3]=1
        return dict(time_s=t,body_pose_w=p,native_min_contact_distance_m=np.zeros(len(t)),
                    native_nonfoot_contacts=np.zeros(len(t)),foot_found=np.ones((len(t),2)),
                    com_velocity_w_m_s=np.zeros((len(t),3)),body_velocity_w=np.zeros((len(t),1,6)),
                    com_position_w_m=np.zeros((len(t),3)))

    def test_rejects_quiet_fallen_and_body_supported(self):
        a=self.standing();self.assertTrue(gate(a)['pass'])
        a['body_pose_w'][:,:,3]=np.cos(np.pi/4);a['body_pose_w'][:,:,4]=np.sin(np.pi/4)
        self.assertFalse(gate(a)['pass'])
        a=self.standing();a['native_nonfoot_contacts'][-1]=1
        self.assertFalse(gate(a)['pass'])

    def test_contact_and_settling_failures(self):
        for field,value in [('native_min_contact_distance_m',-.0031),('com_velocity_w_m_s',.04),('foot_found',0)]:
            a=self.standing();a[field][-1]=value
            self.assertFalse(gate(a)['pass'],field)
        a=self.standing();self.assertFalse(gate(a,['termination'])['pass'])
        a=self.standing();a['time_s'][-1]=5.9
        self.assertFalse(gate(a)['pass'])

    def test_bounded_signed_commands_and_recovery(self):
        self.assertEqual(envelope(1.99),0);self.assertEqual(envelope(3.61),0)
        self.assertAlmostEqual(envelope(2.6),1)
        for group in ('head','flex','shift','left'):
            self.assertEqual(SEQUENCES[group+'_-1'],{k:-v for k,v in SEQUENCES[group+'_+1'].items()})
        self.assertLessEqual(max(abs(v) for d in SEQUENCES.values() for v in d.values()),.08)
