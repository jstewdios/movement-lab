import unittest
import numpy as np
from movement_lab import transition01c as cmd
from movement_lab.study01c import measure

class TransitionTests(unittest.TestCase):
    def record(self):
        t=np.arange(.005,10.001,.005);n=len(t);p=np.zeros((n,1,7));p[:,:,3]=1
        return dict(time_s=t,body_pose_w=p,native_min_contact_distance_m=np.zeros(n),native_nonfoot_contacts=np.zeros(n),foot_found=np.ones((n,2)),com_velocity_w_m_s=np.zeros((n,3)),com_position_w_m=np.zeros((n,3)),body_velocity_w=np.zeros((n,1,6)),joint_position_rad=np.zeros((n,14)),actor_action=np.zeros((n,14)),foot_force_w_n=np.ones((n,2,3)),joint_force_nm=np.zeros((n,14)),joint_velocity_rad_s=np.zeros((n,14)))
    def metric(self,a,c):
        return measure(a,c,dict(pulse=dict(rise_s=.6,hold_s=1.,return_s=1.4),stop_reasons=[],physics_dt=.005))
    def test_persistent_configuration_is_not_return(self):
        a=self.record();c=self.record();a['joint_position_rad'][a['time_s']>=2,0]=.01
        m=self.metric(a,c);self.assertEqual(m['classification'],'stable changed configuration');self.assertFalse(m['contracts_by_half'])
    def test_no_motion_and_invalid_support(self):
        a=self.record();c=self.record();self.assertFalse(self.metric(a,c)['measurable_response'])
        a['native_min_contact_distance_m'][600]=-.00301
        self.assertFalse(self.metric(a,c)['physical_valid'])
        a=self.record();a['foot_found'][-1]=0
        self.assertFalse(self.metric(a,c)['stable_final'])
    def test_partial_return_and_timing(self):
        a=self.record();c=self.record();w=(a['time_s']>=2)&(a['time_s']<5)
        a['joint_position_rad'][w,0]=.02;a['joint_position_rad'][a['time_s']>=5,0]=.003
        self.assertTrue(self.metric(a,c)['contracts_by_half'])
        self.assertEqual(cmd.envelope(2),0);self.assertAlmostEqual(cmd.envelope(2.6),1);self.assertEqual(cmd.envelope(5),0)
        times=np.arange(0,10,.02);v=np.array([cmd.envelope(t) for t in times])
        self.assertTrue((v[times<2]==0).all());self.assertTrue((v[times>=5]==0).all());self.assertLess(abs(np.diff(v)).max(),.053)
