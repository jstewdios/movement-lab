import unittest
import numpy as np
from movement_lab.departure_metrics01b import metrics


class DepartureCriteriaTests(unittest.TestCase):
    def record(self):
        t=np.arange(.005,6.001,.005);n=len(t);pose=np.zeros((n,1,7));pose[:,:,3]=1
        return dict(time_s=t,body_pose_w=pose,native_min_contact_distance_m=np.zeros(n),native_nonfoot_contacts=np.zeros(n),
            foot_found=np.ones((n,2)),com_velocity_w_m_s=np.zeros((n,3)),com_position_w_m=np.zeros((n,3)),
            body_velocity_w=np.zeros((n,1,6)),joint_position_rad=np.zeros((n,14)),foot_force_w_n=np.ones((n,2,3)),joint_force_nm=np.zeros((n,14)))

    def test_response_requires_actual_motion_and_return(self):
        c=self.record();a=self.record();__import__('json').dumps(metrics(a,c));self.assertFalse(metrics(a,c)['controlled_response'])
        w=(a['time_s']>2)&(a['time_s']<3.6);a['joint_position_rad'][w,0]=.01
        self.assertTrue(metrics(a,c)['controlled_response'])
        a['joint_position_rad'][a['time_s']>=3.6,0]=.009
        self.assertTrue(metrics(a,c)['recovered'])
        self.assertFalse(metrics(a,c)['controlled_response'])  # survival with residual offset is insufficient
        a['joint_position_rad'][a['time_s']>=3.6,0]=.04
        self.assertFalse(metrics(a,c)['recovered'])

    def test_contact_invalidity_and_motionless_fall_rejected(self):
        a=self.record();a['native_min_contact_distance_m'][500]=-.0031
        self.assertFalse(metrics(a)['recovered'])
        a=self.record();a['body_pose_w'][:,:,3]=np.sqrt(.5);a['body_pose_w'][:,:,4]=np.sqrt(.5)
        self.assertFalse(metrics(a)['recovered'])
        a=self.record();a['foot_found'][-1,0]=0
        self.assertFalse(metrics(a)['recovered'])

    def test_final_quietness_without_sustained_return_is_insufficient(self):
        c=self.record();a=self.record();t=a['time_s']
        a['joint_position_rad'][(t>2)&(t<5.9),0]=.05
        self.assertFalse(metrics(a,c)['recovered'])

    def test_unsettled_precommand_is_rejected(self):
        a=self.record();a['com_velocity_w_m_s'][(a['time_s']>1.5)&(a['time_s']<2),0]=.04
        self.assertFalse(metrics(a)['recovered'])
