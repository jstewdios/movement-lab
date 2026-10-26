import unittest
import numpy as np
from movement_lab.measure import summary, validate


class Measurements(unittest.TestCase):
    def fixture(self):
        pose = np.zeros((2, 1, 7))
        pose[..., 2] = .2
        pose[..., 3] = 1
        return dict(time_s=np.array([.005, .01]), body_pose_w=pose,
                    joint_force_nm=np.full((2, 2), 2.), joint_velocity_rad_s=np.full((2, 2), -3.),
                    foot_found=np.array([[1,0],[1,1]]), foot_force_w_n=np.array([[[0,0,4],[0,0,0]],[[0,0,4],[0,0,8]]]))

    def test_known_physics(self):
        s = summary(self.fixture())
        self.assertAlmostEqual(s['absolute_mechanical_work_j'], .12)
        self.assertEqual(s['max_root_tilt_deg'], 0)
        self.assertEqual(s['foot_contact_fraction'], [1., .5])
        self.assertEqual(s['peak_foot_force_n'], [4., 8.])
        self.assertEqual(s['rms_joint_force_nm'], 2.)

    def test_tilt(self):
        a = self.fixture()
        a['body_pose_w'][..., 3:7] = [np.sqrt(.5),np.sqrt(.5),0,0]
        self.assertAlmostEqual(summary(a)['max_root_tilt_deg'], 90)

    def test_reject_corruption(self):
        for mutate in (lambda a: a['time_s'].__setitem__(1, .005),
                       lambda a: a['joint_force_nm'].__setitem__((0,0), np.nan),
                       lambda a: a.update(foot_found=np.zeros((1,2)))):
            a = self.fixture()
            mutate(a)
            with self.assertRaises(AssertionError):
                validate(a)


if __name__ == '__main__':
    unittest.main()
