import json
import os
from pathlib import Path
import unittest
import numpy as np
from movement_lab.disturbance import assess
from movement_lab.measure import validate


class DisturbanceTests(unittest.TestCase):
    def test_recovery_requires_settling_and_final_stability(self):
        a=dict(np.load('results/99-balance/trial.npz'))
        m=json.loads(Path('results/99-balance/metadata.json').read_text())
        m['disturbance']={'requested_time_s':2.}
        self.assertTrue(assess(a,m)['settled'])
        a['body_velocity_w'][-100:,0,0]=.1
        self.assertFalse(assess(a,m)['recovered'])
        self.assertTrue(assess(a,m)['upright'])
        a['body_velocity_w'][300:400,0,0]=.1
        self.assertFalse(assess(a,m)['settled'])

    def test_step_evidence_rejects_sliding_and_requires_lift(self):
        a=dict(np.load('results/99-balance/trial.npz'))
        m=json.loads(Path('results/99-balance/metadata.json').read_text())
        m['disturbance']={'requested_time_s':2.}
        f=m['body_names'].index('ankle_left')
        a['body_pose_w'][450:,f,0]+=.01
        self.assertFalse(assess(a,m)['step_evidence'][0])
        a['foot_found'][440:450,0]=0
        self.assertFalse(assess(a,m)['step_evidence'][0])
        a['body_pose_w'][440:450,f,2]+=.01
        self.assertTrue(assess(a,m)['step_evidence'][0])

    def test_saved_interventions(self):
        fixture_dir = os.environ.get('MOVEMENT_LAB_00B_FIXTURES')
        root = Path(fixture_dir) if fixture_dir is not None else Path('results/00b/standing')
        paths = sorted(root.glob('*/metadata.json'))
        if not paths:
            message = f'No 00B metadata found under {root}'
            if fixture_dir is not None:
                self.fail(message)
            self.skipTest(message + '; saved-record validation not performed')
        for path in paths:
            with self.subTest(trial=str(path.parent)):
                if not (path.parent/'trial.npz').is_file():
                    message = f'Missing raw trajectory: {path.parent / "trial.npz"}'
                    if fixture_dir is not None:
                        self.fail(message)
                    self.skipTest(message + '; saved-record validation not performed. '
                                  'Set MOVEMENT_LAB_00B_FIXTURES to a complete record directory')
                m=json.loads(path.read_text());e=m['disturbance']
                with np.load(path.parent/'trial.npz', allow_pickle=False) as record:
                    a=dict(record)
                validate(a)
                if not e['applied']:
                    self.assertEqual(m['controller'],'default-position')
                    self.assertTrue(m['stopped_on_done'])
                    self.assertLess(a['time_s'][-1],2.)
                    self.assertFalse(assess(a,m)['recovered'])
                    continue
                self.assertLessEqual(abs(e['actual_time_s']-2.),m['physics_dt']*1.01)
                delta=np.array(e['qvel_after'])-e['qvel_before']
                np.testing.assert_allclose(delta[0],e['requested_dv_m_s'],atol=2e-7)
                np.testing.assert_allclose(delta[1:],0,atol=2e-7)
                self.assertTrue(np.isfinite(e['first_integrated_qvel']).all())
                np.testing.assert_allclose(a['qpos'][:,:7],a['body_pose_w'][:,0,:],atol=2e-6)
                np.testing.assert_array_equal(a['qvel'][:,6:],a['joint_velocity_rad_s'])
                np.testing.assert_array_equal(a['actuator_force'],a['joint_force_nm'])
                np.testing.assert_allclose(np.diff(a['time_s']),m['physics_dt'],atol=2e-7)
                self.assertLessEqual(len(a['time_s']),800)
                self.assertEqual(len(a['time_s']) % 4,0)
                np.testing.assert_array_equal(a['command'],0)
                absent=a['foot_found']==0
                np.testing.assert_array_equal(a['foot_force_w_n'][absent],0)
