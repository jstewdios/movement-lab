"""Information boundaries, physical validity, paired curriculum and resumption."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from movement_lab.control02d import Controller, Design, load_basis
from movement_lab.sim02d import Condition, rollout
from movement_lab.sim02b import Sim, DEFAULT_MODEL
from movement_lab.sim02c import BASE, rollout as old_rollout, Design as FixedDesign
from movement_lab.learn02d import conditions
from movement_lab.eval02d import battery, shifts
from movement_lab.eval02c import audit_record
from tests.test_move02c import synthetic_record

PHYSICS = (DEFAULT_MODEL / 'model.mjb').exists() and importlib.util.find_spec('bam') is not None


class ControlTests(unittest.TestCase):
    def test_staged_driver_rejects_changed_reused_helper(self):
        from movement_lab.study02d import require_dependencies, DEPENDENCIES
        from movement_lab.learn02c import digest
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            (root/'dependencies.json').write_text(json.dumps({n:digest(Path('movement_lab')/n) for n in DEPENDENCIES}))
            require_dependencies(root)
            with patch('movement_lab.study02d.digest',return_value='changed'):
                with self.assertRaises(SystemExit):require_dependencies(root)

    def test_only_sensors_drive_response_and_state_resets(self):
        w = np.linspace(-.2,.2,22); basis = load_basis()
        a = Controller(w, basis); b = Controller(w, basis)
        for x in (np.zeros(5), np.ones(5), np.zeros(5)):
            np.testing.assert_array_equal(a(x), b(x))
        self.assertGreater(np.max(np.abs(a(np.zeros(5)) - Controller(w, basis)(np.zeros(5)))), 0)
        self.assertLessEqual(np.abs(a(np.full(5, 1e8))).max(), .8)

    def test_split_seeds_and_stratification(self):
        train = {c.seed for s in (1,2,3,71,72) for g in range(12) for c in conditions(s,g,True)}
        dev = {c.seed for _,c in battery(True)} | {c.seed for _,c in battery(True,True)}
        test = {c.seed for _,c in battery(False)} | {c.seed for _,c in shifts()}
        self.assertFalse(train & dev or train & test or dev & test)
        cs = conditions(1,0)
        self.assertIsNone(cs[0].push_time_s)
        self.assertGreater(cs[4].push_dv[0], cs[1].push_dv[0])
        self.assertGreater(cs[5].pulse_duration_s,0)
        self.assertIsNotNone(cs[7].false_gyro_time_s)
        revised=conditions(1,0,True)
        self.assertEqual(len(revised),12)
        self.assertTrue(all(revised[j].push_time_s is None and revised[j].duration_s==6 for j in (0,8,9,10)))

    def test_fall_flicker_slide_and_unsupported_endpoint_rejected(self):
        r = synthetic_record(); c = Condition(duration_s=6, push_time_s=2)
        self.assertTrue(audit_record(r,c,.118)['recovered_200hz'])
        # A contact transition with no geometric clearance is not a step.
        r['foot_found'][500:510,0] = False
        r['foot_pos'][510:,0,0] = .02
        self.assertFalse(audit_record(r,c,.118)['recovery_with_step'])
        # Even a local qualifying flight and replant cannot erase a prior fall.
        r['clearance_m'][500:510,0] = .01
        self.assertTrue(audit_record(r,c,.118)['recovery_with_step'])
        r['tilt_deg'][400] = 50
        self.assertFalse(audit_record(r,c,.118)['recovered_200hz'])
        self.assertFalse(audit_record(r,c,.118)['recovery_with_step'])
        r['tilt_deg'][:] = 0; r['nonfoot'][400] = 1
        self.assertFalse(audit_record(r,c,.118)['recovered_200hz'])
        r['nonfoot'][:] = 0; r['foot_force_n'][-1] = 0
        self.assertFalse(audit_record(r,c,.118)['recovered_200hz'])


@unittest.skipUnless(PHYSICS, 'local compiled model / BAM unavailable')
class PhysicsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim = Sim(); cls.theta = np.load(BASE)['final']

    def test_legacy_and_zero_compatibility(self):
        c = Condition(seed=123, duration_s=3, push_time_s=1.5, push_dv=(.32,0))
        for mode in ('reflex','oracle02c','imu02c'):
            _, a = rollout(self.sim,self.theta,np.zeros(22),c,record=True,terminate=False,mode=mode)
            b = np.zeros(28) if mode == 'reflex' else load_basis().ravel()
            _, old = old_rollout(self.sim,self.theta,b,c,FixedDesign(knots=2,horizon=.4,trigger='imu' if mode=='imu02c' else 'event'),record=True,terminate=False)
            np.testing.assert_array_equal(a['qpos'],old['qpos'])
        _, zero = rollout(self.sim,self.theta,np.zeros(22),c,record=True,terminate=False)
        _, frozen = rollout(self.sim,self.theta,np.zeros(22),c,record=True,terminate=False,mode='reflex')
        np.testing.assert_array_equal(zero['qpos'],frozen['qpos'])

    def test_no_anticipation_or_privileged_velocity_sensor(self):
        w = np.linspace(-.02,.02,22)
        c = Condition(seed=8,duration_s=2,push_time_s=1.5,push_dv=(.32,0))
        _, pushed = rollout(self.sim,self.theta,w,c,record=True,terminate=False)
        c.push_dv = (0,0)
        _, quiet = rollout(self.sim,self.theta,w,c,record=True,terminate=False)
        # The instantaneous root-translation velocity change is invisible to the IMU.
        pre = pushed['time_s'] <= 1.52 + 1e-9
        np.testing.assert_array_equal(pushed['residual'][pre],quiet['residual'][pre])
        pre_state = pushed['time_s'] <= 1.5 + 1e-9
        np.testing.assert_array_equal(pushed['qpos'][pre_state],quiet['qpos'][pre_state])

    def test_recording_passive_limits_and_disturbances(self):
        c = Condition(seed=10,duration_s=3,pulse_time_s=1.5,pulse_duration_s=.12,pulse_force=(.4,0,0))
        w = np.zeros(22)
        s,r = rollout(self.sim,self.theta,w,c,record=True,terminate=False)
        plain,_ = rollout(self.sim,self.theta,w,c,terminate=False)
        self.assertEqual(s['final_root_xy'],plain['final_root_xy'])
        self.assertTrue(s['torque_within_envelope'] and s['targets_within_limits'])
        c.pulse_force=(0,0,0)
        _,q = rollout(self.sim,self.theta,w,c,record=True,terminate=False)
        np.testing.assert_array_equal(r['qpos'][r['time_s']<=1.5+1e-9],q['qpos'][q['time_s']<=1.5+1e-9])
        self.assertGreater(np.abs(r['qvel']-q['qvel']).max(),.01)

    def test_oracle_is_not_falsely_cued_by_sensor_artifact(self):
        from movement_lab.eval02d import effective_mode
        c=Condition(seed=12,duration_s=3,false_gyro_time_s=1.5,false_gyro_rad_s=.75)
        _,oracle=rollout(self.sim,self.theta,np.zeros(22),c,record=True,terminate=False,mode=effective_mode('oracle02c',c))
        _,trigger=rollout(self.sim,self.theta,np.zeros(22),c,record=True,terminate=False,mode=effective_mode('imu02c',c))
        self.assertEqual(float(np.abs(oracle['residual']).max()),0.)
        self.assertGreater(float(np.abs(trigger['residual']).max()),.1)

    def test_exact_generation_resume_and_staging_ignored(self):
        from movement_lab.learn02d import main
        with tempfile.TemporaryDirectory() as td:
            a,b = Path(td,'a'),Path(td,'b')
            args = ['--iterations','2','--population','4','--elite','1','--workers','1','--revision']
            main(['--out',str(a),*args])
            main(['--out',str(b),'--budget','.1',*args])
            from movement_lab.learn02c import completed_checkpoints
            self.assertEqual(len(completed_checkpoints(b)),1)
            first_bytes=(b/'generation-001.npz').read_bytes()
            (b/'generation-002.tmp.npz').write_bytes(b'interrupted')
            main(['--out',str(b),'--resume',*args])
            self.assertEqual((b/'generation-001.npz').read_bytes(),first_bytes)
            self.assertFalse((b/'generation-002.tmp.npz').exists())
            for gen in (1,2):
                x,y = np.load(a/f'generation-{gen:03d}.npz'),np.load(b/f'generation-{gen:03d}.npz')
                for k in ('candidates','scores','outcomes','mean','std','best','conditions'):
                    np.testing.assert_array_equal(x[k],y[k])
                self.assertGreater(np.std(x['candidates']),0)  # actual varied proposals, not just three inserted controls
            with patch('movement_lab.learn02d.digest',return_value='modified'):
                with self.assertRaises(SystemExit):
                    main(['--out',str(b),'--resume',*args])


if __name__ == '__main__':
    unittest.main()
