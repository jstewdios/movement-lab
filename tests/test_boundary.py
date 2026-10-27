import json
from pathlib import Path
import unittest
import numpy as np
from movement_lab.study00c import choose_probe, metrics


class BoundaryTests(unittest.TestCase):
    def test_nonmonotone_points_are_retained_and_audited(self):
        # Failure at .12 and survival at .24 does not collapse into a threshold.
        outcomes = {.06:False, .12:True, .24:False}
        self.assertEqual(choose_probe(outcomes), .09)
        self.assertEqual(choose_probe(outcomes, audit=True), .03)
        self.assertEqual(outcomes[.24], False)
        self.assertEqual(choose_probe({.06:False, .12:False}), .24)

    def test_return_is_distinct_from_displaced_recovery(self):
        a = dict(np.load('results/99-balance/trial.npz', allow_pickle=False))
        m = json.loads(Path('results/99-balance/metadata.json').read_text())
        m['disturbance'] = dict(actual_time_s=2., requested_time_s=2.)
        a['com_position_w_m'] = a['body_pose_w'][:,0,:3].copy()
        self.assertTrue(metrics(a,m)['returned_to_origin'])
        # A stable new location passes 00B recovery, but does not return.
        a['body_pose_w'][a['time_s']>=2.,0,0] += .02
        s = metrics(a,m)
        self.assertTrue(s['recovered'])
        self.assertFalse(s['returned_to_origin'])
        self.assertFalse(s['failure'])
        self.assertEqual(s['state'], 'recovered displaced/posture changed')

    def test_upright_instability_is_not_called_falling(self):
        a = dict(np.load('results/99-balance/trial.npz', allow_pickle=False))
        m = json.loads(Path('results/99-balance/metadata.json').read_text())
        m['disturbance'] = dict(actual_time_s=2., requested_time_s=2.)
        a['com_position_w_m'] = a['body_pose_w'][:,0,:3].copy()
        a['body_velocity_w'][-100:,0,0] = .1
        s = metrics(a,m)
        self.assertFalse(s['failure'])
        self.assertFalse(s['recovered'])
        self.assertEqual(s['state'], 'upright, recovery unresolved')

    def test_rotated_standing_is_not_return_to_original_orientation(self):
        a = dict(np.load('results/99-balance/trial.npz', allow_pickle=False))
        m = json.loads(Path('results/99-balance/metadata.json').read_text())
        m['disturbance'] = dict(actual_time_s=2., requested_time_s=2.)
        a['com_position_w_m'] = a['body_pose_w'][:,0,:3].copy()
        post = a['time_s'] >= 2.
        q = a['body_pose_w'][post,0,3:7].copy()
        c,s = np.cos(np.pi/8),np.sin(np.pi/8)
        a['body_pose_w'][post,0,3:7] = np.column_stack((c*q[:,0]-s*q[:,3], c*q[:,1]-s*q[:,2], c*q[:,2]+s*q[:,1], c*q[:,3]+s*q[:,0]))
        result = metrics(a,m)
        self.assertTrue(result['recovered'])
        self.assertFalse(result['returned_to_origin'])
        self.assertGreater(result['final_orientation_offset_deg'], 40)

if __name__ == '__main__': unittest.main()


class BoundaryValidationTests(unittest.TestCase):
    def test_terminated_trial_uses_common_pre_event_samples(self):
        import hashlib
        import tempfile
        from movement_lab.validate00b import check
        a = dict(np.load('results/99-balance/trial.npz', allow_pickle=False))
        base = json.loads(Path('results/99-balance/metadata.json').read_text())
        # Regression for the boundary-specific failure: a terminated trial is
        # shorter than its control, while the whole pre-event history matches.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for dv in (0., .06):
                dest = root/f'99-s42-dv{dv:+.2f}'; dest.mkdir()
                m = dict(base, controller='policy')
                m['disturbance'] = dict(applied=True, actual_time_s=2., requested_time_s=2.,
                    requested_dv_m_s=dv, qvel_before=[0.]*20, qvel_after=[dv]+[0.]*19)
                b = {k:v[:500 if dv else len(v)].copy() for k,v in a.items()}
                if dv:
                    i = np.flatnonzero(b['time_s']>2.+1e-6)[0]
                    b['qvel'][i,0] += dv; b['qpos'][i,0] += .001
                np.savez(dest/'trial.npz', **b)
                for file,key in [('evaluate.py','measurement_code_sha256'),('measure.py','analysis_code_sha256'),('disturbance.py','disturbance_code_sha256'),('resolved_config.txt','config_sha256')]:
                    data = b'fixture provenance'
                    (dest/file).write_bytes(data)
                    m[key] = hashlib.sha256(data).hexdigest()
                (dest/'metadata.json').write_text(json.dumps(m))
            self.assertEqual(check(root)['trials'], 2)
            # The length fix must not weaken rejection of pre-event changes.
            dest = root/'99-s42-dv+0.06'
            with np.load(dest/'trial.npz') as f: b = dict(f)
            b['qpos'][100,0] += 1e-5
            np.savez(dest/'trial.npz', **b)
            with self.assertRaises(AssertionError): check(root)
