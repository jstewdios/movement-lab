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

    def test_stopped_window_metrics_without_local_geometry(self):
        for samples in (480,1960):
            with self.subTest(samples=samples):
                c=self.record();a={k:v[:samples].copy() for k,v in c.items()}
                a['native_min_contact_distance_m'][-1]=-.0031
                result=measure(a,c,dict(pulse=dict(rise_s=.6,hold_s=1.,return_s=1.4),
                    stop_reasons=['penetration_over_3mm'],physics_dt=.005))
                self.assertEqual(result['classification'],'physical failure / incomplete')
                for key in ('stable_final','final_joint_rad','final_com_mm','final_joint_delta_rad','final_actor_delta_rad','final_com_delta_m'):
                    self.assertIsNone(result[key],key)
                if samples==480:
                    self.assertIsNone(result['stable_hold'])
                    self.assertIsNone(result['hold_joint_delta_rad'])



class TransitionReportTests(unittest.TestCase):
    """Generate from copied audit inputs in /tmp, never in the preserved study."""

    def setUp(self):
        import json
        from pathlib import Path
        import tempfile
        self.temp = tempfile.TemporaryDirectory(prefix='movement-lab-report-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        fixture = Path(__file__).resolve().parents[1]/'results/01c/pilot-gpu'
        for name in ('independent-validation.json', 'cost.json', 'joint-analysis.json', 'PLAN.json'):
            (self.root/name).write_bytes((fixture/name).read_bytes())
        self.audit = json.loads((self.root/'independent-validation.json').read_text())
        self.original_classifications = {n:r['classification'] for n,r in self.audit.items()}

    def report(self):
        import json
        from movement_lab.write_report01c import write_report
        (self.root/'independent-validation.json').write_text(json.dumps(self.audit))
        write_report(self.root)
        text = (self.root/'REPORT.md').read_text()
        self.assertTrue((self.root/'manifest.json').is_file())
        # The returned headline, not just the later table, must match the audit.
        return text.splitlines()[2]

    def test_complete_five_trial_pilot(self):
        headline = self.report()
        self.assertIn('5 audited trial records', headline)
        for name in ('flex', 'extend', 'head-positive', 'head-negative'):
            self.assertIn(f'{name} — {self.original_classifications[name]}', headline)
        self.assertIn('Unsuccessful tested movement conditions: none.', headline)
        self.assertIn('Untested planned conditions: rock-forward, rock-backward, final-control.', headline)
        self.assertIn('Controls: baseline — zero / below response threshold.', headline)

    def test_budget_limited_missing_flexion_or_head_yaw(self):
        from copy import deepcopy
        complete = deepcopy(self.audit)
        for missing in (('flex',), ('head-positive',), ('head-positive', 'head-negative')):
            with self.subTest(missing=missing):
                self.audit = {n:r for n,r in complete.items() if n not in missing}
                headline = self.report()
                untested = headline.split('Untested planned conditions: ', 1)[1].split('.', 1)[0]
                for name in missing:
                    self.assertIn(name, untested)
                    self.assertNotIn(f'{name} — ', headline)
                for name in set(complete)-set(missing)-{'baseline'}:
                    self.assertIn(f'{name} — {complete[name]["classification"]}', headline)

    def test_completed_trial_with_different_classification(self):
        # A valid completed trial below the response threshold is unsuccessful.
        self.audit['flex']['classification'] = 'zero / below response threshold'
        self.audit['flex']['measurable_response'] = False
        # A changed head configuration is distinct from a partial return.
        self.audit['head-positive']['classification'] = 'stable changed configuration'
        self.audit['head-positive']['contracts_by_half'] = False
        headline = self.report()
        self.assertIn('Unsuccessful tested movement conditions: flex — zero / below response threshold.', headline)
        self.assertNotIn('flex — stable changed configuration', headline)
        self.assertIn('head-positive — stable changed configuration', headline)
        self.assertNotIn('head-positive — stable transition with partial return', headline)

    def test_no_demonstrated_controlled_transition(self):
        for name, record in self.audit.items():
            record['classification'] = 'zero / below response threshold'
            record['measurable_response'] = False
        for names in (tuple(self.audit), ('baseline',)):
            with self.subTest(completed=names):
                self.audit = {n:r for n,r in self.audit.items() if n in names}
                headline = self.report()
                self.assertIn('No controlled transition was demonstrated.', headline)
                self.assertNotIn('Demonstrated:', headline)
                self.assertNotIn('stable changed configuration', headline)
                self.assertNotIn('partial return', headline)
                for name in set(names)-{'baseline'}:
                    self.assertIn(f'{name} — zero / below response threshold', headline)


class SafetyStoppedPipelineTests(unittest.TestCase):
    """CPU-only derived fixtures exercise saved-record auditing and publication."""

    def setUp(self):
        import json
        from pathlib import Path
        import shutil
        import tempfile
        self.json = json
        self.source = Path(__file__).resolve().parents[1]/'results/01c/pilot-gpu'
        if not (self.source/'baseline/replay_model.mjb').exists():
            self.skipTest('requires local 01C trajectories and compiled geometry')
        self.temp = tempfile.TemporaryDirectory(prefix='01c-safety-pipeline-', dir='/tmp')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for name in ('baseline', 'flex', 'head-negative'):
            shutil.copytree(self.source/name, self.root/name, copy_function=shutil.copyfile,
                            ignore=shutil.ignore_patterns('matplotlib'))
        for name in ('PLAN.json', 'cost.json'):
            shutil.copyfile(self.source/name, self.root/name)

    def refresh_manifest(self, out):
        from movement_lab.evaluate import sha
        entries = {str(p.relative_to(out)):sha(p) for p in out.rglob('*')
                   if p.is_file() and p.name != 'manifest.json'}
        (out/'manifest.json').write_text(self.json.dumps(entries))

    def stop_movement(self, samples=480, out_name='flex', sequence='flex_+1'):
        import mujoco
        from movement_lab.controlled01b import gate
        from movement_lab.discovery_metrics import analyze
        from movement_lab.study01c import observed_safety_stops
        out = self.root/out_name
        trial = out/sequence
        a = {k:v[:samples].copy() for k,v in dict(np.load(self.source/out_name/sequence/'trial.npz')).items()}
        m = self.json.loads((self.source/out_name/sequence/'metadata.json').read_text())
        model = mujoco.MjModel.from_binary_path(str(out/'replay_model.mjb'))
        d = mujoco.MjData(model)
        # Explicitly synthetic last state: lean past the unchanged 12-degree gate.
        a['qpos'][-1,3:7] = [np.cos(np.radians(13)/2), 0, np.sin(np.radians(13)/2), 0]
        d.qpos[:] = a['qpos'][-1]; d.qvel[:] = a['qvel'][-1]
        mujoco.mj_forward(model, d)
        ids = [mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'robot/'+n) for n in m['body_names']]
        a['body_pose_w'][-1,:,:3] = d.xpos[ids]
        a['body_pose_w'][-1,:,3:7] = d.xquat[ids]
        mass = np.array(m['mass_kg'])
        a['com_position_w_m'][-1] = (d.xipos[ids]*mass[:,None]).sum(0)/mass.sum()
        terrain = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,'terrain')
        count = nonfoot = 0
        feet = np.zeros(2,dtype=int)
        for contact in d.contact:
            if terrain not in (contact.geom1,contact.geom2): continue
            count += 1
            g = contact.geom2 if contact.geom1 == terrain else contact.geom1
            gn = mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_GEOM,g) or ''
            for i,n in enumerate(('left_foot_collision','right_foot_collision')): feet[i] += int(n in gn)
            nonfoot += int(not any(n in gn for n in ('left_foot_collision','right_foot_collision')))
        a['native_ground_contacts'][-1] = count
        a['native_nonfoot_contacts'][-1] = nonfoot
        a['native_foot_contacts'][-1] = feet
        a['foot_found'][-1] = feet
        a['native_min_contact_distance_m'][-1] = min([0.]+[float(c.dist) for c in d.contact])
        m['stop_reasons'] = sorted(observed_safety_stops(a,np.array(m['joint_limits_rad'])))
        self.assertIn('tilt_over_12deg',m['stop_reasons'])
        m['gate'] = gate(a,m['stop_reasons'],False)
        m['summary'] = analyze(a,dict(np.load(trial/'initial.npz')),np.array(m['joint_limits_rad']))
        np.savez_compressed(trial/'trial.npz',**a)
        (trial/'metadata.json').write_text(self.json.dumps(m))
        self.refresh_manifest(out)
        return out

    def test_safety_stopped_movement_and_partial_final_window(self):
        from movement_lab.study01c import check
        # No endpoint substitution even when part of [9.5,10] was recorded.
        for samples in (480, 1960):
            with self.subTest(samples=samples):
                out = self.stop_movement(samples)
                result = check(out,self.root/'baseline')
                self.assertEqual(result['classification'],'physical failure / incomplete')
                self.assertFalse(result['physical_valid'])
                for key in ('stable_final','final_joint_rad','final_com_mm','final_joint_delta_rad','final_actor_delta_rad','final_com_delta_m'):
                    self.assertIsNone(result[key],key)
                self.assertIn('tilt_over_12deg',result['stop_reasons'])
                self.assertAlmostEqual(result['recorded_seconds'],samples*.005,places=3)

    def test_mixed_complete_and_safety_stopped_reports(self):
        from movement_lab.report01c import generate_report
        from movement_lab.write_report01c import write_report
        self.stop_movement()
        generate_report(self.root)
        write_report(self.root)
        audit = self.json.loads((self.root/'independent-validation.json').read_text())
        self.assertEqual(audit['head-negative']['classification'],'stable transition with partial return')
        self.assertEqual(audit['flex']['classification'],'physical failure / incomplete')
        for key in ('absolute_final_return_01b_tolerances','maintained_final_joint_configuration','maintained_hold_response'):
            self.assertIsNone(audit['flex'][key],key)
        detail = self.json.loads((self.root/'joint-analysis.json').read_text())['flex']
        self.assertIsNone(detail['load_hold_delta_n'])
        self.assertIsNone(detail['joints'][0]['final_delta_rad'])
        self.assertIsNone(detail['foot_origin_measurements']['ankle_left']['final_matched_origin_delta_m'])
        text = (self.root/'REPORT.md').read_text()
        self.assertIn('flex — physical failure / incomplete',text)
        self.assertIn('head-negative — stable transition with partial return',text)
        self.assertIn('unavailable',text)
        atlas = (self.root/'comparison.html').read_text()
        self.assertIn('physical failure / incomplete',atlas)
        self.assertIn('unavailable',atlas)
        self.assertIn('d.t.length-1',atlas)
        # Playback has only actual recorded frames for the stopped movement.
        payload = self.json.loads(atlas.split('const data=',1)[1].split(',meta=',1)[0])
        self.assertEqual(len(payload['flex']['t']),120)
        self.assertEqual(len(payload['head-negative']['t']),500)

    def test_incomplete_baseline_is_hard_failure(self):
        from movement_lab.study01c import check
        from movement_lab.controlled01b import gate
        out = self.root/'baseline'; trial = out/'zero'
        a = {k:v[:480] for k,v in dict(np.load(trial/'trial.npz')).items()}
        np.savez_compressed(trial/'trial.npz',**a)
        m = self.json.loads((trial/'metadata.json').read_text())
        m['gate'] = gate(a,m['stop_reasons'],True)
        (trial/'metadata.json').write_text(self.json.dumps(m))
        self.refresh_manifest(out)
        with self.assertRaisesRegex(AssertionError,'incomplete baseline'):
            check(out,out)

    def test_unexplained_truncation_corrupt_or_missing_data_is_not_safety_outcome(self):
        from movement_lab.study01c import check
        trial = self.root/'flex/flex_+1'
        a = {k:v[:480] for k,v in dict(np.load(trial/'trial.npz')).items()}
        from movement_lab.controlled01b import gate
        m = self.json.loads((trial/'metadata.json').read_text())
        m['gate'] = gate(a,[],False)
        (trial/'metadata.json').write_text(self.json.dumps(m))
        np.savez_compressed(trial/'trial.npz',**a)
        self.refresh_manifest(self.root/'flex')
        with self.assertRaisesRegex(AssertionError,'unexplained incomplete movement'):
            check(self.root/'flex',self.root/'baseline')
        # A claimed stop without measured evidence is not an accepted failure.
        m['stop_reasons'] = ['tilt_over_12deg']
        m['gate'] = gate(a,m['stop_reasons'],False)
        (trial/'metadata.json').write_text(self.json.dumps(m))
        self.refresh_manifest(self.root/'flex')
        with self.assertRaisesRegex(AssertionError,'inconsistent safety stop evidence'):
            check(self.root/'flex',self.root/'baseline')
        # A hash mismatch remains corruption, even if a stop reason is present.
        (trial/'metadata.json').write_text('{}')
        import subprocess
        with self.assertRaises(subprocess.CalledProcessError):
            check(self.root/'flex',self.root/'baseline')
        (trial/'trial.npz').unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            check(self.root/'flex',self.root/'baseline')
