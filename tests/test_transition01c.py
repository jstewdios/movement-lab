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
        self.assertIn('Unsuccessful completed movement conditions: none.', headline)
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
        self.assertIn('Unsuccessful completed movement conditions: flex — zero / below response threshold.', headline)
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
