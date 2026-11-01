"""CPU-only regressions for unavailable history and nested deadline cleanup."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
from movement_lab.historical02a import compare_historical, record_historical, HistoricalMismatch
from movement_lab.process02a import run_until


class HistoryTests(unittest.TestCase):
    def fixtures(self, root):
        new, old = root/'new', root/'old'
        for d in (new, old):
            (d/'zero').mkdir(parents=True)
            np.savez(d/'zero/trial.npz', time_s=np.arange(2000)*.005)
            (d/'replay_model.mjb').write_bytes(b'fixture model, not simulation evidence')
            (d/'resolved_config.txt').write_text('fixed fixture configuration')
        return new, old

    def test_all_missing_is_explicitly_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp); new, _ = self.fixtures(r)
            a = record_historical(new, r/'audit.json', r/'absent')
            self.assertEqual(a['status'], 'unavailable')
            self.assertEqual(len(a['missing']), 3)
            self.assertEqual(a['comparisons'], {})
            self.assertEqual(json.loads((r/'audit.json').read_text()), a)

    def test_partial_history_never_passes_but_available_mismatches_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp); new, old = self.fixtures(r)
            (old/'zero/trial.npz').unlink(); (old/'replay_model.mjb').unlink()
            a = compare_historical(new, old)
            self.assertEqual(a['status'], 'incomplete')
            self.assertEqual(a['comparisons'], {'config_equal': True})
            (old/'resolved_config.txt').write_text('wrong config')
            with self.assertRaises(HistoricalMismatch): record_historical(new, r/'audit.json', old)
            self.assertEqual(json.loads((r/'audit.json').read_text())['status'], 'failed')

    def test_complete_history_strict_arrays_model_and_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp); new, old = self.fixtures(r)
            self.assertEqual(compare_historical(new, old)['status'], 'passed')
            for name in ('zero/trial.npz', 'replay_model.mjb', 'resolved_config.txt'):
                original = (old/name).read_bytes()
                if name.endswith('.npz'): np.savez(old/name, time_s=np.arange(2))
                else: (old/name).write_bytes(b'mismatch')
                with self.assertRaises(AssertionError): compare_historical(new, old)
                (old/name).write_bytes(original)

    def test_fresh_checkout_report_records_absence_and_validates_new_baseline(self):
        from movement_lab import report02a
        from movement_lab.historical02a import record_historical as record
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'run'; (root/'baseline/zero').mkdir(parents=True)
            np.savez(root/'baseline/zero/trial.npz', time_s=np.arange(2000)*.005)
            for name, value in {'attempts': [], 'archive': {'adaptive': [], 'control': []},
                                'cost': {'wall_s': 1., 'status': 'budget: insufficient launch reserve'}}.items():
                (root/(name+'.json')).write_text(json.dumps(value))
            def history(base, dest): return record(base, dest, Path(tmp)/'absent')
            with patch.object(sys, 'argv', ['report', str(root)]), patch.object(report02a, 'record_historical', side_effect=history):
                with patch.object(report02a, 'check', return_value={'physical_valid': True}) as check:
                    report02a.main()
                    check.assert_called_once_with(root/'baseline', root/'baseline')
                self.assertIn('comparison unavailable', (root/'REPORT.md').read_text())
                self.assertEqual(json.loads((root/'report-validation.json').read_text())['historical']['status'], 'unavailable')
                with patch.object(report02a, 'check', side_effect=AssertionError('invalid new baseline')):
                    with self.assertRaisesRegex(AssertionError, 'invalid new baseline'): report02a.main()


@unittest.skipUnless(sys.platform == 'linux', 'Linux process group/subreaper contract')
class DeadlineTests(unittest.TestCase):
    def tree(self, root):
        # A validator-like parent -> child -> grandchild; each inherits the group.
        leaf = "import os,time,signal; from pathlib import Path; signal.signal(signal.SIGTERM,signal.SIG_IGN); Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(root/'leaf.pid')
        child = "import os,time,subprocess,sys; from pathlib import Path; subprocess.Popen([sys.executable,'-c',%r]); Path(%r).write_text(str(os.getpid())); time.sleep(60)" % (leaf, str(root/'child.pid'))
        parent = "import os,time,subprocess,sys; from pathlib import Path; subprocess.Popen([sys.executable,'-c',%r]); Path(%r).write_text(str(os.getpid())); print('audit started',flush=True); time.sleep(60)" % (child, str(root/'parent.pid'))
        return [sys.executable, '-c', parent]

    def assert_reaped(self, root):
        for name in ('parent', 'child', 'leaf'):
            pid = int((root/(name+'.pid')).read_text())
            with self.assertRaises(ProcessLookupError): os.kill(pid, 0)

    def test_deadline_before_launch_and_normal_result(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            run_until([sys.executable, '-c', 'raise AssertionError()'], time.monotonic()-1)
        r = run_until([sys.executable, '-c', "print('ok')"], time.monotonic()+5, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout), (0, 'ok\n'))

    def test_execution_and_validation_interruptions_reap_nested_tree_and_keep_attempt(self):
        from movement_lab import study02a
        for phase in ('execution', 'validation'):
            with self.subTest(phase=phase), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); calls = []
                def execute(cmd, deadline, **kwargs):
                    calls.append(deadline)
                    if phase == 'validation' and len(calls) == 1:
                        return subprocess.CompletedProcess(cmd, 0)
                    return run_until(self.tree(root), min(deadline, time.monotonic()+1), **kwargs)
                out = root/'run'
                with patch.object(sys, 'argv', ['study', '--out', str(out), '--budget', '600']), patch.object(study02a, 'run_until', side_effect=execute), patch.object(study02a, 'plot'):
                    study02a.main()
                self.assert_reaped(root)
                self.assertEqual(len(set(calls)), 1, 'execution/validation must share one absolute deadline')
                row, = json.loads((out/'attempts.json').read_text())
                self.assertEqual(row['status'], 'deadline interrupted')
                self.assertEqual(row['interrupted_phase'], phase)
                self.assertTrue(row['subprocess_tree_reaped'])
                self.assertNotIn('validation', row); self.assertNotIn('outcome', row)
                self.assertEqual(json.loads((out/'archive.json').read_text()), {'adaptive': [], 'control': []})
                self.assertIn('hard deadline', json.loads((out/'cost.json').read_text())['status'])
                if phase == 'validation': self.assertIn('audit started', (out/'baseline-validation.log').read_text())
