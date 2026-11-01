"""Publication scope and strict missing-condition safeguards."""
import copy
import json
from pathlib import Path
import unittest
from movement_lab.publication02a import allowed_publication_path
from movement_lab.validate02a import findings

class Publication02aTests(unittest.TestCase):
    def test_scope(self):
        for name in ('movement_lab/validate02a.py','results/02a/pilot-gpu/archive.json','README.md'):
            self.assertTrue(allowed_publication_path(name))
        for name in ('movement_lab/transition01c.py','movement_lab/other02a.py','.aws/credentials','results/01c/REPORT.md'):
            self.assertFalse(allowed_publication_path(name))

    def test_missing_conditions_are_hard_failure(self):
        p = Path('results/02a/pilot-gpu')
        a = json.loads((p/'attempts.json').read_text()); r = json.loads((p/'archive.json').read_text())
        with self.assertRaisesRegex(AssertionError,'conditions'): findings(a[:-1],r)
        changed = copy.deepcopy(r); changed['control'].pop()
        with self.assertRaisesRegex(AssertionError,'incomplete arm'): findings(a,changed)

    def test_redundant_physical_outcome_is_not_command_diversity(self):
        p = Path('results/02a/pilot-gpu')
        a = json.loads((p/'attempts.json').read_text()); r = json.loads((p/'archive.json').read_text())
        stats = findings(a,r)
        self.assertEqual(stats['adaptive']['discoveries'],1)
        self.assertEqual(stats['control']['discoveries'],2)
        self.assertNotEqual(r['adaptive'][0]['action'],r['adaptive'][1]['action'])
        self.assertTrue(r['adaptive'][1]['physical_valid'])
        self.assertFalse(r['adaptive'][1]['discovery'])
