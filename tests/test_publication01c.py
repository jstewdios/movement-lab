"""Publication scope guards reject earlier evidence and near-matching sources."""
import unittest
from movement_lab.publication01c import allowed_publication_path


class Publication01cScopeTests(unittest.TestCase):
    def test_accepts_study_sources_and_evidence(self):
        for name in ('movement_lab/diagnose01b_for01c.py', 'movement_lab/transition01c.py',
                     'movement_lab/validate01c.py', 'tests/test_publication01c.py',
                     'results/01c/pilot-gpu/REPORT.md', 'README.md', '.gitignore'):
            with self.subTest(name=name):
                self.assertTrue(allowed_publication_path(name))

    def test_rejects_earlier_and_unexpected_sources(self):
        for name in ('movement_lab/controlled01b.py', 'movement_lab/other01c.py',
                     'movement_lab/diagnose01b_for01c.py.bak', 'results/01b/REPORT.md',
                     'README.md.bak', '.aws/credentials', 'unrelated/file01c.py'):
            with self.subTest(name=name):
                self.assertFalse(allowed_publication_path(name))
