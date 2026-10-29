"""Publication scope regression: the diagnostic exception is one exact path."""
import unittest
from movement_lab.publication01b import allowed_publication_path


class PublicationScopeTests(unittest.TestCase):
    def test_accepts_legitimate_01b_sources(self):
        for name in ('movement_lab/landing_diagnostics.py', 'movement_lab/landing01b.py',
                     'movement_lab/publication01b.py', 'tests/test_publication01b.py',
                     'results/01b/landing-diagnosis/REPORT.md', 'README.md', '.gitignore'):
            with self.subTest(name=name):
                self.assertTrue(allowed_publication_path(name))

    def test_rejects_unexpected_files_and_near_matches(self):
        for name in ('movement_lab/other_diagnostics.py', 'movement_lab/landing_diagnostics_extra.py',
                     'movement_lab/landing_diagnostics.py.bak', 'tests/landing_diagnostics.py',
                     'results/01a/REPORT.md', 'unrelated/file01b.py', 'README.md.bak'):
            with self.subTest(name=name):
                self.assertFalse(allowed_publication_path(name))
