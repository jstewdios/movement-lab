"""Fixture routing regressions, not substitute evidence for the 00B experiment."""
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import test_disturbance


class SavedFixtureModes(unittest.TestCase):
    def run_saved_test(self, root, explicit=True):
        # Run the real test with an isolated result so expected failures/skips
        # can be checked without misreporting them as experimental validation.
        env = dict(os.environ)
        env.pop('MOVEMENT_LAB_00B_FIXTURES', None)
        if explicit:
            env['MOVEMENT_LAB_00B_FIXTURES'] = str(root)
        real_path = Path

        def select_path(value):
            return root if value == 'results/00b/standing' else real_path(value)

        result = unittest.TestResult()
        with patch.dict(os.environ, env, clear=True), \
                patch.object(test_disturbance, 'Path', side_effect=select_path):
            test_disturbance.DisturbanceTests('test_saved_interventions').run(result)
        return result

    def add_metadata(self, root, name):
        trial = root/name
        trial.mkdir()
        shutil.copyfile('results/00b/standing/99-s42-dv+0.00/metadata.json',
                        trial/'metadata.json')
        return trial

    def test_absent_optional_records_are_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for with_metadata in (False, True):
                with self.subTest(with_metadata=with_metadata):
                    if with_metadata:
                        self.add_metadata(root, 'metadata-only')
                    result = self.run_saved_test(root, explicit=False)
                    self.assertTrue(result.wasSuccessful())
                    self.assertEqual(len(result.skipped), 1)
                    self.assertIn('not performed', result.skipped[0][1])

    def test_explicit_missing_or_empty_root_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            for root in (Path(directory), Path(directory)/'missing'):
                with self.subTest(root=root):
                    result = self.run_saved_test(root)
                    self.assertEqual(len(result.failures), 1)
                    self.assertFalse(result.skipped)
                    self.assertFalse(result.errors)

    def test_explicit_metadata_without_raw_record_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.add_metadata(root, 'metadata-only')
            result = self.run_saved_test(root)
            self.assertEqual(len(result.failures), 1)
            self.assertIn('Missing raw trajectory', result.failures[0][1])
            self.assertFalse(result.skipped)

    def test_partial_directory_still_checks_available_real_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.add_metadata(root, 'a-missing')
            trial = self.add_metadata(root, 'b-mismatched')
            # Deliberately pair a committed five-second 00A trajectory with
            # four-second 00B metadata. It MUST fail the existing sample-count
            # assertion, after loading and validating the real arrays. This is
            # a negative control, never a fabricated passing 00B fixture.
            shutil.copyfile('results/99-balance/trial.npz', trial/'trial.npz')
            for explicit in (False, True):
                with self.subTest(explicit=explicit):
                    result = self.run_saved_test(root, explicit)
                    self.assertFalse(result.errors)
                    self.assertEqual(len(result.failures), 2 if explicit else 1)
                    self.assertEqual(len(result.skipped), 0 if explicit else 1)
                    self.assertIn('1000 not less than or equal to 800',
                                  result.failures[-1][1])

    def test_corrupt_available_archive_is_not_skipped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trial = self.add_metadata(root, 'corrupt')
            (trial/'trial.npz').write_bytes(b'not an NPZ archive')
            for explicit in (False, True):
                with self.subTest(explicit=explicit):
                    result = self.run_saved_test(root, explicit)
                    self.assertEqual(len(result.errors), 1)
                    self.assertFalse(result.skipped)
