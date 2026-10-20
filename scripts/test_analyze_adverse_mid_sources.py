"""Known-answer tests for magnitude/phase error decomposition."""
import unittest

import numpy as np

from scripts.analyze_adverse_mid_sources import band_decomposition


class AdverseMidMetricTests(unittest.TestCase):
    def setUp(self):
        n = 16384
        self.target = np.sin(2.0 * np.pi * 1024.0 * np.arange(n) / n)

    def test_identity_has_zero_decomposed_error(self):
        result = band_decomposition(self.target, self.target.copy(), 1500.0)
        self.assertGreater(result['target_power'], 0.0)
        self.assertEqual(result['absolute_complex_error_power'], 0.0)
        self.assertEqual(result['magnitude_error_power'], 0.0)
        self.assertAlmostEqual(result['phase_contribution_power'], 0.0, delta=1e-12)
        self.assertAlmostEqual(result['decomposition_closure_error'], 0.0, delta=1e-12)

    def test_180_degree_flip_has_equal_magnitude_but_complex_phase_error(self):
        result = band_decomposition(self.target, -self.target, 1500.0)
        self.assertAlmostEqual(result['magnitude_error_power'], 0.0, places=14)
        self.assertGreater(result['absolute_complex_error_power'], 0.0)
        self.assertAlmostEqual(result['phase_contribution_power'],
                               result['absolute_complex_error_power'], places=13)
        self.assertAlmostEqual(result['relative_complex_nmse'], 4.0, places=12)

    def test_null_target_keeps_absolute_error_and_marks_phase_undefined(self):
        result = band_decomposition(np.zeros_like(self.target), self.target, 1500.0)
        self.assertEqual(result['target_power'], 0.0)
        self.assertIsNone(result['relative_complex_nmse'])
        self.assertFalse(result['target_phase_defined'])
        self.assertGreater(result['absolute_complex_error_power'], 0.0)
        self.assertGreater(result['magnitude_error_power'], 0.0)
        self.assertEqual(result['phase_contribution_power'], 0.0)


if __name__ == '__main__':
    unittest.main()
