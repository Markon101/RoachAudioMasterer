"""Known-answer spectral and alignment controls; no model inference."""
import math
import unittest

import numpy as np

from scripts import analyze_long_coherence as audit


class CoherenceAnalysisTests(unittest.TestCase):
    def test_mid_band_gain_inversion_and_erasure(self):
        n = 32
        x = np.sin(2 * np.pi * 3 * np.arange(n) / n)
        half = audit.band_metrics(x, x * 0.5)['mid']
        self.assertAlmostEqual(half['reference_power'], 0.5)
        self.assertAlmostEqual(half['complex_nmse'], 0.25)
        self.assertAlmostEqual(half['target_weighted_lsd_db'], 20 * math.log10(2))
        self.assertAlmostEqual(half['phase_mae_radians'], 0)
        inverted = audit.band_metrics(x, -x)['mid']
        self.assertAlmostEqual(inverted['complex_nmse'], 4)
        self.assertAlmostEqual(inverted['phase_mae_radians'], math.pi)
        erased = audit.band_metrics(x, np.zeros(n))['mid']
        self.assertAlmostEqual(erased['complex_nmse'], 1)
        self.assertEqual(erased['phase_power_coverage'], 0)
        self.assertIsNone(erased['phase_mae_radians'])

    def test_parseval_dc_nyquist_and_closed_boundaries(self):
        n = 32
        t = np.arange(n)
        for x in (np.ones(n), (-1.0) ** t * 0.25,
                  np.sin(2 * np.pi * t / n), np.sin(2 * np.pi * 3 * t / n)):
            scores = audit.band_metrics(x, np.zeros(n))
            self.assertAlmostEqual(sum(s['reference_power'] for s in scores.values()), float(np.mean(x*x)))
            self.assertAlmostEqual(sum(s['absolute_error_power'] for s in scores.values()), float(np.mean(x*x)))
        cutoff = audit.band_metrics(np.sin(2 * np.pi * t / n), np.zeros(n))
        self.assertAlmostEqual(cutoff['low']['reference_power'], 0.5)
        self.assertLess(cutoff['mid']['reference_power'], 1e-28)
        odd = np.sin(2 * np.pi * 3 * np.arange(31) / 31)
        self.assertAlmostEqual(sum(s['reference_power'] for s in audit.band_metrics(odd, odd).values()), float(np.mean(odd*odd)))

    def test_null_target_keeps_absolute_error_and_stereo_swap(self):
        t = np.arange(256)
        x = np.stack([0.1 * np.sin(t * 0.31), 0.04 * np.cos(t * 0.21)])
        scores = audit.compare(np.zeros_like(x), x)
        self.assertIsNone(scores['left']['mid']['complex_nmse'])
        self.assertGreater(scores['left']['mid']['absolute_error_power'], 0)
        a, b = audit.compare(x, x), audit.compare(x[::-1], x[::-1])
        self.assertEqual(a['left'], b['right'])
        self.assertEqual(a['mid'], b['mid'])
        self.assertEqual(a['side'], b['side'])
        with self.assertRaises(ValueError):
            audit.band_metrics(np.zeros(5), np.zeros(6))


if __name__ == '__main__':
    unittest.main()
