"""Meter regressions and independent scalar-reference/seam controls."""
import math
import unittest

import numpy as np

from roach_ears.dsp_metrics import _true_peak_linear, compute_true_peak


class TruePeakTests(unittest.TestCase):
    def test_nyquist_does_not_double_samples(self):
        for n in [127, 128, 8192, 8193]:
            x = (.8 * (-1.) ** np.arange(n)).astype(np.float32)
            peak, overs = _true_peak_linear(x)
            self.assertAlmostEqual(peak, .8, delta=.002)
            self.assertEqual(overs, 0)

    def test_low_level_intersample_peak_and_gain_equivariance(self):
        x = np.cos(np.arange(256) * np.pi / 2 + np.pi / 4).astype(np.float32)
        low, _ = _true_peak_linear(.4 * x)
        high, _ = _true_peak_linear(.8 * x)
        self.assertAlmostEqual(low, .4, delta=.002)
        self.assertGreater(low, float(np.max(np.abs(.4 * x))) * 1.4)
        self.assertAlmostEqual(high, 2 * low, delta=2e-6)

    def test_scalar_reference_and_block_boundaries(self):
        rng = np.random.default_rng(38271)
        x = rng.uniform(-.95, .95, (2, 16417)).astype(np.float32)
        for i in [0, 11, 8191, 8192, 8193, x.shape[-1] - 1]:
            x[0, i] = 1.2
        # Independent scalar double-precision filter for a short subsequence.
        small = x[:, :80]
        peak = float(np.max(np.abs(small)))
        for channel in small:
            for start in range(len(channel) - 23):
                for frac in [.25, .5, .75]:
                    value = sum(float(channel[start+k]) *
                                (math.sin(math.pi*(k-11-frac))/(math.pi*(k-11-frac))) *
                                (.5+.5*math.cos(math.pi*(k-11-frac)/12)) for k in range(24))
                    peak = max(peak, abs(value))
        self.assertAlmostEqual(_true_peak_linear(small)[0], peak, delta=2e-6)
        reference = _true_peak_linear(x, block_size=x.shape[-1])
        for block in [7, 127, 8192]:
            self.assertEqual(_true_peak_linear(x, block_size=block), reference)

    def test_empty_silence_edges_and_validation(self):
        for x in [np.zeros((2, 0)), np.zeros((2, 256))]:
            self.assertEqual(compute_true_peak(x), (-120., 0))
        for n in [1, 23, 24, 25]:
            x = np.zeros(n, dtype=np.float32)
            x[-1] = 1.2
            peak, overs = _true_peak_linear(x)
            self.assertGreaterEqual(peak, 1.2)
            self.assertGreaterEqual(overs, 1)
        with self.assertRaises(ValueError):
            compute_true_peak(np.array([np.nan]))
        with self.assertRaises(ValueError):
            compute_true_peak(np.zeros(24), factor=2)

    def test_evaluated_point_count_includes_each_grid_point_once(self):
        for n in [0, 1, 23, 24, 25, 8192, 8193]:
            _, count = _true_peak_linear(np.full((2, n), 2., dtype=np.float32), block_size=127)
            self.assertEqual(count, 2 * (n + 3 * max(n - 23, 0)))


if __name__ == '__main__':
    unittest.main()
