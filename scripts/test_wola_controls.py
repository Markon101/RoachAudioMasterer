"""Known-answer tests for the source-mirrored WOLA control helpers."""
import importlib.util
import math
import unittest
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("check_wola_controls.py")
SPEC = importlib.util.spec_from_file_location("check_wola_controls", MODULE_PATH)
controls = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(controls)


class WolaControlTests(unittest.TestCase):
    def test_identity_and_stereo_are_preserved(self):
        x = np.stack([np.linspace(-1, 1, 1001), np.cos(np.arange(1001) * 0.07)])
        starts = controls.chunk_starts(1001, 400, 100)
        chunks = [x[:, s:min(s + 400, x.shape[-1])] for s in starts]
        y, norm = controls.normalized_wola(chunks, starts, x.shape[-1], 100)
        self.assertEqual(y.shape, x.shape)
        np.testing.assert_allclose(y, x, rtol=0, atol=2e-15)
        self.assertTrue(np.all(norm > 0))

    def test_identical_residual_is_preserved(self):
        residual = np.sin(np.arange(997) * 0.13)
        starts = controls.chunk_starts(997, 321, 81)
        chunks = [residual[s:min(s + 321, len(residual))] for s in starts]
        y, _ = controls.normalized_wola(chunks, starts, len(residual), 81)
        np.testing.assert_allclose(y, residual, rtol=0, atol=2e-15)

    def test_silence_and_empty_invalid(self):
        starts = controls.chunk_starts(100, 64, 16)
        y, norm = controls.normalized_wola([np.zeros(min(64, 100 - s)) for s in starts],
                                           starts, 100, 16)
        self.assertTrue(np.array_equal(y, np.zeros(100)))
        self.assertTrue(np.all(norm > 0))
        with self.assertRaises(ValueError):
            controls.normalized_wola([], [], 100, 16)

    def test_orthogonal_midpoint_has_half_variance(self):
        n = 4096
        a = np.linspace(-1.0, 1.0, n, endpoint=False)
        a -= a.mean()
        b = np.roll(a, n // 4)
        b -= b.mean()
        b -= a * (np.dot(a, b) / np.dot(a, a))
        b *= math.sqrt(np.dot(a, a) / np.dot(b, b))
        np.testing.assert_allclose(np.dot(a, b) / np.dot(a, a), 0.0, atol=1e-14)
        np.testing.assert_allclose(np.var(0.5 * (a + b)) / np.var(a), 0.5, atol=1e-14)
        self.assertAlmostEqual(10 * math.log10(0.5), -3.01029995664, places=10)

    def test_odd_terminal_anchor_and_non_cola_geometry(self):
        starts = controls.chunk_starts(1013, 401, 101)
        self.assertEqual(starts, [0, 300, 600, 612])
        # The final right-anchored chunk begins only 12 samples after its
        # predecessor, making its actual overlap 389 (not nominal 101).
        self.assertEqual(300 - 0, 300)
        self.assertEqual(600 - 300, 300)
        self.assertEqual(612 - 600, 12)
        self.assertEqual(401 - 12, 389)
        chunks = [np.ones(min(401, 1013 - s)) for s in starts]
        y, weights = controls.normalized_wola(chunks, starts, 1013, 101)
        np.testing.assert_allclose(y, 1.0, rtol=0, atol=2e-15)
        self.assertTrue(np.all(weights > 0))
        # Source applies fades by chunk index/nominal overlap, so the irregular
        # final overlap is normalized from the actual accumulated source weights.
        self.assertTrue(np.any(weights[612:701] > 1.0))

    def test_source_fades_are_sin_squared_cosine_squared(self):
        left = controls.source_weights(12, 4, 0, 2)
        right = controls.source_weights(12, 4, 1, 2)
        expected_left = np.ones(12)
        expected_right = np.ones(12)
        for j in range(4):
            theta = math.pi / 2 * (j / 4)
            expected_left[8 + j] = math.cos(theta) ** 2
            expected_right[j] = math.sin(theta) ** 2
        np.testing.assert_allclose(left, expected_left, rtol=0, atol=0)
        np.testing.assert_allclose(right, expected_right, rtol=0, atol=0)

    def test_per_chunk_projection_does_not_imply_whole_track_lock(self):
        rng = np.random.default_rng(77)
        rate, total, chunk, overlap = 24000, 12037, 4096, 1024
        starts = controls.chunk_starts(total, chunk, overlap)
        raw = rng.normal(size=total)
        locked = []
        for start in starts:
            segment = raw[start:min(start + chunk, total)]
            locked.append(controls.spectral_project(segment, rate, 1500, 8000))
        local_error = max(controls.forbidden_energy_ratio(x, rate, 1500, 8000)
                          for x in locked)
        stitched, _ = controls.normalized_wola(locked, starts, total, overlap)
        whole_error = controls.forbidden_energy_ratio(stitched, rate, 1500, 8000)
        globally_locked = controls.spectral_project(stitched, rate, 1500, 8000)
        repaired_error = controls.forbidden_energy_ratio(globally_locked, rate, 1500, 8000)
        self.assertLess(local_error, 1e-28)
        self.assertGreater(whole_error, 1e-5)
        self.assertLess(repaired_error, 1e-28)

    def test_projection_excludes_exact_cutoff_bins(self):
        rate, n = 24000, 2400  # 10 Hz bin spacing; both cutoffs land exactly on bins.
        t = np.arange(n) / rate
        for hz in (1500.0, 8000.0):
            tone = np.sin(2 * math.pi * hz * t)
            projected = controls.spectral_project(tone, rate, 1500.0, 8000.0)
            self.assertLess(np.max(np.abs(projected)), 2e-12)
        inside = np.sin(2 * math.pi * 2000.0 * t)
        np.testing.assert_allclose(controls.spectral_project(inside, rate, 1500, 8000),
                                   inside, rtol=0, atol=2e-12)

    def test_forbidden_ratio_uses_parseval_dc_and_nyquist_weights(self):
        n, rate = 8, 8000
        t = np.arange(n)
        x = 1.0 + 2.0 * (-1.0) ** t + np.sin(2 * math.pi * t / n)
        # DC power N^2, Nyquist power (2N)^2, and one-sided bin-1
        # power 2*(N/2)^2: forbidden ratio is (64+256)/(64+256+32).
        ratio = controls.forbidden_energy_ratio(x, rate, 0.0, rate / 2)
        self.assertAlmostEqual(ratio, 10.0 / 11.0, places=14)

        # Odd lengths have no Nyquist singleton; the highest positive bin is doubled.
        odd = np.sin(2 * math.pi * 3 * np.arange(7) / 7)
        self.assertAlmostEqual(controls.forbidden_energy_ratio(odd, 7000, 0, 3000),
                               1.0, places=14)


if __name__ == "__main__":
    unittest.main()
