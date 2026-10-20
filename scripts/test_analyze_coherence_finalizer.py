"""Small known-answer controls for finalizer excerpt statistics."""
import math
import unittest

import numpy as np

from scripts import analyze_coherence_finalizer as analysis


class FinalizerAnalysisTests(unittest.TestCase):
    def test_scalar_input_referred_control_only_attenuates_to_common_gain(self):
        x = np.array([[0.2, -0.1, 0.3, 0.4], [0.1, 0.2, -0.2, 0.3]], dtype=np.float64)
        y0, y3 = 2.0 * x, 0.5 * x
        result = analysis.input_referred_attenuation(y0, y3, x)
        self.assertTrue(result['available'])
        self.assertAlmostEqual(result['fitted_gain']['view0'], 2.0)
        self.assertAlmostEqual(result['fitted_gain']['view3'], 0.5)
        self.assertAlmostEqual(result['attenuation_factor']['view0'], 0.25)
        self.assertAlmostEqual(result['attenuation_factor']['view3'], 1.0)
        self.assertLessEqual(max(result['attenuation_factor'].values()), 1.0)
        self.assertAlmostEqual(result['layout_difference_after_attenuation']['rms'], 0.0)

    def test_fixed_onset_windows_report_attack_sustain_rms_peak_and_ratio(self):
        x = np.zeros((2, 100), dtype=np.float64)
        onset, rate = 50, 100
        x[0, 48:58] = 2.0
        x[0, 60:80] = 1.0
        x[1, 48:58] = -4.0
        x[1, 60:80] = -2.0
        result = analysis.event_window_metrics(x, onset, rate)
        self.assertEqual(result['attack_window_samples'], [48, 58])
        self.assertEqual(result['sustain_window_samples'], [60, 80])
        left = result['channels']['left']
        right = result['channels']['right']
        self.assertEqual(left['attack_peak'], 2.0)
        self.assertEqual(left['sustain_peak'], 1.0)
        self.assertAlmostEqual(left['attack_to_sustain_db'], 20 * math.log10(2))
        self.assertAlmostEqual(right['attack_to_sustain_db'], 20 * math.log10(2))

    def test_orthonormal_mid_side_stereo_metrics(self):
        x = np.array([[1.0, -1.0, 1.0, -1.0], [0.0, 0.0, 0.0, 0.0]])
        metrics = analysis.stereo_metrics(x)
        self.assertIsNone(metrics['left_right_correlation'])
        self.assertAlmostEqual(metrics['mid_rms'], math.sqrt(0.5))
        self.assertAlmostEqual(metrics['side_rms'], math.sqrt(0.5))


if __name__ == '__main__':
    unittest.main()
