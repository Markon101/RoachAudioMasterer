import unittest

import numpy as np

from scripts import analyze_stopped_coherence as audit


class StoppedCoherenceTests(unittest.TestCase):
    def test_identity_input_oracle_has_exact_zero_false_residual(self):
        x = np.array([[0.0, -0.0, 0.25, -0.5], [0.125, -0.25, 0.0, 0.75]], dtype=np.float32)
        residual = audit._false_residual(x, x.copy())
        self.assertEqual(residual["rms"], 0.0)
        self.assertEqual(residual["max_abs"], 0.0)
        self.assertEqual(residual["relative_rms_to_input"], 0.0)

    def test_growth_denominator_uses_protocol_power_null_floor(self):
        result = audit._growth(1e-8, 1e-4)
        self.assertEqual(result["null_power_floor"], 1e-14)
        self.assertEqual(result["null_rms_floor"], 1e-7)
        self.assertIsNone(result["ratio"])
        self.assertFalse(result["denominator_defined"])
        self.assertFalse(result["pass"])

    def test_false_residual_growth_fails_even_when_layout_improves(self):
        gates = {
            "continuous_projected_vs_restart_projected_false_residual_rms_growth_max": 0.10,
            "layout_rms_reduction_minimum_for_mechanistic_claim": 0.50,
        }
        result = audit._assess_pair(0.002, 0.0005, 0.001, 0.00111, gates)
        self.assertAlmostEqual(result["layout_rms_reduction"], 0.75)
        self.assertTrue(result["layout_pass"])
        self.assertAlmostEqual(result["false_residual_rms_growth"]["ratio"], 0.11)
        self.assertFalse(result["false_residual_rms_growth"]["pass"])


if __name__ == "__main__":
    unittest.main()
