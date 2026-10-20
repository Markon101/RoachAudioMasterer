"""Small deterministic controls for factorial-gate and alignment logic."""
import unittest
import tempfile
from pathlib import Path

import numpy as np

from scripts import analyze_coherence_factorial as audit
from scripts.long_coherence_fixture import _write_float_wav


class CoherenceFactorialTests(unittest.TestCase):
    def test_protected_gate_uses_absolute_rms_limit_and_reports_relative_ratio(self):
        comparison = {
            channel: {
                band: {"error_rms": (1.01e-6 if channel == "mid" and band == "low" else 2e-7),
                       "reference_power": 0.25}
                for band in ("low", "high")
            }
            for channel in ("left", "right", "mid", "side")
        }
        gate = audit._protected_gate(comparison, 1e-6)
        self.assertFalse(gate["pass"])
        self.assertAlmostEqual(gate["maximum_absolute_error_rms"], 1.01e-6)
        self.assertAlmostEqual(gate["channels"]["mid"]["low"]["relative_error_ratio"], 2.02e-6)
        self.assertTrue(gate["channels"]["left"]["low"]["pass"])

    def test_projection_only_layout_reduction_blocks_excitation_attribution(self):
        gates = {
            "projection_only_layout_rms_reduction_maximum_for_excitation_attribution": 0.10,
            "layout_rms_reduction_vs_restart_minimum": 0.50,
        }
        layouts = {
            "restart": {"rms": 1.0},
            "restart_projected": {"rms": 0.80},
            "continuous": {"rms": 0.30},
            "continuous_projected": {"rms": 0.25},
        }
        result = audit._factorial_layout_gates(layouts, gates)
        self.assertAlmostEqual(result["projection_only_layout_rms_reduction"], 0.20)
        self.assertAlmostEqual(result["continuous_layout_rms_reduction_vs_restart"], 0.70)
        self.assertTrue(result["continuous_layout_gate_pass"])
        self.assertFalse(result["projection_only_attribution_gate_pass"])
        self.assertFalse(result["excitation_attribution_supported_by_this_gate"])

    def test_exact_projection_identity_has_zero_waveform_change(self):
        x = np.array([[0.0, -0.0, 0.25, -0.5], [0.125, -0.25, 0.0, 0.75]], dtype=np.float32)
        result = audit.waveform_difference(x, x.copy())
        self.assertEqual(result["max_abs"], 0.0)
        self.assertEqual(result["rms"], 0.0)
        self.assertEqual(result["relative_rms"], 0.0)

    def test_manifest_crop_alignment_and_stereo_geometry(self):
        full = np.zeros((2, 80), dtype=np.float32)
        full[:, 36:64] = np.arange(56, dtype=np.float32).reshape(2, 28)
        view = full[:, 36:64].copy()
        audit._verify_crop(full, view, 36, "view")
        shifted = view.copy()
        shifted[0, 0] = 1.0
        with self.assertRaisesRegex(ValueError, "not an exact crop"):
            audit._verify_crop(full, shifted, 36, "view")
        self.assertEqual(audit.base.channels(view)["left"].shape, (28,))
        with self.assertRaisesRegex(ValueError, "stereo audio"):
            audit.base.channels(np.zeros((1, 28), dtype=np.float32))

    def test_real_float_wav_read_validates_finiteness_and_frame_alignment(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.wav"
            source = np.array([[0.0, -0.0], [0.125, -0.25], [0.5, 0.75]], dtype=np.float32)
            _write_float_wav(path, source)
            loaded = audit._audio(path, "fixture regression", 3)
            self.assertEqual(loaded.shape, (2, 3))
            self.assertTrue(np.isfinite(loaded).all())
            np.testing.assert_array_equal(loaded.astype(np.float32).T, source)
            with self.assertRaisesRegex(ValueError, "expected 4 frames, found 3"):
                audit._audio(path, "misaligned regression", 4)

            invalid_path = Path(directory) / "nonfinite.wav"
            invalid = source.copy()
            invalid[1, 1] = np.nan
            _write_float_wav(invalid_path, invalid)
            with self.assertRaisesRegex(ValueError, "finite nonempty stereo audio required"):
                audit._audio(invalid_path, "nonfinite regression", 3)


if __name__ == "__main__":
    unittest.main()
