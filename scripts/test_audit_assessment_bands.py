"""Known-answer tests for calibrated whole-clip Parseval band measurements."""
import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.audit_assessment_bands import audit, channel_band_metrics, channel_views
from scripts.long_coherence_fixture import _write_float_wav


class PhysicalBandObserverTests(unittest.TestCase):
    def test_stereo_sine_and_orthonormal_mid_side_rms(self):
        n = 48000
        tone = 0.2 * np.sin(2 * np.pi * 3000 * np.arange(n) / n)
        views = channel_views(np.stack([tone, tone]))
        left = channel_band_metrics(views['left'])
        mid = channel_band_metrics(views['mid_orthonormal'])
        side = channel_band_metrics(views['side_orthonormal'])
        self.assertAlmostEqual(left['waveform_rms_squared'], 0.2 ** 2 / 2, places=12)
        self.assertAlmostEqual(mid['waveform_rms_squared'], 0.2 ** 2, places=12)
        self.assertLess(side['waveform_rms_squared'], 1e-28)
        self.assertAlmostEqual(mid['bands']['mid_500_6000_hz']['power_mean_square'],
                               0.2 ** 2, places=12)
        self.assertTrue(mid['band_power_closure_pass_1e-10'])

    def test_dc_and_nyquist_bins_have_single_parseval_weight(self):
        n = 32
        x = np.ones(n) + 0.25 * (-1.0) ** np.arange(n)
        metrics = channel_band_metrics(x)
        self.assertAlmostEqual(metrics['bands']['dc']['power_mean_square'], 1.0, places=14)
        self.assertAlmostEqual(metrics['bands']['top_22000_24000_hz']['power_mean_square'],
                               0.25 ** 2, places=14)
        self.assertAlmostEqual(metrics['waveform_rms_squared'], 1.0625, places=14)
        self.assertTrue(metrics['band_power_closure_pass_1e-10'])

    def test_odd_length_has_no_nyquist_singleton_and_still_closes(self):
        n = 31
        x = 0.3 * np.sin(2 * np.pi * 5 * np.arange(n) / n)
        metrics = channel_band_metrics(x)
        self.assertAlmostEqual(metrics['waveform_rms_squared'], 0.3 ** 2 / 2, places=14)
        self.assertLess(abs(metrics['band_power_closure_error']), 1e-14)

    def test_silence_is_null_and_nonfinite_samples_are_rejected(self):
        metrics = channel_band_metrics(np.zeros(31))
        self.assertEqual(metrics['waveform_rms_squared'], 0.0)
        self.assertIsNone(metrics['waveform_rms_dbfs'])
        self.assertTrue(all(b['power_below_null_floor'] for b in metrics['bands'].values()))
        with self.assertRaises(ValueError):
            channel_band_metrics(np.array([0.0, math.nan]))

    def test_real_float_wav_geometry_hash_and_rms_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tone = (0.2 * np.sin(2*np.pi*1000*np.arange(4800)/48000)).astype(np.float32)
            path = root/'tone.wav'
            _write_float_wav(path, np.stack([tone, tone], axis=1))
            legacy = root/'assessment.json'
            legacy.write_text(json.dumps({'low_mid_energy_dbfs': 3.0}))
            result = audit(path, legacy)
            self.assertEqual(result['geometry']['frames'], 4800)
            self.assertEqual(result['provenance']['audio_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertAlmostEqual(result['channels']['left']['waveform_rms_squared'], 0.02, places=9)
            self.assertAlmostEqual(result['channels']['mid_orthonormal']['waveform_rms_squared'], 0.04, places=9)
            self.assertEqual(result['channels']['side_orthonormal']['waveform_rms_squared'], 0.0)
            self.assertIn('not authenticated', result['legacy_assessment']['input_binding'])


if __name__ == '__main__':
    unittest.main()
