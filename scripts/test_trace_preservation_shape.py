"""Bounded known answers and failure handling for the offline shape diagnostic."""
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import trace_preservation_shape as trace


class KnownAnswers(unittest.TestCase):
    def setUp(self):
        self.rate = 48000
        time = np.arange(4800, dtype=np.float64) / self.rate
        self.mid = np.sin(2 * np.pi * 1500 * time)
        self.low = np.sin(2 * np.pi * 300 * time)
        self.source = trace.measure_samples(self.mid + self.low, self.rate, 127)

    def test_identity(self):
        row = trace.compare(self.source, self.source, self.source)
        self.assertEqual(row['absolute_shape_deviation_from_source_db'], 0.)
        self.assertEqual(row['signed_increment_from_previous_boundary_db'], 0.)
        self.assertAlmostEqual(self.source['normalized_filtered_rms'], 1 / math.sqrt(2), places=13)

    def test_uniform_gain_cancels(self):
        measured = trace.measure_samples(3 * (self.mid + self.low), self.rate, 127)
        row = trace.compare(measured, self.source, self.source)
        self.assertAlmostEqual(row['signed_shape_change_from_source_db'], 0., places=12)
        for key in ('filtered_band_rms_change_from_source_db', 'broadband_rms_change_from_source_db'):
            self.assertAlmostEqual(row[key], 20 * math.log10(3), places=12)

    def test_mid_boost_known_answer_and_telescope(self):
        boosted = trace.measure_samples(2 * self.mid + self.low, self.rate, 127)
        row = trace.compare(boosted, self.source, self.source)
        expected = 20 * math.log10(2 * math.sqrt(2 / 5))
        self.assertAlmostEqual(row['signed_shape_change_from_source_db'], expected, places=12)
        self.assertFalse(row['analytical_below_unchanged_1db_threshold'])
        final = trace.measure_samples(3 * self.mid + self.low, self.rate, 127)
        last = trace.compare(final, self.source, boosted)
        self.assertAlmostEqual(row['signed_increment_from_previous_boundary_db'] +
                               last['signed_increment_from_previous_boundary_db'],
                               last['signed_shape_change_from_source_db'], places=12)

    def test_filter_skirts_and_support(self):
        measured = trace.combined_response(np.array([0., 700., 850., 1000., 1500., 2500., 2750., 3000., 24000.]))
        np.testing.assert_allclose(measured, [0., 0., .25, 1., 1., 1., .25, 0., 0.], atol=1e-15)

    def test_parseval_dc_nyquist_even_and_odd(self):
        for n in (63, 64):
            values = .25 + np.cos(2 * np.pi * np.arange(n) * (n // 2) / n)
            measured = trace.measure_samples(values, self.rate, 7)
            self.assertLess(measured['parseval_relative_closure_error'], 1e-14)
            self.assertAlmostEqual(measured['full_power'], float(np.mean(values**2)), places=14)

    def test_zero_and_subfloor_are_explicit(self):
        for values in (np.zeros(64), np.ones(64) * 1e-10):
            measured = trace.measure_samples(values, self.rate)
            self.assertTrue(measured['broadband_floor_active'])
            self.assertTrue(measured['filtered_band_floor_active'])
            self.assertEqual(measured['normalized_filtered_rms'], 1.)

    def test_invalid_samples_rejected(self):
        for values in ([], [[1., 2.]], [np.inf], [np.nan]):
            with self.assertRaises(ValueError):
                trace.measure_samples(values, self.rate)

    def test_stereo_reader_uses_first_channel_and_checks_geometry(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'test.wav'
            samples = np.stack((self.mid, 20 * self.low), axis=1).astype('<f4')
            data = samples.tobytes()
            fmt = struct.pack('<HHIIHH', 3, 2, self.rate, self.rate * 8, 8, 32)
            path.write_bytes(b'RIFF' + struct.pack('<I', 36 + len(data)) + b'WAVEfmt ' +
                             struct.pack('<I', 16) + fmt + b'data' + struct.pack('<I', len(data)) + data)
            geometry = {'rate': self.rate, 'channels': 2, 'frames': len(samples)}
            resources = {'max_frames': len(samples), 'input_block_frames': 64,
                         'spectrum_block_bins': 127, 'process_hwm_aim_mib': 512}
            measured = trace.measure_path(path, geometry, resources)
            self.assertAlmostEqual(measured['normalized_filtered_rms'], 1., places=12)
            with self.assertRaisesRegex(ValueError, 'geometry'):
                trace.measure_path(path, {**geometry, 'frames': len(samples) - 1}, resources)
            with self.assertRaisesRegex(ValueError, 'frame limit'):
                trace.measure_path(path, geometry, {**resources, 'max_frames': 1})


class FailureHandling(unittest.TestCase):
    def test_create_only_save_and_output(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            path = directory / 'kept.json'
            trace.save(path, {'original': True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                trace.save(path, {})
            self.assertEqual(path.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                trace.run_trace(directory / 'missing-plan.json', directory)

    def test_every_local_latch_is_honored_and_preserved(self):
        for latch in ('HALT', 'HALT.json', 'STOP', 'STOP.json'):
            with tempfile.TemporaryDirectory() as name:
                directory = Path(name)
                (directory / latch).write_text('preserve')
                with self.assertRaisesRegex(RuntimeError, 'stop latch'):
                    trace.run_trace(directory / 'plan.json', directory / 'new')
                self.assertEqual((directory / latch).read_text(), 'preserve')
                self.assertFalse((directory / 'new').exists())

    def test_changed_pin_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / 'input'
            path.write_bytes(b'first')
            expected = trace.digest(path)
            path.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'pin changed'):
                trace.check_pins({str(path): expected})

    def test_stop_arriving_during_measurement_is_preserved(self):
        with tempfile.TemporaryDirectory() as name:
            directory = Path(name)
            output = directory / 'new'
            plan_path = directory / 'plan.json'
            plan = {'schema': 'first-channel-preservation-shape-trace-v1', 'pins': {},
                    'resources': {'input_block_frames': 64, 'minimum_available_memory_mib': 0},
                    'geometry': {}, 'inputs': [{'path': 'unused'}]}
            plan_path.write_text(json.dumps(plan))
            def stop(*args):
                (output / 'STOP.json').write_text('{"external": true}\n')
                return {}
            with patch.object(trace, 'measure_path', side_effect=stop):
                with self.assertRaisesRegex(RuntimeError, 'stop latch'):
                    trace.run_trace(plan_path, output)
            self.assertEqual((output / 'STOP.json').read_text(), '{"external": true}\n')
            self.assertFalse((output / 'summary.json').exists())


if __name__ == '__main__':
    unittest.main()
