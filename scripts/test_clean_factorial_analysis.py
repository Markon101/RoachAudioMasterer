"""Known-answer interaction and stop/parity gates for the bounded clean factorial."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

PATH = Path(__file__).resolve().parents[1] / 'artifacts/singularity-clean-factorial-20261021/run.py'
SPEC = importlib.util.spec_from_file_location('clean_factorial_analysis', PATH)
analysis = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(analysis)


class CleanFactorialTests(unittest.TestCase):
    def test_additive_log_ratios_have_zero_interaction(self):
        # Independent band/broadband gain ratios compose multiplicatively;
        # their dB differences therefore form an additive factorial.
        fields = ('signed_shape_change_from_source_db',
                  'filtered_band_rms_change_from_source_db',
                  'broadband_rms_change_from_source_db')
        rows = {name: dict(zip(fields, values)) for name, values in {
            'identity': (.25, 2., 1.75), 'guard': (.25, 1., .75),
            'denoise': (.75, 3., 2.25), 'eq': (1.25, 4., 2.75),
            'combined': (1.75, 5., 3.25)}.items()}
        self.assertEqual(analysis.factorial(rows, 'identity'), dict.fromkeys(fields, 0.))
        self.assertEqual(analysis.factorial(rows, 'guard'), dict(zip(fields, (0., -1., -1.))))

    def test_nonadditive_interaction_sign(self):
        key = 'signed_shape_change_from_source_db'
        other = ('filtered_band_rms_change_from_source_db', 'broadband_rms_change_from_source_db')
        rows = {name: {key: value, **dict.fromkeys(other, 0.)}
                for name, value in {'identity': .1, 'denoise': .3, 'eq': .6, 'combined': .5}.items()}
        self.assertAlmostEqual(analysis.factorial(rows, 'identity')[key], -.3)

    def test_combined_hash_mismatch_is_binding(self):
        analysis.require_combined('a' * 64, 'a' * 64)
        with self.assertRaisesRegex(RuntimeError, 'exactly reproduce'):
            analysis.require_combined('b' * 64, 'a' * 64)

    def test_all_local_stop_latches_preserved(self):
        for name in ('HALT', 'HALT.json', 'STOP', 'STOP.json', 'HALTED.json'):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                directory = Path(temporary)
                path = directory / name
                path.write_bytes(b'preserve')
                with self.assertRaises(RuntimeError):
                    analysis.check_latches(directory)
                self.assertEqual(path.read_bytes(), b'preserve')

    def test_create_only_receipts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'receipt.json'
            analysis.save(path, {'first': True})
            original = path.read_bytes()
            with self.assertRaises(FileExistsError):
                analysis.save(path, {})
            self.assertEqual(path.read_bytes(), original)


if __name__ == '__main__':
    unittest.main()
