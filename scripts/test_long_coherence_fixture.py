import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.long_coherence_fixture import (
    DEFAULT_EVENTS, DEFAULT_SEED, RATE, _hash_bytes, _lowpass_once, _render_source, prepare,
)
from scripts.roach_ears.audio_io import read_wav


def _small(out):
    return prepare(out, seed=881000, duration_seconds=0.24,
                   view_specs=[("view0", 0.0, 0.18), ("view3", 0.03, 0.18)],
                   event_times=[0.08, 0.16])


class LongCoherenceFixtureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_fixture_is_deterministic(self):
        a = _small(self.root / "a")
        b = _small(self.root / "b")
        self.assertEqual(a["source_sha256"], b["source_sha256"])
        for name, digest in a["source_sha256"].items():
            path = self.root / "a" / (name.replace("_", "-") + ".wav")
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)

    def test_full_mode_default_and_explicit_mode_preserve_all_audio_hashes(self):
        implicit = prepare(self.root / "implicit", seed=DEFAULT_SEED, duration_seconds=0.24,
                           view_specs=[("view0", 0.0, 0.18)], event_times=[0.08])
        explicit = prepare(self.root / "explicit", seed=DEFAULT_SEED, duration_seconds=0.24,
                           view_specs=[("view0", 0.0, 0.18)], event_times=[0.08],
                           source_mode="full")
        self.assertEqual(implicit["source_sha256"], explicit["source_sha256"])
        self.assertEqual(implicit["source_mode"], "full")
        self.assertEqual(implicit["recipe"]["damage"]["applications"], 1)

    def test_views_are_bit_identical_crops_of_same_physical_source(self):
        root = self.root / "fixture"
        manifest = _small(root)
        target = read_wav(root / "target-full.wav").samples.T
        degraded = read_wav(root / "input-full.wav").samples.T
        for view in manifest["views"]:
            start, count = view["origin_sample"], view["samples"]
            actual_target = read_wav(root / view["target"]).samples.T
            actual_input = read_wav(root / view["input"]).samples.T
            self.assertTrue(np.array_equal(actual_target, target[start:start + count]))
            self.assertTrue(np.array_equal(actual_input, degraded[start:start + count]))
            for event in view["events"]:
                self.assertEqual(event["view_sample"], event["sample"] - start)

    def test_damage_was_applied_once_before_cropping(self):
        root = self.root / "fixture"
        _small(root)
        full = read_wav(root / "target-full.wav").samples.T.astype(np.float64)
        globally_damaged = _lowpass_once(full, 1500.0, 225.0, 2.0).astype(np.float32)
        view = read_wav(root / "view3-input.wav").samples.T
        start = 3 * RATE // 100
        self.assertTrue(np.array_equal(view, globally_damaged[start:start + len(view)]))
        crop_damaged = _lowpass_once(full[start:start + len(view)], 1500.0, 225.0, 2.0).astype(np.float32)
        self.assertGreater(float(np.max(np.abs(view - crop_damaged))), 1e-7)

    def test_source_is_finite_stereo_and_peak_bounded(self):
        source, events = _render_source(int(0.24 * RATE), 881000, [0.08])
        self.assertEqual(source.shape, (int(0.24 * RATE), 2))
        self.assertTrue(np.isfinite(source).all())
        self.assertFalse(np.array_equal(source[:, 0], source[:, 1]))
        self.assertLessEqual(float(np.max(np.abs(source))), 0.620001)
        self.assertEqual(events[0]["sample"], int(round(0.08 * RATE)))

    def test_pitch_varies_source_without_changing_default_recipe(self):
        old, _ = _render_source(8192, 881000, [])
        explicit, _ = _render_source(8192, 881000, [], base_hz=146.8)
        higher, _ = _render_source(8192, 881000, [], base_hz=196.0)
        self.assertTrue(np.array_equal(old.view(np.uint32), explicit.view(np.uint32)))
        self.assertGreater(float(np.sqrt(np.mean((higher - old) ** 2))), 0.01)
        for frequency in (float('nan'), 0.0, 1001.0):
            out = self.root / 'invalid-pitch'
            with self.assertRaises(ValueError):
                prepare(out, duration_seconds=0.1, base_hz=frequency)
            self.assertFalse(out.exists())

    def test_silence_wav_roundtrip(self):
        from scripts.long_coherence_fixture import _write_float_wav

        silence = np.zeros((127, 2), dtype=np.float32)
        path = self.root / "silence.wav"
        _write_float_wav(path, silence)
        decoded = read_wav(path)
        self.assertEqual(decoded.samples.shape, (2, 127))
        self.assertTrue(np.isfinite(decoded.samples).all())
        self.assertFalse(np.any(decoded.samples))

    def test_stopped_mid_is_identity_input_with_only_float_quantization_in_missing_band(self):
        duration = 2.0
        seed = DEFAULT_SEED
        root = self.root / "stopped"
        manifest = prepare(root, seed=seed, duration_seconds=duration, source_mode="stopped_mid")
        target = read_wav(root / "target-full.wav").samples.T
        input_audio = read_wav(root / "input-full.wav").samples.T
        self.assertEqual(target.shape, (int(duration * RATE), 2))
        self.assertTrue(np.isfinite(target).all())
        self.assertTrue(np.array_equal(target.view(np.uint32), input_audio.view(np.uint32)))

        parent, _ = _render_source(len(target), seed, DEFAULT_EVENTS)
        expected = _lowpass_once(parent.astype(np.float64), 1200.0, 100.0, 3.0).astype(np.float32)
        self.assertTrue(np.array_equal(target.view(np.uint32), expected.view(np.uint32)))

        spectrum = np.fft.rfft(target.astype(np.float64), axis=0) / target.shape[0]
        frequencies = np.fft.rfftfreq(target.shape[0], 1.0 / RATE)
        mask = (frequencies > 1500.0) & (frequencies < 8000.0)
        weights = np.full(int(target.shape[0] // 2) + 1, 2.0)
        weights[0] = 1.0
        if target.shape[0] % 2 == 0:
            weights[-1] = 1.0
        band_power_by_channel = np.sum(weights[mask, None] * np.abs(spectrum[mask]) ** 2, axis=0)
        self.assertLess(float(np.max(band_power_by_channel)), 1e-14)
        self.assertEqual(manifest["source_mode"], "stopped_mid")
        self.assertEqual(manifest["recipe"]["damage"], {
            "type": "identity", "applications": 0,
            "note": "input is an exact copy of the stopped-mid target; no manufactured degradation",
        })
        construction = manifest["source_construction"]
        self.assertEqual(construction["parent_recipe"]["parent_source_f32_sha256"],
                         _hash_bytes(parent.tobytes(order="C")))
        self.assertFalse(construction["parent_recipe"]["independent_stems"])
        self.assertFalse(construction["transform"]["renormalization_after_transform"])

    def test_invalid_source_mode_fails_before_creating_output(self):
        out = self.root / "invalid-mode"
        with self.assertRaisesRegex(ValueError, "source_mode must be"):
            prepare(out, duration_seconds=0.24, source_mode="invented")
        self.assertFalse(out.exists())

    def test_invalid_view_scope_and_duration_fail_before_output(self):
        for identifier in ("../escape", "bad/name"):
            out = self.root / "invalid"
            with self.assertRaises(ValueError):
                prepare(out, duration_seconds=0.24, view_specs=[(identifier, 0, 0.1)])
            self.assertFalse(out.exists())
        with self.assertRaises(ValueError):
            prepare(self.root / "too-long", duration_seconds=49)
        self.assertFalse((self.root / "too-long").exists())

    def test_direct_cli_import_and_float_bits(self):
        import os
        import subprocess
        from scripts.long_coherence_fixture import _write_float_wav

        out = self.root / "cli"
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run([os.sys.executable, str(Path(__file__).with_name("long_coherence_fixture.py")),
                                 "--out", str(out), "--duration-seconds", "0.01"],
                                env=env, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        samples = np.array([[0.0, -0.0], [0.125, -0.25]], dtype=np.float32)
        path = self.root / "bits.wav"
        _write_float_wav(path, samples)
        self.assertTrue(np.array_equal(read_wav(path).samples.T.view(np.uint32), samples.view(np.uint32)))


if __name__ == "__main__":
    unittest.main()
