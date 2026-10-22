"""Synthetic known-answer checks; never reads owner audio."""
import argparse
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts import analyze_master_pair as audit

RATE = 48000


class ArrayReader:
    def __init__(self, audio, rate=RATE):
        self.audio = np.asarray(audio, dtype=np.float64)
        self.rate, self.channels, self.frames = rate, self.audio.shape[1], len(self.audio)
        self.max_read = 0

    def read(self, start, end):
        if not 0 <= start < end <= self.frames or end - start > 65536:
            raise ValueError("invalid bounded read")
        self.max_read = max(self.max_read, end - start)
        return self.audio[start:end].copy()


def write_wav(path, audio):
    audio = np.asarray(audio, dtype="<f4")
    data = audio.tobytes()
    fmt = struct.pack("<HHIIHH", 3, 2, RATE, RATE * 8, 8, 32)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        t = np.arange(RATE * 2) / RATE
        self.audio = np.column_stack((.2 * np.sin(2 * np.pi * 1000 * t),
                                      .15 * np.sin(2 * np.pi * 3000 * t)))

    def analyze(self, master, source=None):
        readers = [ArrayReader(self.audio if source is None else source), ArrayReader(master)]
        result = audit.analyze(*readers)
        self.assertTrue(all(reader.max_read <= 65536 for reader in readers))
        json.dumps(result[0], allow_nan=False)
        return result

    def test_identity_exact_residual_and_spectral_zero(self):
        summary, raw = self.analyze(self.audio)
        self.assertAlmostEqual(summary["paired"]["best_nonnegative_global_gain"], 1., places=14)
        self.assertLess(summary["paired"]["gain_adjusted_residual_rms"], 1e-15)
        self.assertEqual(summary["paired"]["raw_difference_rms"], 0.)
        self.assertAlmostEqual(summary["paired"]["waveform_correlation_per_channel_dc_removed"], 1.)
        self.assertTrue(all(row["crest_delta_db"] == 0 for row in raw["transient_100ms"]))
        self.assertTrue(all(row["raw_energy_shift_db"] == 0 for row in summary["spectrum"]["bands"]))

    def test_uniform_gain_raw_level_changes_ratios_cancel(self):
        summary, raw = self.analyze(self.audio * 2)
        paired = summary["paired"]
        self.assertAlmostEqual(paired["best_nonnegative_global_gain"], 2.)
        self.assertLess(paired["gain_adjusted_residual_rms"], 1e-15)
        self.assertAlmostEqual(paired["global_rms_delta_db"], 20 * math.log10(2), places=10)
        self.assertAlmostEqual(paired["global_crest_delta_db"], 0., places=10)
        self.assertTrue(all(abs(row["crest_delta_db"]) < 1e-12 for row in raw["transient_100ms"]))
        for band in summary["spectrum"]["bands"]:
            self.assertAlmostEqual(band["raw_energy_shift_db"], 20 * math.log10(2), places=10)
            self.assertAlmostEqual(band["broadband_normalized_shift_db"], 0., places=10)

    def test_pulse_attenuation_is_visible_in_source_fixed_window(self):
        source = np.full((RATE * 2, 2), .01)
        pulse = RATE // 2 + 100
        source[pulse:pulse + 20] = .5
        master = source.copy()
        master[pulse:pulse + 20] *= .5
        summary, raw = self.analyze(master, source)
        row = raw["transient_100ms"][5]
        self.assertLess(row["crest_delta_db"], 0)
        self.assertAlmostEqual(row["master_sample_peak_dbfs"] - row["source_sample_peak_dbfs"], 20 * math.log10(.5))
        self.assertGreater(summary["paired"]["gain_adjusted_residual_rms"], 0)
        # Excerpt selection depends only on the source, even for severe output changes.
        quiet_summary, _ = self.analyze(source * .001, source)
        self.assertEqual(summary["excerpts"], quiet_summary["excerpts"])
        self.assertEqual(summary["excerpts"][0]["window_index"], 5)

    def test_antiphase_does_not_cancel_pooled_rms(self):
        source = np.column_stack((self.audio[:, 0], -self.audio[:, 0]))
        summary, _ = self.analyze(source, source)
        self.assertGreater(summary["source"]["rms"], .1)
        self.assertAlmostEqual(summary["source"]["lr_correlation"], -1.)
        self.assertEqual(summary["source"]["mid_mean_square"], 0.)
        self.assertEqual(summary["source"]["side_energy_fraction"], 1.)
        self.assertIsNone(summary["source"]["side_to_mid_db"])

    def test_channel_swap_preserves_pooled_spectrum_but_changes_waveform(self):
        summary, _ = self.analyze(self.audio[:, ::-1])
        self.assertAlmostEqual(summary["paired"]["global_rms_delta_db"], 0, places=11)
        self.assertLess(abs(summary["paired"]["waveform_correlation_per_channel_dc_removed"]), 1e-10)
        self.assertGreater(summary["paired"]["raw_difference_rms"], .1)
        self.assertEqual(summary["source"]["side_energy_fraction"], summary["master"]["side_energy_fraction"])
        for band in summary["spectrum"]["bands"]:
            self.assertAlmostEqual(band["raw_energy_shift_db"], 0, places=10)
            self.assertAlmostEqual(band["broadband_normalized_shift_db"], 0, places=10)
        # A vanishing least-squares projection must not cause a spectral shift.
        self.assertAlmostEqual(summary["spectrum"]["normalization_pooled_rms_gain_db"], 0, places=10)

    def test_welch_normalization_matches_stationary_tone_energy(self):
        summary, _ = self.analyze(self.audio)
        spectrum = summary["spectrum"]
        self.assertAlmostEqual(spectrum["source_broadband_mean_square"], summary["source"]["rms"] ** 2, places=8)
        self.assertEqual(spectrum["window_count"], 1 + (len(self.audio) - 4096) // 2048)

    def test_bass_ms_identity_gain_and_energy_partition(self):
        t = np.arange(RATE * 2) / RATE
        source = np.column_stack((.2 * np.sin(2 * np.pi * 46.875 * t),
                                  .1 * np.sin(2 * np.pi * 46.875 * t + .7)))
        for gain in (1., 2.):
            summary, _ = self.analyze(source * gain, source)
            bass = summary["spectrum"]["bass_20_80_ms"]
            self.assertEqual(bass["available_bin_count"], 5)
            self.assertEqual(bass["frequency_resolution_hz"], 11.71875)
            for label in ("source", "master"):
                energy = bass[label]
                self.assertAlmostEqual(energy["mid_mean_square"] + energy["side_mean_square"],
                                       energy["pooled_mean_square"], places=14)
            for key in ("mid_mean_square", "side_mean_square", "pooled_mean_square"):
                self.assertAlmostEqual(bass["master"][key], gain ** 2 * bass["source"][key], places=14)
            self.assertAlmostEqual(bass["master"]["side_energy_fraction"], bass["source"]["side_energy_fraction"], places=14)
            self.assertAlmostEqual(bass["master"]["side_to_mid_db"], bass["source"]["side_to_mid_db"], places=12)

    def test_bass_ms_antiphase_channel_swap_and_silence(self):
        t = np.arange(RATE * 2) / RATE
        tone = .2 * np.sin(2 * np.pi * 46.875 * t)
        source = np.column_stack((tone, -tone))
        summary, _ = self.analyze(source[:, ::-1], source)
        bass = summary["spectrum"]["bass_20_80_ms"]
        for label in ("source", "master"):
            self.assertEqual(bass[label]["mid_mean_square"], 0.)
            self.assertGreater(bass[label]["side_mean_square"], .01)
            self.assertEqual(bass[label]["side_energy_fraction"], 1.)
            self.assertIsNone(bass[label]["side_to_mid_db"])
        self.assertEqual(bass["source"], bass["master"])
        summary, _ = self.analyze(np.zeros_like(source), np.zeros_like(source))
        bass = summary["spectrum"]["bass_20_80_ms"]
        self.assertIsNone(bass["source"]["side_energy_fraction"])
        self.assertIsNone(bass["master"]["side_to_mid_db"])

    def test_silence_and_zero_master_nulls_are_explicit(self):
        silence = np.zeros_like(self.audio)
        summary, raw = self.analyze(silence, silence)
        for key in ("best_nonnegative_global_gain", "gain_adjusted_residual_rms", "waveform_correlation_per_channel_dc_removed"):
            self.assertIsNone(summary["paired"][key])
        self.assertIsNone(summary["source"]["crest_db"])
        self.assertIsNone(summary["source"]["lr_correlation"])
        self.assertEqual(summary["excerpts"], [])
        self.assertTrue(all(row["crest_delta_db"] is None for row in raw["transient_100ms"]))
        summary, _ = self.analyze(silence)
        self.assertEqual(summary["paired"]["best_nonnegative_global_gain"], 0)
        self.assertIsNone(summary["paired"]["gain_db"])
        self.assertIsNone(summary["paired"]["global_rms_delta_db"])

    def test_negative_gain_hits_nonnegative_boundary(self):
        summary, _ = self.analyze(-self.audio)
        self.assertEqual(summary["paired"]["best_nonnegative_global_gain"], 0)
        self.assertAlmostEqual(summary["paired"]["unconstrained_gain"], -1)
        self.assertAlmostEqual(summary["paired"]["waveform_correlation_per_channel_dc_removed"], -1)

    def test_dc_and_near_fullscale_counts(self):
        source = np.tile((.25, -.125), (RATE, 1))
        source[:3, 0] = 1.
        source[:5, 1] = -1.
        summary, _ = self.analyze(source, source)
        np.testing.assert_allclose(summary["source"]["dc_per_channel"], np.mean(source, axis=0), atol=1e-12)
        self.assertEqual(summary["source"]["near_fullscale_counts_per_channel"], [3, 5])

    def test_one_second_contrast_and_partial_exclusion(self):
        source = np.concatenate((self.audio[:RATE], self.audio[:RATE] * .1, self.audio[:123]))
        summary, raw = self.analyze(source, source)
        contrast = summary["paired"]["source_one_second_rms_dbfs_distribution"]
        self.assertEqual(contrast["valid_count"], 2)
        self.assertAlmostEqual(contrast["p90_minus_p10_db"], 16., places=10)
        self.assertEqual(len(raw["blocks_1s"]), 3)
        self.assertFalse(raw["blocks_1s"][-1]["complete"])
        self.assertEqual(raw["envelope_10ms"][-1]["end_sample"], len(source))

    def test_unequal_geometry_and_nonfinite_fail(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            self.analyze(self.audio[:-1])
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nan.wav"
            audio = self.audio[:1000].copy()
            audio[4, 0] = np.nan
            write_wav(path, audio)
            reader = audit.WavReader(path)
            try:
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    audit.analyze(reader, reader)
            finally:
                reader.close()

    def test_short_audio_has_no_fabricated_spectrum(self):
        summary, _ = self.analyze(self.audio[:1000], self.audio[:1000])
        self.assertEqual(summary["spectrum"]["window_count"], 0)
        self.assertTrue(all(b["raw_energy_shift_db"] is None for b in summary["spectrum"]["bands"]))

    def test_cli_artifact_create_only_stop_latch_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "source.wav"
            write_wav(path, self.audio[:5000])
            args = argparse.Namespace(source=path, master=path, out=tmp / "report", compare_summary=None,
                                      prepare_source=False, excerpt_plan=None)
            audit.run(args)
            self.assertTrue((args.out / "summary.json").is_file())
            self.assertIn("<svg", (args.out / "report.html").read_text())
            self.assertTrue((args.out / "envelope_10ms.csv").is_file())
            with self.assertRaises(FileExistsError):
                audit.run(args)
            (args.out / "HALT").touch()
            with self.assertRaisesRegex(RuntimeError, "stop latch"):
                audit.run(args)

    def test_source_only_preparation_and_frozen_plan_verification(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            path = tmp / "source.wav"
            write_wav(path, self.audio[:10000])
            args = argparse.Namespace(source=path, master=tmp / "DOES_NOT_EXIST.wav", out=tmp / "plan",
                                      compare_summary=None, prepare_source=True, excerpt_plan=None)
            audit.run(args)
            plan = args.out / "excerpts.json"
            self.assertTrue(plan.is_file())
            args.prepare_source, args.master, args.out, args.excerpt_plan = False, path, tmp / "analysis", plan
            audit.run(args)
            self.assertIn("excerpt_plan", json.loads((args.out / "summary.json").read_text()))
            bad_plan = json.loads(plan.read_text())
            bad_plan["source_sha256"] = "bad"
            wrong = tmp / "wrong.json"
            wrong.write_text(json.dumps(bad_plan))
            args.out, args.excerpt_plan = tmp / "bad-analysis", wrong
            with self.assertRaisesRegex(ValueError, "source excerpt plan"):
                audit.run(args)
            self.assertTrue((args.out / "FAILED.json").is_file())
            (tmp / "STOP").touch()
            args.out = tmp / "parent-stopped"
            with self.assertRaisesRegex(RuntimeError, "stop latch"):
                audit.run(args)

    def test_stop_latch_interrupts_stream_and_no_cross_version_waveform_pairing(self):
        calls = []
        def latch():
            calls.append(1)
            if len(calls) == 2:
                raise RuntimeError("stopped")
        with self.assertRaisesRegex(RuntimeError, "stopped"):
            audit.analyze(ArrayReader(self.audio), ArrayReader(self.audio), latch)
        one, _ = self.analyze(self.audio)
        two, _ = self.analyze(self.audio[:RATE], self.audio[:RATE])
        comparison = audit.compare_summaries(one, two)
        self.assertNotEqual(comparison["previous_geometry"]["frames"], comparison["current_geometry"]["frames"])
        self.assertIn("no sample correlation", comparison["scope"])


if __name__ == "__main__":
    unittest.main()
