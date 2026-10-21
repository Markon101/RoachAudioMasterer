"""Known-answer controls for source-selected attack comparisons."""
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from scripts import matched_attack_audit as audit


RATE = 48000
DESIGN_PATH = Path(__file__).resolve().parents[1] / "artifacts/matched-attack-20261021/design.json"
V2_DESIGN_PATH = DESIGN_PATH.parent.parent / "matched-attack-paired-20261021/design.json"


def write_wav(path, audio, extensible=False):
    data = np.asarray(audio, dtype="<f4").tobytes()
    if extensible:
        fmt = struct.pack("<HHIIHHHHI", 0xFFFE, 2, RATE, RATE * 8, 8, 32, 22, 32, 3)
        fmt += struct.pack("<I", 3) + bytes.fromhex("00001000800000aa00389b71")
    else:
        fmt = struct.pack("<HHIIHH", 3, 2, RATE, RATE * 8, 8, 32)
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", len(data)) + data
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


class ArrayReader:
    def __init__(self, audio):
        self.audio = np.asarray(audio, dtype=np.float64)
        self.rate = RATE
        self.channels = self.audio.shape[1]
        self.frames = len(self.audio)

    def read(self, start, end):
        if not 0 <= start < end <= self.frames:
            raise ValueError("array test boundary failure")
        return self.audio[start:end].copy()


class MatchedAttackTests(unittest.TestCase):
    def setUp(self):
        self.design = json.loads(DESIGN_PATH.read_text())
        # A 1 kHz carrier has an integer number of cycles in every analysis
        # window. Opposite-polarity channels explicitly test no mono cancellation.
        t = np.arange(RATE, dtype=np.float64) / RATE
        tone = np.sin(2 * np.pi * 1000 * t)
        self.audio = np.column_stack((0.1 * tone, -0.1 * tone))
        self.onset = RATE // 2
        self.audio[self.onset:self.onset + 960] *= 4
        self.event = {"event_id": "E0001", "onset_sample": self.onset,
                      "onset_seconds": 0.5, "rise_db": 20 * math.log10(4)}
        self.filters = {name: None if band is None else audit.make_bandpass(RATE, band, 4097)
                        for name, band in self.design["bands_hz"].items()}

    def paired(self, control, candidate):
        observations = {name: audit.measure_event(ArrayReader(audio), self.onset, self.design, self.filters)
                        for name, audio in (("source", self.audio), ("control", control), ("candidate", candidate))}
        return audit.pair_event(self.event, observations, 1e-9, self.design)

    def test_identity_all_bands_and_windows_have_exact_zero_paired_delta(self):
        row = self.paired(self.audio, self.audio)
        self.assertEqual(row["windows_samples"]["attack"], [23760, 25680])
        for band in row["bands"].values():
            self.assertTrue(all(value == 0 for value in band["candidate_minus_control_db"].values() if value is not None))
            self.assertEqual(band["source"]["windows"], band["candidate"]["windows"])

    def test_independent_positive_gains_cancel_in_ratio_and_crest_differences(self):
        row = self.paired(2 * self.audio, 0.25 * self.audio)
        expected_level_delta = 20 * math.log10(0.25 / 2)
        for band in row["bands"].values():
            for metric in audit.RATIO_WINDOWS:
                value = band["candidate_minus_control_db"][metric]
                if value is not None:
                    self.assertAlmostEqual(value, 0, places=9)
            for window in ("pre", "attack", "sustain"):
                self.assertAlmostEqual(band["candidate_minus_control_db"][f"{window}_crest_db"], 0, places=9)
                self.assertAlmostEqual(band["candidate_minus_control_db"][f"{window}_rms_dbfs"], expected_level_delta, places=9)

    def test_known_half_gain_attack_only_reduces_fullband_contrast_by_six_db(self):
        candidate = self.audio.copy()
        lo = self.onset - audit.ms_samples(5, RATE)
        hi = self.onset + audit.ms_samples(35, RATE)
        candidate[lo:hi] *= 0.5
        row = self.paired(self.audio, candidate)
        delta = row["bands"]["fullband"]["candidate_minus_control_db"]
        expected = 20 * math.log10(0.5)
        for metric in ("attack_rms_dbfs", "attack_peak_dbfs", "attack_to_sustain_rms_db",
                       "attack_to_sustain_peak_db", "attack_peak_to_sustain_rms_db", "attack_to_pre_rms_db"):
            self.assertAlmostEqual(delta[metric], expected, places=11)
        self.assertAlmostEqual(delta["attack_crest_db"], 0, places=11)
        self.assertAlmostEqual(delta["sustain_rms_dbfs"], 0, places=11)
        self.assertAlmostEqual(delta["pre_rms_dbfs"], 0, places=11)

    def test_source_detector_finds_known_jump_despite_antiphase_channels(self):
        blocks = audit.input_block_rms(ArrayReader(self.audio), 240)
        selection = audit.select_events(blocks, RATE, RATE, self.design["detector"],
                                        self.design["windows_ms"], 2048)
        self.assertEqual([row["onset_sample"] for row in selection["accepted"]], [self.onset])
        self.assertAlmostEqual(selection["accepted"][0]["rise_db"], 20 * math.log10(4), places=10)

    def test_source_boundary_and_stronger_event_exclusions_are_recorded(self):
        blocks = np.full(200, 0.01)
        blocks[[10, 40, 50, 188]] = 0.1
        selection = audit.select_events(blocks, RATE, RATE, self.design["detector"],
                                        self.design["windows_ms"], 2048)
        self.assertEqual([row["frame_index"] for row in selection["accepted"]], [40])
        statuses = {row["frame_index"]: row["status"] for row in selection["candidates"]}
        self.assertEqual(statuses[10], "insufficient_window_or_FIR_boundary_support")
        self.assertEqual(statuses[188], "insufficient_window_or_FIR_boundary_support")
        self.assertEqual(statuses[50], "suppressed_by_stronger_source_event")
        self.assertEqual(selection["exclusion_counts"]["suppressed_by_stronger_source_event"], 1)

    def test_shared_source_sustain_null_prevents_output_driven_ratio_selection(self):
        source_windows = {name: audit.window_metrics(np.ones((20, 2))) for name in ("pre", "attack", "sustain")}
        source_windows["sustain"] = audit.window_metrics(np.zeros((20, 2)))
        output_windows = {name: audit.window_metrics(np.full((20, 2), 0.5)) for name in ("pre", "attack", "sustain")}
        values, reasons = audit.ratios(output_windows, source_windows, 1e-6)
        self.assertIsNone(values["attack_to_sustain_rms_db"])
        self.assertEqual(reasons["attack_to_sustain_rms_db"], "shared_source_window_below_activity_floor")
        self.assertEqual(values["attack_to_pre_rms_db"], 0)

    def test_output_exact_zero_is_null_with_reason_without_event_removal(self):
        row = self.paired(self.audio, np.zeros_like(self.audio))
        self.assertEqual(row["event_id"], "E0001")
        band = row["bands"]["fullband"]
        self.assertIsNone(band["candidate_minus_control_db"]["attack_to_sustain_rms_db"])
        self.assertEqual(band["candidate"]["ratio_null_reasons"]["attack_to_sustain_rms_db"],
                         "output_or_source_exact_zero_metric")

    def test_linear_fir_is_centered_and_has_no_circular_wrap(self):
        audio = np.zeros((RATE, 2))
        audio[self.onset] = (1, -1)
        coefficients = self.filters["bass_30_200"]
        y = audit.filtered_patch(ArrayReader(audio), self.onset - 2500, self.onset + 2501, coefficients)
        np.testing.assert_allclose(y[452:4549, 0], coefficients, rtol=0, atol=1e-17)
        self.assertLess(float(np.max(np.abs(y[:452]))), 1e-17)
        self.assertEqual(int(np.argmax(np.abs(y[:, 0]))), 2500)

    def test_alignment_identity_and_known_integer_delay(self):
        rng = np.random.default_rng(742019)
        reference = rng.normal(size=(1000, 2))
        for delay in (0, 7, -9):
            extended = np.zeros((1040, 2))
            extended[20 + delay:1020 + delay] = reference
            result = audit.normalized_lag_search(reference, extended, 20, 2)
            self.assertEqual(result["best_lag_samples"], delay)
            self.assertAlmostEqual(result["best_correlation"], 1, places=12)
        self.assertGreater(audit.normalized_lag_search(reference, np.pad(reference, ((20, 20), (0, 0))), 20, 2)
                           ["zero_peak_margin_outside_neighborhood"], 0.8)

    def test_wav_reader_extensible_and_simple_float_roundtrip_and_bound(self):
        with tempfile.TemporaryDirectory() as tmp:
            for extensible in (False, True):
                path = Path(tmp) / f"test-{extensible}.wav"
                write_wav(path, self.audio, extensible)
                reader = audit.WavReader(path)
                try:
                    self.assertEqual((reader.rate, reader.channels, reader.frames), (RATE, 2, RATE))
                    np.testing.assert_array_equal(reader.read(100, 1000), self.audio[100:1000].astype(np.float32))
                    with self.assertRaises(ValueError):
                        reader.read(-1, 100)
                    with self.assertRaises(ValueError):
                        audit.verify_geometry({"test": reader}, {"sample_rate": RATE, "channels": 2, "frames": RATE + 1})
                finally:
                    reader.close()

    def test_robust_summary_retains_signs_quantiles_and_nulls(self):
        result = audit.robust_summary([-3, -1, 0, 1, 3, None], 6)
        self.assertEqual((result["count"], result["null_count"]), (5, 1))
        self.assertEqual(result["median"], 0)
        self.assertEqual((result["negative_count"], result["positive_count"], result["near_zero_count_at_1e-9_db"]), (2, 2, 1))
        self.assertEqual(result["median_absolute_delta"], 1)

    def test_containment_latches_and_changed_pins_stop_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            audit.check_latches(output)
            for name in ("HALT", "HALT.json", "STOP", "STOP.json"):
                latch = output / name
                latch.write_text("stop")
                with self.assertRaisesRegex(RuntimeError, "stop latch"):
                    audit.check_latches(output)
                latch.unlink()
            pinned = output / "pin"
            pinned.write_text("original")
            pins = {str(pinned): audit.sha256(pinned)}
            audit.check_pins(pins)
            pinned.write_text("changed")
            with self.assertRaisesRegex(ValueError, "pin changed"):
                audit.check_pins(pins)

    def test_v2_preregistered_detector_windows_bands_fir_and_nulls_match_v1(self):
        v2 = json.loads(V2_DESIGN_PATH.read_text())
        for key in ("detector", "windows_ms", "window_semantics", "bands_hz", "filter"):
            self.assertEqual(v2[key], self.design[key])
        self.assertEqual(v2["metrics"]["null_policy"], self.design["metrics"]["null_policy"])
        self.assertEqual(v2["alignment"]["pairs"], ["control/candidate"])
        self.assertEqual(audit.comparison_scope(self.design), audit.V1_SCOPE)
        self.assertEqual(audit.comparison_scope(v2), audit.V2_SCOPE)
        with self.assertRaises(ValueError):
            audit.comparison_scope({"comparison_scope": "misspelled"})

    def test_v2_omits_source_relative_ratios_while_v1_default_is_preserved(self):
        observations = {name: audit.measure_event(ArrayReader(audio), self.onset, self.design, self.filters)
                        for name, audio in (("source", self.audio), ("control", self.audio),
                                            ("candidate", self.audio * 0.5))}
        v1_row = audit.pair_event(self.event, observations, 1e-9, self.design)
        v2 = copy.deepcopy(self.design)
        v2["comparison_scope"] = audit.V2_SCOPE
        v2_row = audit.pair_event(self.event, observations, 1e-9, v2)
        for band in self.design["bands_hz"]:
            self.assertIn("control_minus_source_ratios_db", v1_row["bands"][band])
            self.assertIn("ratios_db", v1_row["bands"][band]["source"])
            self.assertNotIn("control_minus_source_ratios_db", v2_row["bands"][band])
            self.assertNotIn("candidate_minus_source_ratios_db", v2_row["bands"][band])
            self.assertNotIn("ratios_db", v2_row["bands"][band]["source"])
            self.assertEqual(v1_row["bands"][band]["candidate_minus_control_db"],
                             v2_row["bands"][band]["candidate_minus_control_db"])

    def test_v2_presence_shift_is_retained_descriptively_and_not_excluded(self):
        v2 = json.loads(V2_DESIGN_PATH.read_text())
        rng = np.random.default_rng(160731)
        control = rng.normal(size=(RATE, 2))
        candidate = np.roll(control, 7, axis=0)
        readers = {"control": ArrayReader(control), "candidate": ArrayReader(candidate)}
        timing = audit.event_presence_timing(readers, self.event, v2, self.filters)
        self.assertTrue(timing["available"])
        self.assertEqual(timing["best_lag_samples"], 7)
        self.assertAlmostEqual(timing["best_correlation"], 1, places=10)
        self.assertEqual(timing["waveform_shift_applied_samples"], 0)
        self.assertTrue(timing["event_retained_regardless_of_timing"])
        observations = {name: audit.measure_event(ArrayReader(audio), self.onset, v2, self.filters)
                        for name, audio in (("source", self.audio), ("control", control), ("candidate", candidate))}
        row = audit.pair_event(self.event, observations, 1e-9, v2)
        row["event_local_presence_timing"] = timing
        summary = audit.summarize([row], v2)
        self.assertEqual(summary["all"]["paired_event_count"], 1)
        self.assertEqual(summary["all"]["event_local_presence_timing"]["best_lag_samples"]["nonzero_count"], 1)
        self.assertEqual(list(summary["all"]["bands"]["fullband"]), ["candidate_minus_control_db"])

    def test_v2_degenerate_timing_is_null_and_event_is_retained(self):
        v2 = json.loads(V2_DESIGN_PATH.read_text())
        zero = ArrayReader(np.zeros_like(self.audio))
        timing = audit.event_presence_timing({"control": zero, "candidate": zero}, self.event, v2, self.filters)
        self.assertFalse(timing["available"])
        self.assertIn("degenerate", timing["null_reason"])
        self.assertTrue(timing["event_retained_regardless_of_timing"])
        summary = audit.timing_summary([{"event_local_presence_timing": timing}])
        self.assertEqual(summary["event_count"], 1)
        self.assertEqual(summary["best_lag_samples"], {"count": 0, "null_count": 1})


if __name__ == "__main__":
    unittest.main()
