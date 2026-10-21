import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from scripts import run_real_coherence_gate as gate


class FakeProcess:
    pid = 41937

    def __init__(self):
        self.statuses = [None, 0]
        self.terminated = False

    def poll(self):
        return self.statuses.pop(0) if self.statuses else 0

    def terminate(self):
        self.terminated = True
        self.statuses = [-15]

    def wait(self, timeout=None):
        return self.poll()

    def kill(self):
        self.terminated = True
        self.statuses = [-9]


class RealCoherenceGateTests(unittest.TestCase):
    def test_native_f32_parameters_and_older_receipt_fields(self):
        cond = {"steps": 8, "strength": 0.6, "mid_cutoff": 1500,
                "mid_ceiling": 8000, "chunk_seconds": 10.0, "overlap_seconds": 2.0}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            gate._write_float_wav(out/"restored.wav", np.zeros((16, 2), np.float32))
            meta = {"steps": 8, "strength": float(np.float32(0.6)),
                    "mid_cutoff": 1500.0, "mid_ceiling": 8000.0,
                    "controlled": False, "denoise": False, "auto_eq": False}
            job = {"name": "baseline857-plain-probe", "frames": 16,
                   "arm": "restart", "origin": 0, "quiet_abstention": False}
            (out/"restoration.json").write_text(json.dumps(meta))
            gate._check_render(job, out, cond)
            # Adjacent representable strengths remain a configuration mismatch.
            meta["strength"] = float(np.nextafter(np.float32(0.6), np.float32(1)))
            (out/"restoration.json").write_text(json.dumps(meta))
            with self.assertRaises(gate.GateStop):
                gate._check_render(job, out, cond)

    def test_parseval_dc_nyquist_and_odd_length(self):
        dc = np.ones(8, dtype=np.float64) * 0.25
        nyq = np.array([1, -1] * 4, dtype=np.float64) * 0.4
        odd = np.sin(2 * np.pi * np.arange( nine := 9) / nine)
        self.assertAlmostEqual(gate._one_sided_energy_rms(dc, gate.RATE), 0.25, places=12)
        self.assertAlmostEqual(gate._one_sided_energy_rms(nyq, gate.RATE), 0.4, places=12)
        self.assertAlmostEqual(gate._one_sided_energy_rms(odd, gate.RATE), 1 / np.sqrt(2), places=12)
        self.assertEqual(gate._one_sided_energy_rms(np.zeros(9), gate.RATE), 0.0)

    def test_protected_metric_uses_closed_exact_cutoff_bins(self):
        n = 4800
        t = np.arange(n, dtype=np.float64) / gate.RATE
        source = np.stack([0.2 * np.sin(2 * np.pi * 700 * t),
                           0.15 * np.sin(2 * np.pi * 1000 * t)]).astype(np.float32)
        error = np.stack([0.04 * np.sin(2 * np.pi * 1500 * t),
                          0.03 * np.sin(2 * np.pi * 8000 * t)]).astype(np.float32)
        measured = gate.protected_error_metrics(source, source + error)
        expected = max(0.04, 0.03) / np.sqrt(2)
        self.assertAlmostEqual(measured["error_rms"]["left"], 0.04 / np.sqrt(2), places=6)
        self.assertAlmostEqual(measured["error_rms"]["right"], 0.03 / np.sqrt(2), places=6)
        self.assertAlmostEqual(measured["max_error_rms"], expected, places=6)

    def test_layout_alignment_uses_same_absolute_source_samples(self):
        source = np.stack([np.arange(500), np.arange(500) * -2.0]).astype(np.float32)
        view0, view1 = source[:, :300], source[:, 100:400]
        self.assertEqual(gate.aligned_layout_rms(view0, view1, [0, 100], 150, 250), 0.0)
        changed = view1.copy()
        changed[:, 50:150] += 1
        self.assertGreater(gate.aligned_layout_rms(view0, changed, [0, 100], 150, 250), 0.0)

    def test_residual_growth_and_undefined_zero_baseline(self):
        src = np.zeros((2, 32), dtype=np.float32)
        restart = src.copy()
        candidate = src.copy()
        restart[0] = 0.1
        candidate[0] = 0.105
        result = gate.residual_growth(restart, candidate, src)
        self.assertTrue(result["defined"])
        self.assertAlmostEqual(result["growth_fraction"], 0.05, places=6)
        undefined = gate.residual_growth(src, src, src)
        self.assertFalse(undefined["defined"])
        self.assertIsNone(undefined["growth_fraction"])

    def test_tail_chunk_geometry_and_boundary_residual_receipt_are_descriptive(self):
        self.assertEqual(gate._expected_chunk_starts(979_200), [0, 384_000, 499_200])
        restart = np.zeros((2, 100), dtype=np.float32)
        candidate = restart.copy()
        candidate[:, 41] = 0.1
        candidate[:, 78] = -0.2
        rows = gate.first_boundary_residual_differences(restart, candidate, 5_760_000,
                                                        [40, 80], radius_samples=5)
        self.assertEqual(rows[0]["first_differing_residual_sample_local"], 41)
        self.assertTrue(all(row["descriptive_only"] for row in rows))
        self.assertIsNone(rows[0].get("pass"))

    def test_protocol_tail_and_absolute_origin_are_pinned(self):
        protocol = json.loads(gate.PROTOCOL_PATH.read_text(encoding="utf-8"))
        fixed = gate._validate_protocol(protocol)
        self.assertEqual(fixed["tail"]["source_start_sample"], 5_760_000)
        self.assertEqual(fixed["tail"]["local_chunk_starts_samples"], [0, 384_000, 499_200])
        self.assertEqual(fixed["views"]["source_start_sample"] + fixed["views"]["view_starts_samples"][1],
                         2_448_384)

    def test_fixed_command_set_has_absolute_view_and_tail_origins(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            candidate, baseline, model = root / "candidate", root / "baseline", root / "model"
            candidate.write_bytes(b"candidate")
            baseline.write_bytes(b"baseline")
            model.write_bytes(b"model")
            inputs = {name: {"path": str(root / f"{name}.wav"), "sha256": "0" * 64}
                      for name in ("view0", "view1", "tail120to140_4", "source_probe_1s", "true_silence_16384")}
            protocol = {
                "conditioning": {"mid_cutoff": 1500, "mid_ceiling": 8000, "steps": 8,
                    "strength": 0.6, "chunk_seconds": 10, "overlap_seconds": 2},
                "views": {"source_start_sample": 2_304_000, "view_frames": 1_248_000,
                    "view_starts_samples": [0, 144_384]},
                "tail_probe": {"source_start_sample": 5_760_000, "frames": 979_200},
            }
            plan = {"candidate_binary": {"path": str(candidate)},
                    "baseline_binary": {"path": str(baseline)}, "model": {"path": str(model)}}
            jobs = gate._commands(plan, protocol, inputs, root)
            self.assertEqual(len(jobs), 11)
            view1 = next(j for j in jobs if j["name"] == "view1-continuous_with_final_trusted_band_lock")
            tail = next(j for j in jobs if j["name"] == "tail120-continuous_with_final_trusted_band_lock")
            self.assertEqual(view1["origin"], 2_448_384)
            self.assertEqual(tail["origin"], 5_760_000)
            argv = gate._child_command(plan, protocol, view1, root / "render")
            self.assertIn("--quiet-abstention", argv)
            self.assertIn("--final-band-lock", argv)
            self.assertEqual(argv[argv.index("--input-origin-samples") + 1], "2448384")
            restart = next(j for j in jobs if j["name"] == "view1-restart")
            restart_argv = gate._child_command(plan, protocol, restart, root / "restart")
            self.assertNotIn("--continuous-excitation", restart_argv)
            self.assertNotIn("--final-band-lock", restart_argv)

    def test_derived_float_wav_roundtrip_fingerprint(self):
        with tempfile.TemporaryDirectory() as tmp:
            samples = np.stack([np.linspace(-0.2, 0.2, 17, dtype=np.float32),
                                np.linspace(0.3, -0.1, 17, dtype=np.float32)])
            row = gate._write_derived(Path(tmp), "tiny", samples, 1234)
            restored = gate._read_stereo_float(Path(row["path"]), 17)
            self.assertTrue(np.array_equal(restored.view(np.uint32), samples.view(np.uint32)))
            self.assertEqual(row["sample_rate_hz"], gate.RATE)

    def test_fresh_root_and_halt_stop_are_non_destructive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "new-run"
            gate._fresh_root(root)
            self.assertFalse(root.exists())
            root.mkdir()
            (root / "keep.bin").write_bytes(b"evidence")
            with self.assertRaises(FileExistsError):
                gate._fresh_root(root)
            self.assertEqual((root / "keep.bin").read_bytes(), b"evidence")

            halted = Path(tmp) / "halted-run"
            halted.mkdir()
            (halted / "HALT").write_text("stop")
            fake = FakeProcess()
            log = halted / "child.log"

            def trip_halt(_pid):
                (halted / "HALT").write_text("stop")
                return {"VmRSS": 1024, "VmHWM": 2048}

            with mock.patch.object(gate.subprocess, "Popen", return_value=fake) as popen, \
                    mock.patch.object(gate, "memory", side_effect=trip_halt), \
                    mock.patch.object(gate.time, "sleep", return_value=None):
                with self.assertRaises(gate.GateStop):
                    gate._run_child("mock", ["fake-bin"], log, halted, [], timeout=1)
            popen.assert_not_called()  # HALT is checked before any child launch.
            self.assertTrue((halted / "HALT").exists())
            self.assertFalse(fake.terminated)

    def test_child_monitor_terminates_mock_child_on_live_halt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake = FakeProcess()
            def set_halt(_pid):
                (root / "HALT").write_text("stop")
                return {"VmRSS": 1024, "VmHWM": 2048}
            with mock.patch.object(gate.subprocess, "Popen", return_value=fake), \
                    mock.patch.object(gate, "memory", side_effect=set_halt), \
                    mock.patch.object(gate.time, "sleep", return_value=None):
                with self.assertRaises(gate.GateStop):
                    gate._run_child("mock", ["fake-bin"], root / "child.log", root, [], timeout=1)
            self.assertTrue(fake.terminated)
            stopped = json.loads((root / "child.stopped.json").read_text())
            self.assertIn("HALT", stopped["reason"])
            self.assertTrue((root / "child.log").exists())

    def test_halted_latch_preserves_reason_and_partial_child_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "halted"
            root.mkdir()
            partial = [{"name": "view0-restart", "log_sha256": "a" * 64}]
            gate._latch(root, "test stop", partial)
            marker = json.loads((root / "HALTED.json").read_text())
            self.assertEqual(marker["reason"], "test stop")
            self.assertEqual(marker["completed_children"], ["view0-restart"])
            gate._latch(root, "second reason must not overwrite", [])
            self.assertEqual(json.loads((root / "HALTED.json").read_text())["reason"], "test stop")


if __name__ == "__main__":
    unittest.main()
