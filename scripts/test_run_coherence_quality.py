import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest import mock

from scripts import run_coherence_quality as runner


def sparse_float_wav(path: Path, frames: int, extensible: bool = False) -> None:
    data_bytes = frames * runner.CHANNELS * runner.SAMPLE_BYTES
    fmt = struct.pack("<HHIIHH", 3, runner.CHANNELS, runner.RATE,
                      runner.RATE * runner.CHANNELS * runner.SAMPLE_BYTES,
                      runner.CHANNELS * runner.SAMPLE_BYTES, 32)
    if extensible:
        fmt = struct.pack("<H", 0xFFFE) + fmt[2:]
        fmt += struct.pack("<HHI", 22, 32, 3) + bytes.fromhex("0300000000001000800000aa00389b71")
    body = b"WAVEfmt " + struct.pack("<I", len(fmt)) + fmt
    body += b"data" + struct.pack("<I", data_bytes)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body) + data_bytes) + body)
        f.truncate(8 + len(body) + data_bytes)


class FakeProcess:
    pid = 12345

    def __init__(self, returncode=0):
        self.returncode = returncode
        self.terminated = False

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.returncode = -9


class QualityRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root / "native-bin"
        self.model = self.root / "mid-model.json"
        self.source = self.root / "fixture-source.py"
        self.binary.write_bytes(b"native binary pin")
        self.model.write_bytes(b"model pin")
        self.source.write_bytes(b"fixture generator pin")
        self.jobs = []
        for expected in runner.EXPECTED:
            ident = expected["id"]
            inp = self.root / f"{ident}-input.wav"
            target = self.root / f"{ident}-target.wav"
            sparse_float_wav(inp, expected["samples"])
            sparse_float_wav(target, expected["samples"])
            self.jobs.append({**expected, "input": str(inp), "input_sha256": runner.digest(inp),
                              "target": str(target), "target_sha256": runner.digest(target)})
        self.plan_path = self.root / "quality-plan.json"
        self.run_root = self.root / "fresh-run"
        self.plan = {"schema": "long-coherence-quality-v1", "root": str(self.run_root),
                     "binary": str(self.binary), "binary_sha256": runner.digest(self.binary),
                     "model": str(self.model), "model_sha256": runner.digest(self.model),
                     "source_pins": [{"name": "fixture-generator", "path": str(self.source),
                                      "sha256": runner.digest(self.source)}],
                     "jobs": self.jobs}
        self.plan["target_set_sha256"] = runner.target_set_digest(self.jobs)
        self.write_plan()
        self.launched = []

    def tearDown(self):
        self.temp.cleanup()

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")

    def fake_launch(self, command, stdout, stderr, env):
        self.assertTrue((self.run_root / "plan.json").is_file())
        self.assertTrue((self.run_root / "environment.json").is_file())
        self.assertTrue((self.run_root / "logs").is_dir())
        self.assertTrue((self.run_root / "renders").is_dir())
        self.assertTrue((self.run_root / "receipts").is_dir())
        self.launched.append((command, dict(env)))
        out = Path(command[-1])
        ident = out.name
        job = next(j for j in self.jobs if j["id"] == ident)
        sparse_float_wav(out / "restored.wav", job["samples"], extensible=True)
        (out / "restoration.json").write_text(json.dumps({
            "steps": 8, "strength": 1.0, "total_seconds": job["samples"] / runner.RATE,
            "elapsed_seconds": 17.0, "controlled": False, "denoise": False,
            "auto_eq": False,
            "excitation_mode": 'continuous' if job.get('continuous_excitation', False) else 'restart_per_chunk',
            "input_origin_samples": job.get('input_origin_samples', 0),
        }))
        sizes = runner._expected_chunks(job["samples"], job["chunk_seconds"], job["overlap_seconds"])
        chunks = out / "chunks"
        chunks.mkdir()
        for index, count in enumerate(sizes):
            prefix = 'chunk_continuous_' if job.get('continuous_excitation', False) else 'chunk_'
            with (chunks / f"{prefix}{index:04}.bin").open("wb") as f:
                f.truncate(count * runner.CHANNELS * runner.SAMPLE_BYTES)
        return FakeProcess()

    def launch_patch(self):
        return mock.patch.object(runner.subprocess, "Popen", side_effect=self.fake_launch)

    def test_preflight_hash_mismatch_latches_before_any_child(self):
        self.jobs[1]["input_sha256"] = "0" * 64
        self.plan["target_set_sha256"] = runner.target_set_digest(self.jobs)
        self.write_plan()
        with self.launch_patch() as popen:
            with self.assertRaises(runner.SafetyStop):
                runner.run(self.plan_path)
        self.assertEqual(popen.call_count, 0)
        halted = json.loads((self.run_root / "HALTED.json").read_text())
        self.assertIn("hash mismatch", halted["reason"])
        self.assertTrue((self.run_root / "plan.json").is_file())

    def test_extensible_float_matches_native_container_and_rejects_other_guid(self):
        path = self.root / "native-like.wav"
        sparse_float_wav(path, 17, extensible=True)
        info = runner.wav_info(path)
        self.assertEqual(info['format'], 3)
        self.assertEqual(info['container_format'], 65534)
        self.assertEqual(info['frames'], 17)
        runner._assert_finite_float_wav(path, info)
        raw = path.read_bytes().replace(bytes.fromhex("0300000000001000800000aa00389b71"),
                                        bytes.fromhex("0100000000001000800000aa00389b71"))
        path.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'subformat'):
            runner.wav_info(path)

    def test_manual_halt_marker_stops_before_first_job_and_is_preserved(self):
        self.run_root.mkdir()
        (self.run_root / "HALT").write_text("owner stop")
        with self.launch_patch() as popen:
            with self.assertRaisesRegex(runner.SafetyStop, "before first job"):
                runner.run(self.plan_path)
        self.assertEqual(popen.call_count, 0)
        self.assertEqual((self.run_root / "HALT").read_text(), "owner stop")
        self.assertTrue((self.run_root / "HALTED.json").is_file())

    def test_hash_drift_after_child_exit_latches_and_prevents_later_jobs(self):
        def mutate_then_launch(command, stdout, stderr, env):
            result = self.fake_launch(command, stdout, stderr, env)
            self.model.write_bytes(b"model changed during first job")
            return result

        with mock.patch.object(runner, "_environment", return_value={"test": True}), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=mutate_then_launch) as popen, \
                mock.patch.object(runner, "memory", return_value={"VmRSS": 100, "VmHWM": 200}):
            with self.assertRaisesRegex(runner.SafetyStop, "hash changed"):
                runner.run(self.plan_path)
        self.assertEqual(popen.call_count, 1)
        halted = json.loads((self.run_root / "HALTED.json").read_text())
        self.assertEqual(halted["jobs_completed"], [])

    def test_memory_breach_latches_and_terminates_child(self):
        class RunningProcess(FakeProcess):
            def poll(self):
                return None if not self.terminated else self.returncode

        process = RunningProcess()

        def launch(command, stdout, stderr, env):
            self.assertTrue((self.run_root / "environment.json").is_file())
            self.assertTrue((self.run_root / "renders").is_dir())
            self.launched.append(command)
            return process

        with mock.patch.object(runner, "_environment", return_value={"test": True}), \
                mock.patch.object(runner.subprocess, "Popen", side_effect=launch), \
                mock.patch.object(runner, "memory", return_value={
                    "VmRSS": runner.MEMORY_LIMIT_KIB + 1, "VmHWM": 10,
                }):
            with self.assertRaisesRegex(runner.SafetyStop, "768 MiB"):
                runner.run(self.plan_path)
        self.assertTrue(process.terminated)
        self.assertEqual(len(self.launched), 1)
        self.assertTrue((self.run_root / "HALTED.json").is_file())

    def test_success_artifact_schema_and_fixed_sequential_commands(self):
        with mock.patch.object(runner, "_environment", return_value={"test": True}), \
                self.launch_patch() as popen, \
                mock.patch.object(runner, "memory", return_value={"VmRSS": 100, "VmHWM": 200}), \
                mock.patch.dict(os.environ, {"ROACH_PROFILE_MID": "1", "ROACH_MID_CACHE_MIB": "1"}):
            complete = runner.run(self.plan_path)
        self.assertEqual(complete["job_count"], 3)
        self.assertEqual(complete["jobs_completed"], ["view0", "view3", "full"])
        self.assertTrue((self.run_root / "COMPLETE.json").is_file())
        self.assertTrue((self.run_root / "environment.json").is_file())
        self.assertTrue((self.run_root / "renders/view0/restored.wav").is_file())
        self.assertTrue((self.run_root / "renders/view3/restored.wav").is_file())
        self.assertTrue((self.run_root / "renders/full/restored.wav").is_file())
        self.assertFalse((self.run_root / "HALTED.json").exists())
        self.assertEqual(popen.call_count, 3)
        for (command, env), expected in zip(self.launched, runner.EXPECTED):
            self.assertIn("--backend", command)
            self.assertEqual(command[command.index("--backend") + 1], "cpu")
            self.assertEqual(command[command.index("--steps") + 1], "8")
            self.assertEqual(command[command.index("--strength") + 1], "1")
            self.assertEqual(command[command.index("--chunk-seconds") + 1], str(expected["chunk_seconds"]))
            self.assertEqual(command[command.index("--overlap-seconds") + 1], str(expected["overlap_seconds"]))
            self.assertNotIn("ROACH_PROFILE_MID", env)
            self.assertNotIn("ROACH_MID_CACHE_MIB", env)
        for path in (self.run_root / "receipts").glob("*.json"):
            receipt = json.loads(path.read_text())
            self.assertNotIn("elapsed_seconds", json.dumps(receipt))
            self.assertEqual(len(receipt["artifacts"]["chunk_files"]),
                             len(receipt["artifacts"]["chunk_sample_counts"]))

    def test_explicit_continuous_mode_and_origins_reach_child_and_receipt(self):
        self.plan['schema'] = 'long-coherence-quality-v2'
        for job in self.jobs:
            job['continuous_excitation'] = True
            job['input_origin_samples'] = 144384 if job['id'] == 'view3' else 0
        self.write_plan()
        with mock.patch.object(runner, '_environment', return_value={'test': True}), \
                self.launch_patch(), mock.patch.object(runner, 'memory', return_value={'VmRSS': 100, 'VmHWM': 200}):
            runner.run(self.plan_path)
        for (command, _), job in zip(self.launched, self.jobs):
            self.assertIn('--continuous-excitation', command)
            self.assertEqual(command[command.index('--input-origin-samples') + 1], str(job['input_origin_samples']))
            r = json.loads((self.run_root/'receipts'/f"{job['id']}.json").read_text())
            self.assertEqual(r['artifacts']['restoration_settings']['excitation_mode'], 'continuous')

    def test_v2_gpu_prefix_keeps_fixed_bounds_and_records_actual_job_count(self):
        self.plan.update(schema='long-coherence-quality-v2', backend='opencl')
        self.plan['jobs'] = self.jobs[:1]
        self.plan['target_set_sha256'] = runner.target_set_digest(self.plan['jobs'])
        self.write_plan()
        with mock.patch.object(runner, '_environment', return_value={'test': True}), \
                self.launch_patch(), mock.patch.object(runner, 'memory', return_value={'VmRSS': 100, 'VmHWM': 200}):
            done = runner.run(self.plan_path)
        self.assertEqual(done['job_count'], 1)
        self.assertEqual(done['backend_requested'], 'opencl')
        command, env = self.launched[0]
        self.assertEqual(command[command.index('--backend') + 1], 'opencl')
        self.assertEqual(env['OCL_ICD_ASSUME_ICD_EXTENSION'], '1')


if __name__ == "__main__":
    unittest.main()
