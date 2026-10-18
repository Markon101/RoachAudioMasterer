import hashlib
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


MODULE_PATH = Path(__file__).with_name('run_mid_profile.py')
spec = importlib.util.spec_from_file_location('run_mid_profile', MODULE_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


class FakeProcess:
    pid = 12345
    returncode = 0

    def poll(self):
        return 0

    def terminate(self):
        pass

    def wait(self, timeout=None):
        return 0

    def kill(self):
        pass


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.binary = self.root/'runner-bin'
        self.model = self.root/'model.json'
        self.input = self.root/'input.wav'
        self.binary.write_bytes(b'binary')
        self.model.write_bytes(b'model')
        self.input.write_bytes(b'procedural fixture')
        self.launched = []
        self.profile_rows = None

        def popen(command, stdout, stderr, env):
            self.launched.append((command, env.copy()))
            out = Path(command[-1])
            out.mkdir(parents=True, exist_ok=True)
            rows = self.profile_rows
            if rows is None:
                requested = int(env['ROACH_MID_BATCH_ROWS'])
                rows = [self.profile_row(requested)]
            (out/'refinement_profile.jsonl').write_text(
                ''.join(json.dumps(row)+'\n' for row in rows))
            return FakeProcess()

        self.popen_patch = mock.patch.object(runner.subprocess, 'Popen', side_effect=popen)
        self.popen_patch.start()
        self.affinity_patch = mock.patch.object(runner.os, 'sched_getaffinity', return_value={0, 1})
        self.affinity_patch.start()

    def tearDown(self):
        self.popen_patch.stop()
        self.affinity_patch.stop()
        self.temp.cleanup()

    def profile_row(self, host_rows=8192, hwm=100, include_host=True):
        refinement = {
            'cache_cap_bytes': 64*1024*1024,
            'predictor_batch_count': 7,
            'memory_after': {'vm_rss_kib': 80, 'process_high_water_kib': hwm},
        }
        if include_host:
            refinement['host_batch_rows'] = host_rows
        return {'disk_chunk_cache_hit': False, 'refinement': refinement}

    def plan(self, jobs=None):
        plan = {
            'binary': str(self.binary), 'binary_sha256': sha(self.binary),
            'model': str(self.model), 'model_sha256': sha(self.model),
            'cutoff': 1500, 'ceiling': 8000,
            'jobs': jobs or [self.job('out')],
        }
        path = self.root/'plan.json'
        path.write_text(json.dumps(plan))
        return path

    def job(self, name, **kwargs):
        job = {'input': str(self.input), 'sha256': sha(self.input),
               'backend': 'cpu', 'out': str(self.root/name)}
        job.update(kwargs)
        return job

    def test_foreground_guard_prevents_child_launch(self):
        path = self.plan()
        with self.assertRaisesRegex(ValueError, 'foreground confirmation'):
            runner.run(path, False)
        self.assertEqual(self.launched, [])

    def test_invalid_later_job_fails_preflight_before_first_launch(self):
        jobs = [self.job('first'), self.job('second', backend='vulkan')]
        path = self.plan(jobs)
        with self.assertRaisesRegex(ValueError, 'backend'):
            runner.run(path, True)
        self.assertEqual(self.launched, [])
        self.assertFalse((self.root/'first').exists())

    def test_job_environment_pins_rows_over_inherited_value_and_receipt_counts_calls(self):
        path = self.plan([self.job('out', backend='opencl', host_batch_rows=16384)])
        with mock.patch.dict(os.environ, {'ROACH_MID_BATCH_ROWS': '65536'}):
            receipts = runner.run(path, True)
        env = self.launched[0][1]
        self.assertEqual(env['ROACH_MID_BATCH_ROWS'], '16384')
        receipt = receipts[0]
        self.assertEqual(receipt['host_batch_rows'], 16384)
        self.assertEqual(receipt['actual_host_call_count'], 7)
        self.assertEqual(receipt['gpu_device_capacity_rows'], 8192)

    def test_actual_profile_rows_must_match_request(self):
        self.profile_rows = [self.profile_row(8192)]
        path = self.plan([self.job('out', host_batch_rows=16384)])
        with self.assertRaisesRegex(ValueError, 'does not match request'):
            runner.run(path, True)

    def test_final_profile_hwm_is_checked_after_process_exit(self):
        self.profile_rows = [self.profile_row(8192, hwm=10*1024+1)]
        path = self.plan([self.job('out', rss_limit_mib=10)])
        with self.assertRaisesRegex(RuntimeError, 'final profile RSS/HWM'):
            runner.run(path, True)

    def test_old_profile_without_host_field_is_default_only(self):
        self.profile_rows = [self.profile_row(include_host=False)]
        path = self.plan([self.job('default')])
        self.assertEqual(len(runner.run(path, True)), 1)

        self.profile_rows = [self.profile_row(include_host=False)]
        path = self.plan([self.job('nondefault', host_batch_rows=16384)])
        with self.assertRaisesRegex(ValueError, 'omits host_batch_rows'):
            runner.run(path, True)

    def test_preflight_refuses_existing_output_log_and_changed_input(self):
        out = self.root/'occupied'
        out.mkdir()
        with self.assertRaises(FileExistsError):
            runner.run(self.plan([self.job('occupied')]), True)
        out.rmdir()

        log = self.root/'logged.log'
        log.write_text('prior run')
        with self.assertRaises(FileExistsError):
            runner.run(self.plan([self.job('logged')]), True)
        log.unlink()

        changed_plan = self.plan([self.job('changed')])
        self.input.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed input'):
            runner.run(changed_plan, True)
        self.assertEqual(self.launched, [])

    def test_non_integer_and_boolean_limits_are_rejected(self):
        for key, values in (
                ('host_batch_rows', (True, 8192.0, '16384', 12288)),
                ('cache_mib', (True, 64.0, '64')),
                ('rss_limit_mib', (False, 128.0, '128'))):
            for value in values:
                with self.subTest(key=key, value=value):
                    with self.assertRaises(ValueError):
                        runner.run(self.plan([self.job('bad', **{key: value})]), True)
        self.assertEqual(self.launched, [])


if __name__ == '__main__':
    unittest.main()
