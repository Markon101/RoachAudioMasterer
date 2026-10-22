"""Mocked containment and dispatch checks for the two-track runner."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / 'artifacts/around-mastering-20261022/run.py'
PLAN = SCRIPT.with_name('plan.json')
HELPER = Path(__file__).resolve().parents[1] / 'artifacts/listening-singularity-20261021/render_pair.py'


class Child:
    def __init__(self, command, finish):
        self.command, self.finish = command, finish
        self.pid, self.returncode, self.polls, self.events = 54321, None, 0, []

    def poll(self):
        self.polls += 1
        if self.polls > 2 and self.returncode is None:
            self.finish()
            self.returncode = 0
        return self.returncode

    def terminate(self):
        self.events.append('terminate')

    def wait(self, timeout=None):
        self.events.append(('wait', timeout))
        if self.returncode is None:
            self.returncode = -15
        return self.returncode

    def kill(self):
        self.events.append('kill')
        self.returncode = -9


class AroundRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        spec = importlib.util.spec_from_file_location('_around_runner_fixture', SCRIPT)
        self.runner = importlib.util.module_from_spec(spec)
        with patch.object(sys, 'path', list(sys.path)):
            spec.loader.exec_module(self.runner)
        self.runner.ROOT = self.root
        script = self.root / 'artifacts/around-mastering-20261022/run.py'
        self.write(script, SCRIPT.read_bytes())
        self.runner.__file__ = str(script)
        self.runner.HERE = script.parent
        self.plan = json.loads(PLAN.read_text())
        self.run = self.root / self.plan['run_root']
        self.children, self.commands = [], []
        self.sources = []
        for i, row in enumerate(self.plan['tracks']):
            src = self.root / f'source-{i}.wav'
            self.write(src, f'source-{i}'.encode())
            row['path'], row['sha256'] = str(src), self.digest(src)
            self.sources.append(src)
        binary = self.root / self.plan['binary']
        model = self.root / self.plan['model']
        low = self.root / self.plan['low_model']
        for p in (binary, model, low):
            self.write(p, ('fixture:' + p.name).encode())
        self.plan['binary_sha256'] = self.digest(binary)
        self.plan['model_sha256'] = self.digest(model)
        self.plan['low_model_sha256'] = self.digest(low)
        parity = self.root / self.plan['additional_parity']['path']
        self.write_json(parity, {
            'all_wav_sha256_equal': True,
            'all_non_timing_report_fields_equal': True,
            'complete_ph_audit_and_finalizer_metadata_exact': True,
            'real10s_complete_ph_audit_and_finalizer_metadata_exact': True,
            'pins': {str(binary): self.plan['binary_sha256']}})
        self.plan['additional_parity']['sha256'] = self.digest(parity)
        self.write_json(script.with_name('plan.json'), self.plan)
        proof = self.root / 'artifacts/listening-singularity-20261021/render_pair.py'
        self.write(proof, HELPER.read_bytes())

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_json(self, path, obj):
        self.write(path, (json.dumps(obj) + '\n').encode())

    def invoke(self, *, preservation_bad=False, memory_bad=False, halt_late=False,
               config_bad=False, mutate_target=False, pin_bad=False, prehalt=False):
        script = Path(self.runner.__file__)
        plan_path = script.with_name('plan.json')
        if mutate_target:
            self.plan['target_lufs'] = -12.0
            self.write_json(plan_path, self.plan)
        if pin_bad:
            self.sources[0].write_bytes(b'changed after plan pin')
        if prehalt:
            self.write(self.root / 'artifacts/around-mastering-20261022/HALTED.json', b'latched')
        real_read = Path.read_text

        def read_text(path, *a, **kw):
            if path == Path('/proc/meminfo'):
                return 'MemAvailable: 4194304 kB\n'
            return real_read(path, *a, **kw)

        def spawn(command, **kw):
            self.commands.append(command)
            out = Path(command[command.index('--out') + 1])
            def finish():
                if halt_late and len(self.children) == 1:
                    self.write(self.run / 'HALT', b'late stop')
                stats = {'lufs_integrated': -13.95, 'true_peak_dbtp': -1.0}
                checks = {k: True for k in ('mono_compatibility_passed', 'known_band_preserved', 'transient_timing_passed')}
                if preservation_bad and len(self.children) == 1:
                    checks['known_band_preserved'] = False
                summary = {'mid_model': str(self.root / self.plan['model']),
                           'master_stats': {'after': stats}, 'preservation': checks,
                           'settings': {'mid_coherence_applied': False, 'force_specialists': False,
                                        'target_lufs': (-13 if config_bad else -14),
                                        'mid_model': str(self.root / self.plan['model'])}}
                self.write_json(out / 'summary.json', summary)
                self.write(out / 'mastered.wav', b'wav')
                self.write(out / 'mastered.flac', b'flac')
            child = Child(command, finish)
            self.children.append(child)
            return child

        def proc_memory(_pid):
            peak = (self.plan['limits']['process_hwm_mib'] + 1) * 1024 if memory_bad else 100 * 1024
            return {'VmRSS': peak, 'VmHWM': peak}

        with patch.object(Path, 'read_text', read_text), \
             patch.object(self.runner.shutil, 'disk_usage', return_value=SimpleNamespace(free=8 * 1024**3)), \
             patch.object(self.runner, 'available_mib', return_value=4096), \
             patch.object(self.runner, 'memory', side_effect=proc_memory), \
             patch.object(self.runner, 'probe', side_effect=lambda p: (
                 next({k: row[k] for k in ('rate', 'channels', 'frames')}
                      for row in self.plan['tracks'] if row['path'] == str(p))
                 if str(p) in {row['path'] for row in self.plan['tracks']} else
                 {k: self.plan['tracks'][0 if Path(p).parent.name == 'around-v1' else 1][k]
                  for k in ('rate', 'channels', 'frames')})), \
             patch.object(self.runner, 'wav_info', return_value={}), \
             patch.object(self.runner, '_assert_finite_float_wav'), \
             patch.object(self.runner.subprocess, 'Popen', side_effect=spawn), \
             patch.object(self.runner.time, 'monotonic', return_value=0.), \
             patch.object(self.runner.time, 'sleep'):
            return self.runner.main()

    def test_two_independent_tracks_complete_sequentially(self):
        self.invoke()
        complete = json.loads((self.run / 'COMPLETE.json').read_text())
        self.assertEqual([r['id'] for r in complete['rows']], ['around-v1', 'around-v2'])
        self.assertFalse(complete['production_promoted'])
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(c.polls >= 2 for c in self.children))

    def test_preservation_failure_is_recorded_and_other_track_runs(self):
        self.invoke(preservation_bad=True)
        rows = json.loads((self.run / 'COMPLETE.json').read_text())['rows']
        self.assertFalse(rows[0]['qualified'])
        self.assertFalse(rows[0]['quality_checks']['known_band_preserved'])
        self.assertEqual(len(self.children), 2)

    def test_memory_failure_reaps_child_and_blocks_next_track(self):
        with self.assertRaisesRegex(RuntimeError, 'process memory ceiling'):
            self.invoke(memory_bad=True)
        self.assertEqual(len(self.children), 1)
        self.assertEqual(self.children[0].events, ['terminate', ('wait', 5)])
        self.assertTrue((self.run / 'HALTED.json').exists())

    def test_late_halt_prevents_complete_and_preserves_first_latch(self):
        with self.assertRaisesRegex(RuntimeError, 'latched experiment'):
            self.invoke(halt_late=True)
        receipt = (self.run / 'HALTED.json').read_bytes()
        self.assertFalse((self.run / 'COMPLETE.json').exists())
        with self.assertRaisesRegex(RuntimeError, 'fresh local run directory'):
            self.invoke(prehalt=True)
        self.assertEqual((self.root / 'artifacts/around-mastering-20261022/HALTED.json').read_bytes(), b'latched')
        self.assertEqual((self.run / 'HALTED.json').read_bytes(), receipt)

    def test_target_configuration_mismatch_is_fatal(self):
        with self.assertRaisesRegex(RuntimeError, 'wrong loudness target'):
            self.invoke(config_bad=True)
        self.assertEqual(len(self.children), 1)
        self.assertTrue((self.run / 'HALTED.json').exists())

    def test_source_pin_mismatch_refuses_before_dispatch(self):
        with self.assertRaisesRegex(RuntimeError, 'original source pin mismatch'):
            self.invoke(pin_bad=True)
        self.assertEqual(self.children, [])
        self.assertFalse(self.run.exists())


if __name__ == '__main__':
    unittest.main()
