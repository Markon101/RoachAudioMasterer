import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np


MODULE = Path(__file__).with_name('run_passive_routing_audit.py')
spec = importlib.util.spec_from_file_location('passive_routing', MODULE)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class DoneProcess:
    pid = 999999
    returncode = 0
    def poll(self): return 0


class PassiveRoutingTests(unittest.TestCase):
    def test_gain_policy_preserves_identity_and_zero_without_amplification(self):
        x = np.array([[0.0, 0.25, -0.5], [0.0, 0.125, -0.25]], dtype=np.float32)
        np.testing.assert_array_equal(runner.scale_samples(x, 0), x)
        quiet = runner.scale_samples(np.zeros((1, 9), dtype=np.float32), -40)
        self.assertEqual(np.count_nonzero(quiet), 0)
        self.assertLessEqual(float(np.max(np.abs(runner.scale_samples(x, -20)))),
                             float(np.max(np.abs(x))))
        for bad in (1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                runner.scale_samples(x, bad)

    def make_plan(self, root):
        binary = root/'native'
        binary.write_bytes(b'fake native binary')
        identity = root/'identity.wav'
        runner.write_wav(identity, np.zeros((2, 26*runner.RATE), dtype=np.float32), channels_first=True)
        entries, pins = [], [{'path': str(identity), 'sha256': sha(identity)}]
        for seed in runner.EXPECTED_SEEDS:
            case_dir = root/f'seed-{seed}'
            case_dir.mkdir()
            recipe = case_dir/'recipe.json'
            recipe.write_text(json.dumps({'seed': seed, 'family': f'family-{seed}'}))
            pins.append({'path': str(recipe), 'sha256': sha(recipe)})
            files = {}
            for role in ('target', 'degraded_input'):
                channels = 1 if seed == 860015 else 2
                samples = (np.arange(channels*16_384, dtype=np.float32).reshape(channels, 16_384)
                           % 31) / 100
                wav = case_dir/f'{role}.wav'
                raw = case_dir/f'{role}.planar-f32le'
                runner.write_wav(wav, samples, channels_first=True)
                raw.write_bytes(np.asarray(samples, dtype='<f4').tobytes())
                files[role] = {'wav': {'path': f'{case_dir.name}/{wav.name}', 'sha256': sha(wav)},
                               'planar_f32le': {'path': f'{case_dir.name}/{raw.name}', 'sha256': sha(raw)}}
                pins.extend(({'path': str(wav), 'sha256': sha(wav)},
                             {'path': str(raw), 'sha256': sha(raw)}))
            entries.append({'seed': seed, 'family': f'family-{seed}',
                            'recipe_path': f'{case_dir.name}/{recipe.name}',
                            'recipe_sha256': sha(recipe), 'files': files})
        manifest = root/'manifest.json'
        manifest.write_text(json.dumps({'schema': 'mid-procedural-source-bundle-v1', 'cases': entries}))
        plan = {'binary': str(binary), 'binary_sha256': sha(binary),
                'bundle_manifest': str(manifest), 'bundle_sha256': sha(manifest),
                'long_identity': str(identity), 'long_identity_sha256': sha(identity),
                'source_pins': pins, 'gains_db': [0, -20, -40]}
        plan_path = root/'plan.json'
        plan_path.write_text(json.dumps(plan))
        return plan_path, plan, entries

    def test_preflight_constructs_fixed45_and_rejects_hash_drift(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path, _, entries = self.make_plan(root)
            plan, rows = runner.load_plan(plan_path, root/'fresh')
            self.assertEqual(len(rows), 45)
            self.assertEqual(sum(row['role'] == 'target' for row in rows), 21)
            self.assertEqual(sum(row['role'] == 'degraded_input' for row in rows), 21)
            self.assertEqual(sum(row['seed'] is None for row in rows), 3)
            wav = root/'seed-860002'/'target.wav'
            wav.write_bytes(wav.read_bytes()+b'x')
            halted = root/'halted'
            with self.assertRaisesRegex(ValueError, 'source pin mismatch'):
                runner.run(plan_path, halted)
            self.assertTrue((halted/'HALTED.json').is_file())
            self.assertFalse((halted/'COMPLETE.json').exists())

    def test_preflight_rejects_wrong_known_source_geometry(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path, plan, _ = self.make_plan(root)
            identity = Path(plan['long_identity'])
            runner.write_wav(identity, np.zeros((2, 8), dtype=np.float32), channels_first=True)
            plan['long_identity_sha256'] = sha(identity)
            plan['source_pins'] = [p for p in plan['source_pins'] if Path(p['path']) != identity]
            plan['source_pins'].append({'path': str(identity), 'sha256': sha(identity)})
            plan_path.write_text(json.dumps(plan))
            with self.assertRaisesRegex(ValueError, '26-second'):
                runner.load_plan(plan_path, root/'fresh')

    def test_one_native_row_roundtrips_and_records_base_only_schema(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            binary = root/'native'
            binary.write_bytes(b'fake')
            plan = {'binary': str(binary), 'binary_sha256': sha(binary)}
            def fake_popen(command, stdout, stderr):
                out = Path(command[-1])
                out.mkdir(parents=True)
                (out/'assessment.json').write_text(json.dumps({
                    'authorities': {'mid_flow_authority': 0.0},
                    'sub_energy_dbfs': -2.0, 'low_mid_energy_dbfs': -3.0,
                    'mid_energy_dbfs': -4.0, 'high_energy_dbfs': -5.0,
                    'ultrasonic_energy_dbfs': -6.0}))
                (out/'tuning_card.json').write_text(json.dumps({
                    'recommended_parameters': {'mid_flow_authority': 0.0},
                    'stage_actions': {'stage2_mid_flow_action': 'ABSTAIN'}}))
                return DoneProcess()
            row = {'id': 'tiny', 'seed': 860002, 'role': 'target', 'family': 'fixture',
                   'gain_db': 0, 'samples': np.zeros((2, 16), dtype=np.float32),
                   'channels': 2, 'origin_sha256': 'a'*64}
            row_dir = root/'row'
            row_dir.mkdir()
            input_wav = row_dir/'input.wav'
            runner.write_wav(input_wav, row['samples'], channels_first=True)
            row['expected_input_sha256'] = sha(input_wav)
            with mock.patch.object(runner.subprocess, 'Popen', side_effect=fake_popen):
                result = runner._run_row(plan, row, row_dir)
            self.assertEqual(result['intent_scope'], 'base-only; MusicalIntentCard modulation not applied')
            self.assertEqual(result['geometry']['channels'], 2)
            self.assertEqual(result['base_stage_actions']['stage2_mid_flow_action'], 'ABSTAIN')
            self.assertEqual(result['level']['rms'], 0.0)
            self.assertIsNone(result['sampled_peak_rss_kib'])
            self.assertIsNone(result['sampled_process_hwm_kib'])

    def test_run_local_halt_stops_active_child_and_preserves_partial_trace(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path, _, _ = self.make_plan(root)
            out = root/'out'
            class RunningProcess:
                pid = 999999
                returncode = None
                terminated = False
                def poll(self): return self.returncode
                def terminate(self):
                    self.terminated = True
                    self.returncode = -15
                def wait(self, timeout=None): return self.returncode
            process = RunningProcess()
            def start_then_halt(*args, **kwargs):
                (out/'HALT').write_text('stop this session')
                return process
            with mock.patch.object(runner.subprocess, 'Popen', side_effect=start_then_halt), \
                 mock.patch.object(runner, 'memory', return_value={}):
                with self.assertRaisesRegex(RuntimeError, 'HALT'):
                    runner.run(plan_path, out)
            self.assertTrue(process.terminated)
            self.assertTrue((out/'HALT').is_file())
            self.assertTrue((out/'HALTED.json').is_file())
            self.assertTrue(list((out/'rows').glob('*/input.wav')))
            self.assertFalse((out/'COMPLETE.json').exists())

    def test_mocked45_row_run_reaches_complete_with_observer_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            plan_path, plan, _ = self.make_plan(root)
            launched = []
            def fake_popen(command, stdout, stderr):
                launched.append(command)
                out = Path(command[-1])
                out.mkdir(parents=True)
                (out/'assessment.json').write_text(json.dumps({
                    'authorities': {'mid_flow_authority': 0.0},
                    'sub_energy_dbfs': -2.0, 'low_mid_energy_dbfs': -3.0,
                    'mid_energy_dbfs': -4.0, 'high_energy_dbfs': -5.0,
                    'ultrasonic_energy_dbfs': -6.0}))
                (out/'tuning_card.json').write_text(json.dumps({
                    'recommended_parameters': {'mid_flow_authority': 0.0},
                    'stage_actions': {'stage2_mid_flow_action': 'ABSTAIN'}}))
                return DoneProcess()
            with mock.patch.object(runner.subprocess, 'Popen', side_effect=fake_popen):
                runner.run(plan_path, root/'out')
            complete = json.loads((root/'out'/'COMPLETE.json').read_text())
            self.assertEqual(complete['row_count'], 45)
            self.assertEqual(len(launched), 45)
            rows = [json.loads(line) for line in (root/'out'/'rows.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 45)
            self.assertTrue(all((root/'out'/'rows'/row['id']/'observer.json').is_file()
                                for row in rows))
            self.assertTrue(all(row['sampled_peak_rss_kib'] is None for row in rows))
            self.assertIsNotNone(rows[0]['calibrated_mid_bands']['mid_500_6000_hz']['rms_dbfs'])


if __name__ == '__main__':
    unittest.main()
