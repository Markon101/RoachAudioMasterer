"""Containment checks for the fresh listening recipe; no subprocess/audio work."""
from contextlib import ExitStack
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


RECIPE = (Path(__file__).resolve().parents[1] /
          'artifacts/listening-singularity-20261021/render_pair.py')


class FakeChild:
    """Expose one observable running poll, then a controllable exit/stop."""
    def __init__(self, command, on_exit, stay_running=False, require_kill=False):
        self.command = command
        self.pid = 123456
        self.returncode = None
        self.on_exit = on_exit
        self.stay_running = stay_running
        self.require_kill = require_kill
        self.polls = 0
        self.events = []

    def poll(self):
        self.polls += 1
        if self.returncode is None and not self.stay_running and self.polls > 1:
            self.returncode = 0
            self.on_exit()
        return self.returncode

    def terminate(self):
        self.events.append('terminate')

    def wait(self, timeout=None):
        self.events.append(('wait', timeout))
        if self.require_kill and self.returncode is None:
            raise subprocess.TimeoutExpired(self.command, timeout)
        if self.returncode is None:
            self.returncode = -15
        return self.returncode

    def kill(self):
        self.events.append('kill')
        self.returncode = -9


class RenderListeningPairTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        spec = importlib.util.spec_from_file_location('_isolated_listening_recipe', RECIPE)
        self.recipe = importlib.util.module_from_spec(spec)
        # The recipe imports local helpers by extending sys.path. Keep that
        # extension scoped to this import rather than altering other tests.
        with patch.object(sys, 'path', list(sys.path)):
            spec.loader.exec_module(self.recipe)
        self.recipe.ROOT = self.root
        fixture_script = self.root / 'artifacts/listening-singularity-20261021/render_pair.py'
        self.write(fixture_script, RECIPE.read_bytes())
        self.recipe.__file__ = str(fixture_script)
        self.plan = json.loads(RECIPE.with_name('plan.json').read_text())
        source = self.root / 'source.flac'
        binary, model = self.root / self.plan['binary'], self.root / self.plan['model']
        for path in (source, binary, model,
                     self.root / 'artifacts/rich-low-sfht-v2/basis.json'):
            self.write(path, b'isolated fixture: ' + path.name.encode())
        self.plan['source']['path'] = str(source)
        self.plan['source']['sha256'] = self.digest(source)
        self.plan['binary_sha256'] = self.digest(binary)
        self.plan['model_sha256'] = self.digest(model)
        self.model = model
        self.write_json(fixture_script.with_name('plan.json'), self.plan)
        self.write_json(fixture_script.with_name('attack_prerequisite.json'),
                        {'eligible_to_continue_unchanged_cross_song': True})
        self.write_json(self.root / 'runs/listening-coherence-20261020/real-gate-v3/COMPLETE.json',
                        {'eligible_for_bounded_listening': True})
        self.write_json(self.root / 'runs/ph-memory-20261021/parity-v1/COMPLETE.json', {
            'all_wav_sha256_equal': True,
            'all_non_timing_report_fields_equal': True,
            'complete_ph_audit_and_finalizer_metadata_exact': True})
        self.run = self.root / 'runs/listening-singularity-20261021/fullsong-v1'
        self.geometry = {k: self.plan['source'][k] for k in ('rate', 'channels', 'frames')}
        self.children = []
        self.decode_calls = []

    @staticmethod
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def write(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def write_json(self, path, data):
        self.write(path, (json.dumps(data) + '\n').encode())

    def save_plan(self):
        self.write_json(Path(self.recipe.__file__).with_name('plan.json'), self.plan)

    def add_parity(self):
        path = self.root / 'artifacts/new-parity/COMPLETE.json'
        proof = {key: True for key in (
            'all_wav_sha256_equal', 'all_non_timing_report_fields_equal',
            'complete_ph_audit_and_finalizer_metadata_exact',
            'real10s_complete_ph_audit_and_finalizer_metadata_exact')}
        proof['pins'] = {str(self.root / self.plan['binary']): self.plan['binary_sha256']}
        self.write_json(path, proof)
        self.plan['additional_parity'] = {
            'path': str(path.relative_to(self.root)), 'sha256': self.digest(path)}
        self.save_plan()
        return path, proof

    def add_prefix(self):
        self.plan['control_prefix_sha256'] = {
            name: hashlib.sha256(('control prefix: ' + name).encode()).hexdigest()
            for name in sorted(self.recipe.CONTROL_PREFIX_FILES)}
        self.save_plan()

    def assert_refused_before_dispatch(self, reason):
        with self.assertRaisesRegex(RuntimeError, reason):
            self.execute()
        self.assertEqual(self.decode_calls, [])
        self.assertEqual(self.children, [])
        self.assertFalse(self.run.exists())

    def execute(self, *, halt_at_exit=False, halt_during_validation=False,
                mutate_final_pin=False, stop=None, require_kill=False,
                preservation_failure=False, plan_path=None,
                mutate_parity_after_decode=False, missing_prefix=None):
        real_read_text = Path.read_text
        real_save = self.recipe.save

        def read_text(path, *args, **kwargs):
            if path == Path('/proc/meminfo'):
                return 'MemAvailable: 4194304 kB\n'
            return real_read_text(path, *args, **kwargs)

        def decode(command, **kwargs):
            self.decode_calls.append(command)
            self.assertEqual(command[0], 'ffmpeg')
            self.assertTrue(kwargs['check'])
            self.write(Path(command[-1]), b'decoded input fixture')
            if mutate_parity_after_decode:
                path = self.root / self.plan['additional_parity']['path']
                self.write(path, b'changed receipt during decoder')

        def spawn(command, **kwargs):
            self.assertEqual(kwargs['cwd'], self.root)
            out = Path(command[command.index('--out') + 1])
            self.assertFalse(out.exists())
            candidate = out.name == 'candidate'
            summary = {
                'mid_model': str(self.model),
                'master_stats': {'after': {'lufs_integrated': -13., 'true_peak_dbtp': -1.}},
                'preservation': {k: True for k in (
                    'mono_compatibility_passed', 'known_band_preserved', 'transient_timing_passed')},
                'settings': {
                    'backend': 'opencl', 'steps': 8, 'target_lufs': -13.,
                    'force_specialists': False,
                    'master_controls': {'quiet_mid_denoise': True},
                    'mid_coherence_requested': candidate, 'mid_coherence_applied': candidate,
                    'explicit_cli_parameters': ['mid_coherence'] if candidate else []}}
            if preservation_failure and not candidate:
                summary['preservation']['known_band_preserved'] = False
            self.write_json(out / 'summary.json', summary)
            self.write_json(out / 'master.json', {'fixture': out.name})
            self.write(out / 'mastered.wav', out.name.encode())
            for name in self.recipe.CONTROL_PREFIX_FILES:
                if not candidate and name == missing_prefix:
                    continue
                self.write(out / name, (out.name + ' prefix: ' + name).encode())

            def on_exit():
                if candidate and halt_at_exit:
                    self.write(self.run / 'HALT', b'stop at candidate exit')

            child = FakeChild(command, on_exit, stay_running=stop is not None,
                              require_kill=require_kill)
            self.children.append(child)
            return child

        def memory(_pid):
            if stop == 'halt':
                self.write(self.run / 'HALT', b'stop running child')
            kib = ((self.plan['limits']['process_hwm_mib'] + 1) * 1024
                   if stop == 'memory' else 100 * 1024)
            return {'VmRSS': kib, 'VmHWM': kib}

        def validate(path, _info):
            if path.parent.name == 'candidate' and halt_during_validation:
                self.write(self.run / 'HALT', b'stop during candidate validation')

        def save(path, value):
            real_save(path, value)
            if path.name == 'candidate-receipt.json' and mutate_final_pin:
                self.model.write_bytes(b'model changed after final child')

        with ExitStack() as stack:
            stack.enter_context(patch.object(Path, 'read_text', read_text))
            stack.enter_context(patch.object(self.recipe.shutil, 'disk_usage',
                                            return_value=SimpleNamespace(free=8 * 1024**3)))
            stack.enter_context(patch.object(self.recipe.subprocess, 'run', side_effect=decode))
            stack.enter_context(patch.object(self.recipe.subprocess, 'Popen', side_effect=spawn))
            stack.enter_context(patch.object(self.recipe, 'memory', side_effect=memory))
            stack.enter_context(patch.object(self.recipe.time, 'monotonic', return_value=0.))
            stack.enter_context(patch.object(self.recipe.time, 'sleep'))
            stack.enter_context(patch.object(self.recipe, 'probe', return_value=self.geometry))
            stack.enter_context(patch.object(self.recipe, 'wav_info', return_value={}))
            stack.enter_context(patch.object(self.recipe, '_assert_finite_float_wav', side_effect=validate))
            stack.enter_context(patch.object(self.recipe, 'save', side_effect=save))
            self.recipe.main(plan_path)

    def assert_halted(self, reason):
        self.assertFalse((self.run / 'COMPLETE.json').exists())
        halted = self.run / 'HALTED.json'
        preserved = halted.read_bytes()
        receipt = json.loads(preserved)
        self.assertIn(reason, receipt['reason'])
        # A stopped session must remain latched, including its first receipt.
        with patch.object(self.recipe.subprocess, 'run') as decode, \
                patch.object(self.recipe.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'existing run or latched session'):
                self.recipe.main()
            decode.assert_not_called()
            spawn.assert_not_called()
        self.assertEqual(halted.read_bytes(), preserved)
        return receipt

    def test_normal_mocked_pair_completes(self):
        self.execute()
        receipt = json.loads((self.run / 'COMPLETE.json').read_text())
        self.assertTrue(receipt['eligible_for_listening'])
        self.assertFalse(receipt['production_promoted'])
        self.assertEqual([r['name'] for r in receipt['rows']], ['control', 'candidate'])
        self.assertNotEqual(receipt['rows'][0]['mastered_sha256'],
                            receipt['rows'][1]['mastered_sha256'])
        self.assertFalse((self.run / 'HALTED.json').exists())
        self.assertEqual(len(self.children), 2)
        self.assertTrue(all(not child.events for child in self.children))

    def test_failed_control_preserves_execution_receipt_before_quality_gate(self):
        with self.assertRaisesRegex(RuntimeError, 'native preservation gate failed'):
            self.execute(preservation_failure=True)
        self.assert_halted('native preservation gate failed')
        execution = json.loads((self.run / 'control-execution.json').read_text())
        self.assertEqual(execution['child_returncode'], 0)
        self.assertEqual(execution['peak_sampled_process_hwm_mib'], 100.)
        self.assertFalse(execution['speed_experiment'])
        self.assertEqual(len(self.children), 1)
        self.assertFalse((self.run / 'candidate').exists())

    def test_optional_parity_is_verified_and_pinned(self):
        path, _ = self.add_parity()
        self.execute()
        protocol = json.loads((self.run / 'protocol.json').read_text())
        self.assertEqual(protocol['pins'][str(path)], self.plan['additional_parity']['sha256'])
        self.assertTrue((self.run / 'COMPLETE.json').exists())

    def test_malformed_additional_parity_declarations_refused(self):
        for declaration in (None, [], {}, {'path': 'proof.json', 'sha256': 'bad'},
                            {'path': '', 'sha256': '0' * 64}):
            with self.subTest(declaration=declaration):
                self.plan['additional_parity'] = declaration
                self.save_plan()
                self.assert_refused_before_dispatch('malformed additional parity')

    def test_changed_additional_parity_refused_before_dispatch(self):
        path, _ = self.add_parity()
        self.write(path, b'changed receipt')
        self.assert_refused_before_dispatch('additional parity pin mismatch')

    def test_malformed_additional_parity_receipts_refused(self):
        path, _ = self.add_parity()
        for raw in (b'not JSON', b'[]', b'null'):
            with self.subTest(raw=raw):
                self.write(path, raw)
                self.plan['additional_parity']['sha256'] = self.digest(path)
                self.save_plan()
                self.assert_refused_before_dispatch('additional.*parity')

    def test_additional_parity_requires_every_flag_exactly_true(self):
        path, proof = self.add_parity()
        for key in tuple(key for key in proof if key != 'pins'):
            for value in (False, 1, 'true', None):
                with self.subTest(key=key, value=value):
                    changed = {**proof, key: value}
                    self.write_json(path, changed)
                    self.plan['additional_parity']['sha256'] = self.digest(path)
                    self.save_plan()
                    self.assert_refused_before_dispatch('additional memory-fix parity')
            with self.subTest(key=key, missing=True):
                self.write_json(path, {name: value for name, value in proof.items() if name != key})
                self.plan['additional_parity']['sha256'] = self.digest(path)
                self.save_plan()
                self.assert_refused_before_dispatch('additional memory-fix parity')

    def test_additional_parity_requires_matching_binary_binding(self):
        path, proof = self.add_parity()
        for binding in (None, {}, {str(self.root / self.plan['binary']): '0' * 64},
                        {self.plan['binary']: self.plan['binary_sha256']}):
            with self.subTest(binding=binding):
                self.write_json(path, {**proof, 'pins': binding})
                self.plan['additional_parity']['sha256'] = self.digest(path)
                self.save_plan()
                self.assert_refused_before_dispatch('not bound to planned binary')

    def test_parity_change_during_decode_blocks_first_native_child(self):
        self.add_parity()
        with self.assertRaisesRegex(RuntimeError, 'pin changed'):
            self.execute(mutate_parity_after_decode=True)
        self.assert_halted('pin changed')
        self.assertEqual(len(self.decode_calls), 1)
        self.assertEqual(len(self.children), 0)

    def test_matching_control_prefix_allows_distinct_candidate_prefix(self):
        self.add_parity()
        self.add_prefix()
        self.execute()
        receipt = json.loads((self.run / 'COMPLETE.json').read_text())
        self.assertEqual(receipt['rows'][0]['control_prefix_sha256'],
                         self.plan['control_prefix_sha256'])
        self.assertNotIn('control_prefix_sha256', receipt['rows'][1])
        self.assertEqual(len(self.children), 2)

    def test_malformed_prefix_names_and_digests_refused_before_dispatch(self):
        for pins in (None, [], {}, {'../stage2_in.wav': '0' * 64},
                     {'/stage2_in.wav': '0' * 64}, {'mastered.wav': '0' * 64},
                     {'stage2_in.wav': 'bad'}):
            with self.subTest(pins=pins):
                self.plan['control_prefix_sha256'] = pins
                self.save_plan()
                self.assert_refused_before_dispatch('malformed control prefix')

    def test_control_prefix_mismatch_halts_after_one_child(self):
        self.add_prefix()
        self.plan['control_prefix_sha256']['stage2_mid/restored.wav'] = '0' * 64
        self.save_plan()
        with self.assertRaisesRegex(RuntimeError, 'control prefix mismatch: stage2_mid/restored.wav'):
            self.execute()
        self.assert_halted('control prefix mismatch')
        self.assertEqual(len(self.children), 1)
        self.assertTrue((self.run / 'control-execution.json').exists())
        self.assertFalse((self.run / 'candidate').exists())

    def test_missing_control_prefix_halts_before_candidate(self):
        self.add_prefix()
        with self.assertRaisesRegex(RuntimeError, 'control prefix mismatch'):
            self.execute(missing_prefix='stage4_spatial.wav')
        self.assert_halted('control prefix mismatch')
        self.assertEqual(len(self.children), 1)
        self.assertFalse((self.run / 'candidate').exists())

    def test_explicit_plan_uses_its_own_prerequisite_and_run_root(self):
        path = self.root / 'artifacts/second-source/plan.json'
        self.plan['run_root'] = 'runs/second-source'
        self.write_json(path, self.plan)
        self.write_json(path.with_name('attack_prerequisite.json'),
                        {'eligible_to_continue_unchanged_cross_song': True})
        self.run = self.root / 'runs/second-source/fullsong-v1'
        self.execute(plan_path=path)
        protocol = json.loads((self.run / 'protocol.json').read_text())
        self.assertEqual(protocol['plan']['run_root'], 'runs/second-source')
        self.assertIn(str(path), protocol['pins'])
        self.assertTrue((self.run / 'COMPLETE.json').exists())

    def test_run_root_cannot_escape_local_runs(self):
        path = Path(self.recipe.__file__).with_name('plan.json')
        self.plan['run_root'] = '../outside-run'
        self.write_json(path, self.plan)
        with patch.object(self.recipe.subprocess, 'run') as decode, \
                patch.object(self.recipe.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'run root must remain inside local runs'):
                self.recipe.main()
            decode.assert_not_called()
            spawn.assert_not_called()

    def test_valid_revision_is_pinned_and_changes_only_prerequisite_gate(self):
        plan_path = Path(self.recipe.__file__).with_name('plan.json')
        revision_path = plan_path.with_name('plan_revision_v2.json')
        revision = {
            'original_plan_sha256': self.digest(plan_path),
            'replacement_prerequisite': 'Fixture reviewed prerequisite; all other gates unchanged.'}
        self.write_json(revision_path, revision)
        self.execute()
        protocol = json.loads((self.run / 'protocol.json').read_text())
        expected_plan = json.loads(json.dumps(self.plan))
        expected_plan['gates'][0] = revision['replacement_prerequisite']
        self.assertEqual(protocol['plan'], expected_plan)
        self.assertEqual(protocol['plan_revision'], revision)
        self.assertEqual(protocol['pins'][str(revision_path)], self.digest(revision_path))
        self.assertEqual(protocol['pins'][str(plan_path)], self.digest(plan_path))
        self.assertEqual(json.loads(plan_path.read_text()), self.plan)
        self.assertTrue((self.run / 'COMPLETE.json').is_file())

    def test_invalid_revision_refuses_before_decoder_or_native_dispatch(self):
        revision_path = Path(self.recipe.__file__).with_name('plan_revision_v2.json')
        self.write_json(revision_path, {
            'original_plan_sha256': '0' * 64,
            'replacement_prerequisite': 'Invalid fixture revision must never be used.'})
        with patch.object(self.recipe.subprocess, 'run') as decode, \
                patch.object(self.recipe.subprocess, 'Popen') as spawn:
            with self.assertRaisesRegex(RuntimeError, 'revision does not match original frozen plan'):
                self.recipe.main()
            decode.assert_not_called()
            spawn.assert_not_called()
        self.assertFalse(self.run.exists())

    def test_halt_at_child_exit_prevents_completion(self):
        with self.assertRaisesRegex(RuntimeError, 'latched session'):
            self.execute(halt_at_exit=True)
        receipt = self.assert_halted('latched session')
        self.assertEqual(len(receipt['completed_rows']), 1)
        self.assertEqual(len(self.children), 2)
        self.assertTrue((self.run / 'HALT').exists())
        self.assertFalse((self.run / 'candidate-receipt.json').exists())

    def test_halt_during_final_candidate_validation_prevents_completion(self):
        with self.assertRaisesRegex(RuntimeError, 'latched session'):
            self.execute(halt_during_validation=True)
        receipt = self.assert_halted('latched session')
        self.assertEqual(len(receipt['completed_rows']), 2)
        self.assertTrue((self.run / 'HALT').exists())
        self.assertFalse((self.run / 'candidate-receipt.json').exists())

    def test_final_pin_mutation_prevents_completion(self):
        with self.assertRaisesRegex(RuntimeError, 'pin changed at completion'):
            self.execute(mutate_final_pin=True)
        receipt = self.assert_halted('pin changed at completion')
        self.assertEqual(len(receipt['completed_rows']), 2)
        self.assertTrue((self.run / 'candidate-receipt.json').exists())

    def test_memory_stop_terminates_and_reaps_child(self):
        with self.assertRaisesRegex(RuntimeError, 'resource/HALT stop'):
            self.execute(stop='memory')
        self.assert_halted('resource/HALT stop')
        self.assertEqual(len(self.children), 1)
        self.assertEqual(self.children[0].events, ['terminate', ('wait', 5)])
        self.assertIsNotNone(self.children[0].returncode)

    def test_halt_stop_kills_and_reaps_unresponsive_child(self):
        with self.assertRaisesRegex(RuntimeError, 'resource/HALT stop'):
            self.execute(stop='halt', require_kill=True)
        self.assert_halted('resource/HALT stop')
        self.assertEqual(len(self.children), 1)
        self.assertEqual(self.children[0].events,
                         ['terminate', ('wait', 5), 'kill', ('wait', None)])
        self.assertEqual(self.children[0].returncode, -9)
        self.assertTrue((self.run / 'HALT').exists())


if __name__ == '__main__':
    unittest.main()
