"""Bounded local tests; real FFmpeg smoke uses procedural audio only."""
import copy
import csv
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from io import StringIO
import json
import math
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

import blind_listening as blind


class ListeningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
        self.manifest = {
            'version': 1, 'mode': 'historical_playback', 'seed': 219,
            'target_lufs': -23.0,
            'tracks': [{'id': 'secret_song', 'variants': [
                {'label': 'secret_old', 'path': 'private_old.wav', 'provenance': {'note': 'historical'}},
                {'label': 'secret_new', 'path': 'private_new.wav', 'provenance': {'note': 'historical'}}],
                'excerpts': [{'id': 'onsets', 'kind': 'diagnostic', 'start_sample': 0, 'frames': 240000},
                             {'id': 'context', 'kind': 'long', 'start_sample': 0, 'frames': 960000}]}]}
        for name in ['private_old.wav', 'private_new.wav']:
            (self.root / name).write_bytes(b'procedural placeholder')
        self.manifest_path = self.root / 'manifest.json'
        self.listener, self.organizer = self.root / 'public', self.root / 'private'

    def write_manifest(self):
        self.manifest_path.write_text(json.dumps(self.manifest))

    def fake_prepare(self):
        self.write_manifest()
        # Probe encoded duration by the fake encoder's stored interval.
        encoded = {}
        def encode(source, destination, excerpt, gain):
            self.assertLessEqual(gain, 0)
            destination.write_bytes((source.name + str(excerpt) + str(gain)).encode())
            encoded[destination] = excerpt['frames']
        def probe(path, **_):
            return {'rate': 8000, 'channels': 1, 'frames': encoded.get(path, 1040000)}
        def measure(path, excerpt=None):
            return {'lufs': -23. if path.name in {'A.wav', 'B.wav'} else -18., 'true_peak_dbfs': -3.}
        def fake_run(args):
            return (f"{Path(args[0]).name} version mocked\n", '')
        with patch.object(blind, 'probe', side_effect=probe), patch.object(blind, 'measure', side_effect=measure), \
                patch.object(blind, 'encode', side_effect=encode), \
                patch.object(blind, 'run', side_effect=fake_run):
            blind.prepare(self.manifest_path, self.listener, self.organizer)

    def procedural_manifest(self):
        manifest = copy.deepcopy(self.manifest)
        manifest['mode'] = 'procedural_diagnostic'
        for variant in manifest['tracks'][0]['variants']:
            variant['provenance'] = {
                'procedural': True,
                'run_record': 'runs/diagnostic/summary.json',
                'git_commit': 'a' * 40,
                'source_sha256': 'b' * 64,
            }
        manifest['tracks'][0]['excerpts'] = [
            {'id': 'short_control', 'kind': 'diagnostic', 'start_sample': 0, 'frames': 13 * 8000},
            {'id': 'longer_diagnostic', 'kind': 'diagnostic', 'start_sample': 0, 'frames': 26 * 8000},
        ]
        return manifest

    def test_blinding_attenuation_and_determinism(self):
        self.fake_prepare()
        self.assertTrue((self.listener / 'READY').is_file())
        public_files = [p for p in self.listener.rglob('*') if p.is_file()]
        for path in public_files:
            self.assertNotIn('secret_', str(path.relative_to(self.listener)))
            if path.suffix != '.wav':
                self.assertNotIn('secret_', path.read_text())
                self.assertNotIn('LUFS', path.read_text())
        key = json.loads((self.organizer / 'key.private.json').read_text())
        session = json.loads((self.organizer / 'session.private.json').read_text())
        self.assertTrue(session['tool_versions']['ffmpeg'].startswith('ffmpeg version'))
        self.assertTrue(session['tool_versions']['ffprobe'].startswith('ffprobe version'))
        self.assertEqual(len(key['trials']), 4)
        self.assertEqual(sum(t['identity'] for t in key['trials']), 2)
        self.assertTrue(all(m['gain_db'] == -5 for t in key['trials'] for m in t['members'].values()))
        self.listener, self.organizer = self.root / 'public2', self.root / 'private2'
        self.fake_prepare()
        rerun_key = json.loads((self.organizer / 'key.private.json').read_text())
        self.assertEqual(key, rerun_key)

    def test_legacy_assignments_remain_exactly_compatible(self):
        tracks = [{'id': 'fixture', 'variants': [{}, {}, {}],
                   'excerpts': [{'id': 'first'}, {'id': 'second'}]}]
        # Frozen from the original independent assignment algorithm.
        expected = [('first', 0, 2), ('second', 1, 1), ('first', 1, 0),
                    ('second', 1, 2), ('second', 2, 0), ('first', 2, 1),
                    ('first', 2, 2), ('second', 1, 0)]
        for policy in (None, 'independent'):
            args = () if policy is None else (policy,)
            actual = [(e['id'], a, b) for _, e, a, b in blind.assignments(tracks, 219, *args)]
            self.assertEqual(actual, expected)

    def test_counterbalanced_manifest_reaches_packager(self):
        self.manifest['assignment_policy'] = 'counterbalanced_pairs'
        self.fake_prepare()
        key = json.loads((self.organizer / 'key.private.json').read_text())
        pairs = [t for t in key['trials'] if not t['identity']]
        self.assertEqual(Counter(t['members']['A']['variant']['label'] for t in pairs),
                         Counter({'secret_old': 1, 'secret_new': 1}))
        self.assertEqual(Counter(t['excerpt']['id'] for t in key['trials'] if t['identity']),
                         Counter({'onsets': 1, 'context': 1}))

    def test_counterbalanced_pairs_alternate_independently_per_track_and_pair(self):
        for excerpt_count in (2, 5):
            tracks = [{'id': f'track-{n}', 'variants': [{} for _ in range(variant_count)],
                       'excerpts': [{'id': str(e)} for e in range(excerpt_count)]}
                      for n, variant_count in enumerate((3, 4))]
            trials = blind.assignments(tracks, 219, 'counterbalanced_pairs')
            groups, identities = defaultdict(dict), Counter()
            for track, excerpt, a, b in trials:
                if a == b:
                    identities[(track['id'], excerpt['id'])] += 1
                else:
                    pair = (track['id'], min(a, b), max(a, b))
                    self.assertNotIn(int(excerpt['id']), groups[pair])
                    groups[pair][int(excerpt['id'])] = a < b
            self.assertEqual(len(groups), 3 + 6)
            self.assertEqual(identities, Counter({(t['id'], e['id']): 1
                                                 for t in tracks for e in t['excerpts']}))
            for orientations in groups.values():
                ordered = [orientations[e] for e in range(excerpt_count)]
                self.assertTrue(all(a != b for a, b in zip(ordered, ordered[1:])))
                self.assertLessEqual(abs(2 * sum(ordered) - excerpt_count), 1)
            expected_count = excerpt_count * ((3 + 1) + (6 + 1))
            self.assertEqual(len(trials), expected_count)

    def test_counterbalanced_seed_reproducibility_and_shuffled_trial_order(self):
        tracks = self.manifest['tracks']
        def signature(seed):
            return [(t['id'], e['id'], a, b) for t, e, a, b in
                    blind.assignments(tracks, seed, 'counterbalanced_pairs')]
        self.assertEqual(signature(219), signature(219))
        signatures = [signature(seed) for seed in range(16)]
        self.assertGreater(len({tuple(s) for s in signatures}), 1)
        first_orientations = {next((a, b) for _, e, a, b in s
                                   if e == 'onsets' and a != b) for s in signatures}
        self.assertEqual(first_orientations, {(0, 1), (1, 0)})
        # At least one order interleaves excerpts, proving ordering is shuffled
        # after creating their comparisons and identity controls.
        self.assertTrue(any(s[0][1] != s[1][1] for s in signatures))

    def test_unknown_assignment_policy_rejected_before_output(self):
        for policy in ('counterbalanced', '', None, [], {}):
            self.manifest['assignment_policy'] = policy
            self.write_manifest()
            with self.assertRaisesRegex(ValueError, 'assignment_policy'):
                blind.prepare(self.manifest_path, self.listener, self.organizer)
            self.assertFalse(self.listener.exists())
            self.assertFalse(self.organizer.exists())
            with self.assertRaisesRegex(ValueError, 'assignment_policy'):
                blind.assignments(self.manifest['tracks'], 219, policy)

    def test_listener_notes_guidance_keeps_categorical_response_schema(self):
        for instructions in (blind.INSTRUCTIONS, blind.PROCEDURAL_DIAGNOSTIC_INSTRUCTIONS):
            self.assertIn('comfortable listening level', instructions)
            self.assertIn('If the files remain too quiet, stop and report that before recording judgments.',
                          instructions)
            self.assertIn('keep the same playback device, settings and volume', instructions)
            self.assertIn('notes.txt', instructions)
            self.assertIn('categorical choices unchanged', instructions)
        self.fake_prepare()
        (self.listener / 'notes.txt').write_text('T0001: Free prose stays here.\n')
        response_path = self.listener / 'responses.csv'
        with response_path.open(newline='') as source:
            reader = csv.DictReader(source)
            self.assertEqual(reader.fieldnames, blind.FIELDS)
            rows = list(reader)
        for row in rows:
            row.update({k: 'indistinguishable' for k in blind.FIELDS[1:5]})
            row.update(confidence='low', fatigue='not_assessed')
        rows[0]['preference'] = 'A because it sounds clearer'
        with response_path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=blind.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'categorical'):
            blind.reveal(self.organizer, response_path)
        rows[0]['preference'] = 'A'
        with response_path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=blind.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.assertTrue(blind.reveal(self.organizer, response_path).is_file())

    def test_refuse_overwrite_and_nested_private(self):
        self.listener.mkdir()
        (self.listener / 'keep').write_text('preserve')
        with self.assertRaisesRegex(ValueError, 'existing'):
            blind.prepare(self.manifest_path, self.listener, self.organizer)
        self.assertEqual((self.listener / 'keep').read_text(), 'preserve')
        with self.assertRaisesRegex(ValueError, 'outside'):
            blind.prepare(self.manifest_path, self.listener, self.listener / 'key')

    def test_reveal_requires_valid_complete_responses(self):
        self.fake_prepare()
        response_path = self.listener / 'responses.csv'
        with self.assertRaisesRegex(ValueError, 'categorical'):
            blind.reveal(self.organizer, response_path)
        with response_path.open(newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            row.update({k: 'indistinguishable' for k in blind.FIELDS[1:5]})
            row.update(confidence='low', fatigue='none')
        def save(rows):
            with response_path.open('w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=blind.FIELDS)
                writer.writeheader()
                writer.writerows(rows)
        save(rows[:-1])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            blind.reveal(self.organizer, response_path)
        rows[0]['fatigue'] = 'invalid'
        save(rows)
        with self.assertRaisesRegex(ValueError, 'categorical'):
            blind.reveal(self.organizer, response_path)
        rows[0]['fatigue'] = 'not_assessed'
        save(rows)
        first_audio = self.listener / rows[0]['trial'] / 'A.wav'
        original = first_audio.read_bytes()
        first_audio.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'changed'):
            blind.reveal(self.organizer, response_path)
        first_audio.write_bytes(original)
        output = blind.reveal(self.organizer, response_path)
        self.assertTrue(output.is_file())
        with self.assertRaises(FileExistsError):
            blind.reveal(self.organizer, response_path)

    def test_verify_rejects_missing_audio_despite_ready_marker(self):
        self.fake_prepare()
        self.assertTrue((self.listener / 'READY').is_file())
        (self.listener / 'T0001' / 'A.wav').unlink()
        with self.assertRaisesRegex(ValueError, 'audio file is missing'):
            blind.verify(self.organizer)

    def test_verify_rejects_changed_audio_hash(self):
        self.fake_prepare()
        (self.listener / 'T0001' / 'B.wav').write_bytes(b'changed audio')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            blind.verify(self.organizer)

    def test_verify_requires_ready_marker(self):
        self.fake_prepare()
        (self.listener / 'READY').unlink()
        with self.assertRaisesRegex(ValueError, 'session is not ready'):
            blind.verify(self.organizer)

    def test_verify_cli_prints_only_neutral_counts(self):
        self.fake_prepare()
        output = StringIO()
        with patch('sys.argv', ['blind_listening.py', 'verify', '--organizer', str(self.organizer)]), \
                redirect_stdout(output):
            blind.main()
        self.assertEqual(output.getvalue(), 'ready=true trials=4 audio_files=8\n')
        self.assertNotIn('secret_', output.getvalue())
        self.assertNotIn('LUFS', output.getvalue())

    def test_mismatches_and_declared_common_interval(self):
        def validate(specs):
            with patch.object(blind, 'probe', side_effect=specs):
                return blind.validate(copy.deepcopy(self.manifest), self.root)
        a = {'rate': 8000, 'channels': 1, 'frames': 1040000}
        for b in [dict(a, rate=48000), dict(a, channels=2), dict(a, frames=1040001)]:
            with self.assertRaises(ValueError):
                validate([a, b])
        self.manifest['tracks'][0]['common_trim'] = {'start_sample': 0, 'frames': 1040000}
        self.assertTrue(validate([a, dict(a, frames=1040001)]))
        self.manifest['tracks'][0]['common_trim']['start_sample'] = 1
        with self.assertRaisesRegex(ValueError, 'trim'):
            validate([a, dict(a, frames=1040001)])

    def test_architecture_provenance_and_no_pooling(self):
        self.manifest['mode'] = 'equal_target_architecture'
        spec = {'rate': 8000, 'channels': 1, 'frames': 1040000}
        with patch.object(blind, 'probe', return_value=spec):
            with self.assertRaisesRegex(ValueError, 'architecture'):
                blind.validate(copy.deepcopy(self.manifest), self.root)
            for v in self.manifest['tracks'][0]['variants']:
                v['provenance'] = {'git_commit': 'a' * 40, 'source_sha256': 'b' * 64, 'target_lufs': -11}
            self.assertTrue(blind.validate(copy.deepcopy(self.manifest), self.root))
            self.manifest['tracks'][0]['variants'][0]['provenance']['target_lufs'] = -13
            with self.assertRaisesRegex(ValueError, 'same original'):
                blind.validate(copy.deepcopy(self.manifest), self.root)

    def test_old_modes_still_reject_short_only_excerpts(self):
        spec = {'rate': 8000, 'channels': 1, 'frames': 26 * 8000}
        for mode in ('historical_playback', 'equal_target_architecture'):
            manifest = copy.deepcopy(self.manifest)
            manifest['mode'] = mode
            manifest['tracks'][0]['excerpts'] = [
                {'id': '13s', 'kind': 'diagnostic', 'start_sample': 0, 'frames': 13 * 8000},
                {'id': '26s', 'kind': 'long', 'start_sample': 0, 'frames': 26 * 8000},
            ]
            if mode == 'equal_target_architecture':
                for variant in manifest['tracks'][0]['variants']:
                    variant['provenance'] = {'git_commit': 'a' * 40,
                                             'source_sha256': 'b' * 64,
                                             'target_lufs': -11.0}
            with patch.object(blind, 'probe', return_value=spec):
                with self.assertRaisesRegex(ValueError, '30s diagnostic|>=120s long'):
                    blind.validate(manifest, self.root)

    def test_procedural_diagnostic_accepts_13_and_26_seconds(self):
        self.manifest = self.procedural_manifest()
        self.write_manifest()
        self.fake_prepare()
        key = json.loads((self.organizer / 'key.private.json').read_text())
        self.assertEqual(key['mode'], 'procedural_diagnostic')
        self.assertEqual(len(key['trials']), 4)
        self.assertEqual(sum(t['identity'] for t in key['trials']), 2)
        self.assertTrue(all(member['gain_db'] <= 0
                            for trial in key['trials'] for member in trial['members'].values()))
        self.assertTrue(all(trial['members']['A']['sha256'] == trial['members']['B']['sha256']
                            for trial in key['trials'] if trial['identity']))
        instructions = (self.listener / 'INSTRUCTIONS.txt').read_text()
        self.assertIn('do not assess fatigue', instructions)
        self.assertIn('long-session claims', instructions)
        with (self.listener / 'responses.csv').open(newline='') as response_file:
            rows = list(csv.DictReader(response_file))
        self.assertTrue(all(row['fatigue'] == 'not_assessed' for row in rows))
        blind.verify(self.organizer)
        for row in rows:
            row.update({k: 'indistinguishable' for k in blind.FIELDS[1:5]})
            row['confidence'] = 'low'
        response_path = self.listener / 'responses.csv'
        with response_path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=blind.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        self.assertTrue(blind.reveal(self.organizer, response_path).is_file())

    def test_procedural_diagnostic_rejects_long_or_fatigue_excerpt_types(self):
        spec = {'rate': 8000, 'channels': 1, 'frames': 200 * 8000}
        base = self.procedural_manifest()
        for kind in ('long', 'fatigue'):
            manifest = copy.deepcopy(base)
            manifest['tracks'][0]['excerpts'][0]['kind'] = kind
            with patch.object(blind, 'probe', return_value=spec):
                with self.assertRaisesRegex(ValueError, '8 to 48 seconds'):
                    blind.validate(manifest, self.root)
        manifest = copy.deepcopy(base)
        manifest['tracks'][0]['excerpts'][0]['frames'] = 49 * 8000
        with patch.object(blind, 'probe', return_value=spec):
            with self.assertRaisesRegex(ValueError, '8 to 48 seconds'):
                blind.validate(manifest, self.root)

    def test_procedural_mode_requires_declared_provenance_not_metadata_claims(self):
        spec = {'rate': 8000, 'channels': 1, 'frames': 26 * 8000}
        base = self.procedural_manifest()
        mutations = [
            ('procedural', False),
            ('run_record', ''),
            ('git_commit', 'short'),
            ('source_sha256', 'bad'),
            ('git_commit', None),
            ('source_sha256', None),
        ]
        for field, value in mutations:
            manifest = copy.deepcopy(base)
            manifest['tracks'][0]['variants'][0]['provenance'][field] = value
            with patch.object(blind, 'probe', return_value=spec):
                with self.assertRaisesRegex(ValueError, 'procedural diagnostic requires'):
                    blind.validate(manifest, self.root)
        manifest = copy.deepcopy(base)
        manifest['tracks'][0]['variants'][1]['provenance']['source_sha256'] = 'c' * 64
        with patch.object(blind, 'probe', return_value=spec):
            with self.assertRaisesRegex(ValueError, 'same procedural source hash'):
                blind.validate(manifest, self.root)

    def test_procedural_diagnostic_reveal_rejects_fatigue_judgment(self):
        self.manifest = self.procedural_manifest()
        self.write_manifest()
        self.fake_prepare()
        response_path = self.listener / 'responses.csv'
        with response_path.open(newline='') as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            row.update({k: 'indistinguishable' for k in blind.FIELDS[1:5]})
            row['confidence'] = 'low'
        rows[0]['fatigue'] = 'mild'
        with response_path.open('w', newline='') as destination:
            writer = csv.DictWriter(destination, fieldnames=blind.FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(ValueError, 'fatigue must remain not_assessed'):
            blind.reveal(self.organizer, response_path)

    def test_shared_target_never_amplifies_and_preserves_peak_headroom(self):
        stats = [{'lufs': -18, 'true_peak_dbfs': 1}, {'lufs': -22, 'true_peak_dbfs': -3}]
        self.assertEqual(blind.matching_target(stats, -20), -22)
        self.assertAlmostEqual(blind.matching_target(stats[:1], -10), -19.2)
        with self.assertRaisesRegex(ValueError, 'amplification'):
            blind.encode(Path('unused'), Path('unused'), {}, .1)

    def test_failed_encoded_audit_removes_public_ready(self):
        self.write_manifest()
        for index, (loudness_b, peak) in enumerate([(-23., 0.), (-22.7, -3.)]):
            encoded = {}
            def encode(source, destination, excerpt, gain):
                destination.write_bytes(b'audio')
                encoded[destination] = excerpt['frames']
            def probe(path, **_):
                return {'rate': 8000, 'channels': 1, 'frames': encoded.get(path, 1040000)}
            def measure(path, excerpt=None):
                return {'lufs': (-23. if path.name == 'A.wav' else loudness_b) if path in encoded else -18.,
                        'true_peak_dbfs': peak if path in encoded else -3.}
            listener, organizer = self.root / f'failed-public-{index}', self.root / f'failed-private-{index}'
            with patch.object(blind, 'probe', side_effect=probe), patch.object(blind, 'measure', side_effect=measure), patch.object(blind, 'encode', side_effect=encode):
                with self.assertRaisesRegex(ValueError, 'clipping' if peak == 0 else 'loudness'):
                    blind.prepare(self.manifest_path, listener, organizer)
            self.assertFalse(listener.exists())

    def test_storage_preflight_leaves_no_output(self):
        self.write_manifest()
        with patch.object(blind, 'probe', return_value={'rate': 8000, 'channels': 1, 'frames': 1040000}), patch.object(blind.shutil, 'disk_usage', return_value=type('Usage', (), {'free': 0})()):
            with self.assertRaisesRegex(ValueError, 'storage'):
                blind.prepare(self.manifest_path, self.listener, self.organizer)
        self.assertFalse(self.listener.exists())
        self.assertFalse(self.organizer.exists())

    def test_real_ffmpeg_procedural_session(self):
        for variant, amplitude in zip(self.manifest['tracks'][0]['variants'], [4000, 2000]):
            samples = b''.join(struct.pack('<h', round(amplitude * math.sin(2 * math.pi * 220 * n / 8000))) for n in range(8000))
            source = self.root / variant['path']
            with wave.open(str(source), 'wb') as f:
                f.setnchannels(1)
                f.setsampwidth(2)
                f.setframerate(8000)
                f.writeframes(samples * 130)
        self.manifest['target_lufs'] = -26.
        self.write_manifest()
        blind.prepare(self.manifest_path, self.listener, self.organizer)
        key = json.loads((self.organizer / 'key.private.json').read_text())
        self.assertTrue((self.listener / 'READY').is_file())
        for trial in key['trials']:
            for slot, member in trial['members'].items():
                path = self.listener / trial['trial'] / f'{slot}.wav'
                self.assertEqual(blind.probe(path, expected_pcm24=True)['frames'], trial['excerpt']['frames'])
                self.assertLessEqual(abs(member['encoded']['lufs'] - trial['target_lufs']), .15)
                self.assertLess(member['encoded']['true_peak_dbfs'], 0)
                stdout, _ = blind.run(['ffprobe', '-v', 'error', '-show_format', '-of', 'json', str(path)])
                self.assertNotIn('tags', json.loads(stdout)['format'])


if __name__ == '__main__':
    unittest.main()
