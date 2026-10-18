"""Bounded local tests; real FFmpeg smoke uses procedural audio only."""
import copy
import csv
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
        with patch.object(blind, 'probe', side_effect=probe), patch.object(blind, 'measure', side_effect=measure), patch.object(blind, 'encode', side_effect=encode):
            blind.prepare(self.manifest_path, self.listener, self.organizer)

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
