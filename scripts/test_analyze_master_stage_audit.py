import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

MODULE = Path(__file__).with_name('analyze_master_stage_audit.py')
spec = importlib.util.spec_from_file_location('stage_analysis', MODULE)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)

from long_coherence_fixture import _write_float_wav


def metrics(channels):
    all_samples = np.concatenate(channels)
    rms = float(np.sqrt(np.mean(all_samples.astype(np.float64)**2)))
    peak = float(np.max(np.abs(all_samples)))
    return {'rms_dbfs': 20*math.log10(rms) if rms else -180.0,
            'sample_peak_dbfs': 20*math.log10(peak) if peak else -180.0,
            'lufs_integrated': -20.0, 'true_peak_dbtp': -1.0, 'crest_factor_db': 6.0}


def make_audit(root: Path, dynamic: bool = False, onsets=(2_000, 12_000), silence=False):
    audit_dir = root/'audit'
    audit_dir.mkdir()
    n = 24_000
    t = np.arange(n, dtype=np.float64) / 48_000
    x = (np.zeros(n, dtype=np.float32) if silence else
         (0.12*np.sin(2*np.pi*110*t) + 0.05*np.sin(2*np.pi*900*t)).astype(np.float32))
    for onset in onsets:
        x[onset:onset+120] += np.linspace(0.0, 0.65, 120, dtype=np.float32)
    stages = [('input', x.copy(), None), ('pre_shaped', x.copy(), None)]
    if dynamic:
        driven = x.copy()
        for onset in onsets:
            driven[onset:onset+960] *= 0.25
        stages.extend((('pre_limiter', driven, 0), ('post_limiter', driven*0.95, 0)))
        final_gain, limiter_passes, bypasses = 0.95, 1, 0
        limiter = {'kind': 'legacy', 'lookahead_samples': 96, 'hold_samples': 0, 'release_ms': 60.0}
        input_stage = 'pre_limiter'
    else:
        driven = x*0.5
        stages.extend((('gain_only_input', driven, 0), ('gain_only_output', driven.copy(), 0)))
        final_gain, limiter_passes, bypasses = 1.0, 0, 1
        limiter = None
        input_stage = 'gain_only_input'
    stages.extend((('pre_final_trim', stages[-1][1].copy(), None),
                   ('final_output', stages[-1][1]*final_gain, None)))
    record_rows = []
    final_hash = None
    for seq, (stage, signal, iteration) in enumerate(stages):
        channels = [signal, signal*0.9]
        wav_path = audit_dir/f'{seq:02}_{stage}.wav'
        _write_float_wav(wav_path, np.stack(channels, axis=1))
        meta = {'stage': stage, 'iteration': iteration,
                'limiter': limiter if stage in ('pre_limiter', 'post_limiter') else None,
                'base_drive_db': 6.0 if stage in ('pre_limiter', 'post_limiter', 'gain_only_input', 'gain_only_output') else None,
                'quiet_makeup_curve_enabled': dynamic and stage in ('pre_limiter', 'post_limiter'),
                'effective_ceiling_linear': 0.89 if stage in ('pre_limiter','post_limiter','gain_only_input','gain_only_output') else None,
                'gain_reduction_db': (0.2 if stage == 'post_limiter' else 0.0 if stage == 'gain_only_output' else None),
                'final_trim_gain_linear': final_gain if stage == 'final_output' else None}
        fnv = analysis.fnv_planar([np.asarray(c, dtype=np.float32) for c in channels])
        row = {'sequence': seq, 'meta': meta, 'rate': 48_000, 'channels': 2, 'frames': n,
               'metrics': metrics(channels), 'planar_f32le_fnv1a64': fnv,
               'snapshot': {'file': wav_path.name, 'bytes': wav_path.stat().st_size}}
        record_rows.append(row)
        final_hash = fnv
    trace = audit_dir/'observations.jsonl'
    trace.write_text(''.join(json.dumps(row)+'\n' for row in record_rows))
    (audit_dir/'master_audit.json').write_text(json.dumps({
        'schema': 'master-stage-audit-v1', 'complete': True, 'rate': 48_000,
        'channels': 2, 'frames': n, 'record_count': len(stages),
        'iteration_count': 1, 'limiter_pass_count': limiter_passes,
        'gain_only_bypass_count': bypasses, 'last_retained_iteration': 0,
        'snapshots_enabled': True,
        'snapshot_wav_bytes': sum(row['snapshot']['bytes'] for row in record_rows),
        'final_planar_f32le_fnv1a64': final_hash,
        'result': {'before_metrics': record_rows[0]['metrics'],
                   'after_metrics': record_rows[-1]['metrics']},
    }))
    fixed_path = root/'fixed.json'
    input_hash = analysis.digest(audit_dir/'00_input.wav')
    fixed_path.write_text(json.dumps({'schema': 'master-fixed-onsets-v1',
        'sample_rate_hz': 48_000, 'input_sha256': input_hash,
        'samples': list(onsets), 'transient_ms': 20,
        'sustain_start_ms': 80, 'sustain_end_ms': 200}))
    return audit_dir, fixed_path


class MasterStageAnalysisTests(unittest.TestCase):
    def test_gain_only_scalar_changes_level_not_fixed_transient_sustain_ratio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit_dir, fixed = make_audit(root, dynamic=False)
            result = analysis.analyze(audit_dir, fixed, root/'analysis.json')
            pair = next(x for x in result['comparisons'] if x['comparison'] == 'gain_only_bypass_iteration_0')
            self.assertAlmostEqual(pair['rms_ratio_to_from'], 1.0, places=7)
            self.assertAlmostEqual(pair['transient_sustain_ratio_multiplier'], 1.0, places=6)
            self.assertAlmostEqual(result['stage_records'][0]['measured']['rms_all_channels'] /
                                   result['stage_records'][2]['measured']['rms_all_channels'], 2.0, places=6)
            self.assertAlmostEqual(pair['input_rms_match_gain_linear_for_to_stage_only'], 2.0, places=5)

    def test_time_varying_transient_attenuation_changes_ratio_without_rms_matching_audio(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit_dir, fixed = make_audit(root, dynamic=True)
            result = analysis.analyze(audit_dir, fixed, root/'analysis.json')
            pair = next(x for x in result['comparisons'] if x['comparison'] == 'pre_shaped_to_pre_limiter_iteration_0')
            self.assertLess(pair['transient_sustain_ratio_to'], pair['transient_sustain_ratio_from'])
            self.assertNotAlmostEqual(pair['rms_ratio_to_from'], 1.0, places=3)
            self.assertIn('input_rms_match_gain_linear_for_to_stage_only', pair)

    def test_wrong_input_binding_and_incomplete_summary_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit_dir, fixed = make_audit(root, dynamic=False)
            onset = json.loads(fixed.read_text())
            onset['input_sha256'] = '0'*64
            fixed.write_text(json.dumps(onset))
            with self.assertRaisesRegex(ValueError, 'not bound'):
                analysis.analyze(audit_dir, fixed, root/'wrong.json')
            second = root/'second'
            second.mkdir()
            audit_dir, fixed = make_audit(second, dynamic=False)
            summary_path = audit_dir/'master_audit.json'
            summary = json.loads(summary_path.read_text()); summary['complete'] = False
            summary_path.write_text(json.dumps(summary))
            with self.assertRaisesRegex(ValueError, 'not complete'):
                analysis.analyze(audit_dir, fixed, root/'incomplete.json')
            fresh = root/'halted'; fresh.mkdir()
            audit_dir, fixed = make_audit(fresh, dynamic=False)
            (audit_dir/'HALTED.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'halted'):
                analysis.analyze(audit_dir, fixed, fresh/'halted-analysis.json')

    def test_misordered_trace_and_snapshot_hash_drift_are_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit_dir, fixed = make_audit(root, dynamic=False)
            trace = audit_dir/'observations.jsonl'
            trace.write_text('\n'.join(reversed(trace.read_text().splitlines()))+'\n')
            with self.assertRaisesRegex(ValueError, 'start at input'):
                analysis.analyze(audit_dir, fixed, root/'misordered.json')
            other = root/'other'; other.mkdir()
            audit_dir, fixed = make_audit(other, dynamic=False)
            snapshot = audit_dir/'02_gain_only_input.wav'
            data = bytearray(snapshot.read_bytes()); data[-1] ^= 1; snapshot.write_bytes(data)
            with self.assertRaisesRegex(ValueError, 'fingerprint mismatch'):
                analysis.analyze(audit_dir, fixed, other/'drift.json')

    def test_silence_accepts_no_onsets_and_emits_null_event_statistics(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            audit_dir, fixed = make_audit(root, dynamic=False, onsets=(), silence=True)
            result = analysis.analyze(audit_dir, fixed, root/'analysis.json')
            self.assertEqual(result['fixed_onsets']['record_count'], 0)
            for record in result['stage_records']:
                event = record['measured']['fixed_onsets']
                self.assertEqual(event['record_count'], 0)
                self.assertIsNone(event['transient_mean_square'])
                self.assertIsNone(event['sustain_mean_square'])
                self.assertIsNone(event['transient_to_sustain_energy_ratio'])
                self.assertIsNone(event['transient_rms'])
                self.assertIsNone(event['sustain_rms'])
            self.assertTrue(all(x['transient_sustain_ratio_from'] is None
                                for x in result['comparisons']))


if __name__ == '__main__':
    unittest.main()
