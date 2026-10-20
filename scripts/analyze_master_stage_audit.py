#!/usr/bin/env python3
"""Verify and analyze a completed native finalizer stage audit offline."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_coherence_quality import wav_info
from roach_ears.audio_io import read_wav

RATE = 48_000
MAX_FRAMES = 48 * RATE
MAX_RECORDS = 32


def digest(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def finite(value) -> bool:
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def fnv_planar(channels) -> str:
    h = 0xcbf29ce484222325
    for channel in channels:
        for byte in channel.astype('<f4', copy=False).tobytes():
            h ^= byte
            h = (h * 0x100000001b3) & 0xffffffffffffffff
    return f'{h:016x}'


def _rms(channels) -> float:
    arrays = [np.asarray(channel, dtype=np.float64) for channel in channels]
    energy = sum(float(np.dot(channel, channel)) for channel in arrays)
    count = sum(channel.size for channel in arrays)
    return math.sqrt(energy / count) if count else 0.0


def _window_energy(channels, onsets: list[int], begin_ms: float, end_ms: float) -> float | None:
    begin = int(round(begin_ms * RATE / 1000.0))
    end = int(round(end_ms * RATE / 1000.0))
    total, count = 0.0, 0
    for onset in onsets:
        lo, hi = onset + begin, onset + end
        for channel in channels:
            part = np.asarray(channel[lo:hi], dtype=np.float64)
            total += float(np.dot(part, part))
            count += part.size
    return total / count if count else None


def _snapshot_metrics(path: Path, expected: dict, onsets: list[int], windows: dict) -> dict:
    info = wav_info(path)
    require(info['rate'] == RATE and 1 <= info['channels'] <= 2
            and 0 < info['frames'] <= MAX_FRAMES and info['format'] == 3
            and info['bits_per_sample'] == 32,
            f'{path.name}: expected bounded 48 kHz float32 mono/stereo WAV')
    audio = read_wav(path)
    channels = audio.samples
    require(audio.sample_rate == RATE and audio.channels == info['channels']
            and audio.num_samples == info['frames'], f'{path.name}: decoder/header geometry mismatch')
    channel_arrays = [np.asarray(channel, dtype=np.float64) for channel in channels]
    require(all(np.isfinite(channel).all() for channel in channel_arrays),
            f'{path.name}: nonfinite audio')
    frames = info['frames']
    for onset in onsets:
        require(0 <= onset < frames, f'{path.name}: onset outside audio')
        for start_ms, end_ms in windows.values():
            end = onset + int(round(end_ms * RATE / 1000.0))
            require(end <= frames, f'{path.name}: fixed onset window exceeds snapshot')
    transient = _window_energy(channels, onsets,
        windows['transient'][0], windows['transient'][1])
    sustain = _window_energy(channels, onsets,
        windows['sustain'][0], windows['sustain'][1])
    peak = max((float(np.max(np.abs(c))) for c in channel_arrays), default=0.0)
    rms = _rms(channel_arrays)
    per_channel = {}
    for idx, channel in enumerate(channels):
        key = 'left' if idx == 0 and len(channels) == 2 else 'right' if idx == 1 else 'mono'
        values = np.asarray(channel, dtype=np.float64)
        power = float(np.dot(values, values) / values.size)
        per_channel[key] = {'mean_square': power, 'rms': math.sqrt(power)}
    native = expected['metrics']
    if 'rms_dbfs' in native and rms > 0:
        measured_db = 20.0 * math.log10(rms)
        require(abs(measured_db - float(native['rms_dbfs'])) < 2e-4,
                f'{path.name}: stored native RMS does not match snapshot')
    if 'sample_peak_dbfs' in native and peak > 0:
        measured_db = 20.0 * math.log10(peak)
        require(abs(measured_db - float(native['sample_peak_dbfs'])) < 2e-4,
                f'{path.name}: stored native sample peak does not match snapshot')
    return {
        'snapshot_sha256': digest(path), 'fnv1a_planar_f32le': fnv_planar(channels),
        'geometry': {'sample_rate_hz': RATE, 'channels': len(channels), 'frames': frames},
        'mean_square_all_channels': rms*rms, 'rms_all_channels': rms,
        'peak_sample': peak, 'sampled_crest_ratio_peak_over_rms': peak/rms if rms > 0 else None,
        'mean_square_by_channel': per_channel,
        'fixed_onsets': {'record_count': len(onsets), 'transient_ms': list(windows['transient']),
            'sustain_ms': list(windows['sustain']), 'transient_mean_square': transient,
            'sustain_mean_square': sustain,
            'transient_to_sustain_energy_ratio': (transient/sustain
                if transient is not None and sustain is not None and sustain > 0 else None),
            'transient_rms': math.sqrt(transient) if transient is not None else None,
            'sustain_rms': math.sqrt(sustain) if sustain is not None else None},
        'native_metrics': native,
        'qualification': 'fixed-onset transient/sustain energy proxies; onset list supplied by caller, not redetected'
    }


def _stage(row: dict) -> str:
    meta = row.get('meta', {})
    return meta.get('stage')


def _pair(rows: list[dict], index: int, first: str, second: str) -> tuple[dict, dict]:
    require(index + 1 < len(rows), f'missing {second} after {first}')
    a, b = rows[index], rows[index+1]
    require(_stage(a) == first and _stage(b) == second,
            f'expected ordered pair {first}/{second}, got {_stage(a)}/{_stage(b)}')
    return a, b


def _compare(name: str, a: dict, b: dict, stats: dict, input_rms: float) -> dict:
    x, y = stats[a['sequence']], stats[b['sequence']]
    xa, xb = x['fixed_onsets']['transient_to_sustain_energy_ratio'], y['fixed_onsets']['transient_to_sustain_energy_ratio']
    ra, rb = x['rms_all_channels'], y['rms_all_channels']
    return {'comparison': name, 'from_sequence': a['sequence'], 'to_sequence': b['sequence'],
        'from_stage': _stage(a), 'to_stage': _stage(b),
        'rms_ratio_to_from': rb/ra if ra > 0 else None,
        'rms_ratio_to_input': rb/input_rms if input_rms > 0 else None,
        'input_rms_match_gain_linear_for_to_stage_only': input_rms/rb if rb > 0 else None,
        'transient_sustain_ratio_from': xa, 'transient_sustain_ratio_to': xb,
        'transient_sustain_ratio_multiplier': xb/xa if xa is not None and xa > 0 and xb is not None else None,
        'limiter_reduction_db': b.get('meta', {}).get('gain_reduction_db'),
        'final_trim_gain_linear': b.get('meta', {}).get('final_trim_gain_linear')}


def analyze(audit_dir: Path, onset_path: Path, out: Path) -> dict:
    require(not out.exists(), f'refusing existing output: {out}')
    require(not (audit_dir/'HALT').exists() and not (audit_dir/'HALTED.json').exists(),
            'audit directory is halted')
    summary_path, trace_path = audit_dir/'master_audit.json', audit_dir/'observations.jsonl'
    require(summary_path.is_file() and trace_path.is_file(), 'audit summary or trace is missing')
    summary = json.loads(summary_path.read_text())
    fixed = json.loads(onset_path.read_text())
    require(summary.get('schema') == 'master-stage-audit-v1' and summary.get('complete') is True,
            'audit summary is not complete')
    require(fixed.get('schema') == 'master-fixed-onsets-v1', 'unsupported fixed-onsets schema')
    require(fixed.get('sample_rate_hz') == RATE and fixed.get('transient_ms') == 20
            and fixed.get('sustain_start_ms') == 80 and fixed.get('sustain_end_ms') == 200,
            'fixed-onset timing protocol mismatch')
    onsets = fixed.get('samples')
    require(isinstance(onsets, list) and len(onsets) <= 128
            and all(type(x) is int and x >= 0 for x in onsets)
            and onsets == sorted(set(onsets)), 'fixed onset samples must be 1..128 sorted unique nonnegative integers')
    rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
    require(1 <= len(rows) <= MAX_RECORDS and summary.get('record_count') == len(rows),
            'trace record count differs or exceeds cap')
    require(all(finite(r) for r in rows) and finite(summary) and finite(fixed), 'nonfinite JSON value')
    require(rows[0].get('sequence') == 0 and _stage(rows[0]) == 'input', 'trace must start at input')
    input_snapshot = rows[0].get('snapshot')
    require(isinstance(input_snapshot, dict), 'WAV snapshots are required')
    input_name = input_snapshot.get('file')
    require(isinstance(input_name, str) and Path(input_name).name == input_name,
            'input snapshot path must be a direct child filename')
    require(input_name == '00_input.wav', 'sequence-zero snapshot must be named 00_input.wav')
    input_path = audit_dir/input_name
    require(digest(input_path) == fixed.get('input_sha256'), 'fixed onset list is not bound to input snapshot SHA-256')
    channels = summary.get('channels')
    frames = summary.get('frames')
    require(summary.get('rate') == RATE and channels in (1, 2)
            and type(frames) is int and 0 < frames <= MAX_FRAMES, 'summary geometry outside bounds')
    require(summary.get('snapshots_enabled') is True, 'audit summary must declare snapshots enabled')
    windows = {'transient': (0.0, float(fixed['transient_ms'])),
               'sustain': (float(fixed['sustain_start_ms']), float(fixed['sustain_end_ms']))}
    stats, snapshot_receipts, snapshot_names = {}, [], set()
    for idx, row in enumerate(rows):
        require(row.get('sequence') == idx, 'trace sequence is not contiguous and ordered')
        require(row.get('rate') == RATE and row.get('channels') == channels and row.get('frames') == frames,
                f'record {idx}: geometry mismatch')
        snap = row.get('snapshot')
        require(isinstance(snap, dict), f'record {idx}: snapshot required')
        name = snap.get('file')
        require(isinstance(name, str) and Path(name).name == name, f'record {idx}: unsafe snapshot filename')
        require(name == f'{idx:02}_{_stage(row)}.wav', f'record {idx}: snapshot filename differs from stage/sequence')
        require(name not in snapshot_names, f'record {idx}: duplicate snapshot filename')
        snapshot_names.add(name)
        path = audit_dir/name
        require(path.is_file() and path.stat().st_size == snap.get('bytes'), f'record {idx}: snapshot file/size mismatch')
        measured = _snapshot_metrics(path, row, onsets, windows)
        require(measured['geometry'] == {'sample_rate_hz': RATE, 'channels': channels, 'frames': frames},
                f'record {idx}: snapshot geometry differs from summary')
        require(measured['fnv1a_planar_f32le'] == row.get('planar_f32le_fnv1a64'),
                f'record {idx}: snapshot FNV fingerprint mismatch')
        stats[idx] = measured
        snapshot_receipts.append({'sequence': idx, 'file': name, 'bytes': snap['bytes'],
                                  'sha256': measured['snapshot_sha256'],
                                  'fnv1a_planar_f32le': measured['fnv1a_planar_f32le']})
    require(stats[0]['snapshot_sha256'] == fixed['input_sha256'], 'input snapshot hash mismatch')
    require(rows[-1]['meta']['stage'] == 'final_output', 'trace does not end at final_output')
    require(rows[-1]['planar_f32le_fnv1a64'] == summary.get('final_planar_f32le_fnv1a64'),
            'final output fingerprint differs from summary')
    require(stats[len(rows)-1]['fnv1a_planar_f32le'] == summary.get('final_planar_f32le_fnv1a64'),
            'final snapshot does not match summary fingerprint')
    require(sum(x['bytes'] for x in snapshot_receipts) == summary.get('snapshot_wav_bytes'),
            'snapshot byte total differs from summary')
    require(rows[0]['metrics'] == summary.get('result', {}).get('before_metrics'),
            'summary before metrics differ from input snapshot record')
    require(rows[-1]['metrics'] == summary.get('result', {}).get('after_metrics'),
            'summary after metrics differ from final snapshot record')

    require(len(rows) >= 5 and _stage(rows[1]) == 'pre_shaped', 'missing pre_shaped observation')
    comparisons = [_compare('input_to_pre_shaped', rows[0], rows[1], stats, stats[0]['rms_all_channels'])]
    i = 2
    iterations = []
    while i < len(rows)-2:
        stage = _stage(rows[i])
        if stage == 'pre_limiter':
            a, b = _pair(rows, i, 'pre_limiter', 'post_limiter')
            kind = a['meta'].get('limiter')
            require(isinstance(kind, dict), f'iteration {a["sequence"]}: limiter metadata missing')
            require(a['meta'].get('iteration') == b['meta'].get('iteration'), 'limiter pair iteration mismatch')
            require(type(a['meta'].get('iteration')) is int, 'limiter iteration must be integer')
            require(a['meta'].get('limiter') == b['meta'].get('limiter'), 'limiter spec changed within pair')
            for key in ('base_drive_db', 'quiet_makeup_curve_enabled', 'effective_ceiling_linear'):
                require(a['meta'].get(key) == b['meta'].get(key), f'limiter pair {key} mismatch')
            require(a['meta'].get('gain_reduction_db') is None
                    and isinstance(b['meta'].get('gain_reduction_db'), (int, float)),
                    'limiter reduction metadata misplaced')
            iterations.append(a['meta']['iteration'])
            comparisons.append(_compare(f'pre_shaped_to_pre_limiter_iteration_{a["meta"]["iteration"]}',
                                        rows[1], a, stats, stats[0]['rms_all_channels']))
            comparisons.append(_compare(f'limiter_iteration_{a["meta"]["iteration"]}', a, b, stats,
                                        stats[0]['rms_all_channels']))
            i += 2
        elif stage == 'gain_only_input':
            a, b = _pair(rows, i, 'gain_only_input', 'gain_only_output')
            require(a['meta'].get('limiter') is None and b['meta'].get('limiter') is None,
                    'gain-only bypass must declare no limiter')
            require(a['meta'].get('iteration') == b['meta'].get('iteration'), 'gain-only pair iteration mismatch')
            require(type(a['meta'].get('iteration')) is int, 'gain-only iteration must be integer')
            for key in ('base_drive_db', 'quiet_makeup_curve_enabled', 'effective_ceiling_linear'):
                require(a['meta'].get(key) == b['meta'].get(key), f'gain-only pair {key} mismatch')
            require(b['meta'].get('gain_reduction_db') == 0.0, 'gain-only bypass must report zero limiter reduction')
            require(a['planar_f32le_fnv1a64'] == b['planar_f32le_fnv1a64'],
                    'gain-only bypass changed waveform')
            iterations.append(a['meta']['iteration'])
            comparisons.append(_compare(f'pre_shaped_to_gain_only_input_iteration_{a["meta"]["iteration"]}',
                                        rows[1], a, stats, stats[0]['rms_all_channels']))
            comparisons.append(_compare(f'gain_only_bypass_iteration_{a["meta"]["iteration"]}', a, b,
                                        stats, stats[0]['rms_all_channels']))
            i += 2
        else:
            break
    require(iterations and iterations == list(range(len(iterations))), 'retry iterations are not sequential from zero')
    require(summary.get('iteration_count') == len(iterations)
            and summary.get('limiter_pass_count') == sum(_stage(rows[j]) == 'pre_limiter' for j in range(2, i, 2))
            and summary.get('gain_only_bypass_count') == sum(_stage(rows[j]) == 'gain_only_input' for j in range(2, i, 2)),
            'summary retry counts differ from trace')
    require(summary.get('last_retained_iteration') == iterations[-1], 'last retained iteration mismatch')
    require(i + 2 == len(rows), 'unexpected or missing records after retry sequence')
    a, b = _pair(rows, i, 'pre_final_trim', 'final_output')
    require(a['meta'].get('final_trim_gain_linear') is None
            and isinstance(b['meta'].get('final_trim_gain_linear'), (int, float)),
            'final trim metadata misplaced')
    comparisons.append(_compare('retained_output_to_pre_final_trim', rows[i-1], a, stats,
                                stats[0]['rms_all_channels']))
    comparisons.append(_compare('pre_final_trim_to_final_output', a, b, stats,
                                stats[0]['rms_all_channels']))
    final_trim = float(b['meta']['final_trim_gain_linear'])
    require(final_trim > 0 and math.isfinite(final_trim), 'invalid final trim gain')

    compact_rows = []
    for row in rows:
        idx = row['sequence']
        compact_rows.append({'sequence': idx, 'meta': row['meta'], 'snapshot': snapshot_receipts[idx],
                             'measured': stats[idx], 'native_metrics': row['metrics']})
    master_file = Path(__file__).resolve().parents[1]/'src'/'master.rs'
    audit_file = Path(__file__).resolve().parents[1]/'src'/'master_audit.rs'
    result = {
        'schema': 'master-stage-audit-analysis-v1',
        'scope': 'offline comparison of saved snapshots; fixed caller onsets; no waveform modification or onset redetection',
        'provenance': {'audit_dir': str(audit_dir.resolve()),
            'master_audit_json_sha256': digest(summary_path), 'observations_jsonl_sha256': digest(trace_path),
            'fixed_onsets_path': str(onset_path.resolve()), 'fixed_onsets_sha256': digest(onset_path),
            'analysis_source_sha256': digest(Path(__file__)),
            'analysis_checkout_master_source_sha256': digest(master_file),
            'analysis_checkout_master_audit_source_sha256': digest(audit_file),
            'wav_header_reader_source_sha256': digest(Path(__file__).with_name('run_coherence_quality.py')),
            'wav_decoder_source_sha256': digest(HERE/'roach_ears'/'audio_io.py'),
            'source_hash_scope': 'local analysis-worktree source hashes; not authenticated source provenance for the compiled native binary',
            'native_command_argv_descriptive_only': summary.get('command_argv'),
            'snapshot_sha256': snapshot_receipts},
        'geometry': {'sample_rate_hz': RATE, 'channels': channels, 'frames': frames,
                     'record_count': len(rows), 'iteration_count': len(iterations),
                     'last_retained_iteration': iterations[-1]},
        'fixed_onsets': {'record_count': len(onsets),
            'input_snapshot_sha256': fixed['input_sha256'], 'samples': onsets,
            'transient_ms': 20, 'sustain_ms': [80, 200],
            'selection': 'supplied onset sample positions; no redetection'},
        'stage_records': compact_rows,
        'comparisons': comparisons,
        'final_trim_gain_linear': final_trim,
        'interpretation_limits': [
            'Each retry input is compared with pre_shaped independently; retries are not treated as sequential production stages.',
            'quiet_makeup_curve_enabled identifies a time-varying gain mode; it does not encode the framewise curve. Saved waveforms permit measurement, while JSON metadata alone cannot reconstruct that curve.',
            'Level-match values are hypothetical scalar gains to the first input RMS; no audio was altered for comparison.',
            'Transient/sustain ratios use identical fixed windows and can change under nonlinear/time-varying gain; they are proxies, not perceptual or mastering quality scores.',
            'FNV is used only to verify the audit implementation fingerprint; SHA-256 receipts bind saved files.'
        ]
    }
    require(finite(result), 'analysis produced nonfinite values')
    require(out.parent.is_dir(), 'output parent directory must exist')
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    result['analysis_output_sha256'] = digest(out)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit-dir', required=True, type=Path)
    parser.add_argument('--fixed-onsets', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.audit_dir, args.fixed_onsets, args.out)
    print(json.dumps({'out': str(args.out), 'sha256': result['analysis_output_sha256'],
                      'records': result['geometry']['record_count'],
                      'comparisons': len(result['comparisons'])}, indent=2))


if __name__ == '__main__':
    main()
