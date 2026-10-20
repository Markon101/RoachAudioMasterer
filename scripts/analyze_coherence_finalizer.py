#!/usr/bin/env python3
"""Offline receipt-checked analysis of the eight native finalizer outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
from pathlib import Path

import numpy as np

if __package__:
    from .analyze_long_coherence import compare, rms, waveform_difference, RATE
    from .roach_ears.audio_io import read_wav
else:
    from analyze_long_coherence import compare, rms, waveform_difference, RATE
    from roach_ears.audio_io import read_wav


ARMS = ('restart', 'restart_projected', 'continuous', 'continuous_projected')
VIEWS = ('view0', 'view3')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def finite_json(value) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(finite_json(child) for child in value.values())
    if isinstance(value, list):
        return all(finite_json(child) for child in value)
    return True


def repo_root(session: Path) -> Path:
    parents = session.resolve().parents
    require(len(parents) >= 3 and parents[1].name == 'runs',
            'session must be under <repo>/runs/<coherence-run>/<session>')
    return parents[2]


def find_protocol(root: Path, run_name: str, expected_sha: str) -> tuple[Path, dict]:
    artifact_dir = root / 'artifacts' / run_name
    matches = []
    for path in artifact_dir.glob('finalizer*protocol.json'):
        if digest(path) == expected_sha:
            matches.append((path, json.loads(path.read_text())))
    require(len(matches) == 1, 'COMPLETE protocol hash does not identify exactly one retained protocol')
    return matches[0]


def arg_value(command: list[str], flag: str) -> str:
    require(flag in command and command.index(flag) + 1 < len(command),
            f'native command is missing {flag}')
    return command[command.index(flag) + 1]


def wav_float32_geometry(path: Path) -> dict:
    """Read small RIFF headers only; sample data is validated separately."""
    with path.open('rb') as source:
        header = source.read(12)
        require(len(header) == 12 and header[:4] == b'RIFF' and header[8:] == b'WAVE',
                f'not RIFF/WAVE: {path}')
        riff_size = struct.unpack_from('<I', header, 4)[0]
        fmt = None
        data_bytes = None
        while True:
            chunk = source.read(8)
            if len(chunk) != 8:
                break
            name, size = chunk[:4], struct.unpack_from('<I', chunk, 4)[0]
            if name == b'fmt ':
                fmt = source.read(size)
            elif name == b'data':
                data_bytes = size
                source.seek(size + (size & 1), 1)
            else:
                source.seek(size + (size & 1), 1)
            if size & 1 and name == b'fmt ':
                source.seek(1, 1)
        require(fmt is not None and len(fmt) >= 16 and data_bytes is not None,
                f'missing WAV fmt/data: {path}')
        tag, channels, rate, _, align, bits = struct.unpack_from('<HHIIHH', fmt)
        format_name = 'ieee_float' if tag == 3 else None
        if tag == 0xFFFE and len(fmt) >= 40:
            extension_size = struct.unpack_from('<H', fmt, 16)[0]
            valid_bits = struct.unpack_from('<H', fmt, 18)[0]
            float_guid = bytes.fromhex('0300000000001000800000aa00389b71')
            if extension_size >= 22 and valid_bits == 32 and fmt[24:40] == float_guid:
                format_name = 'ieee_float_extensible'
        require(format_name is not None and bits == 32 and align == channels * 4,
                f'mastered WAV is not packed IEEE float32: {path}')
        frames, remainder = divmod(data_bytes, align)
        require(remainder == 0 and path.stat().st_size == 8 + riff_size,
                f'invalid WAV frame/file geometry: {path}')
        return {'format': format_name, 'bits_per_sample': bits, 'channels': channels,
                'rate': rate, 'frames': frames, 'data_bytes': data_bytes}


def read_stereo(path: Path, frames: int) -> np.ndarray:
    geometry = wav_float32_geometry(path)
    require((geometry['rate'], geometry['channels'], geometry['frames']) == (RATE, 2, frames),
            f'expected 48 kHz stereo {frames}-frame float WAV: {path}')
    audio = read_wav(path)
    samples = np.asarray(audio.samples, dtype=np.float64)
    require(audio.sample_rate == RATE and samples.shape == (2, frames),
            f'decoded WAV geometry mismatch: {path}')
    require(np.isfinite(samples).all(), f'nonfinite samples in {path}')
    return samples


def stereo_metrics(audio: np.ndarray) -> dict:
    x = np.asarray(audio, dtype=np.float64)
    require(x.ndim == 2 and x.shape[0] == 2 and x.shape[1] > 0 and np.isfinite(x).all(),
            'stereo metrics require finite nonempty [2,samples] audio')
    left, right = x
    if np.std(left) > 1e-15 and np.std(right) > 1e-15:
        correlation = float(np.corrcoef(left, right)[0, 1])
    else:
        correlation = None
    mid = (left + right) / math.sqrt(2.0)
    side = (left - right) / math.sqrt(2.0)
    return {'left_right_correlation': correlation,
            'mid_rms': rms(mid), 'side_rms': rms(side)}


def event_window_metrics(audio: np.ndarray, onset_local: int, rate: int = RATE) -> dict:
    """Fixed attack [-20,+80) ms and sustain [+100,+300) ms source windows."""
    x = np.asarray(audio, dtype=np.float64)
    attack_start = onset_local - int(round(0.020 * rate))
    attack_end = onset_local + int(round(0.080 * rate))
    sustain_start = onset_local + int(round(0.100 * rate))
    sustain_end = onset_local + int(round(0.300 * rate))
    require(attack_start >= 0 and sustain_end <= x.shape[1],
            'source onset window extends outside audio')
    views = {'left': x[0], 'right': x[1],
             'mid': (x[0] + x[1]) / math.sqrt(2.0),
             'side': (x[0] - x[1]) / math.sqrt(2.0)}
    rows = {}
    for name, channel in views.items():
        attack = channel[attack_start:attack_end]
        sustain = channel[sustain_start:sustain_end]
        attack_rms, sustain_rms = rms(attack), rms(sustain)
        ratio_db = (20.0 * math.log10(attack_rms / sustain_rms)
                    if sustain_rms > 1e-15 else None)
        rows[name] = {
            'attack_rms': attack_rms, 'attack_peak': float(np.max(np.abs(attack))),
            'sustain_rms': sustain_rms, 'sustain_peak': float(np.max(np.abs(sustain))),
            'attack_to_sustain_db': ratio_db,
        }
    return {'onset_local_sample': onset_local,
            'attack_window_samples': [attack_start, attack_end],
            'sustain_window_samples': [sustain_start, sustain_end],
            'channels': rows}


def input_referred_attenuation(y0: np.ndarray, y3: np.ndarray, x: np.ndarray) -> dict:
    """Match only the fitted scalar projection on a shared input, by attenuation."""
    x = np.asarray(x, dtype=np.float64)
    y0, y3 = np.asarray(y0, dtype=np.float64), np.asarray(y3, dtype=np.float64)
    require(x.shape == y0.shape == y3.shape and x.ndim == 2 and x.shape[0] == 2,
            'input-referred gain control requires aligned stereo arrays')
    denom = float(np.sum(x * x))
    if not math.isfinite(denom) or denom <= 1e-20:
        return {'available': False, 'reason': 'shared input energy is null'}
    g0 = float(np.sum(y0 * x) / denom)
    g3 = float(np.sum(y3 * x) / denom)
    if not (math.isfinite(g0) and math.isfinite(g3) and g0 > 0.0 and g3 > 0.0):
        return {'available': False, 'reason': 'nonpositive or nonfinite input-referred fitted gain',
                'fitted_gain': {'view0': g0, 'view3': g3}}
    common = min(g0, g3)
    f0, f3 = common / g0, common / g3
    matched0, matched3 = y0 * f0, y3 * f3
    return {
        'available': True,
        'method': 'scalar least-squares output gain dot(y,x)/dot(x,x) against the same aligned procedural input; attenuate higher fitted gain to lower; no boost and no target-based scaling',
        'fitted_gain': {'view0': g0, 'view3': g3},
        'common_fitted_gain': common,
        'attenuation_factor': {'view0': f0, 'view3': f3},
        'attenuation_db': {'view0': 20.0 * math.log10(f0), 'view3': 20.0 * math.log10(f3)},
        'layout_difference_after_attenuation': waveform_difference(matched0, matched3),
    }


def validate_protocol_and_receipt(session: Path, fixture: Path) -> tuple[dict, dict, dict, Path, Path]:
    complete_path = session / 'COMPLETE.json'
    require(complete_path.is_file(), 'finalizer session lacks COMPLETE.json')
    complete = json.loads(complete_path.read_text())
    require(complete.get('schema') in {'coherence-finalizer-complete-v1',
                                       'coherence-finalizer-stress-complete-v1'},
            'unsupported finalizer completion schema')
    root = repo_root(session)
    protocol_path, protocol = find_protocol(root, session.parent.name,
                                            complete.get('protocol_sha256', ''))
    require(protocol.get('schema') in {'coherence-finalizer-condition-v1',
                                       'coherence-finalizer-limiter-stress-v1'},
            'unsupported finalizer protocol schema')
    require(protocol.get('promotion') is False and protocol.get('source_selection') ==
            'all predeclared arms andviews, no target selection',
            'protocol does not declare all-arm non-target selection')
    is_stress = protocol.get('schema') == 'coherence-finalizer-limiter-stress-v1'
    require((complete.get('schema') == 'coherence-finalizer-stress-complete-v1') == is_stress,
            'COMPLETE and protocol stress schemas disagree')
    require(protocol.get('arms') == list(ARMS) and protocol.get('views') == list(VIEWS),
            'protocol arm/view declaration changed')
    binary_path = Path(protocol['binary'])
    if not binary_path.is_absolute():
        binary_path = root / binary_path
    require(binary_path.is_file() and digest(binary_path) == complete.get('binary_sha256'),
            'retained native binary is missing or its hash differs from COMPLETE')
    require(complete.get('binary_sha256') and complete.get('protocol_sha256'),
            'completion receipt lacks binary/protocol hashes')

    manifest_path = fixture / 'manifest.json'
    require(manifest_path.is_file(), 'fixture manifest is missing')
    manifest = json.loads(manifest_path.read_text())
    require(manifest.get('sample_rate') == RATE and manifest.get('channels') == 2 and
            manifest.get('duration_seconds') == 32.0 and manifest.get('seed') == 881000,
            'expected seed-881000 32-second stereo fixture')
    require(manifest.get('view_origins_hop_aligned') is True and
            manifest.get('comparison_interior_global_samples') == [8 * RATE, 21 * RATE],
            'fixture common interior or hop-alignment declaration changed')
    require([v.get('id') for v in manifest.get('views', [])] == list(VIEWS),
            'fixture must declare view0/view3 in order')
    full_input = fixture / 'input-full.wav'
    require(digest(full_input) == manifest['source_sha256']['input_full'],
            'fixture input hash differs from manifest')
    return complete, protocol, manifest, binary_path, full_input


def analyze(session_path: str | Path, fixture_path: str | Path,
            include_spectral: bool = False) -> dict:
    session, fixture = Path(session_path).resolve(), Path(fixture_path).resolve()
    complete, protocol, manifest, binary_path, full_input_path = validate_protocol_and_receipt(session, fixture)
    complete_path = session / 'COMPLETE.json'
    protocol_path, _ = find_protocol(repo_root(session), session.parent.name,
                                     complete['protocol_sha256'])
    common_begin, common_end = manifest['comparison_interior_global_samples']
    frames = 26 * RATE
    require(common_end - common_begin == 13 * RATE, 'fixed comparison interval must be 13 seconds')
    full_input = read_stereo(full_input_path, manifest['samples'])
    shared_input = full_input[:, common_begin:common_end]
    by_view = {view['id']: view for view in manifest['views']}
    events = [int(event['sample']) for event in manifest.get('events', [])
              if common_begin <= int(event['sample']) < common_end]
    require(events, 'fixture has no declared onsets in the fixed comparison interval')

    expected = {(arm, view) for arm in ARMS for view in VIEWS}
    jobs = complete.get('jobs')
    require(isinstance(jobs, list) and len(jobs) == len(expected),
            'completion receipt must contain exactly eight arm/view jobs')
    indexed = {}
    for job in jobs:
        key = (job.get('arm'), job.get('view'))
        require(key in expected and key not in indexed, 'unknown or duplicate finalizer arm/view job')
        indexed[key] = job
    require(set(indexed) == expected, 'completion receipt is missing an arm/view pair')

    arms = {}
    outputs = {}
    finite_all = True
    peak_pass_all = True
    ceiling = float(protocol['ceiling_dbtp'])
    target_lufs = float(protocol['target_lufs'])
    true_peak_linear_max = float(protocol['gates']['true_peak_linear_max'])
    for arm in ARMS:
        arms[arm] = {'views': {}}
        outputs[arm] = {}
        for view_id in VIEWS:
            job = indexed[(arm, view_id)]
            command = job['command']
            require(command and Path(command[0]).resolve() == binary_path.resolve(),
                    f'{arm}/{view_id} command does not use pinned binary')
            input_path = Path(arg_value(command, '--input')).resolve()
            output_dir = Path(arg_value(command, '--out')).resolve()
            require(output_dir == (session / arm / view_id).resolve(),
                    f'{arm}/{view_id} command output path differs from receipt layout')
            require(float(arg_value(command, '--target-lufs')) == target_lufs and
                    float(arg_value(command, '--ceiling-db')) == ceiling,
                    f'{arm}/{view_id} command target/ceiling differs from protocol')
            for flag, key in (('--glue-threshold-db', 'glue_threshold_db'),
                              ('--glue-ratio', 'glue_ratio'),
                              ('--sidechain-hp-hz', 'sidechain_hp_hz')):
                require(float(arg_value(command, flag)) == float(protocol[key]),
                        f'{arm}/{view_id} command {flag} differs from protocol')
            for flag in ('--clock-transient-punch', '--kick-transient-guard',
                         '--quiet-mid-denoise'):
                require((flag in command) == (flag in protocol['flags']),
                        f'{arm}/{view_id} command flags differ from protocol')
            input_layout = {
                'restart': Path('native-baseline-v1') / 'renders' / view_id / 'restored.wav',
                'continuous': Path('continuous-gpu-v1') / 'renders' / view_id / 'restored.wav',
                'restart_projected': Path('projection-factorial-v1') / 'restart' / view_id / 'projected.wav',
                'continuous_projected': Path('projection-factorial-v1') / 'continuous' / view_id / 'projected.wav',
            }[arm]
            require(input_path == (session.parent / input_layout).resolve(),
                    f'{arm}/{view_id} input path does not match its declared arm')
            require(job.get('input_sha256') == digest(input_path),
                    f'{arm}/{view_id} input hash differs from COMPLETE')
            output_path = output_dir / 'mastered.wav'
            metadata_path = output_dir / 'master.json'
            require(digest(output_path) == job.get('output_sha256'),
                    f'{arm}/{view_id} mastered WAV hash differs from COMPLETE')
            require(digest(metadata_path) == job.get('master_metadata_sha256'),
                    f'{arm}/{view_id} master metadata hash differs from COMPLETE')
            metadata = json.loads(metadata_path.read_text())
            require(finite_json(metadata), f'{arm}/{view_id} master metadata contains nonfinite values')
            require(metadata.get('schema') == 'highband-post-master-v1' and
                    Path(metadata.get('input', '')).resolve() == input_path,
                    f'{arm}/{view_id} master metadata does not identify command input')
            require(metadata.get('before') == job.get('before') and
                    metadata.get('after') == job.get('after') and
                    metadata.get('limiter') == job.get('limiter') and
                    metadata.get('glue') == job.get('glue'),
                    f'{arm}/{view_id} COMPLETE metrics differ from master metadata')
            require(float(metadata.get('target_lufs', math.nan)) == target_lufs and
                    float(metadata.get('ceiling_dbtp', math.nan)) == ceiling,
                    f'{arm}/{view_id} metadata target/ceiling differs from protocol')
            for flag, control in (('--clock-transient-punch', 'clock_transient_punch'),
                                  ('--kick-transient-guard', 'kick_transient_guard'),
                                  ('--quiet-mid-denoise', 'quiet_mid_denoise')):
                require(metadata['controls'].get(control) == (flag in protocol['flags']),
                        f'{arm}/{view_id} metadata feature settings differ from protocol')
            before = read_stereo(input_path, frames)
            after = read_stereo(output_path, frames)
            finite = bool(np.isfinite(before).all() and np.isfinite(after).all())
            finite_all &= finite
            limiter_gr = float(metadata['limiter']['max_gain_reduction_db'])
            after_true_peak = float(metadata['after']['true_peak_dbtp'])
            after_true_peak_linear = 10.0 ** (after_true_peak / 20.0)
            peak_pass = after_true_peak_linear <= true_peak_linear_max
            peak_pass_all &= peak_pass

            spec = by_view[view_id]
            view_origin = int(spec['origin_sample'])
            local_onsets = [sample - view_origin for sample in events]
            require(all(0 <= local < frames for local in local_onsets),
                    f'{arm}/{view_id} event origin mismatch')
            before_events = [{'global_onset_sample': sample, 'global_onset_seconds': sample / RATE,
                              **event_window_metrics(before, local)}
                             for sample, local in zip(events, local_onsets)]
            after_events = [{'global_onset_sample': sample, 'global_onset_seconds': sample / RATE,
                             **event_window_metrics(after, local)}
                            for sample, local in zip(events, local_onsets)]
            interior = after[:, common_begin - view_origin:common_end - view_origin]
            require(interior.shape == (2, 13 * RATE), f'{arm}/{view_id} common interior slice mismatch')
            outputs[arm][view_id] = interior.copy()
            arms[arm]['views'][view_id] = {
                'command': command,
                'input_sha256': job['input_sha256'],
                'output_sha256': job['output_sha256'],
                'master_metadata_sha256': job['master_metadata_sha256'],
                'input_metrics_before_finalizer': metadata['before'],
                'native_metrics_after_finalizer': metadata['after'],
                'limiter': {**metadata['limiter'], 'engaged': limiter_gr > 0.0},
                'glue': metadata['glue'],
                'pre_gain_db': metadata.get('pre_gain_db'),
                'target_shortfall_lu': metadata.get('target_shortfall_lu'),
                'audio_geometry': {'rate': RATE, 'channels': 2, 'frames': frames,
                                   'format': 'IEEE float32'},
                'finite_samples': finite,
                'true_peak_gate': {'ceiling_dbtp': ceiling,
                                   'protocol_linear_max': true_peak_linear_max,
                                   'observed_dbtp': after_true_peak,
                                   'observed_linear': after_true_peak_linear,
                                   'pass': peak_pass},
                'stereo_before_finalizer': stereo_metrics(before),
                'stereo_after_finalizer': stereo_metrics(after),
                'fixed_source_onset_windows_before_finalizer': before_events,
                'fixed_source_onset_windows_after_finalizer': after_events,
            }
    layout = {}
    for arm in ARMS:
        y0, y3 = outputs[arm]['view0'], outputs[arm]['view3']
        layout[arm] = {'raw_layout_difference': waveform_difference(y0, y3)}
        matched = input_referred_attenuation(y0, y3, shared_input)
        layout[arm]['input_referred_attenuation_control'] = matched
        if include_spectral:
            layout[arm]['raw_layout_band_metrics'] = compare(y0, y3)
    return {
        'schema': 'coherence-finalizer-offline-analysis-v1',
        'scope': 'one 32-second seed-881000 procedural fixture; two aligned 26-second views; fixed 13-second common interior; no production-wide or perceptual claim',
        'provenance': {
            'session_complete_sha256': digest(complete_path),
            'protocol_sha256': digest(protocol_path),
            'analysis_source_sha256': digest(Path(__file__).resolve()),
            'binary_sha256': complete['binary_sha256'],
            'binary_path': str(binary_path),
            'fixture_manifest_sha256': digest(fixture / 'manifest.json'),
            'fixture_generator_sha256': manifest.get('generator_sha256'),
            'fixture_input_full_sha256': manifest['source_sha256']['input_full'],
            'fixture_seed': manifest['seed'],
        },
        'condition': {'schema': protocol['schema'], 'target_lufs': target_lufs,
                      'ceiling_dbtp': ceiling,
                      'true_peak_linear_max': true_peak_linear_max,
                      'limiter_stress_is_not_a_production_recommendation':
                          protocol['schema'] == 'coherence-finalizer-limiter-stress-v1',
                      'reason': protocol.get('reason'),
                      'finalizer_scope': protocol.get('scope'),
                      'promotion': protocol.get('promotion')},
        'interior_global_samples': [common_begin, common_end],
        'source_onsets_global_samples': events,
        'gates': {'all_output_samples_finite': finite_all,
                  'all_native_after_true_peaks_within_protocol_linear_limits': peak_pass_all,
                  'limiter_engaged_anywhere': any(
                      arms[a]['views'][v]['limiter']['engaged'] for a in ARMS for v in VIEWS),
                  'limiter_engaged_by_arm_view': {
                      a: {v: arms[a]['views'][v]['limiter']['engaged'] for v in VIEWS} for a in ARMS}},
        'arms': arms,
        'layout_comparisons': layout,
        'gain_control_note': 'The optional scalar attenuation is input-referred to the same degraded procedural input in the shared interior. It is a gain sensitivity control, not a perceptual loudness match; raw differences remain primary.',
        'measurement_limit': 'Before values are the native Mid outputs immediately before this finalizer; they are not a pre-limiter snapshot. The true-peak field is the project finite 4x estimate, not a continuous-time guarantee.',
        'spectral_metrics_included': include_spectral,
        'target_audio_read': False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path,
                        help='new JSON file; existing paths are rejected')
    parser.add_argument('--spectral', action='store_true',
                        help='include compact band metrics for the aligned final-output layout comparison')
    args = parser.parse_args()
    result = analyze(args.session, args.fixture, include_spectral=args.spectral)
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
