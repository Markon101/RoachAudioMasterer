#!/usr/bin/env python3
"""Offline first-channel preservation-shape trace; never renders or changes gates."""
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys

os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np

from matched_attack_audit import WavReader, memory_receipt


ROOT = Path(__file__).resolve().parents[1]
RMS_FLOOR = 1e-8


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def check_latches(*directories):
    for directory in directories:
        present = [name for name in ('HALT', 'HALT.json', 'STOP', 'STOP.json')
                   if (Path(directory) / name).exists()]
        if present:
            raise RuntimeError(f'diagnostic-local stop latch in {directory}: {present}')


def combined_response(frequencies):
    """Analytical float64 amplitude response of the two native Fourier filters."""
    low_t = np.clip((frequencies - 2500.) / 500., 0., 1.)
    high_t = np.clip((1000. - frequencies) / 300., 0., 1.)
    low = np.maximum(0., .5 + .5 * np.cos(np.pi * low_t)) ** 2
    high = np.maximum(0., .5 + .5 * np.cos(np.pi * high_t)) ** 2
    return low * high


def measure_samples(samples, rate, spectrum_block_bins=65536, cap_mib=None):
    """Measure one real channel with bounded spectral temporaries and Parseval."""
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 1 or not len(samples) or not np.all(np.isfinite(samples)):
        raise ValueError('finite nonempty first channel required')
    if rate != 48000 or spectrum_block_bins <= 0:
        raise ValueError('native 48000 Hz and positive spectrum block required')
    n = len(samples)
    full_power = float(np.dot(samples, samples)) / n
    spectrum = np.fft.rfft(samples)
    memory_receipt(cap_mib)
    weighted_full, weighted_band = 0., 0.
    for start in range(0, len(spectrum), spectrum_block_bins):
        end = min(start + spectrum_block_bins, len(spectrum))
        z = spectrum[start:end]
        power = np.square(z.real)
        power += np.square(z.imag)
        power *= 2.
        if start == 0:
            power[0] *= .5
        if n % 2 == 0 and end == len(spectrum):
            power[-1] *= .5
        frequencies = np.arange(start, end, dtype=np.float64) * (rate / n)
        response = combined_response(frequencies)
        weighted_full += float(np.sum(power))
        weighted_band += float(np.dot(power, response * response))
    del spectrum
    band_power = weighted_band / n**2
    spectral_full_power = weighted_full / n**2
    raw_full_rms, raw_band_rms = math.sqrt(full_power), math.sqrt(band_power)
    full_rms, band_rms = max(raw_full_rms, RMS_FLOOR), max(raw_band_rms, RMS_FLOOR)
    return {
        'full_power': full_power, 'filtered_band_power': band_power,
        'spectral_full_power': spectral_full_power,
        'parseval_relative_closure_error': (abs(spectral_full_power - full_power) / full_power
                                            if full_power > 0 else None),
        'raw_full_rms': raw_full_rms, 'raw_filtered_band_rms': raw_band_rms,
        'full_rms_native_floor': full_rms, 'filtered_band_rms_native_floor': band_rms,
        'broadband_floor_active': raw_full_rms < RMS_FLOOR,
        'filtered_band_floor_active': raw_band_rms < RMS_FLOOR,
        'filtered_energy_fraction': band_power / full_power if full_power > 0 else None,
        'normalized_filtered_rms': band_rms / full_rms,
    }


def measure_path(path, geometry, resources):
    reader = WavReader(path)
    try:
        if {k: getattr(reader, k) for k in ('rate', 'channels', 'frames')} != geometry:
            raise ValueError(f'geometry mismatch: {path}')
        if reader.frames > resources['max_frames']:
            raise ValueError('frame limit exceeded before allocating first channel')
        channel = np.empty(reader.frames, dtype=np.float64)
        for start in range(0, reader.frames, resources['input_block_frames']):
            end = min(start + resources['input_block_frames'], reader.frames)
            channel[start:end] = reader.read(start, end)[:, 0]
            memory_receipt(resources['process_hwm_aim_mib'])
        metadata = reader.metadata()
    finally:
        reader.close()
    metrics = measure_samples(channel, geometry['rate'], resources['spectrum_block_bins'],
                              resources['process_hwm_aim_mib'])
    del channel
    gc.collect()
    return {'wav': metadata, **metrics,
            'memory': memory_receipt(resources['process_hwm_aim_mib'])}


def db_ratio(value, reference):
    return 20. * math.log10(value / reference)


def compare(current, source, previous):
    signed = db_ratio(current['normalized_filtered_rms'], source['normalized_filtered_rms'])
    return {
        'signed_shape_change_from_source_db': signed,
        'absolute_shape_deviation_from_source_db': abs(signed),
        'analytical_below_unchanged_1db_threshold': abs(signed) < 1.,
        'signed_increment_from_previous_boundary_db': db_ratio(
            current['normalized_filtered_rms'], previous['normalized_filtered_rms']),
        'filtered_band_rms_change_from_source_db': db_ratio(
            current['filtered_band_rms_native_floor'], source['filtered_band_rms_native_floor']),
        'broadband_rms_change_from_source_db': db_ratio(
            current['full_rms_native_floor'], source['full_rms_native_floor']),
        'filtered_band_rms_increment_db': db_ratio(
            current['filtered_band_rms_native_floor'], previous['filtered_band_rms_native_floor']),
        'broadband_rms_increment_db': db_ratio(
            current['full_rms_native_floor'], previous['full_rms_native_floor']),
    }


def check_pins(pins):
    for path, expected in pins.items():
        if digest(ROOT / path) != expected:
            raise ValueError(f'frozen pin changed: {path}')


def report_text(summary):
    lines = [
        '# Singularity first-channel preservation-shape trace', '',
        'This is a float64 analytical-response stage diagnostic. The original native HALTED run remains stopped; this diagnostic earns no candidate render or listening package.', '',
        '| Cached boundary | Signed shape vs source (dB) | Absolute (dB) | Increment (dB) | Band RMS vs source (dB) | Broadband RMS vs source (dB) |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for row in summary['rows']:
        lines.append(f"| {row['id']} | {row['signed_shape_change_from_source_db']:.6f} | "
                     f"{row['absolute_shape_deviation_from_source_db']:.6f} | "
                     f"{row['signed_increment_from_previous_boundary_db']:.6f} | "
                     f"{row['filtered_band_rms_change_from_source_db']:.6f} | "
                     f"{row['broadband_rms_change_from_source_db']:.6f} |")
    final = summary['native_comparison']
    lines += ['', f"First observed >=1 dB analytical crossing: **{summary['first_crossing_boundary']}**.", '',
              f"Final analytical absolute deviation: {final['analytical_absolute_db']:.9f} dB. "
              f"Recorded native: {final['recorded_native_absolute_db']:.9f} dB. "
              f"Analytical minus native: {final['analytical_minus_native_db']:+.9f} dB.", '',
              'The native field `max_passband_deviation_db` is one absolute broadband-normalized filtered-RMS ratio on channel 0. It is not a maximum over frequencies, passbands or channels. The LP/HP skirt support spans approximately 700–3000 Hz, with full gain at 1000–2500 Hz.', '',
              'A boundary is the first cached observation, not isolated algorithm attribution. Clean polish combines denoise and auto-EQ; final mastering combines nonlinear dynamics and other controls. No gain-only or EQ-only counterfactual was rendered. Positive uniform gain cancels above the RMS floors, but frequency-selective changes, signal-dependent gain and stereo redistribution can change this proxy. Signed increments telescope; absolute deviations can cancel or accumulate.', '',
              'NumPy float64/complex128 Parseval analysis does not reproduce native sequential complex32 FFT/IFFT, f32 frequency/gain evaluation or f32 reductions byte-for-byte. Equal WAV geometry is checked; temporal/file alignment is unsupported and no waveform shift is applied. The result does not establish perceptual damage, preference, source fidelity or protection of each spectral bin.', '',
              f"Maximum observed process HWM: {summary['maximum_observed_hwm_mib']:.3f} MiB (512 MiB aim).", '']
    return '\n'.join(lines)


def run_trace(plan_path, output):
    plan_path, output = Path(plan_path).resolve(), Path(output).resolve()
    check_latches(plan_path.parent, output)
    if output.exists():
        raise FileExistsError('fresh output required; no resume or overwrite')
    plan = json.loads(plan_path.read_text())
    if plan['schema'] != 'first-channel-preservation-shape-trace-v1':
        raise ValueError('unsupported diagnostic plan')
    if plan['resources']['input_block_frames'] > 65536:
        raise ValueError('reader block exceeds bounded limit')
    pins = dict(plan['pins'])
    pins[str(plan_path)] = digest(plan_path)
    pins[str(Path(__file__).resolve())] = digest(__file__)
    check_pins(pins)
    protocols = [path for path in plan['pins'] if path.endswith('/protocol.json')]
    raw_source = None
    if protocols:
        if len(protocols) != 1:
            raise ValueError('ambiguous frozen source protocol')
        raw_source = json.loads((ROOT / protocols[0]).read_text())['plan']['source']
        pins[raw_source['path']] = raw_source['sha256']
        check_pins(pins)
    available = int(next(row.split()[1] for row in Path('/proc/meminfo').read_text().splitlines()
                         if row.startswith('MemAvailable:')))
    if available < plan['resources']['minimum_available_memory_mib'] * 1024:
        raise MemoryError('insufficient available phone memory for bounded offline diagnostic')
    output.mkdir(parents=True)
    started = utc()
    rows = []
    try:
        save(output / 'manifest.json', {'schema': plan['schema'], 'plan': plan,
             'pins': pins, 'started_utc': started, 'argv': sys.argv,
             'raw_source_from_frozen_protocol': raw_source,
             'python_version': sys.version, 'numpy_version': np.__version__,
             'platform': platform.platform(), 'openblas_num_threads': os.environ['OPENBLAS_NUM_THREADS']})
        reference, previous = None, None
        with (output / 'stages.jsonl').open('x') as stream:
            for spec in plan['inputs']:
                check_latches(plan_path.parent, output)
                check_pins(pins)
                measured = measure_path(ROOT / spec['path'], plan['geometry'], plan['resources'])
                check_latches(plan_path.parent, output)
                reference = measured if reference is None else reference
                previous = measured if previous is None else previous
                row = {**spec, 'sha256': pins[spec['path']], **measured,
                       **compare(measured, reference, previous)}
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
                rows.append(row)
                previous = measured
        native = json.loads((ROOT / plan['native_reference']['summary_path']).read_text())['preservation']
        if native['max_passband_deviation_db'] != plan['native_reference']['absolute_shape_deviation_db']:
            raise ValueError('recorded native reference differs from frozen plan')
        check_pins(pins)
        check_latches(plan_path.parent, output)
        final = rows[-1]['absolute_shape_deviation_from_source_db']
        summary = {
            'schema': plan['schema'], 'started_utc': started, 'completed_utc': utc(),
            'plan_sha256': pins[str(plan_path)], 'rows': rows,
            'first_crossing_boundary': next((r['id'] for r in rows
                if not r['analytical_below_unchanged_1db_threshold']), None),
            'native_comparison': {'recorded_native_absolute_db': native['max_passband_deviation_db'],
                'analytical_absolute_db': final,
                'analytical_minus_native_db': final - native['max_passband_deviation_db'],
                'native_preservation_record_unchanged': native,
                'byte_native_reproduction_claimed': False},
            'maximum_observed_hwm_mib': max(r['memory']['observed_hwm_mib'] for r in rows),
            'alignment_status': 'geometry only; waveform/time alignment unsupported; no shift',
            'original_gate_failure_remains_binding': True,
        }
        save(output / 'summary.json', summary)
        with (output / 'report.md').open('x') as stream:
            stream.write(report_text(summary))
        return summary
    except BaseException as error:
        if not (output / 'STOP.json').exists():
            save(output / 'STOP.json', {'stopped_utc': utc(), 'error_type': type(error).__name__,
                 'reason': str(error), 'completed_rows': len(rows), 'memory': memory_receipt()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    summary = run_trace(args.plan, args.out)
    print(json.dumps({'first_crossing_boundary': summary['first_crossing_boundary'],
                      'native_comparison': summary['native_comparison'],
                      'maximum_observed_hwm_mib': summary['maximum_observed_hwm_mib']}))


if __name__ == '__main__':
    main()
