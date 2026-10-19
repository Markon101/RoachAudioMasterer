#!/usr/bin/env python3
"""Offline, aligned long-phrase measurements; never runs a neural model."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

if __package__:
    from .check_wola_controls import chunk_starts, normalized_wola, source_weights
    from .roach_ears.audio_io import read_wav
else:
    from check_wola_controls import chunk_starts, normalized_wola, source_weights
    from roach_ears.audio_io import read_wav

RATE = 48000
FLOOR = 1e-7
NULL_POWER = 1e-14
BANDS = {'low': (0.0, 1500.0), 'mid': (1500.0, 8000.0), 'high': (8000.0, 24000.0)}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def channels(x):
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] != 2 or not x.shape[1] or not np.isfinite(x).all():
        raise ValueError('finite nonempty stereo audio required')
    return {'left': x[0], 'right': x[1], 'mid': (x[0] + x[1]) / math.sqrt(2),
            'side': (x[0] - x[1]) / math.sqrt(2)}


def band_metrics(reference, output):
    """Normalized one-sided FFT; reported power equals waveform mean square."""
    if len(reference) != len(output):
        raise ValueError('unaligned lengths')
    n = len(reference)
    a, b = np.fft.rfft(np.asarray(reference, dtype=np.float64)) / n, np.fft.rfft(np.asarray(output, dtype=np.float64)) / n
    f = np.fft.rfftfreq(n, 1 / RATE)
    weights = np.full(len(f), 2.0)
    weights[0] = 1
    if n % 2 == 0:
        weights[-1] = 1
    result = {}
    for name, (low, high) in BANDS.items():
        mask = (f > low) & (f < high) if name == 'mid' else (f >= low) & (f <= high)
        aa, bb, w = a[mask], b[mask], weights[mask]
        power = float(np.sum(w * np.abs(aa) ** 2))
        error = float(np.sum(w * np.abs(aa - bb) ** 2))
        db = 20 * np.log10(np.maximum(np.abs(aa), FLOOR) / np.maximum(np.abs(bb), FLOOR))
        phase_target = np.abs(aa) >= max(FLOOR, 1e-4 * float(np.max(np.abs(aa), initial=0)))
        phase_valid = phase_target & (np.abs(bb) >= FLOOR)
        masked_power = float(np.sum(w[phase_target] * np.abs(aa[phase_target]) ** 2))
        valid_power = float(np.sum(w[phase_valid] * np.abs(aa[phase_valid]) ** 2))
        phase_error = float(np.sum(w[phase_valid] * np.abs(aa[phase_valid]) ** 2 *
                                   np.abs(np.angle(bb[phase_valid] * aa[phase_valid].conj()))))
        result[name] = {'bins': int(mask.sum()), 'reference_power': power,
                        'absolute_error_power': error, 'error_rms': math.sqrt(error),
                        'complex_nmse': error / power if power > NULL_POWER else None,
                        'target_weighted_lsd_db': math.sqrt(float(np.sum(w * np.abs(aa) ** 2 * db ** 2)) / power) if power > NULL_POWER else None,
                        'phase_target_bins': int(phase_target.sum()), 'phase_valid_bins': int(phase_valid.sum()),
                        'phase_power_coverage': valid_power / masked_power if masked_power > NULL_POWER else None,
                        'phase_mae_radians': phase_error / valid_power if valid_power > NULL_POWER else None}
    return result


def compare(reference, output):
    a, b = channels(reference), channels(output)
    return {name: band_metrics(a[name], b[name]) for name in a}


def waveform_difference(a, b):
    delta = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
    return {'samples_per_channel': a.shape[1], 'max_abs': float(np.max(np.abs(delta))),
            'rms': rms(delta), 'reference_rms': rms(a),
            'relative_rms': rms(delta) / rms(a) if rms(a) > 1e-10 else None}


def inspect_chunks(render, input_audio, origin, chunk_samples, overlap_samples):
    n = input_audio.shape[1]
    starts = chunk_starts(n, chunk_samples, overlap_samples)
    chunks, residuals, rows = [], [], []
    for index, start in enumerate(starts):
        count = min(chunk_samples, n - start)
        path = render / 'chunks' / f'chunk_{index:04d}.bin'
        raw = path.read_bytes()
        if len(raw) != count * 2 * 4:
            raise ValueError('unexpected chunk byte count')
        audio = np.frombuffer(raw, dtype='<f4').reshape(2, count).astype(np.float64)
        x = input_audio[:, start:start + count]
        chunks.append(audio)
        residuals.append(audio - x)
        rows.append({'index': index, 'start_sample': start, 'global_start_sample': origin + start,
                     'samples': count, 'sha256': hashlib.sha256(raw).hexdigest(),
                     'residual_rms': rms(audio - x), 'vs_input': compare(x, audio)})
    stitched, norm = normalized_wola(chunks, starts, n, overlap_samples)
    overlaps = []
    for index in range(len(chunks) - 1):
        begin = starts[index + 1]
        end = min(starts[index] + chunks[index].shape[1], starts[index + 1] + chunks[index + 1].shape[1])
        center = (begin + end) // 2
        start, stop = max(begin, center - RATE // 10), min(end, center + RATE // 10)
        if stop <= start:
            continue
        r1 = residuals[index][:, start - starts[index]:stop - starts[index]]
        r2 = residuals[index + 1][:, start - starts[index + 1]:stop - starts[index + 1]]
        w1 = source_weights(chunks[index].shape[1], overlap_samples, index, len(chunks))[start - starts[index]:stop - starts[index]]
        w2 = source_weights(chunks[index + 1].shape[1], overlap_samples, index + 1, len(chunks))[start - starts[index + 1]:stop - starts[index + 1]]
        denominator = norm[start:stop]
        a, b = r1 * (w1 / denominator), r2 * (w2 / denominator)
        energy_product = float(np.sum(a * a) * np.sum(b * b))
        upper = rms(a) + rms(b)
        overlaps.append({'indices': [index, index + 1], 'global_interval_samples': [origin + begin, origin + end],
                         'global_center_window_samples': [origin + start, origin + stop],
                         'residual_rms': [rms(r1), rms(r2)],
                         'weighted_uncentered_correlation': float(np.sum(a * b)) / math.sqrt(energy_product) if energy_product > 1e-30 else None,
                         'blended_residual_rms': rms(a + b), 'coherent_upper_rms': upper,
                         'blend_to_coherent_upper': rms(a + b) / upper if upper > 1e-10 else None})
    return stitched, rows, overlaps


def analyze(session, fixture):
    manifest = json.loads((fixture / 'manifest.json').read_text())
    if (session / 'HALTED.json').exists():
        raise ValueError('halted session requires separate explicit evidence review')
    if not (session / 'COMPLETE.json').is_file():
        raise ValueError('session is not complete')
    views = manifest['views']
    specs = [dict(v, chunk_seconds=8, overlap_seconds=2) for v in views]
    specs.append({'id': 'full', 'input': 'input-full.wav', 'target': 'target-full.wav',
                  'origin_sample': 0, 'samples': manifest['samples'],
                  'chunk_seconds': 32, 'overlap_seconds': 0})
    cases, audio, inputs = {}, {}, {}
    for spec in specs:
        name = spec['id']
        render = session / 'renders' / name
        x = read_wav(fixture / spec['input']).samples.astype(np.float64)
        target = read_wav(fixture / spec['target']).samples.astype(np.float64)
        y = read_wav(render / 'restored.wav').samples.astype(np.float64)
        if x.shape != y.shape or y.shape != target.shape or y.shape[1] != spec['samples']:
            raise ValueError('unaligned native output')
        stitched, chunks, overlaps = inspect_chunks(render, x, spec['origin_sample'],
                                                    round(spec['chunk_seconds'] * RATE), round(spec['overlap_seconds'] * RATE))
        delta = waveform_difference(y, stitched)
        if delta['max_abs'] > 5e-7:
            raise ValueError('Python WOLA reconstruction disagrees with native output')
        cases[name] = {'output_sha256': digest(render / 'restored.wav'),
                       'output_rms': rms(y), 'output_peak': float(np.max(np.abs(y))),
                       'vs_input': compare(x, y), 'vs_target': compare(target, y),
                       'identity_vs_target': compare(target, x),
                       'python_f64_stitch_vs_native_f32': delta, 'chunks': chunks, 'overlaps': overlaps}
        audio[name], inputs[name] = y, x
    begin, end = manifest['comparison_interior_global_samples']
    def aligned(collection, name, origin):
        return collection[name][:, begin - origin:end - origin]
    a, b = views
    x0, x3 = aligned(inputs, a['id'], a['origin_sample']), aligned(inputs, b['id'], b['origin_sample'])
    if not np.array_equal(x0, x3):
        raise ValueError('physical input differs between aligned views')
    y0, y3 = aligned(audio, a['id'], a['origin_sample']), aligned(audio, b['id'], b['origin_sample'])
    yf = audio['full'][:, begin:end]
    residual0, residual3 = y0 - x0, y3 - x3
    return {'schema': 'long-coherence-native-analysis-v1', 'fixture_manifest_sha256': digest(fixture / 'manifest.json'),
            'scope': 'one procedural continuous phrase; fixed Euler8 CPU; single-clip render is a counterfactual, not ground truth',
            'definitions': {'rate': RATE, 'band_boundaries_hz': BANDS, 'spectrum': 'float64 rFFT/n, one-sided Parseval weights',
                            'power_null_floor': NULL_POWER, 'amplitude_floor': FLOOR,
                            'phase_mask': 'reference>=max(1e-7,1e-4 band peak),output>=1e-7',
                            'overlap_statistic': '200ms centered overlap; cosine similarity of weighted residual vectors, not statistical independence'},
            'cases': cases, 'interior_global_samples': [begin, end], 'aligned_input_identical': True,
            'boundary_layout_difference': waveform_difference(y0, y3),
            'residual_layout_difference': waveform_difference(residual0, residual3),
            'boundary_layout_bands': compare(y0, y3),
            'full_counterfactual_difference': {'view0': waveform_difference(yf, y0), 'view3': waveform_difference(yf, y3)},
            'promotion': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.session, args.fixture)
    with args.out.open('x') as target:
        json.dump(result, target, indent=2, allow_nan=False)
        target.write('\n')
