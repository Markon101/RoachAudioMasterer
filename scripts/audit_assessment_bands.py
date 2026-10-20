#!/usr/bin/env python3
"""Read-only calibrated whole-clip band-power observer for native WAV files."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

if __package__:
    from .run_coherence_quality import wav_info
    from .roach_ears.audio_io import read_wav
else:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from run_coherence_quality import wav_info
    from roach_ears.audio_io import read_wav

RATE = 48_000
MAX_FRAMES = 48 * RATE
NULL_POWER = 1e-14
BANDS = (
    ('dc', 0.0, 0.0),
    ('sub_0_120_hz', 0.0, 120.0),
    ('low_mid_120_500_hz', 120.0, 500.0),
    ('mid_500_6000_hz', 500.0, 6000.0),
    ('high_6000_14000_hz', 6000.0, 14000.0),
    ('upper_14000_22000_hz', 14000.0, 22000.0),
    ('top_22000_24000_hz', 22000.0, 24000.0),
)
LEGACY_FIELDS = ('sub_energy_dbfs', 'low_mid_energy_dbfs', 'mid_energy_dbfs',
                 'high_energy_dbfs', 'ultrasonic_energy_dbfs')


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def channel_band_metrics(samples: np.ndarray, rate: int = RATE) -> dict:
    """Return physical mean-square band powers from an f64-normalized rFFT."""
    x = np.asarray(samples, dtype=np.float64)
    require(x.ndim == 1 and x.size > 0 and np.isfinite(x).all(),
            'band metrics require finite nonempty mono samples')
    n = x.size
    spectrum = np.fft.rfft(x) / n
    frequency = np.fft.rfftfreq(n, d=1.0 / rate)
    weight = np.full(frequency.shape, 2.0, dtype=np.float64)
    weight[0] = 1.0
    if n % 2 == 0:
        weight[-1] = 1.0
    bands = {}
    for name, low, high in BANDS:
        select = (frequency == 0.0) if name == 'dc' else (frequency > low) & (frequency <= high)
        power = float(np.sum(weight[select] * np.abs(spectrum[select]) ** 2))
        rms_value = math.sqrt(max(power, 0.0))
        bands[name] = {
            'frequency_hz': [low, high],
            'power_mean_square': power,
            'rms': rms_value,
            'rms_dbfs': 20.0 * math.log10(rms_value) if rms_value > 0.0 else None,
            'power_below_null_floor': power <= NULL_POWER,
        }
    waveform_power = float(np.mean(x * x))
    band_power_sum = sum(row['power_mean_square'] for row in bands.values())
    closure = band_power_sum - waveform_power
    return {
        'sample_count': int(n),
        'waveform_rms_squared': waveform_power,
        'waveform_rms': math.sqrt(max(waveform_power, 0.0)),
        'waveform_rms_dbfs': 10.0 * math.log10(waveform_power) if waveform_power > 0.0 else None,
        'band_power_sum': band_power_sum,
        'band_power_closure_error': closure,
        'band_power_closure_pass_1e-10': abs(closure) <= 1e-10,
        'bands': bands,
    }


def channel_views(samples: np.ndarray) -> dict[str, np.ndarray]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] not in (1, 2):
        raise ValueError('decoded audio must be mono or stereo [channels,samples]')
    if samples.shape[0] == 1:
        return {'mono': samples[0]}
    left, right = samples
    scale = math.sqrt(2.0)
    return {'left': left, 'right': right,
            'mid_orthonormal': (left + right) / scale,
            'side_orthonormal': (left - right) / scale}


def legacy_assessment(path: Path | None) -> dict | None:
    if path is None:
        return None
    raw = path.read_bytes()
    assessment = json.loads(raw)
    return {
        'sha256': hashlib.sha256(raw).hexdigest(),
        'fields': {name: assessment.get(name) for name in LEGACY_FIELDS},
        'unit_qualification': 'Legacy assess.rs fields are raw windowed FFT-bin mean power divided by frame/bin counts; values are not calibrated physical dBFS band powers.',
        'input_binding': 'Caller-supplied assessment JSON; the JSON contains no input hash, so association with --audio is not authenticated.',
    }


def audit(audio_path: Path, assessment_path: Path | None = None) -> dict:
    info = wav_info(audio_path)
    require(info['rate'] == RATE and info['channels'] in (1, 2),
            'expected native 48 kHz mono/stereo WAV')
    require(0 < info['frames'] <= MAX_FRAMES, 'audio must contain 1 sample through 48 seconds')
    require(info['format'] in (1, 3), 'unsupported WAV encoding')
    decoded = read_wav(audio_path)
    samples = np.asarray(decoded.samples, dtype=np.float64)
    require(decoded.sample_rate == RATE and samples.shape == (info['channels'], info['frames']),
            'decoded WAV geometry differs from header')
    require(np.isfinite(samples).all(), 'audio contains nonfinite samples')
    views = {name: channel_band_metrics(x) for name, x in channel_views(samples).items()}
    require(all(v['band_power_closure_pass_1e-10'] for v in views.values()),
            'one-sided Parseval band powers do not close to waveform mean square')
    raw_sha = hashlib.sha256(audio_path.read_bytes()).hexdigest()
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    legacy = legacy_assessment(assessment_path)
    return {
        'schema': 'assessment-physical-band-observer-v1',
        'scope': 'whole-clip physical mean-square band observer; no routing, mastering, or model inference',
        'provenance': {'audio_path': str(audio_path.resolve()), 'audio_sha256': raw_sha,
                       'observer_source_sha256': source_sha,
                       'assessment_sha256': legacy['sha256'] if legacy else None},
        'geometry': {'sample_rate_hz': RATE, 'channels': info['channels'],
                     'frames': info['frames'], 'duration_seconds': info['frames'] / RATE,
                     'wav_format_tag': info['format'], 'bits_per_sample': info['bits_per_sample']},
        'spectrum': {'transform': 'float64 rFFT(x)/N over whole clip',
                     'one_sided_parseval_weights': 'DC=1; Nyquist=1 for even N; other positive bins=2',
                     'band_edges': 'disjoint: DC, then (low,high] through Nyquist',
                     'null_power_floor': NULL_POWER,
                     'rms_reference': '20*log10(rms/1); a peak-one sine has RMS -3.0103 dBFS. Power is mean-square normalized samples, not acoustic watts or SPL.',
                     'orthonormal_mid_side_note': 'M/S channels can exceed unit sample magnitude; their dBFS RMS is a linear level reference, not a clipping assertion.'},
        'channels': views,
        'legacy_assessment': legacy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audio', required=True, type=Path)
    parser.add_argument('--assessment', type=Path, help='optional existing legacy assessment JSON')
    parser.add_argument('--out', required=True, type=Path, help='fresh exclusive JSON output path')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'refusing existing output: {args.out}')
    result = audit(args.audio, args.assessment)
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
