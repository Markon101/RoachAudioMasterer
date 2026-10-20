#!/usr/bin/env python3
"""Offline parity and error-decomposition for short Mid source bundles."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

import numpy as np

RATE, SAMPLES, NULL_POWER = 48_000, 16_384, 1e-14
OLD_SEEDS = {860002, 860003, 860004, 860005, 860007}
SOURCE_ONLY_SEEDS = {860015, 860019}
CHANNELS = ('left', 'right', 'mid', 'side')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fnv1a64(raw: bytes) -> str:
    h = 0xcbf29ce484222325
    for b in raw:
        h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
    return f'noncryptographic-fnv1a64-f32le:{h:016x}'


def read_planar(path: Path) -> tuple[np.ndarray, bytes]:
    raw = path.read_bytes()
    if len(raw) != 2 * SAMPLES * 4:
        raise ValueError(f'{path}: wrong planar-f32le size')
    audio = np.frombuffer(raw, dtype='<f4').reshape(2, SAMPLES)
    if not np.isfinite(audio).all():
        raise ValueError(f'{path}: nonfinite samples')
    return audio.astype(np.float64), raw


def channels(audio: np.ndarray) -> dict[str, np.ndarray]:
    l, r = np.asarray(audio, dtype=np.float64)
    return {'left': l, 'right': r, 'mid': (l + r) / np.sqrt(2.0),
            'side': (l - r) / np.sqrt(2.0)}


def band_decomposition(target: np.ndarray, output: np.ndarray, cutoff_hz: float,
                       rate: int = RATE) -> dict:
    """Parseval-scaled f64 rFFT: complex error = magnitude + phase terms."""
    if target.shape != output.shape or target.ndim != 1 or target.size != SAMPLES:
        raise ValueError('expected aligned mono 16384-sample signals')
    n = target.size
    f = np.fft.rfftfreq(n, d=1.0 / rate)
    select = (f > cutoff_hz) & (f < 8000.0)
    w = np.full(f.shape, 2.0); w[0] = 1.0
    if n % 2 == 0:
        w[-1] = 1.0
    t = np.fft.rfft(np.asarray(target, dtype=np.float64))[select] / n
    y = np.fft.rfft(np.asarray(output, dtype=np.float64))[select] / n
    w, at, ay = w[select], np.abs(t), np.abs(y)
    target_power = float(np.sum(w * at**2))
    complex_error = float(np.sum(w * np.abs(y - t)**2))
    magnitude_error = float(np.sum(w * (ay - at)**2))
    product = at * ay
    cosine = np.zeros_like(product)
    np.divide(np.real(y * np.conj(t)), product, out=cosine, where=product > 0)
    phase_error = float(np.sum(w * 2 * product * (1 - np.clip(cosine, -1, 1))))
    return {'bins': int(select.sum()), 'cutoff_hz': float(cutoff_hz),
            'target_power': target_power, 'absolute_complex_error_power': complex_error,
            'relative_complex_nmse': complex_error / target_power if target_power > NULL_POWER else None,
            'magnitude_error_power': magnitude_error, 'phase_contribution_power': phase_error,
            'phase_fraction_of_complex_error': phase_error / complex_error if complex_error > 0 else None,
            'decomposition_closure_error': complex_error - magnitude_error - phase_error,
            'target_phase_defined': target_power > NULL_POWER}


def old_rows(path: Path, seeds: set[int]) -> dict:
    rows = {}
    with gzip.open(path, 'rt', encoding='utf-8') as src:
        for line in src:
            row = json.loads(line)
            seed = row.get('seed')
            if seed not in seeds:
                continue
            if row['schema'] == 'native-mid-solver-case-v1' and row.get('solver') == 'euler8':
                key = seed, row['condition'], 'euler8'
            elif row['schema'] == 'native-mid-solver-control-v1':
                key = seed, row['condition'], row['control']
            else:
                continue
            if key in rows:
                raise ValueError(f'duplicate archived row {key}')
            rows[key] = row
    return rows


def recipe_matches(old: dict, new: dict) -> bool:
    keys = ('seed', 'family', 'no_upper', 'no_lower', 'mono', 'gap')
    def f32_value(value):
        if isinstance(value, dict):
            return {k: f32_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [f32_value(v) for v in value]
        if isinstance(value, (float, int)) and not isinstance(value, bool):
            return struct.pack('<f', float(value))
        return value
    return (all(old.get(k) == new.get(k) for k in keys)
            and old.get('version') == new.get('generator_version')
            and f32_value(old.get('damage')) == f32_value(new.get('damage')))


def source_records(bundle_dir: Path, entry: dict) -> tuple[dict, dict, dict]:
    recipe_path = bundle_dir / entry['recipe_path']
    if sha256(recipe_path) != entry['recipe_sha256']:
        raise ValueError(f"seed {entry['seed']}: recipe SHA256 mismatch")
    recipe = json.loads(recipe_path.read_text())
    signals, records = {}, {}
    for name in ('target', 'degraded_input'):
        spec = entry['files'][name]
        raw_path = bundle_dir / spec['planar_f32le']['path']
        wav_path = bundle_dir / spec['wav']['path']
        raw_hash, wav_hash = sha256(raw_path), sha256(wav_path)
        if raw_hash != spec['planar_f32le']['sha256'] or wav_hash != spec['wav']['sha256']:
            raise ValueError(f"seed {entry['seed']}: {name} bundle hash mismatch")
        signals[name], raw = read_planar(raw_path)
        records[name] = {'planar_sha256': raw_hash, 'wav_sha256': wav_hash,
                         'fnv1a64': fnv1a64(raw)}
    return recipe, signals, {'recipe_sha256': entry['recipe_sha256'], 'files': records}


def analyze_old_seed(seed: int, entry: dict, recipe: dict, signals: dict,
                     records: dict, rows: dict, samples_root: Path) -> dict:
    conditions = {}
    for condition, output in (('manufactured_mid_damage', signals['degraded_input']),
                              ('healthy_fullband_forced_head', signals['target'])):
        identity_key = seed, condition, 'identity_degraded_or_healthy'
        prior_key = seed, condition, 'dsp_harmonic_prior'
        solver_key = seed, condition, 'euler8'
        if any(k not in rows for k in (identity_key, prior_key, solver_key)):
            raise ValueError(f'seed {seed}: missing archived identity/prior/Euler8 row')
        identity, prior, solver = (rows[identity_key], rows[prior_key], rows[solver_key])
        old_recipe = solver['recipe']
        if not recipe_matches(old_recipe, recipe) or solver['family'] != entry['family']:
            raise ValueError(f'seed {seed}: regenerated family/recipe differs from old audit')
        source_name = 'degraded_input' if condition == 'manufactured_mid_damage' else 'target'
        if identity['waveform_fingerprint'] != records['files'][source_name]['fnv1a64']:
            raise ValueError(f'seed {seed}: source FNV differs from archived identity control')
        raw_spec = solver['result']['samples']
        old_path = samples_root / raw_spec['path']
        old_audio, old_bytes = read_planar(old_path)
        old_fnv = fnv1a64(old_bytes)
        if old_fnv != solver['result']['waveform_fingerprint'] or old_fnv != raw_spec['waveform_fingerprint']:
            raise ValueError(f'seed {seed}: archived Euler8 FNV mismatch')
        target_ch, output_ch = channels(signals['target']), channels(old_audio)
        identity_ch = channels(output)
        cutoff = float(recipe['damage']['cutoff'])
        conditions[condition] = {
            'identity_control_old_rust_f32_metrics': identity['metrics'],
            'identity_control_new_f64_vs_target': {
                c: band_decomposition(target_ch[c], identity_ch[c], cutoff) for c in CHANNELS},
            'dsp_harmonic_prior_old_rust_f32_metrics': prior['metrics'],
            'dsp_harmonic_prior_waveform_archived': False,
            'euler8_old_rust_f32_metrics': solver['result']['metrics'],
            'euler8_new_f64_vs_target': {
                c: band_decomposition(target_ch[c], output_ch[c], cutoff) for c in CHANNELS},
            'euler8_raw': {'path': raw_spec['path'], 'sha256': sha256(old_path), 'fnv1a64': old_fnv},
            'identity_fingerprint_matches_regenerated_source': True,
            'metric_scope': 'archived values are original Rust f32; newly computed values are NumPy f64 rFFT/n with one-sided Parseval weights',
        }
    return conditions


def analyze(bundle_path: Path, cases_path: Path, samples_root: Path) -> dict:
    manifest_path = bundle_path / 'manifest.json' if bundle_path.is_dir() else bundle_path
    bundle_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'mid-procedural-source-bundle-v1':
        raise ValueError('unsupported source bundle schema')
    entries = manifest.get('cases', [])
    seeds = [case.get('seed') for case in entries]
    if set(seeds) != OLD_SEEDS | SOURCE_ONLY_SEEDS or len(seeds) != len(set(seeds)):
        raise ValueError('expected exact five old seeds plus source-only 860015/860019')
    rows = old_rows(cases_path, OLD_SEEDS)
    old_manifest_path = samples_root / 'manifest.json'
    old_manifest = json.loads(old_manifest_path.read_text())
    case_reports = []
    for entry in entries:
        seed = entry['seed']
        if (entry.get('samples_per_channel'), entry.get('sample_rate_hz')) != (SAMPLES, RATE):
            raise ValueError(f'seed {seed}: unexpected sample geometry')
        recipe, signals, records = source_records(bundle_dir, entry)
        report = {'seed': seed, 'family_index': entry['family_index'], 'family': entry['family'],
                  'recipe': recipe, **records, 'short_clip_seconds': SAMPLES / RATE,
                  'scope_note': '0.3413-second source bundle; not long-coherence evidence'}
        if seed in OLD_SEEDS:
            report['recipe_matches_old_audit'] = recipe_matches(
                rows[(seed, 'manufactured_mid_damage', 'euler8')]['recipe'], recipe)
            if not report['recipe_matches_old_audit']:
                raise ValueError(f'seed {seed}: recipe does not match old audit')
            report['conditions'] = analyze_old_seed(seed, entry, recipe, signals, records, rows, samples_root)
            report['historical_fingerprint_limit'] = 'FNV is a noncryptographic parity check, not audit authentication; SHA256 pins cover source bundle and old Euler8 files.'
        else:
            report['conditions'] = None
            report['source_only_reason'] = 'no corresponding 860000..860011 native audit rows'
        case_reports.append(report)
    return {
        'schema': 'adverse-mid-source-analysis-v1',
        'scope': 'offline source parity; no neural inference; phase contribution is algebraic only and makes no stochastic aspiration phase-recovery claim',
        'provenance': {'analysis_source_sha256': sha256(Path(__file__)),
                       'bundle_manifest_sha256': sha256(manifest_path),
                       'old_cases_gzip_sha256': sha256(cases_path),
                       'old_run_manifest_sha256': sha256(old_manifest_path),
                       'old_source_head': old_manifest.get('source_head'),
                       'old_checkpoint_sha256': old_manifest.get('checkpoint_sha256'),
                       'source_sha256': manifest.get('provenance', {}).get('source_sha256'),
                       'old_samples_root': str(samples_root)},
        'source_decomposition_note': manifest.get('source_decomposition'),
        'metric_definition': 'strict cutoff < f < 8000 Hz; normalized f64 rFFT; one-sided Parseval weights; |Y-T|^2 = (|Y|-|T|)^2 + 2|Y||T|(1-cos phase delta); phase fraction is descriptive algebra, not recovered-phase evidence; relative NMSE null at target power <=1e-14, absolute terms retained',
        'cases': case_reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path, help='bundle directory or manifest.json')
    parser.add_argument('--cases', required=True, type=Path, help='archived native_solver_cpu_cases.jsonl.gz')
    parser.add_argument('--samples-root', required=True, type=Path, help='root for old waveform paths')
    parser.add_argument('--out', required=True, type=Path, help='exclusive JSON output path')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f'refusing existing output: {args.out}')
    result = analyze(args.bundle, args.cases, args.samples_root)
    with args.out.open('x', encoding='utf-8') as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write('\n')


if __name__ == '__main__':
    main()
