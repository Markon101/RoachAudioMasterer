#!/usr/bin/env python3
"""Run a hash-pinned, passive 45-case assessment routing matrix.

This calls only the native ``assess`` command. It does not run restoration or
intent modulation. Outputs are diagnostic records, not quality evaluations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from long_coherence_fixture import _write_float_wav
from roach_ears.audio_io import read_wav
from audit_assessment_bands import audit
from run_mid_profile import memory
from run_coherence_quality import wav_info

RATE = 48_000
GAINS_DB = (0.0, -20.0, -40.0)
EXPECTED_SEEDS = (860002, 860003, 860004, 860005, 860007, 860015, 860019)
RSS_LIMIT_KIB = 384 * 1024
TIMEOUT_SECONDS = 60


def digest(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_wav(path: Path, samples: np.ndarray, channels_first: bool) -> None:
    """Use canonical fixture writer for stereo and equivalent float WAV for mono."""
    x = np.asarray(samples, dtype=np.float32)
    if channels_first:
        x = x.T
    if x.ndim != 2 or x.shape[1] not in (1, 2):
        raise ValueError('expected sample-major mono or stereo float32')
    if x.shape[1] == 2:
        _write_float_wav(path, x)
        return
    data = np.asarray(x[:, 0], dtype='<f4').tobytes()
    fmt = struct.pack('<HHIIHH', 3, 1, RATE, RATE * 4, 4, 32)
    body = b'WAVE' + b'fmt ' + struct.pack('<I', len(fmt)) + fmt
    body += b'data' + struct.pack('<I', len(data)) + data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'RIFF' + struct.pack('<I', len(body)) + body)


def scale_samples(samples: np.ndarray, gain_db: float) -> np.ndarray:
    if type(gain_db) not in (int, float) or not math.isfinite(float(gain_db)) or gain_db > 0:
        raise ValueError('gain must be finite and nonpositive')
    x = np.asarray(samples, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('source has nonfinite samples')
    factor = np.float32(10.0 ** (float(gain_db) / 20.0))
    result = np.asarray(x * factor, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('gain produced nonfinite samples')
    return result


def load_plan(path: Path, out: Path) -> tuple[dict, list[dict]]:
    plan = json.loads(Path(path).read_text())
    required = ('binary', 'binary_sha256', 'bundle_manifest', 'bundle_sha256',
                'long_identity', 'long_identity_sha256', 'source_pins', 'gains_db')
    if any(k not in plan for k in required):
        raise ValueError('plan is missing required pinned paths or gains')
    if tuple(plan['gains_db']) != GAINS_DB:
        raise ValueError('gain policy must be exactly [0, -20, -40] dB')
    binary = Path(plan['binary'])
    manifest_path = Path(plan['bundle_manifest'])
    identity_path = Path(plan['long_identity'])
    for p, expected, label in ((binary, plan['binary_sha256'], 'binary'),
                               (manifest_path, plan['bundle_sha256'], 'bundle manifest'),
                               (identity_path, plan['long_identity_sha256'], 'long identity')):
        if not p.is_file() or digest(p) != expected:
            raise ValueError(f'{label} is missing or hash-mismatched')
    if not isinstance(plan['source_pins'], list) or not plan['source_pins']:
        raise ValueError('source_pins must be a nonempty list')
    pinned = {}
    for pin in plan['source_pins']:
        p = Path(pin['path'])
        if not p.is_file() or digest(p) != pin['sha256']:
            raise ValueError(f'source pin mismatch: {p}')
        pinned[p.resolve()] = pin['sha256']
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('schema') != 'mid-procedural-source-bundle-v1':
        raise ValueError('unsupported procedural bundle manifest')
    cases = manifest.get('cases', [])
    if {int(c.get('seed', -1)) for c in cases} != set(EXPECTED_SEEDS) or len(cases) != 7:
        raise ValueError('bundle must contain the fixed seven procedural seeds')
    identity_info = wav_info(identity_path)
    if (identity_info['rate'] != RATE or identity_info['frames'] != 26 * RATE
            or identity_info['channels'] != 2 or identity_info['format'] != 3
            or identity_info['bits_per_sample'] != 32):
        raise ValueError('long identity must be 26-second 48 kHz stereo float32 WAV')
    identity = read_wav(identity_path)
    if identity.sample_rate != RATE or identity.channels != 2 or not np.isfinite(identity.samples).all():
        raise ValueError('long identity contains invalid decoded values')
    if out.exists():
        raise FileExistsError(f'refusing existing output path: {out}')
    prepared = []
    root = manifest_path.parent
    for entry in sorted(cases, key=lambda c: int(c['seed'])):
        seed = int(entry['seed'])
        recipe_path = root / entry['recipe_path']
        if digest(recipe_path) != entry['recipe_sha256']:
            raise ValueError(f'seed {seed}: recipe hash mismatch')
        if pinned.get(recipe_path.resolve()) != entry['recipe_sha256']:
            raise ValueError(f'seed {seed}: recipe is absent from source_pins')
        recipe = json.loads(recipe_path.read_text())
        family = entry.get('family', recipe.get('family', 'unspecified'))
        if recipe.get('seed') != seed or recipe.get('family') != family:
            raise ValueError(f'seed {seed}: recipe seed/family differs from manifest')
        for role in ('target', 'degraded_input'):
            spec = entry['files'][role]
            raw_path = root / spec['planar_f32le']['path']
            wav_path = root / spec['wav']['path']
            if digest(raw_path) != spec['planar_f32le']['sha256'] or digest(wav_path) != spec['wav']['sha256']:
                raise ValueError(f'seed {seed}/{role}: bundle source hash mismatch')
            if (pinned.get(raw_path.resolve()) != spec['planar_f32le']['sha256']
                    or pinned.get(wav_path.resolve()) != spec['wav']['sha256']):
                raise ValueError(f'seed {seed}/{role}: audio is absent from source_pins')
            info = wav_info(wav_path)
            if (info['rate'] != RATE or info['frames'] != 16_384
                    or info['channels'] not in (1, 2) or info['format'] != 3
                    or info['bits_per_sample'] != 32):
                raise ValueError(f'seed {seed}/{role}: expected 16384-frame float32 WAV at 48 kHz')
            audio = read_wav(wav_path)
            raw = np.frombuffer(raw_path.read_bytes(), dtype='<f4')
            if (audio.sample_rate != RATE or audio.channels not in (1, 2)
                    or raw.size != audio.channels * audio.num_samples
                    or not np.isfinite(audio.samples).all()):
                raise ValueError(f'seed {seed}/{role}: invalid audio geometry or values')
            planar = raw.reshape(audio.channels, audio.num_samples)
            if not np.array_equal(planar.view(np.uint32), audio.samples.view(np.uint32)):
                raise ValueError(f'seed {seed}/{role}: WAV differs from pinned planar source')
            for gain in GAINS_DB:
                prepared.append({'id': f'seed-{seed}-{role}-g{int(gain)}', 'seed': seed,
                                 'role': role, 'family': family, 'gain_db': gain,
                                 'samples': audio.samples.copy(), 'channels': audio.channels,
                                 'origin_sha256': spec['wav']['sha256']})
    if pinned.get(identity_path.resolve()) != plan['long_identity_sha256']:
        raise ValueError('long identity is absent from source_pins')
    for gain in GAINS_DB:
        prepared.append({'id': f'long-identity-g{int(gain)}', 'seed': None,
                         'role': 'target_as_input_no_added_damage',
                         'family': 'long_coherence_procedural_identity', 'gain_db': gain,
                         'samples': identity.samples.copy(), 'channels': identity.channels,
                         'origin_sha256': plan['long_identity_sha256']})
    if len(prepared) != 45:
        raise ValueError('internal matrix size is not 45')
    return plan, prepared


def _halt(out: Path, reason: str, row_id: str | None, completed: int) -> None:
    marker = out / 'HALTED.json'
    if not marker.exists():
        marker.write_text(json.dumps({'schema': 'passive-routing-halted-v1',
            'reason': reason, 'row_id': row_id, 'completed_rows': completed}, indent=2) + '\n')


def _check_pins(plan: dict) -> None:
    for path, key in ((plan['binary'], 'binary_sha256'),
                      (plan['bundle_manifest'], 'bundle_sha256'),
                      (plan['long_identity'], 'long_identity_sha256')):
        if digest(Path(path)) != plan[key]:
            raise ValueError(f'pinned source changed: {path}')
    for pin in plan['source_pins']:
        if digest(Path(pin['path'])) != pin['sha256']:
            raise ValueError(f'pinned source changed: {pin["path"]}')


def _band_mid_metrics(observer: dict, channels: int) -> dict:
    key = 'mid_orthonormal' if channels == 2 else 'mono'
    bands = observer['channels'][key]['bands']
    names = ('sub_0_120_hz', 'low_mid_120_500_hz', 'mid_500_6000_hz')
    return {name: {'rms_dbfs': bands[name]['rms_dbfs'],
                   'power_below_null_floor': bands[name]['power_below_null_floor']}
            for name in names}


def _run_row(plan: dict, row: dict, row_dir: Path) -> dict:
    if not row_dir.is_dir():
        row_dir.mkdir(parents=True)
    source = row['samples']
    scaled = scale_samples(source, row['gain_db'])
    wav = row_dir / 'input.wav'
    if not wav.is_file():
        raise ValueError('preflight input WAV is missing')
    if digest(wav) != row['expected_input_sha256']:
        raise ValueError('preflight input WAV hash changed')
    decoded = read_wav(wav)
    if decoded.channels != row['channels'] or decoded.num_samples != scaled.shape[1] or decoded.sample_rate != RATE:
        raise ValueError('generated WAV geometry changed')
    if not np.array_equal(decoded.samples.view(np.uint32), scaled.view(np.uint32)):
        raise ValueError('generated WAV did not round-trip exact f32 samples')
    assessment_dir = row_dir / 'assessment'
    log = row_dir / 'native_assess.log'
    command = [str(plan['binary']), 'assess', '--input', str(wav), '--out', str(assessment_dir)]
    start = time.monotonic()
    peak_rss = peak_hwm = 0
    memory_observed = False
    with log.open('x') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while process.poll() is None:
                counters = memory(process.pid)
                memory_observed = memory_observed or any(k in counters for k in ('VmRSS', 'VmHWM'))
                peak_rss = max(peak_rss, counters.get('VmRSS', 0))
                peak_hwm = max(peak_hwm, counters.get('VmHWM', 0))
                if (time.monotonic() - start > TIMEOUT_SECONDS
                        or max(peak_rss, peak_hwm) > RSS_LIMIT_KIB):
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise RuntimeError('native assess timeout or RSS/HWM cap exceeded')
                if (row_dir.parent.parent / 'HALT').exists():
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait()
                    raise RuntimeError('explicit HALT marker observed')
                time.sleep(0.1)
        except BaseException:
            if process.poll() is None:
                process.terminate()
                try: process.wait(timeout=3)
                except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if process.returncode != 0:
        raise RuntimeError(f'native assess exited {process.returncode}')
    if row_dir.parent.parent.joinpath('HALT').exists():
        raise RuntimeError('explicit HALT marker observed')
    assessment_path, card_path = assessment_dir/'assessment.json', assessment_dir/'tuning_card.json'
    if not assessment_path.is_file() or not card_path.is_file():
        raise ValueError('native assess outputs are incomplete')
    assessment = json.loads(assessment_path.read_text())
    card = json.loads(card_path.read_text())
    observer = audit(wav, assessment_path)
    observer_path = row_dir/'observer.json'
    observer_path.write_text(json.dumps(observer, allow_nan=False, separators=(',', ':'))+'\n')
    authorities = assessment['authorities']
    params = card['recommended_parameters']
    row_record = {
        'schema': 'passive-routing-row-v1', 'id': row['id'], 'seed': row['seed'],
        'role': row['role'], 'family': row['family'], 'gain_db': row['gain_db'],
        'intent_scope': 'base-only; MusicalIntentCard modulation not applied',
        'origin_sha256': row['origin_sha256'], 'input_sha256': digest(wav),
        'binary_sha256': plan['binary_sha256'],
        'assessment_sha256': digest(assessment_path), 'tuning_card_sha256': digest(card_path),
        'native_log_sha256': digest(log),
        'geometry': {'sample_rate_hz': RATE, 'channels': decoded.channels, 'samples': decoded.num_samples},
        'level': {'peak_sample': float(np.max(np.abs(scaled))),
                  'rms': float(np.sqrt(np.mean(np.asarray(scaled, dtype=np.float64) ** 2)))},
        'legacy_scores_uncalibrated': {k: assessment.get(k) for k in (
            'sub_energy_dbfs', 'low_mid_energy_dbfs', 'mid_energy_dbfs',
            'high_energy_dbfs', 'ultrasonic_energy_dbfs')},
        'base_authorities': authorities,
        'base_tuning_parameters': params,
        'base_stage_actions': card['stage_actions'],
        'calibrated_mid_bands': _band_mid_metrics(observer, decoded.channels),
        'observer_source_sha256': observer['provenance']['observer_source_sha256'],
        'observer_sha256': digest(observer_path),
        'observer_scope': 'whole-clip physical band measurement; diagnostic, not a routing score',
        'sampled_peak_rss_kib': peak_rss if memory_observed else None,
        'sampled_process_hwm_kib': peak_hwm if memory_observed else None,
    }
    return row_record


def run(plan_path: Path, out: Path) -> None:
    if out.exists():
        raise FileExistsError(f'refusing existing output root: {out}')
    try:
        plan, rows = load_plan(plan_path, out)
    except BaseException as exc:
        out.mkdir(parents=True)
        (out/'plan.json').write_bytes(Path(plan_path).read_bytes())
        _halt(out, str(exc), None, 0)
        raise
    out.mkdir(parents=True)
    (out/'plan.json').write_bytes(Path(plan_path).read_bytes())
    (out/'rows').mkdir()
    completed = 0
    try:
        # Materialize and hash the complete derived-input set before the first
        # native child starts, so a bad late row cannot follow partial runs.
        for row in rows:
            if (out/'HALT').exists():
                raise RuntimeError('explicit HALT marker observed during input preflight')
            row_dir = out/'rows'/row['id']
            row_dir.mkdir()
            scaled = scale_samples(row['samples'], row['gain_db'])
            wav = row_dir/'input.wav'
            write_wav(wav, scaled, channels_first=True)
            decoded = read_wav(wav)
            if (decoded.channels != row['channels'] or decoded.num_samples != scaled.shape[1]
                    or decoded.sample_rate != RATE or not np.array_equal(decoded.samples.view(np.uint32), scaled.view(np.uint32))):
                raise ValueError(f'{row["id"]}: generated input failed exact WAV roundtrip')
            row['expected_input_sha256'] = digest(wav)
        for row in rows:
            if (out/'HALT').exists():
                raise RuntimeError('explicit HALT marker observed')
            _check_pins(plan)
            row_dir = out/'rows'/row['id']
            if (not row_dir.is_dir() or not (row_dir/'input.wav').is_file()
                    or (row_dir/'assessment').exists() or (row_dir/'native_assess.log').exists()
                    or (row_dir/'observer.json').exists()):
                raise FileExistsError(f'row output is not a clean preflight directory: {row_dir}')
            record = _run_row(plan, row, row_dir)
            with (out/'rows.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(record, allow_nan=False, separators=(',', ':'))+'\n')
                stream.flush(); os.fsync(stream.fileno())
            completed += 1
    except BaseException as exc:
        _halt(out, str(exc), row.get('id') if 'row' in locals() else None, completed)
        raise
    try:
        _check_pins(plan)
        complete = {'schema': 'passive-routing-complete-v1', 'row_count': completed,
                    'source_pins': plan['source_pins'], 'binary_sha256': plan['binary_sha256'],
                    'bundle_sha256': plan['bundle_sha256'],
                    'long_identity_sha256': plan['long_identity_sha256'],
                    'runner_source_sha256': digest(Path(__file__)),
                    'observer_source_sha256': digest(HERE/'audit_assessment_bands.py'),
                    'scope': 'passive base assessment and base tuning card; no intent modulation or model inference'}
        with (out/'COMPLETE.json').open('x', encoding='utf-8') as stream:
            json.dump(complete, stream, indent=2, allow_nan=False); stream.write('\n')
    except BaseException as exc:
        _halt(out, str(exc), None, completed)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    run(args.plan, args.out)


if __name__ == '__main__':
    main()
