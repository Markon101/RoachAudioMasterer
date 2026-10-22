#!/usr/bin/env python3
"""Two independent, sequential full-song quality tests; no speed experiment."""
from pathlib import Path
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_mid_profile import memory, digest
from run_coherence_quality import wav_info, _assert_finite_float_wav
from blind_listening import probe


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def available_mib():
    return int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                    if line.startswith('MemAvailable:'))) / 1024


def main():
    plan_path = HERE / 'plan.json'
    plan = json.loads(plan_path.read_text())
    if plan['target_lufs'] != -14 or plan['ceiling_dbtp'] != -1:
        raise RuntimeError('this experiment requires requested -14 LUFS / -1 dBTP plan')
    run = (ROOT / plan['run_root']).resolve()
    if not run.is_relative_to(ROOT / 'runs') or run.exists():
        raise RuntimeError('fresh local run directory required')
    if (HERE / 'HALT').exists() or (HERE / 'HALTED.json').exists():
        raise RuntimeError('latched experiment')
    spec = importlib.util.spec_from_file_location('qualified_retry', ROOT /
        'artifacts/listening-singularity-20261021/render_pair.py')
    helpers = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helpers)
    binary, model, low = [ROOT / plan[key] for key in ['binary', 'model', 'low_model']]
    for path, expected in [(binary, plan['binary_sha256']), (model, plan['model_sha256']),
                           (low, plan['low_model_sha256'])]:
        if digest(path) != expected:
            raise RuntimeError(f'pin mismatch: {path}')
    extra_pins = helpers.additional_parity_pins(plan, binary)
    for track in plan['tracks']:
        if digest(track['path']) != track['sha256']:
            raise RuntimeError('original source pin mismatch')
        if probe(track['path']) != {k: track[k] for k in ['rate', 'channels', 'frames']}:
            raise RuntimeError('source geometry mismatch')
    pins = {str(path): digest(path) for path in [plan_path, Path(__file__), binary,
            model, low, ROOT / 'artifacts/listening-singularity-20261021/render_pair.py']}
    pins.update(extra_pins)
    pins.update({track['path']: track['sha256'] for track in plan['tracks']})
    limits = plan['limits']
    run.mkdir(parents=True)
    rows = []

    def check():
        if any((base / name).exists() for base in [HERE, run]
               for name in ['HALT', 'HALTED.json']):
            raise RuntimeError('latched experiment')
        for path, expected in pins.items():
            if digest(path) != expected:
                raise RuntimeError(f'pin changed: {path}')

    save(run / 'protocol.json', {'schema': 'around-run-protocol-v1', 'plan': plan, 'pins': pins})
    env = os.environ.copy()
    env['LD_PRELOAD'] = '/data/data/com.termux/files/usr/lib/libc++_shared.so'
    env['OCL_ICD_ASSUME_ICD_EXTENSION'] = '1'
    for key in ['ROACH_PROFILE_MID', 'ROACH_MID_CACHE_MIB', 'ROACH_MID_CACHE_BUILD',
                'ROACH_MID_BATCH_ROWS', 'ROACH_FOREGROUND_CONFIRMED']:
        env.pop(key, None)
    try:
        for track in plan['tracks']:
            check()
            if shutil.disk_usage(ROOT).free < limits['minimum_free_storage_gib'] * 1024**3:
                raise RuntimeError('storage reserve failed')
            if available_mib() < limits['minimum_memavailable_mib']:
                raise RuntimeError('available-memory reserve failed')
            out = run / track['id']
            command = [str(binary), 'auto-master', '--input', track['path'], '--out', str(out),
                '--auto-tune', '--backend', 'opencl', '--sfht-model', str(low), '--mid-model', str(model),
                '--target-lufs', '-14', '--ceiling-db', '-1', '--steps', '8',
                '--chunk-seconds', '10', '--overlap-seconds', '2', '--morphic-mode', 'ph-lie-v2',
                '--clock-transient-punch', '--kick-transient-guard', '--quiet-mid-denoise',
                '--phase-consistency', '0.5', '--no-export-sdcard']
            peak, seen, start = 0, False, time.monotonic()
            min_available = available_mib()
            with (run / (track['id'] + '.log')).open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        mem = memory(child.pid)
                        seen |= bool({'VmRSS', 'VmHWM'} & mem.keys())
                        peak = max(peak, mem.get('VmRSS', 0), mem.get('VmHWM', 0))
                        min_available = min(min_available, available_mib())
                        if peak > limits['process_hwm_mib'] * 1024:
                            raise RuntimeError('process memory ceiling exceeded')
                        if min_available < limits['emergency_memavailable_mib']:
                            raise RuntimeError('phone memory emergency reserve failed')
                        if time.monotonic() - start > limits['child_timeout_s']:
                            raise RuntimeError('child timeout')
                        if any((base / name).exists() for base in [HERE, run]
                               for name in ['HALT', 'HALTED.json']):
                            raise RuntimeError('HALT during child')
                        time.sleep(.25)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    save(run / (track['id'] + '-execution.json'), {'argv': command,
                        'returncode': child.returncode, 'observed_process_hwm_mib': peak / 1024 if seen else None,
                        'minimum_observed_phone_available_mib': min_available,
                        'elapsed_seconds_watchdog_only': time.monotonic() - start, 'speed_experiment': False})
            check()
            if child.returncode != 0 or not seen or time.monotonic() - start > limits['child_timeout_s']:
                raise RuntimeError('native process failed or monitoring unavailable')
            summary = json.loads((out / 'summary.json').read_text())
            stats = summary['master_stats']['after']
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in stats.values()):
                raise RuntimeError('nonfinite master statistics')
            if probe(out / 'mastered.wav') != {k: track[k] for k in ['rate', 'channels', 'frames']}:
                raise RuntimeError('output geometry mismatch')
            _assert_finite_float_wav(out / 'mastered.wav', wav_info(out / 'mastered.wav'))
            settings = summary['settings']
            if settings['mid_coherence_applied'] or settings['force_specialists']:
                raise RuntimeError('unexpected experimental/forced processing')
            if settings['target_lufs'] != -14 or Path(summary['mid_model']).resolve() != model.resolve():
                raise RuntimeError('wrong loudness target or model')
            checks = summary['preservation']
            gates = {k: checks[k] is True for k in ['mono_compatibility_passed',
                     'known_band_preserved', 'transient_timing_passed']}
            gates['true_peak'] = stats['true_peak_dbtp'] <= plan['validation']['max_native_true_peak_dbtp']
            gates['target_loudness'] = abs(stats['lufs_integrated'] - plan['target_lufs']) <= plan['validation']['max_target_error_lu']
            row = {'id': track['id'], 'argv': command, 'source_sha256': track['sha256'],
                'mastered_sha256': digest(out / 'mastered.wav'), 'flac_sha256': digest(out / 'mastered.flac'),
                'summary_sha256': digest(out / 'summary.json'), 'native_stats': stats,
                'preservation': checks, 'quality_checks': gates, 'qualified': all(gates.values()),
                'observed_process_hwm_mib': peak / 1024, 'settings': settings}
            check()
            save(run / (track['id'] + '-receipt.json'), row)
            rows.append(row)
            print(json.dumps({'track': track['id'], 'render_complete': True, 'qualified': row['qualified']}), flush=True)
        check()
        save(run / 'COMPLETE.json', {'schema': 'around-full-tests-complete-v1',
            'rows': rows, 'protocol_sha256': digest(run / 'protocol.json'),
            'all_qualified': all(row['qualified'] for row in rows), 'production_promoted': False})
    except BaseException as error:
        save(run / 'HALTED.json', {'reason': str(error), 'completed_tracks': rows})
        raise


if __name__ == '__main__':
    main()
