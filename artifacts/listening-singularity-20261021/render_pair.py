#!/usr/bin/env python3
"""Fresh, sequential quality-only rendering using the already pinned native binary.

Reuses the established /proc memory reader and native preservation receipts.
No speed comparison, retry, cap increase, or DSP configuration search.
"""
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from run_mid_profile import memory
from blind_listening import probe
from run_coherence_quality import wav_info, _assert_finite_float_wav


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


CONTROL_PREFIX_FILES = frozenset({
    'stage2_in.wav', 'stage2_mid/restored.wav', 'stage3_clean.wav',
    'stage3_5_microstructure.wav', 'stage4_spatial.wav',
})


def valid_sha256(value):
    return (isinstance(value, str) and len(value) == 64 and
            all(char in '0123456789abcdef' for char in value))


def additional_parity_pins(plan, binary):
    if 'additional_parity' not in plan:
        return {}
    declaration = plan['additional_parity']
    if (not isinstance(declaration, dict) or
        not isinstance(declaration.get('path'), str) or not declaration['path'] or
        not valid_sha256(declaration.get('sha256'))):
        raise RuntimeError('malformed additional parity declaration')
    path = (ROOT / declaration['path']).resolve()
    try:
        if sha(path) != declaration['sha256']:
            raise RuntimeError('additional parity pin mismatch')
        proof = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError('unreadable additional parity receipt') from error
    flags = ['all_wav_sha256_equal', 'all_non_timing_report_fields_equal',
             'complete_ph_audit_and_finalizer_metadata_exact',
             'real10s_complete_ph_audit_and_finalizer_metadata_exact']
    if not isinstance(proof, dict) or not all(proof.get(key) is True for key in flags):
        raise RuntimeError('additional memory-fix parity qualification absent')
    if (not isinstance(proof.get('pins'), dict) or
        proof['pins'].get(str(binary)) != plan['binary_sha256']):
        raise RuntimeError('additional parity receipt is not bound to planned binary')
    return {str(path): declaration['sha256']}


def control_prefix_pins(plan):
    if 'control_prefix_sha256' not in plan:
        return {}
    pins = plan['control_prefix_sha256']
    if (not isinstance(pins, dict) or not pins or
        not all(name in CONTROL_PREFIX_FILES and valid_sha256(expected)
                for name, expected in pins.items())):
        raise RuntimeError('malformed control prefix pins')
    return pins


def main(plan_path=None):
    plan_path = Path(plan_path).resolve() if plan_path else Path(__file__).with_name('plan.json')
    plan = json.loads(plan_path.read_text())
    revision_path = plan_path.with_name('plan_revision_v2.json')
    if revision_path.exists():
        revision = json.loads(revision_path.read_text())
        if revision['original_plan_sha256'] != sha(plan_path):
            raise RuntimeError('revision does not match original frozen plan')
        plan['gates'][0] = revision['replacement_prerequisite']
    limits = plan['limits']
    base = (ROOT / plan.get('run_root', 'runs/listening-singularity-20261021')).resolve()
    if not base.is_relative_to(ROOT / 'runs'):
        raise RuntimeError('run root must remain inside local runs/')
    run = base / 'fullsong-v1'
    if run.exists() or (base / 'HALT').exists() or (base / 'HALTED.json').exists():
        raise RuntimeError('existing run or latched session; no resume/retry')
    # This marker is produced after the separately scoped prior-song diagnostic.
    prerequisite = plan_path.with_name('attack_prerequisite.json')
    if json.loads(prerequisite.read_text())['eligible_to_continue_unchanged_cross_song'] is not True:
        raise RuntimeError('prior-song diagnostic not reviewed')
    if shutil.disk_usage(ROOT).free < limits['required_free_storage_gib'] * 1024**3:
        raise RuntimeError('insufficient free storage')
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:')))
    if available < limits['required_available_memory_mib'] * 1024:
        raise RuntimeError('insufficient available memory')
    original = Path(plan['source']['path'])
    binary, model = ROOT / plan['binary'], ROOT / plan['model']
    low = ROOT / 'artifacts/rich-low-sfht-v2/basis.json'
    for p, expected in [(original, plan['source']['sha256']),
                        (binary, plan['binary_sha256']), (model, plan['model_sha256'])]:
        if sha(p) != expected:
            raise RuntimeError(f'pin mismatch: {p}')
    gate = ROOT / 'runs/listening-coherence-20261020/real-gate-v3/COMPLETE.json'
    parity = ROOT / 'runs/ph-memory-20261021/parity-v1/COMPLETE.json'
    if not json.loads(gate.read_text())['eligible_for_bounded_listening']:
        raise RuntimeError('prior real geometry qualification absent')
    proof = json.loads(parity.read_text())
    if not all(proof[k] for k in ['all_wav_sha256_equal', 'all_non_timing_report_fields_equal',
                                  'complete_ph_audit_and_finalizer_metadata_exact']):
        raise RuntimeError('memory-fix parity qualification absent')
    extra_pins = additional_parity_pins(plan, binary)
    prefix_pins = control_prefix_pins(plan)
    run.mkdir(parents=True)
    env = os.environ.copy()
    env['LD_PRELOAD'] = '/data/data/com.termux/files/usr/lib/libc++_shared.so'
    env['OCL_ICD_ASSUME_ICD_EXTENSION'] = '1'
    for key in ['ROACH_PROFILE_MID', 'ROACH_MID_CACHE_MIB', 'ROACH_MID_CACHE_BUILD',
                'ROACH_MID_BATCH_ROWS', 'ROACH_FOREGROUND_CONFIRMED']:
        env.pop(key, None)
    rows = []
    def check_halt():
        if any((p / marker).exists() for p in [base, run]
               for marker in ['HALT', 'HALTED.json']):
            raise RuntimeError('latched session')
    try:
        check_halt()
        source = run / 'input-decoded.wav'
        decode = ['ffmpeg', '-nostdin', '-v', 'error', '-n', '-i', str(original),
                  '-map', '0:a:0', '-map_metadata', '-1', '-c:a', 'pcm_s16le', str(source)]
        # Source is already 48k/stereo/PCM16 FLAC; no resampling or lossy codec.
        subprocess.run(decode, env=env, check=True, timeout=60)
        geometry = {'rate': plan['source']['rate'], 'channels': plan['source']['channels'],
                    'frames': plan['source']['frames']}
        if probe(source) != geometry:
            raise RuntimeError('decoded geometry mismatch')
        args = ['auto-master', '--input', str(source), '--sfht-model', str(low),
                '--mid-model', str(model), '--backend', 'opencl', '--target-lufs', '-13',
                '--ceiling-db', '-1', '--steps', '8', '--chunk-seconds', '10',
                '--overlap-seconds', '2', '--morphic-mode', 'ph-lie-v2',
                '--clock-transient-punch', '--kick-transient-guard', '--quiet-mid-denoise',
                '--no-export-sdcard']
        pins = {str(p): sha(p) for p in [original, source, binary, model, low, gate,
                                         parity, prerequisite, plan_path, Path(__file__)]}
        # Retain the declared digest, including across decoder execution.
        pins.update(extra_pins)
        if revision_path.exists():
            pins[str(revision_path)] = sha(revision_path)
        save(run / 'protocol.json', {'plan': plan, 'pins': pins, 'shared_args': args,
            'prerequisite_review': json.loads(prerequisite.read_text()),
            'plan_revision': json.loads(revision_path.read_text()) if revision_path.exists() else None,
            'decode_argv': decode, 'speed_experiment': False})
        for name, extra in [('control', []), ('candidate', ['--mid-coherence'])]:
            for p, expected in pins.items():
                if sha(p) != expected:
                    raise RuntimeError(f'pin changed: {p}')
            check_halt()
            out = run / name
            command = [str(binary), *args, '--out', str(out), *extra]
            peak, observed, start = 0, False, time.monotonic()
            with (run / f'{name}.log').open('x') as log:
                child = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    while child.poll() is None:
                        m = memory(child.pid)
                        observed |= bool({'VmRSS', 'VmHWM'} & m.keys())
                        peak = max(peak, m.get('VmRSS', 0), m.get('VmHWM', 0))
                        if (peak > limits['process_hwm_mib'] * 1024 or
                            time.monotonic() - start > limits['child_timeout_seconds'] or
                            any((p / marker).exists() for p in [base, run] for marker in ['HALT', 'HALTED.json'])):
                            raise RuntimeError(f'resource/HALT stop: observed_peak_kib={peak}')
                        time.sleep(.2)
                finally:
                    if child.poll() is None:
                        child.terminate()
                        try:
                            child.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                    save(run / f'{name}-execution.json', {
                        'argv': command, 'child_returncode': child.returncode,
                        'peak_sampled_process_hwm_mib': peak / 1024 if observed else None,
                        'elapsed_seconds_watchdog_only': time.monotonic() - start,
                        'speed_experiment': False})
            if child.returncode != 0 or not observed:
                raise RuntimeError(f'{name} failed or process memory unobserved: {child.returncode}')
            check_halt()
            if time.monotonic() - start > limits['child_timeout_seconds']:
                raise RuntimeError('child timeout at final poll')
            summary = json.loads((out / 'summary.json').read_text())
            stats, settings = summary['master_stats']['after'], summary['settings']
            if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in stats.values()):
                raise RuntimeError('nonfinite master metrics')
            if stats['true_peak_dbtp'] > -.99 or abs(stats['lufs_integrated'] + 13) > .1:
                raise RuntimeError('master level/peak gate failed')
            if probe(out / 'mastered.wav') != geometry:
                raise RuntimeError('master geometry mismatch')
            _assert_finite_float_wav(out / 'mastered.wav', wav_info(out / 'mastered.wav'))
            checks = summary['preservation']
            if not all(checks[k] is True for k in ['mono_compatibility_passed',
                        'known_band_preserved', 'transient_timing_passed']):
                raise RuntimeError('native preservation gate failed')
            if settings['force_specialists'] or not settings['master_controls']['quiet_mid_denoise']:
                raise RuntimeError('routing/control invariant failed')
            if settings['mid_coherence_applied'] is not (name == 'candidate'):
                raise RuntimeError('unexpected intervention application or Mid abstention')
            if Path(summary['mid_model']).resolve() != model.resolve():
                raise RuntimeError('model path mismatch')
            prefix_observed = {}
            if name == 'control':
                for filename, expected in prefix_pins.items():
                    path = out / filename
                    if not path.is_file() or sha(path) != expected:
                        raise RuntimeError(f'control prefix mismatch: {filename}')
                    prefix_observed[filename] = expected
            rows.append({'name': name, 'argv': command, 'settings': settings,
                         'stats': stats, 'preservation': checks,
                         'peak_sampled_process_hwm_mib': peak / 1024,
                         'mastered_sha256': sha(out / 'mastered.wav'),
                         'master_json_sha256': sha(out / 'master.json'),
                         'summary_sha256': sha(out / 'summary.json')})
            if prefix_observed:
                rows[-1]['control_prefix_sha256'] = prefix_observed
            check_halt()
            save(run / f'{name}-receipt.json', rows[-1])
        omit = {'mid_coherence_requested', 'mid_coherence_applied', 'explicit_cli_parameters'}
        if {k:v for k,v in rows[0]['settings'].items() if k not in omit} != {
                k:v for k,v in rows[1]['settings'].items() if k not in omit}:
            raise RuntimeError('paired settings differ beyond declared intervention')
        if rows[0]['mastered_sha256'] == rows[1]['mastered_sha256']:
            raise RuntimeError('candidate is identical to control')
        for p, expected in pins.items():
            if sha(p) != expected:
                raise RuntimeError(f'pin changed at completion: {p}')
        check_halt()
        save(run / 'COMPLETE.json', {'schema': 'cross-song-coherence-eligible-v1',
            'eligible_for_listening': True, 'source_sha256': sha(original),
            'decoded_input_sha256': sha(source), 'geometry': geometry, 'rows': rows,
            'protocol_sha256': sha(run / 'protocol.json'), 'production_promoted': False,
            'owner_judgments_received': False})
    except BaseException as error:
        save(run / 'HALTED.json', {'reason': str(error), 'completed_rows': rows})
        raise


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path)
    main(parser.parse_args().plan)
