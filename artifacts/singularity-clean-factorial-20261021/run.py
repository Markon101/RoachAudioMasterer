"""One create-only, sequential native clean factorial; no listening or mastering."""
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
from run_mid_profile import memory
from matched_attack_audit import WavReader, memory_receipt
import trace_preservation_shape as trace
import numpy as np


def save(path, value):
    with Path(path).open('x') as file:
        json.dump(value, file, indent=2, allow_nan=False)
        file.write('\n')


def check_latches(*paths):
    trace.check_latches(*paths)
    for path in paths:
        if (Path(path) / 'HALTED.json').exists():
            raise RuntimeError(f'latched session: {path}')


def check_pins(pins):
    for path, expected in pins.items():
        if trace.digest(ROOT / path) != expected:
            raise RuntimeError(f'pin changed: {path}')


def require_combined(actual, expected):
    if actual != expected:
        raise RuntimeError('combined output does not exactly reproduce frozen stage3_clean.wav')


def factorial(rows, baseline):
    fields = ['signed_shape_change_from_source_db', 'filtered_band_rms_change_from_source_db',
              'broadband_rms_change_from_source_db']
    return {field: rows['combined'][field] - rows['denoise'][field] - rows['eq'][field]
                   + rows[baseline][field] for field in fields}


def scan(path, geometry):
    reader = WavReader(path)
    try:
        if {key: getattr(reader, key) for key in geometry} != geometry:
            raise RuntimeError(f'geometry/alignment mismatch: {path}')
        global_peak = 0.
        powers = np.zeros(2, dtype=np.float64)
        for start in range(0, reader.frames, 65536):
            values = reader.read(start, min(reader.frames, start + 65536))
            global_peak = max(global_peak, float(np.max(np.abs(values))))
            powers += np.sum(values * values, axis=0)
        return {'global_peak': global_peak, 'channel_full_rms': np.sqrt(powers / reader.frames).tolist()}
    finally:
        reader.close()


def run_arm(binary, spec, plan, run, pins, deadline):
    check_latches(HERE, run, run.parent)
    check_pins(pins)
    out = run / spec['id']
    command = [str(binary), '--input', str(ROOT / plan['mid_input']),
               '--out', str(out), '--arm', spec['id']]
    start, peak, observed, child = time.monotonic(), 0, False, None
    try:
        with (run / (spec['id'] + '.log')).open('x') as log:
            child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            while True:
                sample = memory(child.pid)
                observed |= bool({'VmHWM', 'VmRSS'} & sample.keys())
                peak = max(peak, sample.get('VmHWM', 0), sample.get('VmRSS', 0))
                result = child.poll()
                check_latches(HERE, run, run.parent)
                if peak > plan['limits']['child_hwm_mib'] * 1024:
                    raise RuntimeError('native child exceeds unchanged 512 MiB aim')
                if time.monotonic() - start > plan['limits']['child_timeout_seconds'] or time.monotonic() > deadline:
                    raise RuntimeError('native child/overall deadline exceeded')
                if result is not None:
                    break
                time.sleep(.05)
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        if child is not None:
            save(run / (spec['id'] + '-execution.json'), {
                'argv': command, 'child_returncode': child.returncode,
                'sampled_hwm_mib': peak / 1024 if observed else None,
                'watchdog_elapsed_seconds': time.monotonic() - start, 'speed_experiment': False})
    if child.returncode != 0:
        raise RuntimeError(f'native child failed: {spec["id"]}: {child.returncode}')
    native = json.loads((out / 'arm.json').read_text())
    if native['native_hwm_kib'] > plan['limits']['child_hwm_mib'] * 1024:
        raise RuntimeError('native end receipt exceeds unchanged 512 MiB aim')
    check_pins(pins)
    output_hash = trace.digest(out / 'output.wav')
    if spec['id'] == 'combined':
        require_combined(output_hash, plan['pins'][plan['cached_combined']])
    if spec['id'] == 'identity' and output_hash != plan['pins'][plan['mid_input']]:
        raise RuntimeError('raw identity bytecopy failed')
    return {'id': spec['id'], 'argv': command, 'output_sha256': output_hash,
            'native': native, 'maximum_hwm_mib': max(peak, native['native_hwm_kib']) / 1024}


def report(summary):
    lines = ['# Singularity clean-stage factorial', '',
             'Combined output exactly matches the frozen cached clean-stage WAV. This is a single cached full-song diagnostic, with no native-gate replacement, repair or listening claim.', '',
             '| Arm | Shape vs source dB | Increment vs Mid dB | Band RMS vs Mid dB | Broadband RMS vs Mid dB | Global peak |',
             '|---|---:|---:|---:|---:|---:|']
    for row in summary['rows'].values():
        lines.append(f"| {row['id']} | {row['signed_shape_change_from_source_db']:+.9f} | {row['signed_increment_from_previous_boundary_db']:+.9f} | {row['filtered_band_rms_increment_db']:+.9f} | {row['broadband_rms_increment_db']:+.9f} | {row['global_peak']:.9f} |")
    primary = summary['interaction_with_raw_identity']['signed_shape_change_from_source_db']
    guarded = summary['interaction_with_guard_only']['signed_shape_change_from_source_db']
    lines += ['', f'Explicit signed-shape interaction, combined − denoise − EQ + raw identity: **{primary:+.9f} dB**. Using guard-only as the operational (D=0,E=0) baseline: **{guarded:+.9f} dB**.', '',
              'Raw band and broadband RMS, powers, floor status and per-channel RMS are retained in summary.json. The first-channel analytic NumPy Parseval proxy matches the earlier trace definition; it approximates native f32 Fourier filtering and reduction, rather than replacing the native <1 dB gate.', '',
              'Every clean-function arm includes the native global 0.99 peak guard. Guard-only is not the auto-master bypass: auto-master skips clean when both flags are false. Its exact positive scale and f32 rounding can change raw RMS; positive uniform gain cancels in the normalized shape proxy above its floors. Denoise includes high-band and dual-zone quiet-mid denoising. EQ follows denoise in the combined arm, so its input-dependent curve and quiet gating can interact. Per-arm pre-guard peaks/gains are not instrumented.', '',
              'Exact geometry and construction from the same cached input are checked; no shifts, resampling, event alignment fitting or temporal-preservation claim. Branch contrasts describe these implementations, including STFT roundoff, quiet gating and final headroom protection; they do not isolate audible damage or perceptual benefit.', '',
              f"Maximum native child HWM: {summary['maximum_native_hwm_mib']:.3f} MiB; analysis HWM: {summary['analysis_hwm_mib']:.3f} MiB. These are containment receipts, not speed measurements.", '',
              'The original Singularity HALTED record and failure remain binding. No settings were fitted, candidate mastered, listening package created or production source changed.', '']
    return '\n'.join(lines)


def main():
    plan_path = HERE / 'plan.json'
    check_latches(HERE)
    plan = json.loads(plan_path.read_text())
    run = ROOT / plan['run_root']
    check_latches(run, run.parent)
    if run.exists():
        raise RuntimeError('fresh raw run required; no resume or overwrite')
    executions = []
    try:
        check_pins(plan['pins'])
        build = json.loads((HERE / 'build.json').read_text())
        if build['plan_sha256'] != trace.digest(plan_path):
            raise RuntimeError('build receipt does not match frozen plan')
        if shutil.disk_usage(ROOT).free < plan['limits']['minimum_free_storage_gib'] * 1024**3:
            raise RuntimeError('insufficient free phone storage')
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))
        if available < plan['limits']['minimum_available_memory_mib'] * 1024:
            raise RuntimeError('insufficient available phone memory')
        run.mkdir(parents=True)
        binary = run / 'clean-factorial-v1'
        with Path(build['binary']).open('rb') as source, binary.open('xb') as destination:
            shutil.copyfileobj(source, destination)
        binary.chmod(0o700)
        if trace.digest(binary) != build['binary_sha256']:
            raise RuntimeError('built binary pin mismatch')
        pins = {**plan['pins'], str(plan_path): trace.digest(plan_path),
                str(HERE / 'build.json'): trace.digest(HERE / 'build.json'),
                str(binary): build['binary_sha256']}
        save(HERE / 'manifest.json', {'plan': plan, 'pins': pins, 'build': build,
             'started_utc': trace.utc(), 'python': sys.version, 'numpy': np.__version__,
             'platform': platform.platform(), 'environment': {key: os.environ.get(key) for key in ('LD_PRELOAD', 'OPENBLAS_NUM_THREADS', 'PYTHONDONTWRITEBYTECODE')}})
        deadline = time.monotonic() + plan['limits']['overall_timeout_seconds']
        with (HERE / 'executions.jsonl').open('x') as stream:
            for spec in plan['arms_execution_order']:
                row = run_arm(binary, spec, plan, run, pins, deadline)
                executions.append(row)
                stream.write(json.dumps(row, allow_nan=False) + '\n')
                stream.flush()
        # No contrast is measured before combined byte parity has passed.
        resources = {'max_frames': plan['geometry']['frames'], 'input_block_frames': 65536,
                     'spectrum_block_bins': 65536, 'process_hwm_aim_mib': 512}
        source = trace.measure_path(ROOT / plan['raw_decoded_source'], plan['geometry'], resources)
        mid = trace.measure_path(ROOT / plan['mid_input'], plan['geometry'], resources)
        rows = {}
        for name in plan['arms_report_order']:
            check_latches(HERE, run, run.parent)
            check_pins(pins)
            started = time.monotonic()
            path = run / name / 'output.wav'
            measured = trace.measure_path(path, plan['geometry'], resources)
            scanned = scan(path, plan['geometry'])
            if time.monotonic() - started > plan['limits']['child_timeout_seconds'] or time.monotonic() > deadline:
                raise RuntimeError('analysis/overall deadline exceeded')
            if measured['parseval_relative_closure_error'] is None or measured['parseval_relative_closure_error'] > 1e-10:
                raise RuntimeError('Parseval closure failed')
            rows[name] = {'id': name, 'path': str(path), 'sha256': trace.digest(path),
                          **measured, **scanned, **trace.compare(measured, source, mid)}
        check_pins(pins)
        check_latches(HERE, run, run.parent)
        summary = {'schema': plan['schema'], 'completed_utc': trace.utc(), 'rows': rows,
                   'source_metrics': source, 'mid_metrics': mid,
                   'raw_source_scan': scan(ROOT / plan['raw_decoded_source'], plan['geometry']),
                   'interaction_with_raw_identity': factorial(rows, 'identity'),
                   'interaction_with_guard_only': factorial(rows, 'guard'),
                   'combined_matches_cached_wav_exactly': True,
                   'maximum_native_hwm_mib': max(row['maximum_hwm_mib'] for row in executions),
                   'analysis_hwm_mib': memory_receipt(512)['observed_hwm_mib'],
                   'native_failure_remains_binding': True, 'speed_experiment': False}
        save(HERE / 'summary.json', summary)
        with (HERE / 'report.md').open('x') as file:
            file.write(report(summary))
        save(HERE / 'COMPLETE.json', {'completed_utc': trace.utc(), 'summary_sha256': trace.digest(HERE / 'summary.json'),
             'report_sha256': trace.digest(HERE / 'report.md'), 'manifest_sha256': trace.digest(HERE / 'manifest.json'),
             'pins_reverified': True, 'run_root': str(run), 'combined_matches_cached_wav_exactly': True})
        print(json.dumps({'interaction': summary['interaction_with_raw_identity'],
                          'maximum_native_hwm_mib': summary['maximum_native_hwm_mib']}))
    except BaseException as error:
        if not (HERE / 'STOP.json').exists():
            save(HERE / 'STOP.json', {'stopped_utc': trace.utc(), 'reason': str(error),
                 'error_type': type(error).__name__, 'completed_arms': executions})
        raise


if __name__ == '__main__':
    main()
