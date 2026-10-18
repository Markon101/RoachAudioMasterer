#!/usr/bin/env python3
"""Run an explicitly authorized, hash-pinned sequential Mid-CFM profile plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


DEFAULT_HOST_BATCH_ROWS = 8192
GPU_DEVICE_CAPACITY_ROWS = 8192
MAX_HOST_BATCH_ROWS = 65536


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def memory(pid):
    try:
        rows = Path(f'/proc/{pid}/status').read_text().splitlines()
        return {name: int(next(r.split()[1] for r in rows if r.startswith(name+':')))
                for name in ['VmRSS', 'VmHWM']}
    except (OSError, StopIteration):
        return {}


def sensors():
    result = {}
    for p in Path('/sys/class/thermal').glob('thermal_zone*'):
        try:
            kind = (p/'type').read_text().strip()
            if any(x in kind.lower() for x in ['cpu', 'gpu', 'soc', 'battery']):
                result[f'{p.name}:{kind}'] = (p/'temp').read_text().strip()
        except OSError:
            pass
    for path in ['/sys/class/kgsl/kgsl-3d0/gpuclk',
                 '/sys/class/devfreq/3d00000.qcom,kgsl-3d0/cur_freq']:
        try:
            result[path] = Path(path).read_text().strip()
        except OSError:
            pass
    return result


def _bounded_int(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')
    return value


def preflight(plan_path):
    """Validate every job before creating output directories or launching work."""
    plan = json.loads(Path(plan_path).read_text())
    for key in ['binary', 'model']:
        if digest(plan[key]) != plan[key+'_sha256']:
            raise ValueError(f'{key} changed since plan creation')
    jobs = plan['jobs'] + ([plan['cpu_match']] if plan.get('cpu_match') else [])
    if not 1 <= len(jobs) <= 16:
        raise ValueError('plan requires 1..16 sequential jobs')

    prepared = []
    seen_outputs = set()
    for job in jobs:
        backend = job.get('backend')
        if backend not in ('cpu', 'opencl'):
            raise ValueError('backend must be cpu or opencl')
        rows = _bounded_int(job.get('host_batch_rows', DEFAULT_HOST_BATCH_ROWS),
                            'host_batch_rows', DEFAULT_HOST_BATCH_ROWS, MAX_HOST_BATCH_ROWS)
        if rows % DEFAULT_HOST_BATCH_ROWS:
            raise ValueError('host_batch_rows must be an exact multiple of 8192')
        cache_mib = _bounded_int(job.get('cache_mib', 64), 'cache_mib', 1, 384)
        build = job.get('cache_build', 'serial')
        if build not in ('serial', 'parallel'):
            raise ValueError('cache_build must be serial or parallel')
        rss_limit_mib = _bounded_int(job.get('rss_limit_mib', 1536),
                                     'rss_limit_mib', 1, 1536)
        out = Path(job['out'])
        log = out.parent/(out.name+'.log')
        output_key = out.resolve()
        if output_key in seen_outputs:
            raise ValueError('duplicate output path in plan')
        seen_outputs.add(output_key)
        if out.exists() or log.exists():
            raise FileExistsError(f'existing output or log; refusing run: {out} / {log}')
        if digest(job['input']) != job['sha256']:
            raise ValueError('changed input; refusing run')
        prepared.append({
            'job': job, 'backend': backend, 'host_batch_rows': rows,
            'cache_mib': cache_mib, 'cache_build': build,
            'rss_limit_mib': rss_limit_mib, 'rss_limit_kib': rss_limit_mib*1024,
            'out': out, 'log': log,
        })
    return plan, prepared


def _profile_memory_kib(rows, name):
    """Return the largest available per-chunk memory_after observation."""
    samples = []
    for row in rows:
        refinement = row.get('refinement')
        if not isinstance(refinement, dict):
            continue
        after = refinement.get('memory_after')
        if not isinstance(after, dict):
            continue
        value = after.get(name)
        if type(value) is int and value >= 0:
            samples.append(value)
    return max(samples, default=0)


def run(plan_path, foreground_confirmed):
    if not foreground_confirmed:
        raise ValueError('explicit owner foreground confirmation is required')
    plan, jobs = preflight(plan_path)
    env_base = os.environ.copy()
    env_base.update(ROACH_PROFILE_MID='1', ROACH_FOREGROUND_CONFIRMED='1',
                    OCL_ICD_ASSUME_ICD_EXTENSION='1')
    receipts = []
    for index, config in enumerate(jobs):
        job = config['job']
        out, log = config['out'], config['log']
        out.parent.mkdir(parents=True, exist_ok=True)
        command = [plan['binary'], 'rich-mid-restore', '--model', plan['model'],
                   '--input', job['input'], '--backend', config['backend'],
                   '--mid-cutoff', str(plan['cutoff']), '--mid-ceiling', str(plan['ceiling']),
                   '--steps', '8', '--strength', '1', '--chunk-seconds', '10',
                   '--overlap-seconds', '2', '--out', str(out)]
        env = env_base.copy()
        env.update(ROACH_MID_CACHE_MIB=str(config['cache_mib']),
                   ROACH_MID_CACHE_BUILD=config['cache_build'],
                   ROACH_MID_BATCH_ROWS=str(config['host_batch_rows']))
        start_sensors = sensors()
        start = time.monotonic()
        peak_rss, peak_hwm = 0, 0
        with log.open('x') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=env)
            affinity = sorted(os.sched_getaffinity(process.pid)) if hasattr(os, 'sched_getaffinity') else None
            try:
                while process.poll() is None:
                    counters = memory(process.pid)
                    peak_rss = max(peak_rss, counters.get('VmRSS', 0))
                    peak_hwm = max(peak_hwm, counters.get('VmHWM', 0))
                    if time.monotonic()-start > 180 or max(peak_rss, peak_hwm) > config['rss_limit_kib']:
                        raise RuntimeError('bounded run exceeded 180 seconds or declared RSS/HWM limit')
                    time.sleep(.25)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
        elapsed = time.monotonic()-start
        if process.returncode:
            raise RuntimeError(f'job failed; inspect {log}')
        rows = [json.loads(line) for line in (out/'refinement_profile.jsonl').read_text().splitlines()]
        if not rows or any(row.get('disk_chunk_cache_hit') for row in rows):
            raise ValueError('missing refinement or reused audio cache invalidates timing')
        refinements = [row.get('refinement') for row in rows]
        if any(not isinstance(refinement, dict) for refinement in refinements):
            raise ValueError('profile is missing an actual refinement record')
        if any(refinement.get('cache_cap_bytes') != config['cache_mib']*1024*1024
               for refinement in refinements):
            raise ValueError('requested cache budget was not applied')
        for refinement in refinements:
            if ('host_batch_rows' not in refinement and
                    config['host_batch_rows'] != DEFAULT_HOST_BATCH_ROWS):
                raise ValueError('profile omits host_batch_rows for a non-default request')
            actual_rows = refinement.get('host_batch_rows', DEFAULT_HOST_BATCH_ROWS)
            if (type(actual_rows) is not int or
                    actual_rows != config['host_batch_rows']):
                raise ValueError('actual refinement host_batch_rows does not match request')
        profile_rss = _profile_memory_kib(rows, 'vm_rss_kib')
        profile_hwm = _profile_memory_kib(rows, 'process_high_water_kib')
        peak_rss = max(peak_rss, profile_rss)
        peak_hwm = max(peak_hwm, profile_hwm)
        if max(peak_rss, peak_hwm) > config['rss_limit_kib']:
            raise RuntimeError('final profile RSS/HWM exceeded declared limit')
        actual_host_call_count = sum(refinement.get('predictor_batch_count', 0)
                                     for refinement in refinements)
        receipt = {'job_index': index, 'command': command, 'input_sha256': job['sha256'],
                   'process_wall_seconds': elapsed, 'wall_poll_resolution_seconds': .25,
                   'sampled_peak_rss_kib': peak_rss, 'sampled_process_hwm_kib': peak_hwm,
                   'cpu_affinity': affinity, 'sensors_start_raw': start_sensors,
                   'sensors_end_raw': sensors(), 'chunks': rows,
                   'binary_sha256': plan['binary_sha256'], 'model_sha256': plan['model_sha256'],
                   'host_batch_rows': config['host_batch_rows'],
                   'actual_host_call_count': actual_host_call_count,
                   'gpu_device_capacity_rows': (GPU_DEVICE_CAPACITY_ROWS
                                                if config['backend'] == 'opencl' else None)}
        receipt.update(cache_mib=config['cache_mib'], cache_build=config['cache_build'],
                       rss_limit_kib=config['rss_limit_kib'])
        (out/'profile_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        receipts.append(receipt)
        print(f'Completed profile job {index+1}/{len(jobs)} ({config["backend"]}).', flush=True)
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--foreground-confirmed', action='store_true')
    args = parser.parse_args()
    run(args.plan, args.foreground_confirmed)
