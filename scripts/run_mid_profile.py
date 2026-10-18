#!/usr/bin/env python3
"""Run an explicitly authorized, hash-pinned sequential Mid-CFM profile plan."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


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


def run(plan_path, foreground_confirmed):
    if not foreground_confirmed:
        raise ValueError('explicit owner foreground confirmation is required')
    plan = json.loads(Path(plan_path).read_text())
    for key in ['binary', 'model']:
        if digest(plan[key]) != plan[key+'_sha256']:
            raise ValueError(f'{key} changed since plan creation')
    jobs = plan['jobs'] + ([plan['cpu_match']] if plan.get('cpu_match') else [])
    if not 1 <= len(jobs) <= 16:
        raise ValueError('plan requires 1..16 sequential jobs')
    for job in jobs:
        if Path(job['out']).exists() or digest(job['input']) != job['sha256']:
            raise ValueError('existing output or changed input; refusing run')
    env = os.environ.copy()
    env.update(ROACH_PROFILE_MID='1', ROACH_FOREGROUND_CONFIRMED='1',
               OCL_ICD_ASSUME_ICD_EXTENSION='1')
    receipts = []
    for index, job in enumerate(jobs):
        out = Path(job['out'])
        out.parent.mkdir(parents=True, exist_ok=True)
        log = out.parent/(out.name+'.log')
        if log.exists():
            raise FileExistsError(log)
        command = [plan['binary'], 'rich-mid-restore', '--model', plan['model'],
                   '--input', job['input'], '--backend', job['backend'],
                   '--mid-cutoff', str(plan['cutoff']), '--mid-ceiling', str(plan['ceiling']),
                   '--steps', '8', '--strength', '1', '--chunk-seconds', '10',
                   '--overlap-seconds', '2', '--out', str(out)]
        cache_mib = int(job.get('cache_mib', 64))
        build = job.get('cache_build', 'serial')
        if not 1 <= cache_mib <= 384 or build not in ['serial', 'parallel']:
            raise ValueError('invalid research cache configuration')
        env.update(ROACH_MID_CACHE_MIB=str(cache_mib), ROACH_MID_CACHE_BUILD=build)
        rss_limit_kib = int(job.get('rss_limit_mib', 1536)) * 1024
        if not 1 <= rss_limit_kib <= 1536*1024:
            raise ValueError('RSS limit must be1..1536MiB')
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
                    if time.monotonic()-start > 180 or max(peak_rss, peak_hwm) > rss_limit_kib:
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
        if not rows or any(row['disk_chunk_cache_hit'] for row in rows):
            raise ValueError('missing refinement or reused audio cache invalidates timing')
        if any(row['refinement']['cache_cap_bytes'] != cache_mib*1024*1024 for row in rows):
            raise ValueError('requested cache budget was not applied')
        receipt = {'job_index': index, 'command': command, 'input_sha256': job['sha256'],
                   'process_wall_seconds': elapsed, 'wall_poll_resolution_seconds': .25,
                   'sampled_peak_rss_kib': peak_rss, 'sampled_process_hwm_kib': peak_hwm,
                   'cpu_affinity': affinity, 'sensors_start_raw': start_sensors,
                   'sensors_end_raw': sensors(), 'chunks': rows,
                   'binary_sha256': plan['binary_sha256'], 'model_sha256': plan['model_sha256']}
        receipt.update(cache_mib=cache_mib, cache_build=build, rss_limit_kib=rss_limit_kib)
        (out/'profile_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
        receipts.append(receipt)
        print(f'Completed profile job {index+1}/{len(jobs)} ({job["backend"]}).', flush=True)
    return receipts


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('plan', type=Path)
    parser.add_argument('--foreground-confirmed', action='store_true')
    args = parser.parse_args()
    run(args.plan, args.foreground_confirmed)
