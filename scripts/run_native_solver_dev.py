#!/usr/bin/env python3
"""Bounded, hash-pinned procedural CPU/OpenCL audit; fresh foreground required."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from run_mid_profile import digest, memory, sensors


def run(plan_path, foreground_confirmed):
    if not foreground_confirmed:
        raise ValueError('fresh explicit owner foreground confirmation required')
    plan = json.loads(Path(plan_path).read_text())
    for key in ('binary', 'model'):
        if digest(plan[key]) != plan[key + '_sha256']:
            raise ValueError(f'{key} changed since planning')
    for path, expected in plan['source_sha256'].items():
        if digest(path) != expected:
            raise ValueError(f'source changed since planning: {path}')
    root = Path(plan['out'])
    if root.exists():
        raise FileExistsError('use a fresh audit directory')
    if shutil.disk_usage(root.parent).free < 512 * 1024 * 1024:
        raise ValueError('less than 512 MiB free storage')
    # This executable has one fixed diagnostic session, not an arbitrary job API.
    if plan['seed'] != 860000 or plan['count'] != 12:
        raise ValueError('this session requires the frozen 12-scene development set')
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(('ROACH_PROFILE', 'ROACH_MID_', 'ROACH_FOREGROUND'))}
    env['OCL_ICD_ASSUME_ICD_EXTENSION'] = '1'
    root.mkdir()
    receipts = []
    for backend in ('cpu', 'opencl'):
        out = root / backend
        command = [plan['binary'], 'mid-solver-audit', '--model', plan['model'],
                   '--seed', '860000', '--count', '12', '--backend', backend,
                   '--foreground-confirmed', '--out', str(out)]
        before = sensors()
        peak_kib = 0
        stopped = None
        started = time.monotonic()
        with (root / f'{backend}.log').open('x') as log:
            child = subprocess.Popen(command, env=env, stdout=log, stderr=log)
            while child.poll() is None:
                snapshot = memory(child.pid)
                peak_kib = max(peak_kib, snapshot.get('VmRSS', 0),
                               snapshot.get('VmHWM', 0))
                if peak_kib > 384 * 1024:
                    stopped = 'observed process RSS/HWM exceeded 384 MiB'
                elif time.monotonic() - started > 180:
                    stopped = 'process exceeded 180 seconds'
                if stopped:
                    child.kill()
                    child.wait()
                    break
                time.sleep(0.1)
        receipt = {'backend_requested': backend, 'command': command,
                   'process_wall_seconds': time.monotonic() - started,
                   'exit_code': child.returncode, 'stopped': stopped,
                   'observed_peak_rss_hwm_kib': peak_kib,
                   'sensors_before_raw': before, 'sensors_after_raw': sensors()}
        receipts.append(receipt)
        (root / 'runner_receipt.json').write_text(json.dumps(
            {'schema': 'native-mid-dev-runner-v1', 'plan': plan,
             'jobs': receipts}, indent=2) + '\n')
        if stopped or child.returncode != 0:
            raise RuntimeError(f'{backend} failed; partial evidence retained')
        complete = json.loads((out / 'completion.json').read_text())
        if (complete['actual_case_rows'] != 96 or
                complete['actual_control_rows'] != 48):
            raise RuntimeError('incomplete audit row counts')
        rows = [json.loads(line) for line in (out / 'cases.jsonl').read_text().splitlines()]
        cases = [r for r in rows if r['schema'] == 'native-mid-solver-case-v1']
        controls = [r for r in rows if r['schema'] == 'native-mid-solver-control-v1']
        if len(cases) != 96 or len(controls) != 48:
            raise RuntimeError('JSONL row counts disagree with completion')
        for row in cases:
            if row['result']['status'] == 'ok':
                if row['profile']['nfe'] != row['expected_nfe']:
                    raise RuntimeError('actual neural field count differs from plan')
                samples = row['result']['samples']
                if (out / samples['path']).stat().st_size != samples['bytes']:
                    raise RuntimeError('canonical sample artifact size mismatch')
        print(f'{backend}: complete; solver_failures={complete["solver_failures"]}; '
              f'observed_peak_mib={peak_kib / 1024:.1f}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--foreground-confirmed', action='store_true', required=True)
    args = parser.parse_args()
    run(args.plan, args.foreground_confirmed)
