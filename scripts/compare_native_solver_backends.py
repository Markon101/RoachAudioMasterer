#!/usr/bin/env python3
"""Compare canonical native-audit procedural samples without running inference."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load(directory):
    done = json.loads((directory / 'completion.json').read_text())
    if done['status'] != 'complete' or done['solver_failures']:
        raise ValueError('both audits must be complete without solver failures')
    rows = [json.loads(line) for line in (directory / 'cases.jsonl').read_text().splitlines()]
    cases = [r for r in rows if r['schema'] == 'native-mid-solver-case-v1']
    paired = {(r['seed'], r['condition'], r['solver']): r for r in cases}
    if len(cases) != done['actual_case_rows'] or len(paired) != len(cases):
        raise ValueError('case count mismatch or duplicate case identities')
    return paired


def samples(directory, row):
    record = row['result']['samples']
    path = directory / record['path']
    raw = path.read_bytes()
    if len(raw) != record['bytes'] or len(raw) != 4 * record['channels'] * record['samples_per_channel']:
        raise ValueError('sample shape/size mismatch')
    fingerprint = 0xcbf29ce484222325
    for byte in raw:
        fingerprint = ((fingerprint ^ byte) * 0x100000001b3) & 0xffffffffffffffff
    if f'noncryptographic-fnv1a64-f32le:{fingerprint:016x}' != record['waveform_fingerprint']:
        raise ValueError('samples disagree with recorded fingerprint')
    return record, raw


def compare(root):
    cpu, gpu = root / 'cpu', root / 'opencl'
    a, b = load(cpu), load(gpu)
    if a.keys() != b.keys() or not a:
        raise ValueError('CPU/OpenCL paired cases differ or are empty')
    results = []
    for key in sorted(a):
        ca, x = samples(cpu, a[key])
        ga, y = samples(gpu, b[key])
        if any(ca[k] != ga[k] for k in ('layout', 'channels', 'samples_per_channel', 'rate', 'bytes')):
            raise ValueError('CPU/OpenCL sample schemas differ')
        maximum, squared, count = 0.0, 0.0, 0
        for (u,), (v,) in zip(struct.iter_unpack('<f', x), struct.iter_unpack('<f', y)):
            if not math.isfinite(u) or not math.isfinite(v):
                raise ValueError('nonfinite exported sample')
            delta = float(u) - float(v)
            maximum = max(maximum, abs(delta))
            squared += delta * delta
            count += 1
        results.append({'seed': key[0], 'condition': key[1], 'solver': key[2],
                        'samples': count, 'max_abs': maximum, 'rms': math.sqrt(squared / count),
                        'cpu_sha256': hashlib.sha256(x).hexdigest(),
                        'opencl_sha256': hashlib.sha256(y).hexdigest()})
    return {'schema': 'native-mid-backend-sample-comparison-v1',
            'run': str(root), 'input_jsonl_sha256': {d.name: sha256(d / 'cases.jsonl') for d in (cpu, gpu)},
            'paired_cases': len(results), 'max_abs': max(r['max_abs'] for r in results),
            'max_case_rms': max(r['rms'] for r in results),
            'byte_identical_cases': sum(r['cpu_sha256'] == r['opencl_sha256'] for r in results),
            'qualification': 'Direct procedural sample differences; not a subjective or WOLA-quality result.',
            'cases': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    with args.out.open('x') as target:
        json.dump(compare(args.run), target, indent=2, allow_nan=False)
        target.write('\n')
