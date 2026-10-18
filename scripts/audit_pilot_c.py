#!/usr/bin/env python3
"""Audit frozen toy Pilot C without rerunning/overwriting its historical report.

Requires NumPy, as does the original toy. Counts function evaluations, not time.
Run from the repository root; stdout is an inspectable JSON receipt.
"""
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np


def audit():
    source = Path('analysis/phase3/pilot_c_meanflow_distillation.py')
    original = Path('analysis/phase3/pilot_c_results.json')
    spec = importlib.util.spec_from_file_location('frozen_pilot_c', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    velocity = module.synthetic_teacher_velocity
    calls = []

    def counted(x, t, c):
        calls.append(float(t))
        return velocity(x, t, c)

    module.synthetic_teacher_velocity = counted
    x = np.linspace(-.5, .5, 64)
    target = np.cos(np.arange(64) * .17)
    observed = {}
    for name, expected in [('solve_8step_euler', 8), ('solve_4step_euler', 4),
                           ('solve_2step_meanflow', 4), ('solve_1step_corrector', 1)]:
        calls.clear()
        result = getattr(module, name)(x, target)
        assert len(calls) == expected, (name, calls)
        observed[name] = {'actual_nfe': len(calls), 'times': calls.copy()}
        if name == 'solve_1step_corrector':
            error = float(np.max(np.abs(result - target)))
            assert error < 1e-14
            observed[name]['max_abs_target_error'] = error
    frozen = json.loads(original.read_text())
    row = frozen['results']['student_2step_meanflow']
    assert row['passed'] is False and row['delta_lsd_db'] > .5
    assert frozen['falsification_status'].startswith('SURVIVED')
    return {
        'schema': 'roach-pilot-c-scope-audit-v1',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'historical_results_sha256': hashlib.sha256(original.read_bytes()).hexdigest(),
        'observed': observed,
        'historical_midpoint_gate': 'FAILED; +0.6798029827083267 dB exceeds +0.5 dB',
        'historical_status_contradicts_gate': True,
        'scope': 'Handwritten 64-D toy field; no native model, training, audio, GPU or latency measurement.',
        'interpretation': 'Toy evidence cannot promote or reject a solver for frozen rich-mid-v3 or SFHT.',
        'numpy_version': np.__version__,
    }


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2, allow_nan=False))
