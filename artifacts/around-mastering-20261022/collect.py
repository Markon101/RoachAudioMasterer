#!/usr/bin/env python3
"""Collect numeric receipts only; keep song samples and waveform plots local."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def main():
    plan = read(HERE / 'plan_session2.json')
    run = ROOT / plan['run_root']
    assert not any((p / name).exists() for p in (HERE, run)
                   for name in ('HALT', 'HALTED.json'))
    complete = read(run / 'COMPLETE.json')
    assert complete['protocol_sha256'] == digest(run / 'protocol.json')
    for key in ('binary', 'model', 'low_model'):
        assert digest(ROOT / plan[key]) == plan[key + '_sha256']
    registration = read(HERE / 'analysis_plan.json')
    assert digest(ROOT / 'scripts/analyze_master_pair.py') == registration['analyzer_sha256']
    trace_plan = read(HERE / 'v2-stage-trace/plan_corrected2.json')
    for pins in (trace_plan['pins'], trace_plan['external_provenance_pins']):
        for path, expected in pins.items():
            assert digest(ROOT / path) == expected, path
    trace = read(HERE / 'v2-stage-trace/results-corrected2/summary.json')
    assert trace['plan_sha256'] == digest(HERE / 'v2-stage-trace/plan_corrected2.json')
    previous = 0.0
    for row in trace['rows']:
        assert abs(row['signed_shape_change_from_source_db'] - previous
                   - row['signed_increment_from_previous_boundary_db']) < 1e-10
        previous = row['signed_shape_change_from_source_db']
    meters = read(HERE / 'independent_meters.json')
    rows = []
    for track in plan['tracks']:
        name = track['id']
        receipt = read(run / (name + '-receipt.json'))
        assert digest(track['path']) == track['sha256'] == receipt['source_sha256']
        for filename, key in (('mastered.wav', 'mastered_sha256'),
                              ('mastered.flac', 'flac_sha256'),
                              ('summary.json', 'summary_sha256')):
            assert digest(run / name / filename) == receipt[key]
        analysis_path = ROOT / 'runs/around-analysis-20261022' / name.removeprefix('around-') / 'summary.json'
        analysis = read(analysis_path)
        for path, expected in analysis['provenance']['sha256'].items():
            assert digest(path) == expected, path
        master = read(run / name / 'master.json')
        rows.append({'id': name, 'geometry': analysis['geometry'],
                     'render': receipt, 'execution': read(run / (name + '-execution.json')),
                     'analysis': {k: analysis[k] for k in ('source', 'master', 'paired', 'spectrum', 'memory', 'provenance')},
                     'analysis_summary_sha256': digest(analysis_path),
                     'finalizer_glue': master['glue'], 'finalizer_limiter': master['limiter']})
    result = {'schema': 'around-full-master-analysis-v1', 'entry_commit': plan['source_parent'],
              'native_source_commit': plan['native_source_commit'],
              'plan_sha256': digest(HERE / 'plan_session2.json'), 'tracks': rows,
              'independent_meters': meters,
              'stage_trace_summary_sha256': digest(HERE / 'v2-stage-trace/results-corrected2/summary.json'),
              'verification': {'focused_python_tests': 23, 'all_content_pins_rechecked': True,
                               'trace_increment_arithmetic_rechecked': True},
              'interpretation': {'implementation_evidence_tier': 2, 'measurements_tier': 3,
                                 'subjective_preference_tier': 7, 'speed_experiment': False,
                                 'different_sources_not_treatment_replicates': True,
                                 'production_promoted': False, 'v2_delivery_is_diagnostic': True}}
    target = HERE / 'results.json'
    with target.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'results': str(target), 'all_pins_verified': True}))


if __name__ == '__main__':
    main()
