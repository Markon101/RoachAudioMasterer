#!/usr/bin/env python3
"""Package only this gated pair; keep labels, seed and metrics private."""
import json
from pathlib import Path
import secrets
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
import blind_listening as blind


def verify_qualification(run):
    receipt = json.loads((run / 'COMPLETE.json').read_text())
    if receipt['eligible_for_listening'] is not True:
        raise RuntimeError('full-song gates incomplete')
    if blind.sha256(run / 'protocol.json') != receipt['protocol_sha256']:
        raise RuntimeError('qualified protocol changed')
    protocol = json.loads((run / 'protocol.json').read_text())
    for path, expected in protocol['pins'].items():
        if blind.sha256(Path(path)) != expected:
            raise RuntimeError(f'render qualification pin changed: {path}')
    return receipt, protocol


def main():
    base = ROOT / 'runs/listening-singularity-20261021'
    run = base / 'fullsong-v1'
    for folder in [base, run]:
        if any((folder / name).exists() for name in ['HALT', 'HALTED.json']):
            raise RuntimeError('latched render session')
    receipt, protocol = verify_qualification(run)
    plan = protocol['plan']
    rows = {r['name']: r for r in receipt['rows']}
    for name in ['control', 'candidate']:
        if blind.sha256(run / name / 'mastered.wav') != rows[name]['mastered_sha256']:
            raise RuntimeError('qualified master changed')
    private = base / 'listening-private-v1'
    private.mkdir(mode=0o700)
    variants = []
    for name in ['control', 'candidate']:
        variants.append({'label': name, 'path': str(run / name / 'mastered.wav'),
            'provenance': {'run_record': str(run / 'COMPLETE.json'),
                'git_commit': plan['native_source_commit'],
                'source_sha256': receipt['source_sha256'], 'target_lufs': -13.0,
                'binary_sha256': plan['binary_sha256'],
                'qualification': 'Unchanged qualified native binary, same input/models/settings; candidate only --mid-coherence. Exploratory listening, no promotion. Prior mechanism knowledge; assignments concealed.'}})
    manifest = {'version': 1, 'mode': 'equal_target_architecture',
        'assignment_policy': 'counterbalanced_pairs', 'seed': secrets.randbits(63),
        'target_lufs': -13.0, 'tracks': [{'id': 'track1', 'variants': variants,
            'excerpts': [
                {'id': 'fixed_short', 'kind': 'diagnostic', 'start_sample': 2304000,
                 'frames': 1440000},
                {'id': 'full_song', 'kind': 'long', 'start_sample': 0,
                 'frames': plan['source']['frames']}]}]}
    manifest_path = private / 'manifest.json'
    with manifest_path.open('x') as f:
        json.dump(manifest, f, indent=2)
        f.write('\n')
    manifest_path.chmod(0o600)
    listener, organizer = base / 'listen/session1', private / 'organizer'
    blind.prepare(manifest_path, listener, organizer)
    with (listener / 'notes.txt').open('x') as f:
        f.write('Optional notes. Identify the trial and A/B where relevant.\n'
                'Keep categorical choices in responses.csv.\n\n'
                'Playback setup and comfortable level check:\n\n'
                'T0001:\n\nT0002:\n\nT0003:\n\nT0004:\n')
    counts = blind.verify(organizer)
    key = json.loads((organizer / 'key.private.json').read_text())
    pairs = [t for t in key['trials'] if not t['identity']]
    if (len(pairs) != 2 or
        sorted(t['members']['A']['variant']['label'] for t in pairs) != ['candidate', 'control']):
        raise RuntimeError('source-position balance failed')
    for t in key['trials']:
        # Preserve the native -1dBTP gate for encoded copies, beyond clipping alone.
        if any(m['encoded']['true_peak_dbfs'] > -.99 for m in t['members'].values()):
            raise RuntimeError('encoded peak failed')
    for folder in [base, run]:
        if any((folder / name).exists() for name in ['HALT', 'HALTED.json']):
            raise RuntimeError('session halted during packaging')
    if verify_qualification(run) != (receipt, protocol):
        raise RuntimeError('qualification changed during packaging')
    files = {str(p.relative_to(listener)): blind.sha256(p)
             for p in sorted(listener.rglob('*')) if p.is_file()}
    ready = {'schema': 'cross-song-listening-ready-v1', 'listener_dir': str(listener),
        'counts': counts, 'files_sha256': files,
        'fullsong_complete_sha256': blind.sha256(run / 'COMPLETE.json'),
        'candidate_count': 1, 'source_positions_counterbalanced': True,
        'comfortable_playback_check_requested': True,
        'encoded_loudness_and_peak_checks_passed': True,
        'owner_judgments_received': False, 'production_promoted': False,
        'package_script_sha256': blind.sha256(Path(__file__)),
        'harness_sha256': blind.sha256(ROOT / 'scripts/blind_listening.py')}
    with (base / 'LISTENING_READY.json').open('x') as f:
        json.dump(ready, f, indent=2)
        f.write('\n')
    print(json.dumps({'ready': True, **counts}))


if __name__ == '__main__':
    if (ROOT / 'runs/listening-singularity-20261021/LISTENING_READY.json').exists():
        raise SystemExit('Existing completed package is immutable; refusing repeat invocation.')
    try:
        main()
    except BaseException as error:
        base = ROOT / 'runs/listening-singularity-20261021'
        if base.exists():
            marker = base / 'HALTED.json'
            if not marker.exists():
                with marker.open('x') as f:
                    json.dump({'stage': 'packaging', 'reason': str(error)}, f, indent=2)
                    f.write('\n')
            # Retain all partial audio/key evidence, but never signal listener readiness.
            (base / 'listen/session1/READY').unlink(missing_ok=True)
        raise
