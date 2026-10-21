#!/usr/bin/env python3
"""Record this free-text submission; never coerce it into strict reveal categories.

Run from the repo root with: python artifacts/listening-results-20261021/
record_submission.py --responses PATH --delivered LISTENER_FOLDER
All outputs are create-only. Organizer keys and audio remain local.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from blind_listening import FIELDS, CHOICES, verify_audio


def sha(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--responses', type=Path, required=True)
    parser.add_argument('--delivered', type=Path, required=True)
    args = parser.parse_args()
    organizer = ROOT / 'runs/listening-coherence-20261020/listening-private-v1/organizer'
    raw = args.responses.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode('utf-8-sig'), newline=''))
    if reader.fieldnames != FIELDS:
        raise ValueError('unexpected response columns')
    rows = list(reader)
    key = json.loads((organizer / 'key.private.json').read_text())
    ids = {t['trial'] for t in key['trials']}
    if len(rows) != len(ids) or {r['trial'] for r in rows} != ids:
        raise ValueError('missing/duplicate/unexpected trial')
    if any(set(r) != set(FIELDS) or any(v is None for v in r.values())
           or r['preference'] not in CHOICES for r in rows):
        raise ValueError('malformed response or unsupported preference')
    session = json.loads((organizer / 'session.private.json').read_text())
    verified = {
        'source': verify_audio(Path(session['listener']), key),
        'delivered': verify_audio(args.delivered, key),
    }
    by_id = {t['trial']: t for t in key['trials']}
    trials = []
    for row in rows:
        trial = by_id[row['trial']]
        same = trial['members']['A']['sha256'] == trial['members']['B']['sha256']
        if same != trial['identity']:
            raise ValueError('identity label disagrees with exact audio hashes')
        trials.append({
            'response_verbatim': row,
            'identity_audio': same,
            'duration_seconds': trial['excerpt']['frames'] / 48000,
            'start_sample': trial['excerpt']['start_sample'],
            'members': {slot: {'label': member['variant']['label'],
                               'audio_sha256': member['sha256']}
                        for slot, member in trial['members'].items()},
            'selected_source': (trial['members'][row['preference']]['variant']['label']
                                if row['preference'] in {'A', 'B'} else None),
        })
    if args.responses.read_bytes() != raw:
        raise ValueError('submission changed during verification')
    archive = ROOT / 'runs/listening-results-20261021/submission-v1'
    archive.mkdir(parents=True, exist_ok=False)
    with (archive / 'responses-original.csv').open('xb') as out:
        out.write(raw)
    receipt = {
        'schema': 'roach-owner-listening-free-text-v1',
        'recorded_utc': datetime.now(timezone.utc).isoformat(),
        'submitted_filename': args.responses.name,
        'raw_sha256': sha(raw),
        'raw_archive': str((archive / 'responses-original.csv').relative_to(ROOT)),
        'verification': verified,
        'strict_categorical_reveal': 'not applicable: prose aspects, percentage confidence, free-text fatigue',
        'normalization': 'none; every response value preserved verbatim',
        'owner_clarification': 'Owner explicitly confirmed listening to both A and B for every trial.',
        'trials': trials,
        'identity_trials_with_directional_preference': sum(t['identity_audio'] and
            t['response_verbatim']['preference'] in {'A', 'B'} for t in trials),
        'interpretation': 'Control selected on both nonidentity comparisons; directional choices on both identical pairs confound discrimination. Neither version superiority established.',
        'causal_hum_attribution': 'unestablished; no per-member categorical attribution supplied',
        'epistemic': {'response_and_hash_receipt': 'Tier 2 implementation checks and recorded observations',
                      'comparative_superiority': 'Tier 7 unestablished/confounded'},
        'production_promoted': False,
    }
    with (Path(__file__).parent / 'receipt.json').open('x') as out:
        json.dump(receipt, out, indent=2, allow_nan=False)
        out.write('\n')
    print(json.dumps({'raw_sha256': sha(raw), 'trials': len(trials),
                      'identity_directional_choices': receipt['identity_trials_with_directional_preference']}))


if __name__ == '__main__':
    main()
