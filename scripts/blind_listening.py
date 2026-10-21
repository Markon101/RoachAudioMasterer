#!/usr/bin/env python3
"""Local, fixed-excerpt, attenuation-only blind A/B packaging. Python stdlib only."""
import argparse
import csv
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
from fractions import Fraction

FIELDS = ['trial', 'transient', 'tonal', 'spatial', 'preference',
          'confidence', 'fatigue']
CHOICES = {'A', 'B', 'indistinguishable', 'abstain'}
ASSIGNMENT_POLICIES = ('independent', 'counterbalanced_pairs')
INSTRUCTIONS = """Before judging, check playback at a comfortable listening level.
If the files remain too quiet, stop and report that before recording judgments.
Then keep the same playback device, settings and volume throughout.
Disable playback enhancement, normalization and crossfade. Follow trial order.
Compare A.wav and B.wav freely, then enter one responses.csv row per trial.
For transient, tonal, spatial and preference choose A, B, indistinguishable,
or abstain. Consider onset definition, tonal character and spatial presentation
respectively; use your own musical judgment. 'Indistinguishable' means no reliable
audible difference; 'abstain' means you cannot judge this aspect. Preference is
your overall choice. Confidence: low, medium, high. Fatigue: none, mild, high, not_assessed.
Use not_assessed when fatigue cannot be judged, including short trials. Short
trials cannot establish prolonged fatigue; longer trials provide only a limited
session impression. Take breaks whenever needed. Use listening only; save file
comparisons and diagnostics until judgments are recorded. Do not change file names or trial identifiers.
Keep free-prose comments in a separate notes.txt, keyed by trial identifier;
keep responses.csv columns and categorical choices unchanged. Return the completed
CSV and any notes.txt to the organizer. Do not seek the answer key while judging.
"""
PROCEDURAL_DIAGNOSTIC_INSTRUCTIONS = """This is a short procedural diagnostic comparison, not a fatigue or long-session study.
Before judging, check playback at a comfortable listening level.
If the files remain too quiet, stop and report that before recording judgments.
Then keep the same playback device, settings and volume throughout.
Disable playback enhancement, normalization and crossfade. Follow trial order.
Compare A.wav and B.wav freely, then enter one responses.csv row per trial.
For transient, tonal, spatial and preference choose A, B, indistinguishable,
or abstain. Consider onset definition, tonal character and spatial presentation
respectively; use your own musical judgment. 'Indistinguishable' means no reliable
audible difference; 'abstain' means you cannot judge this aspect. Preference is
your overall choice. Confidence: low, medium, high. Leave fatigue as not_assessed;
these 8-to-48-second excerpts do not assess fatigue or support long-session claims.
Use listening only; save file comparisons and diagnostics until judgments are
recorded. Do not change file names or trial identifiers. Keep free-prose comments
in a separate notes.txt, keyed by trial identifier; keep responses.csv columns
and categorical choices unchanged. Return the completed CSV and any notes.txt
to the organizer. Do not seek the answer key while judging.
"""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha256(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def run(args):
    env = os.environ.copy()
    runtime = Path('/data/data/com.termux/files/usr/lib/libc++_shared.so')
    if runtime.exists():
        env['LD_PRELOAD'] = ' '.join(filter(None, [env.get('LD_PRELOAD'), str(runtime)]))
    result = subprocess.run(args, capture_output=True, text=True, env=env, check=True)
    return result.stdout, result.stderr


def probe(path, expected_pcm24=False):
    stdout, _ = run(['ffprobe', '-v', 'error', '-show_streams', '-of', 'json', str(path)])
    streams = [s for s in json.loads(stdout)['streams'] if s['codec_type'] == 'audio']
    require(len(streams) == 1, 'exactly one audio stream required')
    s = streams[0]
    if expected_pcm24:
        require(s.get('codec_name') == 'pcm_s24le' and s.get('sample_fmt') == 's32'
                and int(s.get('bits_per_sample', 0)) == 24,
                'encoded output is not native 24-bit PCM')
    rate = int(s['sample_rate'])
    frames = Fraction(s['duration_ts']) * Fraction(s['time_base']) * rate
    require(frames.denominator == 1, 'nonintegral sample duration')
    return {'rate': rate, 'channels': int(s['channels']), 'frames': int(frames)}


def trim_filter(excerpt):
    start = excerpt['start_sample']
    return f"atrim=start_sample={start}:end_sample={start + excerpt['frames']},asetpts=PTS-STARTPTS"


def measure(path, excerpt=None):
    filters = ([trim_filter(excerpt)] if excerpt else []) + ['ebur128=peak=true:framelog=verbose']
    _, stderr = run(['ffmpeg', '-nostdin', '-hide_banner', '-i', str(path),
                     '-map', '0:a:0', '-af', ','.join(filters), '-f', 'null', '-'])
    integrated = re.findall(r'\bI:\s*([-+\d.inf]+)\s*LUFS', stderr)
    peaks = re.findall(r'\bPeak:\s*([-+\d.inf]+)\s*dBFS', stderr)
    require(integrated and peaks, 'missing ebur128 summary')
    loudness, peak = float(integrated[-1]), float(peaks[-1])
    require(math.isfinite(loudness) and loudness > -69 and math.isfinite(peak),
            'silence or unmeasurable loudness; choose another fixed excerpt')
    return {'lufs': loudness, 'true_peak_dbfs': peak}


def encode(source, destination, excerpt, gain):
    require(math.isfinite(gain) and gain <= 0, 'amplification forbidden')
    run(['ffmpeg', '-nostdin', '-hide_banner', '-v', 'error', '-n', '-i', str(source),
         '-map', '0:a:0', '-af', trim_filter(excerpt) + f',volume={gain:.8f}dB',
         '-map_metadata', '-1', '-map_chapters', '-1', '-fflags', '+bitexact',
         '-flags:a', '+bitexact', '-c:a', 'pcm_s24le', str(destination)])


def matching_target(stats, ceiling):
    # One shared target per pair: never raise either member; retain peak headroom.
    return min([ceiling] + [s['lufs'] - max(0, s['true_peak_dbfs'] + 0.2) for s in stats])


def integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def validate(manifest, base):
    require(manifest.get('version') == 1, 'manifest version must be 1')
    mode = manifest.get('mode')
    require(mode in {'historical_playback', 'equal_target_architecture', 'procedural_diagnostic'}, 'choose one mode; pooling prohibited')
    require(integer(manifest.get('seed')), 'private integer seed required')
    require(manifest.get('assignment_policy', 'independent') in ASSIGNMENT_POLICIES,
            'unknown assignment_policy')
    require(isinstance(manifest.get('target_lufs'), (int, float)) and
            math.isfinite(manifest['target_lufs']), 'finite target_lufs ceiling required')
    require(isinstance(manifest.get('tracks'), list) and manifest['tracks'], 'tracks required')
    tracks, ids = [], set()
    for track in manifest['tracks']:
        require(isinstance(track.get('id'), str) and track['id'] not in ids, 'unique track id required')
        ids.add(track['id'])
        variants = track.get('variants', [])
        require(2 <= len(variants) <= 6, 'require 2..6 variants per track')
        labels, specs = set(), []
        for v in variants:
            require(isinstance(v.get('label'), str) and v['label'] not in labels, 'unique variant labels required')
            labels.add(v['label'])
            path = (base / v['path']).resolve()
            require(path.suffix.lower() in {'.wav', '.flac'} and path.is_file(), 'local WAV/FLAC required')
            v['path'] = str(path)
            require(isinstance(v.get('provenance'), dict) and v['provenance'], 'explicit provenance required')
            if mode == 'procedural_diagnostic':
                provenance = v['provenance']
                require(provenance.get('procedural') is True,
                        'procedural diagnostic requires procedural provenance true')
                require(isinstance(provenance.get('run_record'), str) and provenance['run_record'].strip(),
                        'procedural diagnostic requires a declared run_record')
                require(isinstance(provenance.get('git_commit'), str) and
                        re.fullmatch(r'[0-9a-f]{40}', provenance['git_commit']) is not None,
                        'procedural diagnostic requires a full git_commit declaration')
                require(isinstance(provenance.get('source_sha256'), str) and
                        re.fullmatch(r'[0-9a-f]{64}', provenance['source_sha256']) is not None,
                        'procedural diagnostic requires a source_sha256 declaration')
            v['sha256'] = sha256(path)
            specs.append(probe(path))
        require(len({(s['rate'], s['channels']) for s in specs}) == 1, 'rate/channel mismatch')
        rate = specs[0]['rate']
        trim = track.get('common_trim')
        if trim is not None:
            require(integer(trim.get('start_sample')) and trim['start_sample'] >= 0 and
                    integer(trim.get('frames')) and trim['frames'] > 0 and
                    all(trim['start_sample'] + trim['frames'] <= s['frames'] for s in specs), 'invalid common trim')
        else:
            require(len({s['frames'] for s in specs}) == 1, 'duration mismatch requires declared common trim')
        if manifest['mode'] == 'equal_target_architecture':
            provenances = [v['provenance'] for v in variants]
            require(all(re.fullmatch(r'[0-9a-f]{40}', p.get('git_commit', '')) and
                        re.fullmatch(r'[0-9a-f]{64}', p.get('source_sha256', '')) and
                        isinstance(p.get('target_lufs'), (int, float)) and math.isfinite(p['target_lufs'])
                        for p in provenances), 'architecture mode requires git_commit/source_sha256/target_lufs')
            require(len({p['source_sha256'] for p in provenances}) == 1 and
                    len({p['target_lufs'] for p in provenances}) == 1,
                    'architecture mode requires same original source and render target')
        if mode == 'procedural_diagnostic':
            provenances = [v['provenance'] for v in variants]
            require(len({p['source_sha256'] for p in provenances}) == 1,
                    'procedural diagnostic variants must declare the same procedural source hash')
        kinds, excerpt_ids = set(), set()
        for e in track.get('excerpts', []):
            require(isinstance(e.get('id'), str) and e['id'] not in excerpt_ids, 'unique excerpt id required')
            excerpt_ids.add(e['id'])
            require(integer(e.get('start_sample')) and e['start_sample'] >= 0 and
                    integer(e.get('frames')) and e['frames'] > 0, 'sample interval must be positive integers')
            if mode == 'procedural_diagnostic':
                require(e.get('kind') == 'diagnostic' and 8 * rate <= e['frames'] <= 48 * rate,
                        'procedural diagnostic requires diagnostic excerpts from 8 to 48 seconds')
            else:
                require((e.get('kind') == 'diagnostic' and e['frames'] == 30 * rate) or
                        (e.get('kind') == 'long' and e['frames'] >= 120 * rate), 'require 30s diagnostic or >=120s long')
            low, high = (trim['start_sample'], trim['start_sample'] + trim['frames']) if trim else (0, specs[0]['frames'])
            require(low <= e['start_sample'] and e['start_sample'] + e['frames'] <= high, 'excerpt outside common interval')
            kinds.add(e['kind'])
        if mode == 'procedural_diagnostic':
            require(kinds == {'diagnostic'}, 'procedural diagnostic accepts diagnostic excerpts only; no long or fatigue study')
        else:
            require(kinds == {'diagnostic', 'long'}, 'each track needs diagnostic and long excerpts')
        track['audio_specs'] = specs
        tracks.append(track)
    require(sum(len(t['excerpts']) * (math.comb(len(t['variants']), 2) + 1) for t in tracks) <= 64, 'session exceeds 64 trials; split manifest')
    return tracks


def assignments(tracks, seed, policy='independent'):
    """Optionally alternate each track/pair's orientation across its excerpts."""
    require(policy in ASSIGNMENT_POLICIES, 'unknown assignment_policy')
    rng, trials = random.Random(seed), []
    for track in tracks:
        # Scope orientations to each track and unordered pair. Leave the legacy
        # branch's random draw order untouched for existing private manifests.
        orientations = ({pair: rng.getrandbits(1) for pair in
                         itertools.combinations(range(len(track['variants'])), 2)}
                        if policy == 'counterbalanced_pairs' else {})
        for excerpt in track['excerpts']:
            pairs = list(itertools.combinations(range(len(track['variants'])), 2))
            identity = rng.randrange(len(track['variants']))
            for a, b in pairs + [(identity, identity)]:
                if policy == 'counterbalanced_pairs' and a != b:
                    reverse = orientations[(a, b)]
                    orientations[(a, b)] ^= 1
                else:
                    reverse = rng.getrandbits(1)
                if reverse:
                    a, b = b, a
                trials.append((track, excerpt, a, b))
    rng.shuffle(trials)
    return trials


def verify_audio(listener, key):
    """Check the listener package against the private key without revealing it."""
    listener = Path(listener)
    require((listener / 'READY').is_file(), 'session is not ready')
    trials = key.get('trials')
    require(isinstance(trials, list) and trials, 'private audio record is invalid')
    trial_ids = [trial.get('trial') for trial in trials if isinstance(trial, dict)]
    require(len(trial_ids) == len(trials) and all(isinstance(t, str) and t for t in trial_ids)
            and len(set(trial_ids)) == len(trial_ids), 'private audio record is invalid')
    for trial in trials:
        members = trial.get('members')
        require(isinstance(members, dict), 'private audio record is invalid')
        for slot in ('A', 'B'):
            member = members.get(slot)
            recorded_hash = member.get('sha256') if isinstance(member, dict) else None
            require(isinstance(recorded_hash, str) and
                    re.fullmatch(r'[0-9a-f]{64}', recorded_hash) is not None,
                    'private audio record is invalid')
            path = listener / trial['trial'] / f'{slot}.wav'
            require(path.is_file(), 'expected listening audio file is missing')
            try:
                digest = sha256(path)
            except OSError as error:
                raise ValueError('expected listening audio file is unavailable') from error
            require(digest == recorded_hash, 'listening audio changed after preparation (hash mismatch)')
    return {'trials': len(trials), 'audio_files': 2 * len(trials)}


def prepare(manifest_path, listener, organizer):
    manifest_path, listener, organizer = map(lambda p: Path(p).resolve(), [manifest_path, listener, organizer])
    require(listener != organizer and listener not in organizer.parents and organizer not in listener.parents,
            'organizer must be outside listener folder')
    require(not listener.exists() and not organizer.exists(), 'refusing existing output directory')
    manifest = json.loads(manifest_path.read_text())
    tracks = validate(manifest, manifest_path.parent)
    storage_root = listener.parent
    while not storage_root.exists():
        storage_root = storage_root.parent
    required_bytes = sum(2 * e['frames'] * t['audio_specs'][0]['channels'] * 3 *
                         (math.comb(len(t['variants']), 2) + 1) for t in tracks for e in t['excerpts'])
    require(shutil.disk_usage(storage_root).free >= required_bytes + 32 * 1024 ** 2,
            'insufficient free storage for uncompressed pairs and safety margin')
    organizer.mkdir(mode=0o700, parents=True)
    listener.mkdir(mode=0o700, parents=True)
    dump(organizer / 'manifest.private.json', manifest)
    key, cache = [], {}
    try:
        for number, (track, excerpt, a, b) in enumerate(assignments(
                tracks, manifest['seed'], manifest.get('assignment_policy', 'independent')), 1):
            trial_id = f'T{number:04d}'
            folder = listener / trial_id
            folder.mkdir()
            variants = [track['variants'][index] for index in [a, b]]
            stats = []
            for v in variants:
                cache_id = (v['path'], excerpt['start_sample'], excerpt['frames'])
                if cache_id not in cache:
                    cache[cache_id] = measure(Path(v['path']), excerpt)
                stats.append(cache[cache_id])
            target = matching_target(stats, manifest['target_lufs'])
            record = {'trial': trial_id, 'track': track['id'], 'excerpt': excerpt,
                      'identity': a == b, 'target_lufs': target, 'members': {}}
            outputs = []
            for slot, variant, stat in zip(['A', 'B'], variants, stats):
                destination = folder / f'{slot}.wav'
                gain = min(0., target - stat['lufs'])
                encode(Path(variant['path']), destination, excerpt, gain)
                spec, actual = probe(destination, expected_pcm24=True), measure(destination)
                require(spec == dict(track['audio_specs'][0], frames=excerpt['frames']), 'encoded sample interval mismatch')
                require(actual['true_peak_dbfs'] < 0, 'encoded output clipping/true-peak failure')
                outputs.append(actual['lufs'])
                record['members'][slot] = {'variant': variant, 'gain_db': gain,
                                          'input': stat, 'encoded': actual, 'sha256': sha256(destination)}
            require(abs(outputs[0] - outputs[1]) <= .15 and
                    all(abs(loudness - target) <= .15 for loudness in outputs), 'encoded loudness matching failed')
            if a == b:
                require(sha256(folder / 'A.wav') == sha256(folder / 'B.wav'), 'identity control differs')
            key.append(record)
        for track in tracks:
            for variant in track['variants']:
                require(sha256(Path(variant['path'])) == variant['sha256'], 'source changed during preparation')
        dump(organizer / 'key.private.json', {'mode': manifest['mode'], 'trials': key})
        with (listener / 'responses.csv').open('x', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({'trial': row['trial'],
                              **({'fatigue': 'not_assessed'} if manifest['mode'] == 'procedural_diagnostic' else {})}
                             for row in key)
        instructions = (PROCEDURAL_DIAGNOSTIC_INSTRUCTIONS
                        if manifest['mode'] == 'procedural_diagnostic' else INSTRUCTIONS)
        (listener / 'INSTRUCTIONS.txt').write_text(instructions)
        dump(organizer / 'session.private.json', {
            'listener': str(listener), 'responses': 'responses.csv',
            'python_version': sys.version,
            'tool_versions': {tool: run([tool, '-version'])[0].splitlines()[0]
                              for tool in ['ffmpeg', 'ffprobe']}})
        # READY is the only completion signal and carries no diagnostics or expectations.
        (listener / 'READY').write_text('Ready for listening. Read INSTRUCTIONS.txt.\n')
    except Exception:
        shutil.rmtree(listener)
        raise


def reveal(organizer, responses):
    organizer, responses = Path(organizer), Path(responses)
    key = json.loads((organizer / 'key.private.json').read_text())
    with responses.open(newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames == FIELDS, 'response columns changed')
        rows = list(reader)
    expected = {t['trial'] for t in key['trials']}
    require(len(rows) == len(expected) and {r['trial'] for r in rows} == expected,
            'complete every trial exactly once before reveal')
    for row in rows:
        require(all(row[k] in CHOICES for k in FIELDS[1:5]) and
                row['confidence'] in {'low', 'medium', 'high'} and
                row['fatigue'] in {'none', 'mild', 'high', 'not_assessed'} and None not in row,
                'invalid or incomplete categorical responses')
    if key.get('mode') == 'procedural_diagnostic':
        require(all(row['fatigue'] == 'not_assessed' for row in rows),
                'procedural diagnostic fatigue must remain not_assessed')
    session = json.loads((organizer / 'session.private.json').read_text())
    listener = Path(session['listener'])
    verify_audio(listener, key)
    output = organizer / 'revealed.private.json'
    with output.open('x') as f:
        json.dump({'key': key, 'responses': rows, 'responses_sha256': sha256(responses)}, f, indent=2)
        f.write('\n')
    return output


def verify(organizer):
    organizer = Path(organizer)
    key = json.loads((organizer / 'key.private.json').read_text())
    session = json.loads((organizer / 'session.private.json').read_text())
    return verify_audio(Path(session['listener']), key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    make = commands.add_parser('prepare')
    make.add_argument('manifest', type=Path)
    make.add_argument('--listener', type=Path, required=True)
    make.add_argument('--organizer', type=Path, required=True)
    show = commands.add_parser('reveal')
    show.add_argument('--organizer', type=Path, required=True)
    show.add_argument('--responses', type=Path, required=True)
    check = commands.add_parser('verify')
    check.add_argument('--organizer', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'prepare':
            prepare(args.manifest, args.listener, args.organizer)
        elif args.command == 'reveal':
            print(reveal(args.organizer, args.responses))
        else:
            counts = verify(args.organizer)
            print(f"ready=true trials={counts['trials']} audio_files={counts['audio_files']}")
    except (ValueError, KeyError, OSError, subprocess.CalledProcessError) as error:
        if args.command == 'verify':
            parser.exit(1, f'Verification failed: {error}\n')
        parser.exit(1, f'Aborted: {error}\n')


if __name__ == '__main__':
    main()
