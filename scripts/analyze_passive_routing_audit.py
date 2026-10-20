#!/usr/bin/env python3
"""Verify and compactly analyze a completed passive routing matrix."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from run_coherence_quality import wav_info

GAINS = (0, -20, -40)
SEEDS = (860002, 860003, 860004, 860005, 860007, 860015, 860019)
SCORE_FIELDS = ('sub_energy_dbfs', 'low_mid_energy_dbfs', 'mid_energy_dbfs',
                'high_energy_dbfs', 'ultrasonic_energy_dbfs')
AUTH_FIELDS = ('sub_bass_authority', 'mid_flow_authority', 'high_field_authority',
               'spatial_cleanup_authority', 'conservative_defizz_authority',
               'master_glue_authority')


def digest(path: Path) -> str:
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def finite_tree(value) -> bool:
    if isinstance(value, dict):
        return all(finite_tree(x) for x in value.values())
    if isinstance(value, list):
        return all(finite_tree(x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def legacy_authorities(profile: dict) -> tuple[dict, dict]:
    """Mirror the frozen src/assess.rs authority branches and retain triggers."""
    a, x = profile['artifacts'], profile['authorities']
    sub_score = profile['sub_energy_dbfs']
    low_mid = profile['low_mid_energy_dbfs']
    cutoff = profile['estimated_cutoff_hz']
    sub_deep = sub_score < -42.0
    sub_mid = sub_score < -32.0
    sub_artifact = a['sub_instability'] > 0.40
    metal_mid = a['metallic_grain'] > 0.35
    low_mid_gate = low_mid < -38.0
    high_cutoff_deep = cutoff < 10_000.0
    high_cutoff_mid = cutoff < 16_000.0
    high_codec = a['codec_swish'] > 0.40
    if sub_deep or sub_artifact:
        sub = 0.85
    elif sub_mid:
        sub = 0.50
    else:
        sub = 0.0
    mid = 0.60 if metal_mid or low_mid_gate else 0.0
    high = 0.90 if high_cutoff_deep else (0.60 if high_cutoff_mid or high_codec else 0.0)
    if profile['geometry_channels'] == 1 or profile['interchannel_correlation'] > 0.99:
        spatial = 0.0
    elif profile['sub_side_leak_ratio'] > 0.06 or a['phase_haze'] > 0.30:
        spatial = 0.90
    else:
        spatial = 0.35
    shimmer = a['ai_shimmer'] > 0.25
    metal_defizz = a['metallic_grain'] > 0.30
    defizz = min(max(a['ai_shimmer'], a['metallic_grain']) * 1.2, 1.0) if shimmer or metal_defizz else 0.0
    crest = profile['crest_factor_db']
    glue = 0.20 if crest < 9.0 else (0.60 if crest < 12.0 else 0.85)
    predicted = {
        'sub_bass_authority': sub, 'mid_flow_authority': mid,
        'high_field_authority': high, 'spatial_cleanup_authority': spatial,
        'conservative_defizz_authority': defizz, 'master_glue_authority': glue,
    }
    basis = {
        'sub': ([name for name, active in (('legacy_sub_lt_-42', sub_deep),
                                            ('sub_instability_gt_0.40', sub_artifact),
                                            ('legacy_sub_lt_-32', sub_mid)) if active]
                or ['no_sub_gate']),
        'mid': ([name for name, active in (('metallic_grain_gt_0.35', metal_mid),
                                            ('legacy_low_mid_lt_-38', low_mid_gate)) if active]
                or ['no_mid_gate']),
        'high': ([name for name, active in (('estimated_cutoff_lt_10000', high_cutoff_deep),
                                             ('estimated_cutoff_lt_16000', high_cutoff_mid),
                                             ('codec_swish_gt_0.40', high_codec)) if active]
                 or ['no_high_gate']),
        'sub_score_at_floor': sub_score <= -119.999,
        'low_mid_score_at_floor': low_mid <= -119.999,
        'mid_score_at_floor': profile['mid_energy_dbfs'] <= -119.999,
        'high_score_at_floor': profile['high_energy_dbfs'] <= -119.999,
        'ultrasonic_score_at_floor': profile['ultrasonic_energy_dbfs'] <= -119.999,
    }
    return predicted, basis


def _is_active(action: str) -> bool:
    return isinstance(action, str) and action.startswith('ACTIVE')


def _role_case(manifest: dict, seed: int, role: str) -> dict:
    return next(c for c in manifest['cases'] if int(c['seed']) == seed)


def _verify(session: Path, protocol_path: Path) -> tuple[dict, dict, dict, dict]:
    complete_path, plan_path = session/'COMPLETE.json', session/'plan.json'
    require(not (session/'HALTED.json').exists(), 'session is halted')
    require(complete_path.is_file() and plan_path.is_file(), 'session is incomplete')
    complete = json.loads(complete_path.read_text())
    plan = json.loads(plan_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    require(complete.get('schema') == 'passive-routing-complete-v1', 'unsupported COMPLETE schema')
    require(plan.get('schema') == 'passive-routing-plan-v1', 'unsupported plan schema')
    require(protocol.get('schema') == 'passive-routing-protocol-v1', 'unsupported protocol schema')
    require(complete.get('row_count') == protocol.get('row_count') == 45, 'expected 45 completed rows')
    require(plan.get('gains_db') == list(GAINS) and protocol.get('gain_policy_db') == list(GAINS),
            'gain plan differs from frozen protocol')
    require(plan.get('binary_sha256') == complete.get('binary_sha256') == protocol.get('binary_sha256'),
            'binary hash differs among plan, COMPLETE, and protocol')
    binary = Path(plan['binary'])
    require(binary.is_file() and digest(binary) == plan['binary_sha256'], 'native binary hash mismatch')
    require(digest(protocol_path) == next((p['sha256'] for p in complete['source_pins']
                                          if Path(p['path']).resolve() == protocol_path.resolve()), None),
            'protocol is not pinned to COMPLETE or its hash changed')
    require(digest(Path(plan['bundle_manifest'])) == plan['bundle_sha256'], 'bundle manifest hash mismatch')
    require(digest(Path(plan['long_identity'])) == plan['long_identity_sha256'], 'long identity hash mismatch')
    pinmap = {}
    for pin in complete.get('source_pins', []):
        path = Path(pin['path'])
        require(path.is_file() and digest(path) == pin['sha256'], f'source pin mismatch: {path}')
        pinmap[path.resolve()] = pin['sha256']
    require(pinmap.get(Path(plan['long_identity']).resolve()) == plan['long_identity_sha256'], 'identity absent from source pins')
    require(plan.get('source_pins') == complete.get('source_pins'), 'plan/COMPLETE source pin lists differ')
    bundle_path = Path(plan['bundle_manifest'])
    bundle = json.loads(bundle_path.read_text())
    require(bundle.get('schema') == 'mid-procedural-source-bundle-v1', 'unsupported bundle schema')
    require({int(c['seed']) for c in bundle.get('cases', [])} == set(SEEDS), 'bundle seed set mismatch')
    require(protocol.get('source_parent') == plan.get('source_parent'), 'protocol source parent differs from plan')
    require(complete.get('runner_source_sha256') in [p['sha256'] for p in complete['source_pins']
            if Path(p['path']).name == 'run_passive_routing_audit.py'],
            'executed runner is not bound to source pins')
    require(complete.get('observer_source_sha256') == digest(Path(__file__).with_name('audit_assessment_bands.py')),
            'observer source hash differs from COMPLETE')
    return complete, plan, protocol, bundle


def analyze(session: Path, protocol_path: Path, out: Path) -> dict:
    require(not out.exists(), f'refusing existing output: {out}')
    complete, plan, protocol, bundle = _verify(session, protocol_path)
    rows_path = session/'rows.jsonl'
    raw_rows = [json.loads(line) for line in rows_path.read_text().splitlines() if line.strip()]
    require(len(raw_rows) == 45, 'rows.jsonl does not contain 45 rows')
    cases = {int(c['seed']): c for c in bundle['cases']}
    pinmap = {Path(pin['path']).resolve(): pin['sha256'] for pin in complete['source_pins']}
    recipe_meta, source_audio_hash, source_geometry = {}, {}, {}
    bundle_root = Path(plan['bundle_manifest']).parent
    for seed, case in cases.items():
        recipe_path = bundle_root/case['recipe_path']
        require(digest(recipe_path) == case['recipe_sha256'], f'seed {seed} recipe hash mismatch')
        recipe = json.loads(recipe_path.read_text())
        require(recipe.get('seed') == seed and recipe.get('family') == case['family'],
                f'seed {seed} recipe identity mismatch')
        recipe_meta[seed] = {'family': case['family'],
            'no_upper': recipe.get('no_upper'), 'no_lower': recipe.get('no_lower'),
            'mono': recipe.get('mono'), 'gap': recipe.get('gap'),
            'recipe_sha256': case['recipe_sha256']}
        for role in ('target', 'degraded_input'):
            spec = case['files'][role]
            wav = bundle_root/spec['wav']['path']
            raw = bundle_root/spec['planar_f32le']['path']
            require(digest(wav) == spec['wav']['sha256'], f'seed {seed}/{role} WAV hash mismatch')
            require(digest(raw) == spec['planar_f32le']['sha256'], f'seed {seed}/{role} planar hash mismatch')
            require(pinmap.get(wav.resolve()) == spec['wav']['sha256'], f'seed {seed}/{role} WAV absent from source pins')
            require(pinmap.get(raw.resolve()) == spec['planar_f32le']['sha256'], f'seed {seed}/{role} planar file absent from source pins')
            source_info = wav_info(wav)
            require(source_info['rate'] == 48_000 and source_info['frames'] == 16_384
                    and source_info['channels'] in (1, 2) and source_info['format'] == 3
                    and source_info['bits_per_sample'] == 32, f'seed {seed}/{role} source geometry invalid')
            source_audio_hash[(seed, role)] = spec['wav']['sha256']
            source_geometry[(seed, role)] = source_info['channels']
    expected_ids = {f'seed-{s}-{r}-g{g}' for s in SEEDS for r in ('target', 'degraded_input') for g in (0, -20, -40)}
    expected_ids |= {f'long-identity-g{g}' for g in GAINS}
    require({r.get('id') for r in raw_rows} == expected_ids, 'fixed row identities differ')
    require(len({r['id'] for r in raw_rows}) == 45, 'duplicate row IDs')
    rowmap, compact_rows = {}, []
    for row in raw_rows:
        rid = row['id']
        require(finite_tree(row), f'{rid}: nonfinite row JSON')
        seed, role, gain = row['seed'], row['role'], int(row['gain_db'])
        if seed is None:
            require(role == 'target_as_input_no_added_damage' and rid == f'long-identity-g{gain}',
                    f'{rid}: unexpected long identity identity')
            expected_origin = plan['long_identity_sha256']
            expected_samples, expected_channels = 26*48_000, 2
        else:
            require(role in ('target', 'degraded_input'), f'{rid}: invalid role')
            require(row['family'] == recipe_meta[int(seed)]['family'], f'{rid}: source family mismatch')
            expected_origin = source_audio_hash[(int(seed), role)]
            expected_samples = 16_384
            expected_channels = source_geometry[(int(seed), role)]
        require(row['origin_sha256'] == expected_origin, f'{rid}: source audio hash mismatch')
        require(row['binary_sha256'] == plan['binary_sha256'], f'{rid}: row binary hash mismatch')
        require(row.get('intent_scope') == 'base-only; MusicalIntentCard modulation not applied',
                f'{rid}: unexpected intent scope')
        case_dir = session/'rows'/rid
        wav = case_dir/'input.wav'
        assessment_path, card_path = case_dir/'assessment'/'assessment.json', case_dir/'assessment'/'tuning_card.json'
        log, observer_path = case_dir/'native_assess.log', case_dir/'observer.json'
        require(wav.is_file() and digest(wav) == row['input_sha256'], f'{rid}: derived input hash mismatch')
        info = wav_info(wav)
        require(info['rate'] == 48_000 and info['frames'] == expected_samples
                and info['channels'] == expected_channels and info['format'] == 3
                and info['bits_per_sample'] == 32, f'{rid}: input WAV header mismatch')
        require(assessment_path.is_file() and digest(assessment_path) == row['assessment_sha256'],
                f'{rid}: assessment hash mismatch')
        require(card_path.is_file() and digest(card_path) == row['tuning_card_sha256'], f'{rid}: card hash mismatch')
        require(log.is_file() and digest(log) == row['native_log_sha256'], f'{rid}: native log hash mismatch')
        require(observer_path.is_file() and digest(observer_path) == row['observer_sha256'], f'{rid}: observer hash mismatch')
        profile, card, observer = (json.loads(assessment_path.read_text()),
                                   json.loads(card_path.read_text()), json.loads(observer_path.read_text()))
        require(finite_tree(profile) and finite_tree(card) and finite_tree(observer), f'{rid}: nonfinite artifact')
        require(profile['authorities'] == row['base_authorities'], f'{rid}: authority snapshot differs from assessment')
        require({k: profile[k] for k in SCORE_FIELDS} == row['legacy_scores_uncalibrated'],
                f'{rid}: legacy score snapshot differs from assessment')
        require(card['recommended_parameters'] == row['base_tuning_parameters'], f'{rid}: base card snapshot mismatch')
        require(card['stage_actions'] == row['base_stage_actions'], f'{rid}: stage action snapshot mismatch')
        require(observer['provenance']['audio_sha256'] == row['input_sha256'], f'{rid}: observer audio binding mismatch')
        require(observer['provenance']['observer_source_sha256'] == row['observer_source_sha256'],
                f'{rid}: observer source hash mismatch')
        band_channel = 'mid_orthonormal' if expected_channels == 2 else 'mono'
        observer_bands = observer['channels'][band_channel]['bands']
        for band_name, band_key in (('sub_0_120_hz', 'sub_0_120_hz'),
                                    ('low_mid_120_500_hz', 'low_mid_120_500_hz'),
                                    ('mid_500_6000_hz', 'mid_500_6000_hz')):
            observed = row['calibrated_mid_bands'][band_key]
            require(observed['rms_dbfs'] == observer_bands[band_name]['rms_dbfs']
                    and observed['power_below_null_floor'] == observer_bands[band_name]['power_below_null_floor'],
                    f'{rid}: calibrated band snapshot differs from observer artifact')
        from roach_ears.audio_io import read_wav
        from run_passive_routing_audit import scale_samples
        import numpy as np
        if seed is None:
            origin = Path(plan['long_identity'])
        else:
            origin = bundle_root/cases[int(seed)]['files'][role]['wav']['path']
        expected_audio = scale_samples(read_wav(origin).samples, gain)
        actual_audio = read_wav(wav).samples
        require(np.array_equal(expected_audio.view(np.uint32), actual_audio.view(np.uint32)),
                f'{rid}: waveform is not exact float32 gain applied to the pinned source')
        info_audio = read_one_audio(wav)
        require(math.isfinite(info_audio[0]) and math.isfinite(info_audio[1]), f'{rid}: nonfinite input samples')
        require(abs(info_audio[0] - row['level']['peak_sample']) <= 1e-7
                and abs(info_audio[1] - row['level']['rms']) <= 1e-7,
                f'{rid}: derived audio level differs from row receipt')
        profile_for_rule = dict(profile)
        profile_for_rule['geometry_channels'] = expected_channels
        predicted, basis = legacy_authorities(profile_for_rule)
        authority_delta = {k: abs(float(predicted[k]) - float(profile['authorities'][k])) for k in AUTH_FIELDS}
        require(all(v <= 1e-6 for v in authority_delta.values()), f'{rid}: authority rule reconstruction mismatch')
        scores = {k: profile[k] for k in SCORE_FIELDS}
        artifacts = {k: profile['artifacts'][k] for k in (
            'metallic_grain', 'codec_swish', 'sub_instability', 'ai_shimmer', 'phase_haze')}
        card_params = card['recommended_parameters']
        actions = card['stage_actions']
        mids = row['calibrated_mid_bands']
        compact = {
            'id': rid, 'seed': seed, 'role': role, 'family': row['family'], 'gain_db': gain,
            'source_sha256': expected_origin, 'input_sha256': row['input_sha256'],
            'assessment_sha256': row['assessment_sha256'],
            'base_tuning_card_sha256': row['tuning_card_sha256'],
            'observer_sha256': row['observer_sha256'], 'native_log_sha256': row['native_log_sha256'],
            'peak': row['level']['peak_sample'], 'rms': row['level']['rms'],
            'legacy_scores_uncalibrated': scores, 'estimated_cutoff_hz': profile['estimated_cutoff_hz'],
            'artifact_critics': artifacts, 'rule_basis': basis,
            'base_authorities': profile['authorities'],
            'base_card_authorities': {'sub': card_params['sub_bass_authority'],
                                      'mid': card_params['mid_flow_authority']},
            'base_stage_active': {'sub': _is_active(actions['stage1_sfht_action']),
                                  'mid': _is_active(actions['stage2_mid_flow_action']),
                                  'high': profile['authorities']['high_field_authority'] > 0,
                                  'spatial': profile['authorities']['spatial_cleanup_authority'] > 0,
                                  'defizz': profile['authorities']['conservative_defizz_authority'] > 0},
            'calibrated_mid_bands': mids,
            'null_score_fields': [k for k, v in scores.items() if v <= -119.999],
            'artifact_paths': {'input_wav': str(wav.relative_to(session)),
                'assessment_json': str(assessment_path.relative_to(session)),
                'base_tuning_card_json': str(card_path.relative_to(session)),
                'observer_json': str(observer_path.relative_to(session)),
                'native_assess_log': str(log.relative_to(session))},
        }
        compact_rows.append(compact)
        rowmap[rid] = {'compact': compact, 'profile': profile, 'card': card, 'basis': basis}

    transition_counts = Counter()
    card_action_transition_counts = Counter()
    transition_examples, subjects = [], {}
    for seed in SEEDS:
        for role in ('target', 'degraded_input'):
            subject = f'{seed}:{role}'
            subjects[subject] = [rowmap[f'seed-{seed}-{role}-g{g}'] for g in GAINS]
    subjects['long_identity'] = [rowmap[f'long-identity-g{g}'] for g in GAINS]
    dimensions = {'sub': 'sub_bass_authority', 'mid': 'mid_flow_authority', 'high': 'high_field_authority'}
    for subject, sequence in subjects.items():
        for index, (g0, g1) in enumerate(zip(GAINS, GAINS[1:])):
            left, right = sequence[index], sequence[index+1]
            p0, p1 = left['profile'], right['profile']
            for dim, field in dimensions.items():
                old, new = p0['authorities'][field], p1['authorities'][field]
                if old == new:
                    continue
                klass = transition_class(dim, left, right)
                transition_counts[(subject.split(':')[-1] if ':' in subject else 'long_identity',
                                   f'{g0}_to_{g1}', dim, klass)] += 1
                transition_examples.append({'subject': subject, 'from_gain_db': g0, 'to_gain_db': g1,
                    'dimension': dim, 'from_authority': old, 'to_authority': new,
                    'cause_class': klass,
                    'from_score': p0['sub_energy_dbfs'] if dim == 'sub' else
                                  p0['low_mid_energy_dbfs'] if dim == 'mid' else p0['estimated_cutoff_hz'],
                    'to_score': p1['sub_energy_dbfs'] if dim == 'sub' else
                                p1['low_mid_energy_dbfs'] if dim == 'mid' else p1['estimated_cutoff_hz']})
            for dim, key in (('sub', 'stage1_sfht_action'), ('mid', 'stage2_mid_flow_action')):
                old_active = _is_active(left['card']['stage_actions'][key])
                new_active = _is_active(right['card']['stage_actions'][key])
                if old_active != new_active:
                    card_action_transition_counts[(subject.split(':')[-1] if ':' in subject else 'long_identity',
                                                   f'{g0}_to_{g1}', dim)] += 1
    grouped_transitions = [
        {'role_group': k[0], 'gain_step': k[1], 'authority': k[2], 'cause_class': k[3], 'count': v}
        for k, v in sorted(transition_counts.items())]
    grouped_card_transitions = [
        {'role_group': k[0], 'gain_step': k[1], 'stage': k[2], 'active_status_flips': v}
        for k, v in sorted(card_action_transition_counts.items())]
    gain_summary = []
    for subject, seq in subjects.items():
        gain_summary.append({'subject': subject,
            'rows': [{'gain_db': GAINS[i], 'authority': {d: seq[i]['profile']['authorities'][f]
                        for d, f in dimensions.items()},
                      'card_auth': {'sub': seq[i]['card']['recommended_parameters']['sub_bass_authority'],
                                    'mid': seq[i]['card']['recommended_parameters']['mid_flow_authority']},
                      'active': seq[i]['compact']['base_stage_active']}
                     for i in range(3)]})
    paired = []
    for seed in SEEDS:
        for gain in GAINS:
            t = rowmap[f'seed-{seed}-target-g{gain}']['compact']
            d = rowmap[f'seed-{seed}-degraded_input-g{gain}']['compact']
            paired.append({'seed': seed, 'family': recipe_meta[seed]['family'], 'gain_db': gain,
                'target_no_added_damage_vs_degraded': {
                    field: {'target': t['base_authorities'][field], 'degraded_input': d['base_authorities'][field],
                            'equal': t['base_authorities'][field] == d['base_authorities'][field]}
                    for field in AUTH_FIELDS},
                'card_sub_mid': {'target': t['base_card_authorities'], 'degraded_input': d['base_card_authorities']}})
    attribution = Counter(x['cause_class'] for x in transition_examples)
    null_summary = []
    for seed in (860004, 860005):
        for role in ('target', 'degraded_input'):
            rows = [rowmap[f'seed-{seed}-{role}-g{g}']['compact'] for g in GAINS]
            null_summary.append({'seed': seed, 'family': recipe_meta[seed]['family'], 'role': role,
                'gains': [{'gain_db': r['gain_db'], 'legacy_score_fields_at_-120': r['null_score_fields'],
                           'authorities': {k:r['base_authorities'][k] for k in AUTH_FIELDS},
                           'card_sub_mid': r['base_card_authorities']} for r in rows]})
    result = {
        'schema': 'passive-routing-analysis-v1',
        'scope': 'development-only passive base assessment; target role means no additional damage, not musical health; no intent or quality claim',
        'provenance': {'session': str(session.resolve()), 'complete_sha256': digest(session/'COMPLETE.json'),
            'plan_sha256': digest(session/'plan.json'), 'rows_jsonl_sha256': digest(rows_path),
            'protocol_path': str(protocol_path.resolve()), 'protocol_sha256': digest(protocol_path),
            'binary_path': plan['binary'], 'binary_sha256': plan['binary_sha256'],
            'native_command_template': [plan['binary'], 'assess', '--input', '<row input.wav>', '--out', '<row assessment directory>'],
            'assess_source_sha256': next((p['sha256'] for p in complete['source_pins']
                if Path(p['path']).name == 'assess.rs'), None),
            'bundle_manifest_sha256': plan['bundle_sha256'], 'source_pins_verified': len(complete['source_pins']),
            'analysis_source_sha256': digest(Path(__file__))},
        'verification': {'rows': len(compact_rows), 'all_derived_audio_assessment_card_observer_and_log_hashes_verified': True,
            'all_native_authorities_reconstructed_from_frozen_assess_rules': True,
            'all_derived_waveforms_exact_f32_gain_from_pinned_source': True,
            'long_identity_geometry_samples': 26*48_000, 'short_source_geometry_samples': 16_384},
        'matrix': {'rows': compact_rows, 'subject_gain_sequences': gain_summary,
                   'target_vs_degraded_role_pairs': paired},
        'source_cases': [{'seed': seed, **recipe_meta[seed],
                          'target_sha256': source_audio_hash[(seed, 'target')],
                          'degraded_input_sha256': source_audio_hash[(seed, 'degraded_input')]}
                         for seed in SEEDS],
        'gain_transition_counts': grouped_transitions,
        'base_card_stage_action_transition_counts': grouped_card_transitions,
        'gain_transition_examples': transition_examples,
        'gain_transition_cause_counts': dict(sorted(attribution.items())),
        'null_and_near_null_controls': null_summary,
        'interpretation_limits': [
            'Raw routing fields are legacy uncalibrated windowed FFT-bin scores; calibrated whole-clip physical mid-band RMS is reported in parallel, never threshold-substituted.',
            'A score-threshold transition shows gain-sensitive behavior under the frozen rule, not by itself a unit-mismatch-caused error or an appropriate new threshold.',
            'Artifact, estimated-cutoff, and score-floor triggers are reported separately where those rules govern the decision.',
            'Exact silence authority reflects the existing score floor plus the unchanged rule, not evidence of harmful audio generation.',
            'Target role means no extra manufactured loss; it does not establish musical health. The long identity is a procedural source, not a certified healthy mix.',
            'Only base Assess/TuningCard actions are measured; MusicalIntentCard and manual auto-master modulation are not applied.'
        ]
    }
    require(finite_tree(result), 'analysis contains nonfinite values')
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')
    return result


def transition_class(dim: str, left: dict, right: dict) -> str:
    p0, p1 = left['profile'], right['profile']
    b0, b1 = left['basis'], right['basis']
    if dim == 'sub':
        if b0['sub_score_at_floor'] or b1['sub_score_at_floor']:
            return 'legacy_score_floor_interaction'
        s0, s1 = p0['sub_energy_dbfs'], p1['sub_energy_dbfs']
        raw_changed = (s0 < -42.0, s0 < -32.0) != (s1 < -42.0, s1 < -32.0)
        artifact_changed = (p0['artifacts']['sub_instability'] > .40) != (p1['artifacts']['sub_instability'] > .40)
        return 'legacy_score_threshold' if raw_changed else ('artifact_threshold' if artifact_changed else 'other_rule_or_mapping')
    if dim == 'mid':
        if b0['low_mid_score_at_floor'] or b1['low_mid_score_at_floor']:
            return 'legacy_score_floor_interaction'
        raw_changed = (p0['low_mid_energy_dbfs'] < -38.0) != (p1['low_mid_energy_dbfs'] < -38.0)
        metal_changed = (p0['artifacts']['metallic_grain'] > .35) != (p1['artifacts']['metallic_grain'] > .35)
        return 'legacy_score_threshold' if raw_changed else ('artifact_threshold' if metal_changed else 'other_rule_or_mapping')
    cut_changed = (p0['estimated_cutoff_hz'] < 10_000.0,
                   p0['estimated_cutoff_hz'] < 16_000.0) != (
                   p1['estimated_cutoff_hz'] < 10_000.0, p1['estimated_cutoff_hz'] < 16_000.0)
    codec_changed = (p0['artifacts']['codec_swish'] > .40) != (p1['artifacts']['codec_swish'] > .40)
    return 'estimated_cutoff_gate' if cut_changed else ('artifact_threshold' if codec_changed else 'other_rule_or_mapping')


def read_one_audio(path: Path) -> tuple[float, float]:
    """Decode one verified WAV at a time and return peak/RMS for finite check."""
    sys.path.insert(0, str(HERE))
    from roach_ears.audio_io import read_wav
    audio = read_wav(path)
    import numpy as np
    samples = np.asarray(audio.samples, dtype=np.float64)
    if samples.size == 0 or not np.isfinite(samples).all():
        return math.nan, math.nan
    return float(np.max(np.abs(samples))), float(np.sqrt(np.mean(samples*samples)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--session', required=True, type=Path)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.session, args.protocol, args.out)
    print(json.dumps({'out': str(args.out), 'rows': result['verification']['rows'],
                      'gain_transition_cause_counts': result['gain_transition_cause_counts']}, indent=2))


if __name__ == '__main__':
    main()
