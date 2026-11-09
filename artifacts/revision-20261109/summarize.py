import hashlib,json
from pathlib import Path
root=Path('runs/revision-20261109')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
summary={'scope':'single procedural fixture and single-start 100-update Mid pilot; no listening or speed conclusion'}
base=root/'baseline-pipeline/mastered.wav'
summary['audio_parity']={name:sha(root/name/'mastered.wav')==sha(base) for name in ['candidate-legacy','candidate-highzero']}
summary['pipeline']={}
for name in ['baseline-pipeline','candidate-legacy','candidate-smooth','candidate-highzero','candidate-high']:
 p=root/name/'summary.json';d=json.loads(p.read_text())
 summary['pipeline'][name]={'master_sha256':sha(root/name/'mastered.wav'),'master_stats':d['master_stats'],'preservation':d['preservation'],'high_field':d.get('settings',{}).get('high_field')}
summary['training']={}
for name in ['baseline-training','candidate-training']:
 p=root/name/'basis.json';summary['training'][name]={'model_sha256':sha(p),'training_receipt_sha256':sha(root/name/'training.json')}
summary['evaluation']={}
for name in ['eval-legacy','eval-independent','eval-legacy-paired','eval-independent-paired']:
 d=json.loads((root/name/'evaluation.json').read_text());rows=d['results']
 totals={k:sum(r.get(k) or 0 for r in rows) for k in ['target_missing_power','candidate_error_power','identity_error_power','prior_error_power']}
 d['pooled_powers']=totals
 d['pooled_complex_nmse']={k:totals[k]/totals['target_missing_power'] for k in ['candidate_error_power','identity_error_power','prior_error_power'] if totals['target_missing_power'] and (k!='prior_error_power' or any(r.get(k) is not None for r in rows))}
 summary['evaluation'][name]=d
Path('artifacts/revision-20261109/results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:{'mean_missing_nmse':v['mean_missing_nmse'],'pooled_complex_nmse':v['pooled_complex_nmse'],'max_passband_err':v['max_passband_err']} for k,v in summary['evaluation'].items()},indent=2))
print(summary['audio_parity'])
