"""Positive limiter-activation control after musical excerpts failed to engage."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
MODULE=Path(__file__).with_name('validate_native.py')
spec=importlib.util.spec_from_file_location('quality',MODULE)
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)


def main():
    out=ROOT/'runs/stage-accountability-20261020/master-observer-engaged-v1'
    if out.exists(): raise FileExistsError(out)
    out.mkdir()
    t=np.arange(96000,dtype=np.float64)/48000
    level=np.where(t<1/3,.02,1.)
    x=level*(.13*np.sin(2*np.pi*130*t)+.07*np.sin(2*np.pi*440*t)+.04*np.sin(2*np.pi*6300*t))
    x[48000:48048]+=1.3
    audio=np.stack([x,.91*x],axis=1).astype(np.float32)
    wave=out/'known-peak-control.wav';quality._write_float_wav(wave,audio)
    baseline=ROOT/'runs/stage-accountability-20261020/baseline-674a472'
    candidate=ROOT/'runs/stage-accountability-20261020/master-observer-candidate-v1'
    cases=[('legacy',[]),('smooth',['--kick-transient-guard','--clock-transient-punch','--quiet-mid-denoise']),('budget-half',['--max-limiter-gr-db','0.5'])]
    variants=['baseline','candidate-disabled','candidate-json','candidate-wavs']
    pins={str(p):quality.digest(p) for p in [baseline,candidate,wave,Path(__file__),MODULE]}
    quality.write_json(out/'protocol.json',{'schema':'master-engaged-limiter-control-v1','reason':'PartA musical excerpts all had zero GR; keep that negative, use declared known peak positive control',
        'fixture':'2s48kstereo:130/440/6300Hz tones .13/.07/.04; .02level first1/3s; +1.3pulse at48000..48048; right=.91left; float32 once; no target fitting',
        'input_sha256':quality.digest(wave),'pins':pins,'variants':variants,'cases':cases,'target_lufs':-9,'ceiling_db':-1,
        'gates':{'exact_output_hash_parity':True,'min_limiter_gr_db':.05,'max_budget_gr_db':.50001,'process_cap_mib':512},
        'fixed_onsets_samples':[48000],'scope':'known positive control, no music/preference/speed evidence'})
    records=[]
    try:
        for name,flags in cases:
            rr=[]
            for v in variants:
                for p,h in pins.items(): assert quality.digest(p)==h, p
                case=out/f'{name}-{v}';audit=case/'audit'
                binary=baseline if v=='baseline' else candidate
                cmd=[str(binary),'master','--input',str(wave),'--out',str(case),'--target-lufs','-9',*flags]
                if v in ('candidate-json','candidate-wavs'):cmd+=['--stage-audit-dir',str(audit)]
                if v=='candidate-wavs':cmd+=['--stage-audit-wavs']
                r=quality.run_child(cmd,out/f'{name}-{v}.log',out)
                r.update({'variant':v,'mastered_sha256':quality.digest(case/'mastered.wav'),
                    'listen_sha256':quality.digest(case/'listen.wav'),'master_json_sha256':quality.digest(case/'master.json')})
                master=json.loads((case/'master.json').read_text())
                gr=master['limiter']['max_gain_reduction_db']
                assert gr>=.05,(name,v,gr)
                if name=='budget-half':assert gr<=.50001,gr
                r['limiter_gain_reduction_db']=gr
                if audit.exists():
                    summary=json.loads((audit/'master_audit.json').read_text());assert summary['complete']
                    r['audit_summary_sha256']=quality.digest(audit/'master_audit.json')
                if v=='candidate-wavs':
                    trace=[json.loads(s) for s in (audit/'observations.jsonl').read_text().splitlines()]
                    quality.write_json(case/'fixed_onsets.json',{'schema':'master-fixed-onsets-v1','sample_rate_hz':48000,
                        'input_sha256':quality.digest(audit/trace[0]['snapshot']['file']),
                        'samples':[48000],'transient_ms':20,'sustain_start_ms':80,'sustain_end_ms':200})
                rr.append(r)
            for key in ['mastered_sha256','listen_sha256','master_json_sha256']:
                assert len({r[key] for r in rr})==1,(name,key)
            records.append({'case':name,'variants':rr,'all_outputs_exact':True})
            quality.write_json(out/f'{name}-receipt.json',records[-1])
        quality.write_json(out/'COMPLETE.json',{'schema':'master-engaged-limiter-complete-v1','cases':records,
            'protocol_sha256':quality.digest(out/'protocol.json'),'native_runs':12,'all_outputs_exact':True,
            'max_sampled_process_hwm_mib':max(r['sampled_process_hwm_mib'] for c in records for r in c['variants'] if r['sampled_process_hwm_mib'] is not None),
            'speed_qualification':False})
    except BaseException as e:
        quality.write_json(out/'HALTED.json',{'reason':str(e),'completed_cases':len(records)});raise

if __name__=='__main__':main()
