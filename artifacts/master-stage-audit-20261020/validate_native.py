"""Bounded procedural parity qualification, not a speed benchmark."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
from roach_ears.audio_io import read_wav
from long_coherence_fixture import _write_float_wav
from run_mid_profile import memory


def digest(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()


def write_json(p, x):
    with Path(p).open('x') as f: json.dump(x, f, indent=2, allow_nan=False); f.write('\n')


def run_child(command, log, session, expect_success=True):
    observed, peak = False, 0
    with log.open('x') as f:
        child = subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT)
        start = time.monotonic()
        try:
            while child.poll() is None:
                m = memory(child.pid)
                observed |= any(k in m for k in ('VmRSS', 'VmHWM'))
                peak = max(peak, m.get('VmRSS', 0), m.get('VmHWM', 0))
                if peak > 512*1024 or time.monotonic()-start > 180 or (session/'HALT').exists():
                    raise RuntimeError('native quality child memory/time/HALT stop')
                time.sleep(.1)
        finally:
            if child.poll() is None:
                child.terminate()
                try: child.wait(timeout=3)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
    if (child.returncode == 0) != expect_success:
        raise RuntimeError(f'unexpected exit {child.returncode}: {log}')
    return {'argv':command, 'exit_code':child.returncode,
            'sampled_process_hwm_mib':peak/1024 if observed else None,
            'log_sha256':digest(log)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--source-tree',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    args.candidate=args.candidate.resolve(); args.baseline=args.baseline.resolve(); args.source_tree=args.source_tree.resolve()
    assert digest(args.baseline)=='1b474970ec278a7db41424d57fb709cf37034e1934b3a5cdf572d2490dca2828'
    out=args.out.resolve()
    if out.exists(): raise FileExistsError(out)
    if shutil.disk_usage(ROOT).free < 3*1024**3: raise RuntimeError('need3GiBfree')
    source=ROOT/'runs/coherence-20261018/fixture-881000-hopaligned-v1/view0-input.wav'
    manifest=source.parent/'manifest.json'
    original=read_wav(source)
    assert original.sample_rate == 48000 and original.samples.shape == (2,26*48000)
    out.mkdir(parents=True)
    excerpt=out/'procedural-8to14.wav'
    _write_float_wav(excerpt, original.samples[:,8*48000:14*48000].T)
    silence=out/'silence.wav'
    _write_float_wav(silence,np.zeros((16384,2),np.float32))
    smooth=['--kick-transient-guard','--clock-transient-punch','--quiet-mid-denoise']
    cases=[('legacy-stress',excerpt,[], -9,'master'),
           ('smooth-stress',excerpt,smooth,-9,'master'),
           ('budget-half',excerpt,['--max-limiter-gr-db','0.5'],-9,'master'),
           ('gain-only',excerpt,['--max-limiter-gr-db','0'],-9,'master'),
           ('silence',silence,[],-13,'master'),
           ('long-normal',source,smooth,-13,'master'),
           ('auto-bypass',excerpt,smooth+['--no-sfht','--no-mid','--no-denoise',
            '--no-auto-eq','--no-spatial','--no-defizz','--no-morphic-gtf','--no-export-sdcard'],-9,'auto-master')]
    pins={str(x):digest(x) for x in [args.baseline,args.candidate,source,manifest,excerpt,silence,
        Path(__file__),*[args.source_tree/x for x in ['src/master.rs','src/main.rs','src/master_audit.rs','Cargo.lock']]]}
    protocol={'schema':'master-observer-native-quality-v1','scope':'procedural output parity and actual observations; no speed or audible-quality claim',
        'baseline_parent':'674a472','pins':pins,'cases':[{'name':n,'input':str(i),'flags':f,'target_lufs':t,'command':c} for n,i,f,t,c in cases],
        'variants':['baseline','candidate-disabled','candidate-json','candidate-wavs'],
        'gates':['exact output WAV SHA parity','same master.json when applicable','complete trace with final-output binding',
                 'zero-budget has no limiter passes','bounded process memory 512MiB','no overwrite; HALT preserves evidence'],
        'frozen_onsets_global_samples':[456000,600000,744000,888000],
        'fixed_window_ms':{'transient':20,'sustain_start':80,'sustain_end':200},
        'interpretation':'Repeated deterministic variants; no model inference or new subjective outcome.'}
    write_json(out/'protocol.json',protocol)
    results=[]
    try:
        for name,input_audio,flags,target,subcmd in cases:
            variants=[]
            for variant in protocol['variants']:
                if (out/'HALT').exists(): raise RuntimeError('manual HALT')
                for path,h in pins.items():
                    if digest(path)!=h: raise RuntimeError(f'source changed: {path}')
                case=out/f'{name}-{variant}'
                binary=args.baseline if variant=='baseline' else args.candidate
                command=[str(binary),subcmd,'--input',str(input_audio),'--out',str(case),
                         '--target-lufs',str(target),*flags]
                audit=case/'audit'
                if variant in ('candidate-json','candidate-wavs'):
                    command+=['--stage-audit-dir',str(audit)]
                if variant=='candidate-wavs': command+=['--stage-audit-wavs']
                record=run_child(command,out/f'{name}-{variant}.log',out)
                wave='mastered.wav'
                if not (case/wave).is_file():
                    choices=list(case.glob('*master*.wav'))
                    if len(choices)!=1: raise RuntimeError(f'no unambiguous mastered output in {case}: {choices}')
                    wave=choices[0].name
                record.update({'variant':variant,'output':str(case/wave),'output_sha256':digest(case/wave)})
                record['listen_sha256']=digest(case/'listen.wav')
                if subcmd=='master': record['master_json_sha256']=digest(case/'master.json')
                if audit.exists():
                    summary=json.loads((audit/'master_audit.json').read_text())
                    assert summary['complete'] is True
                    assert summary['record_count']<=32
                    if name=='gain-only':
                        assert summary['limiter_pass_count']==0 and summary['gain_only_bypass_count']>=1
                    record['audit_summary']=summary
                    record['audit_summary_sha256']=digest(audit/'master_audit.json')
                    record['trace_sha256']=digest(audit/'observations.jsonl')
                    if variant=='candidate-wavs':
                        trace=[json.loads(s) for s in (audit/'observations.jsonl').read_text().splitlines()]
                        first=audit/trace[0]['snapshot']['file']
                        samples=[] if name=='silence' else ([72000,216000] if input_audio==excerpt else [456000,600000,744000,888000])
                        write_json(case/'fixed_onsets.json',{'schema':'master-fixed-onsets-v1','sample_rate_hz':48000,
                            'input_sha256':digest(first),'samples':samples,'transient_ms':20,
                            'sustain_start_ms':80,'sustain_end_ms':200,
                            'provenance':'pre-existing continuous fixture manifest events; excerpt subtracts8seconds; silence hasnoevents'})
                variants.append(record)
            for key in ['output_sha256','listen_sha256']+(['master_json_sha256'] if subcmd=='master' else []):
                assert len({v[key] for v in variants})==1,(name,key)
            results.append({'case':name,'exact_outputs':True,'variants':variants})
            write_json(out/f'{name}-receipt.json',results[-1])
        negatives=[]
        for name,extra in [('same-path',['--stage-audit-dir',str(out/'invalid-same-path')]),
                           ('wavs-without-dir',['--stage-audit-wavs'])]:
            bad=out/('invalid-'+name)
            command=[str(args.candidate),'master','--input',str(excerpt),'--out',str(bad),*extra]
            negatives.append(run_child(command,out/f'{name}-negative.log',out,expect_success=False))
            assert not bad.exists(),bad
        peaks=[v['sampled_process_hwm_mib'] for r in results for v in r['variants'] if v['sampled_process_hwm_mib'] is not None]
        write_json(out/'COMPLETE.json',{'schema':'master-observer-native-quality-complete-v1','cases':results,
            'negative_requests':negatives,'protocol_sha256':digest(out/'protocol.json'),
            'native_runs':28,'all_outputs_exact':True,'max_sampled_process_hwm_mib':max(peaks) if peaks else None,
            'speed_qualification':False})
    except BaseException as e:
        write_json(out/'HALTED.json',{'reason':str(e),'completed_cases':len(results)})
        raise

if __name__=='__main__': main()
