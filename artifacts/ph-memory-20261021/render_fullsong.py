"""Render one matched control and exactly one eligible experimental candidate."""
from pathlib import Path
import hashlib,json,os,subprocess,time,sys,shutil
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from run_mid_profile import memory
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p,v):
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')

def main():
 r=ROOT/'runs/listening-coherence-20261020/fullsong-v3'
 if r.exists():raise FileExistsError(r)
 gate=ROOT/'runs/listening-coherence-20261020/real-gate-v3/COMPLETE.json'
 g=json.loads(gate.read_text());assert g['eligible_for_bounded_listening']
 source=Path('/sdcard/Download/Chasing Horizons.wav')
 baseline=ROOT/'runs/ph-memory-20261021/lifetime-candidate-v1'
 candidate=baseline
 model=ROOT/'artifacts/rich-mid-v3/basis.json';low=ROOT/'artifacts/rich-low-sfht-v2/basis.json'
 assert sha(source)=='1ee053d29036ecba69ddbf0d7cf49483950c717c946a1ec45fd4cb34fc83b559'
 assert sha(baseline)=='7e4c47fe19e589553a8c4e999f988b76683de2967c7b5547f2b58fb9484cf161'
 assert sha(candidate)=='7e4c47fe19e589553a8c4e999f988b76683de2967c7b5547f2b58fb9484cf161'
 assert sha(model)=='e2ca85b64521ca62d72278b33177b426d4386bc0add0ff2d94526f76a5d09ae3'
 if shutil.disk_usage(ROOT).free<3*1024**3:raise RuntimeError('need3GiBfree')
 parity=ROOT/'runs/ph-memory-20261021/parity-v1/COMPLETE.json'
 proof=json.loads(parity.read_text());assert proof['all_wav_sha256_equal'] and proof['all_non_timing_report_fields_equal'] and proof['complete_ph_audit_and_finalizer_metadata_exact']
 pins={str(p):sha(p) for p in [source,baseline,candidate,model,low,gate,parity,ROOT/'src/gtf.rs',Path(__file__)]}
 args=['auto-master','--input',str(source),'--sfht-model',str(low),'--mid-model',str(model),'--backend','opencl','--target-lufs','-13','--ceiling-db','-1','--steps','8','--chunk-seconds','10','--overlap-seconds','2','--morphic-mode','ph-lie-v2','--clock-transient-punch','--kick-transient-guard','--quiet-mid-denoise','--no-export-sdcard']
 available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))
 if available<2560*1024:raise RuntimeError('need2.5GiBavailablebefore1.5GiBboundedretry')
 r.mkdir(parents=True)
 save(r/'protocol.json',{'schema':'one-fullsong-coherence-listening-v1','pins':pins,'shared_args':args,'candidate_only_flag':'--mid-coherence','gate_sha256':sha(gate),'process_hwm_cap_mib':1536,'child_timeout_s':900,'candidate_count':1,'new_baseline_is_control':True,'gates':['nativefinitegeometry','native4xtruepeak<=-0.99dBTP','all3nativepreservationchecks','matchedsettings/models/input','actualMidcoherenceapplied','candidateaudiomustdiffer'],'scope':'listeningeligibilityonly; nopreference/promotion','resource_revision':'four deadSpectrum buffers released afterlastuse; exactparityrequired; same1536MiBcapandallqualityaudits; controlandcandidateuseidenticalfixedbinary; onlymid-coherenceflagdiffers'})
 rows=[]
 try:
  for name,binary,extra in [('control',baseline,[]),('candidate',candidate,['--mid-coherence'])]:
   for p,h in pins.items():assert sha(p)==h,p
   if (r/'HALT').exists():raise RuntimeError('manualHALT')
   out=r/name;cmd=[str(binary),*args,'--out',str(out),*extra]
   peak=0;observed=False;start=time.monotonic()
   env=os.environ.copy();env['OCL_ICD_ASSUME_ICD_EXTENSION']='1'
   for k in ['ROACH_PROFILE_MID','ROACH_MID_CACHE_MIB','ROACH_MID_CACHE_BUILD','ROACH_MID_BATCH_ROWS','ROACH_FOREGROUND_CONFIRMED']:env.pop(k,None)
   with (r/(name+'.log')).open('x') as log:
    child=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    try:
     while child.poll() is None:
      m=memory(child.pid);observed|=any(k in m for k in ['VmRSS','VmHWM']);peak=max(peak,m.get('VmRSS',0),m.get('VmHWM',0))
      if peak>1536*1024 or time.monotonic()-start>900 or (r/'HALT').exists():raise RuntimeError(json.dumps({'reason':'resource_or_HALT_stop','peak_hwm_kib':peak,'elapsed_seconds':time.monotonic()-start,'halt_present':(r/'HALT').exists(),'cap_mib':1536}))
      time.sleep(.2)
    finally:
     if child.poll() is None:
      child.terminate()
      try:child.wait(timeout=5)
      except subprocess.TimeoutExpired:child.kill();child.wait()
   assert child.returncode==0,(name,child.returncode)
   summary=json.loads((out/'summary.json').read_text());meta=json.loads((out/'master.json').read_text())
   stats=summary['master_stats']['after'];assert stats['true_peak_dbtp']<=-.99,stats
   preservation=summary['preservation'];assert all(preservation[k] is True for k in ['mono_compatibility_passed','known_band_preserved','transient_timing_passed']),preservation
   settings=summary['settings'];assert settings['force_specialists'] is False and settings['master_controls']['quiet_mid_denoise'] is True
   assert summary['mid_model']==str(model) or Path(summary['mid_model']).resolve()==model.resolve()
   if name=='candidate':assert settings['mid_coherence_applied'] is True
   rows.append({'name':name,'argv':cmd,'mastered_sha256':sha(out/'mastered.wav'),'summary_sha256':sha(out/'summary.json'),'master_json_sha256':sha(out/'master.json'),'peak_sampled_process_hwm_mib':peak/1024 if observed else None,'stats':stats,'preservation':preservation,'settings':settings})
   save(r/(name+'-receipt.json'),rows[-1])
  drop={'mid_coherence_requested','mid_coherence_applied','explicit_cli_parameters'}
  assert {k:v for k,v in rows[0]['settings'].items() if k not in drop}=={k:v for k,v in rows[1]['settings'].items() if k not in drop},'settings changed'
  assert rows[0]['mastered_sha256']!=rows[1]['mastered_sha256'],'identicalcandidate'
  save(r/'COMPLETE.json',{'schema':'fullsong-coherence-eligible-v1','eligible_for_listening':True,'candidate_count':1,'control_count':1,'input_sha256':sha(source),'protocol_sha256':sha(r/'protocol.json'),'rows':rows,'owner_preference':None,'production_promoted':False})
 except BaseException as e:
  save(r/'HALTED.json',{'reason':str(e),'completed_rows':rows});raise

if __name__=='__main__':main()
