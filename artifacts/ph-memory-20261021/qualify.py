"""Bounded numerical parity for buffer lifetimes; not a speed experiment."""
from pathlib import Path
import json,hashlib,subprocess,time,sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
from run_mid_profile import memory
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 with p.open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def run(cmd,log,root):
 peak=0;seen=False
 with log.open('x') as f:
  c=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT);start=time.monotonic()
  try:
   while c.poll() is None:
    m=memory(c.pid);seen|=bool(m);peak=max(peak,m.get('VmRSS',0),m.get('VmHWM',0))
    if peak>512*1024 or time.monotonic()-start>180 or (root/'HALT').exists():raise RuntimeError('memory/time/HALT')
    time.sleep(.1)
  finally:
   if c.poll() is None:
    c.terminate()
    try:c.wait(timeout=3)
    except subprocess.TimeoutExpired:c.kill();c.wait()
 assert c.returncode==0,(cmd,c.returncode)
 return {'argv':cmd,'log_sha256':sha(log),'sampled_hwm_mib':peak/1024 if seen else None}
def clean(x):
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items() if k!='elapsed_ms'}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
def main():
 old=ROOT/'runs/listening-coherence-20261020/coherence-candidate-v1'
 new=ROOT/'runs/ph-memory-20261021/lifetime-candidate-v1'
 wave=ROOT/'runs/stage-accountability-20261020/master-observer-engaged-v1/known-peak-control.wav'
 r=ROOT/'runs/ph-memory-20261021/parity-v1';assert not r.exists();r.mkdir()
 pins={str(p):sha(p) for p in [old,new,wave,ROOT/'src/gtf.rs',Path(__file__)]}
 save(r/'protocol.json',{'schema':'ph-lifetime-parity-v1','pins':pins,'scope':'CPUprocedural13modes exactaudioandaudits, no speedclaim','expected_source_change':'four explicit drops; no FFT/reductionorder or arithmetic changes'})
 rows=[]
 try:
  reports=[]
  for name,b in [('before',old),('after',new)]:
   for p,h in pins.items():assert sha(p)==h,p
   out=r/name;cmd=[str(b),'morphic-compare','--input',str(wave),'--out',str(out),'--target-lufs','-13']
   rows.append(run(cmd,r/(name+'.log'),r))
   reports.append(json.loads((out/'morphic_comparison.json').read_text()))
  wavs=sorted(p.name for p in (r/'before').glob('*.wav'));assert len(wavs)>=39,len(wavs)
  for n in wavs:assert sha(r/'before'/n)==sha(r/'after'/n),n
  assert clean(reports[0])==clean(reports[1]),'report differed beyond elapsed_ms'
  auto=[]
  for name,b in [('before',old),('after',new)]:
   out=r/(name+'-auto')
   cmd=[str(b),'auto-master','--input',str(wave),'--out',str(out),'--no-sfht','--no-mid','--no-denoise','--no-auto-eq','--no-spatial','--no-defizz','--morphic-mode','ph-lie-v2','--quiet-mid-denoise','--kick-transient-guard','--clock-transient-punch','--target-lufs','-13','--no-export-sdcard']
   rows.append(run(cmd,r/(name+'-auto.log'),r));auto.append(out)
  for n in ['mastered.wav','listen.wav','morphic_audit.json','master.json']:
   assert sha(auto[0]/n)==sha(auto[1]/n),n
  save(r/'COMPLETE.json',{'schema':'ph-lifetime-parity-complete-v1','exact_wav_files':len(wavs),'all_wav_sha256_equal':True,'all_non_timing_report_fields_equal':True,'complete_ph_audit_and_finalizer_metadata_exact':True,'runs':rows,'pins':pins,'speed_qualification':False})
 except BaseException as e:save(r/'HALTED.json',{'reason':str(e),'runs':rows});raise
if __name__=='__main__':main()
