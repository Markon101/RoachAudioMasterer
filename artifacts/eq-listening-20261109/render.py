import hashlib,json,os,subprocess,sys
from pathlib import Path
root=Path.cwd();plan=json.loads(Path('artifacts/eq-listening-20261109/plan.json').read_text());out=Path('runs/eq-listening-20261109')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
env=dict(os.environ,LD_PRELOAD='/data/data/com.termux/files/usr/lib/libc++_shared.so')
records=[]
for idx,source in enumerate(plan['sources']):
 wav=out/f'source-{idx}.wav'
 subprocess.run(['ffmpeg','-v','error','-n','-i',source,'-af','atrim=start=48:end=60,asetpts=PTS-STARTPTS','-ar','48000','-ac','2','-c:a','pcm_f32le',str(wav)],env=env,check=True)
 records.append({'id':idx,'original_path':source,'original_sha256':sha(source),'excerpt':str(wav),'excerpt_sha256':sha(wav),'arms':{}})
 for name,flags in [('bypass',['--no-auto-eq']),('legacy',[]),('smooth',['--smooth-auto-eq'])]:
  folder=out/f'{idx}-{name}'
  args=['auto-master','--input',str(wav),'--no-sfht','--no-mid','--no-spatial','--no-morphic-gtf','--no-defizz','--no-denoise','--no-export-sdcard','--no-glue','--max-limiter-gr-db','0','--target-lufs','-18','--out',str(folder)]+flags
  subprocess.run([sys.executable,'artifacts/revision-20261109/run_child.py','--receipt',f'artifacts/eq-listening-20261109/{idx}-{name}.json','--timeout','180','--',plan['binary']]+args,check=True)
  records[-1]['arms'][name]={'path':str(folder/'mastered.wav'),'run_record':str(folder/'summary.json'),'sha256':sha(folder/'mastered.wav')}
Path('runs/eq-listening-20261109/render-manifest.private.json').write_text(json.dumps(records,indent=2)+'\n')
