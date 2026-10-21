"""Package the single candidate only after full-song gates complete."""
from pathlib import Path
import json,subprocess,sys,time,hashlib,secrets
ROOT=Path(__file__).resolve().parents[2]
r=ROOT/'runs/listening-coherence-20261020/fullsong-v3'
base=ROOT/'runs/listening-coherence-20261020'
start=time.monotonic()
while not (r/'COMPLETE.json').exists():
 if (r/'HALTED.json').exists():raise SystemExit('Full-song qualification halted; no listening package produced')
 if time.monotonic()-start>1850:raise SystemExit('Waiting timeout; no package produced')
 time.sleep(2)
s=json.loads((r/'COMPLETE.json').read_text());assert s['eligible_for_listening'] is True
private=base/'listening-private-v1';private.mkdir(mode=0o700)
variants=[]
source_commit=subprocess.check_output(['git','log','-1','--format=%H','--','src/gtf.rs'],cwd=ROOT,text=True).strip()
for name,commit in [('control',source_commit),('candidate',source_commit)]:
 variants.append({'label':name,'path':str(r/name/'mastered.wav'),'provenance':{'run_record':str(r/'COMPLETE.json'),'git_commit':commit,'source_sha256':s['input_sha256'],'target_lufs':-13.0,'qualification':'Exact matched new control; experimentalcombinedintervention; no ownerjudgment or promotion. Owner has seen mechanismmetrics, so this is source-assignment blinding, not naive-expectation blinding.'}})
m={'version':1,'mode':'equal_target_architecture','seed':secrets.randbits(63),'target_lufs':-23.0,'tracks':[{'id':'track1','variants':variants,'excerpts':[{'id':'fixed_short','kind':'diagnostic','start_sample':2304000,'frames':1440000},{'id':'full_song','kind':'long','start_sample':0,'frames':6739200}]}]}
p=private/'manifest.json';p.write_text(json.dumps(m,indent=2)+'\n');p.chmod(0o600)
listener=base/'listen/session1';organizer=private/'organizer'
subprocess.run([sys.executable,str(ROOT/'scripts/blind_listening.py'),'prepare',str(p),'--listener',str(listener),'--organizer',str(organizer)],cwd=ROOT,check=True)
subprocess.run([sys.executable,str(ROOT/'scripts/blind_listening.py'),'verify','--organizer',str(organizer)],cwd=ROOT,check=True)
with (base/'LISTENING_READY.json').open('x') as f:json.dump({'schema':'one-candidate-listening-ready-v1','listener_dir':str(listener),'instructions':str(listener/'INSTRUCTIONS.txt'),'responses':str(listener/'responses.csv'),'candidate_count':1,'full_song_complete_sha256':hashlib.sha256((r/'COMPLETE.json').read_bytes()).hexdigest(),'owner_judgments_received':False,'production_promoted':False},f,indent=2);f.write('\n')
