"""Execute the declared CPU pilot sequentially with a pinned executable."""
import subprocess, sys
from pathlib import Path
binary=sys.argv[1]
root='runs/revision-20261109/'
def run(name,args):
 subprocess.run([sys.executable,'artifacts/revision-20261109/run_child.py','--receipt',f'artifacts/revision-20261109/{name}.json','--timeout','600','--',binary]+args,check=True)
common=['auto-master','--input',root+'fixture.wav','--no-sfht','--no-mid','--no-spatial','--no-morphic-gtf','--no-defizz','--no-export-sdcard','--kick-transient-guard','--target-lufs','-18']
for name,flags in [('candidate-legacy',[]),('candidate-smooth',['--smooth-auto-eq']),('candidate-highzero',['--high-model','absent.json','--high-strength','0']),('candidate-high',['--high-model','artifacts/rich-field-v1/basis.json','--high-strength','0.5','--chunk-seconds','2','--overlap-seconds','0.5'])]:
 run(name,common+flags+['--out',root+name])
run('candidate-training',['rich-mid-train','--kind','basis','--steps','100','--seed','9100100','--damage-seed','9200100','--backend','cpu','--out',root+'candidate-training'])
for name,model in [('eval-legacy',root+'baseline-training/basis.json'),('eval-independent',root+'candidate-training/basis.json')]:
 run(name,['rich-mid-evaluate','--model',model,'--seed','9300100','--damage-seed','9400100','--count','12','--steps','8','--mid-ceiling','8000','--backend','cpu','--out',root+name])
# Transfer controls use the same unseen sources with the historical paired damage.
for name,model in [('eval-legacy-paired',root+'baseline-training/basis.json'),('eval-independent-paired',root+'candidate-training/basis.json')]:
 run(name,['rich-mid-evaluate','--model',model,'--seed','9300100','--count','12','--steps','8','--mid-ceiling','8000','--backend','cpu','--out',root+name])
