"""Run one bounded sequential child, retain command, output and completion status."""
import argparse, hashlib, json, os, subprocess, time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--receipt',required=True);p.add_argument('--timeout',type=int,default=600);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
out=Path(a.receipt);out.parent.mkdir(parents=True,exist_ok=True)
if out.exists(): raise SystemExit('receipt exists')
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
binary=Path(cmd[0]);sha=hashlib.sha256(binary.read_bytes()).hexdigest()
env=dict(os.environ,LD_PRELOAD='/data/data/com.termux/files/usr/lib/libc++_shared.so',OPENBLAS_NUM_THREADS='1')
receipt={'command':cmd,'binary_sha256':sha,'timeout_seconds':a.timeout,'scope':'correctness/training pilot; elapsed is not benchmark','complete':False}
out.write_text(json.dumps(receipt,indent=2))
with out.with_suffix('.log').open('x') as log:
 try:
  result=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
  receipt.update(complete=True,returncode=result.returncode)
 except subprocess.TimeoutExpired:
  receipt.update(stopped='timeout',returncode=124)
out.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt));raise SystemExit(receipt['returncode'])
