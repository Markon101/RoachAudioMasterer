import importlib.util,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()/'scripts'))
import blind_listening as blind
from matched_attack_audit import WavReader
root=Path('runs/eq-listening-20261109')
def read(p):
 r=WavReader(p);x=np.concatenate([r.read(i,min(i+65536,r.frames)) for i in range(0,r.frames,65536)])
 if (r.rate,r.channels,r.frames)!=(48000,2,576000):raise ValueError('geometry gate')
 return x
records=json.loads((root/'render-manifest.private.json').read_text());report=[]
for item in records:
 ref=read(root/f"{item['id']}-bypass/mastered.wav")
 for name,arm in item['arms'].items():
  x=read(arm['path']);m=blind.measure(Path(arm['path']));native=json.loads(Path(arm['run_record']).read_text())['master_stats']['after']
  assert abs(m['lufs']+18)<=.15 and m['true_peak_dbfs']<=-.99 and native['true_peak_dbtp']<=-.99
  freq=np.fft.rfftfreq(len(x),1/48000); mask=(freq>=800)&(freq<=2500)
  a=np.fft.rfft(ref[:,0]);b=np.fft.rfft(x[:,0]);corr=np.fft.irfft(a.conj()*b*mask,n=len(x));lags=np.arange(-24,25)
  lag=int(lags[np.argmax(corr[lags%len(x)])]);assert lag==0,'timing gate'
  report.append({'track':item['id'],'arm':name,'independent_meter':m,'native_meter':native,'relative_band_timing_lag_samples':lag,'lr_correlation':float(np.corrcoef(x.T)[0,1]),'scope':'local 800..2500Hz timing proxy, not sub-sample or perceptual proof'})
Path('artifacts/eq-listening-20261109/qualification.json').write_text(json.dumps({'eligible_numeric_checks':True,'records':report,'note':'Requires separate corrected EQ test suite pass before listener delivery'},indent=2)+'\n')
print('Nine same-geometry render checks passed')
