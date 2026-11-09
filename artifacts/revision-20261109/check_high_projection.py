import json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path.cwd()))
from scripts.matched_attack_audit import WavReader
root=Path('runs/revision-20261109')
def read(p):
 r=WavReader(p)
 return np.concatenate([r.read(i,min(i+65536,r.frames)) for i in range(0,r.frames,65536)])
a=read(root/'fixture.wav');b=read(root/'candidate-high/stage2_5_high.wav')
f=np.fft.rfftfreq(len(a),1/48000);delta=np.fft.rfft(b-a,axis=0)
low=np.fft.irfft(delta*(f<=6000)[:,None],n=len(a),axis=0)
result={'low_band_error_rms':float(np.sqrt(np.mean(low**2))),'residual_rms':float(np.sqrt(np.mean((b-a)**2))),'geometry_equal':a.shape==b.shape,'all_finite':bool(np.isfinite(b).all()),'scope':'whole-track Fourier projection of source-to-high-stage residual on the declared procedural fixture, before downstream processing'}
assert result['all_finite'] and result['geometry_equal'] and result['low_band_error_rms']<1e-6
Path('artifacts/revision-20261109/high-projection.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
