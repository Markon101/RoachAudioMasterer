"""Fixed bounded procedural signal; source independent of all treatment knobs."""
import math, random, struct, wave
from pathlib import Path
out = Path('runs/revision-20261109/fixture.wav')
if out.exists(): raise SystemExit('refusing overwrite')
rng = random.Random(9112026)
rate=48000
with wave.open(str(out), 'wb') as w:
 w.setparams((2, 2, rate, 0, 'NONE', 'not compressed'))
 for i in range(4*rate):
  t=i/rate
  env=(1-math.exp(-t*30))*min(1,(4-t)*4)
  burst=math.exp(-((t%0.5)*25))
  common=env*(0.13*math.sin(2*math.pi*220*t)+0.07*math.sin(2*math.pi*1100*t)+0.025*math.sin(2*math.pi*9200*t)+0.025*burst*rng.uniform(-1,1))
  side=env*0.01*math.sin(2*math.pi*7300*t)
  w.writeframesraw(struct.pack('<hh', int((common+side)*32767), int((common-side)*32767)))
