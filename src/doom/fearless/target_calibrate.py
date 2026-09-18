"""Engineer-facing calibration of the artificial target-direction channel.

This does not choose Doom actions. It selects, before gameplay, the fixed
LC4/LPLC2 current amplitude and pulse length whose MaleCNS readouts best
separate left, centre and right target cues.
"""
import json, itertools
from pathlib import Path
import numpy as np
from doom.fearless.danger import sided_populations
from doom.fearless.run import GRAPH, TARGET_MANIFEST, atomic_json
from doom.native import NativeBrain

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/fearless/aggressive-mapper-v6-calibration';TK=np.array([8017,8196,36409,134363,136053])
READOUT=np.array([48,146,489,142493],dtype=np.int64); DIRECTIONS=('left','center','right')

def run():
 target=json.loads(TARGET_MANIFEST.read_text());sided=sided_populations(target);rows=[]
 for amp,pulse in itertools.product((10.,20.,40.),(1,3,5)):
  vectors=[]
  for direction in DIRECTIONS:
   b=NativeBrain(GRAPH);total=np.zeros(b.n,np.int64)
   sides=('L','R') if direction=='center' else (('R',) if direction=='left' else ('L',))
   ix=np.unique(np.concatenate([sided[n][s] for n in ('LC4','LPLC2') for s in sides]))
   for tick in range(12):
    drive=np.zeros(b.n,np.float32);drive[ix]=amp if tick<pulse else 0.;c,_=b.step(np.zeros(len(b.retina),np.float32),100/35,extra_drive=drive if tick<pulse else None);total+=c
   vectors.append(total[READOUT].astype(float));
  between=float(np.mean(np.var(np.asarray(vectors),axis=0)));active=float(np.mean([np.sum(v)>0 for v in vectors]));score=between+active
  rows.append({'amplitude':amp,'pulse_ticks':pulse,'readout_vectors':np.asarray(vectors).tolist(),'between_direction_variance':between,'readout_active_fraction':active,'score':score})
 winner=max(rows,key=lambda r:(r['score'],-r['amplitude'], -r['pulse_ticks']))
 result={'schema':1,'status':'calibrated','directions':DIRECTIONS,'candidates':rows,'winner':winner,'targets':'LC4+LPLC2 sided populations','selection':'readout separability and activity only; no Doom button or gameplay score used'}
 atomic_json(OUT/'calibration.json',result);return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
