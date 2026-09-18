"""Open-loop validation of the direct Tk-FruM → pC2l pursuit candidate."""
import json
import numpy as np
from doom.fearless.run import ROOT, GRAPH, atomic_json
from doom.native import NativeBrain

OUT=ROOT/'outputs/fearless/aggression/pathway.json'
TK=np.array([8017,8196,36409,134363,136053]); PC2L=np.array([2734,3472])
# Deterministic, unrelated, ACh cb-intrinsic control: 3 L/2 R, excluding all
# Tk/P1/pC1/pC2/pCd/pIP annotation synonyms. Frozen before this response test.
RANDOM=np.array([126563,136090,110303,126140,7437])
FPS=35

def simulate(indices=None,schedule=None):
 b=NativeBrain(GRAPH); blank=np.zeros(len(b.retina),np.float32); totals=np.zeros(b.n,np.int64)
 schedule=np.zeros(FPS,dtype=np.float32) if schedule is None else np.asarray(schedule,dtype=np.float32)
 for tick,amp in enumerate(schedule,1):
  target=round(tick*10000/FPS); drive=None
  if indices is not None and amp:
   drive=np.zeros(b.n,np.float32);drive[indices]=amp
  c,_=b.step(blank,(target-b.cursor)*b.dt,extra_drive=drive);totals+=c
 return {'stimulated_spikes':int(totals[indices].sum()) if indices is not None else 0,
         'pc2l_spikes':int(totals[PC2L].sum()),
         'dnp20_spikes':int(totals[[48,146]].sum()),'dnpe017_spikes':int(totals[[489,142493]].sum())}

def run(output=OUT):
 # Four five-tick pulses; shuffled control preserves duration and amplitude.
 pulse=np.array(([10]*5+[0]*5)*3+[10]*5,dtype=np.float32)
 shuffled=pulse[np.random.default_rng(20260919).permutation(len(pulse))]
 rows={'baseline':simulate(), 'tk_frum_pulse':simulate(TK,pulse),
       'random_matched_pulse':simulate(RANDOM,pulse), 'tk_frum_temporally_shuffled':simulate(TK,shuffled)}
 checks={'direct_pc2l_annotation':True,'tk_over_baseline':rows['tk_frum_pulse']['pc2l_spikes']>rows['baseline']['pc2l_spikes'],
   'tk_over_random':rows['tk_frum_pulse']['pc2l_spikes']>rows['random_matched_pulse']['pc2l_spikes'],
   'target_spikes':rows['tk_frum_pulse']['stimulated_spikes']>0}
 result={'schema':1,'candidate':{'name':'pC2l pursuit-associated direct target subset','graph_indices':PC2L.tolist(),
   'body_ids':[12922,13723],'annotation':'Nojima 2021: pC2l; direct Tk-FruM targets',
   'literature':'Kohatsu et al., Nature Communications 2015: pC2l priming supports courtship-like following pursuit; not an aggression-specific output.'},
   'tk_frum_indices':TK.tolist(),'matched_random_indices':RANDOM.tolist(),'pulse_schedule':pulse.tolist(),'shuffled_schedule':shuffled.tolist(),'rows':rows,'checks':checks,
   'passed':all(checks.values()),'constraint':'Passing validates an engineered pursuit-associated neural readout only; it does not validate natural aggression or side-to-turn semantics.'}
 atomic_json(output,result);return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
