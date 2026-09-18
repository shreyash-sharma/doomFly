"""Small preregistered stop-gate for Tk-FruM artificial aggression stimulation."""
import json, math
from pathlib import Path
import numpy as np
from doom.engine import NeuralControls
from doom.fearless.danger import packet_from_scene
from doom.fearless.run import GRAPH, MODEL_MANIFEST, TICKS_PER_SECOND, atomic_json, population_event
from doom.game import Game, retinal_samples
from doom.native import NativeBrain

ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'outputs/fearless/aggression/smoke'
CONDS=('normal','aggressive_constant','doom_guy_event'); IDS=np.array([8017,8196,36409,134363,136053],dtype=np.int64)
READOUT={'DNp20':np.array([48,146]),'DNpe017':np.array([489,142493])}

def one(seed,condition,cap=5):
    b=NativeBrain(GRAPH); model=json.loads(MODEL_MANIFEST.read_text()); controls=NeuralControls(model['readouts'],mode='bci'); g=Game(seed=seed,scenario='combat_survival',spectator=True)
    events=[]; old={}; health=g.observation()['health']
    try:
      for tick in range(1,int(cap*TICKS_PER_SECOND)+1):
        before=g.observation(); scene=g.spectator(); truth=packet_from_scene(scene,old,max(0,health-before['health'])); old=truth.pop('distances');health=before['health']
        amp=10.0 if condition=='aggressive_constant' else (10.0*truth['intensity'] if condition=='doom_guy_event' and truth['active'] else 0.0)
        drive=np.zeros(b.n,np.float32); drive[IDS]=amp
        target=round(tick*10000/TICKS_PER_SECOND); steps=target-b.cursor; counts,_=b.step(retinal_samples(g.pixels(),b.uv),steps*b.dt,extra_drive=drive if amp else None)
        decoded=controls.decode(counts,steps*b.dt/1000); action={k:decoded[k] for k in ('turn','forward','attack')};g.act(action)
        events.append({'tick':tick,'amp':amp,'truth':truth,'tk_spikes':int(counts[IDS].sum()),
          'dnp20_spikes':int(counts[READOUT['DNp20']].sum()),'dnpe017_spikes':int(counts[READOUT['DNpe017']].sum()),'action':action,'game':g.observation()})
        if g.observation()['finished']:break
    finally:g.close()
    return {'seed':seed,'condition':condition,'events':events,'metrics':{'ticks':len(events),'tk_spikes':sum(e['tk_spikes'] for e in events),'dnp20_spikes':sum(e['dnp20_spikes'] for e in events),'dnpe017_spikes':sum(e['dnpe017_spikes'] for e in events),'attack_ticks':sum(e['action']['attack'] for e in events),'mean_forward':float(np.mean([e['action']['forward'] for e in events])),'mean_abs_turn':float(np.mean([abs(e['action']['turn']) for e in events])),'survival':len(events)/TICKS_PER_SECOND,'damage':100-events[-1]['game']['health']}}

def run(output=OUT,seeds=(513002,513003)):
 output=Path(output); rows=[one(seed,c) for seed in seeds for c in CONDS]; normal={(r['seed']):r for r in rows if r['condition']=='normal'}
 checks={}
 for c in CONDS[1:]:
  local=[r for r in rows if r['condition']==c]; checks[c+'_tk_active']=all(r['metrics']['tk_spikes']>0 for r in local)
  checks[c+'_decoder_or_behavior_changed']=any(any(r['metrics'][k]!=normal[r['seed']]['metrics'][k] for k in ('dnp20_spikes','dnpe017_spikes','attack_ticks','mean_forward','mean_abs_turn')) for r in local)
 result={'schema':1,'conditions':CONDS,'seeds':list(seeds),'rows':rows,'checks':checks,'passed':all(checks.values()),'stop_rule':'Do not launch cohort if Tk-FruM stimulation fails to change decoder or behavior in this smoke.'};atomic_json(output/'report.json',result);return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
