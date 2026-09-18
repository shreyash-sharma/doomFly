"""Two-seed smoke for a pC2l-only engineered pursuit embodiment mapping."""
import json
from pathlib import Path
import numpy as np
from doom.engine import NeuralControls
from doom.fearless.danger import packet_from_scene
from doom.fearless.run import GRAPH, MODEL_MANIFEST, TICKS_PER_SECOND, atomic_json
from doom.game import Game, retinal_samples
from doom.native import NativeBrain
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/fearless/aggression/adapter-smoke'
TK=np.array([8017,8196,36409,134363,136053]);RANDOM=np.array([126563,136090,110303,126140,7437]);PC2L=np.array([2734,3472])
CONDS=('original_bci','tk_original_bci','tk_pc2l_adapter','random_pc2l_adapter','scripted_motor_positive')
def adapter(counts,seconds):
 # Frozen engineering scale copied from the original forward BCI gain (0.4).
 return {'turn':0.,'forward':float(np.clip(counts[PC2L].sum()/seconds*.4,0,20)),'attack':False}
def motor(scene):
 # Explicit non-biological policy, analytically separate.
 p=scene['player']; enemies=[x for x in scene['objects'] if x['name'] in ('DoomFlyImp','Zombieman')]
 if not enemies:return {'turn':0.,'forward':20.,'attack':True}
 e=min(enemies,key=lambda x:(x['x']-p['x'])**2+(x['y']-p['y'])**2);import math
 a=(math.degrees(math.atan2(e['y']-p['y'],e['x']-p['x']))-p['angle']+180)%360-180
 return {'turn':float(np.clip(a*.15,-6,6)),'forward':20.,'attack':True}
def one(seed,condition,cap=5):
 b=NativeBrain(GRAPH); c=NeuralControls(json.loads(MODEL_MANIFEST.read_text())['readouts'],mode='bci');g=Game(seed=seed,scenario='combat_survival',spectator=True);old={};h=g.observation()['health'];ev=[]
 try:
  for tick in range(1,int(cap*TICKS_PER_SECOND)+1):
   before=g.observation();scene=g.spectator();truth=packet_from_scene(scene,old,max(0,h-before['health']));old=truth.pop('distances');h=before['health']; inds=TK if condition in ('tk_original_bci','tk_pc2l_adapter') else RANDOM
   amp=10*truth['intensity'] if condition in ('tk_original_bci','tk_pc2l_adapter','random_pc2l_adapter') and truth['active'] else 0.;d=None
   if amp:d=np.zeros(b.n,np.float32);d[inds]=amp
   target=round(tick*10000/TICKS_PER_SECOND);steps=target-b.cursor;counts,_=b.step(retinal_samples(g.pixels(),b.uv),steps*b.dt,extra_drive=d)
   if condition=='scripted_motor_positive':action=motor(scene)
   elif condition.endswith('pc2l_adapter'):action=adapter(counts,steps*b.dt/1000)
   else:full=c.decode(counts,steps*b.dt/1000);action={k:full[k] for k in ('turn','forward','attack')}
   g.act(action);ev.append({'tk_or_random_spikes':int(counts[inds].sum()) if amp else 0,'pc2l_spikes':int(counts[PC2L].sum()),'action':action,'game':g.observation()})
   if g.observation()['finished']:break
 finally:g.close()
 return {'seed':seed,'condition':condition,'metrics':{'ticks':len(ev),'stim_spikes':sum(x['tk_or_random_spikes'] for x in ev),'pc2l_spikes':sum(x['pc2l_spikes'] for x in ev),'attack_ticks':sum(x['action']['attack'] for x in ev),'mean_forward':float(np.mean([x['action']['forward'] for x in ev])),'survival':len(ev)/TICKS_PER_SECOND,'damage':100-ev[-1]['game']['health'],'kills':ev[-1]['game']['kills']},'events':ev}
def run(output=OUT,seeds=(513004,513005)):
 rows=[one(s,c) for s in seeds for c in CONDS];base={(r['seed']):r for r in rows if r['condition']=='original_bci'}
 # Stop gate: adapter must increase pC2l-derived forward behavior versus matched random;
 # it is not permitted to invent firing without a supported strike readout.
 checks={'tk_pc2l_over_random':all(next(r for r in rows if r['seed']==s and r['condition']=='tk_pc2l_adapter')['metrics']['pc2l_spikes']>next(r for r in rows if r['seed']==s and r['condition']=='random_pc2l_adapter')['metrics']['pc2l_spikes'] for s in seeds),'no_attack_mapping':all(r['metrics']['attack_ticks']==0 for r in rows if r['condition'].endswith('pc2l_adapter'))}
 result={'schema':1,'conditions':CONDS,'seeds':list(seeds),'rows':rows,'checks':checks,'passed':all(checks.values()),'adapter':'engineered pC2l pursuit activity -> forward only; no target state enters adapter; no strike readout exists','motor_positive':'direct scripted enemy-oriented forward+attack policy; non-biological and separate'};atomic_json(Path(output)/'report.json',result);return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
