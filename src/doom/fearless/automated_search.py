"""Budgeted automated search over small Doom BCIs with a sealed final split.

Game telemetry may create the declared LC4/LPLC2 current and delayed rewards.
For neural systems it never enters ``SearchDecoder.choose`` or its feature input.
The direct-sensory and scripted systems are explicit non-neural controls.
"""
from __future__ import annotations

import argparse, hashlib, json, math
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from doom.fearless.danger import packet_from_scene, sided_populations, threat_drive
from doom.fearless.run import GRAPH, MODEL_MANIFEST, TARGET_MANIFEST, TICKS_PER_SECOND, atomic_json, file_sha256
from doom.game import Game, retinal_samples
from doom.native import NativeBrain

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/fearless/automated-search-v2'
TRAIN=tuple(range(771001,771005)); DEV=tuple(range(772001,772003)); FINAL=tuple(range(773001,773005))
SYSTEMS=('intact','rewired','temporal_shuffle','random_recurrent','direct_sensory')
CAP=5; N_CONFIGS=12; SEARCH_SEED=20260916
MACROS={
 'full':(('noop',0,0,0),('advance',0,20,0),('retreat',0,-20,0),('left',-6,0,0),('right',6,0,0),('attack',0,0,1),('advance_attack',0,20,1),('retreat_attack',0,-20,1)),
 'compact':(('noop',0,0,0),('advance',0,20,0),('left',-6,0,0),('right',6,0,0),('attack',0,0,1),('advance_attack',0,20,1))}

def sha(value): return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
def softmax(x):
 z=np.asarray(x,dtype=np.float64)-np.max(x);q=np.exp(np.clip(z,-30,30));return q/q.sum()

def configurations():
 """Seed-fixed automated design; independent of every episode outcome."""
 rng=np.random.default_rng(SEARCH_SEED); result=[]
 choices={'feature_set':('motor','populations','combined','random14'),'window_ms':(250,500,750,1000),
  'pooling':('counts','difference','mean_max'),'architecture':('linear','rnn8'),'l2':(0.0,1e-4,1e-3),
  'threshold':(0.0,.20,.30),'threat_amplitude':(0.0,10.0,20.0),'vocabulary':('full','compact'),
  'learning_rate':(.005,.02,.05),'kill_reward':(6.,8.,10.),'damage_penalty':(.1,.2,.3),
  'ammo_penalty':(.01,.03,.05),'inactivity_penalty':(.001,.003,.01)}
 for i in range(N_CONFIGS):
  c={'config_id':f'cfg-{i:02d}'}
  for key,values in choices.items(): c[key]=values[int(rng.integers(len(values)))]
  result.append(c)
 return result

def feature_groups(brain):
 model=json.loads(MODEL_MANIFEST.read_text());target=json.loads(TARGET_MANIFEST.read_text())
 motor=[np.asarray([r['index']],dtype=np.int64) for r in model['readouts']]
 pops=[]
 for name in ('LC4','LPLC2','DNp09','DNp01','DNp20','DNpe017'):
  pops.append(np.asarray([r['graph_index'] for r in target['populations'][name]['neurons']],dtype=np.int64))
 rng=np.random.default_rng(88421);random=[np.asarray([i],dtype=np.int64) for i in rng.choice(brain.n,14,replace=False)]
 return {'motor':motor,'populations':pops,'combined':motor+pops,'random14':random}

def rewire(brain):
 """Directed configuration-model control: exact in/out degrees, same weights."""
 rng=np.random.default_rng(99173); original=brain.post.copy(); perm=rng.permutation(len(original)); brain.post[:]=original[perm]
 if not np.array_equal(np.sort(brain.post),np.sort(original)):raise AssertionError('in-degree changed')
 return {'method':'global target-stub permutation','post_before_sha256':sha(original),'post_after_sha256':sha(brain.post),
         'exact_out_degree':True,'exact_in_degree':True,'allows_parallel_and_self_edges':True}

class TemporalFeatures:
 def __init__(self,groups,window_ms,pooling,shuffle=False):
  self.groups=groups;self.length=max(1,round(window_ms*TICKS_PER_SECOND/1000));self.pooling=pooling;self.shuffle=shuffle;self.q=deque(maxlen=self.length)
 def push_values(self,x,tick):
  self.q.append(np.clip(np.asarray(x,dtype=np.float64),0,5));a=np.asarray(self.q)
  if self.shuffle and len(a)>1:
   # Independent causal lags destroy cross-population temporal alignment and
   # affect even sum/mean pooling; no future activity is accessed.
   rng=np.random.default_rng(600000+tick);a=np.asarray([[a[int(rng.integers(len(a))),j] for j in range(a.shape[1])]])
  if self.pooling=='counts':return a.sum(axis=0)/self.length
  if self.pooling=='difference':return np.r_[a.mean(axis=0),a[-1]-a[0]]
  return np.r_[a.mean(axis=0),a.max(axis=0)]
 def push(self,counts,seconds,tick):
  x=np.asarray([counts[g].sum()/max(1,len(g))/seconds/100 for g in self.groups]);return self.push_values(x,tick)

@dataclass
class SearchDecoder:
 w:np.ndarray; recurrent:np.ndarray|None; state:np.ndarray; vocabulary:tuple; threshold:float; l2:float
 @classmethod
 def create(cls,n,config,seed):
  rng=np.random.default_rng(seed); hidden=8 if config['architecture']=='rnn8' else 0
  if hidden:
   recurrent=rng.normal(0,.18,(hidden,hidden+n+1));w=rng.normal(0,.03,(len(MACROS[config['vocabulary']]),hidden+1));state=np.zeros(hidden)
  else:recurrent=None;w=rng.normal(0,.03,(len(MACROS[config['vocabulary']]),n+1));state=np.empty(0)
  return cls(w,recurrent,state,MACROS[config['vocabulary']],config['threshold'],config['l2'])
 def vector(self,x):
  x=np.asarray(x,dtype=np.float64)
  if self.recurrent is not None:self.state=np.tanh(self.recurrent@np.r_[1.,x,self.state])
  return np.r_[1.,self.state if self.recurrent is not None else x]
 def choose(self,x,rng,deterministic=False):
  v=self.vector(x);p=softmax(self.w@v)
  if deterministic:
   i=int(np.argmax(p));i=0 if p[i]<self.threshold else i
  else:i=int(rng.choice(len(p),p=p))
  name,turn,forward,attack=self.vocabulary[i]
  return {'turn':float(turn),'forward':float(forward),'attack':bool(attack)},i,p,v,name
 def update(self,trajectory,lr,gamma=.99):
  returns=np.zeros(len(trajectory));running=0.
  for i in range(len(trajectory)-1,-1,-1):running=trajectory[i]['reward']+gamma*running;returns[i]=running
  advantage=returns-returns.mean();grad=np.zeros_like(self.w)
  for t,a in zip(trajectory,advantage):
   one=np.zeros(len(t['p']));one[t['i']]=1;grad+=a*np.outer(one-t['p'],t['v'])
  grad/=max(1,len(trajectory));grad-=self.l2*self.w
  norm=np.linalg.norm(grad)
  if norm>5:grad*=5/norm
  self.w+=lr*grad

def sensory_vector(light,n,packet):
 base=np.asarray(light if len(light)==n else [light[i] for i in np.linspace(0,len(light)-1,n,dtype=int)],dtype=np.float64)
 # Direct-sensory control receives equal declared threat information explicitly.
 if n>=3:base[:3]=[packet['intensity'],packet['direction']=='left',packet['direction']=='right']
 return base

def train_reward(before,after,engine,action,c):
 kills=max(0,after['kills']-before['kills']);damage=max(0,before['health']-after['health'])
 waste=action['attack'] and after['ammo']==before['ammo'] and engine<=0 and kills==0
 noop=action['turn']==0 and action['forward']==0 and not action['attack']
 return c['kill_reward']*kills+max(0,engine)+.02-c['damage_penalty']*damage-c['ammo_penalty']*waste-c['inactivity_penalty']*noop

def objective(events):
 last=events[-1]['game'];damage=sum(max(0,e['before']['health']-e['game']['health']) for e in events)
 waste=sum(e['action']['attack'] and e['game']['ammo']==e['before']['ammo'] and e['engine_reward']<=0 and e['game']['kills']==e['before']['kills'] for e in events)
 noop=sum(e['macro']=='noop' for e in events);positive=sum(max(0,e['engine_reward']) for e in events)
 score=8*last['kills']+positive+.02*len(events)-.2*damage-.03*waste-.003*noop
 return {'score':float(score),'kills':int(last['kills']),'positive_engine_reward':float(positive),'ticks':len(events),'damage_received':int(damage),'ammo_waste_ticks':int(waste),'noop_ticks':int(noop),'survived':not last['finished']}

def episode(system,seed,config,decoder,learn,run_seed):
 decoder.state.fill(0)
 brain=NativeBrain(GRAPH); provenance={}
 if system=='rewired':provenance['rewire']=rewire(brain)
 groups=feature_groups(brain)[config['feature_set']]; history=TemporalFeatures(groups,config['window_ms'],config['pooling'],system=='temporal_shuffle')
 control_state=np.zeros(len(groups));control_rng=np.random.default_rng(9182)
 control_matrix=control_rng.normal(0,.24,(len(groups),2*len(groups)+1))
 game=Game(seed=seed,scenario='combat_survival',spectator=True);target=json.loads(TARGET_MANIFEST.read_text());sided=sided_populations(target)
 rng=np.random.default_rng(run_seed);old={};health=game.observation()['health'];events=[];trajectory=[]
 try:
  for tick in range(1,CAP*TICKS_PER_SECOND+1):
   before=game.observation();scene=game.spectator();packet=packet_from_scene(scene,old,max(0,health-before['health']));old=packet.pop('distances');health=before['health']
   target_step=round(tick*10000/TICKS_PER_SECOND);steps=target_step-brain.cursor
   if system in ('direct_sensory','random_recurrent'):
    sample_indices=np.linspace(0,len(brain.uv)-1,len(groups),dtype=int)
    light=retinal_samples(game.pixels(),brain.uv[sample_indices])
   else:light=retinal_samples(game.pixels(),brain.uv)
   drive=np.zeros(brain.n,np.float32)
   if config['threat_amplitude'] and system not in ('random_recurrent','direct_sensory'):
    ix,val=threat_drive(packet,sided,config['threat_amplitude']);drive[ix]+=val
   seconds=steps*brain.dt/1000
   if system=='direct_sensory':
    # Explicit control: no MaleCNS computation is needed for its action input.
    brain.cursor=target_step;x=history.push_values(sensory_vector(light,len(groups),packet),tick)
   elif system=='random_recurrent':
    # Matched non-connectome control driven by the same declared sensory vector.
    brain.cursor=target_step;sensory=sensory_vector(light,len(groups),packet)
    control_state=np.tanh(control_matrix@np.r_[1.,sensory,control_state]);x=history.push_values(control_state,tick)
   else:
    counts,_=brain.step(light,steps*brain.dt,extra_drive=drive if np.any(drive) else None);x=history.push(counts,seconds,tick)
   action,i,p,v,macro=decoder.choose(x,rng,deterministic=not learn);engine=game.act(action);after=game.observation()
   reward=train_reward(before,after,engine,action,config);trajectory.append({'reward':reward,'i':i,'p':p,'v':v})
   events.append({'tick':tick,'before':before,'game':after,'action':action,'macro':macro,'engine_reward':engine,'train_reward':reward,'neural_feature':np.asarray(x).round(6).tolist(),'danger_packet':packet})
   if after['finished']:break
 finally:game.close()
 if learn:decoder.update(trajectory,config['learning_rate'])
 return {'seed':seed,'system':system,'config_id':config['config_id'],'learn':learn,'events':events,'objective':objective(events),'provenance':provenance}

def input_size(system,config):
 b=NativeBrain(GRAPH);n=len(feature_groups(b)[config['feature_set']]);
 return n*(2 if config['pooling'] in ('difference','mean_max') else 1)

def save_checkpoint(path,decoder,config,system):
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,w=decoder.w,recurrent=np.asarray([]) if decoder.recurrent is None else decoder.recurrent)
 meta={'system':system,'config':config,'checkpoint_sha256':file_sha256(path),'weights_sha256':sha(decoder.w)};atomic_json(Path(str(path)+'.json'),meta);return meta

def load_winner(system,output=OUT):
 selection=Path(output)/'search'/system/'selection.json';data=json.loads(selection.read_text());winner=data['winner'];c=winner['config']
 path=Path(output)/'search'/system/f'{c["config_id"]}.npz';saved=np.load(path,allow_pickle=False);recurrent=saved['recurrent']
 decoder=SearchDecoder(saved['w'].copy(),None if recurrent.size==0 else recurrent.copy(),
  np.zeros(8) if recurrent.size else np.empty(0),MACROS[c['vocabulary']],c['threshold'],c['l2'])
 if file_sha256(path)!=winner['checkpoint']['checkpoint_sha256']:raise ValueError(f'{system} winner checkpoint hash changed')
 return c,decoder,path,data

def scripted_episode(seed):
 game=Game(seed=seed,scenario='combat_survival',spectator=True);events=[]
 try:
  for tick in range(1,CAP*TICKS_PER_SECOND+1):
   before=game.observation();scene=game.spectator();p=scene['player'];enemies=[o for o in scene['objects'] if o['name'] in ('DoomFlyImp','Zombieman')]
   if enemies:
    q=min(enemies,key=lambda o:math.hypot(o['x']-p['x'],o['y']-p['y']));angle=(math.degrees(math.atan2(q['y']-p['y'],q['x']-p['x']))-p['angle']+180)%360-180
    action={'turn':float(np.clip(angle*.15,-6,6)),'forward':20.,'attack':abs(angle)<18};macro='scripted_geometry'
   else:action={'turn':0.,'forward':20.,'attack':False};macro='scripted_search'
   engine=game.act(action);after=game.observation();events.append({'tick':tick,'before':before,'game':after,'action':action,'macro':macro,'engine_reward':engine})
   if after['finished']:break
 finally:game.close()
 return {'seed':seed,'system':'scripted_benchmark','events':events,'objective':objective(events),'non_biological_direct_policy':True}

def final_once(output=OUT):
 output=Path(output);root=output/'final-test';ledger=root/'ledger.json'
 if ledger.exists():
  existing=json.loads(ledger.read_text())
  if existing['status']=='complete':return existing
  raise RuntimeError('Final evaluation was already opened and is incomplete; automatic rerun forbidden')
 winners={}
 for system in SYSTEMS:
  c,d,p,s=load_winner(system,output);winners[system]={'config_id':c['config_id'],'checkpoint_sha256':file_sha256(p)}
 precommit={'schema':1,'status':'in-progress-no-rerun','final_seeds':FINAL,'winners':winners,'rule':'This ledger was written before opening the first final episode.'}
 atomic_json(ledger,precommit);records=[]
 for system in SYSTEMS:
  c,decoder,path,selection=load_winner(system,output)
  for seed in FINAL:
   rec=episode(system,seed,c,decoder,False,9000000+seed);atomic_json(root/f'{system}-{seed}.json',rec);records.append({'system':system,'seed':seed,'objective':rec['objective']})
 for seed in FINAL:
  rec=scripted_episode(seed);atomic_json(root/f'scripted_benchmark-{seed}.json',rec);records.append({'system':'scripted_benchmark','seed':seed,'objective':rec['objective']})
 done={**precommit,'status':'complete','records':records};atomic_json(ledger,done);return done

def search_system(system,output=OUT):
 if system not in SYSTEMS:raise ValueError(system)
 root=Path(output)/'search'/system;root.mkdir(parents=True,exist_ok=True)
 selection=root/'selection.json'
 if selection.exists():return json.loads(selection.read_text())
 partial=root/'attempts.partial.json'
 attempts=json.loads(partial.read_text())['attempts'] if partial.exists() else []
 completed={a['config']['config_id'] for a in attempts}
 for ci,c in enumerate(configurations()):
  if c['config_id'] in completed:continue
  decoder=SearchDecoder.create(input_size(system,c),c,SEARCH_SEED+ci)
  curves=[]
  for epoch in range(2):
   for seed in TRAIN:
    rec=episode(system,seed,c,decoder,True,1000000*ci+10000*epoch+seed)
    atomic_json(root/f'{c["config_id"]}-train-e{epoch}-{seed}.json',rec)
    curves.append({'epoch':epoch,'seed':seed,'objective':rec['objective']})
  dev=[]
  for seed in DEV:
   rec=episode(system,seed,c,decoder,False,2000000*ci+seed);atomic_json(root/f'{c["config_id"]}-dev-{seed}.json',rec);dev.append(rec['objective'])
  checkpoint=save_checkpoint(root/f'{c["config_id"]}.npz',decoder,c,system)
  attempt={'config':c,'train_curve':curves,'development':dev,'development_mean_score':float(np.mean([x['score'] for x in dev])),
   'development_mean_damage':float(np.mean([x['damage_received'] for x in dev])),'checkpoint':checkpoint}
  attempts.append(attempt);atomic_json(root/'attempts.partial.json',{'status':'running','attempts':attempts})
 winner=min(attempts,key=lambda a:(-a['development_mean_score'],a['development_mean_damage'],a['config']['config_id']))
 result={'schema':1,'status':'development-winner-frozen','system':system,'train_seeds':TRAIN,'development_seeds':DEV,
  'final_seeds_accessed':False,'episode_budget_used':N_CONFIGS*(8+2),'attempts':attempts,'winner':winner}
 atomic_json(root/'selection.json',result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('command',choices=('search','final'));p.add_argument('--system',choices=SYSTEMS);a=p.parse_args()
 if a.command=='search':
  if not a.system:p.error('--system is required for search')
  result=search_system(a.system)
 else:
  if a.system:p.error('--system is not accepted for final')
  result=final_once()
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
