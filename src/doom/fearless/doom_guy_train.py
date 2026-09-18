"""Focused Tk-FruM Doom controller informed by FlyDoom's actor/reward design."""
from __future__ import annotations
import hashlib,json,math
from collections import deque
from pathlib import Path
import numpy as np

from doom.fearless.aggression_pathway import PC2L,RANDOM,TK
from doom.fearless.danger import packet_from_scene,sided_populations,threat_drive
from doom.fearless.run import GRAPH,MODEL_MANIFEST,TARGET_MANIFEST,TICKS_PER_SECOND,atomic_json,file_sha256
from doom.game import Game,retinal_samples
from doom.native import NativeBrain
from doom.fearless.flydoom_ppo import ClippedPPO

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/fearless/doom-guy-fly-v6-combat-guide'
TRAIN=tuple(range(784401,784404));DEV=tuple(range(785701,785703));CAP=5;ACTION_HOLD=4
# name, turn, forward, strafe, attack. Positive strafe is right.
MACROS=(('noop',0,0,0,0),('advance',0,20,0,0),('retreat',0,-20,0,0),
 ('left',-6,0,0,0),('right',6,0,0,0),('attack',0,0,0,1),
 ('advance_attack',0,20,0,1),('strafe_left_attack',0,0,-20,1),
 ('strafe_right_attack',0,0,20,1),('orbit_left_attack',3,12,-20,1),
 ('orbit_right_attack',-3,12,20,1))

def softmax(z):z=np.asarray(z)-np.max(z);q=np.exp(np.clip(z,-30,30));return q/q.sum()
def groups():
 m=json.loads(MODEL_MANIFEST.read_text());t=json.loads(TARGET_MANIFEST.read_text())
 g=[(r['type']+'-'+r['side']+'-'+r['id'],np.asarray([r['index']],dtype=np.int64)) for r in m['readouts']]
 for name in ('LC4','LPLC2'):
  for side in ('L','R'):
   ix=[r['graph_index'] for r in t['populations'][name]['neurons'] if r['soma_side']==side]
   g.append((name+'-'+side,np.asarray(ix,dtype=np.int64)))
 for name,ix in [('Tk-FruM',TK),('pC2l',PC2L)]:g.append((name,np.asarray(ix,dtype=np.int64)))
 return g

class Features:
 def __init__(self):self.groups=groups();self.q=deque(maxlen=18)
 def push(self,counts,seconds):
  x=np.asarray([counts[ix].sum()/len(ix)/seconds/100 for _,ix in self.groups]);self.q.append(np.clip(x,0,5));a=np.asarray(self.q)
  return np.r_[a.mean(0),a[-1]-a[0]]

class ActorCritic:
 def __init__(self,n,seed=20260916):
  r=np.random.default_rng(seed);self.actor=r.normal(0,.03,(len(MACROS),n+1));self.value=np.zeros(n+1)
  self.ppo=ClippedPPO(self.actor,self.value,seed=seed)
 def choose(self,x,rng,deterministic=False):
  v=np.r_[1.,x];p=softmax(self.actor@v);i=int(np.argmax(p) if deterministic else rng.choice(len(p),p=p));name,turn,forward,strafe,attack=MACROS[i]
  return {'turn':float(turn),'forward':float(forward),'strafe':float(strafe),'attack':bool(attack)},name,i,p,v,float(self.value@v)
 def update(self,traj,gamma=.99,actor_lr=.015,value_lr=.025):
  return self.ppo.update(traj,gamma)

def reward(before,after,engine):
 d={'living':-.01,'kill':10*max(0,after['kills']-before['kills']),'hit':.25*max(0,after['hits']-before['hits']),
  'damage_taken':-.05*max(0,after['health']-before['health'])*0} # overwritten below for explicit sign
 d['damage_taken']=-.05*max(0,before['health']-after['health']);d['death']=-5 if after['finished'] else 0
 d['damage_inflicted']=.05*max(0,after['damage_inflicted']-before['damage_inflicted'])
 spent=max(0,before['ammo']-after['ammo']);landed=max(0,after['hits']-before['hits']) or max(0,after['damage_inflicted']-before['damage_inflicted'])
 d['ammo_spent']=-.01*spent;d['wasted_ammo']=-.07*spent if spent and not landed else 0
 d['health_gained']=.02*max(0,after['health']-before['health'])
 return float(sum(d.values())),d

def event_amplitude(packet,before,prior):
 enemy=packet['active'];damage=max(0,prior['health']-before['health']);dealt=max(0,before['damage_inflicted']-prior['damage_inflicted']);kill=max(0,before['kills']-prior['kills'])
 return 10.0*max(packet['intensity'] if enemy else 0,.8 if dealt else 0,1.0 if damage or kill else 0)

def run_episode(seed,policy,condition,learn,run_seed,replay=None,deterministic_eval=True,frame_hook=None,show_hud=False):
 b=NativeBrain(GRAPH);f=Features();g=Game(seed=seed,scenario='combat_survival',spectator=True,show_hud=show_hud);target=json.loads(TARGET_MANIFEST.read_text());sided=sided_populations(target)
 rng=np.random.default_rng(run_seed);old={};prior=g.observation();events=[];traj=[]
 held=None
 try:
  for tick in range(1,CAP*TICKS_PER_SECOND+1):
   before=g.observation();scene=g.spectator();packet=packet_from_scene(scene,old,max(0,prior['health']-before['health']));old=packet.pop('distances')
   lix,lval=threat_drive(packet,sided,20);drive=np.zeros(b.n,np.float32);drive[lix]+=lval
   amp=(float(replay[tick-1]) if tick<=len(replay) else 0.0) if replay is not None else event_amplitude(packet,before,prior)
   stim=TK if condition=='doom_guy' else (RANDOM if condition=='random_control' else np.empty(0,dtype=np.int64));drive[stim]+=amp
   frame=g.pixels();target_step=round(tick*10000/TICKS_PER_SECOND);steps=target_step-b.cursor;counts,_=b.step(retinal_samples(frame,b.uv),steps*b.dt,extra_drive=drive if np.any(drive) else None)
   x=f.push(counts,steps*b.dt/1000)
   if held is None or (tick-1)%ACTION_HOLD==0:held=policy.choose(x,rng,(not learn) and deterministic_eval)
   action,macro,index,p,v,value=held
   if frame_hook is not None:frame_hook(tick,frame,counts,x,action,macro,before,packet)
   engine=g.act(action);after=g.observation();r,parts=reward(before,after,engine)
   traj.append({'reward':r,'index':index,'probability':p,'vector':v,'value':value});events.append({'tick':tick,'action':action,'macro':macro,'reward':r,'reward_components':parts,'before':before,'game':after,'tk_amplitude':amp,'stimulated_indices':stim.tolist(),'features':x.round(6).tolist(),'tk_spikes':int(counts[TK].sum()),'decoder_input_only_neural':True})
   prior=before
   if after['finished']:break
 finally:g.close()
 if learn:policy.update(traj)
 return {'seed':seed,'condition':condition,'events':events,'metrics':metrics(events)}

def metrics(events):
 last=events[-1]['game'];return {'score':float(sum(e['reward'] for e in events)),'ticks':len(events),'survival_seconds':len(events)/35,'survived':not last['finished'],'kills':last['kills'],
  'damage_inflicted':last['damage_inflicted'],'damage_received':sum(max(0,e['before']['health']-e['game']['health']) for e in events),'hits':last['hits'],
  'ammo_spent':sum(max(0,e['before']['ammo']-e['game']['ammo']) for e in events),'attacks':sum(e['action']['attack'] for e in events),'advance_ticks':sum(e['action']['forward']>0 for e in events),'turn_ticks':sum(e['action']['turn']!=0 for e in events),'strafe_ticks':sum(e['action'].get('strafe',0)!=0 for e in events),'macro_counts':{m:sum(e['macro']==m for e in events) for m,_,_,_,_ in MACROS},
  'tk_spikes':sum(e['tk_spikes'] for e in events)}

def train(output=OUT):
 output=Path(output);policy=ActorCritic(2*len(groups()));curve=[]
 checkpoint=output/'doom-guy-decoder.npz';checkpoint.parent.mkdir(parents=True,exist_ok=True)
 if checkpoint.exists() and len(list((output/'training').glob('*.json')))==len(TRAIN)*4:
  saved=np.load(checkpoint);policy.actor[:]=saved['actor'];policy.value[:]=saved['value']
  for path in sorted((output/'training').glob('*.json')):
   rec=json.loads(path.read_text());name=path.stem.split('-');curve.append({'epoch':int(name[0][1:]),'seed':int(name[1]),'metrics':rec['metrics']})
 else:
  for epoch in range(4):
   for seed in TRAIN:
    rec=run_episode(seed,policy,'no_tk',True,epoch*100000+seed);atomic_json(output/'training'/f'e{epoch}-{seed}.json',rec);curve.append({'epoch':epoch,'seed':seed,'metrics':rec['metrics']})
  np.savez_compressed(checkpoint,actor=policy.actor,value=policy.value)
 frozen_hash=file_sha256(checkpoint);rows=[]
 for seed in DEV:
  primary=run_episode(seed,policy,'doom_guy',False,900000+seed);schedule=[e['tk_amplitude'] for e in primary['events']]
  for rec in (primary,run_episode(seed,policy,'no_tk',False,900000+seed),run_episode(seed,policy,'random_control',False,900000+seed,replay=schedule)):
   atomic_json(output/'development'/f'{seed}-{rec["condition"]}.json',rec);rows.append({'seed':seed,'condition':rec['condition'],'metrics':rec['metrics']})
 result={'schema':5,'status':'complete','training_seeds':TRAIN,'development_seeds':DEV,'training_episodes':len(TRAIN)*4,'curve':curve,'decoder_sha256':frozen_hash,'action_hold_ticks':ACTION_HOLD,'real_strafe_axis':True,'hitscan_priority_current':.20,'wasted_ammo_penalty':-.07,'rows':rows,
  'controls':{'no_tk':'same trained decoder and LC4/LPLC2 signal, no Tk current','random_control':'same Tk current schedule delivered to five predetermined unrelated neurons'},
  'claim_boundary':'Engineered frozen-MaleCNS BCI; Doom Guy is a label, not anger, intent, consciousness, or natural perception.'};atomic_json(output/'results.json',result);return result
if __name__=='__main__':print(json.dumps(train(),indent=2))
