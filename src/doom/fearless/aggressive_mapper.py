"""Build a Doom-fly aggressive readout by imitation, then remove the teacher.

The teacher is used only while collecting labelled training trajectories. During
validation/playback the decoder sees MaleCNS spike features only.
"""
import json, math
from pathlib import Path
import numpy as np
from doom.fearless.embodiment import EmbodimentDecoder
from doom.fearless.embodiment_train import _feature_indices
from doom.fearless.danger import packet_from_scene, sided_populations, threat_drive
from doom.fearless.visual_encoder import MotionLoomEncoder
from doom.fearless.run import GRAPH, MODEL_MANIFEST, TICKS_PER_SECOND, atomic_json
from doom.game import Game, retinal_samples
from doom.native import NativeBrain

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'outputs/fearless/aggressive-mapper-v7-visual'
TK=np.array([8017,8196,36409,134363,136053],dtype=np.int64)
TRAIN=(781001,781002,781003);VALID=(782001,782002);CAP=5

def teacher(scene):
 p=scene['player']; es=[o for o in scene['objects'] if o['name'] in ('DoomFlyImp','Zombieman')]
 if not es:return np.array([1,0,0,1],dtype=np.float64),{'turn':0.,'forward':20.,'attack':True}
 e=min(es,key=lambda o:math.hypot(o['x']-p['x'],o['y']-p['y']))
 angle=(math.degrees(math.atan2(e['y']-p['y'],e['x']-p['x']))-p['angle']+180)%360-180
 left,right=angle < -8, angle > 8
 # Existing Doom-fly scripted benchmark fires continuously while pursuing;
 # this is a training label only and is removed before neural validation.
 bits=np.array([1, int(left), int(right), 1],dtype=np.float64)
 return bits,{'turn':float(np.clip(angle*.15,-6,6)),'forward':20.,'attack':True}

def collect(seed,decoder=None,fit=False,rng=None):
 ids,_=_feature_indices(); b=NativeBrain(GRAPH); g=Game(seed=seed,scenario='combat_survival',spectator=True); rows=[];events=[];previous_distances={};previous_health=g.observation()['health']
 target=json.loads((ROOT/'outputs/fearless/target-manifest.json').read_text());sided=sided_populations(target);threat_remaining=0;visual_encoder=MotionLoomEncoder()
 try:
  for tick in range(1,CAP*TICKS_PER_SECOND+1):
   before=g.observation(); scene=g.spectator(); packet=packet_from_scene(scene,previous_distances,max(0,previous_health-before['health']));previous_distances=packet.pop('distances');previous_health=before['health']; target_step=round(tick*10000/TICKS_PER_SECOND);steps=target_step-b.cursor
   frame=g.pixels();visual=visual_encoder.encode(frame)
   visual_direction=max(('left','center','right'),key=lambda k:visual[k]);visual_intensity=max(visual['left'],visual['center'],visual['right'],visual['looming'])
   visual_packet={'active':visual_intensity>0.02,'intensity':visual_intensity,'direction':visual_direction}
   # Directional artificial threat current reaches LC4/LPLC2 only, never buttons.
   inds,values=threat_drive(visual_packet,sided,amplitude=20.);extra=np.zeros(b.n,np.float32)
   if len(inds) and threat_remaining==0:threat_remaining=5
   if threat_remaining>0:
    extra[inds]+=values*2.0  # calibrated 40-unit current; threat_drive base is 20
    threat_remaining-=1
   c,_=b.step(retinal_samples(frame,b.uv),steps*b.dt,extra_drive=extra if len(inds) else None);x=decoder.features(c,steps*b.dt/1000) if decoder else np.r_[1.,np.zeros(len(ids))]
   bits,ta=teacher(scene)
   action=ta if fit else decoder.action(decoder.sample(x,rng)[0])
   engine=g.act(action);after=g.observation();events.append({'tick':tick,'teacher_bits':bits.tolist(),'teacher_action':ta,'action':action,'game':after,'engine_reward':engine,'features':x.tolist(),'visual_features':visual,'threat_packet':visual_packet,'threat_target_indices':inds.tolist(),'threat_amplitude':float(values[0]) if len(values) else 0.0})
   if fit: rows.append((x,bits))
   if after['finished']:break
 finally:g.close()
 return rows,events

def fit_decoder(decoder,rows,lr=.15,l2=1e-4):
 for x,y in rows:
  p=decoder.probabilities(x);decoder.weights += lr*np.outer(y-p,x)-lr*l2*decoder.weights

def play(seed,decoder):
 _,events=collect(seed,decoder,False,np.random.default_rng(seed)); final=events[-1]['game'];
 return {'seed':seed,'events':events,'metrics':{'survival':len(events)/TICKS_PER_SECOND,'kills':final['kills'],'damage_received':100-final['health'],'attacks':sum(e['action']['attack'] for e in events),'teacher_match':float(np.mean([e['action']==e['teacher_action'] for e in events]))}}

def run():
 ids,readouts=_feature_indices(); d=EmbodimentDecoder.initial(ids,seed=20260921); curves=[]
 for epoch in range(3):
  for seed in TRAIN:
   rows,events=collect(seed,d,True);fit_decoder(d,rows);curves.append({'epoch':epoch,'seed':seed,'teacher_steps':len(rows),'teacher_kills':events[-1]['game']['kills']})
 ck=OUT/'decoder-frozen.npz';digest=d.checkpoint(ck,{'label':'frozen aggressive Doom-fly readout','training_seeds':TRAIN,'outputs':['forward','turn_left','turn_right','attack'],'teacher_runtime':False,'readouts':readouts})
 results=[]
 for seed in VALID:
  r=play(seed,d);atomic_json(OUT/'validation'/f'{seed}.json',r);results.append(r['metrics'])
 summary={'schema':1,'status':'frozen-teacher-removed','training_seeds':TRAIN,'validation_seeds':VALID,'decoder_sha256':digest,'teacher_removed_before_validation':True,'directional_artificial_threat':{'targets':'LC4+LPLC2','amplitude':40.0,'pulse_calibration_ticks':5,'direction_encoded':'left/right/center','direct_actions':False,'calibration_artifact':'outputs/fearless/aggressive-mapper-v6-calibration/calibration.json'},'training_curve':curves,'validation_mean':{k:float(np.mean([r[k] for r in results])) for k in results[0]},'claim_boundary':'This is an engineered imitation interface, not fly learning or natural aggression.'}
 atomic_json(OUT/'summary.json',summary);return summary
if __name__=='__main__':print(json.dumps(run(),indent=2))
