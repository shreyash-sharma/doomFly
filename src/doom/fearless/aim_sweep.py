"""Doom-native aiming loophole: neural-gated fire plus deterministic sweep.

Classic Doom resolves hitscan fire immediately and supplies vertical auto-aim;
the sweep supplies only a blind horizontal scan. It never reads enemy geometry
and is an engineered action adapter, not a biological motor claim.
"""
import json, math
from pathlib import Path
import numpy as np
from doom.fearless.embodiment import EmbodimentDecoder
from doom.fearless.visual_encoder import MotionLoomEncoder
from doom.fearless.run import GRAPH, TARGET_MANIFEST, TICKS_PER_SECOND
from doom.fearless.danger import sided_populations, threat_drive
from doom.game import Game, retinal_samples
from doom.native import NativeBrain

ROOT=Path(__file__).resolve().parents[2];CHECKPOINT=ROOT/'outputs/fearless/aggressive-mapper-v7-visual/decoder-frozen.npz';OUT=ROOT/'outputs/fearless/aim-sweep-v2-doomguy';SEEDS=(783101,783102,783103,783104,783105,783106,783107,783108,783109,783110);TK_AMP=20.

def one(seed,sweep):
 d=EmbodimentDecoder.load(CHECKPOINT);b=NativeBrain(GRAPH);g=Game(seed=seed,scenario='combat_survival',spectator=True);enc=MotionLoomEncoder();target=json.loads(TARGET_MANIFEST.read_text());sided=sided_populations(target);events=[]
 try:
  for tick in range(1,10*TICKS_PER_SECOND+1):
   before=g.observation();frame=g.pixels();visual=enc.encode(frame);direction=max(('left','center','right'),key=lambda k:visual[k]);intensity=max(visual['left'],visual['center'],visual['right'],visual['looming']);packet={'active':intensity>.02,'intensity':intensity,'direction':direction};ix,val=threat_drive(packet,sided,TK_AMP);drive=np.zeros(b.n,np.float32);drive[ix]+=val if len(ix) else 0
   step=round(tick*10000/TICKS_PER_SECOND)-b.cursor;c,_=b.step(retinal_samples(frame,b.uv),step*b.dt,extra_drive=drive if len(ix) else None);x=d.features(c,step*b.dt/1000);bits,_,=d.sample(x,np.random.default_rng(seed+tick));decoded=d.action(bits)
   action=dict(decoded)
   if sweep and decoded['attack']: action['turn']=float(3 if (tick//4)%2==0 else -3)
   reward=g.act(action);after=g.observation();events.append({'tick':tick,'decoded_action':decoded,'action':action,'engine_reward':reward,'game':after})
   if after['finished']:break
 finally:g.close()
 final=events[-1]['game'];return {'seed':seed,'sweep':sweep,'events':events,'metrics':{'survival':len(events)/TICKS_PER_SECOND,'kills':final['kills'],'damage_received':100-final['health'],'attacks':sum(e['action']['attack'] for e in events),'sweep_ticks':sum(e['action']['turn']!=e['decoded_action']['turn'] for e in events)}}

def run():
 OUT.mkdir(parents=True,exist_ok=True);rows=[one(s,False) for s in SEEDS]+[one(s,True) for s in SEEDS]
 for r in rows:(OUT/f'{r["seed"]}-sweep-{int(r["sweep"])}.json').write_text(json.dumps(r,indent=2)+'\n')
 summary={'schema':1,'status':'complete','seeds':SEEDS,'decoder_sha256':__import__('hashlib').sha256(CHECKPOINT.read_bytes()).hexdigest(),'rows':[{'seed':r['seed'],'sweep':r['sweep'],'metrics':r['metrics']} for r in rows],'mechanism':'Attack is neural-decoder gated; sweep is fixed ±3 turn oscillation, no enemy state input.','game_mechanics_source':'Doom hitscan with vertical auto-aim; see cited report.'}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary
if __name__=='__main__':print(json.dumps(run(),indent=2))
