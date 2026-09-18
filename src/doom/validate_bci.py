"""Test fixed visual-neuron BCI causality and a real closed game loop."""
import json,time,gc,hashlib
from pathlib import Path
import numpy as np
from doom.native import NativeBrain
from doom.game import Game,retinal_samples
from doom.engine import NeuralControls
ROOT=Path(__file__).resolve().parents[1]
def main():
 m=json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text());path=ROOT/'outputs/doom/malecns_v1/graph.npz'
 game=Game();frame=game.pixels();game.close();trials=[]
 for condition in ['intact','blank_vision','retina_disconnected','all_edges_disconnected']:
  b=NativeBrain(path);d=NeuralControls(m['readouts'],mode='bci');light=retinal_samples(frame,b.uv)
  if condition=='blank_vision':light.fill(0)
  if condition=='retina_disconnected':
   for i in b.retina:b.weight[b.ptr[i]:b.ptr[i+1]]=0
  if condition=='all_edges_disconnected':b.weight.fill(0)
  c,wall=b.step(light,500);a=d.decode(c,.5)
  record={'condition':condition,'input_sha256':hashlib.sha256(light.tobytes()).hexdigest(),'action':{k:a[k] for k in ['turn','forward','attack']},'readouts':a['readouts'],'neural_ms':500,'wall_seconds':round(wall,3)}
  if condition=='all_edges_disconnected':assert a['turn']==a['forward']==0 and not a['attack']
  trials.append(record);print({k:v for k,v in record.items() if k!='readouts'},flush=True);del b;gc.collect()
 assert trials[0]['action']!=trials[1]['action']
 assert trials[0]['action']!=trials[2]['action']
 b=NativeBrain(path);d=NeuralControls(m['readouts'],mode='bci');game=Game();actions=0;spikes=0;total_kills=0;events=[];start=time.monotonic()
 for tick in range(210):
  if game.game.is_episode_finished():total_kills+=game.observation()['kills'];game.new_episode()
  frame=game.pixels();light=retinal_samples(frame,b.uv)
  steps=int(round((tick+1)*10000/35))-b.cursor;c,w=b.step(light,steps*.1);a=d.decode(c,steps*.1/1000);game.act(a)
  active=abs(a['turn'])>1e-9 or abs(a['forward'])>1e-9 or a['attack'];actions+=int(active);spikes+=int(c.sum())
  if tick%35==0:
   event={'tick':tick+1,'action':{k:a[k] for k in ['turn','forward','attack']},'game':game.observation()};events.append(event);print(event,flush=True)
 assert actions>0
 report={'passed':True,'biologically_validated':False,'learning_demonstrated':False,'decoder':'engineered visual-neuron BCI',
 'interpretation':'The fixed BCI receives only activity of retained descending neurons. Black pixels and retinal disconnection change its output; disconnecting every edge abolishes it. This validates the implemented causal loop, not biological motor semantics, realistic vision, or learned game skill.',
 'trials':trials,'closed_loop':{'game_tics':210,'neural_seconds':b.sim_ms/1000,'wall_seconds':round(time.monotonic()-start,3),'action_tics':actions,'spikes':spikes,'kills':total_kills+game.observation()['kills'],'samples':events}}
 (ROOT/'outputs/doom/bci-validation.json').write_text(json.dumps(report,indent=2)+'\n');game.close()
if __name__=='__main__':main()
