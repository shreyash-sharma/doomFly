"""Matched full-graph interventions and untrained closed-loop comparisons."""
import gc, hashlib, json, time
from pathlib import Path
import numpy as np
from doom.native import NativeBrain, BUILD
from doom.game import Game, retinal_samples
from doom.engine import NeuralControls
ROOT=Path(__file__).resolve().parents[1]
PATH=ROOT/'outputs/doom/malecns_v1/graph.npz'
M=json.loads((PATH.parent/'manifest.json').read_text())
def sha(a): return hashlib.sha256(a.tobytes()).hexdigest()
def buttons(a): return {k:a[k] for k in ['turn','forward','attack']}
def main():
    game=Game();frame=game.pixels();assets=game.assets;game.close()
    results=[]
    for condition in ['intact','blank_vision','retina_disconnected','all_edges_disconnected','no_external_drive','no_lamina_bias','ambiguous_sign_reversed','shuffled_pixels']:
        b=NativeBrain(PATH);d=NeuralControls(M['readouts'],mode='bci');light=retinal_samples(frame,b.uv)
        if condition in ['blank_vision','no_external_drive']:light.fill(0)
        if condition=='shuffled_pixels':light=np.random.default_rng(123).permutation(light)
        if condition=='retina_disconnected':
            for i in b.retina:b.weight[b.ptr[i]:b.ptr[i+1]]=0
        if condition=='all_edges_disconnected':b.weight.fill(0)
        if condition=='ambiguous_sign_reversed':
            import pyarrow.feather as feather
            nodes=feather.read_table(ROOT/'connectome_data/malecns_v1/normalized/neurons.feather').to_pandas()
            uncertain=~nodes.neurotransmitter.isin(['acetylcholine','gaba','glutamate','histamine'])
            for i in np.flatnonzero(uncertain):b.weight[b.ptr[i]:b.ptr[i+1]]*=-1
        before=sha(b.weight);total=np.zeros(b.n,dtype=np.int64);start=time.monotonic()
        for tick in range(18):
            steps=round((tick+1)*10000/35)-b.cursor
            counts,_=b.step(light,steps*.1,lamina_bias=0 if condition in ['no_external_drive','no_lamina_bias'] else 12)
            total+=counts;a=d.decode(counts,steps*.1/1000)
        assert sha(b.weight)==before
        if condition=='no_external_drive':assert not total.any()
        if condition=='all_edges_disconnected':
            direct=np.zeros(b.n,dtype=bool);direct[np.r_[b.retina,b.lamina]]=True
            assert not total[~direct].any()
            assert a['turn']==a['forward']==0 and not a['attack']
        result={'condition':condition,'simulation_ms':b.sim_ms,'frame_intervals':18,
          'input_sha256':sha(light),'spikes_sha256':sha(total),'total_spikes':int(total.sum()),
          'active_neurons':int(np.count_nonzero(total)),'receptor_spikes':int(total[b.retina].sum()),
          'voltage_min_mv':float(b.v.min()),'voltage_max_mv':float(b.v.max()),
          'action_at_end':buttons(a),'readouts':a['readouts'],'weights_unchanged':True,'wall_seconds':time.monotonic()-start}
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='readouts'}),flush=True)
        del b;gc.collect()
    loops=[]
    for seed in [41027,41028,41029]:
      for condition in ['intact','blank_vision','controls_clamped']:
        b=NativeBrain(PATH);d=NeuralControls(M['readouts'],mode='bci');game=Game(seed=seed)
        actions=0;fire=0;kills=0;spikes=0;initial_weights=sha(b.weight);start=time.monotonic()
        for tick in range(210):
            if game.game.is_episode_finished():kills+=game.observation()['kills'];game.new_episode()
            light=retinal_samples(game.pixels(),b.uv)
            if condition=='blank_vision':light.fill(0)
            steps=round((tick+1)*10000/35)-b.cursor
            c,_=b.step(light,steps*.1);a=d.decode(c,steps*.1/1000)
            if condition=='controls_clamped':a={'turn':0.,'forward':0.,'attack':False}
            game.act(a);spikes+=int(c.sum());actions+=int(bool(a['turn'] or a['forward'] or a['attack']));fire+=int(a['attack'])
        assert abs(b.sim_ms/1000-210/35)<.0001 and sha(b.weight)==initial_weights
        result={'seed':seed,'condition':condition,'game_tics':210,'neural_seconds':b.sim_ms/1000,
          'kills':kills+game.observation()['kills'],'action_tics':actions,'attack_tics':fire,'spikes':spikes,
          'wall_seconds':time.monotonic()-start,'weights_unchanged':True}
        loops.append(result);print(json.dumps(result),flush=True);game.close();del b;gc.collect()
    report={'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'build':BUILD,'assets':assets,
      'data_audit_sha256':hashlib.sha256((ROOT/'outputs/doom/audit/data-integrity.json').read_bytes()).hexdigest(),
      'fixed_frame_tests':results,'closed_loop_trials':loops,'biologically_validated':False,'learning_demonstrated':False,
      'limits':'Three game seeds, one reconstruction, six seconds each; descriptive checks, not a skill or learning benchmark. Fixed frame interventions use the production 35-Hz input cadence.'}
    (ROOT/'outputs/doom/audit/experiments.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
