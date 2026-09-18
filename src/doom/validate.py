"""Full-graph causal interventions. These checks do not validate fly cognition."""
from pathlib import Path
import gc,hashlib,json,time
import numpy as np
from PIL import Image
from doom.native import NativeBrain
from doom.game import Game,retinal_samples
from doom.engine import NeuralControls
ROOT=Path(__file__).resolve().parents[1]
def main():
    path=ROOT/'outputs/doom/malecns_v1/graph.npz'
    manifest=json.loads((path.parent/'manifest.json').read_text())
    game=Game();frame=game.pixels();game.close()
    results=[];intact=None;zero=None
    for condition in ['intact','blank_vision','retina_disconnected','all_edges_disconnected']:
        b=NativeBrain(path);d=NeuralControls(manifest['readouts']);light=retinal_samples(frame,b.uv)
        if condition=='blank_vision':light.fill(0)
        cut=0
        if condition=='retina_disconnected':
            for i in b.retina:
                cut+=int(b.ptr[i+1]-b.ptr[i]);b.weight[b.ptr[i]:b.ptr[i+1]]=0
        if condition=='all_edges_disconnected':cut=len(b.weight);b.weight.fill(0)
        start=time.perf_counter();counts,_=b.step(light,500);wall=time.perf_counter()-start
        assert b.n==166700 and len(b.weight)==25582938
        action=d.decode(counts,.5)
        if condition=='intact':intact=counts.copy()
        if condition=='blank_vision':
            assert not counts[b.retina].any();assert not np.array_equal(counts,intact)
        if condition=='retina_disconnected':assert not np.array_equal(counts,intact)
        if condition=='all_edges_disconnected':
            spontaneous=np.zeros(b.n,dtype=bool);spontaneous[np.r_[b.retina,b.lamina]]=True
            assert not counts[~spontaneous].any()
            assert action['turn']==0 and action['forward']==0 and not action['attack']
        assert condition=='blank_vision' or counts[b.retina].sum()>0
        result={'condition':condition,'neural_ms':500,'wall_seconds':round(wall,3),'node_count':b.n,'edge_count':len(b.weight),
          'silenced_edges':cut,'total_spikes':int(counts.sum()),'receptor_spikes':int(counts[b.retina].sum()),
          'active_neurons':int(np.count_nonzero(counts)),
          'input_sha256':hashlib.sha256(light.tobytes()).hexdigest(),'spike_counts_sha256':hashlib.sha256(counts.tobytes()).hexdigest(),
          'action':{k:action[k] for k in ['turn','forward','attack']},'readouts':action['readouts']}
        results.append(result);print(json.dumps({k:v for k,v in result.items() if k!='readouts'}),flush=True)
        del b;gc.collect()
    assert results[0]['input_sha256']==results[2]['input_sha256']==results[3]['input_sha256']
    report={'created_at':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'passed':True,'biologically_validated':False,
      'interpretation':'Pixels affect the retained neural model. Brightness activates mapped receptors; removing their edges changes downstream activity. The selected locomotion/mouthpart readouts remained silent. This is not evidence of successful game control, realistic vision, or learning.',
      'checks':['All retained graph counts preserved','Matched inputs for intact and disconnected conditions','Black input silences mapped photoreceptor proxy','Retinal disconnection changes neural activity','Total disconnection eliminates all activity outside directly driven cells','Total disconnection produces no selected game action'],
      'results':results}
    (ROOT/'outputs/doom/validation.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
