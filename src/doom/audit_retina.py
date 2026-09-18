"""Bounded-memory independent reconstruction of the retinal projection."""
import json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc
ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/'connectome_data/malecns_v1'
    graph=np.load(ROOT/'outputs/doom/malecns_v1/graph.npz')
    ids=graph['ids']
    a=feather.read_table(root/'annotations.feather').to_pandas().set_index('bodyId').loc[ids]
    receptor=a.type.eq('R1-R6').to_numpy()
    anchor=(a.type.isin(['L1','L2','L3']) & a.assignedOlHex1.notna() & a.assignedOlHex2.notna()).to_numpy()
    reader=ipc.open_file(pa.memory_map(str(root/'normalized/edges.arrow')))
    pieces=[];cross_side=0;pairs=Counter()
    for k in range(reader.num_record_batches):
        b=reader.get_batch(k)
        pre,post,count=[b.column(b.schema.get_field_index(c)).to_numpy() for c in ['pre_index','post_index','synapse_count']]
        selected=receptor[pre]&anchor[post]
        if not selected.any():continue
        p=pre[selected];q=post[selected]
        ps=a.rootSide.iloc[p].fillna('unknown').astype(str).to_numpy();qs=a.somaSide.iloc[q].fillna('unknown').astype(str).to_numpy()
        pairs.update(zip(ps,qs))
        cross_side+=int((((ps=='L')&(qs=='R'))|((ps=='R')&(qs=='L'))).sum())
        pieces.append(pd.DataFrame({'pre':p,'h1':a.assignedOlHex1.to_numpy()[q],
          'h2':a.assignedOlHex2.to_numpy()[q],'count':count[selected]}))
    contacts=pd.concat(pieces,ignore_index=True)
    grouped=contacts.groupby(['pre','h1','h2'],sort=False,as_index=False)['count'].sum()
    winners=grouped.loc[grouped.groupby('pre')['count'].idxmax()].sort_values('pre')
    mapped=winners.pre.to_numpy()
    np.testing.assert_array_equal(graph['retina'],mapped)
    np.testing.assert_array_equal(graph['hexes'],winners[['h1','h2']].to_numpy())
    confidence=winners['count'].to_numpy()/contacts.groupby('pre')['count'].sum().reindex(mapped).to_numpy()
    np.testing.assert_allclose(graph['confidence'],confidence,rtol=0,atol=1e-12)
    xy=np.column_stack([winners.h1-.5*winners.h2,np.sqrt(3)/2*winners.h2])
    sides=a.rootSide.to_numpy()[mapped];uv=np.empty_like(xy)
    assert set(sides)=={'L','R'}
    for side in ['L','R']:
        mask=sides==side;local=xy[mask];local=(local-local.min(axis=0))/np.ptp(local,axis=0)
        uv[mask,0]=.6*local[:,0] if side=='L' else .4+.6*(1-local[:,0])
        uv[mask,1]=1-local[:,1]
    np.testing.assert_allclose(graph['uv'],uv,atol=4e-8,rtol=0)
    report={'passed':True,'mapped':len(mapped),'unmapped':int(receptor.sum())-len(mapped),
      'left':int((sides=='L').sum()),'right':int((sides=='R').sum()),
      'confidence_below_point8':int((confidence<.8).sum()),'cross_side_anchor_edges':cross_side,
      'side_annotation_fields':'receptor rootSide, anchor somaSide','anchor_side_pairs':{str(k):v for k,v in pairs.items()},'algorithm_matches':True,'optically_calibrated':False}
    (ROOT/'outputs/doom/audit/retinal-projection.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))
if __name__=='__main__':main()
