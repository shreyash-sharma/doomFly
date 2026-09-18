"""Independent source → normalized → runtime graph audit. No importer helpers."""
import hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pyarrow.ipc as ipc

ROOT = Path(__file__).resolve().parents[1]

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()

def main():
    root = ROOT/'connectome_data/malecns_v1'
    lock = json.loads((root/'source.lock.json').read_text())
    hashes = {}
    for name, info in lock.items():
        hashes[name] = digest(root/name)
        assert hashes[name] == info['sha256']
        assert (root/name).stat().st_size == info['bytes']
    raw = feather.read_table(root/'annotations.feather').to_pandas()
    nodes = feather.read_table(root/'normalized/neurons.feather').to_pandas()
    wanted = raw[raw.superclass.notna() & raw.superclass.ne('')].sort_values('bodyId')
    ids = nodes.source_id.to_numpy()
    np.testing.assert_array_equal(wanted.bodyId, ids)
    assert wanted.status.eq('Glia').sum() == 0
    assert len(np.unique(ids)) == len(ids)
    np.testing.assert_array_equal(nodes.node_index, np.arange(len(ids)))
    np.testing.assert_array_equal(wanted.type.fillna(''), nodes.cell_type.fillna(''))
    nt = feather.read_table(root/'neurotransmitters.feather').to_pandas().set_index('body')
    assert nt.index.is_unique
    np.testing.assert_array_equal(nt.reindex(ids).consensus_nt.fillna(''), nodes.neurotransmitter.fillna(''))
    index = pd.Index(ids)
    source = ipc.open_file(pa.memory_map(str(root/'edges.feather')))
    normalized = ipc.open_file(pa.memory_map(str(root/'normalized/edges.arrow')))
    assert source.num_record_batches == normalized.num_record_batches
    total = dict(source_edges=0, retained_edges=0, source_contacts=0, retained_contacts=0, weight_one_edges=0, self_edges=0)
    for k in range(source.num_record_batches):
        b = source.get_batch(k)
        pre, post, count = [b.column(b.schema.get_field_index(c)).to_numpy() for c in ['body_pre','body_post','weight']]
        i, j = index.get_indexer(pre), index.get_indexer(post)
        keep = (i >= 0) & (j >= 0)
        n = normalized.get_batch(k)
        for actual, expected in zip(n.columns, [i[keep], j[keep], count[keep]]):
            np.testing.assert_array_equal(actual.to_numpy(), expected)
        total['source_edges'] += len(pre); total['retained_edges'] += int(keep.sum())
        total['source_contacts'] += int(count.sum()); total['retained_contacts'] += int(count[keep].sum())
        total['weight_one_edges'] += int((count[keep] == 1).sum())
        total['self_edges'] += int((i[keep] == j[keep]).sum())
    print('All raw edge batches match normalized export.', flush=True)
    graph = np.load(ROOT/'outputs/doom/malecns_v1/graph.npz')
    np.testing.assert_array_equal(graph['ids'], ids)
    table = normalized.read_all()
    pre, post, count = [table.column(c).to_numpy() for c in ['pre_index','post_index','synapse_count']]
    order = np.argsort(pre, kind='stable')
    np.testing.assert_array_equal(graph['ptr'], np.r_[0, np.cumsum(np.bincount(pre, minlength=len(ids)))])
    np.testing.assert_array_equal(graph['post'], post[order])
    # Check the implemented sign convention, not its biological validity.
    sign = np.where(nodes.neurotransmitter.isin(['gaba','glutamate','histamine']), -1., 1.).astype(np.float32)
    expected = (count[order].astype(np.float32)*sign[pre[order]]*.275).astype(np.float32)
    np.testing.assert_array_equal(graph['weight'], expected)
    a = raw.set_index('bodyId').loc[ids]
    assert a.iloc[graph['retina']].type.eq('R1-R6').all()
    assert a.iloc[graph['lamina']].type.isin(['L1','L2','L3','L5']).all()
    assert a.iloc[graph['sugar']].type.eq('LB3c').all()
    assert np.isfinite(graph['uv']).all() and ((graph['uv'] >= 0) & (graph['uv'] <= 1)).all()
    manifest = json.loads((ROOT/'outputs/doom/malecns_v1/manifest.json').read_text())
    for r in manifest['readouts']:
        assert str(ids[r['index']]) == r['id']
        assert a.iloc[r['index']].type == r['type']
        assert a.iloc[r['index']].somaSide == r['side']
    report = dict(passed=True, checked_at=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
      dataset='MaleCNS v1.0 male brain and VNC', neurons=len(ids), retained_glia=0,
      source_sha256=hashes, graph_sha256=digest(ROOT/'outputs/doom/malecns_v1/graph.npz'),
      edge_checks=total, all_source_rows_checked=True, all_runtime_edges_checked=True,
      checks=['Source SHA-256 and sizes match lock','Exact source neuron IDs and cell types',
        'Consensus neurotransmitter join','Every source edge rejoined independently using pandas index',
        'Every normalized endpoint and contact count','Every runtime CSR endpoint and signed weight',
        'Readout IDs, types and sides','Sensory cell classes and UV bounds'],
      biological_dynamics_validated=False)
    out = ROOT/'outputs/doom/audit/data-integrity.json'
    out.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

if __name__ == '__main__': main()
