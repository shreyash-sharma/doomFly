"""Resolve FEARLESS populations and freeze a graph-matched control."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra


ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"
DATA = ROOT / "connectome_data/malecns_v1"
OUTPUT = ROOT / "outputs/fearless/target-manifest.json"
REQUESTED = ("LC4", "LPLC2", "DNp09", "DNp01", "DNp20", "DNpe017")
EXCLUDED_CONTROL_TYPES = set(REQUESTED)
CONTROL_SEED = 20260915


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def exact_population(nodes, annotations, label):
    indices = np.flatnonzero(nodes.cell_type.eq(label).to_numpy())
    if not len(indices):
        raise ValueError(f"No exact canonical cell_type match for {label}")
    if len(np.unique(indices)) != len(indices):
        raise ValueError(f"Duplicate graph index for {label}")
    source_ids = nodes.source_id.iloc[indices].to_numpy(dtype=np.int64)
    raw = annotations.reindex(source_ids)
    if raw.index.has_duplicates or raw.index.isna().any() or raw.type.isna().all():
        raise ValueError(f"Ambiguous raw annotation join for {label}")
    if not raw.type.eq(label).all():
        raise ValueError(f"Raw exact type mismatch for {label}")
    return indices.astype(np.int32), raw


def output_statistics(ptr, weight):
    cumulative = np.r_[0.0, np.cumsum(np.abs(weight), dtype=np.float64)]
    signed = np.r_[0.0, np.cumsum(weight, dtype=np.float64)]
    return (
        np.diff(ptr).astype(np.int64),
        cumulative[ptr[1:]] - cumulative[ptr[:-1]],
        signed[ptr[1:]] - signed[ptr[:-1]],
    )


def distances_to_targets(ptr, post, targets):
    n = len(ptr) - 1
    graph = csr_matrix((np.ones(len(post), dtype=np.uint8), post, ptr), shape=(n, n))
    reverse = graph.transpose().tocsr()
    return dijkstra(reverse, directed=True, indices=np.asarray(targets),
                    unweighted=True, min_only=True)


def direct_summary(ptr, post, weight, sources, targets):
    target_mask = np.zeros(len(ptr) - 1, dtype=bool)
    target_mask[np.asarray(targets)] = True
    rows = []
    for source in sources:
        start, stop = int(ptr[source]), int(ptr[source + 1])
        local = np.flatnonzero(target_mask[post[start:stop]]) + start
        rows.extend(local.tolist())
    rows = np.asarray(rows, dtype=np.int64)
    selected = weight[rows] if len(rows) else np.empty(0, dtype=weight.dtype)
    contacts = np.rint(np.abs(selected.astype(np.float64)) / 0.275).astype(np.int64)
    return {
        "edge_rows": int(len(rows)),
        "contacts": int(contacts.sum()),
        "signed_weight_sum": float(selected.sum(dtype=np.float64)),
        "absolute_weight_sum": float(np.abs(selected).sum(dtype=np.float64)),
    }


def select_matched_control(nodes, annotations, graph, populations, seed=CONTROL_SEED):
    ptr, post, weight = graph["ptr"], graph["post"], graph["weight"]
    n = len(nodes)
    out_edges, out_abs, _ = output_statistics(ptr, weight)
    in_edges = np.bincount(post, minlength=n).astype(np.int64)
    in_abs = np.bincount(post, weights=np.abs(weight), minlength=n)
    gf = populations["DNp01"]
    bci = np.r_[populations["DNp20"], populations["DNpe017"]]
    distance_gf = distances_to_targets(ptr, post, gf)
    distance_bci = distances_to_targets(ptr, post, bci)
    raw = annotations.reindex(nodes.source_id.to_numpy()).reset_index()
    sides = raw.somaSide.fillna("").astype(str).to_numpy()
    transmitters = nodes.neurotransmitter.fillna("missing").astype(str).to_numpy()
    cell_types = nodes.cell_type.fillna("").astype(str).to_numpy()
    eligible = nodes.superclass.eq("visual_projection").to_numpy()
    eligible &= ~np.isin(cell_types, list(EXCLUDED_CONTROL_TYPES))
    excluded_indices = np.unique(np.r_[graph["retina"], graph["lamina"], graph["sugar"],
                                      *populations.values()])
    eligible[excluded_indices] = False
    feature = np.column_stack([
        np.log1p(out_edges), np.log1p(out_abs), np.log1p(in_edges), np.log1p(in_abs),
        np.minimum(distance_gf, 20), np.minimum(distance_bci, 20),
    ]).astype(np.float64)
    feature[~np.isfinite(feature)] = 21
    pool = feature[eligible]
    scale = np.std(pool, axis=0)
    scale[scale == 0] = 1
    feature /= scale
    rng = np.random.default_rng(seed)
    lc4 = populations["LC4"].copy()
    lc4 = lc4[rng.permutation(len(lc4))]
    selected = []
    available = eligible.copy()
    tie_break = rng.random(n)
    for target in lc4:
        candidates = np.flatnonzero(available & (sides == sides[target]) &
                                    (transmitters == transmitters[target]))
        if not len(candidates):
            raise ValueError("No exact side/transmitter matched control candidate")
        distance = np.sum((feature[candidates] - feature[target]) ** 2, axis=1)
        order = np.lexsort((tie_break[candidates], distance))
        chosen = int(candidates[order[0]])
        selected.append(chosen)
        available[chosen] = False
    selected = np.asarray(sorted(selected), dtype=np.int32)
    if len(selected) != len(populations["LC4"]) or len(np.unique(selected)) != len(selected):
        raise AssertionError("Invalid matched control selection")
    fields = ["outgoing_edge_rows", "outgoing_absolute_weight_sum", "incoming_edge_rows",
              "incoming_absolute_weight_sum", "distance_to_gf", "distance_to_bci"]
    raw_values = np.column_stack([out_edges, out_abs, in_edges, in_abs,
                                  distance_gf, distance_bci])
    balance = {}
    for column, field in enumerate(fields):
        target_mean = float(np.mean(raw_values[populations["LC4"], column]))
        control_mean = float(np.mean(raw_values[selected, column]))
        balance[field] = {
            "lc4_mean": target_mean,
            "control_mean": control_mean,
            "difference": control_mean - target_mean,
        }
    return selected, {
        "selection_seed": seed,
        "candidate_superclass": "visual_projection",
        "matching": ["somaSide", "neurotransmitter", *fields],
        "selected_graph_indices": selected.tolist(),
        "selected_body_ids": [str(value) for value in graph["ids"][selected]],
        "side_counts": {side: int(np.count_nonzero(sides[selected] == side)) for side in ["L", "R", ""]},
        "transmitter_counts": {value: int(np.count_nonzero(transmitters[selected] == value))
                               for value in sorted(set(transmitters[selected]))},
        "balance": balance,
        "activity_matching": "not used; selection frozen before synthetic or behavioral outcomes",
    }


def build_manifest(output=OUTPUT):
    nodes = feather.read_table(DATA / "normalized/neurons.feather").to_pandas()
    annotations = feather.read_table(DATA / "annotations.feather").to_pandas().set_index("bodyId")
    with np.load(GRAPH) as loaded:
        graph = {key: loaded[key] for key in loaded.files}
    if not np.array_equal(graph["ids"], nodes.source_id.to_numpy(dtype=np.int64)):
        raise ValueError("Graph IDs do not match normalized nodes")
    ptr, post, weight = graph["ptr"], graph["post"], graph["weight"]
    out_edges, out_abs, out_signed = output_statistics(ptr, weight)
    populations = {}
    population_records = {}
    def optional_text(value):
        return None if value is None or str(value) in {"nan", "None", "<NA>"} else str(value)
    for label in REQUESTED:
        indices, raw = exact_population(nodes, annotations, label)
        populations[label] = indices
        records = []
        for index, (body_id, row) in zip(indices, raw.iterrows()):
            records.append({
                "body_id": str(body_id), "graph_index": int(index),
                "matched_annotation_field": "type", "matched_annotation_value": str(row.type),
                "instance": optional_text(row.instance),
                "soma_side": optional_text(row.somaSide),
                "root_side": optional_text(row.rootSide),
                "superclass": str(row.superclass),
                "neurotransmitter": optional_text(nodes.neurotransmitter.iloc[index]),
                "outgoing_edge_rows": int(out_edges[index]),
                "outgoing_absolute_weight_sum": float(out_abs[index]),
                "outgoing_signed_weight_sum": float(out_signed[index]),
            })
        population_records[label] = {
            "requested_label": label,
            "match_mode": "exact",
            "count": len(records),
            "side_counts": {
                "L": sum(row["soma_side"] == "L" for row in records),
                "R": sum(row["soma_side"] == "R" for row in records),
                "blank": sum(row["soma_side"] is None for row in records),
            },
            "warnings": (["rootSide blank for all rows"] if all(row["root_side"] is None for row in records) else []),
            "neurons": records,
        }
    forward_from_lc4 = dijkstra(
        csr_matrix((np.ones(len(post), dtype=np.uint8), post, ptr),
                   shape=(len(nodes), len(nodes))),
        directed=True, indices=populations["LC4"], unweighted=True, min_only=True,
    )
    connectivity = {}
    for target in ["DNp01", "DNp20", "DNpe017"]:
        distances = forward_from_lc4[populations[target]]
        connectivity[f"LC4_to_{target}"] = {
            **direct_summary(ptr, post, weight, populations["LC4"], populations[target]),
            "shortest_directed_path_edges": None if not np.isfinite(distances).any() else int(np.min(distances)),
            "per_target_shortest_path_edges": [None if not np.isfinite(x) else int(x) for x in distances],
        }
    connectivity["LPLC2_to_DNp01"] = direct_summary(
        ptr, post, weight, populations["LPLC2"], populations["DNp01"])
    selected, control = select_matched_control(nodes, annotations, graph, populations)
    source_lock = json.loads((DATA / "source.lock.json").read_text())
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
                            capture_output=True, text=True).stdout.strip()
    manifest = {
        "schema": 1,
        "project": "FEARLESS",
        "repository_commit": commit,
        "graph_sha256": sha256_file(GRAPH),
        "source_hashes": {name: value["sha256"] for name, value in source_lock.items()},
        "resolver_sha256": sha256_file(__file__),
        "populations": population_records,
        "pathway_connectivity": connectivity,
        "random_control": control,
        "warnings": [
            "Shortest paths are structural and do not establish propagation under the LIF model.",
            "Contact weights use the repository's coarse presynaptic transmitter-sign convention.",
        ],
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".partial")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print("population count L R blank-root")
    for label, record in population_records.items():
        sides = record["side_counts"]
        print(f"{label:8s} {record['count']:5d} {sides['L']:3d} {sides['R']:3d} "
              f"{sum(row['root_side'] is None for row in record['neurons']):5d}")
    print(json.dumps({"output": str(output), "graph_sha256": manifest["graph_sha256"],
                      "random_control_count": len(selected), "connectivity": connectivity}, indent=2))
    return manifest


if __name__ == "__main__":
    build_manifest()
