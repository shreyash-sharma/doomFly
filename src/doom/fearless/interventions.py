"""Auditable runtime outgoing-synapse clamps for FEARLESS."""

import hashlib
import json
from pathlib import Path

import numpy as np


CONDITIONS = ("intact", "lc4_silenced", "lc4_random_matched")


def sha256_array(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def indices_for_condition(condition, target_manifest):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if condition == "intact":
        return np.empty(0, dtype=np.int64), []
    if condition == "lc4_silenced":
        rows = target_manifest["populations"]["LC4"]["neurons"]
        return np.asarray([row["graph_index"] for row in rows], dtype=np.int64), [row["body_id"] for row in rows]
    control = target_manifest["random_control"]
    return np.asarray(control["selected_graph_indices"], dtype=np.int64), list(control["selected_body_ids"])


def apply_intervention(brain, condition, target_manifest):
    indices, body_ids = indices_for_condition(condition, target_manifest)
    if condition != "intact" and not len(indices):
        raise ValueError("Non-intact intervention has an empty target set")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Duplicate intervention targets")
    if np.any(indices < 0) or np.any(indices >= brain.n):
        raise ValueError("Intervention target out of bounds")
    original = brain.weight.copy()
    before_hash = sha256_array(original)
    affected = np.zeros(len(brain.weight), dtype=bool)
    for index in indices:
        start, stop = int(brain.ptr[index]), int(brain.ptr[index + 1])
        affected[start:stop] = True
        brain.weight[start:stop] = 0
    if not np.array_equal(brain.weight[~affected], original[~affected]):
        raise AssertionError("A non-target outgoing weight changed")
    if len(indices) and np.any(brain.weight[affected] != 0):
        raise AssertionError("A selected outgoing weight was not clamped")
    if condition == "intact" and not np.array_equal(brain.weight, original):
        raise AssertionError("Intact condition mutated weights")
    return {
        "schema": 1,
        "condition": condition,
        "intervention": "none" if condition == "intact" else "runtime outgoing-synapse clamp",
        "selected_body_ids": body_ids,
        "selected_graph_indices": indices.tolist(),
        "selected_neurons": int(len(indices)),
        "affected_edge_rows": int(np.count_nonzero(affected)),
        "pre_clamp_signed_weight_sum": float(original[affected].sum(dtype=np.float64)),
        "pre_clamp_absolute_weight_sum": float(np.abs(original[affected]).sum(dtype=np.float64)),
        "weight_sha256_before": before_hash,
        "weight_sha256_after": sha256_array(brain.weight),
        "nonselected_weights_identical": True,
        "node_count_unchanged": brain.n,
        "edge_array_length_unchanged": len(brain.weight),
    }


def load_target_manifest(path):
    return json.loads(Path(path).read_text())


def build_intervention_manifest(graph_path, target_path, output_path):
    from doom.native import NativeBrain
    target_manifest = load_target_manifest(target_path)
    records = {}
    for condition in CONDITIONS:
        brain = NativeBrain(graph_path)
        records[condition] = apply_intervention(brain, condition, target_manifest)
    target_sha256 = hashlib.sha256(Path(target_path).read_bytes()).hexdigest()
    result = {
        "schema": 1,
        "target_manifest_sha256": target_sha256,
        "graph_sha256": target_manifest["graph_sha256"],
        "conditions": records,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".partial")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output_path)
    return result


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[2]
    value = build_intervention_manifest(
        root / "outputs/doom/malecns_v1/graph.npz",
        root / "outputs/fearless/target-manifest.json",
        root / "outputs/fearless/intervention-manifest.json",
    )
    print(json.dumps({name: {
        "selected_neurons": record["selected_neurons"],
        "affected_edge_rows": record["affected_edge_rows"],
        "weight_sha256_after": record["weight_sha256_after"],
    } for name, record in value["conditions"].items()}, indent=2))
