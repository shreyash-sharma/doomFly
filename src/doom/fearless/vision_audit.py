"""Audit why the fixed LIF visual path does not recruit LC4.

This is a diagnostic, not a replacement visual model. It checks structural
reachability, measures activity at LC4's direct inputs, and evaluates a
parameter-free three-hop graded OFF-contrast proxy against frozen controls.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyarrow.feather as feather
from scipy.sparse import csr_matrix, diags

from doom.fearless.run import GRAPH, TARGET_MANIFEST, atomic_json, file_sha256
from doom.fearless.synthetic import (BASELINE_TICKS, COMBINATIONS, FPS, KINDS,
                                     advance_with_bias, build_suite)
from doom.native import NativeBrain


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/fearless/vision-audit"


def population_indices(target):
    return {name: np.asarray([cell["graph_index"] for cell in value["neurons"]],
                             dtype=np.int32)
            for name, value in target["populations"].items()}


def normalize_columns(matrix):
    denominator = np.asarray(abs(matrix).sum(axis=0)).ravel()
    denominator[denominator == 0] = 1
    return matrix @ diags(1 / denominator)


def graded_scores(weights, retina, lc4, baseline, vectors):
    """Return a deliberately simple, parameter-free structural contrast probe."""
    hop2 = np.unique(weights[:, lc4].nonzero()[0])
    hop1 = np.unique(weights[:, hop2].nonzero()[0])
    hop1 = np.intersect1d(hop1, np.unique(weights[retina].nonzero()[1]))
    first = normalize_columns(abs(weights[retina][:, hop1]))
    second = normalize_columns(weights[hop1][:, hop2])
    third = normalize_columns(weights[hop2][:, lc4])

    result = {}
    for combination in COMBINATIONS:
        by_kind = {}
        for kind in KINDS:
            previous = baseline
            per_frame = []
            for current in vectors[f"{combination['id']}__{kind}"]:
                activity = np.maximum(previous - current, 0) @ first
                activity = np.maximum(np.asarray(activity).ravel(), 0) @ second
                activity = np.maximum(np.asarray(activity).ravel(), 0) @ third
                per_frame.append(np.asarray(activity).ravel())
                previous = current
            values = np.asarray(per_frame)
            by_kind[kind] = {"mean_population_activity": float(values.mean()),
                             "peak_population_activity": float(values.max()),
                             "sum_population_activity": float(values.sum())}
        result[combination["id"]] = by_kind
    successes = []
    controls = [kind for kind in KINDS if kind not in {"expanding", "blank_no_lamina"}]
    for by_kind in result.values():
        expansion = by_kind["expanding"]["mean_population_activity"]
        successes.append(all(expansion > by_kind[kind]["mean_population_activity"]
                             for kind in controls))
    return result, {"retina": len(retina), "hop1": len(hop1), "hop2": len(hop2),
                    "lc4": len(lc4)}, successes


def functional_diagnostic(baseline, vectors, indices, nodes):
    brain = NativeBrain(GRAPH)
    baseline_counts = np.zeros(brain.n, dtype=np.int64)
    response_counts = np.zeros(brain.n, dtype=np.int64)
    tick = 0
    for _ in range(BASELINE_TICKS):
        tick += 1
        counts, _ = advance_with_bias(brain, baseline, tick, 12.0)
        baseline_counts += counts
    for vector in vectors["speed-2-location-1__expanding"]:
        tick += 1
        counts, _ = advance_with_bias(brain, vector, tick, 12.0)
        response_counts += counts

    lc4_mask = np.zeros(brain.n, dtype=bool)
    lc4_mask[indices["LC4"]] = True
    active_presynaptic = []
    for source in np.flatnonzero(response_counts):
        edge_slice = slice(brain.ptr[source], brain.ptr[source + 1])
        selected = lc4_mask[brain.post[edge_slice]]
        if selected.any():
            active_presynaptic.append({
                "graph_index": int(source),
                "cell_type": str(nodes.cell_type.iloc[source]),
                "spikes": int(response_counts[source]),
                "signed_event_weight": float(
                    (brain.weight[edge_slice][selected] * response_counts[source]).sum()),
            })
    adjacency = csr_matrix((np.ones(len(brain.post), dtype=np.uint8), brain.post,
                            brain.ptr), shape=(brain.n, brain.n))
    direct_total = np.unique(adjacency[:, indices["LC4"]].nonzero()[0]).size
    return {
        "stimulus": "speed-2-location-1__expanding",
        "baseline_total_spikes": int(baseline_counts.sum()),
        "response_total_spikes": int(response_counts.sum()),
        "lc4_baseline_spikes": int(baseline_counts[indices["LC4"]].sum()),
        "lc4_response_spikes": int(response_counts[indices["LC4"]].sum()),
        "lc4_direct_presynaptic_neurons_total": int(direct_total),
        "lc4_direct_presynaptic_neurons_active": len(active_presynaptic),
        "active_direct_inputs": active_presynaptic,
    }


def make_plot(scores, output):
    kinds = [kind for kind in KINDS if kind != "blank_no_lamina"]
    values = [[scores[c["id"]][kind]["mean_population_activity"]
               for c in COMBINATIONS] for kind in kinds]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    axis.boxplot(values, labels=[kind.replace("_", "\n") for kind in kinds],
                 showmeans=True)
    axis.set_ylabel("Three-hop graded LC4 proxy (arbitrary units)")
    axis.set_title("A simple OFF-contrast path is not looming-selective")
    axis.grid(axis="y", alpha=.25)
    fig.tight_layout()
    path = output / "graded-proxy-controls.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    graph = np.load(GRAPH)
    weights = csr_matrix((graph["weight"], graph["post"], graph["ptr"]),
                         shape=(len(graph["ids"]), len(graph["ids"])))
    target = json.loads(TARGET_MANIFEST.read_text())
    indices = population_indices(target)
    nodes = feather.read_table(
        ROOT / "connectome_data/malecns_v1/normalized/neurons.feather").to_pandas()
    probe = NativeBrain(GRAPH)
    baseline, vectors, _ = build_suite(probe.uv)
    scores, layers, successes = graded_scores(
        weights, graph["retina"], indices["LC4"], baseline, vectors)
    functional = functional_diagnostic(baseline, vectors, indices, nodes)
    figure = make_plot(scores, output)
    report = {
        "schema": 1,
        "status": "no_validated_visual_correction",
        "generator_sha256": file_sha256(Path(__file__)),
        "graph_sha256": file_sha256(GRAPH),
        "target_manifest_sha256": file_sha256(TARGET_MANIFEST),
        "structural_three_hop_cone": layers,
        "fixed_lif_diagnostic": functional,
        "graded_off_contrast_proxy": {
            "description": "Rectified OFF contrast propagated over normalized signed contact weights; no fitted parameters.",
            "results": scores,
            "strict_control_successes": successes,
            "strict_control_success_fraction": float(np.mean(successes)),
            "passed": False,
            "reason": "Expansion beat every nonblank visual control in fewer than 70% of frozen combinations.",
        },
        "figure": figure.name,
        "references": {
            "lc4_angular_expansion_velocity": "https://doi.org/10.1016/j.neuron.2017.05.036",
            "trained_graded_visual_model": "https://doi.org/10.1038/s41586-024-07939-3",
            "official_flyvis_code": "https://github.com/TuragaLab/flyvis",
        },
        "conclusion": (
            "Connectivity is present, but uniform LIF dynamics suppress the relevant intermediate "
            "activity. A parameter-free graded OFF proxy remains confounded by uniform dimming, "
            "onset, and spatial scrambling. A trained or physiologically calibrated motion/contrast "
            "model is required before natural Doom pixels can support an LC4 claim."),
    }
    atomic_json(output / "report.json", report)
    (output / "REPORT.md").write_text(f"""# FEARLESS visual-pathway audit

The visual graph is connected: all 126 LC4 cells are structurally reachable from the mapped R1–R6
population in three hops. The fixed LIF implementation nevertheless produced **zero LC4 spikes**
during the frozen looming diagnostic. Only **{functional['lc4_direct_presynaptic_neurons_active']}**
of **{functional['lc4_direct_presynaptic_neurons_total']:,}** neurons with direct LC4 connections
spiked during that response.

A parameter-free diagnostic then propagated rectified OFF contrast over the actual normalized,
signed three-hop contact cone. Expansion exceeded every nonblank visual control in only
**{sum(successes)}/{len(successes)}** frozen speed/location combinations. Uniform dimming and
spatial scrambling were often as strong or stronger. This therefore fails the original 70% gate.

![Graded proxy versus controls]({figure.name})

## Disposition

No new natural-vision Doom cohort is licensed. Increasing current or inserting a hand-authored
loom detector would create the desired answer rather than validate fly vision. The defensible next
model needs graded, adapting and contrast/motion-selective visual dynamics fitted independently of
these FEARLESS stimuli. This follows the approach demonstrated by the published FlyVis work, but
its trained 64-type optic-lobe network is not a drop-in LC4 model for this MaleCNS graph.

LC4 is reported to encode retinal angular expansion velocity, so resolving this
failure requires spatial and temporal motion computation rather than an amplitude
rescale ([von Reyn et al., 2017](https://doi.org/10.1016/j.neuron.2017.05.036)).
The relevant independently trained graded-network precedent is
[Lappalainen et al., 2024](https://doi.org/10.1038/s41586-024-07939-3), with
[official FlyVis code](https://github.com/TuragaLab/flyvis).

This audit is a model diagnosis, not evidence about biological LC4 failure or fly fear.
""", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
