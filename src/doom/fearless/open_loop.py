"""Replay one recorded Doom retinal stream without consulting or running Doom."""

import json
from pathlib import Path

import numpy as np

from doom.engine import NeuralControls
from doom.fearless.interventions import CONDITIONS, apply_intervention
from doom.fearless.run import (GRAPH, INTERVENTION_MANIFEST, MODEL_MANIFEST,
                               POPULATIONS, TARGET_MANIFEST, array_sha256,
                               atomic_json, file_sha256, population_indices)
from doom.native import BUILD, NativeBrain


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "outputs/fearless/open-loop/source/runs/41027-intact"
DEFAULT_OUTPUT = ROOT / "outputs/fearless/open-loop"


def replay_condition(condition, vectors, intervals_ms, frame_hashes, target, readouts,
                     output, direct_drive_amplitude=0.0):
    brain = NativeBrain(GRAPH)
    intervention = apply_intervention(brain, condition, target)
    controls = NeuralControls(readouts, mode="bci")
    indices = population_indices(target)
    direct_drive = None
    if direct_drive_amplitude:
        direct_drive = np.zeros(brain.n, dtype=np.float32)
        direct_drive[np.unique(np.concatenate([indices["LC4"], indices["LPLC2"]]))] = direct_drive_amplitude
    events = []
    for tick, (vector, interval_ms, frame_hash) in enumerate(
            zip(vectors, intervals_ms, frame_hashes), 1):
        counts, _ = brain.step(vector, float(interval_ms), extra_drive=direct_drive)
        decoded_full = controls.decode(counts, float(interval_ms) / 1000)
        decoded = {key: decoded_full[key] for key in ("turn", "forward", "attack")}
        populations = {name: int(counts[index].sum()) for name, index in indices.items()}
        events.append({"tick": tick, "neural_interval_ms": float(interval_ms),
                       "source_frame_sha256": str(frame_hash),
                       "input_sha256": array_sha256(vector),
                       "spike_counts_sha256": array_sha256(counts),
                       "populations": populations, "decoded_action_observation_only": decoded})
    path = Path(output) / f"events-{condition}.jsonl"
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w") as stream:
        for event in events:
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")
    partial.replace(path)
    totals = {name: sum(event["populations"][name] for event in events) for name in POPULATIONS}
    return {"condition": condition, "ticks": len(events), "population_total_spikes": totals,
            "intervention": intervention, "events": path.name,
            "events_sha256": file_sha256(path)}, events


def run(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT, direct_drive_amplitude=0.0):
    source, output = Path(source), Path(output)
    source_result = json.loads((source / "run.json").read_text())
    if source_result.get("validation") is not True or source_result.get("condition") != "intact":
        raise ValueError("Source must be a validated intact closed-loop run")
    archive_path = source / source_result["retinal_inputs"]
    with np.load(archive_path) as archive:
        vectors = archive["vectors"].copy()
        intervals = archive["neural_interval_ms"].copy()
        frame_hashes = archive["source_frame_sha256"].copy()
    target = json.loads(TARGET_MANIFEST.read_text())
    readouts = json.loads(MODEL_MANIFEST.read_text())["readouts"]
    output.mkdir(parents=True, exist_ok=True)
    summaries, traces = [], {}
    for condition in CONDITIONS:
        summary, events = replay_condition(condition, vectors, intervals, frame_hashes,
                                           target, readouts, output, direct_drive_amplitude)
        summaries.append(summary); traces[condition] = events
    reference = [(e["input_sha256"], e["source_frame_sha256"], e["neural_interval_ms"])
                 for e in traces["intact"]]
    identical = all(reference == [(e["input_sha256"], e["source_frame_sha256"],
                                   e["neural_interval_ms"]) for e in traces[condition]]
                    for condition in CONDITIONS)
    by_condition = {row["condition"]: row for row in summaries}
    intact = by_condition["intact"]["population_total_spikes"]
    contrasts = {
        condition: {name: by_condition[condition]["population_total_spikes"][name] - intact[name]
                    for name in POPULATIONS}
        for condition in CONDITIONS[1:]
    }
    report = {"schema": 1, "status": "complete", "offline_no_game_instantiated": True,
              "source_run": str(source.resolve().relative_to(ROOT)),
              "source_run_sha256": file_sha256(source / "run.json"),
              "source_retinal_archive_sha256": file_sha256(archive_path),
              "identical_inputs_frames_and_timing": identical,
              "conditions": summaries, "paired_total_spike_contrasts_vs_intact": contrasts,
              "lc4_spiked_in_intact": intact["LC4"] > 0,
              "lc4_spiked_while_output_silenced": by_condition["lc4_silenced"][
                  "population_total_spikes"]["LC4"] > 0,
              "direct_mutation_scope": "selected outgoing weight rows only",
              "direct_drive": {"populations": ["LC4", "LPLC2"],
                               "amplitude": direct_drive_amplitude},
              "graph_sha256": file_sha256(GRAPH), "kernel_build": BUILD,
              "target_manifest_sha256": file_sha256(TARGET_MANIFEST),
              "intervention_manifest_sha256": file_sha256(INTERVENTION_MANIFEST)}
    if not identical:
        raise RuntimeError("Open-loop input or timing mismatch across conditions")
    atomic_json(output / "report.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
