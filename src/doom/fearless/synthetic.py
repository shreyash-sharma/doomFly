"""Preregistered deterministic synthetic looming/selectivity gate for FEARLESS."""

import hashlib
import json
from pathlib import Path

import numpy as np

from doom.fearless.interventions import CONDITIONS, apply_intervention
from doom.fearless.run import (GRAPH, INTERVENTION_MANIFEST, POPULATIONS,
                               TARGET_MANIFEST, array_sha256, atomic_json,
                               file_sha256, population_indices)
from doom.game import retinal_samples
from doom.native import BUILD, NativeBrain


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/fearless/synthetic"
FPS = 35
BASELINE_TICKS = 18
STIMULUS_TICKS = 35
HEIGHT, WIDTH = 120, 160
BACKGROUND, DISC = 180, 25
KINDS = ("expanding", "receding", "static_final", "spatially_scrambled",
         "mean_luminance", "blank", "blank_no_lamina")
COMBINATIONS = tuple(
    {"id": f"speed-{speed_index + 1}-location-{location_index + 1}",
     "final_radius_fraction": final_radius, "center_uv": list(center)}
    for speed_index, final_radius in enumerate((0.24, 0.40))
    for location_index, center in enumerate(((0.3, 0.3), (0.7, 0.3), (0.3, 0.7), (0.7, 0.7)))
)


def stimulus_frame(kind, combination, frame_index):
    """Return a deterministic RGB frame; dimensions and colors are frozen above."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    center_x = combination["center_uv"][0] * (WIDTH - 1)
    center_y = combination["center_uv"][1] * (HEIGHT - 1)
    progress = (frame_index + 1) / STIMULUS_TICKS
    radius_start = 0.025 * min(HEIGHT, WIDTH)
    radius_final = combination["final_radius_fraction"] * min(HEIGHT, WIDTH)
    radius = radius_start + (radius_final - radius_start) * progress
    if kind == "receding":
        radius = radius_start + (radius_final - radius_start) * (1 - frame_index / (STIMULUS_TICKS - 1))
    elif kind == "static_final":
        radius = radius_final
    if kind in {"blank", "blank_no_lamina"}:
        gray = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    else:
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        gray = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
        gray[mask] = DISC
        if kind == "spatially_scrambled":
            seed = int(hashlib.sha256(combination["id"].encode()).hexdigest()[:8], 16)
            permutation = np.random.default_rng(seed).permutation(gray.size)
            gray = gray.ravel()[permutation].reshape(gray.shape)
        elif kind == "mean_luminance":
            gray.fill(int(round(float(gray.mean()))))
    return np.repeat(gray[:, :, None], 3, axis=2)


def build_suite(uv):
    baseline_frame = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    baseline_vector = retinal_samples(baseline_frame, uv)
    vectors, records = {}, []
    for combination in COMBINATIONS:
        for kind in KINDS:
            frames = [stimulus_frame(kind, combination, index) for index in range(STIMULUS_TICKS)]
            inputs = np.asarray([retinal_samples(frame, uv) for frame in frames], dtype=np.float32)
            key = f"{combination['id']}__{kind}"
            vectors[key] = inputs
            records.append({"key": key, "combination": combination, "kind": kind,
                            "frame_sha256": [array_sha256(frame) for frame in frames],
                            "input_sha256": [array_sha256(value) for value in inputs]})
    return baseline_vector, vectors, records


def advance(brain, vector, tick):
    target = int(round(tick * 10000 / FPS))
    steps = target - brain.cursor
    counts, _ = brain.step(vector, steps * brain.dt)
    return counts, steps * brain.dt / 1000


def replay(vectors, baseline_vector, indices, condition="intact", target_manifest=None,
           lamina_bias=12.0):
    brain = NativeBrain(GRAPH)
    intervention = apply_intervention(brain, condition, target_manifest)
    baseline = {name: 0 for name in POPULATIONS}
    response = {name: 0 for name in POPULATIONS}
    tick = 0
    baseline_seconds = response_seconds = 0.0
    for _ in range(BASELINE_TICKS):
        tick += 1
        counts, interval = advance_with_bias(brain, baseline_vector, tick, lamina_bias)
        baseline_seconds += interval
        for name in POPULATIONS:
            baseline[name] += int(counts[indices[name]].sum())
    for vector in vectors:
        tick += 1
        counts, interval = advance_with_bias(brain, vector, tick, lamina_bias)
        response_seconds += interval
        for name in POPULATIONS:
            response[name] += int(counts[indices[name]].sum())
    result = {}
    for name in POPULATIONS:
        neurons = len(indices[name])
        baseline_rate = baseline[name] / (neurons * baseline_seconds)
        response_rate = response[name] / (neurons * response_seconds)
        result[name] = {"baseline_spikes": baseline[name], "response_spikes": response[name],
                        "baseline_rate_hz_per_neuron": baseline_rate,
                        "response_rate_hz_per_neuron": response_rate,
                        "baseline_subtracted_rate_hz_per_neuron": response_rate - baseline_rate}
    return result, intervention


def advance_with_bias(brain, vector, tick, lamina_bias):
    target = int(round(tick * 10000 / FPS))
    steps = target - brain.cursor
    counts, _ = brain.step(vector, steps * brain.dt, lamina_bias=lamina_bias)
    return counts, steps * brain.dt / 1000


def evaluate_gate(rows):
    intact = [row for row in rows if row["condition"] == "intact"]
    medians = {kind: float(np.median([row["populations"]["LC4"][
        "baseline_subtracted_rate_hz_per_neuron"] for row in intact if row["kind"] == kind]))
               for kind in KINDS}
    controls = [kind for kind in KINDS if kind != "expanding"]
    successes = []
    for combination in COMBINATIONS:
        matching = {row["kind"]: row["populations"]["LC4"][
            "baseline_subtracted_rate_hz_per_neuron"] for row in intact
                    if row["combination_id"] == combination["id"]}
        successes.append(matching["expanding"] >= 1 and all(
            matching["expanding"] - matching[kind] >= 1 for kind in controls))
    fraction = float(np.mean(successes))
    passed = (medians["expanding"] >= 1
              and all(medians["expanding"] - medians[kind] >= 1 for kind in controls)
              and fraction >= 0.7)
    return {"passed": bool(passed), "lc4_median_baseline_subtracted_rate": medians,
            "combination_successes": successes, "combination_success_fraction": fraction,
            "thresholds": {"expansion_minimum_hz_per_neuron": 1.0,
                           "control_margin_hz_per_neuron": 1.0,
                           "minimum_combination_fraction": 0.7}}


def run(output=OUTPUT):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    target = json.loads(TARGET_MANIFEST.read_text())
    indices = population_indices(target)
    probe = NativeBrain(GRAPH)
    baseline, vectors, stimulus_records = build_suite(probe.uv)
    archive_partial = output / "retinal-input-suite.npz.partial"
    archive = output / "retinal-input-suite.npz"
    with archive_partial.open("wb") as stream:
        np.savez_compressed(stream, baseline=baseline, **vectors)
    archive_partial.replace(archive)
    rows = []
    for record in stimulus_records:
        bias = 0.0 if record["kind"] == "blank_no_lamina" else 12.0
        populations, intervention = replay(vectors[record["key"]], baseline, indices,
                                           "intact", target, bias)
        rows.append({"combination_id": record["combination"]["id"], "kind": record["kind"],
                     "condition": "intact", "lamina_bias": bias,
                     "populations": populations, "intervention": intervention})
    gate = evaluate_gate(rows)
    # Only a passed gate licenses lesion replay; failure is a hard protocol stop.
    if gate["passed"]:
        for record in stimulus_records:
            if record["kind"] != "expanding":
                continue
            for condition in CONDITIONS[1:]:
                populations, intervention = replay(vectors[record["key"]], baseline, indices,
                                                   condition, target, 12.0)
                rows.append({"combination_id": record["combination"]["id"],
                             "kind": record["kind"], "condition": condition,
                             "lamina_bias": 12.0, "populations": populations,
                             "intervention": intervention})
    result = {"schema": 1, "status": "pass" if gate["passed"] else "fail",
              "cadence_hz": FPS, "baseline_ticks": BASELINE_TICKS,
              "stimulus_ticks": STIMULUS_TICKS, "frame_shape_rgb": [HEIGHT, WIDTH, 3],
              "colors_rgb_uint8": {"background": BACKGROUND, "disc": DISC},
              "combinations": COMBINATIONS, "stimuli": stimulus_records, "results": rows,
              "gate": gate, "retinal_input_archive": archive.name,
              "retinal_input_archive_sha256": file_sha256(archive),
              "generator_sha256": file_sha256(Path(__file__)),
              "graph_sha256": file_sha256(GRAPH),
              "target_manifest_sha256": file_sha256(TARGET_MANIFEST),
              "intervention_manifest_sha256": file_sha256(INTERVENTION_MANIFEST),
              "kernel_build": BUILD,
              "note": "Synthetic selectivity is a model-level gate, not physiological validation."}
    atomic_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    value = run()
    print(json.dumps(value["gate"], indent=2, sort_keys=True))
