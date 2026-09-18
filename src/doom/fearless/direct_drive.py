"""Frozen direct LC4+LPLC2-drive bridge between the public Space and Doomfly."""

import json
from pathlib import Path

import numpy as np

from doom.fearless.interventions import CONDITIONS, apply_intervention
from doom.fearless.run import (GRAPH, POPULATIONS, TARGET_MANIFEST, array_sha256,
                               atomic_json, file_sha256, population_indices)
from doom.native import BUILD, NativeBrain


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/fearless/direct-drive"
FPS = 35
BASELINE_TICKS = 18
DRIVE_TICKS = 35
# Frozen before execution. Units are the existing model's additive input-current units.
AMPLITUDES = (0.0, 5.0, 10.0, 20.0, 30.0)
DRIVEN_POPULATIONS = ("LC4", "LPLC2")


def advance(brain, luminance, tick, extra_drive):
    target = int(round(tick * 10000 / FPS))
    steps = target - brain.cursor
    counts, _ = brain.step(luminance, steps * brain.dt, extra_drive=extra_drive)
    return counts, steps * brain.dt / 1000


def run_one(condition, amplitude, target):
    brain = NativeBrain(GRAPH)
    intervention = apply_intervention(brain, condition, target)
    indices = population_indices(target)
    driven_indices = np.unique(np.concatenate([indices[name] for name in DRIVEN_POPULATIONS]))
    drive = np.zeros(brain.n, dtype=np.float32)
    drive[driven_indices] = amplitude
    blank = np.zeros(len(brain.retina), dtype=np.float32)
    totals = {"baseline": {name: 0 for name in POPULATIONS},
              "drive": {name: 0 for name in POPULATIONS}}
    tick = 0
    baseline_seconds = drive_seconds = 0.0
    for _ in range(BASELINE_TICKS):
        tick += 1
        counts, seconds = advance(brain, blank, tick, None)
        baseline_seconds += seconds
        for name, index in indices.items():
            totals["baseline"][name] += int(counts[index].sum())
    for _ in range(DRIVE_TICKS):
        tick += 1
        counts, seconds = advance(brain, blank, tick, drive if amplitude else None)
        drive_seconds += seconds
        for name, index in indices.items():
            totals["drive"][name] += int(counts[index].sum())
    populations = {}
    for name, index in indices.items():
        baseline_rate = totals["baseline"][name] / (len(index) * baseline_seconds)
        drive_rate = totals["drive"][name] / (len(index) * drive_seconds)
        populations[name] = {"baseline_spikes": totals["baseline"][name],
                             "drive_spikes": totals["drive"][name],
                             "baseline_rate_hz_per_neuron": baseline_rate,
                             "drive_rate_hz_per_neuron": drive_rate,
                             "baseline_subtracted_rate_hz_per_neuron": drive_rate - baseline_rate}
    return {"condition": condition, "amplitude": amplitude,
            "driven_populations": list(DRIVEN_POPULATIONS),
            "driven_neurons": int(len(driven_indices)), "intervention": intervention,
            "populations": populations,
            "fixed_weight_sha256_after": array_sha256(brain.weight)}


def run(output=OUTPUT):
    output = Path(output)
    target = json.loads(TARGET_MANIFEST.read_text())
    rows = [run_one(condition, amplitude, target)
            for amplitude in AMPLITUDES for condition in CONDITIONS]
    contrasts = []
    for amplitude in AMPLITUDES:
        values = {row["condition"]: row for row in rows if row["amplitude"] == amplitude}
        for condition in CONDITIONS[1:]:
            contrasts.append({"amplitude": amplitude, "condition": condition,
                              "dnp01_drive_spike_difference_vs_intact":
                              values[condition]["populations"]["DNp01"]["drive_spikes"] -
                              values["intact"]["populations"]["DNp01"]["drive_spikes"],
                              "lc4_drive_spike_difference_vs_intact":
                              values[condition]["populations"]["LC4"]["drive_spikes"] -
                              values["intact"]["populations"]["LC4"]["drive_spikes"]})
    result = {"schema": 1, "purpose": "exploratory direct-drive bridge; not a visual stimulus",
              "cadence_hz": FPS, "baseline_ticks": BASELINE_TICKS, "drive_ticks": DRIVE_TICKS,
              "amplitudes": AMPLITUDES, "rows": rows, "contrasts": contrasts,
              "graph_sha256": file_sha256(GRAPH), "target_manifest_sha256": file_sha256(TARGET_MANIFEST),
              "kernel_build": BUILD,
              "interpretation_constraint": (
                  "This directly injects the named populations and cannot establish retinal or Doom visual recruitment.")}
    atomic_json(output / "report.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
