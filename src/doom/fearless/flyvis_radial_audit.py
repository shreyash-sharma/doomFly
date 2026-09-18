"""Exploratory spatial-flow audit for official FlyVis outputs.

This intentionally remains upstream of MaleCNS.  It asks a narrower question
than ``flyvis_audit``: after independently calibrating each T4/T5 output's
screen-direction preference with moving stripes, does its *spatial* activity
form an outward field for the frozen FEARLESS expanding-disc controls?
"""

import json
import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flyvis import results_dir
from flyvis.network import NetworkView

from doom.fearless.flyvis_audit import (
    BACKGROUND, DISC, FPS, HEIGHT, KINDS, MODEL, MOTION_TYPES, OUTPUT,
    STIMULUS_TICKS, UPSAMPLE, WIDTH, atomic_json, flyvis_uv, rms_delta,
    run_network, sample_luminance, stimulus_frame,
)

ROOT = Path(__file__).resolve().parents[2]
RADIAL_OUTPUT = ROOT / "outputs/fearless/flyvis-radial-audit"


def translating_stripe(angle, frame_index):
    """A non-FEARLESS calibration stimulus: a dark stripe translating screenwise."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    x, y = xx / (WIDTH - 1), yy / (HEIGHT - 1)
    direction = np.array([math.cos(angle), math.sin(angle)])
    coordinate = (x - .5) * direction[0] + (y - .5) * direction[1]
    position = -0.75 + 1.5 * frame_index / (STIMULUS_TICKS - 1)
    stripe = np.abs(coordinate - position) <= .10
    gray = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    gray[stripe] = DISC
    return np.repeat(gray[:, :, None], 3, axis=2)


def calibrate_directions(network, baseline, uv):
    """Assign each output type its strongest of four measured screen directions."""
    angles = np.arange(4) * math.pi / 2
    responses = {name: [] for name in MOTION_TYPES}
    movies = [[translating_stripe(angle, tick) for tick in range(STIMULUS_TICKS)]
               for angle in angles]
    activities = run_network_batch(network, movies, uv)
    for activity in activities:
        for name in MOTION_TYPES:
            responses[name].append(rms_delta(activity[name], baseline[name]))
    calibration = {}
    for name, values in responses.items():
        values = np.asarray(values)
        best = int(values.argmax())
        calibration[name] = {
            "responses_by_screen_angle_deg": {
                str(int(round(math.degrees(angle)))): float(value)
                for angle, value in zip(angles, values)
            },
            "assigned_screen_angle_deg": int(round(math.degrees(angles[best]))),
            "direction_selectivity_ratio": float(values[best] / (np.median(values) + 1e-12)),
        }
    return calibration


def run_network_batch(network, movies, uv):
    """Simulate independent movies in one batch to avoid repeated state warmups."""
    samples = np.asarray([[sample_luminance(frame, uv) for frame in frames]
                          for frames in movies])
    samples = np.repeat(samples, UPSAMPLE, axis=1)
    movie = torch.from_numpy(samples)[:, :, None]
    with torch.no_grad():
        activity = network.simulate(movie, 1 / (FPS * UPSAMPLE), as_layer_activity=True)
    raw = {name: activity[name].detach().cpu().numpy() for name in MOTION_TYPES}
    return [{name: raw[name][index] for name in MOTION_TYPES}
            for index in range(len(movies))]


def node_uv(network, name):
    indices = network.connectome.nodes.layer_index[name][:]
    u = network.connectome.nodes.u[indices]
    v = network.connectome.nodes.v[indices]
    # This is the same documented axial-to-screen mapping used for the retinal input.
    x = u + .5 * v
    y = math.sqrt(3) / 2 * v
    return np.column_stack(((x - x.min()) / (x.max() - x.min()),
                            1 - (y - y.min()) / (y.max() - y.min())))


def outward_flow(activity, baseline, network, center, calibration):
    """Positive baseline-subtracted local motion projected onto the outward radial axis."""
    total, weight = 0.0, 0.0
    for name in MOTION_TYPES:
        positions = node_uv(network, name)
        radial = positions - np.asarray(center)
        radial /= np.maximum(np.linalg.norm(radial, axis=1, keepdims=True), 1e-6)
        theta = math.radians(calibration[name]["assigned_screen_angle_deg"])
        preferred = np.array([math.cos(theta), math.sin(theta)])
        projection = radial @ preferred
        delta = np.maximum(activity[name] - baseline[name], 0.0)
        total += float(np.sum(delta * projection[None, :]))
        weight += float(np.sum(delta))
    return total / weight if weight else 0.0


def plot(results, output):
    kinds = [kind for kind in KINDS if kind != "blank_no_lamina"]
    values = [[results[combo][kind] for combo in results] for kind in kinds]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.boxplot(values, tick_labels=[kind.replace("_", "\n") for kind in kinds], showmeans=True)
    axis.axhline(0, color="black", linewidth=.8)
    axis.set_title("Exploratory FlyVis spatial outward-flow readout")
    axis.set_ylabel("Positive = local motion points outward (unitless)")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    path = Path(output) / "radial-flow-controls.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


def main(output=RADIAL_OUTPUT):
    output = Path(output)
    model_path = results_dir / MODEL
    view = NetworkView(model_path)
    network = view.init_network()
    network.eval()
    uv = flyvis_uv(network)
    background = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    baseline = run_network(network, [background] * STIMULUS_TICKS, uv)
    calibration = calibrate_directions(network, baseline, uv)

    from doom.fearless.flyvis_audit import SYNTHETIC_REPORT
    synthetic = json.loads(SYNTHETIC_REPORT.read_text())
    stimuli = []
    for combination in synthetic["combinations"]:
        for kind in KINDS:
            frames = [stimulus_frame(kind, combination, tick) for tick in range(STIMULUS_TICKS)]
            stimuli.append((combination, kind, frames))
    activities = run_network_batch(network, [frames for _, _, frames in stimuli], uv)
    results = {combination["id"]: {} for combination in synthetic["combinations"]}
    for (combination, kind, _), activity in zip(stimuli, activities):
        results[combination["id"]][kind] = outward_flow(
            activity, baseline, network, combination["center_uv"], calibration)
    controls = [kind for kind in KINDS if kind not in {"expanding", "blank_no_lamina"}]
    successes = [all(row["expanding"] > row[kind] for kind in controls)
                 for row in results.values()]
    figure = plot(results, output)
    report = {
        "schema": 1,
        "status": "exploratory_spatial_flow_only",
        "purpose": "No MaleCNS bridge, LC4 claim, or Doom run. Spatial feature audit only.",
        "model": str(MODEL),
        "calibration": {
            "stimulus": "dark translating stripe, four screen directions, not a FEARLESS stimulus",
            "method": "Each T4/T5 type assigned its strongest baseline-subtracted RMS direction.",
            "results": calibration,
        },
        "readout": {
            "definition": "positive baseline-subtracted T4/T5 activity weighted by dot(preferred screen direction, local outward radial axis)",
            "controls": controls, "successes": successes,
            "success_fraction": float(np.mean(successes)), "required_fraction": .7,
            "passed": bool(np.mean(successes) >= .7),
        },
        "results": results, "figure": figure.name,
        "conclusion": "This result is not a direct LC4 response and cannot license a cross-model current bridge by itself.",
    }
    atomic_json(output / "report.json", report)
    (output / "REPORT.md").write_text(f"""# FEARLESS exploratory FlyVis radial-flow audit

This is a feature probe, not a fly-escape model. We first assigned each official FlyVis T4/T5
output a screen-direction label using a separate translating-stripe calibration. We then asked
whether positive output activity at each retinal location pointed away from the known centre of
the frozen synthetic disc.

Expanding discs exceeded all nonblank visual controls in **{sum(successes)}/{len(successes)}**
combinations (criterion: 70%). The figure is a spatial-output diagnostic only; it does not
measure LC4, does not drive MaleCNS, and was not used for a Doom result.

![Radial-flow control comparison]({figure.name})
""", encoding="utf-8")
    return report


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
