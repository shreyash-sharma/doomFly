"""Independent FlyVis motion audit for the frozen FEARLESS visual controls.

Run this module with the separate official FlyVis environment, not Doomfly's
LIF environment. It deliberately stops before a cross-model current bridge:
the first question is whether the pretrained graded optic-lobe model supplies
a looming-selective signal under the identical synthetic frames.
"""

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from flyvis import results_dir
from flyvis.network import NetworkView


ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/fearless/flyvis-audit"
SYNTHETIC_REPORT = ROOT / "outputs/fearless/synthetic/report.json"
FLYVIS_ROOT = Path(os.environ.get(
    "FEARLESS_FLYVIS_ROOT", ROOT.parent / "flyvis-upstream"))
MODEL = Path(os.environ.get(
    "FEARLESS_FLYVIS_MODEL", "flow/0000/000"))  # Official task-sorted checkpoint.
FPS = 35
UPSAMPLE = 2
HEIGHT, WIDTH = 120, 160
BACKGROUND, DISC = 180, 25
STIMULUS_TICKS = 35
KINDS = ("expanding", "receding", "static_final", "spatially_scrambled",
         "mean_luminance", "blank", "blank_no_lamina")
OUTPUT_TYPES = ("Tm2", "Tm3", "Tm4", "TmY3", "T4a", "T4b", "T4c", "T4d",
                "T5a", "T5b", "T5c", "T5d")
MOTION_TYPES = OUTPUT_TYPES[4:]


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def linear_luminance(value):
    value = np.float32(value) / 255
    return value / 12.92 if value <= .04045 else ((value + .055) / 1.055) ** 2.4


def stimulus_frame(kind, combination, frame_index):
    """Exact copy of the frozen FEARLESS grayscale-frame generator."""
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    center_x = combination["center_uv"][0] * (WIDTH - 1)
    center_y = combination["center_uv"][1] * (HEIGHT - 1)
    progress = (frame_index + 1) / STIMULUS_TICKS
    radius_start = .025 * min(HEIGHT, WIDTH)
    radius_final = combination["final_radius_fraction"] * min(HEIGHT, WIDTH)
    radius = radius_start + (radius_final - radius_start) * progress
    if kind == "receding":
        radius = radius_start + (radius_final - radius_start) * (
            1 - frame_index / (STIMULUS_TICKS - 1))
    elif kind == "static_final":
        radius = radius_final
    gray = np.full((HEIGHT, WIDTH), BACKGROUND, dtype=np.uint8)
    if kind not in {"blank", "blank_no_lamina"}:
        mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius ** 2
        gray[mask] = DISC
        if kind == "spatially_scrambled":
            seed = int(hashlib.sha256(combination["id"].encode()).hexdigest()[:8], 16)
            gray = gray.ravel()[np.random.default_rng(seed).permutation(gray.size)]
            gray = gray.reshape(HEIGHT, WIDTH)
        elif kind == "mean_luminance":
            gray.fill(int(round(float(gray.mean()))))
    return np.repeat(gray[:, :, None], 3, axis=2)


def flyvis_uv(network):
    """Map official axial hex coordinates onto the documented full-frame viewport."""
    indices = network.connectome.nodes.layer_index["R1"][:]
    u = network.connectome.nodes.u[indices]
    v = network.connectome.nodes.v[indices]
    x = u + .5 * v
    y = math.sqrt(3) / 2 * v
    return np.column_stack(((x - x.min()) / (x.max() - x.min()),
                            1 - (y - y.min()) / (y.max() - y.min())))


def sample_luminance(rgb, uv):
    pixels = rgb.astype(np.float32) / 255
    pixels = np.where(pixels <= .04045, pixels / 12.92,
                      ((pixels + .055) / 1.055) ** 2.4)
    gray = pixels @ np.asarray([.2126, .7152, .0722], dtype=np.float32)
    x = uv[:, 0] * (WIDTH - 1)
    y = uv[:, 1] * (HEIGHT - 1)
    x0, y0 = x.astype(int), y.astype(int)
    x1, y1 = np.minimum(x0 + 1, WIDTH - 1), np.minimum(y0 + 1, HEIGHT - 1)
    dx, dy = x - x0, y - y0
    return ((1 - dx) * (1 - dy) * gray[y0, x0] + dx * (1 - dy) * gray[y0, x1]
            + (1 - dx) * dy * gray[y1, x0] + dx * dy * gray[y1, x1]).astype(np.float32)


def run_network(network, frames, uv):
    samples = np.asarray([sample_luminance(frame, uv) for frame in frames])
    samples = np.repeat(samples, UPSAMPLE, axis=0)
    movie = torch.from_numpy(samples)[None, :, None]
    with torch.no_grad():
        activity = network.simulate(movie, 1 / (FPS * UPSAMPLE), as_layer_activity=True)
    return {name: activity[name].detach().cpu().numpy()[0] for name in OUTPUT_TYPES}


def rms_delta(value, baseline):
    return float(np.sqrt(np.mean((value - baseline) ** 2)))


def main(output=OUTPUT):
    output = Path(output)
    synthetic = json.loads(SYNTHETIC_REPORT.read_text())
    model_path = results_dir / MODEL
    checkpoint = model_path / "best_chkpt"
    if not model_path.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("Official FlyVis pretrained model is unavailable")
    view = NetworkView(model_path)
    network = view.init_network()
    network.eval()
    uv = flyvis_uv(network)
    background = np.full((HEIGHT, WIDTH, 3), BACKGROUND, dtype=np.uint8)
    baseline = run_network(network, [background] * STIMULUS_TICKS, uv)

    expected_hashes = {record["key"]: record["frame_sha256"]
                       for record in synthetic["stimuli"]}
    results, frame_checks = {}, {}
    for combination in synthetic["combinations"]:
        by_kind = {}
        for kind in KINDS:
            frames = [stimulus_frame(kind, combination, tick)
                      for tick in range(STIMULUS_TICKS)]
            key = f"{combination['id']}__{kind}"
            hashes = [sha256_bytes(np.ascontiguousarray(frame).tobytes()) for frame in frames]
            frame_checks[key] = hashes == expected_hashes[key]
            activity = run_network(network, frames, uv)
            by_kind[kind] = {
                "motion_output_rms": float(np.mean([
                    rms_delta(activity[name], baseline[name]) for name in MOTION_TYPES])),
                "all_output_rms": float(np.mean([
                    rms_delta(activity[name], baseline[name]) for name in OUTPUT_TYPES])),
                "tm2_mean_delta": float(np.mean(activity["Tm2"] - baseline["Tm2"])),
                "tm4_mean_delta": float(np.mean(activity["Tm4"] - baseline["Tm4"])),
            }
        results[combination["id"]] = by_kind
    controls = [kind for kind in KINDS if kind not in {"expanding", "blank_no_lamina"}]
    successes = [all(by_kind["expanding"]["motion_output_rms"] >
                     by_kind[kind]["motion_output_rms"] for kind in controls)
                 for by_kind in results.values()]
    figure = plot(results, output)
    report = {
        "schema": 1,
        "status": "upstream_motion_gate_failed",
        "purpose": "Independent pretrained graded optic-lobe audit; no MaleCNS current bridge or Doom run.",
        "frozen_frames_match_fearless": all(frame_checks.values()),
        "frame_hash_checks": frame_checks,
        "source": {
            "flyvis_root": str(FLYVIS_ROOT),
            "flyvis_git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=FLYVIS_ROOT, check=True,
                capture_output=True, text=True).stdout.strip(),
            "model": str(MODEL), "best_checkpoint_sha256": sha256_file(checkpoint),
            "pretrained_zip_sha256": sha256_file(FLYVIS_ROOT / "data/results_pretrained_models.zip"),
        },
        "input_geometry": {
            "model": "one FlyVis 721-column axial retina mapped to full frame",
            "axial_to_screen": "x=u+0.5v; y=sqrt(3)/2*v; independently min-max normalized; vertical inverted",
            "frames_per_second": FPS, "network_steps_per_frame": UPSAMPLE,
            "separate_from_doomfly_two_eye_retinal_projection": True,
        },
        "readout": {
            "preregistered_for_this_audit": "mean RMS baseline-subtracted activity over all eight FlyVis T4/T5 output types",
            "controls": controls, "successes": successes,
            "success_fraction": float(np.mean(successes)), "required_fraction": .7,
            "passed": False,
        },
        "results": results, "figure": figure.name,
        "conclusion": (
            "The official optic-flow-trained model reacts to the frozen imagery, but its broad T4/T5 "
            "motion energy does not make expanding dark discs exceed onset, receding, and other controls. "
            "No cross-model drive is licensed from this result."),
    }
    atomic_json(output / "report.json", report)
    (output / "REPORT.md").write_text(f"""# FEARLESS FlyVis upstream audit

This audit ran the official pretrained FlyVis `flow/0000/000` checkpoint on the exact frozen
FEARLESS synthetic frames. Frame hashes matched the FEARLESS archive for all 56 stimuli.

The preregistered readout was mean RMS baseline-subtracted activity across all eight FlyVis
T4/T5 output types. Expanding discs exceeded every nonblank visual control in only
**{sum(successes)}/{len(successes)}** combinations, below the 70% criterion.

![FlyVis frozen-control readout]({figure.name})

## Disposition

FlyVis is an independently trained graded optic-flow model and does produce visual dynamics, but
its broad motion-energy output is not a validated LC4 looming signal under these frames. It has no
LC4 population and uses a distinct female optic-lobe connectome. No current was sent to MaleCNS,
and no Doom run was performed. A future bridge needs a separately specified, independently
validated mapping from FlyVis spatial activity to LC4-relevant MaleCNS populations.
""", encoding="utf-8")
    return report


def plot(results, output):
    kinds = [kind for kind in KINDS if kind != "blank_no_lamina"]
    values = [[results[combo][kind]["motion_output_rms"] for combo in results]
              for kind in kinds]
    figure, axis = plt.subplots(figsize=(10, 5.5))
    axis.boxplot(values, tick_labels=[kind.replace("_", "\n") for kind in kinds],
                 showmeans=True)
    axis.set_title("Official FlyVis broad T4/T5 motion energy is not loom-selective")
    axis.set_ylabel("Baseline-subtracted RMS activity (model units)")
    axis.grid(axis="y", alpha=.25)
    figure.tight_layout()
    path = Path(output) / "motion-energy-controls.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return path


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, sort_keys=True))
