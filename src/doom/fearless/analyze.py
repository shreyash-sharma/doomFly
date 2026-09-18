"""Validate, summarize, and plot the exploratory direct-drive Doom pilot."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from doom.fearless.run import POPULATIONS, file_sha256, reconstruct_metrics
from doom.game import Game


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "outputs/fearless/direct-doom/pilot"
OPEN_LOOP = ROOT / "outputs/fearless/direct-doom/open-loop-replay/report.json"
CONDITIONS = ("intact", "lc4_random_matched", "lc4_silenced")
LABELS = {"intact": "Intact", "lc4_random_matched": "Matched clamp",
          "lc4_silenced": "LC4 output clamp"}
COLORS = {"intact": "#3b82f6", "lc4_random_matched": "#94a3b8",
          "lc4_silenced": "#ef4444"}
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_SAMPLES = 10_000


def load_runs(root=PILOT):
    rows = []
    for path in sorted(Path(root).glob("runs/*/run.json")):
        result = json.loads(path.read_text())
        events_path = path.parent / result["events"]
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        if result["events_sha256"] != file_sha256(events_path):
            raise ValueError(f"Event hash mismatch: {events_path}")
        if reconstruct_metrics(events, result["cap_game_seconds"]) != result["metrics"]:
            raise ValueError(f"Metric reconstruction mismatch: {path}")
        if result.get("validation") is not True or result["direct_drive"]["amplitude"] != 20.0:
            raise ValueError(f"Invalid graph or stimulation record: {path}")
        metric = result["metrics"]
        row = {"seed": result["seed"], "condition": result["condition"],
               "run_id": result["run_id"], "status": result["status"],
               "restricted_survival_seconds": metric["restricted_survival_seconds"],
               "right_censored": metric["right_censored"],
               "time_to_death_seconds": metric["time_to_death_seconds"],
               "cumulative_damage": metric["cumulative_damage"],
               "health_gained": metric["health_gained"], "final_health": metric["final_health"],
               "kills": metric["kills"], "attack_button_ticks": metric["attack_button_ticks"],
               "attack_fraction": metric["attack_button_ticks"] / metric["game_ticks"],
               "forward_action_mean": metric["forward_action_mean"],
               "forward_active_fraction": metric["forward_active_fraction"],
               "absolute_turn_mean": metric["absolute_turn_mean"],
               "turning_active_fraction": metric["turning_active_fraction"],
               "action_entropy_nats": metric["action_entropy_nats"],
               "game_seconds": metric["game_seconds"],
               "process_wall_seconds": result["process_wall_seconds"],
               "peak_rss_bytes": result["peak_rss_bytes"]}
        for population in POPULATIONS:
            values = metric["populations"][population]
            row[f"{population}_total_spikes"] = values["total_spikes"]
            row[f"{population}_mean_rate_hz_per_neuron"] = values["mean_rate_hz_per_neuron"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    if len(frame) != 30 or set(frame.condition) != set(CONDITIONS):
        raise ValueError("Pilot must contain exactly 30 runs across three conditions")
    counts = frame.groupby(["seed", "condition"]).size()
    if len(counts) != 30 or not (counts == 1).all():
        raise ValueError("Pilot seed-condition cells are not unique and complete")
    return frame


def paired_table(runs):
    metrics = ["restricted_survival_seconds", "cumulative_damage", "final_health", "kills",
               "attack_fraction", "forward_action_mean", "absolute_turn_mean",
               "action_entropy_nats"] + [f"{name}_total_spikes" for name in POPULATIONS] + [
               f"{name}_mean_rate_hz_per_neuron" for name in POPULATIONS]
    rows = []
    for seed in sorted(runs.seed.unique()):
        local = runs[runs.seed == seed].set_index("condition")
        for condition in CONDITIONS[1:]:
            row = {"seed": seed, "condition": condition, "reference": "intact"}
            for metric in metrics:
                row[metric] = local.loc[condition, metric] - local.loc["intact", metric]
            rows.append(row)
    return pd.DataFrame(rows), metrics


def interval(values, rng):
    values = np.asarray(values, dtype=float)
    samples = values[rng.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))]
    estimate = float(values.mean())
    low, high = np.percentile(samples.mean(axis=1), [2.5, 97.5])
    return {"mean_paired_difference": estimate, "median_paired_difference": float(np.median(values)),
            "iqr_paired_difference": [float(np.percentile(values, 25)), float(np.percentile(values, 75))],
            "bootstrap_mean_95_percent_ci": [float(low), float(high)]}


def summarize(runs, paired, metrics):
    condition_summary = {}
    for condition in CONDITIONS:
        local = runs[runs.condition == condition]
        condition_summary[condition] = {
            "n": len(local), "censored": int(local.right_censored.sum()),
            "metrics": {metric: {"median": float(local[metric].median()),
                                  "q1": float(local[metric].quantile(.25)),
                                  "q3": float(local[metric].quantile(.75))}
                        for metric in metrics}}
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    effects = {condition: {metric: interval(
        paired.loc[paired.condition == condition, metric], rng) for metric in metrics}
        for condition in CONDITIONS[1:]}
    return {"schema": 1, "analysis": "exploratory direct LC4+LPLC2 current-20 Doom pilot",
            "n_paired_seeds": 10, "cap_seconds": 30,
            "bootstrap": {"seed": BOOTSTRAP_SEED, "samples": BOOTSTRAP_SAMPLES,
                          "statistic": "mean paired difference"},
            "condition_summary": condition_summary, "paired_effects_vs_intact": effects,
            "resource_summary": {"total_process_wall_seconds": float(runs.process_wall_seconds.sum()),
                                 "maximum_peak_rss_bytes": int(runs.peak_rss_bytes.max())},
            "interpretation_constraint": "Direct current is not visual recruitment or biological fear."}


def style(axis, title, ylabel):
    axis.set_title(title, loc="left", fontweight="bold")
    axis.set_ylabel(ylabel); axis.grid(axis="y", alpha=.22)
    axis.spines[["top", "right"]].set_visible(False)


def paired_lines(runs, metric, title, ylabel, path):
    fig, axis = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(CONDITIONS))
    for seed in sorted(runs.seed.unique()):
        local = runs[runs.seed == seed].set_index("condition")
        axis.plot(x, [local.loc[c, metric] for c in CONDITIONS], color="#94a3b8", alpha=.55, marker="o")
    medians = [runs.loc[runs.condition == c, metric].median() for c in CONDITIONS]
    axis.plot(x, medians, color="#111827", linewidth=3, marker="D", label="Condition median")
    axis.set_xticks(x, [LABELS[c] for c in CONDITIONS]); style(axis, title, ylabel); axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_actions(runs, path):
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.4))
    specs = [("attack_fraction", "Attack fraction"), ("forward_action_mean", "Mean forward action"),
             ("absolute_turn_mean", "Mean |turn|")]
    for axis, (metric, title) in zip(axes, specs):
        values = [runs.loc[runs.condition == c, metric] for c in CONDITIONS]
        box = axis.boxplot(values, patch_artist=True, widths=.55)
        for patch, condition in zip(box["boxes"], CONDITIONS): patch.set_facecolor(COLORS[condition])
        axis.set_xticks(range(1, 4), ["Intact", "Matched", "LC4"], rotation=15)
        style(axis, title, metric.replace("_", " "))
    fig.suptitle("BCI actions under direct LC4+LPLC2 drive (n=10 paired seeds)", fontweight="bold")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_open_loop(path):
    report = json.loads(OPEN_LOOP.read_text())
    names = ["LC4", "LPLC2", "DNp01", "DNp20", "DNpe017"]
    fig, axis = plt.subplots(figsize=(9, 5.2)); x = np.arange(len(names)); width = .25
    by_condition = {row["condition"]: row["population_total_spikes"] for row in report["conditions"]}
    for offset, condition in enumerate(CONDITIONS):
        axis.bar(x + (offset - 1) * width, [by_condition[condition][name] for name in names],
                 width, label=LABELS[condition], color=COLORS[condition])
    axis.set_yscale("symlog", linthresh=10); axis.set_xticks(x, names)
    style(axis, "Same Doom inputs, offline replay", "Total spikes (symlog)"); axis.legend()
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_effect(paired, summary, path):
    fig, axis = plt.subplots(figsize=(8, 5.2)); metric = "restricted_survival_seconds"
    for x, condition in enumerate(CONDITIONS[1:]):
        values = paired.loc[paired.condition == condition, metric].to_numpy()
        stat = summary["paired_effects_vs_intact"][condition][metric]
        axis.scatter(np.full(len(values), x) + np.linspace(-.08, .08, len(values)), values,
                     color=COLORS[condition], alpha=.8)
        low, high = stat["bootstrap_mean_95_percent_ci"]
        axis.errorbar(x, stat["mean_paired_difference"],
                      yerr=[[stat["mean_paired_difference"] - low], [high - stat["mean_paired_difference"]]],
                      fmt="D", color="#111827", capsize=6, linewidth=2)
    axis.axhline(0, color="#111827", linewidth=1)
    axis.set_xticks([0, 1], [LABELS[c] for c in CONDITIONS[1:]])
    style(axis, "Paired restricted-survival effect vs intact", "Difference (seconds)")
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def capture_doom(path):
    game = Game(seed=41027, scenario="combat_survival")
    try:
        Image.fromarray(game.pixels()).save(path)
    finally:
        game.close()


def run(root=PILOT):
    root = Path(root); plots = root / "plots"; plots.mkdir(parents=True, exist_ok=True)
    runs = load_runs(root); paired, metrics = paired_table(runs); summary = summarize(runs, paired, metrics)
    runs.to_csv(root / "runs.csv", index=False); paired.to_csv(root / "paired-differences.csv", index=False)
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    paired_lines(runs, "restricted_survival_seconds", "Paired survival through the 30-second cap",
                 "Restricted survival (seconds)", plots / "paired-survival.png")
    paired_lines(runs, "cumulative_damage", "Paired cumulative damage", "Damage", plots / "damage.png")
    plot_actions(runs, plots / "actions.png"); plot_open_loop(plots / "open-loop-activity.png")
    plot_effect(paired, summary, plots / "paired-effect.png"); capture_doom(plots / "doom-context.png")
    return summary


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
