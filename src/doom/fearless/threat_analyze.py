"""Summarize the event-triggered artificial-threat matched cohort."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from doom.fearless.run import file_sha256
from doom.fearless.threat_run import CONDITIONS

LABELS = {"uninformed": "Uninformed", "threat_informed": "Threat-informed",
          "lc4_silenced": "LC4-silenced", "random_lesion": "Random lesion",
          "false_alarm": "False alarm", "passive_rate_matched": "Rate-matched"}
COLORS = {"uninformed": "#64748b", "threat_informed": "#2563eb", "lc4_silenced": "#dc2626",
          "random_lesion": "#7c3aed", "false_alarm": "#f59e0b", "passive_rate_matched": "#059669"}
PRIMARY_COMPARISONS = ("threat_informed", "lc4_silenced", "random_lesion", "false_alarm", "passive_rate_matched")


def load(root):
    rows = []
    for path in sorted(Path(root).glob("runs/*/run.json")):
        result = json.loads(path.read_text()); event_path = path.parent / result["events"]
        if result.get("validation") is not True or result["events_sha256"] != file_sha256(event_path):
            raise ValueError(f"Invalid run: {path}")
        metric = result["metrics"]
        row = {"seed": result["seed"], "condition": result["condition"], "status": result["status"],
               "right_censored": metric["right_censored"], **{key: value for key, value in metric.items()
               if key not in ("populations", "action_state_counts")}}
        for name, value in metric["populations"].items():
            row[f"{name}_spikes"] = value["total_spikes"]
            row[f"{name}_rate_hz"] = value["mean_rate_hz_per_neuron"]
        rows.append(row)
    frame = pd.DataFrame(rows)
    expected = len(frame.seed.unique()) * len(CONDITIONS)
    if len(frame) != expected or set(frame.condition) != set(CONDITIONS):
        raise ValueError("Incomplete condition-by-seed grid")
    if not (frame.groupby(["seed", "condition"]).size() == 1).all(): raise ValueError("Duplicate seed-condition cell")
    return frame


def bootstrap(values, rng, n=10000):
    values = np.asarray(values, dtype=float); draws = values[rng.integers(0, len(values), (n, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "median": float(np.median(values)),
            "bootstrap_mean_95_percent_ci": [float(x) for x in np.percentile(draws, [2.5, 97.5])]}


def paired(frame):
    metrics = ["restricted_survival_seconds", "cumulative_damage", "distance_travelled", "enemy_exposure_fraction",
               "nearest_enemy_distance_mean", "projectile_signal_ticks", "reaction_latency_seconds",
               "threat_direction_match_fraction", "kills", "resources_collected", "attack_button_ticks",
               "forward_active_fraction", "turning_active_fraction", "absolute_turn_mean"]
    rows = []
    for seed, group in frame.groupby("seed"):
        pivot = group.set_index("condition")
        for condition in PRIMARY_COMPARISONS:
            for metric in metrics:
                left, right = pivot.loc[condition, metric], pivot.loc["uninformed", metric]
                if pd.notna(left) and pd.notna(right): rows.append({"seed": seed, "comparison": condition,
                    "reference": "uninformed", "metric": metric, "difference": float(left-right)})
    return pd.DataFrame(rows), metrics


def plot_paired(frame, metric, title, ylabel, path):
    fig, ax = plt.subplots(figsize=(10, 5)); xs = np.arange(len(CONDITIONS))
    for _, group in frame.groupby("seed"):
        ax.plot(xs, [group.set_index("condition").loc[c, metric] for c in CONDITIONS], color="#94a3b8", alpha=.4, marker="o")
    ax.plot(xs, [frame[frame.condition == c][metric].median() for c in CONDITIONS], color="#111827", linewidth=2.5, marker="D", label="median")
    ax.set_xticks(xs, [LABELS[c] for c in CONDITIONS], rotation=22, ha="right"); ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis="y", alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def plot_actions(frame, path):
    specs = [("attack_button_ticks", "Attack ticks"), ("forward_active_fraction", "Forward active fraction"),
             ("turning_active_fraction", "Turning active fraction"), ("absolute_turn_mean", "Mean |turn|")]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    for ax, (metric, title) in zip(axes, specs):
        values = [frame[frame.condition == c][metric] for c in CONDITIONS]
        box = ax.boxplot(values, patch_artist=True)
        for patch, c in zip(box["boxes"], CONDITIONS): patch.set_facecolor(COLORS[c])
        ax.set_xticks(range(1, 7), [LABELS[c] for c in CONDITIONS], rotation=45, ha="right"); ax.set_title(title); ax.grid(axis="y", alpha=.25)
    fig.suptitle("Decoded BCI behavior; no game-state action commands", fontweight="bold"); fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def run(root):
    root = Path(root); plots = root / "plots"; plots.mkdir(parents=True, exist_ok=True)
    frame = load(root); differences, metrics = paired(frame); frame.to_csv(root / "runs.csv", index=False); differences.to_csv(root / "paired-differences.csv", index=False)
    rng = np.random.default_rng(20260917)
    effects = {comparison: {metric: bootstrap(differences[(differences.comparison == comparison) & (differences.metric == metric)].difference, rng)
        for metric in metrics if len(differences[(differences.comparison == comparison) & (differences.metric == metric)])}
        for comparison in PRIMARY_COMPARISONS}
    summary = {"schema": 1, "analysis": "exploratory event-triggered artificial LC4+LPLC2 stimulation",
               "n_paired_seeds": int(frame.seed.nunique()), "conditions": list(CONDITIONS), "cap_seconds": float(frame.restricted_survival_seconds.max()),
               "condition_medians": {c: {m: float(frame[frame.condition == c][m].median()) for m in metrics} for c in CONDITIONS},
               "paired_effects_vs_uninformed": effects,
               "constraints": ["No natural visual-LC4 claim.", "Observer state supplies LC4/LPLC2 current only.", "Requested and applied game actions are identical in every run."],
               "passive_control": "Neural rate regulator uses stored LC4-silenced action rates and prior decoded-action history only; it never reads Doom state or writes buttons."}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_paired(frame, "restricted_survival_seconds", "Paired restricted survival (n=10)", "seconds through 10-second cap", plots / "survival.png")
    plot_paired(frame, "cumulative_damage", "Paired cumulative damage", "damage", plots / "damage.png")
    plot_paired(frame, "nearest_enemy_distance_mean", "Nearest-enemy distance", "Doom units", plots / "nearest-enemy.png")
    plot_paired(frame, "threat_direction_match_fraction", "Threat-direction matching", "fraction", plots / "direction-match.png")
    plot_actions(frame, plots / "actions.png")
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path); args = parser.parse_args(); print(json.dumps(run(args.root), indent=2))
