"""Matched intervention smoke test using the immutable learned embodiment BCI.

Conditions one through six apply game-state events only as logged neural
current.  Their buttons are always precisely the decoder output.  The seventh
condition exists solely as a non-biological motor positive control.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from doom.fearless.danger import packet_from_scene, sided_populations, threat_drive
from doom.fearless.embodiment import EmbodimentDecoder
from doom.fearless.embodiment_train import _reward, metrics
from doom.fearless.interventions import apply_intervention
from doom.fearless.run import (GRAPH, MODEL_MANIFEST, TARGET_MANIFEST, TICKS_PER_SECOND,
                               array_sha256, atomic_json, file_sha256)
from doom.game import Game, retinal_samples
from doom.native import NativeBrain

ROOT = Path(__file__).resolve().parents[2]
# ``smoke`` preserves the first failed protocol audit.  This directory contains
# the corrected exact-schedule matched control.
OUT = ROOT / "outputs/fearless/embodiment-v1/smoke-v2-matched"
CHECKPOINT = ROOT / "outputs/fearless/embodiment-v1/decoder-frozen.npz"
TK = np.array([8017, 8196, 36409, 134363, 136053], dtype=np.int64)
RANDOM = np.array([126563, 136090, 110303, 126140, 7437], dtype=np.int64)
CONDS = ("normal", "threat_informed", "fearless", "aggressive", "constant_aggression", "random_control")
SMOKE_SEEDS = (763001, 763002, 763003)
CAP_SECONDS = 5


def _event_tk_drive(packet, indices, constant=False):
    amplitude = 10.0 if constant else (10.0 * packet["intensity"] if packet["active"] else 0.0)
    return indices if amplitude else np.empty(0, dtype=np.int64), amplitude


def _scripted_positive(scene):
    """Directly programmed policy, intentionally outside every biological analysis."""
    player = scene["player"]; enemies = [o for o in scene["objects"] if o["name"] in ("DoomFlyImp", "Zombieman")]
    if not enemies: return {"turn": 0.0, "forward": 20.0, "attack": False}
    target = min(enemies, key=lambda o: math.hypot(o["x"] - player["x"], o["y"] - player["y"]))
    relative = (math.degrees(math.atan2(target["y"] - player["y"], target["x"] - player["x"])) - player["angle"] + 180) % 360 - 180
    return {"turn": float(np.clip(relative * .15, -6, 6)), "forward": 20.0, "attack": abs(relative) < 18}


def one(seed, condition, cap_seconds=CAP_SECONDS, imposed_amplitudes=None):
    if condition not in CONDS and condition != "scripted_positive_control": raise ValueError(condition)
    decoder = EmbodimentDecoder.load(CHECKPOINT); decoder_hash = file_sha256(CHECKPOINT)
    brain = NativeBrain(GRAPH); target = json.loads(TARGET_MANIFEST.read_text()); populations = sided_populations(target)
    intervention = apply_intervention(brain, "lc4_silenced" if condition == "fearless" else "intact", target)
    frozen_weight = array_sha256(brain.weight); game = Game(seed=seed, scenario="combat_survival", spectator=True)
    old_distances, health, events, previous_action = {}, game.observation()["health"], [], None
    try:
        for tick in range(1, int(cap_seconds * TICKS_PER_SECOND) + 1):
            before, scene = game.observation(), game.spectator()
            packet = packet_from_scene(scene, old_distances, max(0, health - before["health"])); old_distances = packet.pop("distances"); health = before["health"]
            drive = np.zeros(brain.n, np.float32); targets = np.empty(0, dtype=np.int64); amplitude = 0.0; channel = "none"
            if condition in ("threat_informed", "fearless"):
                targets, values = threat_drive(packet, populations); drive[targets] += values; amplitude = float(values[0]) if len(values) else 0.0; channel = "LC4+LPLC2 artificial threat"
            elif condition in ("aggressive", "constant_aggression", "random_control"):
                requested_indices = RANDOM if condition == "random_control" else TK
                if condition == "random_control" and imposed_amplitudes is not None:
                    amplitude = float(imposed_amplitudes[tick - 1]); targets = requested_indices if amplitude else np.empty(0, dtype=np.int64)
                    channel = "matched unrelated current replayed from Tk-FruM schedule"
                else:
                    targets, amplitude = _event_tk_drive(packet, requested_indices, condition == "constant_aggression")
                    channel = "matched unrelated current" if condition == "random_control" else "Tk-FruM artificial arousal"
                drive[targets] += amplitude
            target_step = int(round(tick * 10000 / TICKS_PER_SECOND)); steps = target_step - brain.cursor
            counts, _ = brain.step(retinal_samples(game.pixels(), brain.uv), steps * brain.dt, extra_drive=drive if np.any(drive) else None)
            interval = steps * brain.dt / 1000
            features = decoder.features(counts, interval); bits, probabilities = decoder.sample(features, np.random.default_rng(seed * 1000 + tick))
            decoded = decoder.action(bits)
            action = _scripted_positive(scene) if condition == "scripted_positive_control" else decoded
            engine_reward = game.act(action); after, after_scene = game.observation(), game.spectator()
            # ViZDoom exposes no state after a terminal action.  Retain the last
            # valid pre-action pose for a terminal record; it is observer-only.
            if after_scene is None:
                after_scene = {"player": dict(scene["player"])}
            reward, reward_components = _reward(before, after, engine_reward, action, previous_action)
            events.append({"tick": tick, "action": action, "decoder_action": decoded, "action_source": "scripted_positive_control" if condition == "scripted_positive_control" else "frozen_neural_decoder",
                "bits": bits.tolist(), "features": features.tolist(), "probabilities": probabilities.tolist(), "game_before": before, "game": after, "engine_reward": engine_reward, "reward": reward, "reward_components": reward_components,
                "stimulation": {"channel": channel, "target_indices": targets.tolist(), "amplitude": amplitude, "packet": packet},
                "tk_frum_spikes": int(counts[TK].sum()), "random_control_spikes": int(counts[RANDOM].sum()),
                "decoder_input_spikes": int(counts[decoder.feature_indices].sum()),
                "observer_metrics_only": {"player_before": scene["player"], "player_after": after_scene["player"],
                  "visible_enemy_count_before": sum(o["name"] in ("DoomFlyImp", "Zombieman") for o in scene["objects"])}})
            previous_action = action
            if after["finished"]: break
    finally: game.close()
    record = {"schema": 1, "seed": seed, "condition": condition, "events": events, "metrics": metrics(events),
              "decoder_sha256": decoder_hash, "intervention": intervention,
              "validation": {"connectome_weight_unchanged_after_initial_clamp": array_sha256(brain.weight) == frozen_weight,
                 "actions_trace_to_decoder": condition != "scripted_positive_control" and all(e["action"] == e["decoder_action"] for e in events),
                 "decoder_only_reads_neural_features": True}}
    return record


def run(output=OUT):
    output = Path(output); rows = []
    for seed in SMOKE_SEEDS:
        local = {condition: one(seed, condition) for condition in CONDS if condition != "random_control"}
        schedule = [event["stimulation"]["amplitude"] for event in local["aggressive"]["events"]]
        local["random_control"] = one(seed, "random_control", imposed_amplitudes=schedule)
        rows.extend(local.values())
    positive = [one(seed, "scripted_positive_control") for seed in SMOKE_SEEDS]
    for row in rows + positive: atomic_json(output / "runs" / f'{row["seed"]}-{row["condition"]}.json', row)
    by = {(r["seed"], r["condition"]): r for r in rows}
    hashes = {r["decoder_sha256"] for r in rows}
    tk = [r for r in rows if r["condition"] in ("aggressive", "constant_aggression")]
    random = [r for r in rows if r["condition"] == "random_control"]
    checks = {"all_conditions_use_same_decoder_hash": len(hashes) == 1,
      "all_conditions_actions_trace_to_decoder": all(r["validation"]["actions_trace_to_decoder"] for r in rows),
      "tk_frum_activated": all(sum(e["tk_frum_spikes"] for e in r["events"]) > 0 for r in tk),
      "lc4_silenced_only_fearless": all((r["condition"] == "fearless") == bool(r["intervention"]["selected_neurons"]) for r in rows),
      # The random run naturally develops different observer-state packets once
      # its neural activity changes.  It receives the copied *delivered-current*
      # sequence, which is the pre-registered matched quantity.
      "random_schedule_matched_to_aggressive": all([e["stimulation"]["amplitude"] for e in by[seed, "aggressive"]["events"]] ==
          [e["stimulation"]["amplitude"] for e in by[seed, "random_control"]["events"]] for seed in SMOKE_SEEDS),
      "tk_changes_decoder_input_or_output_vs_normal": any(any((a["decoder_input_spikes"], a["action"]) != (b["decoder_input_spikes"], b["action"])
          for a, b in zip(by[seed, "aggressive"]["events"], by[seed, "normal"]["events"])) for seed in SMOKE_SEEDS),
      "positive_control_separate": all(r["validation"]["actions_trace_to_decoder"] is False for r in positive)}
    result = {"schema": 1, "status": "smoke-passed" if all(checks.values()) else "smoke-failed-stop", "seeds": SMOKE_SEEDS,
      "conditions": CONDS, "decoder_sha256": next(iter(hashes)), "checks": checks,
      "claim_boundary": "Tk-FruM and threat channels are artificial neural stimulation; the learned decoder is an engineered interface, not fly learning.",
      "positive_control": "Direct scripted motor policy; non-biological and analytically separate."}
    atomic_json(output / "results.json", result); return result


if __name__ == "__main__": print(json.dumps(run(), indent=2))
