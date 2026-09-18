"""Pre-registered neutral training and competence gate for learned embodiment."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np

from doom.fearless.embodiment import EmbodimentDecoder, OUTPUTS
from doom.fearless.run import GRAPH, MODEL_MANIFEST, TICKS_PER_SECOND, atomic_json
from doom.game import Game, retinal_samples
from doom.native import NativeBrain

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fearless/embodiment-v1"
TRAIN_SEEDS = (761001, 761002, 761003, 761004, 761005, 761006)
VALIDATION_SEEDS = (762001, 762002, 762003)
CAP_SECONDS = 5
LEARNING_RATE = 0.02
DECODER_SEED = 20260916

REWARD_FUNCTION = {
    "enemy_damage_proxy": "+0.50 * max(0, ViZDoom make_action reward)",
    "kill": "+5.00 per KILLCOUNT increment",
    "survival": "+0.005 per 35-Hz control tick while alive",
    "player_damage": "-0.20 per HEALTH decrease",
    "death": "-5.00 when episode finishes",
    "repeated_action": "-0.010 when exact button vector equals previous tick",
    "ammo_waste": "-0.030 for attack with no ammo decrease, positive engine reward, or kill",
    "inactivity": "-0.003 for all-zero button vector",
    "note": "Game telemetry is delayed reinforcement only and is never a decoder input.",
}


def _feature_indices():
    model = json.loads(MODEL_MANIFEST.read_text())
    # Existing declared public motor readouts, in manifest order; no feature was
    # selected by its correlation with a desired Doom button.
    indices = np.asarray([row["index"] for row in model["readouts"]], dtype=np.int64)
    return indices, model["readouts"]


def _angle_delta(after, before):
    return (after - before + 180.0) % 360.0 - 180.0


def _reward(before, after, engine_reward, action, previous_action):
    damage = max(0, before["health"] - after["health"])
    kills = max(0, after["kills"] - before["kills"])
    same = previous_action is not None and action == previous_action
    inactive = action["turn"] == 0 and action["forward"] == 0 and not action["attack"]
    waste = bool(action["attack"] and after["ammo"] == before["ammo"] and engine_reward <= 0 and kills == 0)
    values = {"enemy_damage_proxy": .5 * max(0.0, engine_reward), "kill": 5.0 * kills,
              "survival": .005, "player_damage": -.2 * damage,
              "death": -5.0 if after["finished"] else 0.0,
              "repeated_action": -.01 if same else 0.0,
              "ammo_waste": -.03 if waste else 0.0, "inactivity": -.003 if inactive else 0.0}
    return float(sum(values.values())), values


def run_episode(seed, decoder, rng, cap_seconds=CAP_SECONDS, learn=False):
    """Run neutral Doom only.  No condition current or observer state reaches decoder."""
    brain = NativeBrain(GRAPH); game = Game(seed=seed, scenario="combat_survival", spectator=True)
    events, trajectory, previous_action = [], [], None
    try:
        for tick in range(1, int(cap_seconds * TICKS_PER_SECOND) + 1):
            before = game.observation(); scene = game.spectator()
            target_step = int(round(tick * 10000 / TICKS_PER_SECOND)); steps = target_step - brain.cursor
            counts, _ = brain.step(retinal_samples(game.pixels(), brain.uv), steps * brain.dt)
            interval = steps * brain.dt / 1000
            features = decoder.features(counts, interval); bits, probabilities = decoder.sample(features, rng)
            action = decoder.action(bits); engine_reward = game.act(action); after = game.observation(); after_scene = game.spectator()
            reward, components = _reward(before, after, engine_reward, action, previous_action)
            event = {"tick": tick, "features": features.round(6).tolist(), "bits": bits.tolist(),
                     "probabilities": probabilities.round(6).tolist(), "action": action, "game_before": before,
                     "game": after, "engine_reward": engine_reward, "reward": reward,
                     "reward_components": components,
                     "observer_metrics_only": {"player_before": scene["player"], "player_after": after_scene["player"],
                       "visible_enemy_count_before": sum(o["name"] in ("DoomFlyImp", "Zombieman") for o in scene["objects"])}}
            events.append(event); trajectory.append({"features": features, "bits": bits, "probabilities": probabilities, "reward": reward})
            previous_action = action
            if after["finished"]: break
    finally:
        game.close()
    return events, trajectory


def metrics(events):
    actions = [event["action"] for event in events]
    distance = sum(math.hypot(b["observer_metrics_only"]["player_after"]["x"] - a["observer_metrics_only"]["player_after"]["x"],
                              b["observer_metrics_only"]["player_after"]["y"] - a["observer_metrics_only"]["player_after"]["y"])
                   for a, b in zip(events, events[1:]))
    orientation_changed = sum(abs(_angle_delta(e["observer_metrics_only"]["player_after"]["angle"], e["observer_metrics_only"]["player_before"]["angle"])) > .01
                              and abs(e["action"]["turn"]) > 0 for e in events)
    visible_attack = sum(e["action"]["attack"] and e["observer_metrics_only"]["visible_enemy_count_before"] > 0 for e in events)
    damage_or_kill = sum((e["engine_reward"] > 0 or e["game"]["kills"] > e["game_before"]["kills"])
                         and e["observer_metrics_only"]["visible_enemy_count_before"] > 0 for e in events)
    states = [(a["turn"], a["forward"], a["attack"]) for a in actions]
    counts = {state: states.count(state) for state in set(states)}
    entropy = -sum((n / len(states)) * math.log(n / len(states)) for n in counts.values()) if states else 0.0
    final = events[-1]["game"]
    return {"ticks": len(events), "return": float(sum(e["reward"] for e in events)), "survival": len(events) / TICKS_PER_SECOND,
            "kills": final["kills"], "final_health": final["health"], "damage_received": 100 - final["health"],
            "distance_travelled": float(distance), "turn_effect_ticks": orientation_changed,
            "attacks_visible_enemy": visible_attack, "damage_or_kill_visible_enemy_ticks": damage_or_kill,
            "action_entropy": float(entropy)}


def _mean(rows, key): return float(np.mean([row["metrics"][key] for row in rows]))


def evaluate(label, decoder, seeds, output):
    rows = []
    for seed in seeds:
        events, _ = run_episode(seed, decoder, np.random.default_rng(hash((label, seed, DECODER_SEED)) & 0xffffffff))
        path = output / "validation" / label / f"{seed}.json"
        atomic_json(path, {"schema": 1, "seed": seed, "label": label, "events": events, "metrics": metrics(events)})
        rows.append(json.loads(path.read_text()))
    return rows


def train(output=OUT):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    indices, readouts = _feature_indices(); initial = EmbodimentDecoder.initial(indices, DECODER_SEED)
    decoder = EmbodimentDecoder(initial.weights.copy(), indices); curves = []
    for epoch in range(3):
        for seed in TRAIN_SEEDS:
            events, trajectory = run_episode(seed, decoder, np.random.default_rng(100000 * epoch + seed), learn=True)
            update = decoder.update(trajectory, LEARNING_RATE)
            curves.append({"epoch": epoch, "seed": seed, "metrics": metrics(events), "update": update})
    checkpoint = output / "decoder-frozen.npz"; checkpoint_hash = decoder.checkpoint(checkpoint, {
        "schema": 1, "label": "frozen learned embodiment interface", "outputs": OUTPUTS,
        "feature_readouts": readouts, "training_seeds": TRAIN_SEEDS, "epochs": 3,
        "learning_rate": LEARNING_RATE, "reward_function": REWARD_FUNCTION,
        "connectome_trainable": False, "decoder_inputs": "neural spike-rate features only"})
    initial_rows = evaluate("untrained", initial, VALIDATION_SEEDS, output)
    random_rows = evaluate("random_action_baseline", EmbodimentDecoder.initial(indices, 20260917), VALIDATION_SEEDS, output)
    trained_rows = evaluate("trained", decoder, VALIDATION_SEEDS, output)
    comparisons = {label: {key: _mean(rows, key) for key in ("return", "distance_travelled", "turn_effect_ticks", "attacks_visible_enemy", "damage_or_kill_visible_enemy_ticks", "kills", "survival")}
                   for label, rows in (("untrained", initial_rows), ("random_action_baseline", random_rows), ("trained", trained_rows))}
    # Pre-registered gate: all claims must appear in held-out neutral play.
    competent = (comparisons["trained"]["distance_travelled"] > 0 and comparisons["trained"]["turn_effect_ticks"] > 0
                 and comparisons["trained"]["damage_or_kill_visible_enemy_ticks"] > 0
                 and comparisons["trained"]["return"] > comparisons["untrained"]["return"]
                 and comparisons["trained"]["return"] > comparisons["random_action_baseline"]["return"])
    result = {"schema": 1, "status": "competence-passed" if competent else "competence-not-established-stop",
              "claim_boundary": "The decoder is an engineered embodiment interface; frozen MaleCNS did not learn Doom.",
              "training": {"seeds": TRAIN_SEEDS, "validation_seeds": VALIDATION_SEEDS, "epochs": 3,
                           "learning_rate": LEARNING_RATE, "reward_function": REWARD_FUNCTION, "curve": curves},
              "decoder": {"checkpoint": str(checkpoint.relative_to(output)), "sha256": checkpoint_hash, "outputs": OUTPUTS,
                          "feature_indices": indices.tolist(), "feature_readouts": readouts, "frozen_after_training": True},
              "held_out_comparisons": comparisons,
              "competence_gate": {"movement_changes_position": comparisons["trained"]["distance_travelled"] > 0,
                "turning_changes_orientation": comparisons["trained"]["turn_effect_ticks"] > 0,
                "attacking_sometimes_damages_visible_enemy": comparisons["trained"]["damage_or_kill_visible_enemy_ticks"] > 0,
                "held_out_better_than_untrained": comparisons["trained"]["return"] > comparisons["untrained"]["return"],
                "held_out_better_than_random": comparisons["trained"]["return"] > comparisons["random_action_baseline"]["return"],
                "passed": competent, "rule": "Stop before personality conditions unless every item is true."}}
    atomic_json(output / "results.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(train(), indent=2))
