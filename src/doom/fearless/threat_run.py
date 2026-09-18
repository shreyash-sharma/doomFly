"""Event-triggered artificial-threat FEARLESS runner.

Observer state is permitted here only because the owner explicitly requested an
artificial danger channel. It is transformed solely into logged neural current;
the fixed BCI is still the exclusive source of Doom button values.
"""

import argparse
import json
import math
import time
import uuid
from pathlib import Path

import numpy as np

from doom.engine import NeuralControls
from doom.fearless.danger import (FalseAlarmScheduler, THREAT_AMPLITUDE,
    packet_from_scene, sided_populations, threat_drive)
from doom.fearless.interventions import apply_intervention
from doom.fearless.run import (GRAPH, INTERVENTION_MANIFEST, MODEL_MANIFEST,
    POPULATIONS, TARGET_MANIFEST, TICKS_PER_SECOND, array_sha256, atomic_json,
    file_sha256, population_event, population_indices, reconstruct_metrics)
from doom.game import Game, retinal_samples
from doom.native import BUILD, NativeBrain
from doom.provenance import provenance


ROOT = Path(__file__).resolve().parents[2]
CONDITIONS = ("uninformed", "threat_informed", "lc4_silenced", "random_lesion",
              "false_alarm", "passive_rate_matched")
INTERVENTION = {"uninformed": "intact", "threat_informed": "intact",
                "lc4_silenced": "lc4_silenced", "random_lesion": "lc4_random_matched",
                "false_alarm": "intact", "passive_rate_matched": "intact"}


def _motor_indices(model):
    return {"DNp20": np.asarray([r["index"] for r in model["readouts"] if r["type"] == "DNp20"], dtype=np.int64),
            "DNpe017": np.asarray([r["index"] for r in model["readouts"] if r["type"] == "DNpe017"], dtype=np.int64)}


class RateMatcher:
    """Closed-loop neural-current regulator; it never writes game actions."""
    def __init__(self, target, indices):
        self.target = {**target,
            "attack_fraction": target.get("attack_fraction", target["attack_button_ticks"] / target["game_ticks"])}
        self.indices = indices; self.history = []

    def drive(self, n):
        if not self.history:
            return np.zeros(n, dtype=np.float32)
        actions = self.history
        attack = np.mean([a["attack"] for a in actions]); forward = np.mean([abs(a["forward"]) > 1e-9 for a in actions])
        turning = np.mean([abs(a["turn"]) > 1e-9 for a in actions])
        drive = np.zeros(n, dtype=np.float32)
        # A low-gain, bounded current uses only decoded-action history and the
        # stored fearless target rates. It does not inspect Doom state or set a button.
        pe = (self.target["attack_fraction"] - attack) + (self.target["forward_active_fraction"] - forward)
        pt = self.target["turning_active_fraction"] - turning
        drive[self.indices["DNpe017"]] = np.clip(4.0 * pe, -4.0, 4.0)
        drive[self.indices["DNp20"]] = np.clip(2.0 * pt, -2.0, 2.0)
        return drive

    def observe(self, action):
        self.history.append(dict(action))


def _nearest_scene_summary(scene):
    player = scene["player"]
    enemies = [obj for obj in scene["objects"] if obj["name"] in ("DoomFlyImp", "Zombieman")]
    distances = [math.hypot(obj["x"] - player["x"], obj["y"] - player["y"]) for obj in enemies]
    return {"player_x": player["x"], "player_y": player["y"],
            "nearest_enemy_distance": min(distances) if distances else None,
            "enemy_count": len(enemies)}


def _extra_metrics(events):
    metrics = reconstruct_metrics(events, events[0]["cap_game_seconds"])
    ticks = events[1:]
    positions = [(e["threat_scene"]["player_x"], e["threat_scene"]["player_y"]) for e in ticks]
    metrics["distance_travelled"] = float(sum(math.hypot(x1-x0, y1-y0)
        for (x0, y0), (x1, y1) in zip(positions, positions[1:])))
    nearest = [e["threat_scene"]["nearest_enemy_distance"] for e in ticks if e["threat_scene"]["nearest_enemy_distance"] is not None]
    metrics["nearest_enemy_distance_mean"] = float(np.mean(nearest)) if nearest else None
    metrics["nearest_enemy_distance_min"] = float(np.min(nearest)) if nearest else None
    active = [e for e in ticks if e["danger"]["delivered"]["active"]]
    metrics["enemy_exposure_fraction"] = len(active) / len(ticks) if ticks else 0.0
    metrics["danger_signal_ticks"] = len(active)
    metrics["projectile_signal_ticks"] = sum("incoming_projectile" in e["danger"]["delivered"]["event_types"] for e in ticks)
    metrics["damage_signal_ticks"] = sum("taking_damage" in e["danger"]["truth"]["event_types"] for e in ticks)
    matches = []
    latencies = []
    for index, event in enumerate(ticks):
        packet = event["danger"]["delivered"]
        if not packet["active"] or packet["direction"] == "center":
            continue
        sign = -1 if packet["direction"] == "left" else 1
        future = ticks[index:index + TICKS_PER_SECOND]
        match = next((offset for offset, candidate in enumerate(future)
                      if candidate["applied"]["turn"] * sign > 1e-9), None)
        matches.append(match is not None)
        if match is not None:
            latencies.append(match / TICKS_PER_SECOND)
    metrics["threat_direction_match_fraction"] = float(np.mean(matches)) if matches else None
    metrics["reaction_latency_seconds"] = float(np.mean(latencies)) if latencies else None
    final = events[-1]["game"]
    metrics["resources_collected"] = int(final.get("ammo_pickups", 0))
    return metrics


def run_condition(seed, condition, cap_game_seconds, output, passive_target=None):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if cap_game_seconds <= 0 or cap_game_seconds * TICKS_PER_SECOND != round(cap_game_seconds * TICKS_PER_SECOND):
        raise ValueError("cap must be positive and representable at 35 Hz")
    if condition == "passive_rate_matched" and passive_target is None:
        raise ValueError("Passive rate-control needs the completed matched LC4-silenced target")
    directory = Path(output) / "runs" / f"{seed}-{condition}"; directory.mkdir(parents=True, exist_ok=True)
    result_path, partial, final = directory / "run.json", directory / "events.jsonl.partial", directory / "events.jsonl"
    if result_path.exists() and final.exists():
        record = json.loads(result_path.read_text())
        if record.get("validation") is True and record.get("status") in ("complete", "censored-at-cap"):
            return record
    started = time.monotonic(); game = None
    try:
        target = json.loads(TARGET_MANIFEST.read_text()); model = json.loads(MODEL_MANIFEST.read_text())
        brain = NativeBrain(GRAPH); before_weight = array_sha256(brain.weight)
        intervention = apply_intervention(brain, INTERVENTION[condition], target)
        if INTERVENTION[condition] == "intact" and before_weight != array_sha256(brain.weight):
            raise AssertionError("Intact condition changed a weight")
        fixed_weight = array_sha256(brain.weight)
        populations = population_indices(target); sided = sided_populations(target)
        controls = NeuralControls(model["readouts"], mode="bci")
        matcher = RateMatcher(passive_target, _motor_indices(model)) if passive_target else None
        game = Game(seed=seed, scenario="combat_survival", spectator=True)
        origin = provenance(GRAPH, BUILD, game.assets); scheduler = FalseAlarmScheduler(seed)
        prior_distances = {}; prior_health = game.observation()["health"]; events = []; tick = 0
        zero = {name: {"spikes": 0, "rate_hz_per_neuron": 0.0, "neurons": int(len(ix))} for name, ix in populations.items()}
        with partial.open("w") as stream:
            initial = {"schema": 1, "record_kind": "initial_state", "tick": 0, "seed": seed,
                       "condition": condition, "cap_game_seconds": cap_game_seconds, "game": game.observation(),
                       "populations": zero, "danger_channel": "artificial observer-state to LC4+LPLC2 current only",
                       "provenance": origin, "intervention": intervention}
            events.append(initial); stream.write(json.dumps(initial, separators=(",", ":")) + "\n")
            while tick < round(cap_game_seconds*TICKS_PER_SECOND) and not game.observation()["finished"]:
                before = game.observation(); scene = game.spectator()
                if scene is None: raise RuntimeError("Threat observer unavailable")
                truth = packet_from_scene(scene, prior_distances, max(0, prior_health - before["health"]))
                prior_distances = truth.pop("distances"); prior_health = before["health"]
                if condition in ("uninformed",): delivered = {**truth, "active": False, "intensity": 0.0, "event_types": []}
                elif condition == "false_alarm": delivered = scheduler.push(truth); delivered.pop("distances", None)
                else: delivered = truth
                indices, values = threat_drive(delivered, sided)
                extra = np.zeros(brain.n, dtype=np.float32)
                extra[indices] += values
                rate_drive = matcher.drive(brain.n) if matcher else np.zeros(brain.n, dtype=np.float32)
                extra += rate_drive
                tick += 1; target_step = int(round(tick * 10000 / TICKS_PER_SECOND)); steps = target_step - brain.cursor
                frame = game.pixels(); light = retinal_samples(frame, brain.uv)
                counts, neural_wall = brain.step(light, steps * brain.dt, extra_drive=extra if np.any(extra) else None)
                full = controls.decode(counts, steps * brain.dt / 1000); action = {key: full[key] for key in ("turn", "forward", "attack")}
                game.act(action); after = game.observation();
                if matcher: matcher.observe(action)
                interval = steps * brain.dt / 1000
                event = {"schema": 1, "record_kind": "tick", "tick": tick, "seed": seed, "condition": condition,
                         "neural_interval_ms": steps*brain.dt, "neural_wall_seconds": neural_wall,
                         "game_before": before, "game": after, "threat_scene": _nearest_scene_summary(scene),
                         "danger": {"truth": truth, "delivered": delivered, "target_indices": indices.tolist(),
                                    "current_amplitude": float(values[0]) if len(values) else 0.0,
                                    "rate_matcher_current_l1": float(np.abs(rate_drive).sum())},
                         "populations": {n: population_event(counts, ix, interval) for n, ix in populations.items()},
                         "requested": action, "applied": dict(action)}
                events.append(event); stream.write(json.dumps(event, separators=(",", ":")) + "\n")
        partial.replace(final)
        metrics = _extra_metrics(events)
        lc4 = populations["LC4"]
        validation = (array_sha256(brain.weight) == fixed_weight and
                      all(e["requested"] == e["applied"] for e in events[1:]) and
                      all(set(e["danger"]["target_indices"]).issubset(set(np.concatenate([sided["LC4"]["L"], sided["LC4"]["R"], sided["LPLC2"]["L"], sided["LPLC2"]["R"]]))) for e in events[1:]))
        result = {"schema": 1, "status": "complete" if events[-1]["game"]["finished"] else "censored-at-cap",
                  "run_id": str(uuid.uuid4()), "seed": seed, "condition": condition,
                  "cap_game_seconds": cap_game_seconds, "events": final.name, "events_sha256": file_sha256(final),
                  "metrics": metrics, "intervention": intervention, "lc4_output_clamp": condition == "lc4_silenced",
                  "danger_channel": {"kind": "artificial observer-state current", "amplitude": THREAT_AMPLITUDE,
                    "false_alarm": condition == "false_alarm", "no_direct_action_commands": True,
                    "rate_matcher": condition == "passive_rate_matched"},
                  "provenance": origin, "validation": validation, "process_wall_seconds": time.monotonic()-started}
        atomic_json(result_path, result); return result
    except BaseException as error:
        atomic_json(result_path, {"status": "failed", "seed": seed, "condition": condition,
                                  "error_type": type(error).__name__, "error": str(error)})
        raise
    finally:
        if game is not None: game.close()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", required=True, type=int); parser.add_argument("--cap", type=float, default=30)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    records = []
    # LC4-silenced runs precede their seed-matched passive controls by design.
    for seed in args.seeds:
        local = {}
        for condition in ("uninformed", "threat_informed", "lc4_silenced", "random_lesion", "false_alarm"):
            local[condition] = run_condition(seed, condition, args.cap, args.output)
        target = local["lc4_silenced"]["metrics"]
        local["passive_rate_matched"] = run_condition(seed, "passive_rate_matched", args.cap, args.output, target)
        records.extend(local.values())
    atomic_json(args.output / "manifest.json", {"schema": 1, "status": "complete", "seeds": args.seeds,
      "conditions": CONDITIONS, "cap_game_seconds": args.cap, "records": [{"seed": r["seed"], "condition": r["condition"], "status": r["status"]} for r in records]})


if __name__ == "__main__": main()
