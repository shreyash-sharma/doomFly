"""Isolated, resumable FEARLESS closed-loop experiment runner."""

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
import uuid
from collections import Counter
from pathlib import Path

import numpy as np

from doom.engine import NeuralControls
from doom.fearless.interventions import CONDITIONS, apply_intervention
from doom.game import Game, retinal_samples
from doom.native import BUILD, NativeBrain
from doom.provenance import provenance


ROOT = Path(__file__).resolve().parents[2]
GRAPH = ROOT / "outputs/doom/malecns_v1/graph.npz"
MODEL_MANIFEST = GRAPH.parent / "manifest.json"
TARGET_MANIFEST = ROOT / "outputs/fearless/target-manifest.json"
INTERVENTION_MANIFEST = ROOT / "outputs/fearless/intervention-manifest.json"
POPULATIONS = ("LC4", "LPLC2", "DNp09", "DNp01", "DNp20", "DNpe017")
TICKS_PER_SECOND = 35


def bytes_sha256(value):
    return hashlib.sha256(value).hexdigest()


def array_sha256(value):
    return bytes_sha256(np.ascontiguousarray(value).tobytes())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def software_record():
    configuration = io.StringIO()
    with contextlib.redirect_stdout(configuration):
        np.show_config()
    sources = {
        str(path.relative_to(ROOT)): file_sha256(path)
        for path in sorted((ROOT / "doom/fearless").glob("*.py"))
    }
    return {
        "python": sys.version,
        "numpy": np.__version__,
        "numpy_configuration": configuration.getvalue(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "thread_environment": {key: os.environ.get(key) for key in (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "fearless_source_sha256": sources,
    }


def population_indices(target_manifest):
    return {
        name: np.asarray([row["graph_index"] for row in
                          target_manifest["populations"][name]["neurons"]], dtype=np.int64)
        for name in POPULATIONS
    }


def population_event(counts, indices, interval_seconds):
    total = int(counts[indices].sum())
    return {
        "spikes": total,
        "rate_hz_per_neuron": total / (len(indices) * interval_seconds),
        "neurons": int(len(indices)),
    }


def reconstruct_metrics(events, cap_game_seconds):
    """Derive all run metrics from the event stream, the sole source of truth."""
    if not events or events[0].get("record_kind") != "initial_state":
        raise ValueError("Missing tick-zero initial-state record")
    ticks = [int(event["tick"]) for event in events]
    if ticks != list(range(len(events))):
        raise ValueError("Event ticks are not contiguous from zero")
    actions = events[1:]
    health = [int(event["game"]["health"]) for event in events]
    drops = [max(0, before - after) for before, after in zip(health, health[1:])]
    gains = [max(0, after - before) for before, after in zip(health, health[1:])]
    first_damage = next((index / TICKS_PER_SECOND for index, value in enumerate(drops, 1)
                         if value > 0), None)
    final = events[-1]["game"]
    died = bool(final["finished"])
    elapsed = len(actions) / TICKS_PER_SECOND
    states = Counter()
    for event in actions:
        action = event["applied"]
        turn = -1 if action["turn"] < -1e-9 else (1 if action["turn"] > 1e-9 else 0)
        states[(turn, abs(action["forward"]) > 1e-9, bool(action["attack"]))] += 1
    entropy = 0.0
    if actions:
        entropy = -sum((count / len(actions)) * math.log(count / len(actions))
                       for count in states.values())
    populations = {}
    neural_seconds = sum(float(event["neural_interval_ms"]) for event in actions) / 1000
    for name in POPULATIONS:
        values = [event["populations"][name] for event in actions]
        total = sum(value["spikes"] for value in values)
        neurons = values[0]["neurons"] if values else events[0]["populations"][name]["neurons"]
        populations[name] = {
            "total_spikes": total,
            "mean_rate_hz_per_neuron": total / (neurons * neural_seconds) if neural_seconds else 0.0,
            "peak_tick_rate_hz_per_neuron": max(
                (value["rate_hz_per_neuron"] for value in values), default=0.0),
        }
    wall_seconds = sum(float(event.get("neural_wall_seconds", 0)) for event in actions)
    return {
        "game_ticks": len(actions), "game_seconds": elapsed,
        "neural_seconds": neural_seconds, "neural_compute_seconds": wall_seconds,
        "effective_real_time_factor": neural_seconds / wall_seconds if wall_seconds else None,
        "survived_to_cap": not died, "right_censored": not died,
        "time_to_death_seconds": elapsed if died else None,
        "restricted_survival_seconds": min(elapsed, cap_game_seconds),
        "time_to_first_damage_seconds": first_damage,
        "cumulative_damage": sum(drops), "health_gained": sum(gains),
        "final_health": health[-1], "kills": int(final["kills"]),
        "attack_button_ticks": sum(bool(e["applied"]["attack"]) for e in actions),
        "forward_action_mean": float(np.mean([e["applied"]["forward"] for e in actions])) if actions else 0.0,
        "forward_active_fraction": float(np.mean([abs(e["applied"]["forward"]) > 1e-9 for e in actions])) if actions else 0.0,
        "absolute_turn_mean": float(np.mean([abs(e["applied"]["turn"]) for e in actions])) if actions else 0.0,
        "turning_active_fraction": float(np.mean([abs(e["applied"]["turn"]) > 1e-9 for e in actions])) if actions else 0.0,
        "action_entropy_nats": entropy, "action_state_counts": {str(key): value for key, value in states.items()},
        "populations": populations,
    }


def validated_existing(run_directory, expected=None):
    result_path = Path(run_directory) / "run.json"
    events_path = Path(run_directory) / "events.jsonl"
    if not result_path.exists() or not events_path.exists():
        return None
    result = json.loads(result_path.read_text())
    if (result.get("status") not in {"complete", "censored-at-cap"}
            or result.get("validation") is not True
            or result.get("events_sha256") != file_sha256(events_path)):
        return None
    if expected is not None:
        actual = {"seed": result.get("seed"), "condition": result.get("condition"),
                  "cap_game_seconds": result.get("cap_game_seconds"),
                  "direct_drive_amplitude": result.get("direct_drive", {}).get("amplitude", 0.0)}
        if actual != expected:
            return None
    return result


def run_condition(seed, condition, max_game_seconds, output, record_retinal_inputs=False,
                  direct_drive_amplitude=0.0):
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown condition: {condition}")
    if not math.isfinite(direct_drive_amplitude) or direct_drive_amplitude < 0:
        raise ValueError("Direct-drive amplitude must be finite and nonnegative")
    maximum_ticks_float = max_game_seconds * TICKS_PER_SECOND
    if max_game_seconds <= 0 or maximum_ticks_float != round(maximum_ticks_float):
        raise ValueError("Cap must be positive and exactly representable at 35 Hz")
    run_directory = Path(output) / "runs" / f"{seed}-{condition}"
    run_directory.mkdir(parents=True, exist_ok=True)
    expected_run = {"seed": seed, "condition": condition,
                    "cap_game_seconds": max_game_seconds,
                    "direct_drive_amplitude": direct_drive_amplitude}
    existing = validated_existing(run_directory, expected_run)
    if existing is not None:
        return existing
    run_id = str(uuid.uuid4())
    events_partial = run_directory / "events.jsonl.partial"
    events_final = run_directory / "events.jsonl"
    result_path = run_directory / "run.json"
    inputs_partial = run_directory / "retinal-inputs.npz.partial"
    inputs_final = run_directory / "retinal-inputs.npz"
    started = time.monotonic()
    game = None
    try:
        target = json.loads(TARGET_MANIFEST.read_text())
        intervention_manifest = json.loads(INTERVENTION_MANIFEST.read_text())
        if intervention_manifest["target_manifest_sha256"] != file_sha256(TARGET_MANIFEST):
            raise ValueError("Intervention manifest is not bound to current target manifest")
        model = json.loads(MODEL_MANIFEST.read_text())
        brain = NativeBrain(GRAPH)
        topology_before = {key: array_sha256(getattr(brain, key)) for key in ("ptr", "post", "ids")}
        intervention = apply_intervention(brain, condition, target)
        expected = intervention_manifest["conditions"][condition]
        for key in ("selected_body_ids", "selected_graph_indices", "affected_edge_rows",
                    "weight_sha256_before", "weight_sha256_after"):
            if intervention[key] != expected[key]:
                raise ValueError(f"Runtime intervention differs from manifest: {key}")
        weight_after_clamp = array_sha256(brain.weight)
        controls = NeuralControls(model["readouts"], mode="bci")
        game = Game(seed=seed, scenario="combat_survival")
        origin = provenance(GRAPH, BUILD, game.assets)
        populations = population_indices(target)
        direct_drive = None
        if direct_drive_amplitude:
            direct_drive = np.zeros(brain.n, dtype=np.float32)
            direct_drive[np.unique(np.concatenate([populations["LC4"], populations["LPLC2"]]))] = direct_drive_amplitude
        initial = game.observation()
        zero_populations = {name: {"spikes": 0, "rate_hz_per_neuron": 0.0,
                                   "neurons": int(len(indices))}
                            for name, indices in populations.items()}
        all_events = []
        retinal_vectors, frame_hashes, intervals = [], [], []
        with events_partial.open("w") as stream:
            event = {"schema": 2, "record_kind": "initial_state", "run_id": run_id,
                     "seed": seed, "condition": condition, "tick": 0, "neural_ms": 0.0,
                     "wall_seconds": 0.0, "game": initial, "populations": zero_populations,
                     "direct_drive": {"populations": ["LC4", "LPLC2"],
                                      "amplitude": direct_drive_amplitude},
                     "intervention_manifest_sha256": file_sha256(INTERVENTION_MANIFEST),
                     "provenance": origin}
            all_events.append(event)
            stream.write(json.dumps(event, separators=(",", ":")) + "\n")
            tick = 0
            while tick < int(round(maximum_ticks_float)) and not game.observation()["finished"]:
                before = game.observation()
                frame = game.pixels()
                light = retinal_samples(frame, brain.uv)
                tick += 1
                target_step = int(round(tick * 10000 / TICKS_PER_SECOND))
                steps = target_step - brain.cursor
                counts, neural_wall = brain.step(light, steps * brain.dt, extra_drive=direct_drive)
                requested_full = controls.decode(counts, steps * brain.dt / 1000)
                requested = {key: requested_full[key] for key in ("turn", "forward", "attack")}
                applied = dict(requested)
                game.act(applied)
                after = game.observation()
                interval_seconds = steps * brain.dt / 1000
                event = {"schema": 2, "record_kind": "tick", "run_id": run_id,
                         "seed": seed, "condition": condition, "tick": tick,
                         "neural_ms": round(brain.sim_ms, 6),
                         "neural_interval_ms": round(steps * brain.dt, 6),
                         "wall_seconds": time.monotonic() - started,
                         "neural_wall_seconds": neural_wall,
                         "source_frame_sha256": array_sha256(frame),
                         "input_sha256": array_sha256(light),
                         "spike_counts_sha256": array_sha256(counts),
                         "total_spikes": int(counts.sum()),
                         "populations": {name: population_event(counts, indices, interval_seconds)
                                         for name, indices in populations.items()},
                         "requested": requested, "applied": applied,
                         "game_before": before, "game": after}
                all_events.append(event)
                stream.write(json.dumps(event, separators=(",", ":")) + "\n")
                if record_retinal_inputs:
                    retinal_vectors.append(light.copy()); frame_hashes.append(event["source_frame_sha256"])
                    intervals.append(event["neural_interval_ms"])
                if tick % TICKS_PER_SECOND == 0:
                    stream.flush()
            stream.flush()
        events_partial.replace(events_final)
        if record_retinal_inputs:
            with inputs_partial.open("wb") as stream:
                np.savez_compressed(stream, vectors=np.asarray(retinal_vectors, dtype=np.float32),
                                    source_frame_sha256=np.asarray(frame_hashes),
                                    neural_interval_ms=np.asarray(intervals, dtype=np.float64))
            inputs_partial.replace(inputs_final)
        metrics = reconstruct_metrics(all_events, max_game_seconds)
        topology_after = {key: array_sha256(getattr(brain, key)) for key in topology_before}
        final_weight = array_sha256(brain.weight)
        validation = topology_before == topology_after and final_weight == weight_after_clamp
        if not validation:
            raise RuntimeError("Graph topology or post-clamp weights changed during run")
        result = {"schema": 2, "run_id": run_id,
                  "status": "complete" if all_events[-1]["game"]["finished"] else "censored-at-cap",
                  "seed": seed, "condition": condition, "cap_game_seconds": max_game_seconds,
                  "events": events_final.name, "events_sha256": file_sha256(events_final),
                  "retinal_inputs": inputs_final.name if record_retinal_inputs else None,
                  "intervention": intervention, "intervention_manifest_sha256": file_sha256(INTERVENTION_MANIFEST),
                  "target_manifest_sha256": file_sha256(TARGET_MANIFEST), "provenance": origin,
                  "direct_drive": {"populations": ["LC4", "LPLC2"],
                                   "amplitude": direct_drive_amplitude},
                  "software": software_record(), "metrics": metrics,
                  "process_wall_seconds": time.monotonic() - started,
                  "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                  "validation": validation}
        atomic_json(result_path, result)
        return result
    except BaseException as error:
        atomic_json(result_path, {"schema": 2, "run_id": run_id,
                    "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    "seed": seed, "condition": condition,
                    "events_partial": events_partial.name if events_partial.exists() else None,
                    "error_type": type(error).__name__, "error": str(error),
                    "process_wall_seconds": time.monotonic() - started})
        raise
    finally:
        if game is not None:
            game.close()


def orchestrate(args):
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    order = []
    for ordinal, seed in enumerate(args.seeds):
        rotated = args.conditions[ordinal % len(args.conditions):] + args.conditions[:ordinal % len(args.conditions)]
        order.extend({"seed": seed, "condition": condition} for condition in rotated)
    stage0_real_time_factor = 1.232
    estimated_wall_seconds = (len(order) * args.max_game_seconds / stage0_real_time_factor)
    manifest = {"schema": 2, "status": "running", "conditions": args.conditions,
                "seeds": args.seeds, "execution_order": order,
                "max_game_seconds": args.max_game_seconds, "scenario": args.scenario,
                "decoder": args.decoder,
                "direct_drive": {"populations": ["LC4", "LPLC2"],
                                 "amplitude": args.direct_drive_amplitude}, "pilot_seed_generation":
                "random.Random(20260915).sample(range(1_000_000, 2_000_000), 10)",
                "eta_basis": {"stage0_real_time_factor": stage0_real_time_factor,
                              "estimated_wall_seconds_if_full_caps": estimated_wall_seconds,
                              "warning": "Direct-drive overhead was not represented in Stage 0."}}
    atomic_json(output / "manifest.json", manifest)
    print(json.dumps({"event": "eta", **manifest["eta_basis"]}, sort_keys=True), flush=True)
    failures = []
    for item in order:
        run_directory = output / "runs" / f"{item['seed']}-{item['condition']}"
        expected_run = {"seed": item["seed"], "condition": item["condition"],
                        "cap_game_seconds": args.max_game_seconds,
                        "direct_drive_amplitude": args.direct_drive_amplitude}
        if validated_existing(run_directory, expected_run) is not None:
            continue
        run_directory.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "doom.fearless.run", "--single-seed", str(item["seed"]),
                   "--single-condition", item["condition"], "--max-game-seconds", str(args.max_game_seconds),
                   "--output", str(output)]
        if args.direct_drive_amplitude:
            command.extend(["--direct-drive-amplitude", str(args.direct_drive_amplitude)])
        if args.record_retinal_inputs and item == order[0]:
            command.append("--record-retinal-inputs")
        with (run_directory / "stdout.log").open("a") as stdout, (run_directory / "stderr.log").open("a") as stderr:
            completed = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       env={**os.environ, "OPENBLAS_NUM_THREADS": "1"})
        if completed.returncode:
            failures.append({**item, "returncode": completed.returncode})
    manifest["status"] = "failed" if failures else "complete"
    manifest["failures"] = failures
    atomic_json(output / "manifest.json", manifest)
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--max-game-seconds", type=float, required=True)
    parser.add_argument("--scenario", choices=["combat_survival"], default="combat_survival")
    parser.add_argument("--decoder", choices=["bci"], default="bci")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--record-retinal-inputs", action="store_true")
    parser.add_argument("--direct-drive-amplitude", type=float, default=0.0,
                        help="Exploratory constant current to LC4+LPLC2; not a visual stimulus.")
    parser.add_argument("--single-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--single-condition", choices=CONDITIONS, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.single_seed is None) != (args.single_condition is None):
        parser.error("Both hidden single-run arguments are required together")
    if args.single_seed is not None:
        result = run_condition(args.single_seed, args.single_condition, args.max_game_seconds,
                               args.output.resolve(), args.record_retinal_inputs,
                               args.direct_drive_amplitude)
        print(json.dumps({"run_id": result["run_id"], "status": result["status"]}, sort_keys=True))
    else:
        if not args.seeds:
            parser.error("--seeds is required")
        orchestrate(args)


if __name__ == "__main__":
    main()
