"""Gate the small event-threat validation cohort before held-out seeds."""

import json
from pathlib import Path

from doom.fearless.threat_run import CONDITIONS
from doom.fearless.run import file_sha256


def run(root):
    root = Path(root); records = {}
    for path in root.glob("runs/*/run.json"):
        row = json.loads(path.read_text()); records[row["condition"]] = (row, path.parent)
    if set(records) != set(CONDITIONS):
        raise ValueError("Validation cohort is incomplete")
    checks = {}
    for condition, (row, directory) in records.items():
        events_path = directory / row["events"]
        events = [json.loads(line) for line in events_path.read_text().splitlines()]
        checks[f"{condition}_record_valid"] = row["validation"] is True and row["events_sha256"] == file_sha256(events_path)
        checks[f"{condition}_no_action_override"] = all(e["requested"] == e["applied"] for e in events[1:])
    informed = records["threat_informed"][0]; uninformed = records["uninformed"][0]
    silenced = records["lc4_silenced"][0]; random = records["random_lesion"][0]
    false_row, false_dir = records["false_alarm"]
    false_events = [json.loads(line) for line in (false_dir / false_row["events"]).read_text().splitlines()][1:]
    checks["informed_signal_and_lc4_lplc2_activation"] = (
        informed["metrics"]["danger_signal_ticks"] > 0 and
        informed["metrics"]["populations"]["LC4"]["total_spikes"] > 0 and
        informed["metrics"]["populations"]["LPLC2"]["total_spikes"] > 0)
    checks["uninformed_has_no_artificial_signal"] = uninformed["metrics"]["danger_signal_ticks"] == 0
    checks["lc4_output_rows_zero_only_in_lc4_silenced"] = (
        silenced["intervention"]["affected_edge_rows"] == 25477 and
        random["intervention"]["affected_edge_rows"] == 27783 and
        random["intervention"]["selected_graph_indices"] != silenced["intervention"]["selected_graph_indices"])
    checks["false_alarm_differs_from_truth"] = any(
        event["danger"]["truth"]["intensity"] != event["danger"]["delivered"]["intensity"] or
        event["danger"]["truth"]["direction"] != event["danger"]["delivered"]["direction"]
        for event in false_events)
    passive = records["passive_rate_matched"][0]["metrics"]; target = silenced["metrics"]
    checks["passive_attack_rate_within_0_10_of_lc4_silenced"] = abs(
        passive["attack_button_ticks"] / passive["game_ticks"] -
        target["attack_button_ticks"] / target["game_ticks"]) <= .10
    result = {"schema": 1, "status": "passed" if all(checks.values()) else "failed",
              "checks": checks, "conditions": {name: {"signal_ticks": row["metrics"]["danger_signal_ticks"],
                "lc4_spikes": row["metrics"]["populations"]["LC4"]["total_spikes"],
                "lplc2_spikes": row["metrics"]["populations"]["LPLC2"]["total_spikes"],
                "validation": row["validation"]} for name, (row, _) in records.items()},
              "constraint": "The channel issues additive neural current only; requested and applied buttons must be identical."}
    (root / "validation.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    (root / "VALIDATION.md").write_text("# Event-threat validation\n\n" +
        ("Passed" if result["status"] == "passed" else "Failed") +
        ". This small cohort checks telemetry timing, current targets, stimulation, clamps, false alarms, and the no-action-override invariant before held-out seeds.\n")
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument("root", type=Path); args = parser.parse_args()
    print(json.dumps(run(args.root), indent=2, sort_keys=True))
