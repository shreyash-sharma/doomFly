"""Re-audit saved smoke-v2 raw records after an audit-only check correction."""
import json
from pathlib import Path

from doom.fearless.embodiment_smoke import CONDS, OUT, SMOKE_SEEDS
from doom.fearless.run import atomic_json


def run(output=OUT):
    output = Path(output)
    def load(seed, condition):
        return json.loads((output / "runs" / f"{seed}-{condition}.json").read_text())
    rows = [load(seed, condition) for seed in SMOKE_SEEDS for condition in CONDS]
    positive = [load(seed, "scripted_positive_control") for seed in SMOKE_SEEDS]
    by = {(r["seed"], r["condition"]): r for r in rows}; hashes = {r["decoder_sha256"] for r in rows}
    tk = [r for r in rows if r["condition"] in ("aggressive", "constant_aggression")]
    checks = {"all_conditions_use_same_decoder_hash": len(hashes) == 1,
      "all_conditions_actions_trace_to_decoder": all(r["validation"]["actions_trace_to_decoder"] for r in rows),
      "tk_frum_activated": all(sum(e["tk_frum_spikes"] for e in r["events"]) > 0 for r in tk),
      "lc4_silenced_only_fearless": all((r["condition"] == "fearless") == bool(r["intervention"]["selected_neurons"]) for r in rows),
      "random_schedule_matched_to_aggressive": all(
        [e["stimulation"]["amplitude"] for e in by[seed, "aggressive"]["events"]] ==
        [e["stimulation"]["amplitude"] for e in by[seed, "random_control"]["events"]] for seed in SMOKE_SEEDS),
      "tk_changes_decoder_input_or_output_vs_normal": any(any((a["decoder_input_spikes"], a["action"]) != (b["decoder_input_spikes"], b["action"])
        for a, b in zip(by[seed, "aggressive"]["events"], by[seed, "normal"]["events"])) for seed in SMOKE_SEEDS),
      "positive_control_separate": all(not r["validation"]["actions_trace_to_decoder"] for r in positive)}
    result = {"schema": 1, "status": "smoke-passed" if all(checks.values()) else "smoke-failed-stop", "seeds": SMOKE_SEEDS,
      "conditions": CONDS, "decoder_sha256": next(iter(hashes)), "checks": checks,
      "audit_note": "This reads the completed smoke-v2 raw records without rerunning Doom. It corrects only the schedule audit to compare delivered currents, not divergent observer packets.",
      "claim_boundary": "Tk-FruM and threat channels are artificial neural stimulation; the learned decoder is an engineered interface, not fly learning.",
      "positive_control": "Direct scripted motor policy; non-biological and analytically separate."}
    atomic_json(output / "results.json", result); return result


if __name__ == "__main__": print(json.dumps(run(), indent=2))
