"""Ten-seed matched cohort, permitted only after embodiment smoke-v2 passes."""
import json
from pathlib import Path

from doom.fearless.embodiment_smoke import CHECKPOINT, CONDS, one
from doom.fearless.embodiment_smoke_reaudit import run as audit_smoke
from doom.fearless.run import atomic_json, file_sha256

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/fearless/embodiment-v1/cohort"
SEEDS = tuple(range(764001, 764011))


def run(output=OUT):
    smoke = audit_smoke()
    if smoke["status"] != "smoke-passed": raise RuntimeError("Full cohort forbidden: smoke did not pass")
    output = Path(output); records = []
    for seed in SEEDS:
        local = {condition: one(seed, condition) for condition in CONDS if condition != "random_control"}
        schedule = [event["stimulation"]["amplitude"] for event in local["aggressive"]["events"]]
        local["random_control"] = one(seed, "random_control", imposed_amplitudes=schedule)
        local["scripted_positive_control"] = one(seed, "scripted_positive_control")
        for record in local.values():
            atomic_json(output / "runs" / f'{seed}-{record["condition"]}.json', record)
            records.append({"seed": seed, "condition": record["condition"], "metrics": record["metrics"]})
    manifest = {"schema": 1, "status": "complete", "seeds": SEEDS, "neural_conditions": CONDS,
      "positive_control": "scripted_positive_control is non-biological and analytically separate.",
      "decoder_sha256": file_sha256(CHECKPOINT), "records": records}
    atomic_json(output / "manifest.json", manifest); return manifest


if __name__ == "__main__": print(json.dumps(run(), indent=2))
