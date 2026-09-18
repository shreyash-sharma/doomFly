"""Audit the public static VIDraft fruitfly-brain artifact without inventing its engine."""

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT.parent / "hf-fruitfly-brain"
DEFAULT_OUTPUT = ROOT / "outputs/fearless/hf-audit"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text)
    temporary.replace(path)


def snapshot_summary(encoded):
    raw = base64.b64decode(encoded)
    values = np.frombuffer(raw, dtype=np.uint8)
    return {
        "encoded_characters": len(encoded), "decoded_neurons": int(len(values)),
        "decoded_sha256": hashlib.sha256(raw).hexdigest(),
        "nonzero_neurons": int(np.count_nonzero(values)),
        "above_display_threshold_neurons": int(np.count_nonzero(values / 255 > 0.02)),
        "mean_normalized_activity": float(values.mean() / 255),
        "maximum_normalized_activity": float(values.max() / 255),
    }


def audit(source=DEFAULT_SOURCE, output=DEFAULT_OUTPUT):
    source, output = Path(source), Path(output)
    baked = json.loads((source / "data/baked.json").read_text())
    meta = json.loads((source / "data/meta.json").read_text())
    index = (source / "index.html").read_text()
    readme = (source / "README.md").read_text()
    baseline = float(baked["sig"][""])
    effects = {condition or "intact": {
        "escape_signal": float(value),
        "difference_from_intact": float(value) - baseline,
        "relative_change_percent": 100 * (float(value) / baseline - 1),
    } for condition, value in baked["sig"].items()}
    snapshots = {
        condition or "intact": [snapshot_summary(value) for value in frames]
        for condition, frames in baked["act"].items()
    }
    type_path = source / "data/soma_type.u16"
    type_bytes = type_path.stat().st_size
    lesion_snapshot_checks = {}
    if type_bytes == 2 * int(meta["n"]):
        type_indices = np.fromfile(type_path, dtype="<u2")
        names = {"lc4": "LC4", "lplc2": "LPLC2", "lc11": "LC11", "lc6": "LC6"}
        for condition, cell_type in names.items():
            type_index = meta["types"].index(cell_type)
            target = type_indices == type_index
            final_intact = np.frombuffer(base64.b64decode(baked["act"][""][-1]), dtype=np.uint8)
            final_lesioned = np.frombuffer(base64.b64decode(baked["act"][condition][-1]), dtype=np.uint8)
            lesion_snapshot_checks[condition] = {
                "cell_type": cell_type, "drawn_target_neurons": int(np.count_nonzero(target)),
                "target_neurons_in_activity_span": int(np.count_nonzero(target[:len(final_intact)])),
                "final_intact_target_nonzero": int(np.count_nonzero(final_intact[target[:len(final_intact)]])),
                "final_lesioned_target_nonzero": int(np.count_nonzero(final_lesioned[target[:len(final_lesioned)]])),
                "final_intact_target_activity_sum": int(final_intact[target[:len(final_intact)]].sum()),
                "final_lesioned_target_activity_sum": int(final_lesioned[target[:len(final_lesioned)]].sum()),
            }
    files = [source / item for item in ("README.md", "index.html", "data/baked.json", "data/meta.json")]
    tracked = {str(path.relative_to(source)): sha256(path) for path in files}
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=source,
                                check=True, capture_output=True, text=True).stdout.strip()
    expected_static_files = {str(path.relative_to(source)) for path in source.rglob("*") if path.is_file()}
    engine_files = sorted(name for name in expected_static_files if name.endswith((".py", ".cpp", ".c", ".rs")))
    report = {
        "schema": 1, "source_repository": "https://huggingface.co/spaces/VIDraft/fruitfly-brain",
        "source_commit": git_commit, "source_file_sha256": tracked,
        "public_artifact_kind": "static Space with pre-measured payload",
        "baked_note": baked.get("note"), "baked_threat_groups": baked["threat"],
        "baked_frames": baked["frames"], "reported_escape_effects_recomputed": effects,
        "activity_snapshot_summary": snapshots,
        "control_snapshot_byte_identity": {
            condition: [baked["act"][condition][frame] == baked["act"][""][frame]
                        for frame in range(len(baked["frames"]))]
            for condition in ("lc11", "lc6")},
        "lesion_snapshot_checks": lesion_snapshot_checks,
        "metadata": {"drawn_neurons": meta["n"], "cell_types": len(meta["types"]),
                     "documented_simulated_neurons": 173023,
                     "activity_snapshot_neuron_count": snapshots["intact"][0]["decoded_neurons"]},
        "static_source_observations": {
            "weights_released": False,
            "engine_source_files_present": engine_files,
            "baked_payload_present": True,
            "index_has_live_api_fallback": "/api/stimulate" in index,
            "index_loads_baked_payload": "data/baked.json" in index,
            "readme_states_direct_threat_drive": "LPLC2 + LC4 drive" in readme,
            "readme_states_firing_rate_model": "leaky firing-rate model" in readme,
        },
        "reproducibility_conclusion": (
            "The published 0.840-to-0.091 arithmetic and static activity payload are reproducible; "
            "the full simulation cannot be independently rerun from this public repository because "
            "it contains neither the weight matrix nor simulation engine source."),
        "scope_conclusion": (
            "This artifact estimates a direct LC4+LPLC2-drive total effect on a DNp01-DNp04 firing "
            "readout. It does not establish retinal/visual recruitment of LC4 or observed escape behavior."),
        "lesion_semantics_caveat": (
            "The public final LC4 and LPLC2 lesion snapshots retain some nonzero activity in cells "
            "labeled as their target type. Without released engine code, this is consistent with an "
            "outgoing-output clamp or another implementation detail, and does not establish literal "
            "cellular silence."),
    }
    atomic_write(output / "report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown = """# VIDraft fruitfly-brain public-artifact audit

## Reproducible from the released files

The static payload reproduces the published escape-signal arithmetic: intact
`{intact:.6f}`, LC4-lesioned `{lc4:.6f}`, a `{drop:.3f}%` LC4 change. It also contains four
precomputed activity snapshots for each intact, LC4, LPLC2, LC11, and LC6 condition.

## Not independently reproducible from the released files

The repository explicitly labels `data/baked.json` as pre-measured and states that weights are
not included. The released files contain no simulation-engine source file. The browser code loads
the baked payload and has a `/api/stimulate` fallback, but the deployed public Space is static.
Thus the simulation run that generated the numbers cannot be rerun or perturbed from this artifact.

## Meaning for FEARLESS

The public result is a direct `LPLC2 + LC4` input experiment with a DNp01-DNp04 firing readout.
It supports an important downstream-circuit hypothesis, but does not test image-to-retina-to-LC4
recruitment or observed behavior. FEARLESS should not start its Doom pilot until this distinction
is carried into an explicitly separate, exploratory design.
""".format(intact=baseline, lc4=float(baked["sig"]["lc4"]),
           drop=effects["lc4"]["relative_change_percent"])
    atomic_write(output / "REPORT.md", markdown)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(audit(args.source, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
