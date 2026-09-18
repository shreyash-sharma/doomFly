"""Resolve the only exact aggression-associated MaleCNS mapping used by DOOM GUY FLY."""
import json
from pathlib import Path
import pandas as pd
from doom.fearless.run import ROOT, atomic_json

OUT = ROOT / "outputs/fearless/aggression/audit.json"
TK_IDS = (18792, 18987, 50960, 530190, 533662)

def run(output=OUT):
    neurons = pd.read_feather(ROOT / "connectome_data/malecns_v1/normalized/neurons.feather")
    annotations = pd.read_feather(ROOT / "connectome_data/malecns_v1/annotations.feather").fillna("")
    merged = neurons.merge(annotations, left_on="source_id", right_on="bodyId", suffixes=("_graph", "_annotation"))
    rows = merged[merged.source_id.isin(TK_IDS)].sort_values("source_id")
    if tuple(rows.source_id) != TK_IDS or not rows.retained.all(): raise ValueError("Tk-FruM IDs are missing or unretained")
    if not rows.synonyms.str.contains("Asahina 2014: TK-FruM", regex=False).all(): raise ValueError("Annotation evidence is not exact")
    result = {"schema": 1, "population": "Tk-FruM / AVLP727m", "purpose": "biologically associated aggression/arousal label only",
      "evidence": {"MaleCNS_annotation": "synonyms exactly contain 'Asahina 2014: TK-FruM'",
        "primary_literature": [{"citation": "Asahina et al., Cell 2014, DOI 10.1016/j.cell.2013.12.045",
          "finding": "activation and silencing of male-specific FruM+ Tk neurons increased/decreased intermale aggression; authors describe aggressive arousal"},
          {"citation": "Hoopfer et al., eLife 2015, DOI 10.7554/eLife.11346",
           "finding": "P1 activation promotes a persistent state that enhances aggression, but no exact P1 subset was selected here"}]},
      "neurons": [{"body_id": int(r.source_id), "graph_index": int(r.node_index), "type": r.type,
        "instance": r.instance, "side": r.somaSide, "fruDsx": r.fruDsx, "dimorphism": r.dimorphism,
        "neurotransmitter": r.neurotransmitter, "quality": r.quality} for _, r in rows.iterrows()],
      "constraint": "This is a five-neuron annotation mapping, not a claim that Doom entities are flies or that model stimulation recreates biological aggression."}
    atomic_json(output, result); return result

if __name__ == '__main__': print(json.dumps(run(), indent=2))
