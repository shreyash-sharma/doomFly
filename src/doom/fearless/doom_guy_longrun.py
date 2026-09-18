"""Long, frozen-v5 DoomFly evaluation; never overwrites the final test."""
import json
from pathlib import Path
import numpy as np
import doom.fearless.doom_guy_train as trainer
from doom.fearless.run import atomic_json,file_sha256

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'outputs/fearless/doom-guy-fly-v5-strafe/doom-guy-decoder.npz'
OUT=ROOT/'outputs/fearless/doom-guy-fly-v5-longrun'
SEEDS=(788001,788002,788003)
EPISODE_SECONDS=30

def run():
    trainer.CAP=EPISODE_SECONDS
    policy=trainer.ActorCritic(2*len(trainer.groups()))
    saved=np.load(CHECKPOINT);policy.actor[:]=saved['actor'];policy.value[:]=saved['value']
    rows=[]
    for seed in SEEDS:
        result=trainer.run_episode(seed,policy,'doom_guy',False,920000+seed,deterministic_eval=False)
        atomic_json(OUT/f'{seed}-doom_guy.json',result)
        rows.append({'seed':seed,'metrics':result['metrics']})
    summary={'schema':1,'status':'complete','purpose':'long frozen-v5 gameplay evaluation',
             'checkpoint_sha256':file_sha256(CHECKPOINT),'seeds':SEEDS,
             'episode_seconds':EPISODE_SECONDS,'condition':'doom_guy',
             'rows':rows,'claim_boundary':'No PPO weights changed; this is evaluation only.'}
    atomic_json(OUT/'summary.json',summary);return summary

if __name__=='__main__':print(json.dumps(run(),indent=2))
