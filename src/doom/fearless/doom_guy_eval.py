"""Fixed-seed stochastic evaluation of the frozen v3 PPO checkpoint."""
import json
from pathlib import Path
import numpy as np
from doom.fearless.doom_guy_train import ActorCritic,groups,run_episode
from doom.fearless.run import atomic_json,file_sha256

ROOT=Path(__file__).resolve().parents[2]
CHECKPOINT=ROOT/'outputs/fearless/doom-guy-fly-v5-strafe/doom-guy-decoder.npz'
OUT=ROOT/'outputs/fearless/doom-guy-fly-v5-final-test'
SEEDS=tuple(range(787001,787011))

def run():
 p=ActorCritic(2*len(groups()));saved=np.load(CHECKPOINT);p.actor[:]=saved['actor'];p.value[:]=saved['value'];rows=[]
 for seed in SEEDS:
  primary=run_episode(seed,p,'doom_guy',False,910000+seed,deterministic_eval=False)
  schedule=[e['tk_amplitude'] for e in primary['events']]
  for rec in (primary,run_episode(seed,p,'no_tk',False,910000+seed,deterministic_eval=False),
              run_episode(seed,p,'random_control',False,910000+seed,replay=schedule,deterministic_eval=False)):
   atomic_json(OUT/f'{seed}-{rec["condition"]}.json',rec);rows.append({'seed':seed,'condition':rec['condition'],'metrics':rec['metrics']})
 result={'schema':1,'status':'complete','seeds':SEEDS,'decoder_sha256':file_sha256(CHECKPOINT),
         'evaluation':'stochastic sampling with fixed per-seed RNG; frozen weights','rows':rows}
 atomic_json(OUT/'results.json',result);return result

if __name__=='__main__':print(json.dumps(run(),indent=2))
