"""Paired analysis of the completed frozen-decoder embodiment cohort."""
import csv, json, math
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from doom.fearless.run import atomic_json

ROOT=Path(__file__).resolve().parents[2]
IN=ROOT/'outputs/fearless/embodiment-v1/cohort'
OUT=ROOT/'outputs/fearless/embodiment-v1/analysis'
NEURAL=('normal','threat_informed','fearless','aggressive','constant_aggression','random_control')
POSITIVE='scripted_positive_control'

def row(record):
 e=record['events'];m=record['metrics']; actions=[x['action'] for x in e]
 visible=[x['observer_metrics_only']['visible_enemy_count_before']>0 for x in e]
 pursuits=sum(a['forward']>0 and v for a,v in zip(actions,visible))
 retreats=sum(a['forward']<0 for a in actions)
 return {'seed':record['seed'],'condition':record['condition'], **m,
  'attacks':sum(a['attack'] for a in actions),'forward_ticks':sum(a['forward']>0 for a in actions),
  'turning_ticks':sum(a['turn']!=0 for a in actions),'pursuit_ticks':pursuits,'retreat_ticks':retreats,
  'pursuit_fraction_visible':pursuits/max(1,sum(visible)),'attack_per_minute':sum(a['attack'] for a in actions)*35*60/max(1,len(e)),
  'kills_per_damage':m['kills']/max(1,m['damage_received']),'tk_frum_spikes':sum(x['tk_frum_spikes'] for x in e),
  'decoder_input_spikes':sum(x['decoder_input_spikes'] for x in e),
  'stimulated_ticks':sum(x['stimulation']['amplitude']>0 for x in e)}

def mean(rows,key):return float(np.mean([r[key] for r in rows]))
def paired(rows,a,b,key):
 x={(r['seed'],r['condition']):r for r in rows};v=[x[s,a][key]-x[s,b][key] for s in sorted({r['seed'] for r in rows})]
 return {'mean_difference':float(np.mean(v)),'median_difference':float(np.median(v)),'wins':sum(q>0 for q in v),'losses':sum(q<0 for q in v),'ties':sum(q==0 for q in v),'values':v}

def run():
 records=[json.loads(p.read_text()) for p in sorted((IN/'runs').glob('*.json'))]; rows=[row(r) for r in records]
 keys=('survival','kills','damage_received','attacks','attack_per_minute','forward_ticks','turning_ticks','pursuit_ticks','pursuit_fraction_visible','retreat_ticks','distance_travelled','action_entropy','kills_per_damage','tk_frum_spikes','decoder_input_spikes')
 summary={c:{k:mean([r for r in rows if r['condition']==c],k) for k in keys} for c in (*NEURAL,POSITIVE)}
 comparisons={name:{k:paired(rows,a,b,k) for k in ('survival','kills','damage_received','attacks','pursuit_ticks','forward_ticks','distance_travelled','kills_per_damage')}
  for name,a,b in [('threat_vs_normal','threat_informed','normal'),('fearless_vs_threat','fearless','threat_informed'),('aggressive_vs_normal','aggressive','normal'),('constant_vs_event','constant_aggression','aggressive'),('aggressive_vs_random','aggressive','random_control')]}
 OUT.mkdir(parents=True,exist_ok=True)
 with (OUT/'per_run.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=sorted(rows[0]));w.writeheader();w.writerows(rows)
 result={'schema':1,'cohort_manifest':str((IN/'manifest.json').relative_to(ROOT)),'n_seeds':10,'conditions':NEURAL,
  'separate_positive_control':POSITIVE,'condition_means':summary,'paired_comparisons':comparisons,
  'measurement_limitations':['This runner did not log nearest-enemy distance, enemy geometry, resource collection, or ammo spent; those cannot be reconstructed honestly from these raw records.','Damage inflicted is represented only by positive engine-reward/kills, not exact enemy hit points.'],
  'claim_boundary':'Conditions are artificial neural interventions under one frozen engineered embodiment interface. They do not demonstrate fear, anger, intent, consciousness, or natural Doom vision.'}
 atomic_json(OUT/'summary.json',result)
 fig,axes=plt.subplots(1,3,figsize=(13,3.6)); labels=list(NEURAL)+[POSITIVE]
 for ax,key,title in zip(axes,['kills','damage_received','pursuit_ticks'],['Kills','Damage received','Forward while enemy visible']):
  ax.bar(range(len(labels)),[summary[c][key] for c in labels],color=['#4c78a8']*6+['#e45756']);ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right');ax.set_title(title)
 fig.tight_layout();fig.savefig(OUT/'condition-means.png',dpi=180);plt.close(fig)
 lines=['# Frozen embodiment-interface cohort','',f'10 matched unseen seeds; 5-second episodes; decoder hash is fixed in the cohort manifest.','', '## Plain-English result','']
 for c in NEURAL:
  s=summary[c];lines.append(f'- {c}: {s["kills"]:.2f} kills, {s["damage_received"]:.1f} damage, {s["attacks"]:.1f} attacks, {s["pursuit_ticks"]:.1f} forward-visible-enemy ticks.')
 lines+=['','Threat, FEARLESS, Tk-FruM, and random conditions are neural-current/clamp interventions only; their buttons came from the same frozen decoder. The scripted policy is a deliberately non-biological benchmark.','', '## Interpretation guardrail','', 'Use the paired-comparison JSON rather than visual differences alone. This short cohort supports descriptive comparisons, not claims about fly emotion or natural perception.']
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
 return result
if __name__=='__main__':print(json.dumps(run(),indent=2))
