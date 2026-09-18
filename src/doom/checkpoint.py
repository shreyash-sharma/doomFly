"""Atomic recovery of a fixed brain; an interrupted game starts a new round.

This format intentionally rejects learning subclasses and mismatched releases.
It is process recovery for the baseline, not a validated memory mechanism.
"""
import hashlib
import json
import re
import shutil
import uuid
from pathlib import Path
import numpy as np

FIELDS=['weight','v','g','refractory','drive','previous_drive','queue','queue_count',
        'counts','luminance','active','active_flag','nactive','last']


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


class Checkpoints:
    def __init__(self,directory,identity):
        self.directory=Path(directory);self.identity=identity
        self.directory.mkdir(parents=True,exist_ok=True)

    def save(self,brain,controls,game,record):
        from doom.native import NativeBrain
        if type(brain) is not NativeBrain or game.scenario!='combat_survival':
            raise ValueError('Only the fixed baseline and unlimited combat arena support this recovery format')
        generation=uuid.uuid4().hex
        stage=self.directory/(generation+'.partial');stage.mkdir()
        try:
            np.savez(stage/'brain.npz',**{k:getattr(brain,k) for k in FIELDS},decoder_rates=controls.rates)
            data={'schema':1,'recovery':'neural-state-with-new-arena','identity':self.identity,'record':record,'game':game.observation(),
                  'brain':{k:getattr(brain,k) for k in ['cursor','sim_ms','total_spikes']},
                  'sha256':{'brain.npz':digest(stage/'brain.npz')}}
            (stage/'state.json').write_text(json.dumps(data,indent=2)+'\n')
            stage.rename(self.directory/generation)
            pointer=self.directory/'latest.partial'
            pointer.write_text(json.dumps({'generation':generation})+'\n');pointer.replace(self.directory/'latest.json')
            generations=sorted([p for p in self.directory.iterdir() if re.fullmatch('[a-f0-9]{32}',p.name)],key=lambda p:p.stat().st_mtime,reverse=True)
            for old in generations[2:]:shutil.rmtree(old)
            return generation
        except Exception:
            shutil.rmtree(stage,ignore_errors=True)
            raise

    def restore(self,brain,controls,game):
        from doom.native import NativeBrain
        if type(brain) is not NativeBrain or game.scenario!='combat_survival':raise ValueError('Unsupported recovery model')
        pointer=self.directory/'latest.json'
        if not pointer.exists():return None
        generation=json.loads(pointer.read_text())['generation']
        if not re.fullmatch('[a-f0-9]{32}',generation):raise ValueError('Invalid checkpoint identifier')
        path=self.directory/generation
        data=json.loads((path/'state.json').read_text())
        if data['schema']!=1 or data['identity']!=self.identity:raise ValueError('Checkpoint belongs to another model, graph or protocol')
        for name in ['brain.npz']:
            if digest(path/name)!=data['sha256'][name]:raise ValueError('Checkpoint checksum mismatch')
        with np.load(path/'brain.npz',allow_pickle=False) as saved:
            arrays={k:saved[k] for k in FIELDS+['decoder_rates']}
        for name,array in arrays.items():
            target=controls.rates if name=='decoder_rates' else getattr(brain,name)
            if target.shape!=array.shape or target.dtype!=array.dtype or not np.isfinite(array).all():raise ValueError('Checkpoint array mismatch')
        if np.any(arrays['queue_count']<0) or np.any(arrays['queue_count']>brain.n) or np.any(arrays['queue']<0) or np.any(arrays['queue']>=brain.n):raise ValueError('Invalid queued neuron')
        # ViZDoom save advances physics by two tics. Do not use it to pretend
        # recovery is uninterrupted. The caller supplies a fresh arena and a new
        # run ID; the previous partial round is explicitly censored.
        game.episode=data['game']['episode']+1;game.tick=0
        for name in FIELDS:getattr(brain,name)[:]=arrays[name]
        controls.rates[:]=arrays['decoder_rates']
        for key in ['cursor','sim_ms','total_spikes']:setattr(brain,key,data['brain'][key])
        return {**data['record'],'interrupted_round':data['game'],'recovery':'neural-state-with-new-arena'}
