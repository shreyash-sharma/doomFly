"""Atomic full learning-state recovery with separately recorded arena restart."""
import json
import re
import shutil
import uuid
from pathlib import Path
import numpy as np
from doom.checkpoint import digest


class TrainingCheckpoints:
    def __init__(self, directory, identity):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.identity = identity

    def save(self, brain, controls, game, record):
        generation = uuid.uuid4().hex
        stage = self.directory/(generation+'.partial')
        stage.mkdir()
        try:
            brain.checkpoint(stage/'brain.npz')
            np.save(stage/'decoder.npy', controls.rates, allow_pickle=False)
            data = {'schema': 1, 'identity': self.identity, 'record': record,
                'game': game.observation(), 'recovery': 'learning-state-with-new-arena',
                'sha256': {name: digest(stage/name) for name in ['brain.npz', 'decoder.npy']}}
            (stage/'state.json').write_text(json.dumps(data, indent=2)+'\n')
            stage.rename(self.directory/generation)
            pointer = self.directory/'latest.partial'
            pointer.write_text(json.dumps({'generation': generation})+'\n')
            pointer.replace(self.directory/'latest.json')
            generations = sorted([p for p in self.directory.iterdir() if re.fullmatch('[a-f0-9]{32}', p.name)], key=lambda p: p.stat().st_mtime, reverse=True)
            for old in generations[2:]:
                shutil.rmtree(old)
            return generation
        except Exception:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    def restore(self, brain, controls, game):
        pointer = self.directory/'latest.json'
        if not pointer.exists():
            return None
        generation = json.loads(pointer.read_text())['generation']
        if not re.fullmatch('[a-f0-9]{32}', generation):
            raise ValueError('Invalid checkpoint identifier')
        path = self.directory/generation
        data = json.loads((path/'state.json').read_text())
        if data['schema'] != 1 or data['identity'] != self.identity:
            raise ValueError('Checkpoint belongs to another model, graph or protocol')
        for name, expected in data['sha256'].items():
            if name not in ['brain.npz', 'decoder.npy'] or digest(path/name) != expected:
                raise ValueError('Checkpoint checksum mismatch')
        if set(data['sha256']) != {'brain.npz', 'decoder.npy'}:
            raise ValueError('Incomplete checkpoint')
        rates = np.load(path/'decoder.npy', allow_pickle=False)
        if rates.shape != controls.rates.shape or rates.dtype != controls.rates.dtype or not np.isfinite(rates).all():
            raise ValueError('Invalid decoder checkpoint')
        with np.load(path/'brain.npz', allow_pickle=False) as a:
            for k in ['weight', *brain.fields]:
                if not np.isfinite(a[k]).all():
                    raise ValueError('Nonfinite brain checkpoint')
            if np.any(a['queue_count'] < 0) or np.any(a['queue_count'] > brain.n) or np.any(a['queue'] < 0) or np.any(a['queue'] >= brain.n):
                raise ValueError('Invalid delayed events')
        brain.restore(path/'brain.npz')
        controls.rates[:] = rates
        game.episode = data['game']['episode']+1
        game.tick = 0
        return {**data['record'], 'interrupted_round': data['game'],
            'recovery': 'learned and fast neural state retained; new arena; interrupted round censored'}
