"""Live adapter for the explicitly unvalidated v6 memory hypothesis.

Only images enter the visual model. Nonfatal health loss schedules next-step
PPL101 current; the fixed decoder receives only actual neural spike counts.
Terminal damage and interrupted pulses are recorded, not paired with a fresh
round's unrelated image. No game state selects buttons or directly sets weights.
"""
import hashlib
import numpy as np


class DamageTraining:
    def __init__(self, brain, enabled=True):
        self.brain = brain
        self.enabled = bool(enabled)
        self.brain.weights_frozen = not self.enabled
        self.until = 0
        self.events = 0
        self.delivered_steps = 0
        self.last_steps = 0
        self.terminal_events = 0
        self.cancelled_steps = 0

    def state(self):
        return {k: getattr(self, k) for k in ['until', 'events', 'delivered_steps',
            'last_steps', 'terminal_events', 'cancelled_steps']}

    def restore(self, state):
        if set(state) != set(self.state()) or any(type(v) is not int or v < 0 for v in state.values()):
            raise ValueError('Invalid reinforcement checkpoint')
        for k, v in state.items():
            setattr(self, k, v)

    def new_round(self):
        self.cancelled_steps += max(0, self.until - self.brain.cursor)
        self.until = self.brain.cursor
        self.last_steps = 0

    def step(self, rgb, steps):
        if type(steps) is not int or steps <= 0:
            raise ValueError('Positive integer integration steps required')
        b = self.brain
        active = min(steps, max(0, self.until - b.cursor))
        counts = np.zeros(b.n, dtype=np.int32)
        wall = 0.
        # Split at the exact pulse boundary, including inside a 35 Hz game tic.
        for n, stimulus in [(active, (b.circuit['dan'], 4.)), (steps-active, None)]:
            if n:
                c, t = b.rgb_step(rgb, n*.1, learning=self.enabled, stimulation=stimulus)
                counts += c
                wall += t
        b.counts[:] = counts
        self.last_steps = active
        self.delivered_steps += active
        return counts, wall

    def observe(self, before, after):
        damage = max(0, before['health'] - max(0, after['health']))
        if damage:
            if after['finished']:
                self.terminal_events += 1
            else:
                self.events += 1
                # Repeated hits extend the pulse; dose is measured, not inferred
                # by multiplying event count. Duration = 2,000 x 0.1 ms.
                self.until = self.brain.cursor + 2000
        return damage

    def telemetry(self):
        b = self.brain
        m = b.memory()
        ratios = b.weight[b.circuit['edges']] / b.baseline_plastic
        if not np.isfinite(ratios).all():
            raise RuntimeError('Nonfinite memory efficacy')
        return {**m, 'enabled': self.enabled, 'validated': False,
            'maximum_efficacy': float(ratios.max()),
            'mean_absolute_change': float(np.abs(ratios-1).mean()),
            'bound_edges': int(np.count_nonzero((ratios <= .10001) | (ratios >= 1.99999))),
            'efficacy_histogram': np.histogram(ratios, bins=20, range=(.1, 2.))[0].tolist(),
            'histogram_range': [.1, 2.],
            'damage_events': self.events, 'delivered_ms': round(self.delivered_steps*.1, 3),
            'stimulus_active': self.last_steps > 0, 'stimulus_steps_last_tic': self.last_steps,
            'terminal_events_excluded': self.terminal_events,
            'cancelled_ms_at_round_reset': round(self.cancelled_steps*.1, 3),
            'KC_spikes_last_tic': int(b.counts[b.circuit['kc']].sum()),
            'DAN_spikes_last_tic': b.counts[b.circuit['dan']].tolist(),
            'MBON_spikes_last_tic': b.counts[b.circuit['mb']].tolist()}


def candidate_provenance(brain, root):
    paths = []
    for directory in ['doom_learning', 'doom_learning_v6']:
        paths += list((root/directory).glob('*.py')) + list((root/directory).glob('*.cpp'))
    from doom_learning_v6.brain import PARAMETERS
    return {'model': 'adaptive-centered-v6-live-v1', 'validated': False,
        'kernel': brain.build, 'parameters': {**PARAMETERS, 'eta': brain.eta},
        'configuration': brain.configuration_signature(), 'calibration': brain.calibration,
        'visual': brain.visual_report, 'circuit': brain.circuit['report'],
        'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'reinforcement': 'Nonfatal damage: next-step +4 mV-equivalent PPL101 current for 200 ms. Overlapping hits extend exposure. Terminal damage excluded; pending exposure cancelled at reset.',
        'rounds': 'Continuous fast neural state, efficacy state and memory traces across normal game resets; no per-round equilibration. Different from the historical isolated-trial pilot.',
        'claim': 'Experimental plasticity is enabled. Useful vision, associative learning and survival improvement have not been established.'}
