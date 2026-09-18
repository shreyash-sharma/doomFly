"""Artificial sugar conditioning input, independent of the control decoder.

Positive task reward can trigger 200 ms of LB3c current injection. These are
homology-inferred sugar sensory cells in MaleCNS. A pulse alone is not learning:
there is no plasticity rule in this runtime and no weight changes are made.
"""
class SugarReinforcement:
    def __init__(self,enabled=False):self.enabled=enabled;self.until_ms=0.;self.pulses=0
    def observe(self,reward,neural_ms):
        if self.enabled and reward>0:
            self.until_ms=max(self.until_ms,neural_ms+200);self.pulses+=1
    def active(self,neural_ms):return self.enabled and neural_ms<self.until_ms
