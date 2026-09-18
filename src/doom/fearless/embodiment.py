"""Frozen-connectome, learned embodiment interface for the FEARLESS follow-up.

This is deliberately a small *engineered* BCI: a four-output linear Bernoulli
readout from a fixed list of existing motor-readout neurons.  It never receives
pixels, observer geometry, game variables, or a condition label.  Doom feedback
is used only after an action to update this readout during neutral training.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


OUTPUTS = ("forward", "turn_left", "turn_right", "attack")


def sigmoid(values):
    return 1.0 / (1.0 + np.exp(-np.clip(values, -30.0, 30.0)))


def feature_vector(counts, indices, interval_seconds):
    """Rate features from an a-priori, existing transparent motor readout set."""
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be positive")
    rates = np.asarray(counts, dtype=np.float64)[indices] / interval_seconds
    # Fixed scale prevents a rare spike burst from dominating a linear output.
    return np.r_[1.0, np.clip(rates / 100.0, 0.0, 5.0)]


@dataclass
class EmbodimentDecoder:
    """Four independent, stochastic binary outputs trained by policy gradient."""
    weights: np.ndarray
    feature_indices: np.ndarray

    @classmethod
    def initial(cls, feature_indices, seed=20260916):
        rng = np.random.default_rng(seed)
        indices = np.asarray(feature_indices, dtype=np.int64)
        # Small, seed-fixed asymmetry supports exploration but has no task policy.
        weights = rng.normal(0.0, 0.03, size=(len(OUTPUTS), len(indices) + 1))
        return cls(weights.astype(np.float64), indices)

    def features(self, counts, interval_seconds):
        return feature_vector(counts, self.feature_indices, interval_seconds)

    def probabilities(self, features):
        return sigmoid(self.weights @ np.asarray(features, dtype=np.float64))

    def sample(self, features, rng):
        probabilities = self.probabilities(features)
        bits = rng.random(len(OUTPUTS)) < probabilities
        return bits.astype(np.int8), probabilities

    def action(self, bits):
        bits = np.asarray(bits, dtype=np.int8)
        return {"turn": float(6 * (bits[2] - bits[1])),
                "forward": float(20 * bits[0]), "attack": bool(bits[3])}

    def update(self, trajectory, learning_rate, gamma=0.99):
        """One REINFORCE update.  The only trainable parameters are ``weights``."""
        if not trajectory:
            return {"gradient_l2": 0.0, "mean_return": 0.0}
        rewards = np.asarray([row["reward"] for row in trajectory], dtype=np.float64)
        returns = np.zeros_like(rewards); running = 0.0
        for index in range(len(rewards) - 1, -1, -1):
            running = rewards[index] + gamma * running
            returns[index] = running
        advantage = returns - returns.mean()
        gradient = np.zeros_like(self.weights)
        for row, value in zip(trajectory, advantage):
            gradient += value * np.outer(row["bits"] - row["probabilities"], row["features"])
        gradient /= len(trajectory)
        # Declared numerical safety bound, not a behavior-tuning mechanism.
        norm = float(np.linalg.norm(gradient))
        if norm > 5.0:
            gradient *= 5.0 / norm
        self.weights += float(learning_rate) * gradient
        return {"gradient_l2": float(np.linalg.norm(gradient)), "mean_return": float(returns.mean())}

    def checkpoint(self, path, metadata):
        path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, feature_indices=self.feature_indices)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        Path(str(path) + ".json").write_text(json.dumps({**metadata, "checkpoint_sha256": digest}, indent=2, sort_keys=True) + "\n")
        return digest

    @classmethod
    def load(cls, path):
        data = np.load(Path(path), allow_pickle=False)
        return cls(np.asarray(data["weights"], dtype=np.float64), np.asarray(data["feature_indices"], dtype=np.int64))
