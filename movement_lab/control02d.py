"""Compact sensor-only feedback over 02C's learned motor vocabulary.

The two old waypoints are a basis, NOT a scheduled trajectory. Two coefficients
are recomputed every 20 ms. No clock, event flag, root velocity, contact force,
or world position enters this controller. The state is an 80 ms exponential
average of the five noisy IMU features, initialized to the first observation.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np

OPTION = Path('results/02c/runs/cem-s1/final.npz')


@dataclass(frozen=True)
class Design:
    memory: bool = True
    tau_s: float = .08
    bound: float = .8

    @property
    def dim(self):
        return 22 if self.memory else 12


class Controller:
    def __init__(self, weights, basis, design=Design()):
        self.design = design
        self.weights = np.asarray(weights).reshape(2, design.dim // 2)
        self.basis = np.asarray(basis).reshape(2, 14)
        self.history = None
        self.coefficients = np.zeros(2)

    def __call__(self, imu_features):
        x = np.clip(np.asarray(imu_features), -10, 10)
        if self.history is None:
            self.history = x.copy()
        else:
            self.history += (1 - np.exp(-.02 / self.design.tau_s)) * (x - self.history)
        features = np.r_[x, self.history, 1.] if self.design.memory else np.r_[x, 1.]
        self.coefficients = np.tanh(self.weights @ features)
        return np.clip(self.coefficients @ self.basis, -self.design.bound, self.design.bound)


def load_basis():
    return np.load(OPTION)['best'].reshape(2, 14)
