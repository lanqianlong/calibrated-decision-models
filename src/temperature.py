from dataclasses import dataclass, asdict
import json
import numpy as np
from scipy.optimize import minimize_scalar

@dataclass
class TemperatureArtifact:
    temperature: float
    sample_ids: list
    def transform(self, logits):
        z = np.asarray(logits, dtype=float) / self.temperature
        return 1.0 / (1.0 + np.exp(-z))
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f: json.dump(asdict(self), f, indent=2)

def fit_temperature(labels, logits, sample_ids=None):
    y, z = np.asarray(labels), np.asarray(logits, dtype=float)
    def loss(t):
        p = 1.0 / (1.0 + np.exp(-z / t)); p = np.clip(p, 1e-7, 1-1e-7)
        return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
    result = minimize_scalar(loss, bounds=(0.05, 20.0), method="bounded")
    return TemperatureArtifact(float(result.x), list(sample_ids or []))

