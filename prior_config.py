"""Configuration for the optional training-only OT-FM branch."""
from dataclasses import asdict, dataclass
import json
import math


@dataclass(frozen=True)
class PriorConfig:
    enabled: bool = False
    weight: float = 0.005

    def __post_init__(self):
        if type(self.enabled) is not bool:
            raise ValueError("enabled must be a boolean")
        if isinstance(self.weight, bool) or not isinstance(self.weight, (int, float)):
            raise ValueError("weight must be a finite nonnegative number")
        if not math.isfinite(self.weight) or self.weight < 0:
            raise ValueError("weight must be a finite nonnegative number")

    @property
    def active(self):
        return self.enabled and self.weight > 0

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or set(values) != {"enabled", "weight"}:
            raise ValueError("prior config must contain exactly enabled and weight")
        return cls(**values)


def load_prior_config(path=None):
    if path is None:
        return PriorConfig()
    with open(path) as handle:
        return PriorConfig.from_dict(json.load(handle))
