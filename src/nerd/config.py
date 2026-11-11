from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    width: int = 64
    n_residual: int = 2


@dataclass(frozen=True)
class SamplerConfig:
    n_walkers: int = 4096
    decorrelation: int = 60
    equilibration: int = 600
    initial_step: float = 0.2
    target_acceptance: float = 0.57


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 25000
    learning_rate: float = 1e-2
    decay_fraction: float = 0.8
    final_ratio: float = 6e-5
    b1: float = 0.95
    b2: float = 0.999
    nce_weight: float = 1.0
    seed: int = 0


def paper_steps(molecule):
    return max(25000, 5000 * molecule.n_elec)


def to_dict(*configs):
    return {type(c).__name__: asdict(c) for c in configs}
