from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True, eq=False)
class Molecule:
    coords: np.ndarray
    charges: np.ndarray
    n_up: int
    n_down: int
    spins: np.ndarray = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, "coords", np.asarray(self.coords, dtype=float).reshape(-1, 3))
        object.__setattr__(self, "charges", np.asarray(self.charges, dtype=float).reshape(-1))
        object.__setattr__(self, "spins", np.repeat([0, 1], [self.n_up, self.n_down]))

    @classmethod
    def neutral(cls, coords, charges, spin=0):
        n_elec = int(round(float(np.sum(charges))))
        return cls(coords, charges, (n_elec + spin) // 2, (n_elec - spin) // 2)

    @property
    def n_elec(self):
        return self.n_up + self.n_down

    @property
    def n_nuc(self):
        return len(self.charges)

    @property
    def spin_counts(self):
        return np.array([self.n_up, self.n_down])

    def to_dict(self):
        return {
            "coords": self.coords.tolist(),
            "charges": self.charges.tolist(),
            "n_up": self.n_up,
            "n_down": self.n_down,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["coords"], d["charges"], d["n_up"], d["n_down"])
