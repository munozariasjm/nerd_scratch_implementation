import jax
import pytest

jax.config.update("jax_enable_x64", True)

from nerd import Molecule, becke_grid
from nerd.wavefunctions import analytic


@pytest.fixture(scope="session")
def helium():
    return analytic.helium_like(2.0)


@pytest.fixture(scope="session")
def triplet():
    return analytic.triplet_1s2p(3.0)


@pytest.fixture(scope="session")
def hydrogen():
    return analytic.hydrogen_molecule_like(1.0, 1.4)


@pytest.fixture(scope="session")
def carbon_dimer():
    return Molecule([[0.0, 0.0, -1.2], [0.0, 0.0, 1.2]], [6.0, 6.0], 6, 6)


@pytest.fixture(scope="session")
def grid():
    cache = {}

    def build(molecule, level=5):
        key = (molecule.coords.tobytes(), molecule.charges.tobytes(), level)
        if key not in cache:
            cache[key] = becke_grid(molecule, level)
        return cache[key]

    return build
