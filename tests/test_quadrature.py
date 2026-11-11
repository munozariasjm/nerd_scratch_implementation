import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.parametrize("name", ["helium", "triplet", "hydrogen"])
@pytest.mark.parametrize("level", [3, 5])
def test_exact_densities_integrate_to_electron_count(name, level, request, grid):
    system = request.getfixturevalue(name)
    points, weights = grid(system.molecule, level)
    total = np.sum(weights * np.asarray(system.density(jnp.asarray(points))))
    assert abs(total / system.molecule.n_elec - 1) < 1e-5
