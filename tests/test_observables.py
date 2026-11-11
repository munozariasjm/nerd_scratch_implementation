import jax
import jax.numpy as jnp
import numpy as np

from nerd import make_model
from nerd.io import load, save
from nerd.observables import (
    contact_densities,
    dipole,
    effective_potential,
    hellmann_feynman_forces,
    make_density,
    nuclear_potential_energy,
    quadrupole,
    summary,
    total_variation,
)
from oracles import helium_spin_densities, oracle_model


def test_helium_observables(helium, grid):
    molecule = helium.molecule
    points, weights = grid(molecule, 6)
    points = jnp.asarray(points)
    rho = helium.density(points)
    assert abs(float(nuclear_potential_energy(rho, points, weights, molecule)) + 8.0) < 1e-6
    np.testing.assert_allclose(dipole(rho, points, weights, molecule), 0, atol=1e-10)
    np.testing.assert_allclose(quadrupole(rho, points, weights), 0, atol=1e-8)
    np.testing.assert_allclose(hellmann_feynman_forces(rho, points, weights, molecule), 0, atol=1e-8)
    density = make_density(oracle_model(helium_spin_densities(helium)), 0.0, 0.0)
    np.testing.assert_allclose(contact_densities(density, molecule), 16 / np.pi, rtol=1e-12)


def test_hydrogen_symmetry(hydrogen, grid):
    molecule = hydrogen.molecule
    points, weights = grid(molecule, 6)
    points = jnp.asarray(points)
    rho = hydrogen.density(points)
    np.testing.assert_allclose(dipole(rho, points, weights, molecule), 0, atol=1e-8)
    forces = np.asarray(hellmann_feynman_forces(rho, points, weights, molecule))
    np.testing.assert_allclose(forces.sum(0), 0, atol=1e-6)
    np.testing.assert_allclose(forces[:, :2], 0, atol=1e-8)
    assert np.trace(np.asarray(quadrupole(rho, points, weights))) < 1e-8


def test_total_variation(helium, grid):
    points, weights = grid(helium.molecule, 5)
    rho = helium.density(jnp.asarray(points))
    assert float(total_variation(rho, rho, weights)) == 0.0
    assert abs(float(total_variation(rho, 0.9 * rho, weights)) - 0.1) < 1e-6


def test_effective_potential_of_one_electron_atom():
    z = 3.0
    model = oracle_model(lambda r: jnp.stack([jnp.exp(-2 * z * jnp.linalg.norm(r, axis=-1))] * 2, -1))
    for r in (jnp.array([0.3, 0.1, -0.2]), jnp.array([1.5, -0.7, 2.0])):
        expected = z**2 / 2 - z / jnp.linalg.norm(r)
        np.testing.assert_allclose(effective_potential(model, 0.0, r, 0), expected, rtol=1e-10)


def test_summary_normalizes_and_round_trips(tmp_path, hydrogen, grid):
    molecule = hydrogen.molecule
    model = make_model(molecule)
    params = model.init(jax.random.PRNGKey(0))
    save(tmp_path, params, {"molecule": molecule.to_dict()})
    loaded, metadata = load(tmp_path, model.init(jax.random.PRNGKey(5)))
    assert jax.tree.all(jax.tree.map(lambda a, b: bool(jnp.all(a == b)), params, loaded))
    points, weights = grid(molecule, 4)
    result = summary(model, loaded, molecule, points, weights)
    np.testing.assert_allclose(result["electron_counts"], [1.0, 1.0], rtol=1e-10)
    assert metadata["molecule"]["n_up"] == 1
