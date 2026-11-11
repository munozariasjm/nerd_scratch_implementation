import jax
import jax.numpy as jnp
import numpy as np

from nerd import Molecule, make_noise
from nerd.geometry import stratified_components


def test_normalized(grid, carbon_dimer, hydrogen):
    for molecule in (carbon_dimer, hydrogen.molecule, Molecule([[0, 0, 0], [0, 0, 2.0]], [6.0, 1.0], 4, 3)):
        noise = make_noise(molecule)
        points, weights = grid(molecule, 6)
        assert abs(np.sum(weights * np.exp(noise.log_prob(jnp.asarray(points)))) - 1) < 1e-6


def test_stratified_components_are_exact_for_neutral_molecules():
    charges = jnp.array([6.0, 6.0, 6.0, 1.0, 1.0, 1.0, 1.0])
    counts = np.bincount(np.asarray(stratified_components(charges, 22 * 512)), minlength=7)
    np.testing.assert_array_equal(counts, 512 * np.array([6, 6, 6, 1, 1, 1, 1]))


def test_samples_follow_log_prob(grid):
    molecule = Molecule([[0, 0, 0], [0, 0, 2.0]], [6.0, 2.0], 4, 4)
    noise = make_noise(molecule)
    samples = noise.sample(jax.random.PRNGKey(0), 400000)
    points, weights = grid(molecule, 6)
    p = np.exp(noise.log_prob(jnp.asarray(points)))
    for f in (lambda r: r[:, 2], lambda r: r[:, 0] ** 2, lambda r: np.linalg.norm(r, axis=-1)):
        values = np.asarray(f(samples))
        expected = np.sum(weights * p * f(points))
        assert abs(values.mean() - expected) < 4 * values.std() / np.sqrt(len(values))


def test_single_center_radial_moments():
    noise = make_noise(Molecule([[1.0, -2.0, 0.5]], [3.0], 2, 1))
    radius = np.linalg.norm(np.asarray(noise.sample(jax.random.PRNGKey(1), 200000)) - [1.0, -2.0, 0.5], axis=-1)
    assert abs(radius.mean() - 3) < 0.02
    assert abs((radius**2).mean() - 12) < 0.15
