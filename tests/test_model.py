import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nerd import ModelConfig, make_model
from nerd.model import cusp, features


@pytest.fixture(params=["hydrogen", "carbon_dimer"])
def system(request, hydrogen, carbon_dimer):
    molecule = hydrogen.molecule if request.param == "hydrogen" else carbon_dimer
    model = make_model(molecule)
    return molecule, model, model.init(jax.random.PRNGKey(1))


def fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    theta = math.pi * (1 + 5**0.5) * i
    return np.stack([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=-1)


def test_shapes(system):
    molecule, model, params = system
    r = jax.random.normal(jax.random.PRNGKey(0), (5, 7, 3))
    assert model.log_density(params, r).shape == (5, 7, 2)
    assert model.spin_log_density(params, r, jnp.zeros(7, int)).shape == (5, 7)
    assert model.score(params, r, jnp.zeros(7, int)).shape == (5, 7, 3)


def test_parameter_count(system):
    molecule, model, params = system
    n_params = sum(x.size for x in jax.tree.leaves(params))
    width = ModelConfig().width
    expected = (4 * molecule.n_nuc + 1) * width + 2 * (width + 1) * width + (width + 1) * 2 + 4 * molecule.n_nuc
    assert n_params == expected


def test_features_match_paper(system):
    molecule, _, _ = system
    r = jnp.array([0.3, -0.2, 0.9])
    offsets = r - jnp.asarray(molecule.coords)
    s = jnp.sqrt(1 + jnp.sum(offsets**2, -1))
    expected = jnp.concatenate([offsets * (jnp.log1p(s) / s)[:, None], jnp.log1p(s)[:, None]], -1).reshape(-1)
    np.testing.assert_allclose(features(offsets), expected, rtol=1e-12)


def test_spin_selection(system):
    _, model, params = system
    r = jax.random.normal(jax.random.PRNGKey(2), (4, 3))
    both = model.log_density(params, r)
    np.testing.assert_allclose(model.spin_log_density(params, r, 0), both[:, 0])
    spins = jnp.array([0, 1, 1, 0])
    np.testing.assert_allclose(model.spin_log_density(params, r, spins), both[jnp.arange(4), spins])


def test_score_matches_finite_differences(system):
    _, model, params = system
    r = jax.random.normal(jax.random.PRNGKey(3), (3, 3)) + 0.1
    spin = jnp.array([0, 1, 0])
    h = 1e-6
    def log_rho(x):
        return model.spin_log_density(params, x, spin)

    numerical = jnp.stack([(log_rho(r + h * e) - log_rho(r - h * e)) / (2 * h) for e in jnp.eye(3)], axis=-1)
    np.testing.assert_allclose(model.score(params, r, spin), numerical, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("spin", [0, 1])
def test_kato_cusp(system, spin):
    molecule, model, params = system
    directions = jnp.asarray(fibonacci_sphere(2000))
    for center, charge in zip(molecule.coords, molecule.charges):
        values = []
        for h in (1e-4, 2e-4):
            outer = model.spin_log_density(params, center + h * directions, spin)
            values.append(jnp.mean(outer - model.spin_log_density(params, jnp.asarray(center), spin)) / h)
        slope = 2 * values[0] - values[1]
        assert abs(slope + 2 * charge) < 1e-3 * charge


def test_cusp_term_is_flat_far_away(system):
    molecule, model, params = system
    far = jnp.array([[40.0, 0.0, 0.0]]) - jnp.asarray(molecule.coords)
    expected = -2 * math.sqrt(math.pi) * molecule.n_nuc
    np.testing.assert_allclose(cusp(far, jnp.asarray(molecule.charges)), expected, rtol=1e-12)


def test_initial_envelope_exponent(system):
    _, _, params = system
    np.testing.assert_allclose(jax.nn.softplus(params["envelope"]["exponent"]), math.log1p(math.e))
