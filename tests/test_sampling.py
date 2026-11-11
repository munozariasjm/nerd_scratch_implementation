import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nerd import SamplerConfig, make_metropolis


def moments(r):
    return {"z2": r[..., 2] ** 2, "x2": r[..., 0] ** 2, "radius": np.linalg.norm(r, axis=-1)}


def grid_moments(system, grid):
    points, weights = grid(system.molecule, 6)
    rho = np.asarray(system.density(jnp.asarray(points))) / system.molecule.n_elec
    return {k: float(np.sum(weights * rho * v)) for k, v in moments(points).items()}


@pytest.mark.parametrize("name", ["helium", "triplet", "hydrogen"])
def test_exact_samplers_match_density(name, request, grid):
    system = request.getfixturevalue(name)
    r = np.asarray(system.sample(jax.random.PRNGKey(0), 100000)).reshape(-1, 3)
    for key, expected in grid_moments(system, grid).items():
        values = moments(r)[key]
        assert abs(values.mean() - expected) < 5 * values.std() / np.sqrt(len(values))


def test_triplet_sampler_is_antisymmetric(triplet):
    r = np.asarray(triplet.sample(jax.random.PRNGKey(1), 100000))
    distance = np.linalg.norm(r[:, 0] - r[:, 1], axis=-1)
    assert np.quantile(distance, 0.001) > 0.01


@pytest.mark.parametrize("name", ["helium", "triplet"])
def test_metropolis_matches_density(name, request, grid):
    system = request.getfixturevalue(name)
    config = SamplerConfig(n_walkers=2048, decorrelation=20, equilibration=400)
    source = make_metropolis(system.log_psi, system.molecule, config)
    walkers = source.init(jax.random.PRNGKey(0))
    batches = []
    for key in jax.random.split(jax.random.PRNGKey(1), 30):
        walkers, r, target = source.next(key, walkers)
        batches.append(np.asarray(r).reshape(-1, 3))
    r = np.concatenate(batches)
    expected = grid_moments(system, grid)
    for key, values in moments(r).items():
        assert abs(values.mean() - expected[key]) < 0.02 * expected[key]
    assert 0.05 < float(walkers.step_size) < 2.0


def test_metropolis_score_target(helium):
    config = SamplerConfig(n_walkers=16, decorrelation=5, equilibration=10)
    source = make_metropolis(helium.log_psi, helium.molecule, config)
    walkers, r, target = source.next(jax.random.PRNGKey(2), source.init(jax.random.PRNGKey(0)))
    expected = -4.0 * r / jnp.linalg.norm(r, axis=-1, keepdims=True)
    np.testing.assert_allclose(target, expected, rtol=1e-10)
