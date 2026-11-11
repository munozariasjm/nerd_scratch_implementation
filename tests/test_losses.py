import jax
import jax.numpy as jnp
import numpy as np

from nerd import make_model, make_noise, noise_contrastive, score_matching
from nerd.sampling import score_target
from oracles import helium_spin_densities, oracle_model, triplet_spin_densities


def nce_curve(model, noise, molecule, r, shifts, key):
    spins = jnp.asarray(molecule.spins)
    active = jnp.asarray(molecule.spin_counts > 0, dtype=float)
    noise_r = noise.sample(key, r.shape[0] * molecule.n_elec)
    return np.array([float(noise_contrastive(model, noise, c, r, spins, noise_r, active)) for c in shifts])


def test_score_matching_is_zero_at_model_score(hydrogen):
    molecule = hydrogen.molecule
    model = make_model(molecule)
    params = model.init(jax.random.PRNGKey(0))
    r = hydrogen.sample(jax.random.PRNGKey(1), 64)
    spins = jnp.asarray(molecule.spins)
    assert float(score_matching(model, params, r, model.score(params, r, spins), spins)) == 0.0


def test_nce_minimum_is_the_normalized_density(helium):
    molecule = helium.molecule
    model = oracle_model(helium_spin_densities(helium))
    r = helium.sample(jax.random.PRNGKey(0), 100000)
    shifts = np.linspace(-0.1, 0.1, 41)
    curve = nce_curve(model, make_noise(molecule), molecule, r, shifts, jax.random.PRNGKey(1))
    assert abs(shifts[np.argmin(curve)]) <= 0.015


def test_nce_ignores_empty_spin_channel(triplet):
    molecule = triplet.molecule
    model = oracle_model(triplet_spin_densities(triplet))
    r = triplet.sample(jax.random.PRNGKey(0), 50000)
    shifts = np.linspace(-0.1, 0.1, 41)
    curve = nce_curve(model, make_noise(molecule), molecule, r, shifts, jax.random.PRNGKey(1))
    assert abs(shifts[np.argmin(curve)]) <= 0.02
    lifted = oracle_model(lambda x: triplet_spin_densities(triplet)(x).at[..., 1].multiply(1e3))
    lifted_curve = nce_curve(lifted, make_noise(molecule), molecule, r, shifts, jax.random.PRNGKey(1))
    np.testing.assert_allclose(curve, lifted_curve, rtol=1e-12)


def test_score_target_is_unbiased_for_the_marginal_score(triplet):
    r = triplet.sample(jax.random.PRNGKey(0), 200000)
    target = score_target(triplet.log_psi)(r).reshape(-1, 3)
    points = r.reshape(-1, 3)
    marginal = jax.vmap(jax.grad(lambda x: jnp.log(triplet.density(x))))(points)
    probe = jnp.tanh(points) * jnp.array([1.0, 0.5, 2.0])
    residual = jnp.sum((target - marginal) * probe, axis=-1)
    assert abs(float(residual.mean())) < 4 * float(residual.std()) / np.sqrt(residual.size)
    s_only = jax.vmap(jax.grad(lambda x: -3.0 * jnp.linalg.norm(x)))(points)
    wrong = jnp.sum((target - s_only) * probe, axis=-1)
    assert abs(float(wrong.mean())) > 10 * float(wrong.std()) / np.sqrt(wrong.size)
