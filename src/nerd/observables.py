import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import logsumexp


def evaluate(fn, points, chunk=32768):
    points = jnp.asarray(points)
    n = points.shape[0]
    padded = jnp.concatenate([points, jnp.zeros(((-n) % chunk, *points.shape[1:]), points.dtype)])
    values = jax.lax.map(fn, padded.reshape(-1, chunk, *points.shape[1:]))
    return values.reshape(-1, *values.shape[2:])[:n]


def log_electron_counts(model, params, points, weights):
    log_rho = evaluate(jax.jit(lambda x: model.log_density(params, x)), points)
    return logsumexp(log_rho, b=jnp.asarray(weights)[:, None], axis=0)


def normalization_shift(log_counts, molecule):
    counts = jnp.asarray(molecule.spin_counts, dtype=log_counts.dtype)
    return jnp.where(counts > 0, log_counts - jnp.log(jnp.maximum(counts, 1)), jnp.inf)


def normalized_density(model, params, molecule, points, weights):
    shift = normalization_shift(log_electron_counts(model, params, points, weights), molecule)
    return make_density(model, params, shift)


def make_density(model, params, log_norm):
    return lambda r: jnp.exp(model.log_density(params, r) - log_norm)


def integrate(values, weights):
    return jnp.tensordot(jnp.asarray(weights), values, axes=(0, 0))


def contact_densities(density, molecule):
    return jnp.sum(density(jnp.asarray(molecule.coords)), axis=-1)


def dipole(rho, points, weights, molecule):
    nuclear = jnp.sum(jnp.asarray(molecule.charges)[:, None] * jnp.asarray(molecule.coords), axis=0)
    return nuclear - integrate(rho[:, None] * points, weights)


def quadrupole(rho, points, weights):
    outer = 3 * points[:, :, None] * points[:, None, :]
    trace = jnp.sum(points**2, axis=-1)[:, None, None] * jnp.eye(3)
    return integrate(rho[:, None, None] * (outer - trace), weights)


def nuclear_potential_energy(rho, points, weights, molecule):
    distances = jnp.linalg.norm(points[:, None, :] - molecule.coords, axis=-1)
    potential = -jnp.sum(jnp.asarray(molecule.charges) / distances, axis=-1)
    return integrate(rho * potential, weights)


def hellmann_feynman_forces(rho, points, weights, molecule):
    coords = jnp.asarray(molecule.coords)
    charges = jnp.asarray(molecule.charges)
    pair = coords[:, None, :] - coords[None, :, :]
    distance = jnp.linalg.norm(pair, axis=-1) + jnp.eye(len(charges))
    off_diagonal = 1 - jnp.eye(len(charges))
    strength = charges[:, None] * charges[None, :] * off_diagonal / distance**3
    nuclear = jnp.sum(strength[..., None] * pair, axis=1)
    offsets = points[:, None, :] - coords
    field = offsets / jnp.linalg.norm(offsets, axis=-1, keepdims=True) ** 3
    electronic = charges[:, None] * integrate(rho[:, None, None] * field, weights)
    return nuclear + electronic


def total_variation(rho_a, rho_b, weights):
    return 0.5 * integrate(jnp.abs(rho_a - rho_b), weights)


def effective_potential(model, params, r, spin):
    def log_rho(x):
        return model.spin_log_density(params, x, spin)

    gradient = jax.grad(log_rho)(r)
    laplacian = jnp.trace(jax.hessian(log_rho)(r))
    return jnp.sum(gradient**2) / 8 + laplacian / 4


def summary(model, params, molecule, points, weights):
    log_counts = log_electron_counts(model, params, points, weights)
    density = make_density(model, params, normalization_shift(log_counts, molecule))
    rho_spin = evaluate(jax.jit(density), points)
    rho = jnp.sum(rho_spin, axis=-1)
    points = jnp.asarray(points)
    return {
        "raw_electron_counts": np.exp(np.asarray(log_counts)).tolist(),
        "electron_counts": np.asarray(integrate(rho_spin, weights)).tolist(),
        "contact_densities": np.asarray(contact_densities(density, molecule)).tolist(),
        "dipole": np.asarray(dipole(rho, points, weights, molecule)).tolist(),
        "quadrupole": np.asarray(quadrupole(rho, points, weights)).tolist(),
        "nuclear_potential_energy": float(nuclear_potential_energy(rho, points, weights, molecule)),
        "hellmann_feynman_forces": np.asarray(hellmann_feynman_forces(rho, points, weights, molecule)).tolist(),
    }
