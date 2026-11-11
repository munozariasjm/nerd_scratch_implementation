import jax.numpy as jnp


def safe_norm(x):
    squared = jnp.sum(x**2, axis=-1)
    positive = squared > 0
    return jnp.where(positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0)


def smooth_norm(x):
    return jnp.sqrt(1.0 + jnp.sum(x**2, axis=-1))


def nuclear_offsets(r, coords):
    return r[..., None, :] - coords


def stratified_components(charges, n):
    weights = charges / jnp.sum(charges)
    quantiles = (jnp.arange(n) + 0.5) / n
    return jnp.minimum(jnp.searchsorted(jnp.cumsum(weights), quantiles), len(charges) - 1)
