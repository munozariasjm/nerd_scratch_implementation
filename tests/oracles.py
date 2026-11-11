import jax.numpy as jnp

from nerd import Model


def oracle_model(spin_densities):
    def log_density(params, r):
        return jnp.log(spin_densities(r)) + params

    def spin_log_density(params, r, spin):
        spin = jnp.broadcast_to(spin, r.shape[:-1])
        return jnp.take_along_axis(log_density(params, r), spin[..., None], axis=-1)[..., 0]

    return Model(None, log_density, spin_log_density, None)


def helium_spin_densities(system):
    return lambda r: jnp.stack([system.density(r) / 2] * 2, axis=-1)


def triplet_spin_densities(system):
    return lambda r: jnp.stack([system.density(r), jnp.full(r.shape[:-1], 1e-3)], axis=-1)
