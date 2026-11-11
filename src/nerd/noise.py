import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

from .geometry import nuclear_offsets, safe_norm, stratified_components


class Noise(NamedTuple):
    sample: Callable
    log_prob: Callable


def make_noise(molecule):
    coords = jnp.asarray(molecule.coords)
    charges = jnp.asarray(molecule.charges)
    log_weights = jnp.log(charges / jnp.sum(charges))

    def log_prob(r):
        distances = safe_norm(nuclear_offsets(r, coords))
        return logsumexp(log_weights - distances, axis=-1) - math.log(8 * math.pi)

    def sample(key, n):
        key_radius, key_direction = jax.random.split(key)
        radius = jax.random.gamma(key_radius, 3.0, (n,))
        direction = jax.random.normal(key_direction, (n, 3))
        direction = direction / jnp.linalg.norm(direction, axis=-1, keepdims=True)
        return coords[stratified_components(charges, n)] + radius[:, None] * direction

    return Noise(sample, log_prob)
