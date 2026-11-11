import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp
from jax.scipy.special import erf, logsumexp

from .config import ModelConfig
from .geometry import nuclear_offsets, safe_norm, smooth_norm


class Model(NamedTuple):
    init: Callable
    log_density: Callable
    spin_log_density: Callable
    score: Callable


def _linear(key, n_in, n_out):
    initializer = jax.nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
    return {"w": initializer(key, (n_in, n_out)), "b": jnp.zeros(n_out)}


def _apply(layer, x):
    return x @ layer["w"] + layer["b"]


def features(offsets):
    s = smooth_norm(offsets)
    scale = jnp.log1p(s) / s
    per_nucleus = jnp.concatenate([offsets * scale[..., None], (s * scale)[..., None]], axis=-1)
    return per_nucleus.reshape(*per_nucleus.shape[:-2], -1)


def network(params, x):
    h = _apply(params["input"], x)
    for layer in params["residual"]:
        h = h + _apply(layer, jax.nn.silu(h))
    return _apply(params["output"], h)


def envelope(params, offsets):
    s = smooth_norm(offsets)[..., None, :]
    return logsumexp(params["log_coefficient"] - jax.nn.softplus(params["exponent"]) * s, axis=-1)


def cusp(offsets, charges):
    return -2 * math.sqrt(math.pi) * jnp.sum(erf(charges * safe_norm(offsets) / 2), axis=-1)


def make_model(molecule, config=ModelConfig()):
    coords = jnp.asarray(molecule.coords)
    charges = jnp.asarray(molecule.charges)
    n_nuc = molecule.n_nuc

    def init(key):
        keys = jax.random.split(key, config.n_residual + 2)
        return {
            "envelope": {
                "exponent": jnp.ones((2, n_nuc)),
                "log_coefficient": jnp.zeros((2, n_nuc)),
            },
            "mlp": {
                "input": _linear(keys[0], 4 * n_nuc, config.width),
                "residual": [_linear(k, config.width, config.width) for k in keys[1:-1]],
                "output": _linear(keys[-1], config.width, 2),
            },
        }

    def log_density(params, r):
        offsets = nuclear_offsets(r, coords)
        smooth = network(params["mlp"], features(offsets)) + envelope(params["envelope"], offsets)
        return smooth + cusp(offsets, charges)[..., None]

    def spin_log_density(params, r, spin):
        spin = jnp.broadcast_to(spin, r.shape[:-1])
        return jnp.take_along_axis(log_density(params, r), spin[..., None], axis=-1)[..., 0]

    def score(params, r, spin):
        return jax.grad(lambda x: spin_log_density(params, x, spin).sum())(r)

    return Model(init, log_density, spin_log_density, score)
