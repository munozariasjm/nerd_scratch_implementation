import math
from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from ..sampling import Source, score_target
from ..system import Molecule


class ExactSystem(NamedTuple):
    molecule: Molecule
    log_psi: Callable
    density: Callable
    sample: Callable


def _directions(key, n):
    v = jax.random.normal(key, (*n, 3))
    return v / jnp.linalg.norm(v, axis=-1, keepdims=True)


def _sample_s(key, shape, exponent):
    key_radius, key_direction = jax.random.split(key)
    radius = jax.random.gamma(key_radius, 3.0, shape) / (2 * exponent)
    return radius[..., None] * _directions(key_direction, shape)


def _sample_pz(key, shape, exponent):
    key_radius, key_cos, key_phi = jax.random.split(key, 3)
    radius = jax.random.gamma(key_radius, 5.0, shape) / exponent
    u = jax.random.uniform(key_cos, shape, minval=-1.0, maxval=1.0)
    cos = jnp.sign(u) * jnp.abs(u) ** (1 / 3)
    sin = jnp.sqrt(1 - cos**2)
    phi = jax.random.uniform(key_phi, shape, maxval=2 * math.pi)
    return radius[..., None] * jnp.stack([sin * jnp.cos(phi), sin * jnp.sin(phi), cos], axis=-1)


def _first_accepted(candidates, accept):
    index = jnp.argmax(accept, axis=-1)
    return jnp.take_along_axis(candidates, index[..., None, None], axis=-2)[..., 0, :]


def helium_like(z=2.0):
    molecule = Molecule([[0.0, 0.0, 0.0]], [z], 1, 1)

    def log_psi(r):
        return -z * jnp.sum(jnp.linalg.norm(r, axis=-1))

    def density(r):
        return 2 * z**3 / math.pi * jnp.exp(-2 * z * jnp.linalg.norm(r, axis=-1))

    def sample(key, n):
        return _sample_s(key, (n, 2), z)

    return ExactSystem(molecule, log_psi, density, jax.jit(sample, static_argnums=1))


def triplet_1s2p(z=3.0, n_candidates=64):
    molecule = Molecule([[0.0, 0.0, 0.0]], [z], 2, 0)

    def orbitals(r):
        d = jnp.linalg.norm(r, axis=-1)
        s = jnp.sqrt(z**3 / math.pi) * jnp.exp(-z * d)
        p = jnp.sqrt(z**5 / (32 * math.pi)) * r[..., 2] * jnp.exp(-z * d / 2)
        return s, p

    def determinant(r1, r2):
        s1, p1 = orbitals(r1)
        s2, p2 = orbitals(r2)
        return s1 * p2 - p1 * s2

    def log_psi(r):
        return jnp.log(jnp.abs(determinant(r[0], r[1])))

    def density(r):
        s, p = orbitals(r)
        return s**2 + p**2

    def sample_density(key, shape):
        key_choice, key_s, key_p = jax.random.split(key, 3)
        choose_s = jax.random.bernoulli(key_choice, 0.5, shape)
        return jnp.where(choose_s[..., None], _sample_s(key_s, shape, z), _sample_pz(key_p, shape, z))

    def sample(key, n):
        key_first, key_candidates, key_accept = jax.random.split(key, 3)
        r1 = sample_density(key_first, (n,))
        candidates = sample_density(key_candidates, (n, n_candidates))
        probability = determinant(r1[:, None], candidates) ** 2 / (2 * density(r1)[:, None] * density(candidates))
        accept = jax.random.uniform(key_accept, probability.shape) < probability
        return jnp.stack([r1, _first_accepted(candidates, accept)], axis=1)

    return ExactSystem(molecule, log_psi, density, jax.jit(sample, static_argnums=1))


def hydrogen_molecule_like(exponent=1.0, bond=1.4, n_candidates=32):
    coords = jnp.array([[0.0, 0.0, -bond / 2], [0.0, 0.0, bond / 2]])
    kato_charge = exponent / (1 + math.exp(-exponent * bond))
    molecule = Molecule(coords, [kato_charge, kato_charge], 1, 1)
    x = exponent * bond
    overlap = math.exp(-x) * (1 + x + x**2 / 3)
    norm = 2 * math.pi / exponent**3 * (1 + overlap)

    def atomic(r):
        return jnp.exp(-exponent * jnp.linalg.norm(r[..., None, :] - coords, axis=-1))

    def orbital(r):
        return jnp.sum(atomic(r), axis=-1)

    def log_psi(r):
        return jnp.sum(jnp.log(orbital(r)))

    def density(r):
        return 2 * orbital(r) ** 2 / norm

    def sample(key, n):
        key_center, key_s, key_accept = jax.random.split(key, 3)
        shape = (n, 2, n_candidates)
        center = jax.random.bernoulli(key_center, 0.5, shape).astype(int)
        candidates = coords[center] + _sample_s(key_s, shape, exponent)
        a = atomic(candidates)
        probability = orbital(candidates) ** 2 / (2 * jnp.sum(a**2, axis=-1))
        accept = jax.random.uniform(key_accept, shape) < probability
        return _first_accepted(candidates, accept)

    return ExactSystem(molecule, log_psi, density, jax.jit(sample, static_argnums=1))


def exact_source(system, n_walkers):
    target = score_target(system.log_psi)

    def init(key, r=None):
        return None

    @jax.jit
    def next(key, state):
        r = system.sample(key, n_walkers)
        return state, r, target(r)

    return Source(init, next)
