from typing import Callable, NamedTuple

import jax
import jax.numpy as jnp

from .config import SamplerConfig
from .geometry import stratified_components


class Walkers(NamedTuple):
    r: jax.Array
    log_psi: jax.Array
    step_size: jax.Array


class Source(NamedTuple):
    init: Callable
    next: Callable


def score_target(log_psi):
    return jax.vmap(jax.grad(lambda r: 2 * log_psi(r)))


def initial_positions(key, molecule, n_walkers):
    coords = jnp.asarray(molecule.coords)
    centers = coords[stratified_components(jnp.asarray(molecule.charges), molecule.n_elec)]
    return centers + jax.random.normal(key, (n_walkers, molecule.n_elec, 3))


def make_metropolis(log_psi, molecule, config=SamplerConfig()):
    batched_log_psi = jax.vmap(log_psi)
    target = score_target(log_psi)
    n_blocks = max(1, config.equilibration // config.decorrelation)

    def move(walkers, key):
        key_proposal, key_accept = jax.random.split(key)
        proposal = walkers.r + walkers.step_size * jax.random.normal(key_proposal, walkers.r.shape)
        proposal_log_psi = batched_log_psi(proposal)
        log_u = jnp.log(jax.random.uniform(key_accept, walkers.log_psi.shape))
        accept = log_u < 2 * (proposal_log_psi - walkers.log_psi)
        r = jnp.where(accept[:, None, None], proposal, walkers.r)
        new_log_psi = jnp.where(accept, proposal_log_psi, walkers.log_psi)
        return Walkers(r, new_log_psi, walkers.step_size), jnp.mean(accept)

    def run(key, walkers):
        walkers, acceptance = jax.lax.scan(move, walkers, jax.random.split(key, config.decorrelation))
        return walkers, jnp.mean(acceptance)

    def adapt(walkers, key):
        walkers, acceptance = run(key, walkers)
        step_size = walkers.step_size * jnp.exp(acceptance - config.target_acceptance)
        return walkers._replace(step_size=step_size), acceptance

    @jax.jit
    def equilibrate(key, r):
        walkers = Walkers(r, batched_log_psi(r), jnp.asarray(config.initial_step, r.dtype))
        walkers, acceptance = jax.lax.scan(adapt, walkers, jax.random.split(key, n_blocks))
        return walkers, acceptance[-1]

    def init(key, r=None):
        key_position, key_equilibrate = jax.random.split(key)
        if r is None:
            r = initial_positions(key_position, molecule, config.n_walkers)
        walkers, _ = equilibrate(key_equilibrate, jnp.asarray(r)[: config.n_walkers])
        return walkers

    @jax.jit
    def next(key, walkers):
        walkers, _ = run(key, walkers)
        return walkers, walkers.r, target(walkers.r)

    return Source(init, next)
