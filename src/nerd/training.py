import time

import jax
import jax.numpy as jnp
import optax

from .losses import noise_contrastive, score_matching


def make_optimizer(config):
    schedule = optax.cosine_decay_schedule(
        config.learning_rate, int(config.decay_fraction * config.steps), alpha=config.final_ratio
    )
    return optax.adam(schedule, b1=config.b1, b2=config.b2)


def make_loss(model, noise, molecule, nce_weight):
    spins = jnp.asarray(molecule.spins)
    active_spins = jnp.asarray(molecule.spin_counts > 0, dtype=float)

    def loss(params, key, r, target):
        sm = score_matching(model, params, r, target, spins)
        if not nce_weight:
            return sm, {"score_matching": sm, "nce": jnp.zeros(())}
        noise_r = noise.sample(key, r.shape[0] * molecule.n_elec).astype(r.dtype)
        nce = noise_contrastive(model, noise, params, r, spins, noise_r, active_spins)
        return sm + nce_weight * nce, {"score_matching": sm, "nce": nce}

    return loss


def make_update(loss, optimizer):
    @jax.jit
    def update(params, opt_state, key, r, target):
        (value, metrics), grads = jax.value_and_grad(loss, has_aux=True)(params, key, r, target)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, {"loss": value, **metrics}

    return update


def fit(key, model, noise, molecule, source, config, initial_r=None, log_every=0, log=print, callback=None):
    key_init, key_source, key = jax.random.split(key, 3)
    params = model.init(key_init)
    optimizer = make_optimizer(config)
    opt_state = optimizer.init(params)
    update = make_update(make_loss(model, noise, molecule, config.nce_weight), optimizer)
    state = source.init(key_source, initial_r) if initial_r is not None else source.init(key_source)
    history = []
    start = time.perf_counter()
    for step in range(config.steps):
        key, key_data, key_loss = jax.random.split(key, 3)
        state, r, target = source.next(key_data, state)
        params, opt_state, metrics = update(params, opt_state, key_loss, r, target)
        if log_every and (step % log_every == 0 or step == config.steps - 1):
            record = {"step": step, "time": time.perf_counter() - start, **{k: float(v) for k, v in metrics.items()}}
            history.append(record)
            log(" ".join(f"{k}={v:.6g}" for k, v in record.items()))
        if callback is not None:
            callback(step, params, history)
    return params, history
