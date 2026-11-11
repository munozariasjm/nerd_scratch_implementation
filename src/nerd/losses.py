import jax
import jax.numpy as jnp


def score_matching(model, params, r, target, spins):
    return jnp.mean(jnp.sum((model.score(params, r, spins) - target) ** 2, axis=-1))


def noise_contrastive(model, noise, params, r, spins, noise_r, active_spins):
    log_nu = jnp.log(noise_r.shape[0] / r.shape[0])
    data_logits = model.spin_log_density(params, r, spins) - noise.log_prob(r) - log_nu
    noise_logits = model.log_density(params, noise_r) - noise.log_prob(noise_r)[:, None] - log_nu
    data_term = jnp.sum(jax.nn.softplus(-data_logits))
    noise_term = jnp.sum(jax.nn.softplus(noise_logits) * active_spins)
    return (data_term + noise_term) / (data_logits.size + noise_r.shape[0] * jnp.sum(active_spins))
