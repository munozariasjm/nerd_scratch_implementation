from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nerd import SamplerConfig, TrainConfig, fit, make_metropolis, make_model, make_noise
from nerd.observables import (
    contact_densities,
    log_electron_counts,
    make_density,
    normalization_shift,
    total_variation,
)
from nerd.wavefunctions.analytic import exact_source

pytestmark = pytest.mark.slow


def train(system, steps, walkers, source=None, nce_weight=1.0):
    molecule = system.molecule
    model = make_model(molecule)
    config = replace(TrainConfig(), steps=steps, nce_weight=nce_weight)
    source = source or exact_source(system, walkers)
    params, history = fit(jax.random.PRNGKey(0), model, make_noise(molecule), molecule, source, config, log_every=0)
    return model, params


def assess(system, model, params, grid):
    molecule = system.molecule
    points, weights = grid(molecule, 5)
    log_counts = log_electron_counts(model, params, points, weights)
    density = make_density(model, params, normalization_shift(log_counts, molecule))
    rho = jnp.sum(density(jnp.asarray(points)), axis=-1)
    exact_rho = system.density(jnp.asarray(points))
    contact = np.asarray(contact_densities(density, molecule))
    exact_contact = np.asarray(system.density(jnp.asarray(molecule.coords)))
    raw = np.exp(np.asarray(log_counts))[molecule.spin_counts > 0] / molecule.spin_counts[molecule.spin_counts > 0]
    return {
        "contact_error": np.abs(contact / exact_contact - 1),
        "total_variation": float(total_variation(rho, exact_rho, weights)) / molecule.n_elec,
        "count_error": np.abs(raw - 1),
    }


def test_helium_like(helium, grid):
    result = assess(helium, *train(helium, 3000, 256), grid)
    assert result["contact_error"].max() < 0.01
    assert result["total_variation"] < 0.01
    assert result["count_error"].max() < 0.02


def test_triplet_with_nodes(triplet, grid):
    result = assess(triplet, *train(triplet, 3000, 256), grid)
    assert result["contact_error"].max() < 0.02
    assert result["total_variation"] < 0.02
    assert result["count_error"].max() < 0.03


def test_hydrogen_molecule_like(hydrogen, grid):
    result = assess(hydrogen, *train(hydrogen, 3000, 256), grid)
    assert result["contact_error"].max() < 0.01
    assert result["total_variation"] < 0.01
    assert result["count_error"].max() < 0.02


def test_metropolis_end_to_end(helium, grid):
    config = SamplerConfig(n_walkers=256, decorrelation=10, equilibration=500)
    source = make_metropolis(helium.log_psi, helium.molecule, config)
    result = assess(helium, *train(helium, 3000, 256, source=source), grid)
    assert result["contact_error"].max() < 0.015
    assert result["total_variation"] < 0.015


def test_score_matching_alone_leaves_normalization_free(helium, grid):
    result = assess(helium, *train(helium, 1500, 256, nce_weight=0.0), grid)
    assert result["total_variation"] < 0.02
