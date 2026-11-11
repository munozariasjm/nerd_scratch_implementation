from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable, NamedTuple

import deepqmc
import jax
import jax.numpy as jnp
import numpy as np
from deepqmc.app import instantiate_ansatz
from deepqmc.log import CheckpointStore
from deepqmc.postprocess.workdir import last_checkpoint_iteration
from deepqmc.types import PhysicalConfiguration
from hydra.utils import instantiate
from omegaconf import OmegaConf

from ..system import Molecule

VERSION = version(deepqmc.__name__)


class DeepQMCWavefunction(NamedTuple):
    molecule: Molecule
    log_psi: Callable
    walkers: Any
    stored_log_psi: Any
    step: int


def _root_config(training_dir):
    config = OmegaConf.load(training_dir / ".hydra" / "config.yaml")
    while config.task.get("restdir"):
        training_dir = Path(config.task.restdir)
        config = OmegaConf.load(training_dir / ".hydra" / "config.yaml")
    return config


def load(workdir, step=None, state=0):
    training_dir = Path(workdir) / "training"
    config = _root_config(training_dir)
    hamiltonian = instantiate(config.hamil, _recursive_=True, _convert_="all")
    ansatz = instantiate_ansatz(hamiltonian, instantiate(config.ansatz, _recursive_=True, _convert_="all"))
    step = last_checkpoint_iteration(training_dir) if step is None else step
    _, train_state = CheckpointStore.load(training_dir / f"chkpt-{step}.pt", deserialize=False)
    params = jax.tree.map(lambda x: x[state], train_state.params)
    electrons = train_state.sampler["elec"]
    coords = jnp.asarray(hamiltonian.mol.coords)
    molecule = Molecule(np.asarray(hamiltonian.mol.coords), np.asarray(hamiltonian.mol.charges),
                        int(hamiltonian.n_up), int(hamiltonian.n_down))

    def log_psi(r):
        return ansatz.apply(params, PhysicalConfiguration(coords, r, jnp.array(0))).log

    return DeepQMCWavefunction(molecule, log_psi, electrons["r"][0, state], electrons["psi"].log[0, state], int(step))


def reload_error(wavefunction, n=256):
    reloaded = jax.vmap(wavefunction.log_psi)(wavefunction.walkers[:n])
    return float(jnp.max(jnp.abs(reloaded - wavefunction.stored_log_psi[:n])))
