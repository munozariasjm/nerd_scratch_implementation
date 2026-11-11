import argparse
from pathlib import Path
from dataclasses import replace

import jax

from nerd import ModelConfig, SamplerConfig, TrainConfig, make_metropolis, make_model, make_noise
from nerd.config import paper_steps, to_dict
from nerd.io import save
from nerd.training import fit
from nerd.wavefunctions import deepqmc


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--checkpoint-step", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--walkers", type=int, default=SamplerConfig.n_walkers)
    parser.add_argument("--decorrelation", type=int, default=SamplerConfig.decorrelation)
    parser.add_argument("--equilibration", type=int, default=SamplerConfig.equilibration)
    parser.add_argument("--nce-weight", type=float, default=TrainConfig.nce_weight)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    parser.add_argument("--log-every", type=int, default=500)
    return parser.parse_args()


def main():
    jax.config.update("jax_default_matmul_precision", "highest")
    args = parse()
    wavefunction = deepqmc.load(args.workdir, args.checkpoint_step)
    molecule = wavefunction.molecule
    model_config = ModelConfig()
    sampler_config = SamplerConfig(args.walkers, args.decorrelation, args.equilibration)
    steps = args.steps or paper_steps(molecule)
    train_config = replace(TrainConfig(), steps=steps, nce_weight=args.nce_weight, seed=args.seed)
    model = make_model(molecule, model_config)
    source = make_metropolis(wavefunction.log_psi, molecule, sampler_config)
    initial_r = wavefunction.walkers if wavefunction.walkers.shape[0] >= args.walkers else None
    params, history = fit(
        jax.random.PRNGKey(args.seed), model, make_noise(molecule), molecule, source, train_config,
        initial_r=initial_r, log_every=args.log_every,
    )
    metadata = {
        "molecule": molecule.to_dict(),
        "config": to_dict(model_config, sampler_config, train_config),
        "wavefunction": {
            "workdir": str(Path(args.workdir).resolve()),
            "step": wavefunction.step,
            "deepqmc": deepqmc.VERSION,
            "reload_error": deepqmc.reload_error(wavefunction),
        },
        "history": history,
    }
    save(args.out, params, metadata)


if __name__ == "__main__":
    main()
