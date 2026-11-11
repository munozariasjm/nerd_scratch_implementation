import argparse
import json

import jax

from nerd import ModelConfig, Molecule, becke_grid, make_model
from nerd.io import load
from nerd.observables import summary


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--level", type=int, default=6)
    parser.add_argument("--out", required=True)
    return parser.parse_args()


def main():
    jax.config.update("jax_enable_x64", True)
    args = parse()
    with open(f"{args.run}/metadata.json") as f:
        metadata = json.load(f)
    molecule = Molecule.from_dict(metadata["molecule"])
    model = make_model(molecule, ModelConfig(**metadata["config"]["ModelConfig"]))
    params, _ = load(args.run, model.init(jax.random.PRNGKey(0)))
    params = jax.tree.map(lambda x: jax.numpy.asarray(x, jax.numpy.float64), params)
    points, weights = becke_grid(molecule, args.level)
    result = {"run": args.run, "grid_level": args.level, **summary(model, params, molecule, points, weights)}
    with open(args.out, "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
