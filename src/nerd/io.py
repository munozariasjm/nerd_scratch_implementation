import json
from pathlib import Path

import jax
import numpy as np


def _flatten(params):
    leaves, _ = jax.tree_util.tree_flatten_with_path(params)
    return {jax.tree_util.keystr(path): np.asarray(leaf) for path, leaf in leaves}


def save(directory, params, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez(directory / "params.npz", **_flatten(params))
    (directory / "metadata.json").write_text(json.dumps(metadata, indent=1))


def load(directory, template):
    directory = Path(directory)
    stored = np.load(directory / "params.npz")
    leaves, treedef = jax.tree_util.tree_flatten_with_path(template)
    params = jax.tree_util.tree_unflatten(treedef, [stored[jax.tree_util.keystr(path)] for path, _ in leaves])
    return params, json.loads((directory / "metadata.json").read_text())
