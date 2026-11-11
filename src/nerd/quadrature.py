import numpy as np
from pyscf import dft, gto


def becke_grid(molecule, level=5):
    atoms = [(max(1, int(round(z))), tuple(c)) for z, c in zip(molecule.charges, molecule.coords)]
    n_elec = sum(a[0] for a in atoms)
    mol = gto.M(atom=atoms, unit="Bohr", basis="sto-3g", spin=n_elec % 2, verbose=0)
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.prune = None
    grids.build(with_non0tab=False)
    return np.asarray(grids.coords), np.asarray(grids.weights)
