# NERD Implementation

A JAX implementation of neural electron real-space densities (NERD). It follows Cheng et al., *J. Chem. Phys.* 162,
034120 (2025), arXiv:2409.01306.

Each spin density is modelled as

```
log ρσ(r) = MLPσ(features) + log Σ_I π_Iσ exp(-ζ_Iσ s_I) - 2√π Σ_I erf(Z_I |r - R_I| / 2),   s_I = √(1 + |r - R_I|²)
```

and trained on samples of a fixed wave function Ψ with two losses:

- **Score matching** against `2 ∇_i log|Ψ|`.
- **Noise-contrastive estimation** against `p_n(r) ∝ Σ_I Z_I exp(-|r - R_I|)`.

## Layout

```
.
├── pyproject.toml
├── src/nerd/
│   ├── system.py            Molecule: nuclei, charges, spin counts
│   ├── config.py            ModelConfig, SamplerConfig, TrainConfig, paper step rule
│   ├── geometry.py          distances, smooth norm, stratified split by charge
│   ├── model.py             features, residual MLP, envelope, cusp, spin heads, score
│   ├── noise.py             NCE noise: stratified sampling, normalized log density
│   ├── losses.py            score matching, NCE
│   ├── sampling.py          all-electron Metropolis source of (r, 2∇log|Ψ|) batches
│   ├── training.py          Adam with cosine decay, update step, fit loop
│   ├── quadrature.py        Becke-partitioned Lebedev grids (PySCF)
│   ├── observables.py       normalization, contact densities, dipole, quadrupole,
│   │                        nuclear potential, Hellmann–Feynman forces,
│   │                        total variation, effective potential
│   ├── io.py                parameter and metadata storage
│   └── wavefunctions/
│       ├── analytic.py      exactly sampled wave functions with known densities
│       └── deepqmc.py       DeepQMC 1.3 checkpoint loader
├── scripts/
│   ├── fit.py               command-line fit
│   └── evaluate.py          command-line evaluation
├── tests/
│   ├── conftest.py          fixtures: test molecules, quadrature grids
│   ├── oracles.py           exact densities wrapped as models
│   ├── test_model.py
│   ├── test_noise.py
│   ├── test_losses.py
│   ├── test_sampling.py
│   ├── test_quadrature.py
│   ├── test_observables.py
│   └── test_training.py     end-to-end fits (slow)
└── checks/he_deepqmc/
    ├── run.sh               Slurm job: fit, evaluate, compare
    └── compare.py           comparison with the exact contact density of a He Psiformer
```

## Architecture

- MLP | input layer, 2 pre-activation residual SiLU blocks, output layer; width 64; 2 heads | paper, oneqmc |
- Envelope init | ζ = softplus(1), π = 1 per nucleus and spin | oneqmc |
- Optimizer | Adam (0.95, 0.999); cosine decay from 1e-2 over 0.8 × steps, α = 6e-5 | paper |
- Steps | max(25000, 5000 N) | paper (atoms) |
- NCE weight | 1 | paper |
- NCE noise | N noise points per configuration, components stratified by Z, shared by both spin heads | oneqmc |
- NCE ratio | log ν = log(n_noise / n_configurations), so that `exp(log ρσ)` is normalized to Nσ at the optimum | derived |
- Sampler | all-electron Gaussian Metropolis, 4096 walkers, 60 steps between batches; step size adapted to acceptance 0.57 during equilibration, then frozen | oneqmc |
- Evaluation | density renormalized to Nσ by quadrature before every observable | oneqmc |

## Usage

```bash
pip install -e ".[deepqmc,test]"
python scripts/fit.py --workdir <deepqmc run> --out runs/He
python scripts/evaluate.py --run runs/He --level 6 --out runs/He/evaluation.json
pytest -m "not slow"
pytest -m slow
```

From Python:

```python
import jax
from nerd import TrainConfig, fit, make_metropolis, make_model, make_noise

model = make_model(molecule)
source = make_metropolis(log_psi, molecule)
params, history = fit(jax.random.PRNGKey(0), model, make_noise(molecule), molecule, source, TrainConfig())
```

`log_psi` maps one configuration `r` of shape `[N, 3]` to `log|Ψ(r)|`. Electrons are ordered spin-up first.
# nerd_scratch_implementation
