#!/bin/bash
#SBATCH --job-name=nerd_he_check
#SBATCH --account=iaifi_lab
#SBATCH --partition=iaifi_gpu_requeue,gpu_requeue,iaifi_gpu
#SBATCH --time=0-01:30:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=checks/he_deepqmc/slurm-%j.out

set -euo pipefail
ROOT=/n/home02/josemm/RESEARCH/FUTURE/clean_nerd_implementation
PSI=/n/home02/josemm/RESEARCH/FUTURE/rd_path_2/code/runs/psi/He_X3_final
PY=/n/home02/josemm/.conda/envs/envEDENS/bin/python
OUT=$ROOT/checks/he_deepqmc/fit
export PYTHONPATH=$ROOT/src NVIDIA_TF32_OVERRIDE=0
cd "$ROOT"
nvidia-smi --query-gpu=name --format=csv,noheader
$PY scripts/fit.py --workdir "$PSI" --checkpoint-step 5000 --out "$OUT" --steps 3125 --walkers 4096 --decorrelation 30 --log-every 250
$PY scripts/evaluate.py --run "$OUT" --level 6 --out "$ROOT/checks/he_deepqmc/evaluation.json"
$PY checks/he_deepqmc/compare.py
