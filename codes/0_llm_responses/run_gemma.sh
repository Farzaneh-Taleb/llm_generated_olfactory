#!/bin/bash -l
#
#SBATCH -J gemma_olf
#SBATCH -o codes/0_llm_responses/logs/output_%A_%a.out
#SBATCH -e codes/0_llm_responses/logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1

set -euo pipefail

# -------------------- Logging --------------------
mkdir -p codes/0_llm_responses/logs

echo "JobID: $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# -------------------- Modules & Conda --------------------
module --force purge
module load Miniforge3/24.7.1-2-hpc1-bdist

source /software/sse/manual/Miniforge3/24.7.1-2/hpc1-bdist/etc/profile.d/conda.sh
conda activate llm_olfaction

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"

# -------------------- Runtime --------------------
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# HuggingFace cache (recommended)
export HF_HOME="/cfs/klemming/scratch/${USER}/hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Optional (if gated model)
# export HF_TOKEN="$(cat ~/.secrets/hf_token)"

# -------------------- Debug info --------------------
echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
nvidia-smi || true

# -------------------- Project setup --------------------
export PROJ=/proj/rep-learning-robotics/users/x_farzt/llm_generated_olfactory
cd "$PROJ"

export PYTHONPATH="$PROJ/codes:${PYTHONPATH:-}"

python -c "import utils.helpers; print('utils OK')"

# -------------------- Arguments --------------------
DS="${1:-keller2016}"
BPT="${2:-bysmiles}"
REPS="${3:-1}"
TEMP="${4:-0.2}"
TEMP_TYPE="${5:-1}"

# -------------------- Run Gemma --------------------
echo "Running dataset=$DS | prompt=$BPT | repeats=$REPS | temp=$TEMP"

python -m codes.0_llm_responses.llm_response_gemma \
  --ds "$DS" \
  --build-prompt-type "$BPT" \
  --n-repeats "$REPS" \
  --temperature "$TEMP" \
  --temp_type "$TEMP_TYPE"

echo "Done."