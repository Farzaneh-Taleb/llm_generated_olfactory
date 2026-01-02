#!/bin/bash -l
#
#SBATCH -J moljob
#SBATCH -o codes/0_llm_responses/logs/output_%A_%a.out
#SBATCH -e codes/0_llm_responses/logs/error_%A_%a.err
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --gpus-per-node=4
# set -euo pipefail

# Create logs early so SBATCH output paths exist
mkdir -p logs

echo "JobID: $SLURM_JOB_ID"
echo "Node:  $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"

# -------------------- Modules & Conda --------------------
module --force purge
module load Miniforge3/24.7.1-2-hpc1-bdist
source /software/sse/manual/Miniforge3/24.7.1-2/hpc1-bdist/etc/profile.d/conda.sh
conda activate llm_olfaction
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Extra safety: preload the correct C++ runtime (fixes stubborn environments)
export LD_PRELOAD="$CONDA_PREFIX/lib/libstdc++.so.6:$CONDA_PREFIX/lib/libgcc_s.so.1${LD_PRELOAD:+:$LD_PRELOAD}"

# -------------------- Runtime env --------------------
# export TOKENIZERS_PARALLELISM=false
# export PYTHONNOUSERSITE=1
# export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
# export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# Optional but recommended: put HF cache on scratch (avoid home quota/slow FS)
# export HF_HOME="/cfs/klemming/scratch/${USER}/hf_cache"
# export TRANSFORMERS_CACHE="$HF_HOME"
mkdir -p "$HF_HOME"

# Optional: HF token (do NOT hardcode in script)
# export HF_TOKEN="hf_..."

echo "Python: $(which python)"
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device_count', torch.cuda.device_count())"
nvidia-smi || true

# ---- Arguments ----
DS="${1:-keller2016}"
BPT="${2:-bysmiles}"
REPS="${3:-1}"

MODEL_NAME="meta-llama/Llama-3.1-70B-Instruct"

# Generation settings
TEMP="0.2"
TOP_P="1.0"
NEW_TOKENS="256"
DO_SAMPLE="--do_sample"          # leave empty for deterministic (temp 0); set to "--do_sample" if temp>0
SEED="0"

# Performance settings
DTYPE="bf16"
BATCH_SIZE="64"
export PROJ=/proj/rep-learning-robotics/users/x_farzt/llm_generated_olfactory
cd "$PROJ"
export PYTHONPATH="$PROJ:${PYTHONPATH:-}"

export PROJ=/proj/rep-learning-robotics/users/x_farzt/llm_generated_olfactory
cd "$PROJ"

# THIS is the important line
export PYTHONPATH="$PROJ/codes:${PYTHONPATH:-}"
python -c "import custom_utils, custom_utils.helpers; print('utils OK')"
export HF_TOKEN="$(cat ~/.secrets/hf_token)"

python -m codes.0_llm_responses.llm_response_llama \
  --ds "$DS" \
  --model_name "$MODEL_NAME" \
  --build-prompt-type "$BPT" \
  --n-repeats "$REPS" \
  --temperature "$TEMP" \
  --top_p "$TOP_P" \
  --max_new_tokens "$NEW_TOKENS" \
  $DO_SAMPLE \
  --seed "$SEED" \
  --dtype "$DTYPE" \
  --device -1 \
  --batch_size "$BATCH_SIZE" \
  --write_csv \
  --debug