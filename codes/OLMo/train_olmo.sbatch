#!/bin/bash

#SBATCH -A <your-project>
#SBATCH -N 1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=0
#SBATCH -t 24:00:00
#SBATCH -J olmo_scratch
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

module purge
# load what you normally use on Berzelius (cuda handled by DGX image / modules in your env)
source activate <your-conda-env>

export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# train script you create from olmo-core example
torchrun --nproc_per_node=8 train_scratch.py \
  --train_tokens_npy /path/to/tokens/dolma_v1_6-sample_tokens.npy \
  --seq_len 1024 \
  --global_batch_size 256 \
  --run_name scratch_dolma_v1_6_sample


