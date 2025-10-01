#!/bin/bash

# reset modules
source /opt/cray/pe/cpe/23.12/restore_lmod_system_defaults.sh

# load fmri grid
source "/cfs/klemming/projects/supr/olfactory_alignment/olfactory-fmri-alignment-NEW/finetune_reg/fmri_finetune_grid.sh"

compute_total() {
  echo $(( ${#datasets[@]} * ${#subjects[@]} * ${#n_folds[@]} \
         * ${#n_components[@]} * ${#models[@]} * ${#behavior_embeddings[@]} \
         * ${#z_scores[@]} ))
}

mkdir -p logs
total_jobs=$(compute_total)
echo "Submitting $total_jobs jobs..."
# RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)}"
echo "RUN_ID=${RUN_ID}"

sbatch --export=ALL,RUN_ID="$RUN_ID" --array=0-$((total_jobs-1)) "$(dirname "$0")/regresion_behavior_run_job.sh"
