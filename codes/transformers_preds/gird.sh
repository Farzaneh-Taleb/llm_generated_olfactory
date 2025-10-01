#!/bin/bash
# grid.sh – defines the experiment grid
unset -v datasets subjects n_folds models behavior_embeddings unfreeze_layers \
         lrs weight_decays batch_sizes OUT_DIR EPOCHS RUN_ID embed_type \
         z_scores n_components

# Force array semantics (global) so we don't get scalars from env
declare -ag datasets subjects n_folds models behavior_embeddings unfreeze_layers \
             lrs weight_decays batch_sizes z_scores n_components
# Datasets to iterate over
datasets=(
  "sagar2023"
)

# Subjects, folds, and model/hparam grid
subjects=(1 2 3)
n_folds=(10)
models=(
  "ibm/MoLFormer-XL-both-10pct"
  "seyonec/ChemBERTa-zinc-base-v1"
  "HUBioDataLab/SELFormer"
  "jonghyunlee/ChemBERT_ChEMBL_pretrained"
)
behavior_embeddings=("")       # empty string => use default behavior columns inside your script

unfreeze_layers=("adaptive")
OUT_DIR="Sep20"
EPOCHS=40
RUN_ID="11_finetune_by_fmri"
embed_type="can"
lrs=(1e-5)
weight_decays=(0.0)
batch_sizes=(16)
z_scores=(True)
n_components=("None")   
rois=("ALL")
trs=(-1)
finetune_by='fmri'     # 'beh' or 'fmri'   <-- fixed spelling
