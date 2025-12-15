"""
Configuration file for olfactory-fmri-alignment project.
Contains common constants and settings used across the project.
"""

import os

# BASE_DIR = '/cfs/klemming/projects/supr/olfactory_alignment'
BASE_DIR = '/Volumes/work/phd/llm_generated_olfactory'

PROJECT_DIR = os.path.join(BASE_DIR, 'codes')

SEED = 2024

MID_DIR = 'data'

"""
Configuration constants for models and layers.
"""

# Model configurations
MODELS = [
    'openpom', 'behavior', 'molecular_descriptors', 'ChemBERT_ChEMBL_pretrained',
    'ChemBERTa-zinc-base-v1', 'MoLFormer-XL-both-10pct', 'SELFormer',
    'smiles-gpt', 'decoder_BARTSmiles', 'encoder_BARTSmiles', 'molgpt',
    'ChemGPT-4.7M', 'ChemGPT-19M', 'ChemGPT-1.2B'
]

# Layer endpoints for each model
LAYERS_END = [1, 1, 1, 8, 6, 12, 12, 4, 12, 12, 12, 24, 24, 24]

# ROI configurations  
ROIS = ["PirF", "PirT", "AMY", "OFC"]
P_VALUES = [[5, 5, 5], [4, 4, 5], [4, 3, 4], [3, 3, 5]]

INPUT_TYPES_CAN = {''}
INPUT_TYPES_CAN = {m: ("canonicalselfies" if m == "SELFormer" else "canonicalsmiles") for m in MODELS}

INPUT_TYPES_ISO = {''}
INPUT_TYPES_ISO = {m: ("isomericselfies" if m == "SELFormer" else "isomericsmiles") for m in MODELS}

# def get_input_type(model_name: str) -> str:
#     """Return the input type for a model (defaults to 'smiles' if unknown)."""
#     return INPUT_TYPES.get(model_name, "smiles")
SYSTEM_MSG = "Output ONLY valid JSON."
INPUT_TYPE = "isomericsmiles"         # 'isomericsmiles' or 'cid'
BATCH_REGISTRY = f"{BASE_DIR}/results/responses/llm_responses/batch_registry.jsonl"
BATCH_REGISTRY = f"{BASE_DIR}/results/responses/llm_responses/batch_registry.jsonl"
RATE_RANGE = {
    "keller2016": (0.0, 100.0),
    "sagar2023": (-1, 1),
    "leffingwell": (0.0, 1.0),
    "ravia": (0.0, 1.0),
    "snitz2013": (0.0, 1.0),
}
# RATE_RANGE = {
#     "keller2016": (0.0, 100.0),
#     "sagar2023": (-1,1),
#     "leffingwell": (0.0, 1.0),
#     "ravia": (0.0, 1.0),
#     "snitz": (0.0, 1.0),
#     # Add others as needed
# }
# RATE_RANGE = {
#     "keller2016": (0.0, 100.0),
#     "sagar2023": (-1, 1),
#     "leffingwell": (0.0, 1.0),
#     "ravia": (0.0, 1.0),
#     "snitz2013": (0.0, 1.0),
# }


# RATE_RANGE = {
#     "keller2016": (0.0, 100.0),
#     "sagar2023": (-1, 1),
#     "leffingwell": (0.0, 1.0),
#     "ravia": (0.0, 1.0),
#     "snitz2013": (0.0, 1.0),
# }


# RATE_RANGE = {
#     "keller2016": (0.0, 100.0),
#     "sagar2023": (-1,1),
#     "leffingwell": (0.0, 1.0),
#     "ravia": (0.0, 1.0),
#     "snitz": (0.0, 1.0),
#     # Add others as needed
# }

BUILD_PROMPT_CHOICES = ("bysmiles", "byname")

INCLUDE_CONFIDENCE = False



# Pairwise column names
PAIR_CID1 = "cid stimulus 1"
PAIR_CID2 = "cid stimulus 2"
PAIR_SIMILARITY = "similarity"   # human ref (not used in prompt)
PAIR_NAME1 = "name stimulus 1"
PAIR_NAME2 = "name stimulus 2"