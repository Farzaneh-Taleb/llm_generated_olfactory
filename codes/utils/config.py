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

# RIDGE_ALPHAS_RANGE = (2, 7, 16)
# RIDGE_CV_FOLDS = 5

# DEFAULT_PCA_COMPONENTS = 30