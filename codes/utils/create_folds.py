import sys
import os
from pathlib import Path
from sklearn.model_selection import KFold
from config import SEED
import pandas as pd
# --- Make imports work from anywhere ---
REPO_DIR = Path(__file__).resolve().parent
sys.path.append(str(REPO_DIR))  # so `config` is importable
from config import BASE_DIR  # do not shadow later
sys.path.append(str(REPO_DIR))  # ensure utils is on path
from helpers import common_cids_per_ds

import argparse
def save_fold_indices(BASE_DIR, n_fold,ds):
    
    common_cids = common_cids_per_ds(BASE_DIR,ds)
    print("ds",ds,len(common_cids))
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=SEED)
    
    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(common_cids)):
        # map indices -> actual CID values
        train_c = [str(common_cids[i]) for i in train_idx]
        test_c  = [str(common_cids[i]) for i in test_idx]

        
        for cid in train_c:
            rows.append({
                "cid": cid,
                "set": "train",
                "n_fold": n_fold,
                "fold_idx": fold_idx,
                "ds": ds
            })
        for cid in test_c:
            rows.append({
                "cid": cid,
                "set": "test",
                "n_fold": n_fold,
                "fold_idx": fold_idx,
                "ds": ds
            })
    
    all_folds_df = pd.DataFrame(rows)
    all_folds_df.to_csv(
        f"{BASE_DIR}/folds/fold_indices_ds-{ds}_nfold-{n_fold}.csv", index=False)

def main():

    
    for ds in  ["sagar2023"]:
        for n_fold in [5,10]:
            save_fold_indices(BASE_DIR, n_fold,ds)

if __name__ == "__main__":
    main()