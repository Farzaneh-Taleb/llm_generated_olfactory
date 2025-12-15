import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import KFold

from utils.config import SEED, BASE_DIR
from utils.helpers import common_cids_per_ds


def save_fold_indices(base_dir: str, n_fold: int, ds: str):
    # Load dataset
    df = pd.read_csv(f"{base_dir}/datasets/{ds}/{ds}_data.csv")
    df["cid"] = pd.to_numeric(df["cid"], errors="coerce")

    # Optional concentration filter if present
    if "concentration" in df.columns and 'keller' in ds.lower():
        df["concentration"] = df["concentration"].astype(float)
        df = df[df["concentration"] == 0.001].copy()

    # Get CIDs common to the dataset (per your helpers API)
    common_cids = sorted(map(int, common_cids_per_ds(df)))
    print("ds:", ds, "n_common_cids:", len(common_cids))

    kf = KFold(n_splits=n_fold, shuffle=True, random_state=SEED)

    rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(common_cids)):
        train_c = [str(common_cids[i]) for i in train_idx]
        test_c = [str(common_cids[i]) for i in test_idx]

        for cid in train_c:
            rows.append(
                {
                    "cid": cid,
                    "set": "train",
                    "n_fold": n_fold,
                    "fold_idx": fold_idx,
                    "ds": ds,
                }
            )
        for cid in test_c:
            rows.append(
                {
                    "cid": cid,
                    "set": "test",
                    "n_fold": n_fold,
                    "fold_idx": fold_idx,
                    "ds": ds,
                }
            )

    out_path = f"{base_dir}/folds/fold_indices_ds-{ds}_nfold-{n_fold}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print("Saved:", out_path)


def main():
    for ds in ["leffingwell"]:
        for n_fold in [5, 10]:
            save_fold_indices(BASE_DIR, n_fold, ds)


if __name__ == "__main__":
    main()
