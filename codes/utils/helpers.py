import os   
import pandas as pd
import numpy as np
import random
import torch
from .config import SEED, BASE_DIR
def common_cids_per_ds(df):
    """
    Return a sorted list of CIDs that appear in ALL subjects for this dataset (ds).
    Looks under: {BASE_DIR}/embeddings/{ds}/
    """
   

    # Try to find per-subject CSVs
    cid_sets = []
    cid_col ="cid"
    pid_col = "participant_id"
    df = df[[pid_col, cid_col]]
    for _, g in df.groupby("participant_id"):
        cids=set(g["cid"].tolist())
        print(len(cids))
        cid_sets.append(cids)
    

    if not cid_sets:
        return []
    
    filtered = []
    common = set.intersection(*cid_sets)
    for cid in common:
        try:
            filtered.append(int(cid))
        except (ValueError, TypeError):
            continue
    # return as sorted strings (or map to int if you prefer)
    return sorted(filtered, key=lambda x: int(x))


def set_seeds(seed=SEED):
    # Set environment variable for hash-based operations
    os.environ['PYTHONHASHSEED'] = str(seed)

    # Set seed for Python's random module
    random.seed(seed)

    # Set seed for NumPy
    np.random.seed(seed)

    # Set seed for PyTorch
    torch.manual_seed(seed)

    # If using GPUs, set seed for CUDA operations
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups

    # Configure cuDNN for deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
def load_model_embeddings(ds, model_name, cids, layer, embed_type):
    """
    Load embeddings for a given dataset/model, filtered by layer and (optionally) a CID list.

    Args:
        ds (str): Dataset identifier
        model_name (str): Model name used in the embeddings file
        cids (list|None): List of CIDs to keep (order preserved). If None, keep all.
        layer (int): Layer index to select
        embed_type (str): "iso" (isomeric) or "can" (canonical)

    Returns:
        numpy.ndarray: Array of shape (n_rows, n_features) with the selected embeddings,
                       in the same order as `cids` if provided.
    """
    # --- Load CSV as DataFrame ---
    df = pd.read_csv(f"{BASE_DIR}/DATASETS/embeddings/{ds}_{model_name}_embeddings.csv")

    
    embed_type = str(embed_type).lower().strip()
    if embed_type not in {"iso", "can"}:
        raise ValueError("embed_type must be 'iso' or 'can'.")

    # --- Pick the right block of columns ---
    prefix = "iso_e" if embed_type == "iso" else "can_e"
    emb_cols = [c for c in df.columns if c.startswith(prefix)]
    if not emb_cols:
        raise ValueError(f"No embedding columns found with prefix '{prefix}'. "
                         f"Columns present: {list(df.columns)[:12]}...")

    # --- Filter by layer ---
    df = df[df["layer"] == layer].copy()

    # --- Optional: filter and preserve order of CIDs ---
    # Keep only requested CIDs, in the order given by `cids`
    # (drop any CIDs not present in the file)
    df = df.set_index("cid")
    present = [cid for cid in cids if cid in df.index]
    if not present:
        raise ValueError("None of the requested CIDs are present in the embeddings file.")
    df = df.loc[present]
    
    df = df.sort_values(by="cid")

    # --- Extract and return as numpy array ---
    arr = df[emb_cols].to_numpy(dtype=float)
    print("Embeddings shape:", arr.shape, flush=True)
    del df
    return arr

    #reviewed
def load_behavior_embeddings(ds,cids,participant_id, embed_cols, group_by_cid=True):
    """
    Load behavior embeddings with optional filtering.

    Args:
        subject (int|str): Subject identifier (matches values in 'subject' column)
        behavior_embeddings (str|None): Comma-separated embedding column names to select.
                                        If None, selects all numeric columns except ['cid', 'subject'].
        group_by_cid (bool): If True, aggregate duplicate CIDs by mean (within the subject).

    Returns:
        numpy.ndarray: Array of shape (n_rows, n_features) with the selected embeddings.
    """
    # --- Load CSV as DataFrame ---
    df_behavior = pd.read_csv(f"{BASE_DIR}/DATASETS/datasets/{ds}/{ds}_data.csv")
    df_behavior = df_behavior[df_behavior["participant_id"] == participant_id].copy()

    # --- Sort by cid first ---
    df_behavior = df_behavior.sort_values(by="cid").reset_index(drop=True)
    # --- Filter rows for the given subject ---
    print(cids)
    print(df_behavior["cid"].values.tolist())
    if cids is not None:
        df_behavior = df_behavior[df_behavior["cid"].isin(cids)].copy()

    # --- Decide which embedding columns to use ---
    
   

    # --- Optional: group by cid (mean across duplicates within this subject) ---
    if group_by_cid:
        # Keep only the needed columns for aggregation: cid + embeddings
        agg_df = df_behavior[["cid"] + embed_cols].groupby("cid", as_index=False).mean(numeric_only=True)
        # After grouping, drop 'cid' before converting to numpy
        behavior = agg_df[embed_cols].to_numpy()
    else:
        # No grouping: just take embeddings in current row order
        behavior = df_behavior[embed_cols].to_numpy()

    print("Behavior shape:", behavior.shape, flush=True)
    return behavior


def load_fold_cids(n_fold,i_fold, ds):
    """
    Load train and test CIDs for a specific fold from pre-created fold indices.
    
    Args:
        BASE_DIR: Base directory path
        out_dir: Output directory name
        i_fold: Fold index
        ds: Dataset suffix (empty string for main dataset)
        
    Returns:
        tuple: (train_cids, test_cids) as numpy arrays
    """
    fold_file = f"{BASE_DIR}/DATASETS/folds/fold_indices_ds-{ds}_nfold-{n_fold}.csv"
    
   
    
    fold_df = pd.read_csv(fold_file)

    fold_df = fold_df[fold_df['fold_idx']==i_fold]
    train_cids = fold_df[fold_df['set']=='train'][ "cid"].astype(int).tolist()
    test_cids = fold_df[fold_df['set']=='test'][ "cid"].astype(int).tolist()
    
    return train_cids, test_cids
