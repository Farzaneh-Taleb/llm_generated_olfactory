import os   
import pandas as pd
def common_cids_per_ds(BASE_DIR, ds):
    """
    Return a sorted list of CIDs that appear in ALL subjects for this dataset (ds).
    Looks under: {BASE_DIR}/embeddings/{ds}/
    """
    emb_dir = os.path.join(BASE_DIR, "datasets", ds)

    # Try to find per-subject CSVs
    cid_sets = []

  
    combined = os.path.join(emb_dir, f"{ds}_data.csv")
    
    df = pd.read_csv(combined)
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
