import os   
import pandas as pd
import numpy as np
import random
from .config import *
from typing import List, Dict, Any, Optional, Tuple, Union
import time
import json 
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
    #convert participant_id to int in daaframe
    df_behavior['participant_id'] = df_behavior['participant_id'].astype(int)
    df_behavior["cid"] = pd.to_numeric(df_behavior["cid"], errors="coerce")


    if 'concentration' in df_behavior.columns:
        df_behavior['concentration'] = df_behavior['concentration'].astype(float)
        df_behavior = df_behavior[df_behavior['concentration']==0.001].copy()
    df_behavior = df_behavior[df_behavior["participant_id"] == participant_id].copy()
    

    print(f"Total rows for participant {participant_id}: {len(df_behavior)}", flush=True)

    # --- Sort by cid first ---
    df_behavior = df_behavior.sort_values(by="cid").reset_index(drop=True)
    # --- Filter rows for the given subject ---
    print(cids)
    print(df_behavior["cid"].values.tolist())
    if cids is not None:
        df_behavior = df_behavior[df_behavior["cid"].isin(cids)].copy()

    # --- Decide which embedding columns to use ---
    
    print(f"Total rows for participant cid {participant_id}: {len(df_behavior)}", flush=True)

    # --- Optional: group by cid (mean across duplicates within this subject) ---
    if group_by_cid:
        # Keep only the needed columns for aggregation: cid + embeddings
        agg_df = df_behavior[["cid"] + embed_cols].groupby("cid", as_index=False).mean(numeric_only=True)
        # After grouping, drop 'cid' before converting to numpy
        behavior = agg_df[embed_cols].fillna(0).to_numpy()
        print(f"Total rows for participant cid {participant_id}, after mean: {len(df_behavior)}", flush=True)

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



def collect_predictions_rows(
    preds_list,                        # List[np.ndarray] — one array per fold, in the same order as Xs_test
    test_cids_list,                      # List[List[int]] — one list per fold, in the same order as Xs_test
    descriptors,                         # List[str] == embed_cols
    participant_id: int,
    model_name: str,
    ds: str,
    layer: int,
    n_fold: int,
    n_components: int | None,
    z_score: bool,
    run_id: str,
    # input_type_col: str = "isomericsmiles",   # to mirror your GPT CSV columns
    # smiles_lookup: dict[int, str] | None = None,  # optional {cid -> smiles}
    # name_lookup: dict[int, str] | None = None,    # optional {cid -> odor name}
) -> pd.DataFrame:
    """
    Returns a dataframe with columns mirroring your GPT batch output:
    [participant_id, cid, repeat, temperature, isomericsmiles, name, model_name, build_prompt_type, ...descriptors]
    """
    rows = []
    
    # Run the same pipeline used in compute_correlation
    
    # Build rows for each test sample
    for i, cid in enumerate(test_cids_list):
        base = {
            "participant_id": participant_id,
            "cid": int(cid) if cid is not None else None,
            "repeat": 0,                         # to mirror GPT format (no repeats here)
            # "temperature": "",                   # N/A for regression
            # input_type_col: smiles_lookup.get(int(cid), "") if (smiles_lookup and cid is not None) else "",
            # "name": name_lookup.get(int(cid), "") if (name_lookup and cid is not None) else "",
            "model_name": model_name,
            "build_prompt_type": "regression",   # tag to distinguish from "bysmiles"/"byname"
            "ds": ds,
            "layer": layer,
            "n_fold": n_fold,
            "n_components": n_components if n_components is not None else "",
            "z_score": bool(z_score)
           
        }
        # Attach descriptor predictions
        for j, d in enumerate(descriptors):
            base[d] = float(preds_list[i, j])
        rows.append(base)

    # Order columns similar to your GPT CSVs (then add metadata)
    front_cols = [
        "participant_id", "cid", "repeat", "model_name", "build_prompt_type"
    ]
    meta_cols = ["ds", "layer", "n_fold", "n_components", "z_score"]
    cols = front_cols + descriptors + meta_cols

    df = pd.DataFrame(rows)
    # Only keep columns that exist
    df = df.reindex(columns=[c for c in cols if c in df.columns])
    return df


def get_descriptors(ds):
    if ds =='bierling2025':
        return ['intensity','pleasantness','familiar','edible', 'warm','sour', 'cold','sweet','fruit','spices','bakery','garlic', 'fish', 
                    'burnt', 'decayed', 'grass', 'wood', 'chemical','flower', 'musky', 'sweaty', 'ammonia']
    elif ds == 'keller2016':
        return['intensity', 'pleasantness','familiarity','edible','bakery','sweet','fruit','fish','garlic','spices','cold','sour',
               'burnt','acid','warm','musky','sweaty','ammonia','decayed','wood','grass','flower','chemical']
    elif ds== 'sagar2023_v1':
        pass
    elif ds == 'sagar2023_v2':
        pass
    elif ds == 'sagar2023':
        return [ 'intensity', 'pleasantness', 'fishy', 'burnt', 'sour', 'decayed', 'musky',
    'fruity', 'sweaty', 'cool', 'floral', 'sweet', 'warm', 'bakery', 'spicy']
    elif ds == 'leffingwell':
        return [
                "alcoholic","aldehydic","alliaceous","almond","animal","anisic","apple","apricot","aromatic","balsamic",
                "banana","beefy","berry","black currant","brandy","bread","brothy","burnt","buttery","cabbage","camphoreous",
                "caramellic","catty","chamomile","cheesy","cherry","chicken","chocolate","cinnamon","citrus","cocoa","coconut",
                "coffee","cognac","coumarinic","creamy","cucumber","dairy","dry","earthy","ethereal","fatty","fermented","fishy",
                "floral","fresh","fruity","garlic","gasoline","grape","grapefruit","grassy","green","hay","hazelnut","herbal",
                "honey","horseradish","jasmine","ketonic","leafy","leathery","lemon","malty","meaty","medicinal","melon","metallic",
                "milky","mint","mushroom","musk","musty","nutty","odorless","oily","onion","orange","orris","peach","pear",
                "phenolic","pine","pineapple","plum","popcorn","potato","pungent","radish","ripe","roasted","rose","rum","savory",
                "sharp","smoky","solvent","sour","spicy","strawberry","sulfurous","sweet","tea","tobacco","tomato","tropical",
                "vanilla","vegetable","violet","warm","waxy","winey","woody"]
    else:
        raise ValueError("Unsupported dataset: {}".format(ds))
    
    

def get_rate_range(ds: str) -> tuple[float, float]:
    if ds not in RATE_RANGE:
        raise ValueError(f"Unknown dataset: {ds}. Please add it to RATE_RANGE.")
    return RATE_RANGE[ds]

# -------- Registry I/O (unchanged) --------
def log_batch_entry(
    ds: str,
    model_name: str,
    temp: float,
    batch_id: str,
    build_prompt_type: str,
    n_repeats: int,
):
    os.makedirs(os.path.dirname(BATCH_REGISTRY), exist_ok=True)
    entry = {
        "ds": ds,
        "model_name": model_name,
        "temperature": temp,
        "batch_id": batch_id,  # e.g., 'batches/123...'
        "build_prompt_type": build_prompt_type,
        "n_repeats": n_repeats,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(BATCH_REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_registry() -> List[Dict[str, Any]]:
    if not os.path.exists(BATCH_REGISTRY):
        return []
    with open(BATCH_REGISTRY, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def _clean_cell(x) -> str:
    if x is None:
        return ""
    try:
        import pandas as _pd
        if _pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x).strip()




def pair_input_col(which: int, input_type: str) -> str:
    assert which in (1, 2)
    return f"{input_type} stimulus {which}"

def is_pairwise_df(df: pd.DataFrame) -> bool:
    """
    Detect pairwise format by presence of either:
      - CID columns for both stimuli, or
      - Name columns for both stimuli, or
      - Two input columns for the configured INPUT_TYPE,
    and ideally a 'similarity' column.
    """
    has_pair_ids = (PAIR_CID1 in df.columns and PAIR_CID2 in df.columns)
    has_pair_names = (PAIR_NAME1 in df.columns and PAIR_NAME2 in df.columns)
    has_pair_inputs = (
        pair_input_col(1, INPUT_TYPE) in df.columns and
        pair_input_col(2, INPUT_TYPE) in df.columns
    )
    has_sim = PAIR_SIMILARITY in df.columns
    # be permissive: if we have two inputs of either type, treat as pairwise
    return (has_pair_inputs or has_pair_names or has_pair_ids) and has_sim

def row_smiles(row: pd.Series) -> Optional[str]:
    smi = _clean_cell(row.get(INPUT_TYPE, ""))
    return smi or None

def row_name(row: pd.Series) -> Optional[str]:
    nm = _clean_cell(row.get("name", ""))
    return nm or None

def row_pair_inputs(row: pd.Series, build_prompt_type: str) -> Optional[Tuple[str, str]]:
    """
    Return (input1, input2) depending on build_prompt_type.
    - bysmiles -> use f"{INPUT_TYPE} stimulus 1/2"
    - byname   -> use "name stimulus 1/2"
    """
    if build_prompt_type == "bysmiles":
        col1 = pair_input_col(1, INPUT_TYPE)
        col2 = pair_input_col(2, INPUT_TYPE)
    else:
        col1 = PAIR_NAME1
        col2 = PAIR_NAME2
    a = _clean_cell(row.get(col1, ""))
    b = _clean_cell(row.get(col2, ""))
    if not a or not b:
        return None
    return a, b


def row_pair_cids(row: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    c1 = pd.to_numeric(row.get(PAIR_CID1), errors="coerce")
    c2 = pd.to_numeric(row.get(PAIR_CID2), errors="coerce")
    c1 = int(c1) if pd.notna(c1) else None
    c2 = int(c2) if pd.notna(c2) else None
    return c1, c2


# def build_prompt_bysmiles_single(
#     smiles: str,
#     descriptors: List[str],
#     rate_min: float,
#     rate_max: float,
#     include_confidence: bool = False,
# ) -> str:
#     desc_list = ", ".join([f'"{d}"' for d in descriptors])
#     json_lines = [f'  "{d}": <{rate_min}-{rate_max}>' for d in descriptors]
#     if include_confidence:
#         json_lines.append('  "confidence": <0-1>')
#     json_block = "{\n" + ",\n".join(json_lines) + "\n}"

#     return f"""Molecule:
# - ISOMERIC SMILES: {smiles}

# Descriptors (rate each from {rate_min} to {rate_max}): [{desc_list}]

# Output rules:
# - Return ONLY a single valid JSON object. No prose, no markdown.
# - Keys must match the descriptor list exactly.
# - Values must be numbers in [{rate_min},{rate_max}].
# - Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

# Output format:
# {json_block}
# """


def build_prompt_bysmiles_single(
    smiles: str,
    descriptors: List[str],
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
) -> str:
    desc_list = ", ".join([f'"{d}"' for d in descriptors])
    json_lines = [f'  "{d}": <{rate_min}-{rate_max}>' for d in descriptors]
    if include_confidence:
        json_lines.append('  "confidence": <0-1>')
    json_block = "{\n" + ",\n".join(json_lines) + "\n}"

    return f"""Instructions:
    You will be presented with an odor stimulus.
    Please rate the smell of the stimulus on each perceptual dimension listed below.
- Stimulus: {smiles}

Perceptual dimensions (rate each from {rate_min} to {rate_max}): [{desc_list}]

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must match the descriptor list exactly.
- Values must be numbers in [{rate_min},{rate_max}].
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""

# ---------- Prompt builders (single-item) ----------
# def build_prompt_byname_single(
#     name: str,
#     descriptors: List[str],
#     rate_min: float,
#     rate_max: float,
#     include_confidence: bool = False,
# ) -> str:
#     desc_list = ", ".join([f'"{d}"' for d in descriptors])
#     json_lines = [f'  "{d}": <{rate_min}-{rate_max}>' for d in descriptors]
#     if include_confidence:
#         json_lines.append('  "confidence": <0-1>')
#     json_block = "{\n" + ",\n".join(json_lines) + "\n}"

#     return f"""Molecule:
# - Name: {name}

# Descriptors (rate each from {rate_min} to {rate_max}): [{desc_list}]

# Output rules:
# - Return ONLY a single valid JSON object. No prose, no markdown.
# - Keys must match the descriptor list exactly.
# - Values must be numbers in [{rate_min},{rate_max}].
# - Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

# Output format:
# {json_block}
# """


def build_prompt_byname_single(
    name: str,
    descriptors: List[str],
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
) -> str:
    desc_list = ", ".join([f'"{d}"' for d in descriptors])
    json_lines = [f'  "{d}": <{rate_min}-{rate_max}>' for d in descriptors]
    if include_confidence:
        json_lines.append('  "confidence": <0-1>')
    json_block = "{\n" + ",\n".join(json_lines) + "\n}"

    return f"""Instructions:
    You will be presented with an odor stimulus.
    Please rate the smell of the stimulus on each perceptual dimension listed below.
- Stimulus: {name}

Perceptual dimensions (rate each from {rate_min} to {rate_max}): [{desc_list}]

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must match the descriptor list exactly.
- Values must be numbers in [{rate_min},{rate_max}].
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""


# def build_prompt_bysmiles_pairwise(
#     smiles_1: str,
#     smiles_2: str,
#     rate_min: float,
#     rate_max: float,
#     include_confidence: bool = False,
# ) -> str:
#     json_lines = [f'  "similarity": <{rate_min}-{rate_max}>']
#     if include_confidence:
#         json_lines.append('  "confidence": <0-1>')
#     json_block = "{\n" + ",\n".join(json_lines) + "\n}"
#     return f"""Two Molecules:
# - ISOMERIC SMILES (Stimulus 1): {smiles_1}
# - ISOMERIC SMILES (Stimulus 2): {smiles_2}

# Similarity (rate from {rate_min} to {rate_max}):
# - Provide a single continuous similarity rating; higher means more similar.

# Output rules:
# - Return ONLY a single valid JSON object. No prose, no markdown.
# - Keys must be exactly: "similarity".
# - Values must be numbers in [{rate_min},{rate_max}]{' (and confidence in [0,1])' if include_confidence else ''}.
# - Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

# Output format:
# {json_block}
# """

# def build_prompt_byname_pairwise(
#     name_1: str,
#     name_2: str,
#     rate_min: float,
#     rate_max: float,
#     include_confidence: bool = False,
# ) -> str:
#     """
#     Name-based similarity prompt with the same structure as the single-item name prompt.
#     """
#     json_lines = [f'  "similarity": <{rate_min}-{rate_max}>']
#     if include_confidence:
#         json_lines.append('  "confidence": <0-1>')
#     json_block = "{\n" + ",\n".join(json_lines) + "\n}"

#     return f"""Two Molecules:
# - Name (Stimulus 1): {name_1}
# - Name (Stimulus 2): {name_2}

# Similarity (rate from {rate_min} to {rate_max}):
# - Provide a single continuous similarity rating; higher means more similar.

# Output rules:
# - Return ONLY a single valid JSON object. No prose, no markdown.
# - Keys must be exactly: "similarity".
# - Values must be numbers in [{rate_min},{rate_max}]{' (and confidence in [0,1])' if include_confidence else ''}.
# - Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

# Output format:
# {json_block}
# """


def build_prompt_bysmiles_pairwise(
    smiles_1: str,
    smiles_2: str,
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
) -> str:
    json_lines = [f'  "similarity": <{rate_min}-{rate_max}>']
    if include_confidence:
        json_lines.append('  "confidence": <0-1>')
    json_block = "{\n" + ",\n".join(json_lines) + "\n}"
    return f"""Instructions:
    You will be presented with two odor stimuli.
    Please rate the similarity between the smell of the two stimuli:
- Stimulus 1: {smiles_1}
- Stimulus 2: {smiles_2}

Similarity (rate from {rate_min} to {rate_max}):
- Provide a single continuous similarity rating; higher means more similar.

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must be exactly: "similarity".
- Values must be numbers in [{rate_min},{rate_max}]{' (and confidence in [0,1])' if include_confidence else ''}.
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""

def build_prompt_byname_pairwise(
    name_1: str,
    name_2: str,
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
) -> str:
    """
    Name-based similarity prompt with the same structure as the single-item name prompt.
    """
    json_lines = [f'  "similarity": <{rate_min}-{rate_max}>']
    if include_confidence:
        json_lines.append('  "confidence": <0-1>')
    json_block = "{\n" + ",\n".join(json_lines) + "\n}"

    return f"""Instructions:
    You will be presented with two odor stimuli.
    Please rate the similarity between the smell of the two stimuli:
- Stimulus 1: {name_1}
- Stimulus 2: {name_2}

Similarity (rate from {rate_min} to {rate_max}):
- Provide a single continuous similarity rating; higher means more similar.

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must be exactly: "similarity".
- Values must be numbers in [{rate_min},{rate_max}]{' (and confidence in [0,1])' if include_confidence else ''}.
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""



def build_prompt_bysmiles(
    x: Union[str, Tuple[str, str]],
    descriptors: List[str],
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
    *,
    pairwise: bool = False,
) -> str:
    """
    Unified API compatible with previous code paths.
    - For single-item: x is a SMILES string.
    - For pairwise: x is a tuple (stimulus_1, stimulus_2), SMILES strings.
    """
    if pairwise:
        if not (isinstance(x, tuple) and len(x) == 2):
            raise ValueError("For pairwise=True, pass a tuple (stimulus_1, stimulus_2).")
        a, b = x
        return build_prompt_bysmiles_pairwise(a, b, rate_min, rate_max, include_confidence)
    # single-item
    if isinstance(x, tuple):
        raise ValueError("For single-item prompts, pass a single SMILES string, not a tuple.")
    return build_prompt_bysmiles_single(x, descriptors, rate_min, rate_max, include_confidence)

def build_prompt_byname(
    x: Union[str, Tuple[str, str]],
    descriptors: List[str],
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
    *,
    pairwise: bool = False,
) -> str:
    """
    Unified API compatible with previous code paths.
    - For single-item: x is a name string.
    - For pairwise: x is a tuple (stimulus_1, stimulus_2), names.
    """
    if pairwise:
        if not (isinstance(x, tuple) and len(x) == 2):
            raise ValueError("For pairwise=True, pass a tuple (stimulus_1, stimulus_2).")
        a, b = x
        return build_prompt_byname_pairwise(a, b, rate_min, rate_max, include_confidence)
    # single-item
    if isinstance(x, tuple):
        raise ValueError("For single-item prompts, pass a single name string, not a tuple.")
    return build_prompt_byname_single(x, descriptors, rate_min, rate_max, include_confidence)

def validate_response_single(
    resp_text: str,
    descriptors: List[str],
    rate_min: float,
    rate_max: float,
    include_confidence: bool,
) -> Dict[str, float]:
    obj = json.loads(resp_text)
    out: Dict[str, float] = {}
    for d in descriptors:
        if d not in obj:
            raise ValueError(f"Missing key: {d}")
        try:
            v = float(obj[d])
        except Exception:
            raise ValueError(f"Non-numeric value for '{d}': {obj[d]!r}")
        out[d] = max(rate_min, min(rate_max, v))
    if include_confidence:
        if "confidence" in obj:
            try:
                c = float(obj["confidence"])
            except Exception:
                c = 0.0
            out["confidence"] = max(0.0, min(1.0, c))
        else:
            out["confidence"] = None
    return out


def validate_response_pairwise(
    resp_text: str,
    rate_min: float,
    rate_max: float,
    include_confidence: bool,
) -> Dict[str, float]:
    obj = json.loads(resp_text)
    if "similarity" not in obj:
        raise ValueError('Missing key: "similarity"')
    try:
        sim = float(obj["similarity"])
    except Exception:
        raise ValueError(f"Non-numeric value for 'similarity': {obj.get('similarity')!r}")
    out: Dict[str, float] = {"similarity": max(rate_min, min(rate_max, sim))}
    if include_confidence:
        if "confidence" in obj:
            try:
                c = float(obj["confidence"])
            except Exception:
                c = 0.0
            out["confidence"] = max(0.0, min(1.0, c))
        else:
            out["confidence"] = None
    return out


def build_prompt_dispatch(
    x: Union[str, Tuple[str, str]],
    descriptors: List[str],
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
    *,
    pairwise: bool = False,
    byname: bool = False,
) -> str:
    if pairwise:
        if not (isinstance(x, tuple) and len(x) == 2):
            raise ValueError("For pairwise=True, pass a tuple (stimulus_1, stimulus_2).")
        a, b = x
        return (
            build_prompt_byname_pairwise(a, b, rate_min, rate_max, include_confidence)
            if byname
            else build_prompt_bysmiles_pairwise(a, b, rate_min, rate_max, include_confidence)
        )
    else:
        if isinstance(x, tuple):
            raise ValueError("For single-item prompts, pass a single string (SMILES or name).")
        return (
            build_prompt_byname_single(x, descriptors, rate_min, rate_max, include_confidence)
            if byname
            else build_prompt_bysmiles_single(x, descriptors, rate_min, rate_max, include_confidence)
        )