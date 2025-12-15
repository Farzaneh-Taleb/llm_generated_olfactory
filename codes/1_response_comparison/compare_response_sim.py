from __future__ import annotations
import argparse, glob, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

# --- Project utilities ---
from utils.config import BASE_DIR

# ------------------------------
# Filename metadata parsers (reuse style)
# ------------------------------
LLM_RE = re.compile(
    r"(?P<ds>[^_]+)_odor_llm_scores_temp-(?P<temp>[^_]+)_model-(?P<model>[^_]+)_bpt-(?P<bpt>[^_]+)(?:_reps-(?P<reps>\d+))?"
)

def parse_llm_meta(path: str) -> Dict[str, str]:
    m = LLM_RE.search(Path(path).stem)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "ds": d.get("ds"),
        "model_name": d.get("model"),
        "build_prompt_type": d.get("bpt"),
        "temperature": d.get("temp"),
        "reps": d.get("reps"),
    }

# ------------------------------
# Pair helpers
# ------------------------------
PAIR_KEYS = ("cid stimulus 1","cid stimulus 2")

def _to_int(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")

def canonicalize_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """Make unordered pairs canonical: (min, max)."""
    df = df.copy()
    a = _to_int(df[PAIR_KEYS[0]])
    b = _to_int(df[PAIR_KEYS[1]])
    df["cid_a"] = np.minimum(a, b)
    df["cid_b"] = np.maximum(a, b)
    df = df.drop(columns=list(PAIR_KEYS), errors="ignore")
    return df

def merge_on_pairs(left: pd.DataFrame, right: pd.DataFrame, suffixes=("_h","_p")) -> pd.DataFrame:
    lk = ["cid_a","cid_b"]
    return left.merge(right, on=lk, suffixes=suffixes, how="inner")

def safe_pearson(x: pd.Series, y: pd.Series) -> Tuple[float,float,int]:
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    m = x.notna() & y.notna()
    x, y = x[m], y[m]
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return (np.nan, np.nan, int(len(x)))
    r, p = pearsonr(x, y)
    return (float(r), float(p), int(len(x)))

# ------------------------------
# Loaders
# ------------------------------
def load_human_pairs(ds: str) -> pd.DataFrame:
    fp = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"
    print(f"[HUMAN] Loading {fp}")  
    df = pd.read_csv(fp)
    df = canonicalize_pairs(df)
    df = df.rename(columns={"similarity":"similarity_h"})
    return df[["cid_a","cid_b","similarity_h"]].dropna()

def load_llm_pairs_files(ds: str) -> pd.DataFrame:
    """Concatenate all LLM pair files for the dataset; attach parsed meta."""
    glob_pat = f"{BASE_DIR}/results/responses/llm_responses/{ds}_odor_llm_scores_*.csv"
    frames = []
    for fp in glob.glob(glob_pat):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[LLM] Skip {fp}: {e}")
            continue
        meta = parse_llm_meta(fp)
        for k, v in meta.items():
            if k not in df.columns:
                df[k] = v
        # canonicalize
        df = df.rename(columns={
            "cid_stimulus_1": "cid stimulus 1",
            "cid_stimulus_2": "cid stimulus 2",
        })
        df = canonicalize_pairs(df)
        if "similarity" not in df.columns:
            print(f"[LLM] {fp} missing 'similarity'; skipping.")
            continue
        keep = ["cid_a","cid_b","similarity","model_name","build_prompt_type","temperature","repeat"]
        keep = [c for c in keep if c in df.columns]
        frames.append(df[keep].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

def load_transformer_pairs_files(ds: str) -> pd.DataFrame:
    """
    Accepts files like:
      snitz2013_data_<MODEL>_cosine_from_embs.csv
    Each must contain: cid stimulus 1, cid stimulus 2, similarity
    """
    glob_pat = f"{BASE_DIR}/Oct34_transformer_responses/{ds}_data_*_cosine_from_embs_prefer_can.csv"
    frames = []
    for fp in glob.glob(glob_pat):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[TR] Skip {fp}: {e}")
            continue
        # infer model name from stem: e.g., snitz2013_data_SELFormer_cosine_from_embs
        stem = Path(fp).stem
        model_name = stem.replace(f"{ds}_data_", "").replace("_cosine_from_embs","")
        df = canonicalize_pairs(df)
        if "similarity" not in df.columns:
            print(f"[TR] {fp} missing 'similarity'; skipping.")
            continue
        df = df.rename(columns={"similarity":"similarity_tr"})
        df["model_name"] = model_name
        frames.append(df[["cid_a","cid_b","similarity_tr","model_name"]].copy())
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)

# ------------------------------
# Aggregation (LLM repeats / temperature)
# ------------------------------
def average_llm_pairs_repeats(llm: pd.DataFrame) -> pd.DataFrame:
    """
    Average LLM repeats per (cid_a,cid_b,model_name,build_prompt_type[,temperature]).
    """
    if llm is None or llm.empty:
        return llm
    key = ["cid_a","cid_b","model_name","build_prompt_type"]
    if "temperature" in llm.columns:
        key.append("temperature")
    if "repeat" in llm.columns:
        return llm.groupby(key, dropna=False)["similarity"].mean().reset_index()
    return llm[key + ["similarity"]].drop_duplicates()

def aggregate_llm_pairs_over_temperature(llm_avg: pd.DataFrame) -> pd.DataFrame:
    """Average over temperature (if present) to get one score per (cid_a,cid_b,model,bpt)."""
    if llm_avg is None or llm_avg.empty:
        return llm_avg
    if "temperature" not in llm_avg.columns:
        return llm_avg
    key = ["cid_a","cid_b","model_name","build_prompt_type"]
    return llm_avg.groupby(key, dropna=False)["similarity"].mean().reset_index()

# ------------------------------
# Alignment computations (Pearson r)
# ------------------------------
def compute_llm_pairs_alignment(human: pd.DataFrame, llm_agg: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if llm_agg is None or llm_agg.empty:
        return pd.DataFrame()
    # compute per (model, bpt[, temp]) first
    key_full = ["model_name","build_prompt_type","temperature"] if "temperature" in llm_agg.columns else ["model_name","build_prompt_type"]
    for keys, g in llm_agg.groupby(key_full, dropna=False):
        merged = merge_on_pairs(human, g.rename(columns={"similarity":"similarity_p"}))
        if merged.empty:
            continue
        r, p, n = safe_pearson(merged["similarity_h"], merged["similarity_p"])
        rec = {"model_name": keys[0], "build_prompt_type": keys[1], "r": r, "p": p, "n_pairs": n}
        if len(keys) == 3:
            rec["temperature"] = keys[2]
        rows.append(rec)
    return pd.DataFrame(rows)

def compute_tr_pairs_alignment(human: pd.DataFrame, tr_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if tr_df is None or tr_df.empty:
        return pd.DataFrame()
    for model_name, g in tr_df.groupby("model_name", dropna=False):
        merged = merge_on_pairs(human, g)
        if merged.empty:
            continue
        r, p, n = safe_pearson(merged["similarity_h"], merged["similarity_tr"])
        rows.append({"model_name": model_name, "r": r, "p": p, "n_pairs": n})
    return pd.DataFrame(rows)

# ------------------------------
# CLI
# ------------------------------
def main():
    parser = argparse.ArgumentParser(description="Snitz2013 pairwise alignment (human vs LLM/Transformer similarities).")
    parser.add_argument("--ds", default="snitz2013", help="Dataset (default: snitz2013).")
    args = parser.parse_args()

    ds = args.ds
    out_dir = Path(f"{BASE_DIR}/alignment_score/{ds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load human
    human = load_human_pairs(ds)
    if human is None or human.empty:
        print("[HUMAN] No human pairs found; abort.")
        return

    # LLMs
    llm_raw = load_llm_pairs_files(ds)
    if llm_raw is not None and not llm_raw.empty:
        llm_avg = average_llm_pairs_repeats(llm_raw)
        # Save alignment at each temperature (if any)
        llm_temp_align = compute_llm_pairs_alignment(human, llm_avg)
        if not llm_temp_align.empty:
            llm_temp_align.to_csv(out_dir / "llm_pairs_alignment_by_temperature.csv", index=False)
        # Aggregate over temperature for a single number per model & bpt
        llm_agg = aggregate_llm_pairs_over_temperature(llm_avg)
        llm_align = compute_llm_pairs_alignment(human, llm_agg)
        if not llm_align.empty:
            llm_align.to_csv(out_dir / "llm_pairs_alignment.csv", index=False)
            print(f"[LLM] Saved: {out_dir/'llm_pairs_alignment.csv'}")

    else:
        print("[LLM] No LLM pair files found.")

    # Transformers
    tr = load_transformer_pairs_files(ds)
    if tr is not None and not tr.empty:
        tr_align = compute_tr_pairs_alignment(human, tr)
        if not tr_align.empty:
            tr_align.to_csv(out_dir / "tr_pairs_alignment.csv", index=False)
            print(f"[TR] Saved: {out_dir/'tr_pairs_alignment.csv'}")
    else:
        print("[TR] No Transformer pair files found.")

if __name__ == "__main__":
    main()
