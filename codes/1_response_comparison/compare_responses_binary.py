from __future__ import annotations
import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# --- Project utilities ---
from utils.config import BASE_DIR
from utils.ds_utils import get_descriptors


# ------------------------------
# Helpers
# ------------------------------
def _coerce_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _standardize_target_names(name: str) -> str:
    rename_map = {"intensive": "intensity", "pleasant": "pleasantness", "familiar": "familiarity"}
    return rename_map.get(str(name).strip().lower(), str(name))

def _to_binary(s: pd.Series) -> pd.Series:
    """Coerce human labels to {0,1}. Treat >0 as positive."""
    s = pd.to_numeric(s, errors="coerce")
    return (s > 0).astype(int)

def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """
    ROC-AUC with safety:
      - requires >=2 samples and both classes present in y_true
      - returns np.nan if not computable
    """
    y_true = pd.to_numeric(y_true, errors="coerce")
    y_score = pd.to_numeric(y_score, errors="coerce")
    mask = y_true.notna() & y_score.notna()
    y_true, y_score = y_true[mask], y_score[mask]
    if len(y_true) < 2:
        return np.nan
    uniq = pd.unique(y_true)
    if len(uniq) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_score,multi_class='ovr', average='micro'))
    except Exception:
        return np.nan


def _standardize_descriptor_columns(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    rename_map = {
        "intensive": "intensity",
        "pleasant": "pleasantness",
        "familiar": "familiarity",
    }
    cols = {c: rename_map.get(c, c) for c in df.columns}
    return df.rename(columns=cols)


# ------------------------------
# Filename metadata parsers
# ------------------------------
LLM_RE = re.compile(
    r"(?P<ds>[^_]+)_odor_llm_scores_temp-(?P<temp>[^_]+)_model-(?P<model>[^_]+)_bpt-(?P<bpt>[^_]+)(?:_reps-(?P<reps>\d+))?",
)

TR_RE = re.compile(
    r"(?P<ds>[^_]+)_odor_regression_scores_model-(?P<model>[^_]+)_layer-(?P<layer>[^_]+)_nfold-(?P<nfold>[^_]+)_ncomp-(?P<ncomp>[^_]+)_z-(?P<z>[^_]+)",
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

def parse_tr_meta(path: str) -> Dict[str, str]:
    m = TR_RE.search(Path(path).stem)
    if not m:
        return {}
    d = m.groupdict()
    return {
        "ds": d.get("ds"),
        "model_name": d.get("model"),
        "layer": d.get("layer"),
        "n_fold": d.get("nfold"),
        "n_components": d.get("ncomp"),
        "z_score": d.get("z"),
    }


# ------------------------------
# Loading / Preprocessing
# ------------------------------
def load_human(ds: str) -> pd.DataFrame:
    human_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
    df = pd.read_csv(human_csv)
    df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
    df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")

    if "concentration" in df.columns and 'keller' in ds.lower():
        df["concentration"] = df["concentration"].astype(float)
        df = df[df["concentration"] == 0.001].copy()

    df = df.fillna(0)

    if "participant_id" in df.columns and "cid" in df.columns:
        desc_cols = get_descriptors(ds)
        grp = df.groupby(["cid"], dropna=False)[desc_cols]
        avg = grp.mean().reset_index()
        avg["participant_id"] = -1
        df = pd.concat([df, avg], ignore_index=True)

    return df


def load_llm_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    meta = parse_llm_meta(path)
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
    if "cid" in df.columns:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
    if "participant_id" in df.columns:
        df = df.drop(columns=["participant_id"])
    return _standardize_descriptor_columns(df, source="llm")

def load_transformer_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    meta = parse_tr_meta(path)
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
    if "cid" in df.columns:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")
    return _standardize_descriptor_columns(df, source="transformer")


# ------------------------------
# Averaging utilities
# ------------------------------
def average_llm_repeats(llm: pd.DataFrame, descriptors: List[str]) -> pd.DataFrame:
    key = ["cid", "model_name", "build_prompt_type"]
    if "temperature" in llm.columns:
        key.append("temperature")
    if "repeat" in llm.columns:
        return llm.groupby(key, dropna=False)[descriptors].mean().reset_index()
    return llm[key + descriptors].drop_duplicates()

def aggregate_llm_over_temperature(llm_avg: pd.DataFrame, descriptors: List[str]) -> pd.DataFrame:
    if "temperature" not in llm_avg.columns:
        return llm_avg.copy()
    key = ["cid", "model_name", "build_prompt_type"]
    return llm_avg.groupby(key, dropna=False)[descriptors].mean().reset_index()

def average_transformer_repeats(tr: pd.DataFrame, descriptors: List[str], *, by_layer: bool) -> pd.DataFrame:
    if tr is None or tr.empty:
        return tr
    base_keys = ["participant_id", "cid", "model_name"]
    if by_layer and "layer" in tr.columns:
        base_keys.append("layer")
    if "repeat" in tr.columns:
        return tr.groupby(base_keys, dropna=False)[descriptors].mean().reset_index()
    return tr[base_keys + descriptors].drop_duplicates()


# ------------------------------
# Compute AUC
# ------------------------------
def compute_llm_auc(human: pd.DataFrame, llm_avg: pd.DataFrame, descriptors: List[str]):
    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna().copy()
    rows_ppd = []
    key = ["model_name", "build_prompt_type"] + (["temperature"] if "temperature" in llm_avg.columns else [])

    for key_vals, df_llm_group in llm_avg.groupby(key, dropna=False):
        for pid, df_h_p in human_min.groupby("participant_id"):
            merged = df_h_p.merge(df_llm_group, on="cid", suffixes=("_h", "_p"))
            if merged.empty:
                continue
            for d in descriptors:
                y_true = _to_binary(merged[f"{d}_h"])
                y_score = merged[f"{d}_p"]
                auc = _safe_auc(y_true, y_score)
                rec = {"participant_id": pid, "descriptor": d, "auc": auc}
                if isinstance(key_vals, tuple):
                    rec.update(dict(zip(key, key_vals)))
                else:
                    rec.update({key[0]: key_vals})
                rows_ppd.append(rec)

    per_participant_descriptor = pd.DataFrame(rows_ppd)
    if per_participant_descriptor.empty:
        return per_participant_descriptor, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    grp_cols = [c for c in ["model_name", "build_prompt_type", "temperature"] if c in per_participant_descriptor.columns]
    per_participant_avg = per_participant_descriptor.groupby(grp_cols + ["participant_id"])["auc"].mean().reset_index(name="auc_mean")
    grand_average = per_participant_avg.groupby(grp_cols)["auc_mean"].mean().reset_index(name="auc_over_participants")
    per_descriptor_avg = per_participant_descriptor.groupby(grp_cols + ["descriptor"])["auc"].mean().reset_index(name="auc_over_participants")

    return per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg


def compute_transformer_auc(human: pd.DataFrame, tr_avg: pd.DataFrame, descriptors: List[str], *, by_layer: bool):
    if tr_avg is None or tr_avg.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna().copy()
    rows_ppd = []
    group_cols = ["model_name"] + (["layer"] if by_layer and "layer" in tr_avg.columns else [])

    for key_vals, df_tr_grp in tr_avg.groupby(group_cols, dropna=False):
        for pid, df_h_p in human_min.groupby("participant_id"):
            df_tr_p = df_tr_grp[df_tr_grp["participant_id"] == pid]
            merged = df_h_p.merge(df_tr_p, on=["participant_id", "cid"], suffixes=("_h", "_p"))
            if merged.empty:
                continue
            for d in descriptors:
                y_true = _to_binary(merged[f"{d}_h"])
                y_score = merged[f"{d}_p"]
                auc = _safe_auc(y_true, y_score)
                rec = {"participant_id": pid, "descriptor": d, "auc": auc}
                if isinstance(key_vals, tuple):
                    rec.update(dict(zip(group_cols, key_vals)))
                else:
                    rec.update({group_cols[0]: key_vals})
                rows_ppd.append(rec)

    per_participant_descriptor = pd.DataFrame(rows_ppd)
    if per_participant_descriptor.empty:
        return per_participant_descriptor, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    grp_cols = group_cols
    per_participant_avg = per_participant_descriptor.groupby(grp_cols + ["participant_id"])["auc"].mean().reset_index(name="auc_mean")
    grand_average = per_participant_avg.groupby(grp_cols)["auc_mean"].mean().reset_index(name="auc_over_participants")
    per_descriptor_avg = per_participant_descriptor.groupby(grp_cols + ["descriptor"])["auc"].mean().reset_index(name="auc_over_participants")

    return per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg


# ------------------------------
# CLI
# ------------------------------
def main():
    p = argparse.ArgumentParser(description="Compute ROC-AUC for LLM & Transformer vs Human")
    p.add_argument("--ds", required=True)
    args = p.parse_args()

    out_dir = Path(f"{BASE_DIR}/correlation_reports/{args.ds}")
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptors = list(get_descriptors(args.ds))
    human = load_human(args.ds)
    human = human[human["participant_id"] == -1]

    # LLM
    llm_frames = [load_llm_file(fp) for fp in glob.glob(f"{BASE_DIR}/llm_responses/{args.ds}_odor_llm_scores_*.csv")]
    llm = pd.concat(llm_frames, ignore_index=True) if llm_frames else None
    print(f"Loaded LLM data: {llm.shape[0]} rows from {len(llm_frames)} files.")
    if llm is not None and not llm.empty:
        llm_avg = average_llm_repeats(llm, descriptors)
        print(f"After averaging repeats: {llm_avg.shape[0]} rows.")
        llm_ppd, llm_pp, llm_grand, llm_pdesc = compute_llm_auc(human, llm_avg, descriptors)
        llm_ppd.to_csv(out_dir / "llm_auc_per_participant_descriptor.csv", index=False)
        llm_pp.to_csv(out_dir / "llm_auc_per_participant_avg.csv", index=False)
        llm_grand.to_csv(out_dir / "llm_auc_grand_average.csv", index=False)
        llm_pdesc.to_csv(out_dir / "llm_auc_per_descriptor_avg.csv", index=False)

    # Transformer
    # tr_frames = [load_transformer_file(fp) for fp in glob.glob(f"{BASE_DIR}/Oct34_transformer_responses/{args.ds}_odor_regression_scores_model-*.csv")]
    # tr = pd.concat(tr_frames, ignore_index=True) if tr_frames else None
    # if tr is not None and not tr.empty:
    #     tr = tr[tr["participant_id"] == -1]
    #     tr_by_layer = average_transformer_repeats(tr, descriptors, by_layer=True)
    #     L_ppd, L_pp, L_grand, L_pdesc = compute_transformer_auc(human, tr_by_layer, descriptors, by_layer=True)
    #     L_ppd.to_csv(out_dir / "tr_auc_perlayer_per_participant_descriptor.csv", index=False)
    #     L_pp.to_csv(out_dir / "tr_auc_perlayer_per_participant_avg.csv", index=False)
    #     L_grand.to_csv(out_dir / "tr_auc_perlayer_grand_average.csv", index=False)
    #     L_pdesc.to_csv(out_dir / "tr_auc_perlayer_per_descriptor_avg.csv", index=False)


if __name__ == "__main__":
    main()
