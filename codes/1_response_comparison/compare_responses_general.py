from __future__ import annotations
import argparse, glob, re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# --- Project utilities ---
from utils.config import BASE_DIR
from utils.ds_utils import get_descriptors

# ------------------------------
# Custom palette (edit these later as you like)
# Keys are series labels produced by _series_label_key()
#   LLMs:  "<MODEL_NAME> byname" or "<MODEL_NAME> bysmiles"
#   TRs:   "<MODEL_NAME>"
# ------------------------------
CUSTOM_PALETTE: Dict[str, str] = {
    # LLMs (examples)
    "GPT byname": "#1f77b4",
    "GPT bysmiles": "#1f77b4",
    "Gemini byname": "#ff7f0e",
    "Gemini bysmiles": "#ff7f0e",
    # Transformers (examples)
    "MoLFormer": "#2ca02c",
    "ChemBERT": "#d62728",
    "ChemBERTa": "#9467bd",
}

# ------------------------------
# Helpers (shared)
# ------------------------------
def _coerce_int(x) -> Optional[int]:
    try:
        return int(x)
    except Exception:
        return None

def _standardize_descriptor_columns(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Idempotent rename of common descriptor aliases."""
    if df is None or df.empty:
        return df
    rename_map = {
        "intensive": "intensity",
        "pleasant": "pleasantness",
        "familiar": "familiarity",
    }
    return df.rename(columns={c: rename_map.get(c, c) for c in df.columns})

def _to_binary(s: pd.Series) -> pd.Series:
    """Coerce human labels to {0,1}. Treat > 0 as positive."""
    s = pd.to_numeric(s, errors="coerce")
    return (s > 0).astype(int)

def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson r with safety: requires >= 2 points and non-constant variance."""
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 2 or x.nunique() < 2 or y.nunique() < 2:
        return np.nan
    try:
        r, _ = pearsonr(x, y)
        return float(r)
    except Exception:
        return np.nan

def _safe_auc(y_true: pd.Series, y_score: pd.Series) -> float:
    """
    ROC-AUC with safety:
      - requires >= 2 samples and both classes present in y_true
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
        return float(roc_auc_score(y_true, y_score))
    except Exception:
        return np.nan

def metric_for_dataset(ds: str):
    """Return (metric_column_name, scorer_fn, needs_binarize_human: bool)."""
    d = ds.lower()
    if "leffingwell" in d:
        return ("auc", _safe_auc, True)
    if "keller" in d or "sagar" in d:
        return ("corr", _safe_corr, False)
    return ("corr", _safe_corr, False)

# ------------------------------
# Filename metadata parsers
# ------------------------------
LLM_RE = re.compile(
    r"(?P<ds>[^_]+)_odor_llm_scores_temp-(?P<temp>[^_]+)_model-(?P<model>[^_]+)_bpt-(?P<bpt>[^_]+)(?:_reps-(?P<reps>\d+))?"
)
TR_RE = re.compile(
    r"(?P<ds>[^_]+)_odor_regression_scores_model-(?P<model>[^_]+)_layer-(?P<layer>[^_]+)_nfold-(?P<nfold>[^_]+)_ncomp-(?P<ncomp>[^_]+)_z-(?P<z>[^_]+)"
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
    if "cid" in df.columns:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")
    if "concentration" in df.columns and "keller" in ds.lower():
        df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")
        df = df[df["concentration"] == 0.001].copy()
    df = df.fillna(0)

    # Append participant_id = -1 grand-average across descriptors per CID
    if "participant_id" in df.columns and "cid" in df.columns:
        desc_cols = list(get_descriptors(ds))
        grp = df.groupby(["cid"], dropna=False)[desc_cols]
        avg = grp.mean().reset_index()
        avg["participant_id"] = -1
        df = pd.concat([df, avg], ignore_index=True)

    return _standardize_descriptor_columns(df, source="human")

def load_llm_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    meta = parse_llm_meta(path)
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
    if "cid" in df.columns:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
    if "participant_id" in df.columns:  # meaningless for LLM by spec
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

def load_transformer_metrics_files(
    ds: str,
    descriptors: List[str],
    metrics_glob: Optional[str],
) -> pd.DataFrame:
    """Load precomputed transformer correlations (metrics CSVs) into tidy form."""
    metrics_glob = metrics_glob or f"{BASE_DIR}/Oct34_llmprj/metrics_model-*_ds-{ds}*.csv"
    frames = []
    for fp in glob.glob(metrics_glob):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[TR-METRICS] Skipping {fp}: {e}")
            continue

        required = {
            "correlation", "model", "ds", "participant_id", "layer",
            "n_fold", "n_components", "z_score", "target", "target_id"
        }
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[TR-METRICS] {fp} missing columns {missing}, skipping.")
            continue

        df = df[df["ds"] == ds].copy()
        df = df.rename(columns={"model": "model_name"})
        df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")

        targ = df["target"].astype(str).str.strip()
        descriptor_set = set(descriptors)
        targ_is_name = targ.isin(descriptor_set)

        df["target_id"] = pd.to_numeric(df["target_id"], errors="coerce")
        def _map_from_id(tid):
            tid = _coerce_int(tid)
            if tid is None or tid < 0 or tid >= len(descriptors):
                return None
            return descriptors[tid]

        df["descriptor"] = np.where(targ_is_name, targ, df["target_id"].map(_map_from_id))
        df = df.dropna(subset=["descriptor", "correlation"])

        keep = [
            "participant_id", "model_name", "layer", "n_fold", "n_components",
            "z_score", "descriptor", "correlation"
        ]
        frames.append(df[keep].copy())

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    if "layer" in out.columns:
        out["layer"] = out["layer"].astype(str)
    return out

# ------------------------------
# Averaging utilities
# ------------------------------
def average_llm_repeats(llm: pd.DataFrame, descriptors: List[str]) -> pd.DataFrame:
    """Average LLM descriptor scores across repeats for each (cid, model_name, build_prompt_type[, temperature])."""
    key = ["cid", "model_name", "build_prompt_type"]
    if "temperature" in llm.columns:
        key.append("temperature")
    if "repeat" in llm.columns:
        return llm.groupby(key, dropna=False)[descriptors].mean().reset_index()
    return llm[key + descriptors].drop_duplicates()

def aggregate_llm_over_temperature(llm_avg: pd.DataFrame, descriptors: List[str]) -> pd.DataFrame:
    """Aggregate LLMs over temperature → (cid, model_name, build_prompt_type)."""
    if "temperature" not in llm_avg.columns:
        return llm_avg.copy()
    key = ["cid", "model_name", "build_prompt_type"]
    return llm_avg.groupby(key, dropna=False)[descriptors].mean().reset_index()

def average_transformer_repeats(tr: pd.DataFrame, descriptors: List[str], *, by_layer: bool) -> pd.DataFrame:
    """Average transformer scores across repeats."""
    if tr is None or tr.empty:
        return tr
    base = ["participant_id", "cid", "model_name"]
    if by_layer and "layer" in tr.columns:
        base.append("layer")
    if "repeat" in tr.columns:
        return tr.groupby(base, dropna=False)[descriptors].mean().reset_index()
    return tr[base + descriptors].drop_duplicates()

# ------------------------------
# Unified alignment computation
# ------------------------------
def compute_llm_alignment(
    ds: str,
    human: pd.DataFrame,
    llm_avg: pd.DataFrame,
    descriptors: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Compute alignment for LLMs using metric determined by dataset."""
    metric_name, scorer, needs_bin = metric_for_dataset(ds)
    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna(subset=["participant_id", "cid"]).copy()

    rows = []
    key = ["model_name", "build_prompt_type"] + (["temperature"] if "temperature" in llm_avg.columns else [])

    for key_vals, df_llm_group in llm_avg.groupby(key, dropna=False):
        for pid, df_h_p in human_min.groupby("participant_id"):
            merged = df_h_p.merge(df_llm_group, on="cid", suffixes=("_h", "_p"))
            if merged.empty:
                continue
            for d in descriptors:
                y_true = _to_binary(merged[f"{d}_h"]) if needs_bin else merged[f"{d}_h"]
                y_pred = merged[f"{d}_p"]
                val = scorer(y_true, y_pred)
                rec = {"participant_id": pid, "descriptor": d, metric_name: val}
                if isinstance(key_vals, tuple):
                    rec.update(dict(zip(key, key_vals)))
                else:
                    rec.update({key[0]: key_vals})
                rows.append(rec)

    per_participant_descriptor = pd.DataFrame(rows)
    if per_participant_descriptor.empty:
        return per_participant_descriptor, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metric_name

    grp_cols = [c for c in ["model_name", "build_prompt_type", "temperature"] if c in per_participant_descriptor.columns]
    per_participant_avg = (
        per_participant_descriptor.groupby(grp_cols + ["participant_id"], dropna=False)[metric_name]
        .mean().reset_index(name=f"{metric_name}_mean")
    )
    grand_average = (
        per_participant_avg.groupby(grp_cols, dropna=False)[f"{metric_name}_mean"]
        .mean().reset_index(name=f"{metric_name}_over_participants")
    )
    per_descriptor_avg = (
        per_participant_descriptor.groupby(grp_cols + ["descriptor"], dropna=False)[metric_name]
        .mean().reset_index(name=f"{metric_name}_over_participants")
    )
    return per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg, metric_name

def compute_transformer_alignment(
    ds: str,
    human: pd.DataFrame,
    tr_avg: pd.DataFrame,
    descriptors: List[str],
    *,
    by_layer: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    """Compute alignment for Transformers using metric determined by dataset."""
    metric_name, scorer, needs_bin = metric_for_dataset(ds)
    if tr_avg is None or tr_avg.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metric_name

    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna(subset=["participant_id", "cid"]).copy()

    rows = []
    group_cols = ["model_name"] + (["layer"] if by_layer and "layer" in tr_avg.columns else [])

    for key_vals, df_tr_grp in tr_avg.groupby(group_cols, dropna=False):
        for pid, df_h_p in human_min.groupby("participant_id"):
            df_tr_p = df_tr_grp[df_tr_grp["participant_id"] == pid]
            merged = df_h_p.merge(df_tr_p, on=["participant_id", "cid"], suffixes=("_h", "_p"))
            if merged.empty:
                continue
            for d in descriptors:
                y_true = _to_binary(merged[f"{d}_h"]) if needs_bin else merged[f"{d}_h"]
                y_pred = merged[f"{d}_p"]
                val = scorer(y_true, y_pred)
                rec = {"participant_id": pid, "descriptor": d, metric_name: val}
                if isinstance(key_vals, tuple):
                    rec.update(dict(zip(group_cols, key_vals)))
                else:
                    rec.update({group_cols[0]: key_vals})
                rows.append(rec)

    per_participant_descriptor = pd.DataFrame(rows)
    if per_participant_descriptor.empty:
        return per_participant_descriptor, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), metric_name

    grp_cols = group_cols
    per_participant_avg = (
        per_participant_descriptor.groupby(grp_cols + ["participant_id"], dropna=False)[metric_name]
        .mean().reset_index(name=f"{metric_name}_mean")
    )
    grand_average = (
        per_participant_avg.groupby(grp_cols, dropna=False)[f"{metric_name}_mean"]
        .mean().reset_index(name=f"{metric_name}_over_participants")
    )
    per_descriptor_avg = (
        per_participant_descriptor.groupby(grp_cols + ["descriptor"], dropna=False)[metric_name]
        .mean().reset_index(name=f"{metric_name}_over_participants")
    )
    return per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg, metric_name

# ------------------------------
# ROC curve helpers (macro/micro/descriptor)
# ------------------------------
def _stack_binary_targets_and_scores(merged: pd.DataFrame, descriptors: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Stack y_true (binary) and y_score across all descriptors for micro-averaged ROC."""
    y_trues, y_scores = [], []
    for d in descriptors:
        y_t = _to_binary(pd.to_numeric(merged[f"{d}_h"], errors="coerce"))
        y_s = pd.to_numeric(merged[f"{d}_p"], errors="coerce")
        mask = y_t.notna() & y_s.notna()
        y_t, y_s = y_t[mask], y_s[mask]
        if len(y_t) == 0:
            continue
        y_trues.append(y_t.to_numpy())
        y_scores.append(y_s.to_numpy())
    if not y_trues:
        return np.array([]), np.array([])
    return np.concatenate(y_trues), np.concatenate(y_scores)

def _per_descriptor_curves(merged: pd.DataFrame, descriptors: list[str]) -> list[dict]:
    """For each descriptor, compute ROC (fpr, tpr) and AUC if both classes present."""
    curves = []
    for d in descriptors:
        y_true = _to_binary(pd.to_numeric(merged[f"{d}_h"], errors="coerce"))
        y_score = pd.to_numeric(merged[f"{d}_p"], errors="coerce")
        mask = y_true.notna() & y_score.notna()
        y_true, y_score = y_true[mask].to_numpy(), y_score[mask].to_numpy()
        if y_true.size < 2 or len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_score)
        curves.append({"descriptor": d, "fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))})
    return curves

def _macro_curve_from_descriptor_curves(curves: list[dict], *, grid_size: int = 101) -> tuple[np.ndarray, np.ndarray, float, int]:
    """
    Macro ROC:
      - Interpolate each descriptor's TPR to a shared FPR grid
      - Average TPR across descriptors
      - Macro AUC = mean of per-descriptor AUCs
    """
    if not curves:
        return np.array([]), np.array([]), np.nan, 0

    fpr_grid = np.linspace(0.0, 1.0, grid_size)
    tprs = []
    aucs = []
    for c in curves:
        fpr, tpr = c["fpr"], c["tpr"]
        order = np.argsort(fpr)
        fpr_sorted, tpr_sorted = fpr[order], tpr[order]
        uniq_fpr, idx = np.unique(fpr_sorted, return_index=True)
        tpr_dedup = np.maximum.reduceat(tpr_sorted, idx)
        tpr_interp = np.interp(fpr_grid, uniq_fpr, tpr_dedup)
        tpr_interp[0] = 0.0
        tpr_interp[-1] = 1.0
        tprs.append(tpr_interp)
        aucs.append(float(c["auc"]))
    if not tprs:
        return np.array([]), np.array([]), np.nan, 0
    tpr_macro = np.mean(np.vstack(tprs), axis=0)
    auc_macro = float(np.nanmean(aucs)) if len(aucs) else np.nan
    return fpr_grid, tpr_macro, auc_macro, len(tprs)

def compute_llm_roc_curves(
    ds: str,
    human: pd.DataFrame,
    llm_avg: pd.DataFrame,
    descriptors: List[str],
    *,
    mode: str = "macro",             # "macro", "micro", or "descriptor"
    descriptor: Optional[str] = None # required if mode == "descriptor"
) -> pd.DataFrame:
    """ROC curves for each (model_name, build_prompt_type[, temperature]) LLM group."""
    if "leffingwell" not in ds.lower():
        print("[ROC] Only supported for binary datasets (e.g., Leffingwell).")
        return pd.DataFrame()
    if mode == "descriptor" and not descriptor:
        raise ValueError("When roc-mode='descriptor', you must pass --roc-descriptor.")

    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna(subset=["participant_id", "cid"]).copy()

    key_cols = ["model_name", "build_prompt_type"] + (["temperature"] if "temperature" in llm_avg.columns else [])
    rows = []

    for key_vals, df_llm_group in llm_avg.groupby(key_cols, dropna=False):
        merged = human_min.merge(df_llm_group, on="cid", suffixes=("_h", "_p"))
        if merged.empty:
            continue

        if mode == "macro":
            desc_curves = _per_descriptor_curves(merged, descriptors)
            fpr_grid, tpr_macro, auc_macro, n_used = _macro_curve_from_descriptor_curves(desc_curves)
            if n_used == 0:
                continue
            rec = {"fpr": fpr_grid, "tpr": tpr_macro, "auc": auc_macro, "n_descriptors": n_used}
        elif mode == "micro":
            y_true, y_score = _stack_binary_targets_and_scores(merged, descriptors)
            if y_true.size < 2 or len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            rec = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr))}
        else:  # descriptor
            dname = descriptor
            y_true = _to_binary(pd.to_numeric(merged[f"{dname}_h"], errors="coerce"))
            y_score = pd.to_numeric(merged[f"{dname}_p"], errors="coerce")
            mask = y_true.notna() & y_score.notna()
            y_true, y_score = y_true[mask].to_numpy(), y_score[mask].to_numpy()
            if y_true.size < 2 or len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            rec = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)), "descriptor": dname}

        if isinstance(key_vals, tuple):
            rec.update(dict(zip(key_cols, key_vals)))
        else:
            rec.update({key_cols[0]: key_vals})

        rows.append(rec)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).explode(["fpr", "tpr"], ignore_index=True)

def compute_transformer_roc_curves(
    ds: str,
    human: pd.DataFrame,
    tr_avg: pd.DataFrame,
    descriptors: List[str],
    *,
    mode: str = "macro",             # "macro", "micro", or "descriptor"
    descriptor: Optional[str] = None # required if mode == "descriptor"
) -> pd.DataFrame:
    """ROC curves for Transformers per model_name (aggregated across layers beforehand)."""
    if "leffingwell" not in ds.lower():
        print("[ROC] Only supported for binary datasets (e.g., Leffingwell).")
        return pd.DataFrame()
    if tr_avg is None or tr_avg.empty:
        return pd.DataFrame()
    if mode == "descriptor" and not descriptor:
        raise ValueError("When roc-mode='descriptor', you must pass --roc-descriptor.")

    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna(subset=["participant_id", "cid"]).copy()

    rows = []
    for model_name, df_tr_group in tr_avg.groupby(["model_name"], dropna=False):
        merged = human_min.merge(df_tr_group, on=["cid"], suffixes=("_h", "_p"))
        if merged.empty:
            continue

        if mode == "macro":
            desc_curves = _per_descriptor_curves(merged, descriptors)
            fpr_grid, tpr_macro, auc_macro, n_used = _macro_curve_from_descriptor_curves(desc_curves)
            if n_used == 0:
                continue
            rec = {"fpr": fpr_grid, "tpr": tpr_macro, "auc": auc_macro, "model_name": model_name, "n_descriptors": n_used}
        elif mode == "micro":
            y_true, y_score = _stack_binary_targets_and_scores(merged, descriptors)
            if y_true.size < 2 or len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            rec = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)), "model_name": model_name}
        else:  # descriptor
            dname = descriptor
            y_true = _to_binary(pd.to_numeric(merged[f"{dname}_h"], errors="coerce"))
            y_score = pd.to_numeric(merged[f"{dname}_p"], errors="coerce")
            mask = y_true.notna() & y_score.notna()
            y_true, y_score = y_true[mask].to_numpy(), y_score[mask].to_numpy()
            if y_true.size < 2 or len(np.unique(y_true)) < 2:
                continue
            fpr, tpr, _ = roc_curve(y_true, y_score)
            rec = {"fpr": fpr, "tpr": tpr, "auc": float(auc(fpr, tpr)), "model_name": model_name, "descriptor": dname}

        rows.append(rec)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).explode(["fpr", "tpr"], ignore_index=True)

# ------------------------------
# Plot helpers (bars)
# ------------------------------
def _metric_over_participants_col(df: pd.DataFrame) -> str:
    cols = [c for c in df.columns if c.endswith("_over_participants")]
    if not cols:
        raise ValueError("No '*_over_participants' column found in dataframe.")
    return cols[0]

def summarize_for_plot(ds: str, a_pdesc: pd.DataFrame) -> pd.DataFrame:
    metric_col = _metric_over_participants_col(a_pdesc)
    grp = (
        a_pdesc
        .groupby(["model_name", "build_prompt_type"], dropna=False)[metric_col]
        .mean()
        .reset_index()
        .rename(columns={metric_col: "value"})
    )
    grp.insert(0, "dataset", ds)
    return grp

def plot_llm_across_datasets(
    summary_df: pd.DataFrame,
    out_path: Path,
    *,
    models: Optional[List[str]] = None,
    bpt_order: Tuple[str, str] = ("byname", "bysmiles"),
    bar_width: float = 0.08,
    fig_w: float = 11,
    fig_h: float = 5.5,
    colors: Optional[Tuple[str, str]] = ("#4d79a4", "#ecc947"),
    annotate: bool = True,
    decimals: int = 3,
):
    if summary_df is None or summary_df.empty:
        print("[PLOT] No data to plot.")
        return

    datasets = list(summary_df["dataset"].unique())
    model_list = sorted(summary_df["model_name"].unique().tolist())
    if models:
        model_list = [m for m in models if m in model_list]
    if len(model_list) < 2:
        print("[PLOT] Fewer than 2 models found; plotting available models.")
    model_list = model_list[:2]

    x = np.arange(len(datasets))
    total_series = max(1, len(model_list) * len(bpt_order))
    offsets = [((k - (total_series - 1) / 2) * bar_width) for k in range(total_series)]

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.grid(axis="y", linestyle=":", linewidth=0.9, alpha=0.55)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    if colors is None or len(colors) < 2:
        colors = ("#4d79a4", "#ecc947")

    legend_handles, legend_labels = [], []
    series_idx = 0

    for m_i, model in enumerate(model_list):
        color = colors[m_i % len(colors)]
        for b_i, bpt in enumerate(bpt_order):
            pos = x + offsets[series_idx]
            series_idx += 1

            ys, es = [], []
            for ds in datasets:
                sub = summary_df[
                    (summary_df["dataset"] == ds) &
                    (summary_df["model_name"] == model) &
                    (summary_df["build_prompt_type"] == bpt)
                ]
                if not sub.empty:
                    ys.append(float(sub["mean"].mean()))
                    es.append(float(sub["sem"].mean()))
                else:
                    ys.append(np.nan)
                    es.append(np.nan)

            bars = ax.bar(
                pos, ys, width=bar_width, edgecolor="black", linewidth=0.8,
                color=color, alpha=0.95, label=None,
            )

            if b_i == 1:
                for r in bars:
                    r.set_hatch("//")

            ax.errorbar(
                pos, ys, yerr=es, fmt="none", ecolor="black",
                elinewidth=0.9, capsize=3, capthick=0.9,
            )

            if annotate:
                for rect, y in zip(bars, ys):
                    if np.isnan(y):
                        continue
                    ax.annotate(
                        f"{y:.{decimals}f}",
                        xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", va="bottom", fontsize=9,
                    )

            if len(bars):
                legend_handles.append(bars[0])
                legend_labels.append(f"{model} ({bpt})")

    ax.set_xlabel("Dataset", labelpad=8)
    ax.set_ylabel("Alignment (mean ± SEM over descriptors)", labelpad=8)
    ax.set_title("LLM alignment by dataset (colors = models; hatch = prompt type)", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, ha="right")

    finite_means = summary_df["mean"].to_numpy()
    finite_means = finite_means[np.isfinite(finite_means)]
    if finite_means.size > 0:
        vmin, vmax = float(np.nanmin(finite_means)), float(np.nanmax(finite_means))
        if vmin >= 0.0 and vmax <= 1.0:
            ax.set_ylim(0.0, 1.0)
        else:
            margin = 0.05 * (vmax - vmin if vmax > vmin else 1.0)
            ax.set_ylim(vmin - margin, vmax + margin)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    seen, uniq_h, uniq_l = set(), [], []
    for h, l in zip(legend_handles, legend_labels):
        if l not in seen:
            uniq_h.append(h); uniq_l.append(l); seen.add(l)
    ax.legend(
        uniq_h, uniq_l, frameon=False, ncol=2, fontsize=9,
        loc="upper left", bbox_to_anchor=(0.0, 1.02), borderaxespad=0.0, handlelength=1.4, columnspacing=1.2
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path.with_suffix('.png')} and .pdf")

# ------------------------------
# Rounding / CSV helpers
# ------------------------------
DECIMALS = 2

def round_to_2dec(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        df[num_cols] = df[num_cols].round(DECIMALS)
    return df

def save_csv_2dec(df: pd.DataFrame, path: Path) -> None:
    df_rounded = round_to_2dec(df)
    df_rounded.to_csv(path, index=False, float_format=f"%.{DECIMALS}f")

def mean_sem_over_descriptors(
    df: pd.DataFrame,
    *,
    group_cols: List[str],
    descriptor_col: str = "descriptor",
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    metric_col = _metric_over_participants_col(df)
    grp = df.groupby(group_cols, dropna=False)[metric_col]
    out = grp.agg(['mean', 'std', 'count']).reset_index()
    out.rename(columns={'count': 'n'}, inplace=True)
    out['sem'] = out['std'] / np.sqrt(out['n'].clip(lower=1))
    out = out[group_cols + ['mean', 'sem', 'n']]
    num_cols = out.select_dtypes(include=[np.number]).columns
    if len(num_cols):
        out[num_cols] = out[num_cols].round(DECIMALS)
    return out

# ------------------------------
# ROC plotting (split: LLMs vs Transformers)
# ------------------------------
def _series_label_key(row: pd.Series) -> str:
    """Palette key. LLMs include prompt type; Transformers only model_name."""
    if "build_prompt_type" in row and pd.notna(row["build_prompt_type"]):
        return f"{row['model_name']} {row['build_prompt_type']}"
    return f"{row['model_name']}"

def plot_roc_llms(
    curves_df_llm: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    palette: Optional[Dict[str, str]] = None,
    annotate_auc: bool = True,
):
    """Plot ROC for LLMs only. Dashed=byname, Solid=bysmiles."""
    if curves_df_llm is None or curves_df_llm.empty:
        print("[ROC-LLM] No ROC data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.5)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    label_cols = [c for c in ["model_name", "build_prompt_type", "temperature", "descriptor"] if c in curves_df_llm.columns]
    series = curves_df_llm[label_cols].drop_duplicates().to_dict(orient="records")

    for key_dict in series:
        mask = np.ones(len(curves_df_llm), dtype=bool)
        for k, v in key_dict.items():
            mask &= (curves_df_llm[k] == v)
        sub = curves_df_llm[mask].copy()

        auc_val = sub["auc"].dropna().unique()
        auc_str = f"AUC={auc_val[0]:.3f}" if len(auc_val) else "AUC=NA"
        bpt = sub["build_prompt_type"].iloc[0] if "build_prompt_type" in sub.columns else None
        linestyle = "--" if bpt == "byname" else "-"  # dashed for byname, solid for bysmiles/other

        if "descriptor" in sub.columns and sub["descriptor"].notna().any():
            label = f"{sub['model_name'].iloc[0]} {bpt} [{sub['descriptor'].iloc[0]}]"
        else:
            label = f"{sub['model_name'].iloc[0]} {bpt}"
        if annotate_auc:
            label = f"{label} ({auc_str})"

        key_row = sub.iloc[0]
        pal_key = _series_label_key(key_row)
        color = (palette or {}).get(pal_key, None)

        ax.plot(sub["fpr"].astype(float).to_numpy(),
                sub["tpr"].astype(float).to_numpy(),
                label=label,
                linestyle=linestyle,
                color=color)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ROC-LLM] Saved: {out_path.with_suffix('.png')} and .pdf")

def plot_roc_transformers(
    curves_df_tr: pd.DataFrame,
    out_path: Path,
    *,
    title: str,
    palette: Optional[Dict[str, str]] = None,
    annotate_auc: bool = True,
):
    """Plot ROC for Transformers only. Lines are solid."""
    if curves_df_tr is None or curves_df_tr.empty:
        print("[ROC-TR] No ROC data to plot.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, linestyle=":", linewidth=0.9, alpha=0.5)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    label_cols = [c for c in ["model_name", "descriptor"] if c in curves_df_tr.columns]
    series = curves_df_tr[label_cols].drop_duplicates().to_dict(orient="records")

    for key_dict in series:
        mask = np.ones(len(curves_df_tr), dtype=bool)
        for k, v in key_dict.items():
            mask &= (curves_df_tr[k] == v)
        sub = curves_df_tr[mask].copy()

        auc_val = sub["auc"].dropna().unique()
        auc_str = f"AUC={auc_val[0]:.3f}" if len(auc_val) else "AUC=NA"

        if "descriptor" in sub.columns and sub["descriptor"].notna().any():
            label = f"{sub['model_name'].iloc[0]} [{sub['descriptor'].iloc[0]}]"
        else:
            label = f"{sub['model_name'].iloc[0]}"
        if annotate_auc:
            label = f"{label} ({auc_str})"

        key_row = sub.iloc[0]
        pal_key = _series_label_key(key_row)  # just model_name for TRs
        color = (palette or {}).get(pal_key, None)

        ax.plot(sub["fpr"].astype(float).to_numpy(),
                sub["tpr"].astype(float).to_numpy(),
                label=label,
                linestyle="-",
                color=color)

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, fontsize=9, loc="lower right")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[ROC-TR] Saved: {out_path.with_suffix('.png')} and .pdf")

# ------------------------------
# CLI / Orchestration
# ------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Alignment & ROC: AUC (Leffingwell) or Pearson r (Keller/Sagar). Creates ROC plots for LLMs (GPTs) and Transformers separately."
    )
    p.add_argument("--ds", required=True,
                   help="Dataset(s). Comma-separated allowed: leffingwell,keller2016,sagar2023")
    p.add_argument("--tr-source", choices=["predictions", "metrics"], default="predictions",
                   help="Read transformer results from raw predictions or precomputed metrics.")
    p.add_argument("--plot-out", default="llm_alignment_across_datasets",
                   help="Output filename stem for the multi-dataset LLM barplot")
    p.add_argument("--models", nargs="*", default=None,
                   help="Optionally restrict to two LLM model names (order matters) for coloring in bar plot.")

    # ROC flags
    p.add_argument("--plot-roc", action="store_true",
                   help="If set, produce ROC curves for binary datasets (e.g., Leffingwell).")
    p.add_argument("--roc-mode", choices=["macro", "micro", "descriptor"], default="macro",
                   help="Macro: average across descriptors; Micro: pooled labels; or a single descriptor.")
    p.add_argument("--roc-descriptor", default=None,
                   help="Descriptor name to plot when --roc-mode=descriptor.")
    p.add_argument("--tr-roc-models", nargs="*", default=None,
                   help="Limit transformer ROC to these model names (substring match). Defaults to first 3 unique models.")

    args = p.parse_args()

    ds_list = [d.strip() for d in args.ds.split(",") if d.strip()]
    all_plot_rows = []

    for ds in ds_list:
        out_dir = Path(f"{BASE_DIR}/alignment_score/{ds}")
        out_dir.mkdir(parents=True, exist_ok=True)

        descriptors = list(get_descriptors(ds))
        human = load_human(ds)
        human = human[human["participant_id"] == -1]

        # ---- LLM ingest ----
        llm_glob = f"{BASE_DIR}/results/responses/llm_responses/{ds}_odor_llm_scores_*.csv"
        llm_frames: List[pd.DataFrame] = []
        for fp in glob.glob(llm_glob):
            try:
                llm_frames.append(load_llm_file(fp))
            except Exception as e:
                print(f"[LLM] Skipping {fp}: {e}")
        llm = pd.concat(llm_frames, ignore_index=True) if llm_frames else None

        llm_agg = None
        if llm is not None and not llm.empty:
            keep_cols = set(["cid", "model_name", "build_prompt_type", "temperature", "repeat"]) | set(descriptors)
            llm = llm[[c for c in llm.columns if c in keep_cols]]
            llm_avg = average_llm_repeats(llm, descriptors)
            _, _, _, llm_pdesc, _ = compute_llm_alignment(ds, human, llm_avg, descriptors)
            llm_agg = aggregate_llm_over_temperature(llm_avg, descriptors)
            _, _, _, a_pdesc, _ = compute_llm_alignment(ds, human, llm_agg, descriptors)

            llm_desc_stats = mean_sem_over_descriptors(
                a_pdesc,
                group_cols=[c for c in ["model_name", "build_prompt_type", "temperature"] if c in a_pdesc.columns]
            )
            save_csv_2dec(llm_desc_stats, out_dir / "llm_mean_sem_over_descriptors.csv")

            if a_pdesc is not None and not a_pdesc.empty:
                all_plot_rows.append(summarize_for_plot(ds, a_pdesc))

        # ---- Transformer ingest (predictions) ----
        tr = None
        tr_agg = None
        if args.tr_source == "predictions":
            tr_glob = f"{BASE_DIR}/Oct34_transformer_responses/{ds}_odor_regression_scores_model-*_layer-*_nfold-*_ncomp-*_z-*.csv"
            tr_frames: List[pd.DataFrame] = []
            for fp in glob.glob(tr_glob):
                try:
                    tr_frames.append(load_transformer_file(fp))
                except Exception as e:
                    print(f"[TR] Skipping {fp}: {e}")
            tr = pd.concat(tr_frames, ignore_index=True) if tr_frames else None

            if tr is not None and not tr.empty:
                keep_cols = set(["participant_id", "cid", "model_name", "layer", "repeat"]) | set(descriptors)
                tr = tr[[c for c in tr.columns if c in keep_cols]]
                tr["participant_id"] = pd.to_numeric(tr["participant_id"], errors="coerce").fillna(-1).astype(int)
                tr = tr[tr["participant_id"] == -1]  # grand-average rows

                # Per-layer (saved for completeness)
                tr_by_layer = average_transformer_repeats(tr, descriptors, by_layer=True)
                L_ppd, L_pp, L_grand, L_pdesc, metric_name = compute_transformer_alignment(ds, human, tr_by_layer, descriptors, by_layer=True)
                save_csv_2dec(L_ppd,   out_dir / f"trPERLAYER_{metric_name}_per_participant_descriptor.csv")
                save_csv_2dec(L_pp,    out_dir / f"trPERLAYER_{metric_name}_per_participant_avg.csv")
                save_csv_2dec(L_grand, out_dir / f"trPERLAYER_{metric_name}_grand_average.csv")
                save_csv_2dec(L_pdesc, out_dir / f"trPERLAYER_{metric_name}_per_descriptor_avg.csv")
                tr_layer_desc_stats = mean_sem_over_descriptors(L_pdesc, group_cols=["model_name", "layer"])
                save_csv_2dec(tr_layer_desc_stats, out_dir / "trPERLAYER_mean_sem_over_descriptors.csv")

                # Aggregated across layers (used for ROC)
                tr_agg = average_transformer_repeats(tr, descriptors, by_layer=False)
                M_ppd, M_pp, M_grand, M_pdesc, metric_name = compute_transformer_alignment(ds, human, tr_agg, descriptors, by_layer=False)
                save_csv_2dec(M_ppd,   out_dir / f"trAGG_{metric_name}_per_participant_descriptor.csv")
                save_csv_2dec(M_pp,    out_dir / f"trAGG_{metric_name}_per_participant_avg.csv")
                save_csv_2dec(M_grand, out_dir / f"trAGG_{metric_name}_grand_average.csv")
                save_csv_2dec(M_pdesc, out_dir / f"trAGG_{metric_name}_per_descriptor_avg.csv")
                tr_desc_stats = mean_sem_over_descriptors(M_pdesc, group_cols=["model_name"])
                tr_desc_stats.to_csv(out_dir / "tr_mean_sem_over_descriptors.csv", index=False)

        else:  # metrics path (kept for completeness; not used for ROC)
            trm = load_transformer_metrics_files(ds, descriptors, metrics_glob=f"{BASE_DIR}/results/responses/llm_responses/metrics_model-*_ds-{ds}*.csv")
            if trm is None or trm.empty:
                print("[TR-METRICS] No metrics rows found.")
            else:
                if "participant_id" in trm.columns:
                    trm["participant_id"] = pd.to_numeric(trm["participant_id"], errors="coerce")
                    trm = trm[trm["participant_id"] == -1]
                # Save per-layer & aggregated corr summaries (as in previous versions)...
                # (Omitted here for brevity; ROC requires raw predictions.)

        # ---------- ROC plots (separate: LLMs vs Transformers) ----------
        if args.plot_roc and ("leffingwell" in ds.lower()):
            suffix = (
                "macro" if args.roc_mode == "macro"
                else "micro" if args.roc_mode == "micro"
                else f"{args.roc_descriptor}"
            )

            # LLM ROC (own plot)
            if llm_agg is not None and not llm_agg.empty:
                curves_llm = compute_llm_roc_curves(
                    ds, human, llm_agg, descriptors,
                    mode=args.roc_mode, descriptor=args.roc_descriptor
                )
                if curves_llm is not None and not curves_llm.empty:
                    curves_llm.to_csv(out_dir / f"roc_curves_llm_{suffix}.csv", index=False)
                    title_llm = f"ROC (LLMs) — {ds} [{suffix}]  (dashed=byname, solid=bysmiles)"
                    plot_roc_llms(curves_llm, out_dir / f"roc_plot_llm_{suffix}", title=title_llm, palette=CUSTOM_PALETTE)

            # Transformer ROC (own plot, only from predictions)
            if tr_agg is not None and not tr_agg.empty:
                # Filter to 3 models or user-provided subset
                tr_sel = tr_agg
                if args.tr_roc_models:
                    keep_models = []
                    for m in tr_sel["model_name"].dropna().unique():
                        if any(s.lower() in str(m).lower() for s in args.tr_roc_models):
                            keep_models.append(m)
                    tr_sel = tr_sel[tr_sel["model_name"].isin(keep_models)]
                else:
                    uniq = tr_sel["model_name"].dropna().unique().tolist()
                    tr_sel = tr_sel[tr_sel["model_name"].isin(uniq[:3])]

                curves_tr = compute_transformer_roc_curves(
                    ds, human, tr_sel, descriptors,
                    mode=args.roc_mode, descriptor=args.roc_descriptor
                )
                if curves_tr is not None and not curves_tr.empty:
                    curves_tr.to_csv(out_dir / f"roc_curves_tr_{suffix}.csv", index=False)
                    title_tr = f"ROC (Transformers) — {ds} [{suffix}]"
                    plot_roc_transformers(curves_tr, out_dir / f"roc_plot_tr_{suffix}", title=title_tr, palette=CUSTOM_PALETTE)

        # ---------- Multi-dataset LLM barplot ----------
        if all_plot_rows:
            summary_df = pd.concat(all_plot_rows, ignore_index=True)
            plot_dir = Path(f"{BASE_DIR}/alignment_score/_plots")
            plot_llm_across_datasets(summary_df, plot_dir / args.plot_out, models=args.models)
        else:
            print("[PLOT] No LLM data collected across datasets; skipping plot.")

if __name__ == "__main__":
    main()
