from __future__ import annotations
import argparse
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

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

# def _standardize_target_names(name: str) -> str:
#     # Keep in sync with _standardize_descriptor_columns mapping
#     rename_map = {"intensive": "intensity", "pleasant": "pleasantness", "familiar": "familiarity"}
#     return rename_map.get(str(name).strip().lower(), str(name))

def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    """Pearson r with safety: requires >=2 points and non-constant variance.
    Returns np.nan if not computable.
    """
    x = pd.to_numeric(x, errors="coerce")
    y = pd.to_numeric(y, errors="coerce")
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    if len(x) < 2:
        return np.nan
    if x.nunique() < 2 or y.nunique() < 2:
        print("Constant series, cannot compute correlation.",x.nunique(), y.nunique())
        return np.nan
    try:
        r, _ = pearsonr(x, y)
        return float(r)
    except Exception:
        return np.nan


def _standardize_descriptor_columns(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Rename descriptor columns to match human naming.
    - LLM source often uses: intensive→intensity, pleasant→pleasantness, familiar→familiarity.
    - Transformer source should already match, but we run the mapping idempotently.
    """
    rename_map = {
        "intensive": "intensity",
        "pleasant": "pleasantness",
        "familiar": "familiarity",
    }
    cols = {c: rename_map.get(c, c) for c in df.columns}
    out = df.rename(columns=cols)
    return out


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
# Oct34_transformer_responses
#keller2016_odor_regression_scores_model-*_layer-*_nfold-*_ncomp-*_z-*.csv" 
def load_human(ds: str) -> pd.DataFrame:
    human_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
    df = pd.read_csv(human_csv)
    # df = _standardize_descriptor_columns(df, source="human")
    df["cid"]=pd.to_numeric(df["cid"], errors="coerce")
    df["participant_id"]=pd.to_numeric(df["participant_id"], errors="coerce")
    print(df.head(5))
    if "cid" in df.columns:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
    if "participant_id" in df.columns:
        df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")
    if "concentration" in df.columns and 'keller' in ds.lower():
        df["concentration"] = df["concentration"].astype(float)
        df = df[df["concentration"] == 0.001].copy()
    #group by participant_id and cid,model_name, average descriptor and add as a new row with participant_id -1
    print(df.head(5))
    #fill na with 0
    df =   df.fillna(0)
    if "participant_id" in df.columns and "cid" in df.columns:
        # desc_cols = _standardize_target_names(get_descriptors(ds))
        desc_cols = get_descriptors(ds)

        print(df.columns)
        print(desc_cols)
        grp = df.groupby(["cid"], dropna=False)[desc_cols]
        avg = grp.mean().reset_index()
        avg["participant_id"] = -1
        df = pd.concat([df, avg], ignore_index=True)
        print("Human data:", len(df), "rows; participants:", df["participant_id"].nunique() if "participant_id" in df.columns else "N/A")
    
    #standardize descriptor columns
    


    return df


def load_llm_file(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    meta = parse_llm_meta(path)
    for k, v in meta.items():
        if k not in df.columns:
            df[k] = v
    if "cid" in df.columns:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
    if "participant_id" in df.columns:  # meaningless for LLM per spec
        df = df.drop(columns=["participant_id"])
    # df = _standardize_descriptor_columns(df, source="llm")
    return df


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
    # df = _standardize_descriptor_columns(df, source="transformer")
    return df

def load_transformer_metrics_files(
    ds: str,
    descriptors: List[str],
    metrics_glob: Optional[str],
) -> pd.DataFrame:
    """
    Read precomputed transformer correlations from metrics CSVs and return a tidy frame:
      ['participant_id','cid'(absent),'model_name','layer','descriptor','corr']
    """
    
    metrics_glob = f"{BASE_DIR}/Oct34_llmprj/metrics_model-*_ds-{ds}*.csv"

    frames = []
    for fp in glob.glob(metrics_glob):
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[TR-METRICS] Skipping {fp}: {e}")
            continue

        # Normalize required columns
        required = {
            "correlation","model","ds","participant_id","layer",
            "n_fold","n_components","z_score","target","target_id"
        }
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"[TR-METRICS] {fp} missing columns {missing}, skipping.")
            continue

        df = df[df["ds"] == ds].copy()

        # Model name normalization
        df = df.rename(columns={"model": "model_name"})

        # Participant id
        df["participant_id"] = pd.to_numeric(df["participant_id"], errors="coerce")

        # Build descriptor column:
        # 1) if target is a string that matches a descriptor (after standardization), use it
        # 2) else if target_id is numeric, map ordering: descriptors[target_id]
        # 3) otherwise drop row
        targ = df["target"].astype(str).str.strip()
        # targ_std = targ.apply(_standardize_target_names)

        # detect if target is numeric index
        targ_is_int = targ_std.apply(lambda s: s.isdigit())
        df["target_id"] = pd.to_numeric(df["target_id"], errors="coerce")

        # use name if it looks like text and is in set
        descriptor_set = set(descriptors)
        use_name_mask = targ_is_int.eq(False) & targ_std.isin(descriptor_set)

        # fallback to index
        def _map_from_id(row):
            tid = _coerce_int(row.get("target_id"))
            if tid is None or tid < 0 or tid >= len(descriptors):
                return None
            return descriptors[tid]

        df["descriptor"] = np.where(
            use_name_mask, targ_std,
            df.apply(_map_from_id, axis=1)
        )

        df = df.dropna(subset=["descriptor", "correlation"])

        # Keep only relevant columns
        keep = [
            "participant_id", "model_name", "layer", "n_fold", "n_components", "z_score",
            "descriptor", "correlation"
        ]
        frames.append(df[keep].copy())

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    # Standardize descriptor names once more (idempotent)
    # out["descriptor"] = out["descriptor"].map(_standardize_target_names)

    # Ensure layer is str (so groupby behaves)
    if "layer" in out.columns:
        out["layer"] = out["layer"].astype(str)

    return out

def aggregate_transformer_metrics_corr(
    metrics_df: pd.DataFrame,
    *,
    by_layer: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    From metrics_df with columns:
      participant_id, model_name, layer?, descriptor, correlation
    produce:
      per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg
    """
    if metrics_df is None or metrics_df.empty:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame())

    group_cols = ["model_name"] + (["layer"] if by_layer and "layer" in metrics_df.columns else [])

    # 1) each (participant, descriptor, group) corr
    per_participant_descriptor = (
        metrics_df
        .groupby(group_cols + ["participant_id", "descriptor"], dropna=False)["correlation"]
        .mean()
        .reset_index(name="corr")
    )

    # 2) participant-level mean over descriptors
    per_participant_avg = (
        per_participant_descriptor
        .groupby(group_cols + ["participant_id"], dropna=False)["corr"]
        .mean()
        .reset_index(name="corr_mean")
    )

    # 3) grand average over participants
    grand_average = (
        per_participant_avg
        .groupby(group_cols, dropna=False)["corr_mean"]
        .mean()
        .reset_index(name="corr_over_participants")
    )

    # 4) per-descriptor average over participants
    per_descriptor_avg = (
        per_participant_descriptor
        .groupby(group_cols + ["descriptor"], dropna=False)["corr"]
        .mean()
        .reset_index(name="corr_over_participants")
    )

    return per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg

# ------------------------------
# Core correlation logic
# ------------------------------

def average_llm_repeats(llm: pd.DataFrame, descriptors: List[str]) -> pd.DataFrame:
    """Average LLM descriptor scores across repeats for each (cid, model_name, build_prompt_type, temperature)."""
    key = ["cid", "model_name", "build_prompt_type"]
    if "temperature" in llm.columns:
        key.append("temperature")
    if "repeat" in llm.columns:
        grp = llm.groupby(key, dropna=False)[descriptors]
        llm_avg = grp.mean().reset_index()
    else:
        llm_avg = llm[key + descriptors].drop_duplicates()
    return llm_avg


def aggregate_llm_over_temperature(llm_avg: pd.DataFrame, descriptors: List[str]) -> pd.DataFrame:
    """Aggregate LLMs over temperature → (cid, model_name, build_prompt_type)."""
    if "temperature" not in llm_avg.columns:
        return llm_avg.copy()
    key = ["cid", "model_name", "build_prompt_type"]
    return llm_avg.groupby(key, dropna=False)[descriptors].mean().reset_index()


def average_transformer_repeats(tr: pd.DataFrame, descriptors: List[str], *, by_layer: bool) -> pd.DataFrame:
    """Average transformer scores across repeats.
    - by_layer=True: average within (participant_id, cid, model_name, layer) → keep layers separate.
    - by_layer=False: average within (participant_id, cid, model_name) → aggregates over layers.
    """
    if tr is None or tr.empty:
        return tr

    base_keys = ["participant_id", "cid", "model_name"]
    if by_layer and "layer" in tr.columns:
        base_keys = base_keys + ["layer"]

    if "repeat" in tr.columns:
        tr_avg = (
            tr.groupby(base_keys, dropna=False)[descriptors]
            .mean()
            .reset_index()
        )
    else:
        tr_avg = tr[base_keys + descriptors].drop_duplicates()

    return tr_avg


def compute_llm_correlations(
    human: pd.DataFrame,
    llm_avg: pd.DataFrame,
    descriptors: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlations for LLMs per (model_name, build_prompt_type[, temperature])."""
    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna(subset=["participant_id", "cid"]).copy()

    rows_ppd = []
    key = ["model_name", "build_prompt_type"] + (["temperature"] if "temperature" in llm_avg.columns else [])

    for key_vals, df_llm_group in llm_avg.groupby(key, dropna=False):
        for pid, df_h_p in human_min.groupby("participant_id"):
            merged = df_h_p.merge(df_llm_group, on="cid", suffixes=("_h", "_p"))
            if merged.empty:
                continue
            for d in descriptors:

                r = _safe_corr(merged[f"{d}_h"], merged[f"{d}_p"])
                print(f"Participant {pid}, descriptor {d}, correlation: {r}")
                rec = {
                    "participant_id": pid,
                    "descriptor": d,
                    "corr": r,
                }
                if isinstance(key_vals, tuple):
                    rec.update(dict(zip(key, key_vals)))
                else:
                    rec.update({key[0]: key_vals})
                rows_ppd.append(rec)

    per_participant_descriptor = pd.DataFrame(rows_ppd)

    if not per_participant_descriptor.empty:
        grp_cols = [c for c in ["model_name", "build_prompt_type", "temperature"] if c in per_participant_descriptor.columns]
        per_participant_avg = (
            per_participant_descriptor.groupby(grp_cols + ["participant_id"], dropna=False)["corr"].mean().reset_index(name="corr_mean")
        )
        grand_average = (
            per_participant_avg.groupby(grp_cols, dropna=False)["corr_mean"].mean().reset_index(name="corr_over_participants")
        )
        per_descriptor_avg = (
            per_participant_descriptor.groupby(grp_cols + ["descriptor"], dropna=False)["corr"].mean().reset_index(name="corr_over_participants")
        )
    else:
        per_participant_avg = pd.DataFrame()
        grand_average = pd.DataFrame()
        per_descriptor_avg = pd.DataFrame()

    return per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg


def compute_transformer_correlations(
    human: pd.DataFrame,
    tr_avg: pd.DataFrame,
    descriptors: List[str],
    *,
    by_layer: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute correlations for Transformers (per-model or per-model×layer)."""
    if tr_avg is None or tr_avg.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    hcols = ["participant_id", "cid"] + descriptors
    human_min = human[hcols].dropna(subset=["participant_id", "cid"]).copy()

    rows_ppd = []
    group_cols = ["model_name"] + (["layer"] if by_layer and "layer" in tr_avg.columns else [])

    for key_vals, df_tr_grp in tr_avg.groupby(group_cols, dropna=False):
        for pid, df_h_p in human_min.groupby("participant_id"):
            df_tr_p = df_tr_grp[df_tr_grp["participant_id"] == pid]
            print(f"Processing participant {pid} with {len(df_h_p)} human rows and {len(df_tr_p)} transformer rows.")
            if df_tr_p.empty:
                continue
            merged = df_h_p.merge(df_tr_p, on=["participant_id", "cid"], suffixes=("_h", "_p"))
            
            if merged.empty:
                continue
            for d in descriptors:
                r = _safe_corr(merged[f"{d}_h"], merged[f"{d}_p"])
                rec = {
                    "participant_id": pid,
                    "descriptor": d,
                    "corr": r,
                }
                if isinstance(key_vals, tuple):
                    rec.update(dict(zip(group_cols, key_vals)))
                else:
                    rec.update({group_cols[0]: key_vals})
                rows_ppd.append(rec)

    per_participant_descriptor = pd.DataFrame(rows_ppd)

    if not per_participant_descriptor.empty:
        grp_cols = group_cols
        per_participant_avg = (
            per_participant_descriptor.groupby(grp_cols + ["participant_id"], dropna=False)["corr"].mean().reset_index(name="corr_mean")
        )
        grand_average = (
            per_participant_avg.groupby(grp_cols, dropna=False)["corr_mean"].mean().reset_index(name="corr_over_participants")
        )
        per_descriptor_avg = (
            per_participant_descriptor.groupby(grp_cols + ["descriptor"], dropna=False)["corr"].mean().reset_index(name="corr_over_participants")
        )
    else:
        per_participant_avg = pd.DataFrame()
        grand_average = pd.DataFrame()
        per_descriptor_avg = pd.DataFrame()

    return per_participant_descriptor, per_participant_avg, grand_average, per_descriptor_avg


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_llm_descriptor_bars(
    per_desc_df: pd.DataFrame,
    descriptors: list[str],
    out_path: Path,
    *,
    bpt_order: tuple[str, str] = ("byname", "bysmiles"),
    fig_w: float = 14,
    fig_h: float = 6,
    bar_width: float = 0.08,
):
    """
    per_desc_df columns expected: ['model_name','build_prompt_type','descriptor','corr_over_participants']
    Only LLM rows should be passed (your a_pdesc).
    """
    if per_desc_df is None or per_desc_df.empty:
        print("[PLOT] No LLM per-descriptor rows to plot.")
        return

    # Keep only requested prompt types if present
    if "build_prompt_type" in per_desc_df.columns:
        per_desc_df = per_desc_df[per_desc_df["build_prompt_type"].isin(bpt_order)].copy()

    # Ensure descriptor names standardized and ordered
    # per_desc_df["descriptor"] = per_desc_df["descriptor"].map(_standardize_target_names)
    # Pivot convenience (we'll still loop to keep style control)
    models = sorted(per_desc_df["model_name"].unique().tolist())

    # X locations
    x = np.arange(len(descriptors))

    # Per descriptor, we'll place bars for each model × bpt
    total_series = len(models) * len(bpt_order)
    group_span = bar_width * total_series
    offsets = []
    for m_i in range(len(models)):
        for b_i in range(len(bpt_order)):
            k = m_i * len(bpt_order) + b_i
            offsets.append((k - (total_series - 1) / 2) * bar_width)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    legend_handles = []
    legend_labels = []
    colors = ["#ecc947","#4d79a4",'#b07aa0']
    for m_i, model in enumerate(models):
        color = colors[m_i]
        for b_i, bpt in enumerate(bpt_order):
            sub = per_desc_df[
                (per_desc_df["model_name"] == model) &
                (per_desc_df["build_prompt_type"] == bpt)
            ]
            # Build a vector aligned to 'descriptors' order (fill missing with np.nan)
            vals_map = dict(zip(zip(sub["descriptor"], sub["build_prompt_type"]), sub["corr_over_participants"]))
            y = [vals_map.get((d, bpt), np.nan) for d in descriptors]

            pos = x + offsets[m_i * len(bpt_order) + b_i]

            # Style: byname = solid filled bars (default); bysmiles = dashed outline (no fill)
            if bpt == "byname":
                bars = ax.bar(pos, y, width=bar_width, label=None, color=color,edgecolor='black')  # default filled
            else:  # "bysmiles"
                bars = ax.bar(pos, y, width=bar_width, label=None,hatch='//',color=color,edgecolor='black')  # dashed outline
                for r in bars:
                    r.set_linestyle("--")
                    r.set_linewidth(1.5)

            # add one handle per series for legend (first bar is enough)
            if bars:
                legend_handles.append(bars[0])
                legend_labels.append(f"{model} ({bpt})")

    # Axes formatting
    ax.set_xlabel("Descriptor")
    ax.set_ylabel("Avg. correlation (LLM vs human)")
    ax.set_title("LLM alignment by descriptor (solid: byname, dashed: bysmiles)")
    ax.set_xticks(x)
    ax.set_xticklabels(descriptors, rotation=60, ha="right")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=7))
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.6)

    # De-duplicate legend entries while preserving order
    seen = set()
    uniq_handles, uniq_labels = [], []
    for h, l in zip(legend_handles, legend_labels):
        if l not in seen:
            uniq_handles.append(h)
            uniq_labels.append(l)
            seen.add(l)
    ax.legend(uniq_handles, uniq_labels, frameon=False, ncol=2, fontsize=9)

    fig.tight_layout()
    # Save both PNG and PDF
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path.with_suffix(".png"), dpi=300)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    print(f"[PLOT] Saved: {out_path.with_suffix('.png')} and .pdf")


# ------------------------------
# Orchestration / CLI
# ------------------------------

def main():
    p = argparse.ArgumentParser(description="Compute participant-wise descriptor correlations: LLM & Transformer vs Human (globs + per-layer)")
    # p.add_argument("--human_csv", required=True, help="Path to human responses CSV")
    p.add_argument("--ds", required=True, help="Dataset key for get_descriptors (e.g., 'keller2016', 'sagar2023')")
    p.add_argument(
    "--tr-source",
    choices=["predictions", "metrics"],
    default="predictions",
    help="Where to read transformer results from."
)

    args = p.parse_args()
    tr_metrics_glob = f"{BASE_DIR}/llm_responses/metrics_model-*_ds-{args.ds}*.csv" 

    out_dir = f"{BASE_DIR}/correlation_reports/{args.ds}"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    descriptors = list(get_descriptors(args.ds))


    human = load_human(args.ds)
    human = human[human["participant_id"] == -1]
    print("Loaded human data:", len(human), "rows; participants:", human["participant_id"].nunique() if "participant_id" in human.columns else "N/A",human["cid"].nunique() if "cid" in human.columns else "N/A")
    #save human data for reference
    # human.to_csv(out_dir / "human_data.sv", index=False)
    # print(human.head())
    # ---- LLM ingest ----
    llm_frames: List[pd.DataFrame] = []
    llm_glob = f"{BASE_DIR}/llm_responses/{args.ds}_odor_llm_scores_*.csv"
    for fp in glob.glob(llm_glob):
        try:
            llm_frames.append(load_llm_file(fp))
        except Exception as e:
            print(f"[LLM] Skipping {fp}: {e}")
   
    llm = pd.concat(llm_frames, ignore_index=True) if llm_frames else None
    llm =_standardize_descriptor_columns(llm, source="human")

    # ---- Transformer ingest ----
    tr_frames: List[pd.DataFrame] = []
    transformer_glob = f"{BASE_DIR}/Oct34_transformer_responses/{args.ds}_odor_regression_scores_model-*_layer-*_nfold-*_ncomp-*_z-*.csv"
    
    for fp in glob.glob(transformer_glob):
        try:
            tr_frames.append(load_transformer_file(fp))
        except Exception as e:
            print(f"[TR] Skipping {fp}: {e}")



    
    tr = pd.concat(tr_frames, ignore_index=True) if tr_frames else None

    # === LLM path ===
    if llm is not None and not llm.empty:
        keep_cols = set(["cid", "model_name", "build_prompt_type", "temperature", "repeat"]) | set(descriptors)
        llm = llm[[c for c in llm.columns if c in keep_cols]]

        # Average over repeats first
        llm_avg = average_llm_repeats(llm, descriptors)

        # A) Per (model, bpt, temperature)
        llm_ppd, llm_pp, llm_grand, llm_pdesc = compute_llm_correlations(human, llm_avg, descriptors)
        llm_ppd.to_csv(out_dir / "llm_per_participant_descriptor.csv", index=False)
        llm_pp.to_csv(out_dir / "llm_per_participant_avg.csv", index=False)
        llm_grand.to_csv(out_dir / "llm_grand_average.csv", index=False)
        llm_pdesc.to_csv(out_dir / "llm_per_descriptor_avg.csv", index=False)

        # B) Aggregate models across temperatures (model aggregation)
        llm_agg = aggregate_llm_over_temperature(llm_avg, descriptors)
        a_ppd, a_pp, a_grand, a_pdesc = compute_llm_correlations(human, llm_agg, descriptors)
        a_ppd.to_csv(out_dir / "llmAGG_per_participant_descriptor.csv", index=False)
        a_pp.to_csv(out_dir / "llmAGG_per_participant_avg.csv", index=False)
        a_grand.to_csv(out_dir / "llmAGG_grand_average.csv", index=False)
        a_pdesc.to_csv(out_dir / "llmAGG_per_descriptor_avg.csv", index=False)

    # === Transformer path ===
    # === Transformer path ===
    if args.tr_source == "predictions":
        # (existing code path you already have)
        tr_frames: List[pd.DataFrame] = []
        transformer_glob = f"{BASE_DIR}/Oct34_transformer_responses/{args.ds}_odor_regression_scores_model-*_layer-*_nfold-*_ncomp-*_z-*.csv"
        for fp in glob.glob(transformer_glob):
            try:
                tr_frames.append(load_transformer_file(fp))
            except Exception as e:
                print(f"[TR] Skipping {fp}: {e}")
        tr = pd.concat(tr_frames, ignore_index=True) if tr_frames else None

        if tr is not None and not tr.empty:
            keep_cols = set(["participant_id", "cid", "model_name", "layer", "repeat"]) | set(descriptors)
            tr = tr[[c for c in tr.columns if c in keep_cols]]

            tr["participant_id"] = pd.to_numeric(tr["participant_id"], errors="coerce").fillna(-1).astype(int)
            print("Transformer data (predictions):", len(tr), "rows; participants:", tr["participant_id"].unique())
            tr = tr[tr["participant_id"] == -1]  # keep grand-average rows
            pre = "-1"

            # Per-layer
            tr_by_layer = average_transformer_repeats(tr, descriptors, by_layer=True)
            human_gavg = human[human["participant_id"] == -1]
            print("Human grand average data:", len(human_gavg), "rows.")
            L_ppd, L_pp, L_grand, L_pdesc = compute_transformer_correlations(human_gavg, tr_by_layer, descriptors, by_layer=True)
            L_ppd.to_csv(out_dir / f"{pre}trPERLAYER_per_participant_descriptor.csv", index=False)
            L_pp.to_csv(out_dir / f"{pre}trPERLAYER_per_participant_avg.csv", index=False)
            L_grand.to_csv(out_dir / f"{pre}trPERLAYER_grand_average.csv", index=False)
            L_pdesc.to_csv(out_dir / f"{pre}trPERLAYER_per_descriptor_avg.csv", index=False)

            plot_llm_descriptor_bars(
            a_pdesc[["model_name","build_prompt_type","descriptor","corr_over_participants"]].copy(),
            descriptors=descriptors,
            out_path=out_dir / "llm_descriptor_barplot"
                )

            # Aggregate layers
            tr_agg = average_transformer_repeats(tr, descriptors, by_layer=False)
            M_ppd, M_pp, M_grand, M_pdesc = compute_transformer_correlations(human_gavg, tr_agg, descriptors, by_layer=False)
            M_ppd.to_csv(out_dir / f"{pre}trAGG_per_participant_descriptor.csv", index=False)
            M_pp.to_csv(out_dir / f"{pre}trAGG_per_participant_avg.csv", index=False)
            M_grand.to_csv(out_dir / f"{pre}trAGG_grand_average.csv", index=False)
            M_pdesc.to_csv(out_dir / f"{pre}trAGG_per_descriptor_avg.csv", index=False)

    elif args.tr_source == "metrics":
        # New metrics path (no human merge needed)
        trm = load_transformer_metrics_files(args.ds, descriptors, tr_metrics_glob)
        trm["participant_id"] = pd.to_numeric(tr["participant_id"], errors="coerce").fillna(-1).astype(int)

        if trm is None or trm.empty:
            print("[TR-METRICS] No metrics rows found.")
        else:
            # If you want to mirror your previous restriction, keep only -1 rows (grand average):
            trm["participant_id"] = pd.to_numeric(trm["participant_id"], errors="coerce")
            trm = trm[trm["participant_id"] == -1]
            pre = "-1"

            # Per-layer (keep layers separate)
            L_ppd, L_pp, L_grand, L_pdesc = aggregate_transformer_metrics_corr(trm, by_layer=True)
            L_ppd.to_csv(out_dir / f"{pre}trPERLAYER_per_participant_descriptor.csv", index=False)
            L_pp.to_csv(out_dir / f"{pre}trPERLAYER_per_participant_avg.csv", index=False)
            L_grand.to_csv(out_dir / f"{pre}trPERLAYER_grand_average.csv", index=False)
            L_pdesc.to_csv(out_dir / f"{pre}trPERLAYER_per_descriptor_avg.csv", index=False)

            # Aggregate across layers (drop 'layer' in grouping)
            trm_no_layer = trm.copy()
            if "layer" in trm_no_layer.columns:
                # average correlation across layers first
                trm_no_layer = (
                    trm_no_layer
                    .groupby(["participant_id","model_name","descriptor"], dropna=False)["correlation"]
                    .mean().reset_index()
                )
            M_ppd, M_pp, M_grand, M_pdesc = aggregate_transformer_metrics_corr(trm_no_layer, by_layer=False)
            M_ppd.to_csv(out_dir / f"{pre}trAGG_per_participant_descriptor.csv", index=False)
            M_pp.to_csv(out_dir / f"{pre}trAGG_per_participant_avg.csv", index=False)
            M_grand.to_csv(out_dir / f"{pre}trAGG_grand_average.csv", index=False)
            M_pdesc.to_csv(out_dir / f"{pre}trAGG_per_descriptor_avg.csv", index=False)


if __name__ == "__main__":
    main()
