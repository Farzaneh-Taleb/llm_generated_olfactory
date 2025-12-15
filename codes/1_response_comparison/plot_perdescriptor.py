# plot_llm_descriptor_bars.py
from __future__ import annotations
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------
# User inputs (edit paths)
# ---------------------------
SAGAR_CSV = "../alignment_score/sagar2023/llm_alignment score_per_descriptor_avg.csv"
KELLER_CSV = "../alignment_score/keller2016/llm_alignment score_per_descriptor_avg.csv"
OUT_DIR = "figs_llm_descriptor_bars"

# ---------------------------
# Descriptor canonicalization
# ---------------------------
CANON_MAP = {
    # spelling / morphology
    "intensive": "intensity",
    "pleasant": "pleasantness",
    "familiar": "familiarity",
    # synonymy between datasets
    "flower": "floral",
    "fruit": "fruity",
    "fish": "fishy",
    "spices": "spicy",
    "cold": "cool",
    # keep as-is if already canonical
    "floral": "floral",
    "fruity": "fruity",
    "fishy": "fishy",
    "spicy": "spicy",
    "cool": "cool",
    "warm": "warm",
    "sweet": "sweet",
    "sour": "sour",
    "musky": "musky",
    "bakery": "bakery",
    "burnt": "burnt",
    "decayed": "decayed",
    "pleasantness": "pleasantness",
    "intensity": "intensity",
    "familiarity": "familiarity",
    "edible": "edible",
    "chemical": "chemical",
    "grass": "grass",
    "wood": "wood",
    "ammonia": "ammonia",
    "acid": "acid",
    # add more if needed...
}

def canon_desc(s: str) -> str:
    key = str(s).strip().lower()
    return CANON_MAP.get(key, key)

def load_and_standardize(csv_path: str, dataset_name: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Normalize column names in case there are spaces / typos
    df = df.rename(columns={
        "model_name": "model_name",
        "build_prompt_type": "build_prompt_type",
        "temperature": "temperature",
        "descriptor": "descriptor",
        "alignment score_over_participants": "score",
        "alignment_score_over_participants": "score",   # fallback
    })
    # Canonicalize descriptors
    df["descriptor_canon"] = df["descriptor"].apply(canon_desc)
    df["dataset"] = dataset_name
    # Keep only the columns we need
    return df[["model_name", "build_prompt_type", "descriptor_canon", "score", "dataset"]]

def intersect_descriptors(df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[str]:
    # Only plot categories present in BOTH datasets (in any prompt type)
    a_set = set(df_a["descriptor_canon"].unique())
    b_set = set(df_b["descriptor_canon"].unique())
    common = sorted(a_set & b_set)
    return common

def make_per_model_plots(df_all: pd.DataFrame, descriptors: list[str]):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # ---------------- UI constants ----------------
    DATASETS = ["Sagar", "Keller"]
    PROMPT_TYPES = ["bysmiles", "byname"]  # <-- order: SMILES first, Name second
    HATCH_MAP = {"byname": "//", "bysmiles": None}  # <-- Name is hatched
    DATASET_COLORS = {"Sagar": "#4d79a4", "Keller": "#ecc947"}

    LABEL_FS = 16
    TICK_FS = 16
    LEGEND_FS = 16

    # Remap dataset codes → display names (so pivot keys match DATASETS)
    DATASET_REMAP = {
        "sagar2023": "Sagar",
        "keller2016": "Keller",
        "Sagar": "Sagar",
        "Keller": "Keller",
    }
    df_all = df_all.copy()
    df_all["dataset"] = df_all["dataset"].map(lambda x: DATASET_REMAP.get(x, x))

    # Prepare output directory
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # For each model, make one figure
    for model in df_all["model_name"].unique():
        sub = df_all[(df_all["model_name"] == model) &
                     (df_all["descriptor_canon"].isin(descriptors))]

        # Pivot to easier plotting: index=(descriptor), columns=(dataset, prompt_type)
        pivot = (sub
                 .groupby(["descriptor_canon", "dataset", "build_prompt_type"])["score"]
                 .mean()
                 .unstack(["dataset", "build_prompt_type"])
                 .reindex(descriptors))

        # Layout
        n_desc = len(descriptors)
        n_dsets = len(DATASETS)
        n_prompts = len(PROMPT_TYPES)
        bars_per_desc = n_dsets * n_prompts

        x = np.arange(n_desc, dtype=float)
        total_group_width = 0.8
        bar_width = total_group_width / bars_per_desc

        fig, ax = plt.subplots(figsize=(max(10, n_desc * 0.6), 5.5))

        # Draw bars
        for d_i, dname in enumerate(DATASETS):
            dataset_offset = (d_i * n_prompts) * bar_width - total_group_width / 2 + bar_width / 2
            dataset_color = DATASET_COLORS[dname]

            for p_j, ptype in enumerate(PROMPT_TYPES):  # SMILES first, then Name
                offset = dataset_offset + p_j * bar_width
                heights = []
                for desc in descriptors:
                    val = np.nan
                    try:
                        val = pivot.loc[desc, (dname, ptype)]
                    except KeyError:
                        pass
                    heights.append(val)

                bars = ax.bar(
                    x + offset,
                    heights,
                    width=bar_width,
                    edgecolor="black",
                    color=dataset_color,
                )
                if HATCH_MAP[ptype]:
                    for b in bars:
                        b.set_hatch(HATCH_MAP[ptype])

        # Cosmetics: labels, ticks, limits
        ax.set_ylabel("Pearson r", fontsize=LABEL_FS)
        ax.set_xticks(x)
        ax.set_xticklabels(descriptors, rotation=45, ha="right", fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylim(-0.1, 0.65)
        ax.margins(x=0.01)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # --- Legend (dataset colors + prompt hatches) ---
        dataset_proxies = [
            Patch(facecolor=DATASET_COLORS[d], edgecolor="black", label=d)
            for d in DATASETS
        ]
        prompt_proxies = [
            Patch(facecolor="white", edgecolor="black", label="SMILES"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label="Name"),
        ]
        all_handles = dataset_proxies + prompt_proxies

        plt.subplots_adjust(bottom=0.28)
        fig.legend(
            handles=all_handles,
            ncol=4,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.09),
            frameon=False,
            fontsize=LEGEND_FS,
        )

        out_path = Path(OUT_DIR) / f"{model.replace('/', '_')}_descriptor_bars.pdf"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")



def make_per_dataset_plots(df_all: pd.DataFrame, descriptors: list[str]):
    from pathlib import Path
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # ---------------- UI constants ----------------
    DATASETS = ["Sagar", "Keller"]                 # one figure per dataset
    PROMPT_TYPES = ["bysmiles", "byname"]          # <-- SMILES first, Name second
    HATCH_MAP = {"byname": "//", "bysmiles": None} # <-- Name is hatched

    LABEL_FS = 16
    TICK_FS = 16
    LEGEND_FS = 16

    # Remap dataset codes → display names
    DATASET_REMAP = {
        "sagar2023": "Sagar",
        "keller2016": "Keller",
        "Sagar": "Sagar",
        "Keller": "Keller",
    }
    df_all = df_all.copy()
    df_all["dataset"] = df_all["dataset"].map(lambda x: DATASET_REMAP.get(x, x))

    # Prepare output directory
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    # Pick model palette (consistent across figures)
    all_models = list(df_all["model_name"].unique())
    BASE_COLORS = ["#4d79a4", "#ecc947"]
    MODEL_COLORS = {m: BASE_COLORS[i % len(BASE_COLORS)] for i, m in enumerate(all_models)}

    for dataset in DATASETS:
        sub = df_all[
            (df_all["dataset"] == dataset)
            & (df_all["descriptor_canon"].isin(descriptors))
        ].copy()
        if sub.empty:
            print(f"[warn] No rows for dataset {dataset}; skipping.")
            continue

        models_here = list(sub["model_name"].unique())

        # Pivot: rows = descriptor, cols = (model, prompt_type)
        pivot = (
            sub.groupby(["descriptor_canon", "model_name", "build_prompt_type"])["score"]
               .mean()
               .unstack(["model_name", "build_prompt_type"])
               .reindex(descriptors)
        )

        n_desc = len(descriptors)
        n_models = len(models_here)
        n_prompts = len(PROMPT_TYPES)
        bars_per_desc = n_models * n_prompts

        x = np.arange(n_desc, dtype=float)
        total_group_width = 0.8
        bar_width = total_group_width / bars_per_desc

        fig, ax = plt.subplots(figsize=(max(10, n_desc * 0.6), 5.5))

        # Draw bars: group by descriptor, within each descriptor: [model x prompt]
        for m_i, model in enumerate(models_here):
            model_offset = (m_i * n_prompts) * bar_width - total_group_width / 2 + bar_width / 2
            m_color = MODEL_COLORS[model]

            for p_j, ptype in enumerate(PROMPT_TYPES):  # SMILES first, Name second
                offset = model_offset + p_j * bar_width
                heights = []
                for desc in descriptors:
                    val = np.nan
                    if (model, ptype) in pivot.columns:
                        try:
                            val = pivot.loc[desc, (model, ptype)]
                        except KeyError:
                            pass
                    heights.append(val)

                bars = ax.bar(
                    x + offset,
                    heights,
                    width=bar_width,
                    edgecolor="black",
                    color=m_color,
                )
                if HATCH_MAP[ptype]:
                    for b in bars:
                        b.set_hatch(HATCH_MAP[ptype])

        # Cosmetics
        ax.set_ylabel("Pearson r", fontsize=LABEL_FS)
        ax.set_xticks(x)
        ax.set_xticklabels(descriptors, rotation=45, ha="right", fontsize=TICK_FS)
        ax.tick_params(axis="y", labelsize=TICK_FS)
        ax.set_ylim(-0.1, 0.65)
        ax.margins(x=0.01)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Legends: models (colors) + prompt types (hatches)
        model_proxies = [
            Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=m) for m in models_here
        ]
        prompt_proxies = [
            Patch(facecolor="white", edgecolor="black", label="SMILES"),
            Patch(facecolor="white", edgecolor="black", hatch="//", label="Name"),
        ]
        all_handles = model_proxies + prompt_proxies

        plt.subplots_adjust(bottom=0.30)
        fig.legend(
            handles=all_handles,
            ncol=min(4, len(all_handles)),
            loc="lower center",
            bbox_to_anchor=(0.5, -0.11),
            frameon=False,
            fontsize=LEGEND_FS,
            title="",
            title_fontsize=LEGEND_FS,
        )

        out_path = Path(OUT_DIR) / f"{dataset}_descriptor_bars_by_model.pdf"
        fig.tight_layout()
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {out_path}")

def main():
    # Load both datasets
    df_sagar = load_and_standardize(SAGAR_CSV, "Sagar")
    df_keller = load_and_standardize(KELLER_CSV, "Keller")

    # Restrict to the same set of models present across both datasets (safety)
    common_models = sorted(set(df_sagar["model_name"]) & set(df_keller["model_name"]))
    if not common_models:
        print("No common models found across datasets. Will plot models present in either dataset.")
        common_models = sorted(set(df_sagar["model_name"]) | set(df_keller["model_name"]))

    # Intersect descriptor categories across the two datasets
    descriptors_common = intersect_descriptors(df_sagar, df_keller)
    if not descriptors_common:
        raise ValueError("No overlapping descriptor categories after canonicalization. "
                         "Add more entries to CANON_MAP.")

    # Combine
    df_all = pd.concat([df_sagar, df_keller], ignore_index=True)
    # Keep only common models (optional)
    df_all = df_all[df_all["model_name"].isin(common_models)]

    # Make plots per model
    make_per_model_plots(df_all, descriptors_common)
    make_per_dataset_plots(df_all, descriptors_common)
    # make_per_dataset_plots(df_all, descriptors_common)

if __name__ == "__main__":
    main()
