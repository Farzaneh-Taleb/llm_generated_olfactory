import pandas as pd
import numpy as np
from utils.config import BASE_DIR
from utils.helpers import common_cids_per_ds
from utils.ds_utils import get_descriptors
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr

# ---------------- Config ----------------
ds = "sagar2023"
model = "gemini-flash-lite-latest"
temp = 0.0

# ---------------- I/O ----------------
llm_bysmiles_path = f"{BASE_DIR}/results/responses/llm_responses/{ds}_odor_llm_scores_temp-{temp}_model-{model}_bpt-bysmiles.csv"
llm_byname_path   = f"{BASE_DIR}/results/responses/llm_responses/{ds}_odor_llm_scores_temp-{temp}_model-{model}_bpt-byname.csv"
human_path        = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"

llm_bysmiles = pd.read_csv(llm_bysmiles_path)
llm_byname   = pd.read_csv(llm_byname_path)
human        = pd.read_csv(human_path)

# ---------------- Filter to common CIDs ----------------
cids = set(common_cids_per_ds(BASE_DIR, ds))
llm_bysmiles = llm_bysmiles[llm_bysmiles["cid"].isin(cids)].copy()
llm_byname   = llm_byname[llm_byname["cid"].isin(cids)].copy()
human        = human[human["cid"].isin(cids)].copy()

# ---------------- Descriptors & safety ----------------
descriptors = get_descriptors(ds)

# ensure required columns exist
for df, name in [(llm_bysmiles,"llm_bysmiles"), (llm_byname,"llm_byname"), (human,"human")]:
    missing = [col for col in ["cid","participant_id"] if col not in df.columns]
    if missing:
        raise ValueError(f"{name} missing columns: {missing}")
    missing_desc = [d for d in descriptors if d not in df.columns]
    if missing_desc:
        raise ValueError(f"{name} missing descriptor columns: {missing_desc[:5]}{' ...' if len(missing_desc)>5 else ''}")

# coerce descriptor columns to numeric (invalid -> NaN)
def coerce_numeric(df):
    df = df.copy()
    for d in descriptors:
        df[d] = pd.to_numeric(df[d], errors="coerce")
    return df

llm_bysmiles = coerce_numeric(llm_bysmiles)
llm_byname   = coerce_numeric(llm_byname)
human        = coerce_numeric(human)

# ---------------- Long-form once ----------------
id_cols = ["cid", "participant_id"]
llm_byname_m = llm_byname.melt(id_vars=id_cols, value_vars=descriptors,
                               var_name="descriptor", value_name="llm_name_rating")
llm_bysmiles_m = llm_bysmiles.melt(id_vars=id_cols, value_vars=descriptors,
                                   var_name="descriptor", value_name="llm_smiles_rating")
human_m = human.melt(id_vars=id_cols, value_vars=descriptors,
                     var_name="descriptor", value_name="human_rating")

# Merge on cid, participant_id, descriptor (ensures alignment)
merged = (human_m
          .merge(llm_byname_m, on=id_cols+["descriptor"], how="inner")
          .merge(llm_bysmiles_m, on=id_cols+["descriptor"], how="inner"))

# Drop rows with all-NaN ratings
merged = merged.dropna(subset=["human_rating", "llm_name_rating", "llm_smiles_rating"], how="all")

# ---------------- Correlations per participant × descriptor ----------------
def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    # need at least 2 valid points and non-constant
    if len(x) < 2 or np.all(x == x.mean()) or np.all(y == y.mean()):
        return np.nan
    r, _ = pearsonr(x, y)
    return r



grp = merged.groupby(["participant_id", "descriptor"], dropna=False)

corr_name = grp.apply(lambda g: safe_pearson(g["human_rating"], g["llm_name_rating"])).rename("corr_name")
corr_smiles = grp.apply(lambda g: safe_pearson(g["human_rating"], g["llm_smiles_rating"])).rename("corr_smiles")

corr_df = pd.concat([corr_name, corr_smiles], axis=1).reset_index()

# ---------------- Optional: R^2 ----------------
r2_df = corr_df.assign(r2_name=corr_df["corr_name"]**2,
                       r2_smiles=corr_df["corr_smiles"]**2)

# ---------------- Plot: mean across participants with SEM ----------------
plot_df = (corr_df
           .melt(id_vars=["participant_id","descriptor"],
                 value_vars=["corr_name","corr_smiles"],
                 var_name="model_type", value_name="correlation"))

# summary across participants
summary = (plot_df
           .groupby(["descriptor","model_type"], as_index=False)
           .agg(mean_corr=("correlation","mean"),
                sem_corr=("correlation", lambda s: s.std(ddof=1)/np.sqrt(s.notna().sum()) if s.notna().sum()>1 else np.nan)))

# Clean display names
model_name_map = {"corr_name": "LLM by Name", "corr_smiles": "LLM by SMILES"}
summary["model_type"] = summary["model_type"].map(model_name_map)

plt.figure(figsize=(12,6))
sns.barplot(
    data=summary,
    x="descriptor",
    y="mean_corr",
    hue="model_type",
    hue_order=["LLM by Name", "LLM by SMILES"],
    errorbar=None
)
# add error bars manually for SEM
for i, row in summary.iterrows():
    x = i  # not directly usable—need bar positions from Axes
# easier: let seaborn draw CI via estimator + errorbar="se"
plt.xticks(rotation=90)
plt.ylabel("Pearson r (mean across participants)")
plt.xlabel("Descriptor")
plt.title(f"Human vs LLM correlations per descriptor — {ds} (temp={temp}, model={model})")
plt.legend(title="Model Type")
plt.tight_layout()
plt.show()
