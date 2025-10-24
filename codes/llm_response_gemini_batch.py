from __future__ import annotations
import os, json, argparse, time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# --- Gemini SDK ---
from google import genai
from google.genai import types

from utils.helpers import common_cids_per_ds
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import BASE_DIR

# ---------------- Config ----------------
# ---------------- Config ----------------
RATE_RANGE = {
    "keller2016": (0.0, 100.0),
    "sagar2023": (-1,1),
    "leffingwell": (0.0, 1.0),
    "ravia": (0.0, 1.0),
    "snitz": (0.0, 1.0),
    # Add others as needed
}
# global RATE_MIN, RATE_MAX

def get_rate_range(ds: str) -> tuple[float, float]:
    """Return (RATE_MIN, RATE_MAX) for the given dataset."""
    if ds not in RATE_RANGE:
        raise ValueError(f"Unknown dataset: {ds}. Please add it to RATE_RANGE.")
    return RATE_RANGE[ds]

# Example usage
# ds = "leffingwell"

INCLUDE_CONFIDENCE = False
INPUT_TYPE = "isomericsmiles"         # 'isomericsmiles' or 'cid'
SYSTEM_MSG = "You are an olfactory rater. Output ONLY valid JSON."
BATCH_REGISTRY = f"{BASE_DIR}/llm_responses/batch_registry.jsonl"

BUILD_PROMPT_CHOICES = ("bysmiles", "byname")

# Gemini client (reads GEMINI_API_KEY from env)
client = genai.Client()

# -------- Registry I/O (unchanged) --------
def log_batch_entry(ds, model_name, temp, batch_id, build_prompt_type, n_repeats):
    os.makedirs(os.path.dirname(BATCH_REGISTRY), exist_ok=True)
    entry = {
        "ds": ds,
        "model_name": model_name,
        "temperature": temp,
        "batch_id": batch_id,           # e.g. 'batches/123456...'
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

# -------- Helpers (unchanged prompts) --------
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

def row_smiles(row: pd.Series) -> Optional[str]:
    smi = _clean_cell(row.get(INPUT_TYPE, ""))
    return smi or None

def row_name(row: pd.Series) -> Optional[str]:
    nm = _clean_cell(row.get("name", ""))
    return nm or None

def build_prompt(smiles: str, descriptors: List[str], rate_min: float, rate_max: float, include_confidence: bool=False) -> str:
    desc_list = ", ".join([f'"{d}"' for d in descriptors])
    return f"""Molecule:
- ISOMERIC SMILES: {smiles}

Descriptors (rate each from {rate_min} to {rate_max}): [{desc_list}]

Rules:
- Return ONLY a single JSON object with those keys and numeric values in range.
"""

def build_prompt_byname(name: str, descriptors: List[str], rate_min: float, rate_max: float, include_confidence: bool=False) -> str:
    desc_list = ", ".join([f'"{d}"' for d in descriptors])
    return f"""Molecule:
- Name: {name}

Descriptors (rate each from {rate_min} to {rate_max}): [{desc_list}]

Rules:
- Return ONLY a single JSON object with those keys and numeric values in range.
"""

def validate_response(resp_text: str, descriptors: List[str], rate_min: float, rate_max: float, include_confidence: bool) -> Dict[str, float]:
    obj = json.loads(resp_text)
    out: Dict[str, float] = {}
    for d in descriptors:
        if d not in obj:
            raise ValueError(f"Missing key: {d}")
        v = float(obj[d])
        out[d] = max(rate_min, min(rate_max, v))
    if include_confidence:
        c = obj.get("confidence", None)
        if c is None:
            out["confidence"] = None
        else:
            out["confidence"] = max(0.0, min(1.0, float(c)))
    return out

# --------- Structured output schema for Gemini ---------
def make_response_schema(descriptors: list[str], include_confidence: bool, RATE_MIN: float, RATE_MAX: float) -> dict:
    props = {
        d: {"type": "NUMBER", "minimum": RATE_MIN, "maximum": RATE_MAX}
        for d in descriptors
    }
    required = list(descriptors)
    if include_confidence:
        props["confidence"] = {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0}
        # leave it optional (don't add to required)

    return {
        "type": "OBJECT",
        "properties": props,
        "required": required,
    }


# ---------- BATCH: PREP & SUBMIT (Gemini) ----------
def make_jsonl_for_batch(
    df: pd.DataFrame,
    ds_name: str,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    out_jsonl_path: str,
    build_prompt_type: str,
    n_repeats: int = 1,
    *,
    RATE_MIN: float, RATE_MAX: float
) -> tuple[int, int]:
    """
    Write one GenerateContentRequest PER DEDUPED CID row, repeated n_repeats times.
    Each line is: {"key": "...", "request": {...}} for Gemini Batch API.
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    df = df.copy().reset_index(drop=True)
    df["cid"] = pd.to_numeric(df["cid"], errors="coerce").astype("Int64")

    schema = make_response_schema(descriptors, INCLUDE_CONFIDENCE,RATE_MIN, RATE_MAX)
    total_rows = len(df)
    used_reqs = 0

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for row_id, r in df.iterrows():
            cid = r.get("cid")
            if pd.isna(cid):
                continue
            cid = int(cid)

            if build_prompt_type == "bysmiles":
                smi = row_smiles(r)
                if not smi:
                    continue
                prompt = build_prompt(smi, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
            else:
                nm = row_name(r)
                if not nm:
                    continue
                prompt = build_prompt_byname(nm, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)

            for rep in range(n_repeats):
                key = f"{ds_name}|{temperature}|bpt:{build_prompt_type}|row{row_id}|rep{rep}|cid{cid}"
                request = {
                    "system_instruction": {"parts": [{"text": SYSTEM_MSG}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {
                        "temperature": float(temperature),
                        "response_mime_type": "application/json",
                        "response_schema": schema,  # SDK will serialize this
                    },
                }
                f.write(json.dumps({"key": key, "request": request}, default=lambda o: o.__dict__) + "\n")
                used_reqs += 1

    return total_rows, used_reqs

def submit_batch(jsonl_path: str, model_name: str, display_name: str="olf-batch") -> str:
    """
    Upload JSONL and create a Gemini Batch job.
    Returns the job name, e.g. 'batches/123456789'.
    """
    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(display_name=os.path.basename(jsonl_path), mime_type="jsonl"),
    )
    job = client.batches.create(
        model=model_name,          # e.g. "gemini-2.5-flash" or "gemini-2.5-pro"
        src=uploaded.name,         # file name/id from Files API
        config={"display_name": display_name},
    )
    print(f"Submitted batch: {job.name}  (input_file={uploaded.name})")
    return job.name

# ---------- BATCH: COLLECT & MERGE (Gemini) ----------
def download_batch_outputs(job_name: str, save_dir: str) -> Tuple[str, List[str]]:
    """
    Poll once; if finished and results are in a file, download it.
    Returns (state, [paths]).
    """
    try:
        bj = client.batches.get(name=job_name)
    except ValueError as e:
        return "JOB_NOT_FOUND", []
        
    state = bj.state.name  # 'JOB_STATE_*'
    print(f"Batch {job_name} status: {state}")

    os.makedirs(save_dir, exist_ok=True)
    paths: List[str] = []

    if state == "JOB_STATE_SUCCEEDED":
        if bj.dest and getattr(bj.dest, "file_name", None):
            # File results
            out_path = os.path.join(save_dir, f"{job_name.replace('/','_')}_out.jsonl")
            content = client.files.download(file=bj.dest.file_name)  # bytes
            with open(out_path, "wb") as fh:
                fh.write(content)
            print(f"Saved output file: {out_path}")
            paths.append(out_path)
        elif bj.dest and getattr(bj.dest, "inlined_responses", None):
            # Inline results -> persist to a file for unified parsing
            out_path = os.path.join(save_dir, f"{job_name.replace('/','_')}_out.jsonl")
            with open(out_path, "w", encoding="utf-8") as fh:
                for idx, ir in enumerate(bj.dest.inlined_responses):
                    # Store a line mimicking the file result shape
                    fh.write(json.dumps({"key": f"inline-{idx}", "response": ir.response}, default=lambda o: getattr(o, "__dict__", str(o))) + "\n")
            print(f"Saved inline output file: {out_path}")
            paths.append(out_path)

    return state, paths

def parse_batch_jsonl(jsonl_paths: List[str], descriptors: List[str],RATE_MIN, RATE_MAX) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Parse Gemini Batch outputs. For each line:
      { "key": "<...|row{r}|rep{k}|...>", "response": { ... GenerateContentResponse ... } }
    Extracts the model's JSON text and validates it.
    """
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _extract_text_from_response(resp: Dict[str, Any]) -> str:
        # Try the canonical content.parts[0].text
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            texts = []
            for p in parts:
                if "text" in p and isinstance(p["text"], str):
                    texts.append(p["text"])
            return "\n".join(texts)
        except Exception:
            # Fallbacks
            return resp.get("text", "") or ""

    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = rec.get("key", "")
                resp = rec.get("response", {})
                if not key or not resp:
                    # Some lines may be status objects; skip them
                    continue

                # recover row/rep from key
                parts = key.split("|")
                row_part = next((x for x in parts if x.startswith("row")), None)
                rep_part = next((x for x in parts if x.startswith("rep")), None)
                if row_part is None or rep_part is None:
                    continue
                row_id = int(row_part.replace("row", ""))
                rep = int(rep_part.replace("rep", ""))

                try:
                    content_text = _extract_text_from_response(resp)
                    scores = validate_response(content_text, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                except Exception as e:
                    scores = {"error": f"validate_error: {e}"}

                out[(row_id, rep)] = scores

    return out

# ---------- CSV writer (unchanged) ----------
def write_final_csv_by_rowrep(
    df_input: pd.DataFrame,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    rowrep_to_scores: Dict[Tuple[int, int], Dict[str, Any]],
    out_csv: str,
    build_prompt_type: str,
    n_repeats: int,
):
    df = df_input.copy().reset_index(drop=True)
    rows: List[Dict[str, Any]] = []
    for row_id, row in df.iterrows():
        cid = pd.to_numeric(row.get("cid"), errors="coerce")
        cid = int(cid) if pd.notna(cid) else None
        smi = row_smiles(row) or ""
        name = row_name(row) or ""
        for rep in range(n_repeats):
            base = {
                "participant_id": row.get("participant_id"),
                "cid": cid,
                "repeat": rep,
                "temperature": temperature,
                INPUT_TYPE: smi,
                "name": name,
                "model_name": model_name,
                "build_prompt_type": build_prompt_type,
            }
            scores = rowrep_to_scores.get((row_id, rep), {"error": "missing_in_batch_output"})
            rec = {**base, **scores}
            rows.append(rec)

    cols = ["participant_id", "cid", "repeat", "temperature", INPUT_TYPE, "name", "model_name", "build_prompt_type"] + descriptors
    if INCLUDE_CONFIDENCE: cols.append("confidence")
    if any("error" in r for r in rows): cols.append("error")
    out_df = pd.DataFrame(rows)
    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# ---------- CLI (changes: submit uses Gemini submit_batch) ----------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    



    s_submit = sub.add_parser("submit", help="Create JSONL and submit Gemini Batch")
    s_submit.add_argument("--ds", required=True)
    s_submit.add_argument("--temperature", type=float, default=0.0)
    s_submit.add_argument("--model_name", required=True, help="Gemini model, e.g. 'gemini-2.5-flash'")
    s_submit.add_argument("--build-prompt-type", dest="build_prompt_type",
                          choices=BUILD_PROMPT_CHOICES, default="bysmiles")
    s_submit.add_argument("--n-repeats", type=int, default=1)

    s_collect = sub.add_parser("collect", help="Fetch finished batch, parse, and write CSV")
    s_collect.add_argument("--save_dir", required=True)

    args = ap.parse_args()

    if args.cmd == "submit":
        ds = args.ds
        RATE_MIN, RATE_MAX = get_rate_range(ds)
        input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
        df = pd.read_csv(input_csv)
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
        if "concentration" in df.columns and "keller" in ds.lower():
            df["concentration"] = df["concentration"].astype(float)
            df = df[df["concentration"] == 0.001].copy()

        # Keep only common CIDs (your helper may accept df directly)
        cids = sorted(map(int, common_cids_per_ds(df)))
        df = df[df["cid"].isin(cids)].drop_duplicates(subset=["cid"]).reset_index(drop=True)

        model_name = args.model_name
        build_prompt_type = args.build_prompt_type
        n_repeats = int(args.n_repeats)
        descriptors = get_descriptors(ds)

        jsonl_path = f"tmp/{ds}_{model_name}_{args.temperature}_{build_prompt_type}_reps-{n_repeats}.jsonl"
        total, used_reqs = make_jsonl_for_batch(
            df=df,
            ds_name=ds,
            model_name=model_name,
            temperature=args.temperature,
            descriptors=descriptors,
            out_jsonl_path=jsonl_path,
            build_prompt_type=build_prompt_type,
            n_repeats=n_repeats,
        )
        print(f"Prepared JSONL: {jsonl_path}  (dedup_rows={total}, requests={used_reqs})")

        job_name = submit_batch(jsonl_path, model_name=model_name, display_name=f"{ds}-{build_prompt_type}")
        print(f"BATCH_JOB_NAME={job_name}")
        log_batch_entry(ds, model_name, args.temperature, job_name, build_prompt_type, n_repeats)

    elif args.cmd == "collect":
        reg = load_registry()
        for entry in reg:
            ds = entry["ds"]
            RATE_MIN, RATE_MAX = get_rate_range(ds)
            input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
            df = pd.read_csv(input_csv)
            df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
            if "concentration" in df.columns and "keller" in ds.lower():
                df["concentration"] = df["concentration"].astype(float)
                df = df[df["concentration"] == 0.001].copy()

            cids = sorted(map(int, common_cids_per_ds(df)))
            df = df[df["cid"].isin(cids)].drop_duplicates(subset=["cid"]).reset_index(drop=True)

            model_name = entry["model_name"]
            temp = entry["temperature"]
            job_name = entry["batch_id"]   # same key in your registry
            build_prompt_type = entry.get("build_prompt_type", "bysmiles")
            n_repeats = int(entry.get("n_repeats", 1))
            descriptors = get_descriptors(ds)

            state, paths = download_batch_outputs(job_name, args.save_dir)
            if state != "JOB_STATE_SUCCEEDED":
                print(f"Batch {job_name} not completed yet (state={state}). Skipping.")
                continue
            if not paths:
                print(f"No output file(s) for batch {job_name}. Skipping.")
                continue

            rowrep_to_scores = parse_batch_jsonl(paths, descriptors,RATE_MIN, RATE_MAX)
            output_csv = (
                f"{BASE_DIR}/llm_responses/"
                f"{ds}_odor_llm_scores_temp-{temp}_model-{model_name}_bpt-{build_prompt_type}_reps-{n_repeats}.csv"
            )
            write_final_csv_by_rowrep(
                df_input=df,
                model_name=model_name,
                temperature=temp,
                descriptors=descriptors,
                rowrep_to_scores=rowrep_to_scores,
                out_csv=output_csv,
                build_prompt_type=build_prompt_type,
                n_repeats=n_repeats,
            )

if __name__ == "__main__":
    main()
