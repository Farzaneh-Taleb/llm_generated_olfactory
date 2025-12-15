from __future__ import annotations
import sys
print(sys.path)

import os, json, argparse, time
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
# --- Gemini SDK ---
from google import genai
from google.genai import types
from utils.helpers import *
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import *

# ---------------- Config ----------------

GOOGLE_API_KEY="AIzaSyCwRo-CCQ_duuj7FFa3Tpnl_bpPJmvqzLw"

# Gemini client (reads GEMINI_API_KEY from env)
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"])

# --------- Structured output schema for Gemini ---------
def make_response_schema(
    descriptors: list[str],
    include_confidence: bool,
    RATE_MIN: float,
    RATE_MAX: float,
    *,
    pairwise: bool = False
) -> dict:
    if pairwise:
        props = {"similarity": {"type": "NUMBER", "minimum": RATE_MIN, "maximum": RATE_MAX}}
        required = ["similarity"]
    else:
        props = {
            d: {"type": "NUMBER", "minimum": RATE_MIN, "maximum": RATE_MAX}
            for d in descriptors
        }
        required = list(descriptors)

    if include_confidence:
        props["confidence"] = {"type": "NUMBER", "minimum": 0.0, "maximum": 1.0}
        # confidence optional

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
    Write one GenerateContentRequest per row, repeated n_repeats times.
    Pairwise rows are NOT deduped; single-item rows are deduped by CID upstream.
    key encodes ds|temp|bpt|row{row_id}|rep{rep}|meta...
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    df = df.copy().reset_index(drop=True)
    pairwise = is_pairwise_df(df)

    if not pairwise:
        # Single-item convenience typing (used in key only)
        df["cid"] = pd.to_numeric(df.get("cid"), errors="coerce").astype("Int64")

    schema = make_response_schema(descriptors, INCLUDE_CONFIDENCE, RATE_MIN, RATE_MAX, pairwise=pairwise)
    total_rows = len(df)
    used_reqs = 0

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for row_id, r in df.iterrows():
            if pairwise:
                pair_inputs = row_pair_inputs(r, build_prompt_type)
                if not pair_inputs:
                    continue
                a, b = pair_inputs
                if build_prompt_type == "bysmiles":
                    prompt = build_prompt_bysmiles((a, b), [], RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=True)
                else:
                    prompt = build_prompt_byname((a, b), [], RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=True)

                c1, c2 = row_pair_cids(r)
                meta_suffix = ""
                if c1 is not None and c2 is not None:
                    meta_suffix = f"|cid1{c1}_cid2{c2}"
            else:
                # Single-item
                if build_prompt_type == "bysmiles":
                    smi = row_smiles(r)
                    if not smi:
                        continue
                    prompt = build_prompt_bysmiles(smi, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=False)
                else:
                    nm = row_name(r)
                    if not nm:
                        continue
                    prompt = build_prompt_byname(nm, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=False)

                cid = r.get("cid")
                meta_suffix = f"|cid{int(cid)}" if pd.notna(cid) else ""

            for rep in range(n_repeats):
                key = f"{ds_name}|{temperature}|bpt:{build_prompt_type}|row{row_id}|rep{rep}{meta_suffix}"
                request = {
                    "system_instruction": {"parts": [{"text": SYSTEM_MSG}]},
                    "contents": [{"role": "user", "parts": [{"text": prompt}]}],
                    "generation_config": {
                        "temperature": float(temperature),
                        "response_mime_type": "application/json",
                        "response_schema": schema,
                    },
                }
                f.write(json.dumps({"key": key, "request": request}, default=lambda o: getattr(o, "__dict__", str(o))) + "\n")
                used_reqs += 1

    return total_rows, used_reqs

def submit_batch(jsonl_path: str, model_name: str, display_name: str="olf-batch") -> str:
    uploaded = client.files.upload(
        file=jsonl_path,
        config=types.UploadFileConfig(display_name=os.path.basename(jsonl_path), mime_type="jsonl"),
    )
    job = client.batches.create(
        model=model_name,
        src=uploaded.name,
        config={"display_name": display_name},
    )
    print(f"Submitted batch: {job.name}  (input_file={uploaded.name})")
    return job.name

# ---------- BATCH: COLLECT & MERGE (Gemini) ----------
def download_batch_outputs(job_name: str) -> Tuple[str, List[str]]:
    try:
        bj = client.batches.get(name=job_name)
    except ValueError:
        return "JOB_NOT_FOUND", []
    state = bj.state.name
    print(f"Batch {job_name} status: {state}")
    save_dir = f"{BASE_DIR}/results/responses/tmp/batch_outputs"
    os.makedirs(save_dir, exist_ok=True)
    paths: List[str] = []

    if state == "JOB_STATE_SUCCEEDED":
        if bj.dest and getattr(bj.dest, "file_name", None):
            out_path = os.path.join(save_dir, f"{job_name.replace('/','_')}_out.jsonl")
            content = client.files.download(file=bj.dest.file_name)
            with open(out_path, "wb") as fh:
                fh.write(content)
            print(f"Saved output file: {out_path}")
            paths.append(out_path)
        elif bj.dest and getattr(bj.dest, "inlined_responses", None):
            out_path = os.path.join(save_dir, f"{job_name.replace('/','_')}_out.jsonl")
            with open(out_path, "w", encoding="utf-8") as fh:
                for idx, ir in enumerate(bj.dest.inlined_responses):
                    fh.write(json.dumps({"key": f"inline-{idx}", "response": ir.response}, default=lambda o: getattr(o, "__dict__", str(o))) + "\n")
            print(f"Saved inline output file: {out_path}")
            paths.append(out_path)

    return state, paths

def parse_batch_jsonl(
    jsonl_paths: List[str],
    descriptors: List[str],
    *,
    RATE_MIN: float, RATE_MAX: float,
    pairwise: bool
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Returns mapping (row_id, rep) -> validated scores dict.
    Expects keys like: ds|temp|bpt:{...}|row{row_id}|rep{rep}|...
    """
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}

    def _extract_text_from_response(resp: Dict[str, Any]) -> str:
        # Try the canonical content.parts text(s)
        try:
            parts = resp["candidates"][0]["content"]["parts"]
            texts = []
            for p in parts:
                if "text" in p and isinstance(p["text"], str):
                    texts.append(p["text"])
            return "\n".join(texts)
        except Exception:
            return resp.get("text", "") or ""

    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = rec.get("key", "")
                resp = rec.get("response", {})
                if not key or not resp:
                    continue

                parts = key.split("|")
                row_part = next((x for x in parts if x.startswith("row")), None)
                rep_part = next((x for x in parts if x.startswith("rep")), None)
                if row_part is None or rep_part is None:
                    continue
                row_id = int(row_part.replace("row", ""))
                rep = int(rep_part.replace("rep", ""))

                try:
                    content_text = _extract_text_from_response(resp)
                    if pairwise:
                        scores = validate_response_pairwise(content_text, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                    else:
                        scores = validate_response_single(content_text, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                except Exception as e:
                    scores = {"error": f"validate_error: {e}"}

                out[(row_id, rep)] = scores

    return out

# ---------- CSV writer ----------
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
    pairwise = is_pairwise_df(df)
    rows: List[Dict[str, Any]] = []

    for row_id, row in df.iterrows():
        if pairwise:
            cid1, cid2 = row_pair_cids(row)
            inp1, inp2 = row_pair_inputs(row, build_prompt_type) or ("", "")
            for rep in range(n_repeats):
                base = {
                    "participant_id": row.get("participant_id"),
                    "cid_stimulus_1": cid1,
                    "cid_stimulus_2": cid2,
                    f"{('name' if build_prompt_type=='byname' else INPUT_TYPE)} stimulus 1": inp1,
                    f"{('name' if build_prompt_type=='byname' else INPUT_TYPE)} stimulus 2": inp2,
                    "repeat": rep,
                    "temperature": temperature,
                    "model_name": model_name,
                    "build_prompt_type": build_prompt_type,
                }
                scores = rowrep_to_scores.get((row_id, rep), {"error": "missing_in_batch_output"})
                rows.append({**base, **scores})
        else:
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
                rows.append({**base, **scores})

    if pairwise:
        cols = [
            "participant_id",
            "cid_stimulus_1", "cid_stimulus_2",
            f"{('name' if build_prompt_type=='byname' else INPUT_TYPE)} stimulus 1",
            f"{('name' if build_prompt_type=='byname' else INPUT_TYPE)} stimulus 2",
            "repeat", "temperature", "model_name", "build_prompt_type",
            "similarity",
        ]
    else:
        cols = ["participant_id", "cid", "repeat", "temperature", INPUT_TYPE, "name", "model_name", "build_prompt_type"] + descriptors

    if INCLUDE_CONFIDENCE:
        cols.append("confidence")
    # include error column if any
    tmp_rows = rows if rows else [{}]
    if any("error" in r for r in tmp_rows):
        cols.append("error")

    out_df = pd.DataFrame(rows)
    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# ---------- CLI ----------
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
    s_submit.add_argument("--debug", action="store_true", help="Debug mode (no submission)")

    s_collect = sub.add_parser("collect", help="Fetch finished batch, parse, and write CSV")
    # s_collect.add_argument("--save_dir", required=True)

    args = ap.parse_args()

    if args.cmd == "submit":
        ds = args.ds
        RATE_MIN, RATE_MAX = get_rate_range(ds)
        input_csv = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"
        df = pd.read_csv(input_csv)
        if args.debug:
            df = df.head(2)

        pairwise = is_pairwise_df(df)
        if not pairwise:
            # SINGLE-ITEM: optional filters and CID dedup
            df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
            if "concentration" in df.columns and "keller" in ds.lower():
                df["concentration"] = df["concentration"].astype(float)
                df = df[df["concentration"] == 0.001].copy()
            cids = sorted(map(int, common_cids_per_ds(df)))
            df = df[df["cid"].isin(cids)].reset_index(drop=True)
            df = df.drop_duplicates(subset=["cid"]).reset_index(drop=True)
            descriptors = get_descriptors(ds)
        else:
            # PAIRWISE: no dedup, descriptors not used
            descriptors = []

        model_name = args.model_name
        build_prompt_type = args.build_prompt_type
        n_repeats = int(args.n_repeats)

        os.makedirs("tmp", exist_ok=True)
        jsonl_path = f"{BASE_DIR}/results/responses/tmp/{ds}_{model_name}_{args.temperature}_{build_prompt_type}_reps-{n_repeats}.jsonl"
        total, used_reqs = make_jsonl_for_batch(
            df=df,
            ds_name=ds,
            model_name=model_name,
            temperature=args.temperature,
            descriptors=descriptors,
            out_jsonl_path=jsonl_path,
            build_prompt_type=build_prompt_type,
            n_repeats=n_repeats,
            RATE_MIN=RATE_MIN, RATE_MAX=RATE_MAX
        )
        print(f"Prepared JSONL: {jsonl_path}  (rows={total}, requests={used_reqs})")

        job_name = submit_batch(jsonl_path, model_name=model_name, display_name=f"{ds}-{build_prompt_type}")
        print(f"BATCH_JOB_NAME={job_name}")
        log_batch_entry(ds, model_name, args.temperature, job_name, build_prompt_type, n_repeats)

    elif args.cmd == "collect":
        reg = load_registry()
        if not reg:
            print("Registry is empty. Nothing to collect.")
            return

        for entry in reg:
            ds = entry["ds"]
            RATE_MIN, RATE_MAX = get_rate_range(ds)
            input_csv = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"
            df = pd.read_csv(input_csv)
            pairwise = is_pairwise_df(df)

            if not pairwise:
                df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
                if "concentration" in df.columns and "keller" in ds.lower():
                    df["concentration"] = df["concentration"].astype(float)
                    df = df[df["concentration"] == 0.001].copy()
                cids = sorted(map(int, common_cids_per_ds(df)))
                df = df[df["cid"].isin(cids)].reset_index(drop=True)
                df = df.drop_duplicates(subset=["cid"]).reset_index(drop=True)
                descriptors = get_descriptors(ds)
            else:
                descriptors = []

            model_name = entry["model_name"]
            temp = entry["temperature"]
            job_name = entry["batch_id"]
            build_prompt_type = entry.get("build_prompt_type", "bysmiles")
            n_repeats = int(entry.get("n_repeats", 1))
            time = entry.get("time", "unknown")

            state, paths = download_batch_outputs(job_name)
            if state != "JOB_STATE_SUCCEEDED":
                print(f"Batch {job_name} not completed yet (state={state}). Skipping.")
                continue
            if not paths:
                print(f"No output file(s) for batch {job_name}. Skipping.")
                continue

            rowrep_to_scores = parse_batch_jsonl(paths, descriptors, RATE_MIN=RATE_MIN, RATE_MAX=RATE_MAX, pairwise=pairwise)
            os.makedirs(f"{BASE_DIR}/results/responses/llm_responses", exist_ok=True)
            output_csv = (
                f"{BASE_DIR}/results/responses/llm_responses/"
                f"{ds}_odor_llm_scores_temp-{temp}_model-{model_name}_bpt-{build_prompt_type}_reps-{n_repeats}_time-{time}.csv"
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
