from __future__ import annotations
import os, json, argparse, time
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd
from openai import OpenAI

from utils.helpers import *
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import *

# ---------------- Config ----------------

client = OpenAI()  # uses OPENAI_API_KEY from env


# ---------- BATCH: PREP & SUBMIT ----------
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
    Write one /v1/chat/completions request per row (deduped by CID for single-item datasets),
    repeated n_repeats times.
    custom_id encodes ds|temp|bpt|row{row_id}|rep{rep}|meta...
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    df = df.copy().reset_index(drop=True)
    pairwise = is_pairwise_df(df)

    total_rows = len(df)
    used_reqs = 0

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
                # Single-item (descriptor ratings)
                if build_prompt_type == "bysmiles":
                    smi = row_smiles(r)
                    if not smi:
                        continue
                    prompt = build_prompt_bysmiles(smi, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=False)
                else:
                    name = row_name(r)
                    if not name:
                        continue
                    prompt = build_prompt_byname(name, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=False)

                cid = pd.to_numeric(r.get("cid"), errors="coerce")
                meta_suffix = f"|cid{int(cid)}" if pd.notna(cid) else ""

            for rep in range(n_repeats):
                line = {
                    "custom_id": f"{ds_name}|{temperature}|bpt:{build_prompt_type}|row{row_id}|rep{rep}{meta_suffix}",
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model_name,
                        "messages": [
                            {"role": "system", "content": SYSTEM_MSG},
                            {"role": "user", "content": prompt},
                        ],
                        "temperature": temperature,
                        "response_format": {"type": "json_object"},
                    },
                }
                f.write(json.dumps(line, ensure_ascii=False) + "\n")
                used_reqs += 1

    return total_rows, used_reqs

def submit_batch(jsonl_path: str) -> str:
    up = client.files.create(file=open(jsonl_path, "rb"), purpose="batch")
    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
    )
    print(f"Submitted batch: {batch.id}  (input_file_id={up.id})")
    return batch.id

# ---------- BATCH: COLLECT & MERGE ----------
def download_batch_outputs(
    batch_id: str,
    save_dir: str,
    poll_seconds: int = 20,
    poll_interval: float = 1.0,
) -> Tuple[str, List[str]]:
    try:
        b = client.batches.retrieve(batch_id)
    except Exception as e:
        print(f"Error retrieving batch {batch_id}: {e}")
        return "error", []
    state = b.status
    print(f"Batch {batch_id} status: {state}")

    rc = getattr(b, "request_counts", None)
    if rc:
        print(f"request_counts: {rc}")

    os.makedirs(save_dir, exist_ok=True)

    def _save_file(file_id: str, suffix: str) -> str:
        content = client.files.content(file_id)
        path = os.path.join(save_dir, f"{batch_id}_{suffix}.jsonl")
        with open(path, "wb") as fh:
            fh.write(content.read())
        print(f"Saved {suffix} file: {path}")
        return path

    if state not in {"completed", "failed", "cancelled", "expired"}:
        return state, []

    out_id = getattr(b, "output_file_id", None)
    err_id = getattr(b, "error_file_id", None)

    if state == "completed" and not out_id:
        deadline = time.time() + poll_seconds
        while time.time() < deadline and not out_id:
            time.sleep(poll_interval)
            b = client.batches.retrieve(batch_id)
            out_id = getattr(b, "output_file_id", None)
            err_id = getattr(b, "error_file_id", None)
            if out_id:
                break

    paths: List[str] = []
    if out_id:
        paths.append(_save_file(out_id, "out"))
    if err_id:
        paths.append(_save_file(err_id, "errors"))

    if not paths:
        print("No output_file_id or error_file_id available.")
        return state, []

    return state, paths

def parse_chat_completion_jsonl_by_rowrep(
    jsonl_paths: List[str],
    descriptors: List[str],
    *,
    RATE_MIN: float, RATE_MAX: float,
    pairwise: bool
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Returns mapping (row_id, rep) -> validated scores dict.
    Expects custom_id like: ds|temp|bpt:{...}|row{row_id}|rep{rep}|...
    """
    out: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                custom_id = rec.get("custom_id", "")
                parts = custom_id.split("|")

                row_part = next((x for x in parts if x.startswith("row")), None)
                rep_part = next((x for x in parts if x.startswith("rep")), None)
                if row_part is None or rep_part is None:
                    continue

                row_id = int(row_part.replace("row", ""))
                rep = int(rep_part.replace("rep", ""))

                body = rec.get("response", {}).get("body", {})
                content = (
                    body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                ) or ""
                try:
                    if pairwise:
                        scores = validate_response_pairwise(
                            content, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE
                        )
                    else:
                        scores = validate_response_single(
                            content, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE
                        )
                except Exception as e:
                    scores = {"error": f"validate_error: {e}"}

                out[(row_id, rep)] = scores
    return out

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
                rec = {**base, **scores}
                rows.append(rec)
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
                rec = {**base, **scores}
                rows.append(rec)

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

    s_submit = sub.add_parser("submit", help="Create JSONL and submit Batch")
    s_submit.add_argument("--ds", required=True, help="Dataset key for get_descriptors() and path building")
    s_submit.add_argument("--temperature", type=float, default=0.0)
    s_submit.add_argument("--model_name", required=True, help="Model name to use for the batch")
    s_submit.add_argument("--build-prompt-type", dest="build_prompt_type",
                          choices=BUILD_PROMPT_CHOICES, default="bysmiles",
                          help="Prompt construction mode: bysmiles or byname")
    s_submit.add_argument("--n-repeats", type=int, default=1, help="Repeat each item this many times")

    s_collect = sub.add_parser("collect", help="Fetch finished batch, parse, and write CSV")
    s_collect.add_argument("--save_dir", required=True, help="Where to save batch output .jsonl")

    args = ap.parse_args()

    if args.cmd == "submit":
        ds = args.ds
        RATE_MIN, RATE_MAX = get_rate_range(ds)
        input_csv = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"
        df = pd.read_csv(input_csv)

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
            # PAIRWISE: no dedup by CID; descriptors not used
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

        batch_id = submit_batch(jsonl_path)
        print(f"BATCH_ID={batch_id}")
        log_batch_entry(ds, model_name, args.temperature, batch_id, build_prompt_type, n_repeats)

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
            batch_id = entry["batch_id"]
            build_prompt_type = entry.get("build_prompt_type", "bysmiles")
            n_repeats = int(entry.get("n_repeats", 1))

            state, paths = download_batch_outputs(batch_id, args.save_dir)
            if state != "completed":
                print(f"Batch {batch_id} not completed yet (state={state}). Skipping.")
                continue
            if not paths:
                print(f"No output file(s) for batch {batch_id}. Skipping.")
                continue

            out_paths = [p for p in paths if p.endswith("_out.jsonl")]
            err_paths = [p for p in paths if p.endswith("_errors.jsonl")]

            if err_paths and not out_paths:
                print(f"Batch {batch_id} produced only errors. Inspect: {err_paths[0]}")
            if not out_paths:
                continue

            rowrep_to_scores = parse_chat_completion_jsonl_by_rowrep(
                paths, descriptors, RATE_MIN=RATE_MIN, RATE_MAX=RATE_MAX, pairwise=pairwise
            )
            os.makedirs(f"{BASE_DIR}/results/responses/llm_responses", exist_ok=True)
            output_csv = (
                f"{BASE_DIR}/results/responses/llm_responses/"
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
