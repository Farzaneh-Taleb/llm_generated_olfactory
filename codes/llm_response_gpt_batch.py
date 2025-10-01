from __future__ import annotations
import os, json, argparse, time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from openai import OpenAI

from utils.helpers import common_cids_per_ds
from utils.ds_utils import get_descriptors
from utils.config import BASE_DIR

# ---------------- Config ----------------
RATE_MIN, RATE_MAX = -1.0, 1.0
INCLUDE_CONFIDENCE = False
INPUT_TYPE = "isomericsmiles"         # 'isomericsmiles' or 'cid'
SYSTEM_MSG = "You are an olfactory rater. Output ONLY valid JSON."
BATCH_REGISTRY = f"{BASE_DIR}/llm_responses/batch_registry.jsonl"

BUILD_PROMPT_CHOICES = ("bysmiles", "byname")

client = OpenAI()  # uses OPENAI_API_KEY from env

# -------- Registry I/O --------
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
        "batch_id": batch_id,
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

# -------- Helpers --------
def _clean_cell(x) -> str:
    if x is None:
        return ""
    try:
        import pandas as _pd  # local to avoid accidental shadowing
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

def build_prompt(
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
    return f"""Molecule:
- ISOMERIC SMILES: {smiles}

Descriptors (rate each from {rate_min} to {rate_max}): [{desc_list}]

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must match the descriptor list exactly.
- Values must be numbers in [{rate_min},{rate_max}].
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""

def build_prompt_byname(
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
    return f"""Molecule:
- Name: {name}

Descriptors (rate each from {rate_min} to {rate_max}): [{desc_list}]

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must match the descriptor list exactly.
- Values must be numbers in [{rate_min},{rate_max}].
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""

def validate_response(
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
) -> tuple[int, int]:
    """
    Write one /v1/chat/completions request PER DEDUPED CID row, repeated n_repeats times.
    custom_id encodes ds|temp|bpt|row{row_id}|rep{rep}|cid{cid}.
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    df = df.copy().reset_index(drop=True)
    df["cid"] = pd.to_numeric(df["cid"], errors="coerce").astype("Int64")

    total_rows = len(df)
    used_reqs = 0

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
                name = row_name(r)
                if not name:
                    continue
                prompt = build_prompt_byname(name, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)

            for rep in range(n_repeats):
                line = {
                    "custom_id": f"{ds_name}|{temperature}|bpt:{build_prompt_type}|row{row_id}|rep{rep}|cid{cid}",
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
    """
    Uploads the JSONL and creates a batch with 24h window.
    Returns batch_id.
    """
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
    """
    Robustly download batch artifacts.

    Returns:
        (state, paths)
        - If successes exist: includes "<batch_id>_out.jsonl"
        - If any failures exist: also includes "<batch_id>_errors.jsonl"
        - If only failures exist: returns just the errors file
        - If nothing downloadable yet: returns []
    """
    b = client.batches.retrieve(batch_id)
    state = b.status  # "queued" | "in_progress" | "completed" | "failed" | "cancelled" | "expired"
    print(f"Batch {batch_id} status: {state}")

    # Useful diagnostics
    rc = getattr(b, "request_counts", None)
    if rc:
        # Example: {'total': 100, 'completed': 92, 'failed': 8}
        print(f"request_counts: {rc}")

    os.makedirs(save_dir, exist_ok=True)

    def _save_file(file_id: str, suffix: str) -> str:
        content = client.files.content(file_id)
        path = os.path.join(save_dir, f"{batch_id}_{suffix}.jsonl")
        with open(path, "wb") as fh:
            fh.write(content.read())
        print(f"Saved {suffix} file: {path}")
        return path

    # Non-terminal -> nothing to download yet
    if state not in {"completed", "failed", "cancelled", "expired"}:
        return state, []

    out_id = getattr(b, "output_file_id", None)
    err_id = getattr(b, "error_file_id", None)

    # Completed but no output yet: short poll (API can lag in populating file IDs)
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
        # Save errors too (even when completed) so you can inspect partial failures.
        paths.append(_save_file(err_id, "errors"))

    if not paths:
        print("No output_file_id or error_file_id available.")
        return state, []

    return state, paths

def parse_chat_completion_jsonl_by_rowrep(
    jsonl_paths: List[str],
    descriptors: List[str],
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Returns mapping (row_id, rep) -> validated scores dict.
    Expects custom_id like: ds|temp|bpt:{...}|row{row_id}|rep{rep}|cid{cid}
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
                    scores = validate_response(content, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
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
    if INCLUDE_CONFIDENCE:
        cols.append("confidence")
    if any("error" in r for r in rows):
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
    s_submit.add_argument("--n-repeats", type=int, default=1, help="Repeat each deduped CID this many times")

    s_collect = sub.add_parser("collect", help="Fetch finished batch, parse, and write CSV")
    s_collect.add_argument("--save_dir", required=True, help="Where to save batch output .jsonl")

    args = ap.parse_args()

    if args.cmd == "submit":
        ds = args.ds
        model_name = args.model_name
        build_prompt_type = args.build_prompt_type
        n_repeats = int(args.n_repeats)
        descriptors = get_descriptors(ds)

        input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
        df = pd.read_csv(input_csv)

        # Keep only common CIDs (using your helper) and deduplicate by CID
        cids = set(common_cids_per_ds(df))
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce").astype("Int64")
        df = df[df["cid"].isin(cids)].reset_index(drop=True)
        df = df.drop_duplicates(subset=["cid"]).reset_index(drop=True)

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

        batch_id = submit_batch(jsonl_path)
        print(f"BATCH_ID={batch_id}")
        log_batch_entry(ds, model_name, args.temperature, batch_id, build_prompt_type, n_repeats)

    elif args.cmd == "collect":
        reg = load_registry()
        for entry in reg:
            ds = entry["ds"]
            model_name = entry["model_name"]
            temp = entry["temperature"]
            batch_id = entry["batch_id"]
            build_prompt_type = entry.get("build_prompt_type", "bysmiles")
            n_repeats = int(entry.get("n_repeats", 1))

            descriptors = get_descriptors(ds)
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
                continue  # nothing to parse for scores

            rowrep_to_scores = parse_chat_completion_jsonl_by_rowrep(paths, descriptors)
            output_csv = (
                f"{BASE_DIR}/llm_responses/"
                f"{ds}_odor_llm_scores_temp-{temp}_model-{model_name}_bpt-{build_prompt_type}_reps-{n_repeats}.csv"
            )

            # Prepare the same deduped, common-CID dataframe as in 'submit'
            input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
            df = pd.read_csv(input_csv)
            cids = set(common_cids_per_ds(df))
            df["cid"] = pd.to_numeric(df["cid"], errors="coerce").astype("Int64")
            df = df[df["cid"].isin(cids)].reset_index(drop=True)
            df = df.drop_duplicates(subset=["cid"]).reset_index(drop=True)

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
