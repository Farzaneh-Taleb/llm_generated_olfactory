from __future__ import annotations
import os, json, argparse, time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# --- Gemini ---
# pip install google-generativeai
import google.generativeai as genai

from utils.ds_utils import get_descriptors
from utils.config import BASE_DIR

# ---------------- Config ----------------
RATE_MIN, RATE_MAX = -1.0, 1.0
INCLUDE_CONFIDENCE = False
INPUT_TYPE = "isomericsmiles"         # 'isomericsmiles' or 'cid'
SYSTEM_MSG = (
    "You are an olfactory rater. "
    "Return ONLY a single valid JSON object. No prose, no markdown, no code fences."
)
BATCH_REGISTRY = f"{BASE_DIR}/llm_responses/batch_registry.jsonl"
BUILD_PROMPT_CHOICES = ("bysmiles", "byname")

# Configure Gemini (expects GEMINI_API_KEY in env)
GEMINI_MODEL_DEFAULT = "gemini-2.5-pro-preview-03-25"   # change if you prefer ultra/flash etc.
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY in environment.")
genai.configure(api_key=API_KEY)

# -------- Registry I/O (kept for parity) --------
def log_batch_entry(ds: str, model_name: str, temp: float, batch_id: str, build_prompt_type: str):
    os.makedirs(os.path.dirname(BATCH_REGISTRY), exist_ok=True)
    entry = {
        "ds": ds,
        "model_name": model_name,
        "temperature": temp,
        "batch_id": batch_id,
        "build_prompt_type": build_prompt_type,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(BATCH_REGISTRY, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

def load_registry() -> list[dict[str, Any]]:
    if not os.path.exists(BATCH_REGISTRY):
        return []
    with open(BATCH_REGISTRY, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

# -------- Helpers --------
def _clean_cell(x) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
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

# ---------- Gemini call ----------
def gemini_generate_json(
    prompt: str,
    model_name: str,
    temperature: float,
    system_msg: str,
) -> str:
    """
    Calls Gemini and asks for JSON-only output.
    Returns raw text (JSON string). Raises on obvious API/format errors.
    """
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=system_msg,
        generation_config={
            "temperature": float(temperature),
            # Forces JSON-only output (no markdown, no prose)
            "response_mime_type": "application/json",
        },
        # (Optional) You can add safety_settings here if you need custom thresholds
    )
    resp = model.generate_content(prompt)
    # resp.text should be a JSON string per response_mime_type
    if not hasattr(resp, "text") or not resp.text:
        raise ValueError("Empty response from Gemini.")
    return resp.text

# ---------- RUN (row-by-row, preserves duplicates) ----------
def run_immediate(
    df: pd.DataFrame,
    ds_name: str,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    out_csv: str,
    build_prompt_type: str,
) -> None:
    """
    Processes every original row in-order and writes a single CSV with results.
    Repeats CIDs exactly as they appear in the input (no dedup).
    """
    df = df.copy().reset_index(drop=True)
    rows: List[Dict[str, Any]] = []

    for row_id, r in df.iterrows():
        print(f"Processing row {row_id+1}/{len(df)} (cid={r.get('cid')})...")   
        cid = pd.to_numeric(r.get("cid"), errors="coerce")
        cid = int(cid) if not pd.isna(cid) else None
        smi = row_smiles(r) or ""
        nm = row_name(r) or ""

        # Build the appropriate prompt
        if build_prompt_type == "bysmiles":
            if not smi:
                scores = {"error": "missing_smiles"}
            else:
                prompt = build_prompt(smi, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                try:
                    raw = gemini_generate_json(
                        prompt=prompt,
                        model_name=model_name,
                        temperature=temperature,
                        system_msg=SYSTEM_MSG,
                    )
                    scores = validate_response(
                        raw, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE
                    )
                except Exception as e:
                    scores = {"error": f"gemini_error: {e}"}
        else:
            if not nm:
                scores = {"error": "missing_name"}
            else:
                prompt = build_prompt_byname(nm, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                try:
                    raw = gemini_generate_json(
                        prompt=prompt,
                        model_name=model_name,
                        temperature=temperature,
                        system_msg=SYSTEM_MSG,
                    )
                    scores = validate_response(
                        raw, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE
                    )
                except Exception as e:
                    scores = {"error": f"gemini_error: {e}"}

        base = {
            "participant_id": r.get("participant_id"),
            "cid": cid,
            "temperature": temperature,
            INPUT_TYPE: smi,
            "name": nm,
            "model_name": model_name,
            "build_prompt_type": build_prompt_type,
        }
        rows.append({**base, **scores})

    cols = ["participant_id", "cid", "temperature", INPUT_TYPE, "name", "model_name", "build_prompt_type"] + descriptors
    if INCLUDE_CONFIDENCE:
        cols.append("confidence")
    if any("error" in row for row in rows):
        cols.append("error")

    out_df = pd.DataFrame(rows)
    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

# ---------- (Optional) JSONL maker kept for parity ----------
def make_jsonl_for_batch(
    df: pd.DataFrame,
    ds_name: str,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    out_jsonl_path: str,
    build_prompt_type: str,
) -> tuple[int, int]:
    """
    Keeps your JSONL creation for reference or future Gemini Batch use.
    Currently, this is *not* submitted to a Gemini batch endpoint in this script.
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    df = df.copy().reset_index(drop=True)
    df["cid"] = pd.to_numeric(df["cid"], errors="coerce").astype("Int64")

    total_rows = len(df)
    used_rows = 0

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

            # Store the content we would send to Gemini
            line = {
                "custom_id": f"{ds_name}|{temperature}|bpt:{build_prompt_type}|row{row_id}|cid{cid}",
                "gemini_request": {
                    "model": model_name,
                    "system_instruction": SYSTEM_MSG,
                    "generation_config": {
                        "temperature": float(temperature),
                        "response_mime_type": "application/json",
                    },
                    "prompt": prompt,
                },
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
            used_rows += 1

    return total_rows, used_rows

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # Direct run with Gemini (recommended)
    s_run = sub.add_parser("run", help="Process rows now with Gemini and write CSV")
    s_run.add_argument("--ds", required=True, help="Dataset key for get_descriptors() and path building")
    s_run.add_argument("--temperature", type=float, default=0.0)
    s_run.add_argument("--model_name", default=GEMINI_MODEL_DEFAULT, help="Gemini model name, e.g., gemini-1.5-pro")
    s_run.add_argument("--build-prompt-type", dest="build_prompt_type",
                       choices=BUILD_PROMPT_CHOICES, default="bysmiles",
                       help="Prompt construction mode: bysmiles or byname")

    # Keep parity with your old flow (JSONL maker only)
    s_submit = sub.add_parser("submit", help="Create JSONL for potential batch usage (not submitted)")
    s_submit.add_argument("--ds", required=True)
    s_submit.add_argument("--temperature", type=float, default=0.0)
    s_submit.add_argument("--model_name", default=GEMINI_MODEL_DEFAULT)
    s_submit.add_argument("--build-prompt-type", dest="build_prompt_type",
                          choices=BUILD_PROMPT_CHOICES, default="bysmiles")

    s_collect = sub.add_parser("collect", help="(Not implemented) Placeholder for batch collection")
    s_collect.add_argument("--save_dir", required=False, help="Unused here")

    args = ap.parse_args()

    if args.cmd == "run":
        ds = args.ds
        model_name = args.model_name
        build_prompt_type = args.build_prompt_type
        descriptors = get_descriptors(ds)
        input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
        output_csv = f"{BASE_DIR}/llm_responses/{ds}_odor_llm_scores_temp-{args.temperature}_model-{model_name}_bpt-{build_prompt_type}.csv"
        df = pd.read_csv(input_csv)

        run_immediate(
            df=df,
            ds_name=ds,
            model_name=model_name,
            temperature=args.temperature,
            descriptors=descriptors,
            out_csv=output_csv,
            build_prompt_type=build_prompt_type,
        )

    elif args.cmd == "submit":
        ds = args.ds
        model_name = args.model_name
        build_prompt_type = args.build_prompt_type
        descriptors = get_descriptors(ds)
        input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
        df = pd.read_csv(input_csv)
        jsonl_path = f"/tmp/{ds}_{model_name}_{args.temperature}_{build_prompt_type}_gemini.jsonl"

        total, used_rows = make_jsonl_for_batch(
            df=df,
            ds_name=ds,
            model_name=model_name,
            temperature=args.temperature,
            descriptors=descriptors,
            out_jsonl_path=jsonl_path,
            build_prompt_type=build_prompt_type,
        )
        print(f"Prepared JSONL: {jsonl_path}  (rows={total}, used_rows={used_rows})")
        print("Note: This script does NOT submit a Gemini batch job. Use `run` to execute now.")

    elif args.cmd == "collect":
        raise NotImplementedError(
            "Gemini batch collect is not wired here. Use `run` to execute immediately.\n"
            "If you need true batch/offline processing, we can adapt this to Google AI Batch Jobs or Vertex AI."
        )

if __name__ == "__main__":
    main()
