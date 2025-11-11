# llm_response_gemini_sync.py
from __future__ import annotations
import os, json, argparse, time, sys
from typing import Any, Dict, List, Optional, Tuple, Union

import pandas as pd

# --- Gemini SDK ---
from google import genai
from google.genai import types

from utils.helpers import common_cids_per_ds
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import BASE_DIR

# ---------------- Config ----------------
RATE_RANGE = {
    "keller2016": (0.0, 100.0),
    "sagar2023": (-1, 1),
    "leffingwell": (0.0, 1.0),
    "ravia": (0.0, 1.0),
    "snitz2013": (0.0, 1.0),
}
INCLUDE_CONFIDENCE = False
INPUT_TYPE = "isomericsmiles"  # 'isomericsmiles' | 'canonicalsmiles' | 'isomericselfies' | 'canonicalselfies' | 'cid'
SYSTEM_MSG = "You are an olfactory rater. Output ONLY valid JSON."
BUILD_PROMPT_CHOICES = ("bysmiles", "byname")

# Pairwise column names
PAIR_CID1 = "cid stimulus 1"
PAIR_CID2 = "cid stimulus 2"
PAIR_SIMILARITY = "similarity"   # human ref (not used in prompt)
PAIR_NAME1 = "name stimulus 1"
PAIR_NAME2 = "name stimulus 2"

# -------- Gemini client (reads API key from env) --------
def _make_client() -> genai.Client:
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing API key. Set GEMINI_API_KEY or GOOGLE_API_KEY in your environment."
        )
    return genai.Client(api_key=api_key)

client = _make_client()

# ---------------- Utilities ----------------
def get_rate_range(ds: str) -> tuple[float, float]:
    if ds not in RATE_RANGE:
        raise ValueError(f"Unknown dataset: {ds}. Please add it to RATE_RANGE.")
    return RATE_RANGE[ds]

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

# ------------ Single-odorant helpers ------------
def row_smiles(row: pd.Series) -> Optional[str]:
    smi = _clean_cell(row.get(INPUT_TYPE, ""))
    return smi or None

def row_name(row: pd.Series) -> Optional[str]:
    nm = _clean_cell(row.get("name", ""))
    return nm or None

# ------------ Pairwise detection & helpers ------------
def pair_input_col(which: int, input_type: str) -> str:
    assert which in (1, 2)
    return f"{input_type} stimulus {which}"

def is_pairwise_df(df: pd.DataFrame) -> bool:
    has_sim = PAIR_SIMILARITY in df.columns
    if not has_sim:
        return False
    # if similarity exists and we have at least one of the expected input ID/name/SMILES pairs → pairwise
    has_pair_ids = (PAIR_CID1 in df.columns and PAIR_CID2 in df.columns)
    has_pair_names = (PAIR_NAME1 in df.columns and PAIR_NAME2 in df.columns)
    has_pair_inputs = (
        pair_input_col(1, INPUT_TYPE) in df.columns and
        pair_input_col(2, INPUT_TYPE) in df.columns
    )
    return has_pair_ids or has_pair_names or has_pair_inputs

def row_pair_inputs(row: pd.Series, build_prompt_type: str) -> Optional[Tuple[str, str]]:
    if build_prompt_type == "bysmiles":
        col1 = pair_input_col(1, INPUT_TYPE)
        col2 = pair_input_col(2, INPUT_TYPE)
    else:
        col1 = PAIR_NAME1
        col2 = PAIR_NAME2
    a = _clean_cell(row.get(col1, ""))
    b = _clean_cell(row.get(col2, ""))
    if not a or not b:
        return None
    return a, b

def row_pair_cids(row: pd.Series) -> Tuple[Optional[int], Optional[int]]:
    c1 = pd.to_numeric(row.get(PAIR_CID1), errors="coerce")
    c2 = pd.to_numeric(row.get(PAIR_CID2), errors="coerce")
    c1 = int(c1) if pd.notna(c1) else None
    c2 = int(c2) if pd.notna(c2) else None
    return c1, c2

# ---------- Prompt builders (single-item) ----------
def build_prompt_single(
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

def build_prompt_byname_single(
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

# ---------- Prompt builders (pairwise similarity) ----------
def build_prompt_pairwise(
    smiles_1: str,
    smiles_2: str,
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
) -> str:
    json_lines = [f'  "similarity": <{rate_min}-{rate_max}>']
    if include_confidence:
        json_lines.append('  "confidence": <0-1>')
    json_block = "{\n" + ",\n".join(json_lines) + "\n}"
    return f"""Two Molecules:
- ISOMERIC SMILES (Stimulus 1): {smiles_1}
- ISOMERIC SMILES (Stimulus 2): {smiles_2}

Similarity (rate from {rate_min} to {rate_max}):
- Provide a single continuous similarity rating; higher means more similar.

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must be exactly: "similarity".
- Values must be numbers in [{rate_min},{rate_max}]{' (and confidence in [0,1])' if include_confidence else ''}.
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""

def build_prompt_byname_pairwise(
    name_1: str,
    name_2: str,
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
) -> str:
    json_lines = [f'  "similarity": <{rate_min}-{rate_max}>']
    if include_confidence:
        json_lines.append('  "confidence": <0-1>')
    json_block = "{\n" + ",\n".join(json_lines) + "\n}"
    return f"""Two Molecules:
- Name (Stimulus 1): {name_1}
- Name (Stimulus 2): {name_2}

Similarity (rate from {rate_min} to {rate_max}):
- Provide a single continuous similarity rating; higher means more similar.

Output rules:
- Return ONLY a single valid JSON object. No prose, no markdown.
- Keys must be exactly: "similarity".
- Values must be numbers in [{rate_min},{rate_max}]{' (and confidence in [0,1])' if include_confidence else ''}.
- Do NOT add extra keys{' except "confidence"' if include_confidence else ''}.

Output format:
{json_block}
"""

# ---------- Unified dispatchers ----------
def build_prompt_dispatch(
    x: Union[str, Tuple[str, str]],
    descriptors: List[str],
    rate_min: float,
    rate_max: float,
    include_confidence: bool = False,
    *,
    pairwise: bool = False,
    byname: bool = False,
) -> str:
    if pairwise:
        if not (isinstance(x, tuple) and len(x) == 2):
            raise ValueError("For pairwise=True, pass a tuple (stimulus_1, stimulus_2).")
        a, b = x
        return (
            build_prompt_byname_pairwise(a, b, rate_min, rate_max, include_confidence)
            if byname
            else build_prompt_pairwise(a, b, rate_min, rate_max, include_confidence)
        )
    else:
        if isinstance(x, tuple):
            raise ValueError("For single-item prompts, pass a single string (SMILES or name).")
        return (
            build_prompt_byname_single(x, descriptors, rate_min, rate_max, include_confidence)
            if byname
            else build_prompt_single(x, descriptors, rate_min, rate_max, include_confidence)
        )

# ------------ Validators ------------
def validate_response_single(
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

def validate_response_pairwise(
    resp_text: str,
    rate_min: float,
    rate_max: float,
    include_confidence: bool,
) -> Dict[str, float]:
    obj = json.loads(resp_text)
    if "similarity" not in obj:
        raise ValueError('Missing key: "similarity"')
    try:
        sim = float(obj["similarity"])
    except Exception:
        raise ValueError(f"Non-numeric value for 'similarity': {obj.get('similarity')!r}")
    out: Dict[str, float] = {"similarity": max(rate_min, min(rate_max, sim))}
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

# --------- Structured output schema for Gemini ---------
def make_response_schema(
    descriptors: list[str],
    include_confidence: bool,
    RATE_MIN: float,
    RATE_MAX: float,
    *,
    pairwise: bool = False
) -> dict:
    """
    Build a minimal Gemini-compatible JSON schema.
    Gemini supports only simple JSON schema features:
    - "type"
    - "properties"
    - "required"
    - "minimum"/"maximum"
    """
    if pairwise:
        props = {
            "similarity": {
                "type": "number",
                "minimum": RATE_MIN,
                "maximum": RATE_MAX,
            }
        }
        required = ["similarity"]
    else:
        props = {
            d: {
                "type": "number",
                "minimum": RATE_MIN,
                "maximum": RATE_MAX,
            }
            for d in descriptors
        }
        required = list(descriptors)

    if include_confidence:
        props["confidence"] = {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        }

    return {
        "type": "object",
        "properties": props,
        "required": required,
    }

# --------- Gemini call helpers (sync) ---------
def _extract_text(resp) -> str:
    # Prefer .text if available (SDK parses parts for us)
    try:
        if getattr(resp, "text", None):
            return resp.text
    except Exception:
        pass
    # Try candidates/parts
    try:
        parts = resp.candidates[0].content.parts
        texts = []
        for p in parts:
            if getattr(p, "text", None):
                texts.append(p.text)
        if texts:
            return "\n".join(texts)
    except Exception:
        pass
    return ""

def call_gemini_json(prompt: str, model_name: str, temperature: float, schema: dict) -> str:
    cfg = types.GenerateContentConfig(
        temperature=float(temperature),
        response_mime_type="application/json",
        response_schema=schema,
        # 👇 move system prompt into the config
        system_instruction=SYSTEM_MSG,
    )
    resp = client.models.generate_content(
        model=model_name,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=cfg,
    )
    text = _extract_text(resp)
    if not text:
        raise RuntimeError("Empty response from Gemini.")
    return text.strip()


def with_retries(fn, *, retries:int=3, base_delay:float=1.5):
    last_exc = None
    for i in range(retries):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            sleep_s = base_delay * (2 ** i)
            time.sleep(sleep_s)
    raise last_exc if last_exc else RuntimeError("Unknown error after retries")

# ---------- Main runner (sync) ----------
def run_sync(
    ds: str,
    model_name: str,
    temperature: float,
    build_prompt_type: str,
    n_repeats: int,
    max_rows: Optional[int] = None,
    flush_every: int = 50,
):
    RATE_MIN, RATE_MAX = get_rate_range(ds)
    input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
    df = pd.read_csv(input_csv)
    # df = df.head(2)
    out_csv = f"{BASE_DIR}/llm_responses/{ds}_odor_llm_scores_temp-{temperature}_model-{model_name}_bpt-{build_prompt_type}_reps-{n_repeats}.csv"

    pairwise = is_pairwise_df(df)
    if not pairwise:
        # SINGLE-ITEM: optional filters and CID dedup
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
        if "concentration" in df.columns and "keller" in ds.lower():
            df["concentration"] = pd.to_numeric(df["concentration"], errors="coerce")
            df = df[df["concentration"] == 0.001].copy()
        cids = sorted(map(int, common_cids_per_ds(df)))
        df = df[df["cid"].isin(cids)].drop_duplicates(subset=["cid"]).reset_index(drop=True)
        descriptors = get_descriptors(ds)
    else:
        descriptors = []  # not used in pairwise prompting

    schema = make_response_schema(descriptors, INCLUDE_CONFIDENCE, RATE_MIN, RATE_MAX, pairwise=pairwise)
    byname = (build_prompt_type == "byname")

    rows_buffer: List[Dict[str, Any]] = []
    header_written = os.path.exists(out_csv) and os.path.getsize(out_csv) > 0

    total = len(df) if max_rows is None else min(len(df), max_rows)

    for idx in range(total):
        print(f"[{idx+1}/{total}] Processing row {idx}...", file=sys.stderr)
        row = df.iloc[idx]
        base_common = {
            "participant_id": row.get("participant_id"),
            "repeat": None,  # filled below
            "temperature": temperature,
            "model_name": model_name,
            "build_prompt_type": build_prompt_type,
        }

        # Prepare prompt input
        try:
            if pairwise:
                pair_inputs = row_pair_inputs(row, build_prompt_type)
                if not pair_inputs:
                    raise ValueError("Missing pair inputs")
                x = pair_inputs  # tuple
            else:
                x = (row_name(row) if byname else row_smiles(row))
                if not x:
                    raise ValueError("Missing single input")
        except Exception as e:
            # write error rows for all repeats
            for rep in range(n_repeats):
                rec = base_common.copy()
                rec["repeat"] = rep
                if pairwise:
                    cid1, cid2 = row_pair_cids(row)
                    rec["cid_stimulus_1"], rec["cid_stimulus_2"] = cid1, cid2
                    a, b = row_pair_inputs(row, build_prompt_type) or ("", "")
                    if byname:
                        rec[f"name stimulus 1"] = a
                        rec[f"name stimulus 2"] = b
                    else:
                        rec[f"{INPUT_TYPE} stimulus 1"] = a
                        rec[f"{INPUT_TYPE} stimulus 2"] = b
                else:
                    cid = pd.to_numeric(row.get("cid"), errors="coerce")
                    rec["cid"] = int(cid) if pd.notna(cid) else None
                    rec[INPUT_TYPE] = row_smiles(row) or ""
                    rec["name"] = row_name(row) or ""
                rec["error"] = f"input_error: {e}"
                rows_buffer.append(rec)
            continue

        # Call Gemini n_repeats times
        for rep in range(n_repeats):
            prompt = build_prompt_dispatch(
                x=x,
                descriptors=descriptors,
                rate_min=RATE_MIN,
                rate_max=RATE_MAX,
                include_confidence=INCLUDE_CONFIDENCE,
                pairwise=pairwise,
                byname=byname,
            )

            def _do_call():
                return call_gemini_json(prompt, model_name=model_name, temperature=temperature, schema=schema)

            try:
                resp_text = with_retries(_do_call, retries=3, base_delay=1.2)
                if pairwise:
                    scores = validate_response_pairwise(resp_text, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                else:
                    scores = validate_response_single(resp_text, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
            except Exception as e:
                scores = {"error": f"gen_or_validate_error: {e}"}

            rec = base_common.copy()
            rec["repeat"] = rep

            if pairwise:
                cid1, cid2 = row_pair_cids(row)
                rec["cid_stimulus_1"], rec["cid_stimulus_2"] = cid1, cid2
                a, b = row_pair_inputs(row, build_prompt_type) or ("", "")
                if byname:
                    rec["name stimulus 1"] = a
                    rec["name stimulus 2"] = b
                else:
                    rec[f"{INPUT_TYPE} stimulus 1"] = a
                    rec[f"{INPUT_TYPE} stimulus 2"] = b
                rec.update(scores)
            else:
                cid = pd.to_numeric(row.get("cid"), errors="coerce")
                rec["cid"] = int(cid) if pd.notna(cid) else None
                rec[INPUT_TYPE] = row_smiles(row) or ""
                rec["name"] = row_name(row) or ""
                rec.update(scores)

            rows_buffer.append(rec)

        # Flush periodically
        if len(rows_buffer) >= flush_every:
            header_written = _flush_buffer(rows_buffer, out_csv, header_written)

        time.sleep(0.02)  # small delay to avoid bursts

    # Final flush
    if rows_buffer:
        _flush_buffer(rows_buffer, out_csv, header_written)

def _flush_buffer(rows_buffer: List[Dict[str, Any]], out_csv: str, header_written: bool) -> bool:
    df_tmp = pd.DataFrame(rows_buffer)
    # Keep natural column order for the first write
    cols = list(df_tmp.columns)
    df_tmp = df_tmp[cols]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df_tmp.to_csv(out_csv, mode=("a" if header_written else "w"), index=False, header=not header_written)
    rows_buffer.clear()
    return True  # header is written after first flush

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_run = sub.add_parser("run", help="Synchronous Gemini calls row-by-row; writes CSV incrementally")
    s_run.add_argument("--ds", required=True)
    s_run.add_argument("--model_name", required=True, help="Gemini model, e.g. 'gemini-2.0-flash'")
    s_run.add_argument("--temperature", type=float, default=0.0)
    s_run.add_argument("--build-prompt-type", dest="build_prompt_type",
                       choices=BUILD_PROMPT_CHOICES, default="bysmiles")
    s_run.add_argument("--n-repeats", type=int, default=1)
    s_run.add_argument("--max-rows", type=int, default=None)
    s_run.add_argument("--flush-every", type=int, default=50, help="Append to CSV every N rows")

    args = ap.parse_args()

    if args.cmd == "run":
        run_sync(
            ds=args.ds,
            model_name=args.model_name,
            temperature=args.temperature,
            build_prompt_type=args.build_prompt_type,
            n_repeats=int(args.n_repeats),
            max_rows=args.max_rows,
            flush_every=args.flush_every,
        )

if __name__ == "__main__":
    main()
