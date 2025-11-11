from __future__ import annotations
import os, json, argparse, time, csv, re, math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.ds_utils import get_descriptors
from utils.config import BASE_DIR

# ---------------- Config ----------------
RATE_MIN, RATE_MAX = -1.0, 1.0
INCLUDE_CONFIDENCE = False
INPUT_TYPE = "isomericsmiles"  # 'isomericsmiles' or 'cid'
SYSTEM_MSG = "You are an olfactory rater. Output ONLY valid JSON."
BUILD_PROMPT_CHOICES = ("bysmiles", "byname")

# ------------- Helpers (unchanged) -------------
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

# ------------- JSON extraction -------------
_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(s: str) -> Optional[str]:
    """
    Find the first top-level balanced JSON object in s.
    Simple brace counter; avoids grabbing trailing prose if any.
    """
    start = s.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i+1]
    return None

def parse_llama_json(output_text: str) -> Optional[str]:
    # Try balanced extractor first
    j = extract_first_json(output_text)
    if j:
        return j
    # Fallback: greedy regex (may be messy but sometimes works)
    m = _JSON_BLOCK_RE.search(output_text)
    return m.group(0) if m else None

# ------------- HF Inference -------------
class LlamaRater:
    def __init__(
        self,
        model_name: str,
        load_8bit: bool = False,
        load_4bit: bool = False,
        dtype: Optional[str] = "bfloat16",
        device_map: str = "auto",
        seed: Optional[int] = 42,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        kwargs = dict(
            device_map=device_map,
            torch_dtype=(torch.bfloat16 if dtype == "bfloat16" else torch.float16),
        )
        if load_8bit:
            kwargs["load_in_8bit"] = True
            kwargs.pop("torch_dtype", None)
        if load_4bit:
            kwargs["load_in_4bit"] = True
            kwargs.pop("torch_dtype", None)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True, **kwargs
        )

        if self.tokenizer.pad_token is None:
            # LLaMA often has no explicit pad; align with eos.
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()

    @torch.inference_mode()
    def score_one(
        self,
        system_msg: str,
        user_prompt: str,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        stop_json: bool = True,
        max_retries: int = 2,
    ) -> str:
        """
        Returns a validated JSON string (or raises on failure).
        Retries a couple of times if parsing fails.
        """
        for attempt in range(max_retries + 1):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_prompt},
            ]
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

            gen_out = self.model.generate(
                **inputs,
                do_sample=(temperature > 0.0),
                temperature=max(1e-6, temperature),
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                repetition_penalty=repetition_penalty,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
            decoded = self.tokenizer.decode(gen_out[0], skip_special_tokens=True)

            # keep only assistant completion (after the prompt)
            completion = decoded[len(prompt):].strip()

            # try to extract JSON
            json_text = parse_llama_json(completion)
            if json_text is None:
                if attempt < max_retries:
                    continue
                raise ValueError("No JSON object found in model output.")

            # return raw JSON text (validation done by caller)
            return json_text

        # Should not reach here
        raise RuntimeError("Generation loop fell through without returning.")

# ------------- Runner (local, no batches) -------------
def run_local_llama(
    ds: str,
    model_name: str,
    temperature: float,
    build_prompt_type: str,
    batch_size: int = 1,
    max_new_tokens: int = 256,
    load_8bit: bool = False,
    load_4bit: bool = False,
    dtype: str = "bfloat16",
    out_csv: Optional[str] = None,
):
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"--build-prompt-type must be one of {BUILD_PROMPT_CHOICES}")

    descriptors = get_descriptors(ds)
    input_csv = f"{BASE_DIR}/datasets/{ds}/{ds}_data.csv"
    df = pd.read_csv(input_csv)
    df = df.copy().reset_index(drop=True)
    df["cid"] = pd.to_numeric(df["cid"], errors="coerce").astype("Int64")

    if out_csv is None:
        out_csv = f"{BASE_DIR}/llm_responses/{ds}_odor_llm_scores_temp-{temperature}_model-{model_name.replace('/', '_')}_bpt-{build_prompt_type}.csv"
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    llama = LlamaRater(
        model_name=model_name,
        load_8bit=load_8bit,
        load_4bit=load_4bit,
        dtype=dtype,
        device_map="auto",
        seed=42,
    )

    rows_out: List[Dict[str, Any]] = []

    # simple micro-batching over rows (keeps 1:1 mapping)
    n = len(df)
    for i in range(0, n, batch_size):
        chunk = df.iloc[i : i + batch_size]
        for row_id, row in chunk.iterrows():
            cid = row.get("cid")
            if pd.isna(cid):
                rec = _row_record_error(row, model_name, temperature, build_prompt_type, "missing_cid")
                rows_out.append(rec); continue
            cid = int(cid)

            if build_prompt_type == "bysmiles":
                smi = row_smiles(row)
                if not smi:
                    rec = _row_record_error(row, model_name, temperature, build_prompt_type, "missing_smiles")
                    rows_out.append(rec); continue
                prompt = build_prompt(smi, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
            else:
                nm = row_name(row)
                if not nm:
                    rec = _row_record_error(row, model_name, temperature, build_prompt_type, "missing_name")
                    rows_out.append(rec); continue
                prompt = build_prompt_byname(nm, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)

            try:
                raw_json = llama.score_one(
                    system_msg=SYSTEM_MSG,
                    user_prompt=prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    top_p=1.0,
                    repetition_penalty=1.0,
                    stop_json=True,
                    max_retries=2,
                )
                scores = validate_response(
                    raw_json, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE
                )
                rec = _row_record_ok(row, model_name, temperature, build_prompt_type, scores)
            except Exception as e:
                rec = _row_record_error(row, model_name, temperature, build_prompt_type, f"validate_error: {e}")

            rows_out.append(rec)

        # progress
        done = min(i + batch_size, n)
        if done % 50 == 0 or done == n:
            print(f"[{done}/{n}] processed...")

    # order & save
    cols = ["participant_id", "cid", "temperature", INPUT_TYPE, "name", "model_name", "build_prompt_type"] + descriptors
    if INCLUDE_CONFIDENCE:
        cols.append("confidence")
    if any("error" in r for r in rows_out):
        cols.append("error")

    out_df = pd.DataFrame(rows_out)
    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

def _row_record_base(row: pd.Series, model_name: str, temp: float, build_prompt_type: str) -> Dict[str, Any]:
    cid = pd.to_numeric(row.get("cid"), errors="coerce")
    cid = int(cid) if not pd.isna(cid) else None
    smi = row_smiles(row) or ""
    name = row_name(row) or ""
    return {
        "participant_id": row.get("participant_id"),
        "cid": cid,
        "temperature": temp,
        INPUT_TYPE: smi,
        "name": name,
        "model_name": model_name,
        "build_prompt_type": build_prompt_type,
    }

def _row_record_ok(row, model_name, temp, build_prompt_type, scores: Dict[str, Any]) -> Dict[str, Any]:
    base = _row_record_base(row, model_name, temp, build_prompt_type)
    return {**base, **scores}

def _row_record_error(row, model_name, temp, build_prompt_type, err: str) -> Dict[str, Any]:
    base = _row_record_base(row, model_name, temp, build_prompt_type)
    base["error"] = err
    return base

# ------------- CLI -------------
def main():
    ap = argparse.ArgumentParser(description="Run LLaMA (HF) on olfactory rating prompts and save CSV.")
    ap.add_argument("--ds", required=True, help="Dataset key for get_descriptors() and path building")
    ap.add_argument("--model_name", required=True, help="HF model id, e.g. meta-llama/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--build-prompt-type", dest="build_prompt_type",
                    choices=BUILD_PROMPT_CHOICES, default="bysmiles",
                    help="Prompt construction mode: bysmiles or byname")
    ap.add_argument("--batch_size", type=int, default=1, help="Rows per micro-batch (generation is still per row).")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--out_csv", type=str, default=None)
    ap.add_argument("--load_8bit", action="store_true", help="Load model in 8-bit (bitsandbytes).")
    ap.add_argument("--load_4bit", action="store_true", help="Load model in 4-bit (bitsandbytes).")
    ap.add_argument("--dtype", type=str, default="bfloat16", choices=["bfloat16","float16","auto"])

    args = ap.parse_args()

    run_local_llama(
        ds=args.ds,
        model_name=args.model_name,
        temperature=args.temperature,
        build_prompt_type=args.build_prompt_type,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        load_8bit=args.load_8bit,
        load_4bit=args.load_4bit,
        dtype=args.dtype,
        out_csv=args.out_csv,
    )

if __name__ == "__main__":
    main()
