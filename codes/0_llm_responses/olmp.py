from __future__ import annotations
import os, json, argparse
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.helpers import *
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import *


def _extract_first_json_obj(text: str) -> str:
    """
    Robust-ish: grab the first top-level {...} block.
    (You already have validate_response_* which can raise if malformed.)
    """
    if not text:
        return ""
    start = text.find("{")
    if start < 0:
        return text.strip()
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1].strip()
    return text.strip()


@torch.inference_mode()
def run_olmo(
    df: pd.DataFrame,
    ds_name: str,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    build_prompt_type: str,
    n_repeats: int,
    rate_min: float,
    rate_max: float,
    *,
    max_new_tokens: int = 256,
    batch_size: int = 8,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    """
    Returns mapping (row_id, rep) -> scores dict (validated or with 'error').
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    df = df.copy().reset_index(drop=True)
    pairwise = is_pairwise_df(df)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    # OLMo chat models expect chat formatting; apply_chat_template handles tokens properly. :contentReference[oaicite:4]{index=4}
    # For OLMo 2, AI2 demos often use this system prompt (optional). :contentReference[oaicite:5]{index=5}
    system_msg = SYSTEM_MSG  # keep your existing one
    # If you want OLMo2-style:
    # system_msg = "You are OLMo 2, a helpful and harmless AI Assistant built by the Allen Institute for AI."

    jobs: List[Tuple[int, int, Dict[str, Any], str]] = []
    for row_id, r in df.iterrows():
        # build prompt string using your existing helpers
        if pairwise:
            pair_inputs = row_pair_inputs(r, build_prompt_type)
            if not pair_inputs:
                continue
            a, b = pair_inputs
            if build_prompt_type == "bysmiles":
                prompt = build_prompt_bysmiles((a, b), [], rate_min, rate_max, INCLUDE_CONFIDENCE, pairwise=True)
            else:
                prompt = build_prompt_byname((a, b), [], rate_min, rate_max, INCLUDE_CONFIDENCE, pairwise=True)
        else:
            if build_prompt_type == "bysmiles":
                smi = row_smiles(r)
                if not smi:
                    continue
                prompt = build_prompt_bysmiles(smi, descriptors, rate_min, rate_max, INCLUDE_CONFIDENCE, pairwise=False)
            else:
                nm = row_name(r)
                if not nm:
                    continue
                prompt = build_prompt_byname(nm, descriptors, rate_min, rate_max, INCLUDE_CONFIDENCE, pairwise=False)

        for rep in range(n_repeats):
            meta = {"row_id": row_id, "rep": rep}
            jobs.append((row_id, rep, meta, prompt))

    out: Dict[Tuple[int, int], Dict[str, Any]] = {}

    # batched generation
    for i in range(0, len(jobs), batch_size):
        chunk = jobs[i : i + batch_size]

        messages_list = []
        for (_, _, _, prompt) in chunk:
            messages_list.append(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ]
            )

        # Convert each conversation into token IDs with the model's expected template.
        inputs = tokenizer(
            [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages_list],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        do_sample = float(temperature) > 0.0
        gen = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Decode only the newly generated tokens per sample
        for (row_id, rep, _, _), input_ids, gen_ids in zip(chunk, inputs["input_ids"], gen):
            new_tokens = gen_ids[len(input_ids) :]
            text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            text = _extract_first_json_obj(text)

            try:
                if pairwise:
                    scores = validate_response_pairwise(text, rate_min, rate_max, INCLUDE_CONFIDENCE)
                else:
                    scores = validate_response_single(text, descriptors, rate_min, rate_max, INCLUDE_CONFIDENCE)
            except Exception as e:
                scores = {"error": f"validate_error: {e}", "raw": text[:500]}

            out[(row_id, rep)] = scores

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--model_name", required=True)  # e.g. allenai/Olmo-3-7B-Instruct
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--build-prompt-type", dest="build_prompt_type",
                    choices=BUILD_PROMPT_CHOICES, default="bysmiles")
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    args = ap.parse_args()

    ds = args.ds
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

    rowrep_to_scores = run_olmo(
        df=df,
        ds_name=ds,
        model_name=args.model_name,
        temperature=args.temperature,
        descriptors=descriptors,
        build_prompt_type=args.build_prompt_type,
        n_repeats=int(args.n_repeats),
        rate_min=RATE_MIN,
        rate_max=RATE_MAX,
        max_new_tokens=int(args.max_new_tokens),
        batch_size=int(args.batch_size),
    )

    # Reuse your existing CSV writer (same as GPT/Gemini scripts)
    os.makedirs(f"{BASE_DIR}/results/responses/llm_responses", exist_ok=True)
    out_csv = (
        f"{BASE_DIR}/results/responses/llm_responses/"
        f"{ds}_odor_llm_scores_temp-{args.temperature}_model-{args.model_name}_bpt-{args.build_prompt_type}_reps-{args.n_repeats}.csv"
    )
    write_final_csv_by_rowrep(
        df_input=df,
        model_name=args.model_name,
        temperature=args.temperature,
        descriptors=descriptors,
        rowrep_to_scores=rowrep_to_scores,
        out_csv=out_csv,
        build_prompt_type=args.build_prompt_type,
        n_repeats=int(args.n_repeats),
    )


if __name__ == "__main__":
    main()
