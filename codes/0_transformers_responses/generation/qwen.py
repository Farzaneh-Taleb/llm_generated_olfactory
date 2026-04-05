from __future__ import annotations

import os
import json
import argparse
import time
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd

from utils.helpers import *
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import *

# ============================================================
# Backend selection
#   export LLM_BACKEND=vllm
#   export LLM_BACKEND=hf
# ============================================================
BACKEND = os.environ.get("LLM_BACKEND", "vllm").lower()

# Optional generation settings
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", 1024))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", 25))  # save partial CSV every N rows


# ============================================================
# Lazy model globals
# ============================================================
_HF_MODEL = None
_HF_TOKENIZER = None
_VLLM_MODEL = None


# ============================================================
# Utilities
# ============================================================
def safe_model_name_for_path(model_name: str) -> str:
    return model_name.replace("/", "_").replace(":", "_")


def extract_first_json_object(text: str) -> str:
    """
    Try to extract the first top-level JSON object from a possibly messy response.
    If none found, return original text.
    """
    if not text:
        return text

    start = text.find("{")
    if start == -1:
        return text

    depth = 0
    in_str = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]

    return text


def save_partial_csv(
    df_input: pd.DataFrame,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    rowrep_to_scores: Dict[Tuple[int, int], Dict[str, Any]],
    out_csv: str,
    build_prompt_type: str,
    n_repeats: int,
    temp_type: str,
) -> None:
    """
    Same idea as final CSV writer, but can be called during generation.
    """
    write_final_csv_by_rowrep(
        df_input=df_input,
        model_name=model_name,
        temperature=temperature,
        descriptors=descriptors,
        rowrep_to_scores=rowrep_to_scores,
        out_csv=out_csv,
        build_prompt_type=build_prompt_type,
        n_repeats=n_repeats,
        temp_type=temp_type,
    )


# ============================================================
# Hugging Face backend
# ============================================================
def load_hf_model(model_name: str):
    global _HF_MODEL, _HF_TOKENIZER
    if _HF_MODEL is not None and _HF_TOKENIZER is not None:
        return _HF_MODEL, _HF_TOKENIZER

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    _HF_TOKENIZER = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    _HF_MODEL = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
        trust_remote_code=True,
    )
    _HF_MODEL.eval()

    return _HF_MODEL, _HF_TOKENIZER


def generate_one_hf(
    model_name: str,
    system_msg: str,
    user_prompt: str,
    temperature: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    import torch

    model, tokenizer = load_hf_model(model_name)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    chat_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(chat_text, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    do_sample = temperature > 0

    gen_kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id,
    )

    if do_sample:
        gen_kwargs["temperature"] = temperature

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return text.strip()


# ============================================================
# vLLM backend
# ============================================================
def load_vllm_model(model_name: str):
    global _VLLM_MODEL
    if _VLLM_MODEL is not None:
        return _VLLM_MODEL

    from vllm import LLM

    _VLLM_MODEL = LLM(
        model=model_name,
        trust_remote_code=True,
    )
    return _VLLM_MODEL


def generate_one_vllm(
    model_name: str,
    system_msg: str,
    user_prompt: str,
    temperature: float,
    max_new_tokens: int = MAX_NEW_TOKENS,
) -> str:
    from vllm import SamplingParams

    llm = load_vllm_model(model_name)

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_prompt},
    ]

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
    )

    outputs = llm.chat(
        messages=messages,
        sampling_params=sampling_params,
        use_tqdm=False,
    )

    return outputs[0].outputs[0].text.strip()


# ============================================================
# Common generation wrapper
# ============================================================
def generate_one(
    model_name: str,
    system_msg: str,
    user_prompt: str,
    temperature: float,
) -> str:
    if BACKEND == "hf":
        return generate_one_hf(
            model_name=model_name,
            system_msg=system_msg,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    elif BACKEND == "vllm":
        return generate_one_vllm(
            model_name=model_name,
            system_msg=system_msg,
            user_prompt=user_prompt,
            temperature=temperature,
        )
    else:
        raise ValueError(f"Unsupported backend: {BACKEND}")


# ============================================================
# Prompt construction
# ============================================================
def build_prompt_for_row(
    row: pd.Series,
    descriptors: List[str],
    build_prompt_type: str,
    RATE_MIN: float,
    RATE_MAX: float,
    temp_type: str,
    pairwise: bool,
) -> Optional[str]:
    if pairwise:
        pair_inputs = row_pair_inputs(row, build_prompt_type)
        if not pair_inputs:
            return None

        a, b = pair_inputs

        if build_prompt_type == "bysmiles":
            return build_prompt_bysmiles(
                (a, b),
                [],
                RATE_MIN,
                RATE_MAX,
                temp_type,
                INCLUDE_CONFIDENCE,
                pairwise=True,
            )
        elif build_prompt_type == "byname":
            return build_prompt_byname(
                (a, b),
                [],
                RATE_MIN,
                RATE_MAX,
                temp_type,
                INCLUDE_CONFIDENCE,
                pairwise=True,
            )
        elif build_prompt_type == "bycid":
            return build_prompt_bycid(
                (a, b),
                [],
                RATE_MIN,
                RATE_MAX,
                temp_type,
                INCLUDE_CONFIDENCE,
                pairwise=True,
            )
        else:
            raise ValueError(f"Unsupported build_prompt_type: {build_prompt_type}")

    else:
        if build_prompt_type == "bysmiles":
            smi = row_smiles(row)
            if not smi:
                return None
            return build_prompt_bysmiles(
                smi,
                descriptors,
                RATE_MIN,
                RATE_MAX,
                temp_type,
                INCLUDE_CONFIDENCE,
                pairwise=False,
            )

        elif build_prompt_type == "byname":
            name = row_name(row)
            if not name:
                return None
            return build_prompt_byname(
                name,
                descriptors,
                RATE_MIN,
                RATE_MAX,
                temp_type,
                INCLUDE_CONFIDENCE,
                pairwise=False,
            )

        elif build_prompt_type == "bycid":
            cid = pd.to_numeric(row.get("cid"), errors="coerce")
            if pd.isna(cid):
                return None
            return build_prompt_bycid(
                str(int(cid)),
                descriptors,
                RATE_MIN,
                RATE_MAX,
                temp_type,
                INCLUDE_CONFIDENCE,
                pairwise=False,
            )

        else:
            raise ValueError(f"Unsupported build_prompt_type: {build_prompt_type}")


# ============================================================
# Local inference
# ============================================================
def run_local_inference(
    df: pd.DataFrame,
    ds_name: str,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    build_prompt_type: str,
    n_repeats: int,
    temp_type: str,
    output_csv: str,
    *,
    RATE_MIN: float,
    RATE_MAX: float,
) -> Dict[Tuple[int, int], Dict[str, Any]]:
    pairwise = is_pairwise_df(df)
    SYSTEM_MSG = SYSTEM_MSG_temp1 if temp_type == "1" else SYSTEM_MSG_temp2

    rowrep_to_scores: Dict[Tuple[int, int], Dict[str, Any]] = {}

    total_jobs = len(df) * n_repeats
    done_jobs = 0

    for row_id, row in df.iterrows():
        prompt = build_prompt_for_row(
            row=row,
            descriptors=descriptors,
            build_prompt_type=build_prompt_type,
            RATE_MIN=RATE_MIN,
            RATE_MAX=RATE_MAX,
            temp_type=temp_type,
            pairwise=pairwise,
        )

        if prompt is None:
            for rep in range(n_repeats):
                rowrep_to_scores[(row_id, rep)] = {"error": "prompt_construction_failed_or_missing_input"}
                done_jobs += 1
            continue

        for rep in range(n_repeats):
            try:
                raw_text = generate_one(
                    model_name=model_name,
                    system_msg=SYSTEM_MSG,
                    user_prompt=prompt,
                    temperature=temperature,
                )

                cleaned_text = extract_first_json_object(raw_text)

                if pairwise:
                    scores = validate_response_pairwise(
                        cleaned_text,
                        RATE_MIN,
                        RATE_MAX,
                        INCLUDE_CONFIDENCE,
                    )
                else:
                    scores = validate_response_single(
                        cleaned_text,
                        descriptors,
                        RATE_MIN,
                        RATE_MAX,
                        INCLUDE_CONFIDENCE,
                    )

            except Exception as e:
                scores = {
                    "error": f"inference_or_validation_error: {e}"
                }

            rowrep_to_scores[(row_id, rep)] = scores
            done_jobs += 1

        if ((row_id + 1) % SAVE_EVERY == 0) or (row_id == len(df) - 1):
            save_partial_csv(
                df_input=df,
                model_name=model_name,
                temperature=temperature,
                descriptors=descriptors,
                rowrep_to_scores=rowrep_to_scores,
                out_csv=output_csv,
                build_prompt_type=build_prompt_type,
                n_repeats=n_repeats,
                temp_type=temp_type,
            )
            print(f"[progress] processed rows: {row_id + 1}/{len(df)} | jobs: {done_jobs}/{total_jobs}")

    return rowrep_to_scores


# ============================================================
# Final CSV writer
# ============================================================
def write_final_csv_by_rowrep(
    df_input: pd.DataFrame,
    model_name: str,
    temperature: float,
    descriptors: List[str],
    rowrep_to_scores: Dict[Tuple[int, int], Dict[str, Any]],
    out_csv: str,
    build_prompt_type: str,
    n_repeats: int,
    temp_type: str,
):
    df = df_input.copy().reset_index(drop=True)
    pairwise = is_pairwise_df(df)
    rows: List[Dict[str, Any]] = []

    for row_id, row in df.iterrows():
        if pairwise:
            cid1, cid2 = row_pair_cids(row)
            inp_pair = row_pair_inputs(row, build_prompt_type)
            inp1, inp2 = inp_pair if inp_pair else ("", "")

            for rep in range(n_repeats):
                base = {
                    "participant_id": row.get("participant_id"),
                    "cid_stimulus_1": cid1,
                    "cid_stimulus_2": cid2,
                    f"{('name' if build_prompt_type == 'byname' else INPUT_TYPE)} stimulus 1": inp1,
                    f"{('name' if build_prompt_type == 'byname' else INPUT_TYPE)} stimulus 2": inp2,
                    "repeat": rep,
                    "temperature": temperature,
                    "model_name": model_name,
                    "build_prompt_type": build_prompt_type,
                    "temp_type": temp_type,
                }
                scores = rowrep_to_scores.get((row_id, rep), {"error": "missing_output"})
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
                    "temp_type": temp_type,
                }
                scores = rowrep_to_scores.get((row_id, rep), {"error": "missing_output"})
                rows.append({**base, **scores})

    if pairwise:
        cols = [
            "participant_id",
            "cid_stimulus_1",
            "cid_stimulus_2",
            f"{('name' if build_prompt_type == 'byname' else INPUT_TYPE)} stimulus 1",
            f"{('name' if build_prompt_type == 'byname' else INPUT_TYPE)} stimulus 2",
            "repeat",
            "temperature",
            "model_name",
            "build_prompt_type",
            "temp_type",
            "similarity",
        ]
    else:
        cols = [
            "participant_id",
            "cid",
            "repeat",
            "temperature",
            INPUT_TYPE,
            "name",
            "model_name",
            "build_prompt_type",
            "temp_type",
        ] + descriptors

    if INCLUDE_CONFIDENCE:
        cols.append("confidence")

    if rows and any("error" in r for r in rows):
        cols.append("error")

    out_df = pd.DataFrame(rows)
    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


# ============================================================
# Dataset preprocessing
# ============================================================
def load_and_prepare_df(ds: str, debug: bool = False) -> Tuple[pd.DataFrame, List[str], bool, float, float]:
    RATE_MIN, RATE_MAX = get_rate_range(ds)

    input_csv = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"
    df = pd.read_csv(input_csv)

    if debug:
        df = df.head(2).copy()

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

    return df, descriptors, pairwise, RATE_MIN, RATE_MAX


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True, help="Dataset key")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--model_name", required=True, help="HF/vLLM model name, e.g. Qwen/Qwen3.5-27B-FP8")
    ap.add_argument(
        "--build-prompt-type",
        dest="build_prompt_type",
        choices=BUILD_PROMPT_CHOICES,
        default="bysmiles",
        help="Prompt mode",
    )
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--temp-type", dest="temp_type", choices=["1", "2"], default="1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    print(f"Using backend: {BACKEND}")
    print(f"Model: {args.model_name}")

    df, descriptors, pairwise, RATE_MIN, RATE_MAX = load_and_prepare_df(
        ds=args.ds,
        debug=args.debug,
    )

    os.makedirs(f"{BASE_DIR}/results/responses_local/llm_responses", exist_ok=True)
    output_csv = (
        f"{BASE_DIR}/results/responses_local/llm_responses/"
        f"{args.ds}_odor_llm_scores_temp-{args.temperature}"
        f"_model-{safe_model_name_for_path(args.model_name)}"
        f"_bpt-{args.build_prompt_type}_reps-{args.n_repeats}.csv"
    )

    rowrep_to_scores = run_local_inference(
        df=df,
        ds_name=args.ds,
        model_name=args.model_name,
        temperature=args.temperature,
        descriptors=descriptors,
        build_prompt_type=args.build_prompt_type,
        n_repeats=args.n_repeats,
        temp_type=args.temp_type,
        output_csv=output_csv,
        RATE_MIN=RATE_MIN,
        RATE_MAX=RATE_MAX,
    )

    write_final_csv_by_rowrep(
        df_input=df,
        model_name=args.model_name,
        temperature=args.temperature,
        descriptors=descriptors,
        rowrep_to_scores=rowrep_to_scores,
        out_csv=output_csv,
        build_prompt_type=args.build_prompt_type,
        n_repeats=args.n_repeats,
        temp_type=args.temp_type,
    )


if __name__ == "__main__":
    main()