from __future__ import annotations

import os, json, argparse
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from custom_utils.helpers import *
from custom_utils.ds_utils import get_descriptors2 as get_descriptors
from custom_utils.config import *


def load_textgen_pipeline(
    model_name: str,
    hf_token: Optional[str],
    dtype: str,
    device: int,
):
    if dtype == "bf16":
        torch_dtype = torch.bfloat16
    elif dtype == "fp16":
        torch_dtype = torch.float16
    else:
        torch_dtype = torch.float32

    tok = AutoTokenizer.from_pretrained(model_name, token=hf_token, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        torch_dtype=torch_dtype,
        device_map="auto" if device == -1 else None,
    )
    if device != -1:
        model = model.to(f"cuda:{device}")
    model.eval()
    print("n_gpus:", torch.cuda.device_count())
    print("hf_device_map:", getattr(model, "hf_device_map", None))

    textgen = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tok,
        # device=device,  # 0 for cuda:0, -1 for cpu
    )
    return tok, textgen


def make_rows_and_prompts(
    df: pd.DataFrame,
    ds: str,
    build_prompt_type: str,
    n_repeats: int,
) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Returns list of records with:
      - meta fields (row_id, rep, cid info)
      - messages (chat format)
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    RATE_MIN, RATE_MAX = get_rate_range(ds)
    df = df.copy().reset_index(drop=True)
    pairwise = is_pairwise_df(df)

    if not pairwise:
        # match your prior filtering/dedup rules
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

    rows: List[Dict[str, Any]] = []
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

            cid1, cid2 = row_pair_cids(r)

            for rep in range(n_repeats):
                rows.append({
                    "row_id": row_id,
                    "repeat": rep,
                    "participant_id": r.get("participant_id"),
                    "cid_stimulus_1": cid1,
                    "cid_stimulus_2": cid2,
                    "build_prompt_type": build_prompt_type,
                    "pairwise": True,
                    "input_a": a,
                    "input_b": b,
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                })
        else:
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

            cid = pd.to_numeric(r.get("cid"), errors="coerce")
            cid = int(cid) if pd.notna(cid) else None

            for rep in range(n_repeats):
                rows.append({
                    "row_id": row_id,
                    "repeat": rep,
                    "participant_id": r.get("participant_id"),
                    "cid": cid,
                    "build_prompt_type": build_prompt_type,
                    "pairwise": False,
                    INPUT_TYPE: row_smiles(r) or "",
                    "name": row_name(r) or "",
                    "messages": [
                        {"role": "system", "content": SYSTEM_MSG},
                        {"role": "user", "content": prompt},
                    ],
                })

    return df, rows


def _extract_assistant_text(gen_out: Any) -> str:
    """
    HF pipeline output varies by version.
    We handle:
      - [{"generated_text": "..."}]
      - [{"generated_text": [{"role":..., "content":...}, ...]}] (chat-like)
    """
    if isinstance(gen_out, list) and gen_out:
        item = gen_out[0]
        gt = item.get("generated_text", "")
        if isinstance(gt, str):
            return gt
        if isinstance(gt, list) and gt:
            # usually list of messages; last one assistant
            last = gt[-1]
            if isinstance(last, dict) and "content" in last:
                return last["content"]
    return ""


def write_jsonl(path: str, records: List[Dict[str, Any]]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved JSONL: {path} (n={len(records)})")


def write_scores_csv(
    df_input: pd.DataFrame,
    ds: str,
    model_name: str,
    temperature: float,
    build_prompt_type: str,
    n_repeats: int,
    generated: List[Dict[str, Any]],
    out_csv: str,
):
    """
    Optional: parses generated text using your validate_response_* and writes the same CSV format as GPT/Gemini.
    """
    RATE_MIN, RATE_MAX = get_rate_range(ds)
    pairwise = is_pairwise_df(df_input)

    if not pairwise:
        descriptors = get_descriptors(ds)
    else:
        descriptors = []

    rows: List[Dict[str, Any]] = []
    for rec in generated:
        base = {
            "participant_id": rec.get("participant_id"),
            "repeat": rec.get("repeat"),
            "temperature": temperature,
            "model_name": model_name,
            "build_prompt_type": build_prompt_type,
        }
        if rec.get("pairwise"):
            base.update({
                "cid_stimulus_1": rec.get("cid_stimulus_1"),
                "cid_stimulus_2": rec.get("cid_stimulus_2"),
                f"{('name' if build_prompt_type=='byname' else INPUT_TYPE)} stimulus 1": rec.get("input_a", ""),
                f"{('name' if build_prompt_type=='byname' else INPUT_TYPE)} stimulus 2": rec.get("input_b", ""),
            })
        else:
            base.update({
                "cid": rec.get("cid"),
                INPUT_TYPE: rec.get(INPUT_TYPE, ""),
                "name": rec.get("name", ""),
            })

        text = rec.get("generated_text", "") or ""
        try:
            if pairwise:
                scores = validate_response_pairwise(text, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
            else:
                scores = validate_response_single(text, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
        except Exception as e:
            scores = {"error": f"validate_error: {e}"}

        rows.append({**base, **scores})

    out_df = pd.DataFrame(rows)

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
    if "error" in out_df.columns:
        cols.append("error")

    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved CSV: {out_csv} (n={len(out_df)})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--model_name", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--build-prompt-type", dest="build_prompt_type",
                    choices=BUILD_PROMPT_CHOICES, default="bysmiles")
    ap.add_argument("--n-repeats", type=int, default=1)

    # generation controls
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--do_sample", action="store_true", help="Enable sampling (recommended if temperature > 0)")
    ap.add_argument("--seed", type=int, default=0)

    # performance
    ap.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    ap.add_argument("--device", type=int, default=0, help="CUDA device index, or -1 for CPU")
    ap.add_argument("--batch_size", type=int, default=1)

    # auth / output
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--out_dir", default=f"{BASE_DIR}/results/responses/llama31")
    ap.add_argument("--write_csv", action="store_true", help="Also parse outputs and write CSV with scores")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    # determinism
    torch.manual_seed(args.seed)
    np_seed_ok = True
    try:
        import numpy as np
        np.random.seed(args.seed)
    except Exception:
        np_seed_ok = False

    ds = args.ds
    input_csv = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"
    df = pd.read_csv(input_csv)
    print("debug:", args.debug, "dataset:", ds, "n_rows:", len(df), "np_seed_ok:", np_seed_ok)
    if args.debug:
        df = df.head(5)

    df_used, rows = make_rows_and_prompts(
        df=df,
        ds=ds,
        build_prompt_type=args.build_prompt_type,
        n_repeats=int(args.n_repeats),
    )
    if not rows:
        raise RuntimeError("No prompts/rows produced. Check dataset columns and build_prompt_type.")

    tok, textgen = load_textgen_pipeline(
        model_name=args.model_name,
        hf_token=args.hf_token,
        dtype=args.dtype,
        device=args.device,
    )

    # If you pass chat messages to pipeline, newer transformers supports it.
    # We still pass tokenized chat template explicitly for stability across versions.
    generated_records: List[Dict[str, Any]] = []
    bs = max(1, int(args.batch_size))
    safe_model = args.model_name.split("/")[-1]
    job_name = f"{ds}_{safe_model}_temp-{args.temperature}_bpt-{args.build_prompt_type}_reps-{args.n_repeats}"
    time_now = time.strftime("%Y-%m-%d %H:%M:%S")
    log_batch_entry(ds, safe_model, args.temperature, job_name, args.build_prompt_type, args.n_repeats)

    for i in range(0, len(rows), bs):
        print(f"Processing {i}-{min(i+bs, len(rows))} / {len(rows)}",flush=True)
        batch = rows[i:i+bs]

        # Build chat-templated strings (recommended for consistent behavior)
        prompts = [
            tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True)
            for r in batch
        ]

        outputs = textgen(
            prompts,
            max_new_tokens=int(args.max_new_tokens),
            do_sample=bool(args.do_sample),
            temperature=float(args.temperature),
            top_p=float(args.top_p),
            return_full_text=False,
            pad_token_id=tok.pad_token_id,
        )

        # outputs is list aligned with prompts
        for r, out in zip(batch, outputs):
            # Most common output: [{"generated_text": "..."}]
            # But pipeline may return dict already if batched
            if isinstance(out, list):
                gen_text = out[0].get("generated_text", "")
            elif isinstance(out, dict):
                gen_text = out.get("generated_text", "")
            else:
                gen_text = str(out)

            rec = {k: v for k, v in r.items() if k != "messages"}
            rec.update({
                "ds": ds,
                "model_name": args.model_name,
                "temperature": float(args.temperature),
                "top_p": float(args.top_p),
                "max_new_tokens": int(args.max_new_tokens),
                "do_sample": bool(args.do_sample),
                "seed": int(args.seed),
                "generated_text": gen_text,
            })
            generated_records.append(rec)

    os.makedirs(args.out_dir, exist_ok=True)
    
    # out_jsonl = (
    #     f"{args.out_dir}/"
    #     f"{ds}_model-{safe_model}_bpt-{args.build_prompt_type}"
    #     f"_temp-{args.temperature}_topP-{args.top_p}"
    #     f"_newtok-{args.max_new_tokens}_sample-{int(args.do_sample)}"
    #     f"_reps-{args.n_repeats}_seed-{args.seed}.jsonl"
    # )
    jsonl_path = f"{BASE_DIR}/results/responses/tmp2/{ds}_{safe_model}_{args.temperature}_{args.build_prompt_type}_reps-{args.n_repeats}.jsonl"

    write_jsonl(jsonl_path, generated_records)

    if args.write_csv:
        output_csv = (
                f"{BASE_DIR}/results/responses/llm_responses2/"
                f"{ds}_odor_llm_scores_temp-{args.temperature}_model-{safe_model}_bpt-{args.build_prompt_type}_reps-{args.n_repeats}_time-{time_now}.csv"
            )
        write_scores_csv(
            df_input=df_used,
            ds=ds,
            model_name=args.model_name,
            temperature=float(args.temperature),
            build_prompt_type=args.build_prompt_type,
            n_repeats=int(args.n_repeats),
            generated=generated_records,
            out_csv=output_csv,
        )


if __name__ == "__main__":
    main()
