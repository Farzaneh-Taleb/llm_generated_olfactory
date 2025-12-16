from __future__ import annotations

import os, json, argparse
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.helpers import *
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import *


# ---------------- Helpers ----------------

def _extract_first_json_obj(text: str) -> str:
    """Grab the first top-level {...} JSON object from text (best-effort)."""
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


def _format_chat_if_available(tokenizer, system_msg: str, user_msg: str) -> str:
    """
    If tokenizer has a chat template (chat/instruct models), use it.
    Otherwise fallback to plain concatenation.
    """
    try:
        if getattr(tokenizer, "chat_template", None):
            messages = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return f"{system_msg}\n\n{user_msg}\n"


def _pick_torch_dtype(dtype: str):
    d = dtype.lower()
    if d in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if d in {"fp16", "float16"}:
        return torch.float16
    if d in {"fp32", "float32"}:
        return torch.float32
    raise ValueError("dtype must be one of: bf16, fp16, fp32")


# ---------------- OLMo runner (HF local) ----------------

@torch.inference_mode()
def run_olmo_prompts(
    model_name: str,
    prompts: List[str],
    *,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    device: str,          # "auto" | "cuda" | "cpu"
    dtype: str,           # "bf16" | "fp16" | "fp32"
) -> List[str]:
    """
    Run local HF model over a list of already-formatted prompts.
    Returns decoded continuations (attempts to strip prompt prefix).
    """
    torch_dtype = _pick_torch_dtype(dtype)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        device_map="auto" if device == "auto" else None,
    )
    if device in {"cuda", "cpu"}:
        model = model.to(device)
    model.eval()

    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})
        model.resize_token_embeddings(len(tokenizer))

    outs: List[str] = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(model.device) for k, v in enc.items()}

        do_sample = float(temperature) > 0.0
        gen = model.generate(
            **enc,
            max_new_tokens=int(max_new_tokens),
            do_sample=do_sample,
            temperature=float(temperature) if do_sample else None,
            top_p=float(top_p) if do_sample else None,
            repetition_penalty=float(repetition_penalty),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

        decoded = tokenizer.decode(gen[0], skip_special_tokens=True)
        # strip prompt if it appears verbatim
        if decoded.startswith(prompt):
            decoded = decoded[len(prompt):].lstrip()
        outs.append(decoded)

    return outs


# ---------------- PREP: build prompt JSONL ----------------

def make_prompt_jsonl(
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
    Writes one record per row+rep:
      { "key": "...", "system": "...", "user": "...", "model_name": "...", "temperature": ... }
    key encodes ds|temp|bpt|row{row_id}|rep{rep}|meta...
    """
    if build_prompt_type not in BUILD_PROMPT_CHOICES:
        raise ValueError(f"build_prompt_type must be one of {BUILD_PROMPT_CHOICES}")

    df = df.copy().reset_index(drop=True)
    pairwise = is_pairwise_df(df)

    if not pairwise:
        df["cid"] = pd.to_numeric(df.get("cid"), errors="coerce").astype("Int64")

    total_rows = len(df)
    used = 0

    os.makedirs(os.path.dirname(out_jsonl_path), exist_ok=True)
    with open(out_jsonl_path, "w", encoding="utf-8") as f:
        for row_id, r in df.iterrows():
            if pairwise:
                pair_inputs = row_pair_inputs(r, build_prompt_type)
                if not pair_inputs:
                    continue
                a, b = pair_inputs
                if build_prompt_type == "bysmiles":
                    user_prompt = build_prompt_bysmiles((a, b), [], RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=True)
                else:
                    user_prompt = build_prompt_byname((a, b), [], RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=True)

                c1, c2 = row_pair_cids(r)
                meta_suffix = ""
                if c1 is not None and c2 is not None:
                    meta_suffix = f"|cid1{c1}_cid2{c2}"
            else:
                if build_prompt_type == "bysmiles":
                    smi = row_smiles(r)
                    if not smi:
                        continue
                    user_prompt = build_prompt_bysmiles(smi, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=False)
                else:
                    nm = row_name(r)
                    if not nm:
                        continue
                    user_prompt = build_prompt_byname(nm, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE, pairwise=False)

                cid = r.get("cid")
                meta_suffix = f"|cid{int(cid)}" if pd.notna(cid) else ""

            for rep in range(int(n_repeats)):
                key = f"{ds_name}|{temperature}|bpt:{build_prompt_type}|row{row_id}|rep{rep}{meta_suffix}"
                rec = {
                    "key": key,
                    "system": SYSTEM_MSG,
                    "user": user_prompt,
                    "model_name": model_name,
                    "temperature": float(temperature),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                used += 1

    return total_rows, used


# ---------------- RUN: execute local OLMo over prompt JSONL ----------------

def run_local_batch(
    in_jsonl: str,
    out_jsonl: str,
    model_name: str,
    *,
    temperature: float,
    max_new_tokens: int,
    top_p: float,
    repetition_penalty: float,
    device: str,
    dtype: str,
    batch_size: int,
):
    """
    Reads prompt JSONL from submit step, formats chats if possible,
    runs OLMo locally, writes output JSONL:
      { "key": ..., "response_text": ... }
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    keys: List[str] = []
    prompts: List[str] = []

    with open(in_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            key = rec["key"]
            system = rec.get("system", SYSTEM_MSG)
            user = rec["user"]
            full_prompt = _format_chat_if_available(tokenizer, system, user)
            keys.append(key)
            prompts.append(full_prompt)

    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for i in range(0, len(prompts), int(batch_size)):
            chunk_prompts = prompts[i:i + int(batch_size)]
            chunk_keys = keys[i:i + int(batch_size)]

            outs = run_olmo_prompts(
                model_name=model_name,
                prompts=chunk_prompts,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                device=device,
                dtype=dtype,
            )

            for k, resp in zip(chunk_keys, outs):
                out_f.write(json.dumps({"key": k, "response_text": resp}, ensure_ascii=False) + "\n")

    print(f"Saved OLMo outputs: {out_jsonl}  (n={len(prompts)})")


# ---------------- PARSE: output JSONL -> (row, rep) mapping ----------------

def parse_olmo_output_jsonl(
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

    for p in jsonl_paths:
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                key = rec.get("key", "")
                text = rec.get("response_text", "") or ""
                if not key:
                    continue

                parts = key.split("|")
                row_part = next((x for x in parts if x.startswith("row")), None)
                rep_part = next((x for x in parts if x.startswith("rep")), None)
                if row_part is None or rep_part is None:
                    continue

                row_id = int(row_part.replace("row", ""))
                rep = int(rep_part.replace("rep", ""))

                try:
                    jsonish = _extract_first_json_obj(text)
                    if pairwise:
                        scores = validate_response_pairwise(jsonish, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                    else:
                        scores = validate_response_single(jsonish, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                except Exception as e:
                    scores = {"error": f"validate_error: {e}"}

                out[(row_id, rep)] = scores

    return out


# ---------------- CSV writer ----------------

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
                scores = rowrep_to_scores.get((row_id, rep), {"error": "missing_in_output"})
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
                scores = rowrep_to_scores.get((row_id, rep), {"error": "missing_in_output"})
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

    if any("error" in r for r in (rows or [{}])):
        cols.append("error")

    out_df = pd.DataFrame(rows)
    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")


# ---------------- DATA PREP (same logic as your other scripts) ----------------

def load_and_prepare_df(ds: str, debug: bool) -> tuple[pd.DataFrame, bool, List[str], float, float]:
    RATE_MIN, RATE_MAX = get_rate_range(ds)
    input_csv = f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv"
    df = pd.read_csv(input_csv)

    if debug:
        df = df.head(2)

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

    return df, pairwise, descriptors, RATE_MIN, RATE_MAX


# ---------------- CLI ----------------

def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s_all = sub.add_parser("all", help="One-shot: build prompts -> run HF OLMo -> parse -> write CSV")
    s_all.add_argument("--ds", required=True)
    s_all.add_argument("--model_name", required=True, help="HF model id, e.g. allenai/OLMo-1B-hf")
    s_all.add_argument("--temperature", type=float, default=0.0)
    s_all.add_argument("--build-prompt-type", dest="build_prompt_type",
                       choices=BUILD_PROMPT_CHOICES, default="bysmiles")
    s_all.add_argument("--n-repeats", type=int, default=1)
    s_all.add_argument("--debug", action="store_true")

    # generation params
    s_all.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=256)
    s_all.add_argument("--top_p", type=float, default=0.95)
    s_all.add_argument("--repetition_penalty", type=float, default=1.0)
    s_all.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    s_all.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    s_all.add_argument("--batch-size", dest="batch_size", type=int, default=4)

    args = ap.parse_args()

    if args.cmd == "all":
        ds = args.ds
        df, pairwise, descriptors, RATE_MIN, RATE_MAX = load_and_prepare_df(ds, args.debug)

        os.makedirs(f"{BASE_DIR}/results/responses/tmp", exist_ok=True)
        os.makedirs(f"{BASE_DIR}/results/responses/llm_responses", exist_ok=True)

        model_slug = args.model_name.replace("/", "-")
        prompt_jsonl = (
            f"{BASE_DIR}/results/responses/tmp/"
            f"{ds}_{model_slug}_{args.temperature}_{args.build_prompt_type}_reps-{args.n_repeats}.jsonl"
        )
        out_jsonl = (
            f"{BASE_DIR}/results/responses/tmp/"
            f"{ds}_{model_slug}_{args.temperature}_{args.build_prompt_type}_reps-{args.n_repeats}_OUT.jsonl"
        )
        out_csv = (
            f"{BASE_DIR}/results/responses/llm_responses/"
            f"{ds}_odor_llm_scores_temp-{args.temperature}"
            f"_model-{model_slug}"
            f"_bpt-{args.build_prompt_type}"
            f"_reps-{args.n_repeats}"
            f"_OLMo-HF.csv"
        )

        total, used = make_prompt_jsonl(
            df=df,
            ds_name=ds,
            model_name=args.model_name,
            temperature=args.temperature,
            descriptors=descriptors,
            out_jsonl_path=prompt_jsonl,
            build_prompt_type=args.build_prompt_type,
            n_repeats=args.n_repeats,
            RATE_MIN=RATE_MIN, RATE_MAX=RATE_MAX,
        )
        print(f"[ALL] Prepared prompt JSONL: {prompt_jsonl}  (rows={total}, requests={used})")

        run_local_batch(
            in_jsonl=prompt_jsonl,
            out_jsonl=out_jsonl,
            model_name=args.model_name,
            temperature=args.temperature,
            max_new_tokens=args.max_new_tokens,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            device=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
        )

        rowrep_to_scores = parse_olmo_output_jsonl(
            [out_jsonl],
            descriptors,
            RATE_MIN=RATE_MIN,
            RATE_MAX=RATE_MAX,
            pairwise=pairwise,
        )

        write_final_csv_by_rowrep(
            df_input=df,
            model_name=args.model_name,
            temperature=args.temperature,
            descriptors=descriptors,
            rowrep_to_scores=rowrep_to_scores,
            out_csv=out_csv,
            build_prompt_type=args.build_prompt_type,
            n_repeats=args.n_repeats,
        )

        print(f"[ALL] Done.\n  prompts: {prompt_jsonl}\n  outputs: {out_jsonl}\n  csv:     {out_csv}")


if __name__ == "__main__":
    main()
