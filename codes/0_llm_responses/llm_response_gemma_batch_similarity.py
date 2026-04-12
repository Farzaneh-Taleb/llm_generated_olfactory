from __future__ import annotations
import os, json, argparse
from typing import Any, Dict, List
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.helpers import *
from utils.ds_utils import get_descriptors2 as get_descriptors
from utils.config import *

# =======================
# GEMMA MODEL
# =======================
MODEL_NAME = "google/gemma-2b-it"

print("Loading Gemma...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)
print("Gemma loaded.")

# =======================
# PROMPT FORMAT
# =======================
def build_gemma_prompt(system_msg: str, user_msg: str) -> str:
    return (
        "<bos><start_of_turn>system\n"
        f"{system_msg}\n"
        "<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user_msg}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

# =======================
# JSONL CREATION (UNCHANGED LOGIC)
# =======================
def make_jsonl_chunks(
    df,
    ds_name,
    temperature,
    descriptors,
    out_jsonl_path,
    build_prompt_type,
    n_repeats,
    *,
    RATE_MIN,
    RATE_MAX,
    temp_type,
    chunk_size=200,
):
    df = df.copy().reset_index(drop=True)
    pairwise = is_pairwise_df(df)

    SYSTEM_MSG = SYSTEM_MSG_temp1 if temp_type == '1' else SYSTEM_MSG_temp2

    used_reqs = 0
    generated_files = []

    num_chunks = (len(df) + chunk_size - 1) // chunk_size

    for i in range(num_chunks):
        chunk_df = df.iloc[i * chunk_size : (i + 1) * chunk_size]
        if chunk_df.empty:
            continue

        chunk_path = out_jsonl_path.replace(".jsonl", f"_chunk{i}.jsonl")
        os.makedirs(os.path.dirname(chunk_path), exist_ok=True)
        generated_files.append(chunk_path)

        with open(chunk_path, "w") as f:
            for row_id, r in chunk_df.iterrows():

                if pairwise:
                    pair_inputs = row_pair_inputs(r, build_prompt_type)
                    if not pair_inputs:
                        continue
                    a, b = pair_inputs

                    prompt = (
                        build_prompt_bysmiles((a, b), [], RATE_MIN, RATE_MAX, temp_type, INCLUDE_CONFIDENCE, True)
                        if build_prompt_type == "bysmiles"
                        else build_prompt_byname((a, b), [], RATE_MIN, RATE_MAX, temp_type, INCLUDE_CONFIDENCE, True)
                    )

                    c1, c2 = row_pair_cids(r)
                    meta_suffix = f"|cid1{c1}_cid2{c2}"

                else:
                    if build_prompt_type == "bysmiles":
                        smi = row_smiles(r)
                        if not smi:
                            continue
                        prompt = build_prompt_bysmiles(smi, descriptors, RATE_MIN, RATE_MAX, temp_type, INCLUDE_CONFIDENCE, False)
                    else:
                        nm = row_name(r)
                        if not nm:
                            continue
                        prompt = build_prompt_byname(nm, descriptors, RATE_MIN, RATE_MAX, temp_type, INCLUDE_CONFIDENCE, False)

                    cid = r.get("cid")
                    meta_suffix = f"|cid{int(cid)}" if pd.notna(cid) else ""

                for rep in range(n_repeats):
                    key = f"{ds_name}|{temperature}|row{row_id}|rep{rep}{meta_suffix}"

                    request = {
                        "system_instruction": SYSTEM_MSG,
                        "user_prompt": prompt,
                        "temperature": float(temperature),
                    }

                    f.write(json.dumps({"key": key, "request": request}) + "\n")
                    used_reqs += 1

    return generated_files


# =======================
# GEMMA + CSV WRITER
# =======================
def run_gemma_and_write_csv(
    jsonl_path: str,
    df_input: pd.DataFrame,
    descriptors: List[str],
    out_csv: str,
    *,
    RATE_MIN: float,
    RATE_MAX: float,
    temperature: float,
    build_prompt_type: str,
    n_repeats: int,
):
    pairwise = is_pairwise_df(df_input)
    df = df_input.copy().reset_index(drop=True)

    rows = []

    with open(jsonl_path, "r") as f:
        for line in f:
            rec = json.loads(line)

            key = rec["key"]
            req = rec["request"]

            parts = key.split("|")
            row_id = int(parts[2].replace("row", ""))
            rep = int(parts[3].replace("rep", ""))

            system_msg = req["system_instruction"]
            user_msg = req["user_prompt"]
            temp = req["temperature"]

            prompt = build_gemma_prompt(system_msg, user_msg)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=temp,
                    do_sample=True,
                )

            text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # -------- Parse --------
            try:
                if pairwise:
                    scores = validate_response_pairwise(text, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
                else:
                    scores = validate_response_single(text, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
            except Exception as e:
                scores = {"error": str(e)}

            row = df.iloc[row_id]

            # -------- Build row --------
            if pairwise:
                cid1, cid2 = row_pair_cids(row)
                inp1, inp2 = row_pair_inputs(row, build_prompt_type) or ("", "")

                out_row = {
                    "participant_id": row.get("participant_id"),
                    "cid_stimulus_1": cid1,
                    "cid_stimulus_2": cid2,
                    "name stimulus 1": inp1,
                    "name stimulus 2": inp2,
                    "repeat": rep,
                    "temperature": temperature,
                    "model_name": MODEL_NAME,
                    "build_prompt_type": build_prompt_type,
                    "similarity": scores.get("similarity"),
                }

            else:
                cid = int(row["cid"]) if pd.notna(row.get("cid")) else None
                smi = row_smiles(row) or ""
                name = row_name(row) or ""

                out_row = {
                    "participant_id": row.get("participant_id"),
                    "cid": cid,
                    "repeat": rep,
                    "temperature": temperature,
                    "isomericsmiles": smi,
                    "name": name,
                    "model_name": MODEL_NAME,
                    "build_prompt_type": build_prompt_type,
                }

                for d in descriptors:
                    out_row[d] = scores.get(d)

            rows.append(out_row)

    out_df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    out_df.to_csv(out_csv, index=False)

    print(f"Saved CSV: {out_csv}")


# =======================
# MAIN
# =======================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", required=True)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--build-prompt-type", choices=BUILD_PROMPT_CHOICES, default="bysmiles")
    ap.add_argument("--n-repeats", type=int, default=1)
    ap.add_argument("--temp_type", type=str)

    args = ap.parse_args()

    ds = args.ds
    RATE_MIN, RATE_MAX = get_rate_range(ds)

    df = pd.read_csv(f"{BASE_DIR}/data/datasets/{ds}/{ds}_data.csv")
    pairwise = is_pairwise_df(df)

    if not pairwise:
        df["cid"] = pd.to_numeric(df["cid"], errors="coerce")
        descriptors = get_descriptors(ds)
    else:
        descriptors = []

    jsonl_path = f"{BASE_DIR}/results/tmp/{ds}.jsonl"

    files = make_jsonl_chunks(
        df,
        ds,
        args.temperature,
        descriptors,
        jsonl_path,
        args.build_prompt_type,
        args.n_repeats,
        RATE_MIN=RATE_MIN,
        RATE_MAX=RATE_MAX,
        temp_type=args.temp_type,
    )

    final_csv = f"{BASE_DIR}/results/responses/{ds}_{MODEL_NAME}_temp{args.temperature}.csv"

    for fpath in files:
        print(f"Processing {fpath}")
        run_gemma_and_write_csv(
            jsonl_path=fpath,
            df_input=df,
            descriptors=descriptors,
            out_csv=final_csv,
            RATE_MIN=RATE_MIN,
            RATE_MAX=RATE_MAX,
            temperature=args.temperature,
            build_prompt_type=args.build_prompt_type,
            n_repeats=args.n_repeats,
        )


if __name__ == "__main__":
    main()