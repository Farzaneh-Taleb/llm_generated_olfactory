# export_dolma_tokens.py
import argparse, os
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dolma_name", default="v1_6-sample")
    ap.add_argument("--sources", default="")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_docs", type=int, default=50000)
    ap.add_argument("--tokenizer", default="gpt2")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "</s>"})

    keep_sources = {s.strip() for s in args.sources.split(",") if s.strip()}

    ds = load_dataset(
        "allenai/dolma",
        name=args.dolma_name,
        split="train",
        streaming=True,
    )

    tokens = []
    for i, ex in enumerate(tqdm(ds, total=args.max_docs)):
        if i >= args.max_docs:
            break
        if keep_sources and ex.get("source") not in keep_sources:
            continue
        text = ex.get("text", "")
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        ids.append(tokenizer.eos_token_id)
        tokens.extend(ids)

    arr = np.array(tokens, dtype=np.int32)
    out_path = os.path.join(
        args.out_dir, f"dolma_{args.dolma_name}_tokens.npy"
    )
    np.save(out_path, arr)

    print("Saved:", out_path)
    print("Num tokens:", arr.shape[0])

if __name__ == "__main__":
    main()