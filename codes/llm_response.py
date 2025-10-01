from openai import OpenAI
from rdkit import Chem
import json
import pubchempy as pcp
from openai import OpenAI
from utils.ds_utils import get_descriptors
import pandas as pd
from typing import Any, Dict, List, Optional
import os
import numpy as np
from utils.config import BASE_DIR 

# ---------------- Config ----------------
MODEL_NAME = "gpt-4.1-nano"   # change if needed
RATE_MIN, RATE_MAX = -1.0, 1.0
INCLUDE_CONFIDENCE = False
DS_NAME = "sagar2023"         # your dataset key for get_descriptors()
INPUT_TYPE = 'isomericsmiles'  # 'isomericsmiles' or 'cid'


# Initialize OpenAI client (make sure OPENAI_API_KEY is in your environment)
client = OpenAI()


def _clean_cell(x) -> str:
    """Return a safe, stripped string; NaN/None -> ''."""
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
    if not smi:
        return  None
    return smi



   
def build_prompt(smiles: str, descriptors: List[str],
                 rate_min: float, rate_max: float,
                 include_confidence: bool = False) -> str:
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


def call_llm(prompt: str, model: str = MODEL_NAME,temperature: float=0) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an olfactory rater. Output ONLY valid JSON."},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        response_format={"type": "json_object"},
    )
    return (resp.choices[0].message.content or "").strip()




def validate_response(resp_text: str, descriptors: List[str],
                      rate_min: float, rate_max: float,
                      include_confidence: bool) -> Dict[str, float]:
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




def score_row(row: pd.Series, descriptors: List[str],temperature: float) -> Dict[str, Any]:
    cid = row["cid"]
    smi = row_smiles(row)
    prompt = build_prompt(smi, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
    llm_text = call_llm(prompt, model=MODEL_NAME,temperature=temperature)
    scores = validate_response(llm_text, descriptors, RATE_MIN, RATE_MAX, INCLUDE_CONFIDENCE)
    out = {"participant_id": row["participant_id"], "cid": cid, INPUT_TYPE: smi}
    out.update(scores)
    return out




def main(input_csv: str, output_csv: str, debug: bool = False,temperature: float=0):
    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set; export it or pass api_key=... to OpenAI().")

    df = pd.read_csv(input_csv)
    #I want to get unique CIDs only
    descriptors = get_descriptors(DS_NAME)
    if debug:
        df = df.head(5)
    
    
    
    results: List[Dict[str, Any]] = []
    for idx, row in df.iterrows():
        try:
            res = score_row(row, descriptors,temperature)
            results.append(res)
            print(f"OK  row {idx}  CID {row['cid']}, {res}")
        except Exception as e:
            print(f"ERR row {idx} CID {row.get('cid')}: {e}")
            # still record basic info so row alignment is easier later
            results.append({"idx":idx, "participant_id": row.get("participant_id"),
                            "cid": row.get("cid"),"temperature": temperature,
                            INPUT_TYPE: row_smiles(row) or "",
                            "error": str(e)})

    # Build output DataFrame (wide: one row per (participant_id, cid))
    cols = ["idx", "participant_id", "cid","temperature", INPUT_TYPE] + descriptors + (["confidence"] if INCLUDE_CONFIDENCE else []) + (["error"])
    # Only include "error" column if any errors exist
    include_error = any(("error" in r) for r in results)
    if not include_error:
        cols = [c for c in cols if c != "error"]

    out_df = pd.DataFrame(results)
    out_df = out_df.reindex(columns=[c for c in cols if c in out_df.columns])
    out_df.to_csv(output_csv, index=False)
    print(f"Saved: {output_csv}")

if __name__ == "__main__":
    # Example: change paths as needed
    for temp in [0,1]:
        for ds in ['sagar2023','keller2016']:
            INPUT_CSV = f'{BASE_DIR}/datasets/{ds}/{ds}_data.csv'
            OUTPUT_CSV = f"{BASE_DIR}/llm_responses/{ds}_odor_llmmm_scores_temp{temp}.csv"
            main(INPUT_CSV, OUTPUT_CSV,debug=True,temperature=temp)













