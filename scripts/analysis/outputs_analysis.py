import os
import re
import json
import glob
import pandas as pd

CSV_DIR = "../results_evaluation/csv_results"  # folder containing prediction CSVs named like: predictions_<model>_<zero|few>_<kk|en>.csv
JSONL_GOLD = "../processed_data/final_dataset_lastE.jsonl"  # gold dataset with word, sentences, label, idx
OUT_JSONL = "../results_evaluation/combined_outputs/combined_predictions_wide.jsonl"
OUT_JSONL_SORTED = "../results_evaluation/combined_outputs/combined_predictions_wide_sorted.jsonl"
OUT_CSV_SORTED = "../results_evaluation/combined_outputs/combined_predictions_wide_sorted.csv"

def to_bool(x):
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "t", "1"}:
        return True
    if s in {"false", "f", "0"}:
        return False
    return None

FNAME_RE = re.compile(
    r"^predictions_(?P<model>.+)_(?P<prompt_mode>zero|few)_(?P<language>kk|en)\.csv$",
    re.IGNORECASE
)

# load gold jsonl
gold_rows = []
with open(JSONL_GOLD, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        # normalize columns we need
        gold_rows.append({
            "idx": int(obj["idx"]),
            "word": obj.get("word"),
            "sentence1": obj.get("sentence1"),
            "sentence2": obj.get("sentence2"),
            "gold": bool(obj.get("label")) if isinstance(obj.get("label"), bool) else (str(obj.get("label")).lower() == "true"),
        })
gold_df = pd.DataFrame(gold_rows).set_index("idx").sort_index()

# read predictions and pivot to wide per idx
pred_frames = []
model_cols = []

for path in glob.glob(os.path.join(CSV_DIR, "predictions_*.csv")):
    fname = os.path.basename(path)
    m = FNAME_RE.match(fname)
    if not m:
        continue  # skip non-matching files
    model = m.group("model")
    prompt_mode = m.group("prompt_mode").lower()
    language = m.group("language").lower()
    colname = f"{model}__{prompt_mode}__{language}"
    model_cols.append(colname)

    df = pd.read_csv(path)
    # make sure idx present and boolean parse
    if "idx" not in df.columns or "pred" not in df.columns:
        continue
    df = df[["idx", "pred"]].copy()
    df["idx"] = df["idx"].astype(int)
    df[colname] = df["pred"].apply(to_bool)
    df = df.drop(columns=["pred"]).set_index("idx")
    pred_frames.append(df)

# merge all prediction columns
if pred_frames:
    preds_wide = pd.concat(pred_frames, axis=1).sort_index()
else:
    preds_wide = pd.DataFrame(index=gold_df.index)

# align to gold indices
preds_wide = preds_wide.reindex(gold_df.index)

# combine with gold + metadata
combined = gold_df.join(preds_wide, how="left")

# compute disagreement metrics
pred_cols = [c for c in combined.columns if c not in ["word", "sentence1", "sentence2", "gold"]]

def row_disagreement_stats(row):
    vals = [row[c] for c in pred_cols if pd.notna(row[c])]
    # convert to strict booleans; ignore None
    bools = [bool(v) for v in vals if v is not None]
    total = len(bools)
    if total == 0:
        return pd.Series({
            "total_models": 0,
            "majority_vote": None,
            "majority_size": 0,
            "disagree_among_models": 0,
            "num_incorrect_vs_gold": None
        })
    n_true = sum(bools)
    n_false = total - n_true
    majority_vote = (n_true >= n_false)  # ties count as True majority for determinism
    majority_size = max(n_true, n_false)
    disagree_among = total - majority_size  # how many disagree with the majority
    # disagreement vs gold: count of predictions not equal to gold
    gold = row["gold"]
    num_incorrect = None
    if pd.notna(gold):
        num_incorrect = sum(1 for v in bools if v != bool(gold))
    return pd.Series({
        "total_models": total,
        "majority_vote": majority_vote,
        "majority_size": majority_size,
        "disagree_among_models": disagree_among,
        "num_incorrect_vs_gold": num_incorrect
    })

stats = combined.apply(row_disagreement_stats, axis=1)
combined2 = pd.concat([combined, stats], axis=1)

# sort: first by most model disagreement, then by most incorrect vs gold (desc), then by idx
combined_sorted = combined2.sort_values(
    by=["disagree_among_models", "num_incorrect_vs_gold", combined2.index.name],
    ascending=[False, False, True]
)

# write JSONL (wide) unsorted and sorted
def df_to_jsonl(df, path):
    records = df.reset_index().to_dict(orient="records")
    with open(path, "w", encoding="utf-8") as f:
      for r in records:
          # ensure JSON-serializable (convert NaNs)
          cleaned = {k: (None if pd.isna(v) else v) for k, v in r.items()}
          f.write(json.dumps(cleaned, ensure_ascii=False) + "\n")

df_to_jsonl(combined2, OUT_JSONL)
df_to_jsonl(combined_sorted, OUT_JSONL_SORTED)

# also save a CSV for quick inspection
combined_sorted.to_csv(OUT_CSV_SORTED, index=True)

OUT_JSONL, OUT_JSONL_SORTED, OUT_CSV_SORTED
