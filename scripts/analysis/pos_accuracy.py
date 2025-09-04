import os
import re
import json
import glob
import pandas as pd

# path to file names
JSONL_PATH = "../processed_data/final_dataset_lastE.jsonl"
CSV_DIR    = "../results_evaluation/csv_results"  # folder containing prediction to each model with different prompt modes and languages
OUTPUT_SUMMARY_WIDE = "../results_evaluation/accuracy_per_POS/pos_accuracy_by_model_wide.csv"
OUTPUT_SUMMARY_LONG = "../results_evaluation/accuracy_per_POS/pos_accuracy_by_model_long.csv"

# helper functions 
def load_jsonl_to_df(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    df = pd.DataFrame(rows)
    if "idx" in df.columns:
        df["idx"] = df["idx"].astype(int)
    return df

def to_bool(x):
    """
    Robust boolean parser for strings/values like: True/False, 'TRUE'/'FALSE', 'true'/'false', 1/0, etc.
    """
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in {"true", "t", "1"}:
        return True
    if s in {"false", "f", "0"}:
        return False
    try:
        return bool(x)
    except Exception:
        return None

# regex to parse: predictions_<model>_<prompt_mode>_<language>.csv
FNAME_RE = re.compile(
    r"^predictions_(?P<model>.+)_(?P<prompt_mode>zero|few)_(?P<language>kk|en)\.csv$",
    re.IGNORECASE
)

# loading gold and POS labels
gold_df = load_jsonl_to_df(JSONL_PATH)
# keep only idx, pos
pos_map = gold_df[["idx", "pos"]].copy()

# iterate over prediction CSVs
records = []  # long-form rows: one row per (model, mode, lang, POS) with accuracy
overall_rows = []  # to add an overall-accuracy row per (model, mode, lang)

for path in glob.glob(os.path.join(CSV_DIR, "predictions_*.csv")):
    fname = os.path.basename(path)
    m = FNAME_RE.match(fname)
    if not m:
        # skip files that don't match the pattern
        continue

    model       = m.group("model")
    prompt_mode = m.group("prompt_mode")
    language    = m.group("language")

    df = pd.read_csv(path)
    # normalize idx
    if "idx" not in df.columns:
        raise ValueError(f"'idx' column not found in {fname}")

    df["idx"] = df["idx"].astype(int)

    # normalizing gold/pred to booleans and recompute correctness
    df["gold_bool"] = df["gold"].apply(to_bool)
    df["pred_bool"] = df["pred"].apply(to_bool)
    df["correct_recalc"] = (df["gold_bool"] == df["pred_bool"])

    # attach POS
    merged = df.merge(pos_map, on="idx", how="left", validate="many_to_one")

    # overall accuracy
    overall_acc = merged["correct_recalc"].mean()
    overall_n   = merged.shape[0]
    overall_rows.append({
        "model": model,
        "prompt_mode": prompt_mode,
        "language": language,
        "pos": "OVERALL",
        "accuracy": overall_acc,
        "n_items": overall_n
    })

    # per-POS accuracy
    grp = merged.groupby("pos", dropna=False)
    for pos_tag, g in grp:
        acc = g["correct_recalc"].mean()
        n   = g.shape[0]
        records.append({
            "model": model,
            "prompt_mode": prompt_mode,
            "language": language,
            "pos": pos_tag,
            "accuracy": acc,
            "n_items": n
        })

# combining long tables
long_df = pd.DataFrame(records)
overall_df = pd.DataFrame(overall_rows)
long_with_overall = pd.concat([long_df, overall_df], ignore_index=True)

# nice sorting
long_with_overall["model"] = long_with_overall["model"].astype(str)
long_with_overall["prompt_mode"] = pd.Categorical(
    long_with_overall["prompt_mode"], categories=["zero", "few"], ordered=True
)
long_with_overall["language"] = pd.Categorical(
    long_with_overall["language"], categories=["kk", "en"], ordered=True
)

# saving LONG format (one row per model/mode/lang/POS)
long_with_overall.sort_values(["model", "prompt_mode", "language", "pos"], inplace=True)
long_with_overall.to_csv(OUTPUT_SUMMARY_LONG, index=False)

# making a WIDE pivot: rows = model/mode/lang, columns = POS (plus OVERALL), values = accuracy (%)
wide = long_with_overall.pivot_table(
    index=["model", "prompt_mode", "language"],
    columns="pos",
    values="accuracy",
    aggfunc="mean"
)

#Converting to percentage with 2 decimals for readability
wide_pct = (wide * 100).round(2)
wide_pct = wide_pct.sort_index()
wide_pct.to_csv(OUTPUT_SUMMARY_WIDE)

#printing a preview
print("=== POS accuracy by model/mode/lang (percent) ===")
print(wide_pct.fillna("—").to_string())
print(f"\nSaved wide table to: {OUTPUT_SUMMARY_WIDE}")
print(f"Saved long table to: {OUTPUT_SUMMARY_LONG}")
