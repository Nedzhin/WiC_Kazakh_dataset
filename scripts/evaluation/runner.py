import os
import time
import argparse
import pandas as pd

# loading environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

from prompts import PROMPTS
from utils import load_jsonl, parse_bool, compute_metrics
from models.openai_model import OpenAIChat
from models.gemini_model import GeminiChat
from models.llama_model import OllamaChat

def append_metrics_row(xlsx_path: str, row: dict):
    """
    Append a single metrics row to an Excel file (creates it if missing).
    Columns are inferred from the dict keys; future rows should reuse same keys.
    """
    df_row = pd.DataFrame([row])
    if os.path.exists(xlsx_path):
        try:
            old = pd.read_excel(xlsx_path)
            out = pd.concat([old, df_row], ignore_index=True)
        except Exception:
            out = df_row
    else:
        out = df_row
    out.to_excel(xlsx_path, index=False)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    metrics_results_path = "../../results_evaluation/metrics_log.xlsx"

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../processed_data/final_dataset_lastE.jsonl",
                    help="Path to WiC JSONL.")
    ap.add_argument("--provider", required=True,
                    choices=["openai", "gemini", "ollama"])
    ap.add_argument("--model", default=None,
                    help="openai(gpt-4o), gemini(gemini-1.5-pro|gemini-1.5-flash), ollama(llama3 or your quant).")
    ap.add_argument("--mode", choices=["zero","few"], default="few",
                    help="Prompt mode.")
    ap.add_argument("--lang", choices=["kk","en"], default="kk",
                    help="Prompt language.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Evaluate first N items only.")
    ap.add_argument("--outfile", default="predictions.csv",
                    help="CSV path to save predictions.")
    ap.add_argument("--metrics_xlsx", default=metrics_results_path,
                    help="Excel file to append one row of metrics for this run.")
    args = ap.parse_args()

    # picking model based on provider
    if args.provider == "openai":
        model = args.model or "gpt-4o"
        infer = OpenAIChat(model=model).infer
    elif args.provider == "gemini":
        model = args.model or "gemini-1.5-pro"
        infer = GeminiChat(model=model).infer
    else:
        model = args.model or "llama3"
        infer = OllamaChat(model=model).infer

    # data
    df = load_jsonl(args.data)
    if args.limit:
        df = df.head(args.limit).copy()

    template = PROMPTS[(args.mode, args.lang)]

    detailed_rows = []
    preds, golds, times = [], [], []

    for _, row in df.iterrows():
        s1, s2, w = row["sentence1"], row["sentence2"], row["word"]
        gold = bool(row["label"])
        idx_val = row.get("idx", _)

        prompt = template.format(sentence1=s1, sentence2=s2, target_word=w)

        t0 = time.perf_counter()
        raw = infer(prompt)
        dt = time.perf_counter() - t0

        try:
            pred = parse_bool(raw)
        except Exception:
            pred = False  # fallback
            print("the word: ", row["idx"], row["word"])
            print("The message which could not parsed: ", raw)

        preds.append(pred)
        golds.append(gold)
        times.append(dt)

        detailed_rows.append({
            "idx": int(idx_val),
            "word": w,
            "sentence1": s1,
            "sentence2": s2,
            "gold": bool(gold),
            "pred": bool(pred),
            "raw_output": raw,
            "latency_s": round(dt, 6),
        })

    # Save per-item predictions BEFORE metrics
    results_df = pd.DataFrame(detailed_rows)
    results_df["correct"] = results_df["gold"] == results_df["pred"]

    # CSV (predictions)
    csv_output_path = "../../results_evaluation/csv_results/" + args.outfile
    results_df.to_csv(csv_output_path, index=False, encoding="utf-8")
    print(f"Saved CSV predictions to: {csv_output_path}")

    # Compute metrics
    met = compute_metrics(golds, preds)
    avg_latency = sum(times) / len(times) if times else 0.0

    # Prepare metrics row to append
    metrics_row = {
        "provider": args.provider,
        "model": model,
        "prompt_mode": args.mode,
        "language": args.lang,
        "items": len(df),
        "accuracy": met["accuracy"],
        "f1": met["f1"],
        "cohens_kappa": met["cohens_kappa"],
        "macro_f1": met["macro_f1"],
        "weighted_f1": met["weighted_f1"],
        "precision": met["precision"],
        "recall": met["recall"],
        "tp": met["tp"], "tn": met["tn"], "fp": met["fp"], "fn": met["fn"],
        "avg_latency_s": avg_latency,
    }

    append_metrics_row(args.metrics_xlsx, metrics_row)
    print(f"Appended metrics to: {args.metrics_xlsx}")

    # Console report
    print("\n=== WiC Evaluation ===")
    print(f"Provider: {args.provider} | Model: {model}")
    print(f"Prompt mode: {args.mode} | Language: {args.lang}")
    print(f"Items: {len(df)}")
    print(f"Accuracy:       {met['accuracy']:.4f}")
    print(f"Cohen's κ:      {met['cohens_kappa']:.4f}")
    print(f"Macro F1:       {met['macro_f1']:.4f}   Weighted F1: {met['weighted_f1']:.4f}")
    print(f"P: {met['precision']:.4f}  R: {met['recall']:.4f}  F1: {met['f1']:.4f}")
    print(f"TP={met['tp']}  TN={met['tn']}  FP={met['fp']}  FN={met['fn']}")
    print(f"Avg latency:    {avg_latency:.3f}s")

if __name__ == "__main__":
    main()

