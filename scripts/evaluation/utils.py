import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

def load_jsonl(path: str) -> pd.DataFrame:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def parse_bool(text: str) -> bool:
    t = (text or "").strip().lower()
    if t == "true" or t.startswith("true"):
        return True
    if t == "false" or t.startswith("false"):
        return False
    raise ValueError(f"Non-boolean output: {text!r}")

def _as_bool_array(x):
    return np.asarray(x, dtype=bool)

def compute_metrics(y_true, y_pred):
    """
    Returns a dict with:
      - accuracy
      - cohens_kappa
      - precision, recall (binary, positive=True)
      - f1 (binary), macro_f1, weighted_f1
      - confusion counts: tp, tn, fp, fn
    All continuous metrics are fractions in [0,1].
    """
    y_true = _as_bool_array(y_true)
    y_pred = _as_bool_array(y_pred)

    if len(y_true) == 0:
        return {
            "accuracy": 0.0, "cohens_kappa": 0.0,
            "precision": 0.0, "recall": 0.0,
            "f1": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0,
            "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }

    acc = float((y_true == y_pred).mean())

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    # Binary metrics (positive class=True)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)

    # Macro/weighted F1
    f1_macro    = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    kappa = float(cohen_kappa_score(y_true, y_pred))

    return {
        "accuracy": acc,
        "cohens_kappa": kappa,
        "precision": float(prec),
        "recall":    float(rec),
        "f1":        float(f1),
        "macro_f1":  float(f1_macro),
        "weighted_f1": float(f1_weighted),
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }