import os
import re
import json
import pandas as pd

# compiling cleaning regexes
# cleaning sentences: not necessary whitepace, ensures final punctuation
WS_RE = re.compile(r"\s+", flags=re.UNICODE)

def clean_sentence(s: str) -> str:
    """Trim, collapse whitespace, ensure final punctuation."""
    if s is None:
        s = ""
    s = str(s)
    s = WS_RE.sub(" ", s).strip()
    if s and s[-1] not in ".?!":
        s += "."
    return s

# finding spans of kazakh stems with regex
# Allowing Cyrillic letters, Latin fallback, apostrophes, hyphen in suffixes
KZ_LETTERS = r"[A-Za-z\u0400-\u04FF\u2019\u02BC\-]"

def compile_kz_stem_pattern(stem: str) -> re.Pattern:
    """
    Building a regex that matches:
      - the given stem as a word-start, followed by Kazakh-friendly suffixes; AND
      - if the stem ends with 'у' (Kazakh infinitive), also match its truncated form
        without the final 'у' (e.g., 'тұру' -> 'тұр', 'келу' -> 'кел', 'бару' -> 'бар').
    """
    stem = (stem or "").strip()
    if not stem:
        # dummy pattern that never matches, to avoid accidental full mathces
        return re.compile(r"(?!x)x")

    variants = {stem}

    # If infinitive ending with 'у', also allow truncated base without 'у'
    # build vor Verbs as most of them ends with 'y' or 'yy' and their changes in the sentences
    if stem.endswith("у"):
        variants.add(stem[:-1])

    # Escape each variant for safe regex insertion
    escaped = [re.escape(v) for v in variants]
    
    # Alternation between variants
    alternation = "(?:" + "|".join(sorted(escaped, key=len, reverse=True)) + ")"

    # \b ensures we only match at word boundaries
    pattern = rf"\b({alternation}{KZ_LETTERS}*)"
    return re.compile(pattern, flags=re.IGNORECASE | re.UNICODE)

def find_first_span(stem: str, sentence: str):
    """
    Return (start, end) of FIRST occurrence of stem-or-truncated-stem + suffixes; (-1, -1) if none.
    Example:
      stem='тұру' will match 'тұру', 'тұр', 'тұрды', 'тұрады', etc.
    """
    if not stem or not sentence:
        return -1, -1
    m = compile_kz_stem_pattern(stem).search(sentence)
    return (m.start(1), m.end(1)) if m else (-1, -1)

# main part
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    excel_path = os.path.join(base_dir, "../raw_data/Final_WiC_kazakh.xlsx")
    out_dir = os.path.join(base_dir, "../processed_data")
    os.makedirs(out_dir, exist_ok=True)
    jsonl_path = os.path.join(out_dir, "final_dataset_2nd.jsonl")

    # Load Excel
    df = pd.read_excel(excel_path)

    # Column names
    WORD_COL  = "word"
    S1_COL    = "example_1"
    S2_COL    = "example_2"
    LABEL_COL = "label"   # expected as 0.0 / 1.0 floats
    POS_COL   = "POS"     # single POS column for the target word (same POS in both sentences)

    total = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, row in df.iterrows():
            total += 1

            word = str(row.get(WORD_COL, "")).strip()
            s1 = clean_sentence(row.get(S1_COL, ""))
            s2 = clean_sentence(row.get(S2_COL, ""))

            # labels are 0.0 / 1.0
            val = float(row.get(LABEL_COL, 0.0))
            if val not in (0.0, 1.0):
                raise ValueError(f"Unexpected label value at idx {idx}: {val}")
            label = bool(val)

            pos = row.get(POS_COL, None)  # pass-through POS

            start1, end1 = find_first_span(word, s1)
            start2, end2 = find_first_span(word, s2)

            entry = {
                "word": word,
                "sentence1": s1,
                "sentence2": s2,
                "idx": int(idx),
                "label": label,
                "start1": int(start1),
                "end1": int(end1),
                "start2": int(start2),
                "end2": int(end2),
                "pos": pos,          # same POS for both sentences
                "version": 1.1,
            }

            json.dump(entry, f, ensure_ascii=False)
            f.write("\n")

    print(f"Wrote {total} entries to JSONL: {jsonl_path}")
