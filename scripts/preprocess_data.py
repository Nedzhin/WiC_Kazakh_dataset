import pandas as pd
import json
import os
import re

# function for finding the start and end of word in a sentence
def find_flexible_span(word, sentence):
    # Find tokens that start with the base word
    for match in re.finditer(rf'\b({re.escape(word)}\S*)', sentence):
        return match.start(), match.end()
    return -1, -1  # If not found

# Getting the base directory of the script (it is in case someone wanted to run it by themselves)
base_dir = os.path.dirname(os.path.abspath(__file__))

# Constructing path to the raw excel file
excel_path = os.path.join(base_dir, "../raw_data/WiC_kazakh.xlsx")

# Loading the raw excel file
df = pd.read_excel(excel_path) 

output = []
for idx, row in df.iterrows():
    word = row["word"]
    sentence1 = row["example_1"]
    sentence2 = row["example_2"]
    label = bool(row["label"])  # Ensures it's boolean

    start1, end1 = find_flexible_span(word, sentence1)
    start2, end2 = find_flexible_span(word, sentence2)

    entry = {
        "word": word,
        "sentence1": sentence1,
        "sentence2": sentence2,
        "idx": idx,
        "label": label,
        "start1": start1,
        "end1": end1,
        "start2": start2,
        "end2": end2,
        "version": 1.1
    }

    output.append(entry)

# Write to JSONL file
output_path = os.path.join(base_dir, "../processed_data/final_dataset.jsonl")
with open(output_path, "w", encoding="utf-8") as f:
    for entry in output:
        json.dump(entry, f, ensure_ascii=False)
        f.write("\n")
