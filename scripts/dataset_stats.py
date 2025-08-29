import os
import pandas as pd
import numpy as np
import time
import stanza

nlp = stanza.Pipeline(lang='kk', processors='tokenize,pos,lemma')

# Paths
base_dir = os.path.dirname(os.path.abspath(__file__))
jsonl_path = os.path.join(base_dir, "../processed_data/final_dataset_lastE.jsonl")

# Load JSONL dataset
df = pd.read_json(jsonl_path, lines=True)

# Just in case, ensure that stanza is loaded
# if nlp:
#   print("yeah, downloaded")

# col names
s1_col, s2_col, label_col, pos_col, word_col = "sentence1", "sentence2", "label", "pos", "word"

# Getting sentences and labels
contexts = list(df[s1_col].astype(str)) + list(df[s2_col].astype(str))
labels_bool = df[label_col].astype(bool)

# counting number of true/false labels
true_count = int(labels_bool.sum())
false_count = len(labels_bool) - true_count

# Tokenization with Stanza(as it is multilangual and supports Kazakh)
# calculating the time spent on teknization
start_time = time.time()
all_tokens = []
lengths = []

for s in contexts:
    doc = nlp(s)
    tokens = [w.text.lower() for sent in doc.sentences for w in sent.words]
    all_tokens.extend(tokens)
    lengths.append(len(tokens))

unique_words_contexts = len(set(all_tokens))
avg_context_len = float(np.mean(lengths))

# taget unique words
unique_words = df[word_col].nunique()
# counting POS tags
pos_counts = df[pos_col].value_counts(dropna=False) #.to_dict()

end_time = time.time()
elapsed = end_time - start_time

# Printing stats
print("=== WiC-Kazakh Stats (Stanza) ===")
print(f"Instances (pairs): {len(df)}")
print(f"Unique target words: {unique_words}")
print(f"Unique words in context: {unique_words_contexts}")
print(f"Avg context length (per sentence): {avg_context_len:.2f}")
print(f"True labels: {true_count}")
print(f"False labels: {false_count}")

print("\nPOS Tag Counts:")
for pos, count in pos_counts.items():
    print(f"{pos} number: {count}")

print(f"\nTime taken: {elapsed:.2f} seconds")
import matplotlib.pyplot as plt
# Plot POS tag counts
plt.figure(figsize=(10, 6))
pos_counts.plot(kind='bar')
plt.title('POS Tag Counts')
plt.xlabel('POS Tag')
plt.ylabel('Count')
plt.tight_layout()
plt.show()