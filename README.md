# 🧠 WiC-Kazakh Dataset

## Overview

**WiC-Kazakh** is a manually constructed dataset designed to evaluate contextual understanding in the Kazakh language using the **Word-in-Context (WiC)** task format (Pilehvar & Camacho-Collados, 2019). The task involves determining whether a target word carries the same meaning in two different sentence contexts. This dataset aims to provide a benchmark for evaluating the performance of multilingual and Kazakh-specific language models, especially in a low-resource linguistic setting.

Each entry in the dataset consists of:
- A target **word**,
- Its **part-of-speech** tag (Noun, Verb, or Adjective),
- Two **example sentences** where the word appears in slightly different contexts,
- A **label** indicating whether the meaning of the word is the same in both sentences (`TRUE`) or not (`FALSE`),
- Character-level **start and end positions** of the word form in each sentence,
- Metadata such as `idx` and versioning.

## Data Collection Process

The dataset was manually compiled over 1–2 weeks, inspired by the original **English WiC dataset** structure. The following steps were followed:

1. **Target Word Selection**  
   - Words were chosen using the **Oxford Kazakh Dictionary** and **Sordid.kz**, ensuring common and semantically flexible usage.

2. **Sentence Sourcing**  
   - Sentences were collected from publicly available and linguistically diverse sources:
     - [Almaty Corpus](http://web-corpora.net/KazakhCorpus/)
     - [SketchEngine (Kazakh Corpus)](https://www.sketchengine.eu)
     - [Kazakh Wikipedia](https://kk.wikipedia.org/)
     - [Bilim-all.kz](https://bilim-all.kz)

3. **Balancing**  
   - A total of **506 sentence pairs** were compiled, with roughly equal numbers of `TRUE` and `FALSE` labels.
   - Most examples involve **Nouns (N)** and **Verbs (V)**, with a smaller portion of **Adjectives (A)** for semantic variety.

## Preprocessing

The original dataset was stored in Excel format with the following columns:
- `word`, `POS`, `label`, `example_1`, `example_2`

Using a custom Python script (`scripts/preprocess_data.py`), the dataset was converted into **JSONL** format for compatibility with common NLP evaluation tools. The script also:
- Located the **morphologically inflected** form of the word in each sentence using regex-based flexible matching,
- Recorded the **character positions** (`start1`, `end1`, `start2`, `end2`) for each word form,
- Added metadata fields like `idx` and `version`.

## Format (JSONL Example)

```json
{
  "word": "бас",
  "sentence1": "Ол үстелдің басында отырды.",
  "sentence2": "Оның басы ауырды.",
  "idx": 0,
  "label": false,
  "start1": 13,
  "end1": 20,
  "start2": 5,
  "end2": 9,
  "version": 1.1
}
```

## References

- Pilehvar, M.T., & Camacho-Collados, J. (2019). [WiC: The Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations](https://aclanthology.org/N19-1128/). *Proceedings of NAACL-HLT 2019.*
- Oxford Kazakh Dictionary.
- Sordid.kz Online Word Tool.
- Almaty Corpus, SketchEngine (Kazakh), Wikipedia (kk), Bilim-all.kz.

## License and Use

This dataset is released for **research and educational purposes only**. Please cite the sources above if using this dataset for academic publications or benchmarking tasks.
