# 🧠 WiC-Kazakh Dataset

## Overview

Welcome to the **WiC-Kazakh** dataset, designed to evaluate the contextual understanding of the Kazakh language. This dataset follows the **Word-in-Context (WiC)** task format, inspired by the work of Pilehvar & Camacho-Collados (2019). The goal of this task is to determine whether a target word has the same meaning in two different sentence contexts. Our dataset aims to fill a gap in Kazakh language resources by providing a benchmark for multilingual and Kazakh-specific language models, particularly in low-resource linguistic environments.

## Data Construction

The **WiC-Kazakh** dataset was manually curated to include diverse instances of words used in varying contexts. Here's a breakdown of the data creation process:

1. **Word Selection**: We selected target words commonly used in the Kazakh language. Each word is accompanied by its **part-of-speech tag** (noun, verb, adjective).
   
2. **Sentence Collection**: For each word, we collected two example sentences that use the word in different contexts. The goal is to challenge the model to discern whether the word carries the same meaning in both sentences.

3. **Labeling**: We manually labeled the pairs of sentences as either **True** or **False**. If the word's meaning is the same in both contexts, the label is **True**; otherwise, it's **False**.

4. **Positioning Information**: For each word in a sentence, we recorded its **start and end positions** at the character level. This allows for easier tokenization and extraction when training models.

5. **Metadata**: Each entry includes metadata, such as a unique **index (idx)** and version information.

## Dataset Structure

The dataset is organized into several key directories and files:

1. **`raw_data/`**: Contains the raw, unprocessed entries. These are the initial datasets with word-context pairs.
   
2. **`processed_data/`**: This directory holds the cleaned and processed version of the dataset, ready for use in training and evaluation tasks.
   
3. **`scripts/`**: A collection of Python scripts used for preprocessing, data analysis, and model evaluation. These scripts automate many tasks, including text cleaning, feature extraction, and dataset formatting.
   
4. **`annotators_results/`**: Contains the results of the manual annotation process. It provides insights into the labeling and adjudication process, helping to ensure consistency and accuracy.
   
5. **`requirements.txt`**: This file lists the necessary Python packages for running the scripts and models in this repository. It ensures that your environment is properly set up.

6. **`results_evaluation/`**: Holds the evaluation results for different models tested on the dataset. These results include metrics like **accuracy**, **F1 score**, and **Cohen’s Kappa**, helping to assess the model’s performance.

7. **`.env`**: Stores environment-specific variables, such as paths and API keys, used during the data preprocessing and model training phases.

8. **`.gitignore`**: Specifies which files should be ignored by version control (Git), ensuring that temporary files or sensitive data are not tracked.

9. **`README.md`**: This file provides an overview of the dataset, its construction process, and the directory structure.

## Experiment Setup

The **WiC-Kazakh** dataset was used in a series of experiments to evaluate the performance of different **large language models (LLMs)** in the Kazakh language. The models were assessed on the **Word-in-Context (WiC)** task, where the models needed to determine if the target word’s meaning remained consistent across two different sentences.

### Models Evaluated:
- **GPT-4**: A state-of-the-art LLM known for its multilingual capabilities.
- **Gemini 1.5 Pro & Flash**: Another competitive model that emphasizes long-context reasoning and multilingual support.
- **Llama-3 (via Ollama)**: A more localized model with strong performance in specific languages.

### Experiment Settings:
- **Zero-shot & Few-shot Prompting**: Models were tested with both zero-shot (no task-specific training) and few-shot (limited examples in prompts) approaches in both **Kazakh** and **English**.
- **Evaluation Metrics**: The models were evaluated using metrics such as **accuracy**, **F1 score**, and **Cohen’s Kappa** to measure performance in terms of both overall correctness and the balance between precision and recall.
- **Human Evaluation**: To ensure high-quality annotations, a human evaluation process was employed, including inter-annotator agreement (IAA).

## How to Use

To get started, follow these steps:

1. **Install dependencies**: Run the following command to install the necessary Python packages:
   ```bash
   pip install -r requirements.txt
