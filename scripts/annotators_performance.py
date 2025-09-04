import pandas as pd
import os
from sklearn.metrics import cohen_kappa_score

# functions to use later
def convert_to_boolean(labels):
    """Convert 'T'/'F' labels to boolean values."""
    for l in labels:
        if l == "T":
            yield True
        elif l == "F":
            yield False
        else:
            raise ValueError(f"Unexpected label: {l}")

# function to load labels
def load_labels(gold_path, ann1_path, ann2_path):
    """Load gold + annotator labels and convert them to boolean."""
    goldL_file = pd.read_excel(gold_path)
    ann1_file = pd.read_excel(ann1_path)
    ann2_file = pd.read_excel(ann2_path)

    gold_labels = [bool(l) for l in goldL_file["label"].tolist()]
    ann1_labels = list(convert_to_boolean(ann1_file["label(T,F)"].tolist()))
    ann2_labels = list(convert_to_boolean(ann2_file["label(T,F)"].tolist()))
    
    return gold_labels, ann1_labels, ann2_labels

def calculate_performance(gold_labels, ann1_labels, ann2_labels, stage="Before"):
    """Calculating annotator performance vs gold and IAA metrics before and after adjudication."""
    # Annotator vs gold
    ann1_perf = sum(g == a for g,a in zip(gold_labels, ann1_labels)) / len(gold_labels)
    ann2_perf = sum(g == a for g,a in zip(gold_labels, ann2_labels)) / len(gold_labels)

    print(f"\n=== PERFORMANCE OF THE ANNOTATORS ({stage} Adjudication) ===")
    print("Length of the dataset {stage} adjudication:", len(gold_labels))
    print("Annotator 1 results in percent:", ann1_perf * 100)
    print("Annotator 2 results in percent:", ann2_perf * 100)
    print("Human Eval (average):", (ann1_perf*100 + ann2_perf*100) / 2)

    # If before adjudication, it will calculate disagreements + IAA
    if stage == "Before":
        disagreement_indexes = []
        annotators_disagreement = []

        for i, (g, a1, a2) in enumerate(zip(gold_labels, ann1_labels, ann2_labels)):
            if g != a1 or g != a2:
                disagreement_indexes.append((i+2, 0 if g == a1 else 1, 0 if g == a2 else 1))
            if a1 != a2:
                annotators_disagreement.append(i+2)

        print("\nNUMBER OF DISAGREEMENTS")
        print("Disagreements with gold (at least one annotator):", len(disagreement_indexes))
        print("Annotators' disagreement:", len(annotators_disagreement))

        # IAA
        iaa_raw = sum(a1 == a2 for a1, a2 in zip(ann1_labels, ann2_labels)) / len(gold_labels)
        kappa = cohen_kappa_score(ann1_labels, ann2_labels)
        print("\nInter-annotator agreement (raw):", iaa_raw * 100, "%")
        print(f"Cohen's kappa: {kappa:.3f}")

# Main function
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Before adjudication
    gold_before = os.path.join(base_dir, "../raw_data/Before_adjudication_WiC_kazakh.xlsx")
    ann1_before = os.path.join(base_dir, "../annotators_results/annotator_1_before_adjudication_WiC_kazakh.xlsx")
    ann2_before = os.path.join(base_dir, "../annotators_results/annotator_2_before_adjudication_WiC_kazakh.xlsx")

    gold_labels, ann1_labels, ann2_labels = load_labels(gold_before, ann1_before, ann2_before)
    calculate_performance(gold_labels, ann1_labels, ann2_labels, stage="Before")

    # After adjudication
    gold_after = os.path.join(base_dir, "../raw_data/Final_WiC_kazakh.xlsx")
    ann1_after = os.path.join(base_dir, "../annotators_results/annotator_1_WiC_kazakh.xlsx")
    ann2_after = os.path.join(base_dir, "../annotators_results/annotator_2_WiC_kazakh.xlsx")

    gold_labels, ann1_labels, ann2_labels = load_labels(gold_after, ann1_after, ann2_after)
    calculate_performance(gold_labels, ann1_labels, ann2_labels, stage="After")
