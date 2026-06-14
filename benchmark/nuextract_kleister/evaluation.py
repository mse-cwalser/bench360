import csv
import json
import sys
import os

# Dynamically add the parent directory to sys.path to import kleister_utils.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from kleister_utils import compute_kleister_metrics
except ImportError:
    print("Error: Could not import kleister_utils. Ensure it is located at ../kleister_utils.py")
    sys.exit(1)

# --- Configuration ---
CSV_PATH = "kleister_predictions.csv"
OUTPUT_FILE = "evaluation_summary.json"
TARGET_FIELDS = ["effective_date", "jurisdiction", "party", "term"]


def calculate_f1(tp: float, fp: float, fn: float):
    """Calculates precision, recall, and f1 score given tp, fp, fn."""
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


def evaluate_predictions():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Please run the NuExtract script first.")
        sys.exit(1)

    # Dictionaries to aggregate total TP, FP, FN across the whole dataset
    agg_metrics = {
        "doc_tp": 0.0, "doc_fp": 0.0, "doc_fn": 0.0
    }
    for field in TARGET_FIELDS:
        agg_metrics[f"{field}_tp"] = 0.0
        agg_metrics[f"{field}_fp"] = 0.0
        agg_metrics[f"{field}_fn"] = 0.0

    total_docs = 0

    print(f"Loading predictions from {CSV_PATH}...")

    with open(CSV_PATH, mode="r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            total_docs += 1
            try:
                # Parse the JSON strings saved in the CSV
                ground_truth = json.loads(row["ground_truth"])
                prediction = json.loads(row["prediction"])
            except json.JSONDecodeError as e:
                print(f"Skipping document {row.get('filename', 'Unknown')} due to JSON parsing error: {e}")
                continue

            # Calculate document-level metrics using the provided utility
            doc_metrics = compute_kleister_metrics(ground_truth, prediction, TARGET_FIELDS)
            print(doc_metrics, end="\r")

            # Accumulate overall document stats
            agg_metrics["doc_tp"] += doc_metrics.get("doc_tp", 0)
            agg_metrics["doc_fp"] += doc_metrics.get("doc_fp", 0)
            agg_metrics["doc_fn"] += doc_metrics.get("doc_fn", 0)

            # Accumulate entity-specific stats
            for field in TARGET_FIELDS:
                agg_metrics[f"{field}_tp"] += doc_metrics.get(f"{field}_tp", 0)
                agg_metrics[f"{field}_fp"] += doc_metrics.get(f"{field}_fp", 0)
                agg_metrics[f"{field}_fn"] += doc_metrics.get(f"{field}_fn", 0)

    # --- Compute Final Corpus-Level Metrics ---
    final_results = {
        "total_documents_evaluated": total_docs,
        "overall_metrics": {},
        "field_metrics": {}
    }

    # Overall Metrics
    doc_prec, doc_rec, doc_f1 = calculate_f1(
        agg_metrics["doc_tp"], agg_metrics["doc_fp"], agg_metrics["doc_fn"]
    )
    final_results["overall_metrics"] = {
        "precision": round(doc_prec, 4),
        "recall": round(doc_rec, 4),
        "f1_score": round(doc_f1, 4)
    }

    # Per-Field Metrics
    for field in TARGET_FIELDS:
        f_prec, f_rec, f_f1 = calculate_f1(
            agg_metrics[f"{field}_tp"], agg_metrics[f"{field}_fp"], agg_metrics[f"{field}_fn"]
        )
        final_results["field_metrics"][field] = {
            "precision": round(f_prec, 4),
            "recall": round(f_rec, 4),
            "f1_score": round(f_f1, 4)
        }

    # Save to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        json.dump(final_results, outfile, indent=4)

    # Display results
    print("\n==========================================")
    print("EVALUATION SUMMARY")
    print("==========================================")
    print(f"Documents Evaluated : {total_docs}")
    print(f"Overall Precision   : {doc_prec:.4f}")
    print(f"Overall Recall      : {doc_rec:.4f}")
    print(f"Overall F1 Score    : {doc_f1:.4f}")
    print("------------------------------------------")
    for field, metrics in final_results["field_metrics"].items():
        print(f"Field: {field}")
        print(f"  -> Precision : {metrics['precision']:.4f}")
        print(f"  -> Recall    : {metrics['recall']:.4f}")
        print(f"  -> F1 Score  : {metrics['f1_score']:.4f}")
    print("==========================================")
    print(f"Detailed results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    evaluate_predictions()