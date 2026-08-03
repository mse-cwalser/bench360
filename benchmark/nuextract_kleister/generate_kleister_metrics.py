import pandas as pd
import json


def parse_json_col(col_val):
    try:
        return json.loads(col_val)
    except:
        return {}


def compute_em(df_pred, target_fields):
    """Computes the Exact Match (EM) score for predictions against ground truth."""
    em_scores = {f: [] for f in target_fields}
    for _, row in df_pred.iterrows():
        gt = parse_json_col(row['ground_truth'])
        pred = parse_json_col(row['prediction'])

        for f in target_fields:
            gt_val = gt.get(f)
            pred_val = pred.get(f)

            # Handle lists (like parties) vs strings
            if isinstance(gt_val, list) and isinstance(pred_val, list):
                match = 1 if sorted([str(x).strip() for x in gt_val]) == sorted(
                    [str(x).strip() for x in pred_val]) else 0
            elif not isinstance(gt_val, list) and not isinstance(pred_val, list):
                match = 1 if str(gt_val).strip() == str(pred_val).strip() else 0
            else:
                match = 1 if gt_val == pred_val else 0

            em_scores[f].append(match)

    avg_em_per_field = [sum(em_scores[f]) / len(em_scores[f]) for f in target_fields if len(em_scores[f]) > 0]
    avg_field_em = sum(avg_em_per_field) / len(avg_em_per_field) if avg_em_per_field else 0.0
    return avg_field_em


def generate_standalone_metrics():
    print("Loading evaluation summary...")
    with open('evaluation_summary.json', 'r') as f:
        eval_summary = json.load(f)

    print("Loading emissions data...")
    emissions_df = pd.read_csv('emissions.csv')
    emissions = emissions_df.iloc[-1]

    print("Loading predictions and computing Exact Match (EM)...")
    df_pred = pd.read_csv('kleister_predictions.csv')
    target_fields = ["effective_date", "jurisdiction", "party", "term"]
    avg_field_em = compute_em(df_pred, target_fields)

    print("Calculating average F1 score...")
    f1_per_field = [eval_summary['field_metrics'][f]['f1_score'] for f in target_fields]
    avg_field_f1 = sum(f1_per_field) / len(f1_per_field)

    # Compile the final metrics dictionary
    metrics = {
        'model_name': 'NuExtract',
        'task': 'kleister_nda',
        'num_queries': eval_summary['total_documents_evaluated'],
        'total_generation_time_s': emissions['duration'],
        'avg_power_w': emissions['cpu_power'] + emissions['gpu_power'] + emissions['ram_power'],
        'total_energy_wh': emissions['energy_consumed'] * 1000,
        'avg_cpu_util_pct': emissions['cpu_utilization_percent'],
        'avg_gpu_util_pct': emissions['gpu_utilization_percent'],
        'avg_ram_mb': emissions['ram_used_gb'] * 1024,
        'avg_field_f1': avg_field_f1,
        'avg_field_em': avg_field_em
    }

    # Save to CSV
    df_out = pd.DataFrame([metrics])
    output_file = 'kleister_standalone_metrics_output.csv'
    df_out.to_csv(output_file, index=False)

    print(f"\nMetrics successfully saved to {output_file}")
    print("\nGenerated Metrics:")
    print("-" * 40)
    for k, v in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    generate_standalone_metrics()