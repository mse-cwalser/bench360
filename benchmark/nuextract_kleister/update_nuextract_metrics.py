import pandas as pd
import json
import ast
import string
import os
from thefuzz import fuzz


def normalize_value(val):
    if val is None:
        return []
    if isinstance(val, list):
        res = []
        for v in val:
            res.extend(normalize_value(v))
        return res
    if isinstance(val, (str, int, float)):
        s = str(val).upper().replace('_', ' ')
        s = s.translate(str.maketrans('', '', string.punctuation))
        s = " ".join(s.split())
        return [s]
    return []


def extract_and_normalize(json_str, keys_to_extract):
    if pd.isna(json_str): return {}
    json_str = str(json_str).strip()
    start_idx = json_str.find('{')
    end_idx = json_str.rfind('}')
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        json_str = json_str[start_idx:end_idx + 1]
    try:
        data = json.loads(json_str)
        if not isinstance(data, dict): return {}
    except Exception:
        try:
            cleaned_str = json_str.replace('true', 'True').replace('false', 'False').replace('null', 'None')
            data = ast.literal_eval(cleaned_str)
            if not isinstance(data, dict): return {}
        except Exception:
            return {}

    normalized = {}
    for k in keys_to_extract:
        if k in data:
            normalized[k] = normalize_value(data[k])
        else:
            normalized[k] = []
    return normalized


def process_nuextract_metrics(predictions_csv, summary_csv):
    if not os.path.exists(predictions_csv) or not os.path.exists(summary_csv):
        print(f"Skipping: Missing files for {predictions_csv} or {summary_csv}")
        return

    df = pd.read_csv(predictions_csv)

    # 1. Dynamically extract all available keys from ground_truth
    all_keys = set()
    for gt in df['ground_truth'].dropna():
        try:
            all_keys.update(json.loads(gt).keys())
        except:
            pass
    keys_to_extract = list(all_keys)
    print(f"Detected keys for {os.path.basename(predictions_csv)}: {keys_to_extract}")

    # 2. Normalize Ground Truth and Predictions
    df['normalized_generated'] = df['prediction'].apply(lambda x: extract_and_normalize(x, keys_to_extract))
    df['normalized_reference'] = df['ground_truth'].apply(lambda x: extract_and_normalize(x, keys_to_extract))

    # 3. Compute Metrics row by row
    def calc_row(row):
        gen = row['normalized_generated']
        ref = row['normalized_reference']
        tp_val, fp_val, fn_val = 0, 0, 0
        tp_field, fp_field, fn_field = 0, 0, 0
        fuzzy_scores = []

        for k in keys_to_extract:
            gen_list, ref_list = gen.get(k, []), ref.get(k, [])
            gen_set, ref_set = set(gen_list), set(ref_list)
            if not ref_set and not gen_set: continue

            # Field EM (Precision)
            if ref_set == gen_set:
                tp_field += 1
            else:
                if not ref_set and gen_set:
                    fp_field += 1
                elif ref_set and not gen_set:
                    fn_field += 1
                else:
                    fp_field += 1; fn_field += 1

            # Value F1 Tracking
            tp_val += len(gen_set.intersection(ref_set))
            fp_val += len(gen_set - ref_set)
            fn_val += len(ref_set - gen_set)

            # Fuzzy Score Tracking
            if not ref_set or not gen_set:
                fuzzy_scores.append(0.0)
            else:
                gen_str = " ".join(sorted(gen_list))
                ref_str = " ".join(sorted(ref_list))
                score = fuzz.token_sort_ratio(gen_str, ref_str) / 100.0
                fuzzy_scores.append(score)

        # Edge Case: Perfect True Negative Document
        if tp_field == 0 and fp_field == 0 and fn_field == 0:
            return pd.Series({'calculated_document_f1': 1.0, 'calculated_field_em': 1.0, 'calculated_fuzzy_score': 1.0})

        prec = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        rec = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        field_em = tp_field / (tp_field + fp_field) if (tp_field + fp_field) > 0 else 0.0
        avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0

        return pd.Series(
            {'calculated_document_f1': f1, 'calculated_field_em': field_em, 'calculated_fuzzy_score': avg_fuzzy})

    metrics = df.apply(calc_row, axis=1)

    # 4. Update the Standalone Summary CSV
    df_summary = pd.read_csv(summary_csv)
    df_summary['avg_calculated_document_f1'] = metrics['calculated_document_f1'].mean()
    df_summary['avg_calculated_field_em'] = metrics['calculated_field_em'].mean()
    df_summary['avg_calculated_fuzzy_score'] = metrics['calculated_fuzzy_score'].mean()

    df_summary.to_csv(summary_csv, index=False)
    print(f"Successfully updated {summary_csv} with the new calculated metrics.\n")


# Process Kleister NDA
process_nuextract_metrics(
    predictions_csv='kleister_predictions.csv',
    summary_csv='kleister_standalone_metrics_output.csv'
)

# Process VRDU (Adjust the paths to your local VRDU predictions/summary locations)
process_nuextract_metrics(
    predictions_csv='../nuextract_vrdu/vrdu_predictions.csv',
    summary_csv='../nuextract_vrdu/nuextract_vrdu_metrics.csv'
)