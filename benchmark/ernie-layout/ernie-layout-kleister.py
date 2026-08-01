import os
import re
import json
import lzma
import csv
import argparse
import math
import torch
import string
from typing import Any, Dict, List, Union
from transformers import AutoModelForImageTextToText, AutoProcessor
from codecarbon import EmissionsTracker
from dateutil import parser

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# --- Configuration ---
MAIN_DIR = "../data/kleister-nda"


# ==========================================
# Kleister & Utility Functions
# ==========================================

def normalize_answer(s: str) -> str:
    """Standalone string normalization (lowercasing, removing punctuation, standardizing whitespace)."""
    if not isinstance(s, str):
        return str(s)

    s = s.lower()
    s = s.translate(str.maketrans('', '', string.punctuation))
    return " ".join(s.split())


def _safe_json_loads(s: str) -> Any:
    try:
        return json.loads(s)
    except:
        start, end = s.find("{"), s.rfind("}")
        if -1 < start < end:
            try:
                return json.loads(s[start:end + 1])
            except:
                pass
    return {}


def load_kleister_dataset(base_path: str, split: str) -> List[Dict[str, Any]]:
    """Loads and aligns the filenames with their ground truth expectations using safe string splitting."""
    expected_path = os.path.join(base_path, split, "expected.tsv")
    in_path = os.path.join(base_path, split, "in.tsv.xz")

    entries = []
    filenames = []

    if os.path.exists(in_path):
        with lzma.open(in_path, mode='rt', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if parts:
                    filenames.append(parts[0])

    if os.path.exists(expected_path):
        with open(expected_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue

                filename = filenames[i] if i < len(filenames) else f"unknown_{i}.pdf"
                if not filename.endswith('.pdf'):
                    filename += '.pdf'

                gold_dict = {}

                # Split the line by spaces (handling standard Kleister space-delimited format)
                for item in line.strip().split():
                    if '=' not in item:
                        continue

                    # Split only on the FIRST equals sign
                    k, v = item.split('=', 1)

                    # Clean up the value: handle underscores, URL encoding, and literal quotes
                    v_clean = v.replace('_', ' ').replace('%20', ' ')
                    if v_clean.startswith('"') and v_clean.endswith('"'):
                        v_clean = v_clean[1:-1]

                    if k in gold_dict:
                        if isinstance(gold_dict[k], list):
                            gold_dict[k].append(v_clean)
                        else:
                            gold_dict[k] = [gold_dict[k], v_clean]
                    else:
                        gold_dict[k] = v_clean

                entries.append({
                    "split": split,
                    "filename": filename,
                    "annotations": gold_dict,
                    "_pdf_root": os.path.join(base_path, "documents")
                })
    return entries


def get_all_pages_images(pdf_path: str):
    """Renders ALL pages of a PDF as a list of PIL Images at 150 DPI."""
    from PIL import Image
    if fitz is None:
        raise ImportError("PyMuPDF (fitz) is required. Install via: pip install PyMuPDF")

    imgs = []
    try:
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(dpi=150)
            imgs.append(Image.frombytes("RGB", [pix.width, pix.height], pix.samples))
        doc.close()
    except Exception as e:
        print(f"Failed to render PDF {pdf_path}: {e}")
    return imgs


def quality_metrics(generated: str, reference: str) -> Dict[str, float]:
    """Calculates EM and F1 with special handling for Kleister date formats."""
    gold = _safe_json_loads(reference)
    pred = _safe_json_loads(generated)

    gold = gold if isinstance(gold, dict) else {}
    pred = pred if isinstance(pred, dict) else {}

    tp, fp, fn = 0, 0, 0

    def normalize_date(date_str: str) -> str:
        if not date_str: return ""
        try:
            parsed = parser.parse(date_str, fuzzy=True)
            return parsed.strftime("%Y-%m-%d")
        except (ValueError, TypeError, OverflowError):
            return date_str

    for key, gt_val in gold.items():
        if gt_val in [None, ""]: continue
        pred_val = pred.get(key)

        if pred_val in [None, ""]:
            fn += 1
        else:
            list_gt = [str(x) for x in gt_val] if isinstance(gt_val, list) else [str(gt_val)]
            list_pred = [str(x) for x in pred_val] if isinstance(pred_val, list) else [str(pred_val)]

            if key == "effective_date":
                list_gt = [normalize_date(x) for x in list_gt]
                list_pred = [normalize_date(x) for x in list_pred]

            norm_gt = [normalize_answer(x) for x in list_gt]
            norm_pred = [normalize_answer(x) for x in list_pred]

            if sorted(norm_gt) == sorted(norm_pred) or set(norm_pred).issubset(set(norm_gt)):
                tp += 1
            else:
                fp += 1

    for pred_key in pred:
        if pred_key not in gold and pred.get(pred_key) not in [None, ""]:
            fp += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "subset_em": 1.0 if f1 == 1.0 else 0.0,
        "field_f1": f1,
        "field_em": precision,
    }


# ==========================================
# Main Processing Logic (Batched Multi-Image)
# ==========================================

def main(num_docs, batch_size, output_csv, summary_csv, splits):
    print(f"Loading Multi-Image NuExtract for Kleister NDA across splits: {splits}...")

    all_docs_to_process = []
    for split in splits:
        split_entries = load_kleister_dataset(MAIN_DIR, split)
        if not split_entries:
            print(f"Warning: Could not find or parse expected.tsv / in.tsv.xz for split '{split}'.")
            continue

        docs = split_entries[:num_docs] if num_docs > 0 else split_entries
        all_docs_to_process.extend(docs)
        print(f"  -> Added {len(docs)} documents from split '{split}'")

    if not all_docs_to_process:
        print("Error: No documents found to process across any splits.")
        return

    print(f"\nTotal documents to process: {len(all_docs_to_process)} with batch size {batch_size}...")

    print("\nLoading NuExtract-2.0-4B...")
    model_name = "numind/NuExtract-2.0-4B"
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    tracker = EmissionsTracker(project_name="NuExtract_MultiImage_Kleister", measure_power_secs=1)
    tracker.start()

    processed_count = 0
    total_subset_em = 0.0
    total_field_em = 0.0
    target_fields = ["effective_date", "jurisdiction", "party", "term"]

    with open(output_csv, mode="w", newline="", encoding="utf-8") as csvfile:
        fieldnames = ["split", "filename", "ground_truth", "prediction", "subset_em", "field_f1", "field_em"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        num_batches = math.ceil(len(all_docs_to_process) / batch_size)

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            batch_docs = all_docs_to_process[start_idx: start_idx + batch_size]

            print(
                f"\n--- Processing Batch {batch_idx + 1}/{num_batches} (Documents {start_idx + 1} to {min(start_idx + batch_size, len(all_docs_to_process))}) ---")

            batch_imgs = []
            batch_prompts = []
            batch_refs = []
            batch_meta = []

            for ex in batch_docs:
                pdf_filename = ex.get("filename")
                split_name = ex.get("split")
                pdf_path = os.path.join(ex.get("_pdf_root", ""), pdf_filename)

                if not os.path.exists(pdf_path):
                    print(f"File not found: {pdf_path}. Skipping.")
                    continue

                raw_gold_fields = ex.get("annotations", {})
                present_fields = [f for f in target_fields if f in raw_gold_fields]

                if not present_fields:
                    print(f"No target annotations found for {pdf_filename}. Skipping.")
                    continue

                trimmed_fields = {field: raw_gold_fields[field] for field in present_fields}

                # Fetch ALL pages per document at 150 DPI
                imgs = get_all_pages_images(pdf_path)
                if not imgs:
                    print(f"Failed to extract images from {pdf_filename}.")
                    continue

                # Generate a tag for every page extracted
                image_tags = "\n".join(["<|vision_start|><|image_pad|><|vision_end|>" for _ in range(len(imgs))])
                template_dict = {key: "string" for key in present_fields}

                prompt_text = (
                    f"<|im_start|>user\n"
                    f"{image_tags}\n"
                    f"# Template:\n{json.dumps(template_dict, indent=4)}\n"
                    f"# Context:\n"
                    f"<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                )

                batch_imgs.append(imgs)
                batch_prompts.append(prompt_text)
                batch_refs.append(json.dumps(trimmed_fields))
                batch_meta.append((split_name, pdf_filename))

            if not batch_imgs:
                print("No valid documents in this batch. Skipping.")
                continue

            try:
                # Pass the nested list directly to the processor
                inputs = processor(
                    text=batch_prompts,
                    images=batch_imgs,
                    return_tensors="pt",
                    padding=True
                ).to(model.device)

                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=1024, do_sample=False)

                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(inputs.input_ids, outputs)
                ]

                prediction_strs = processor.batch_decode(
                    generated_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )
            except RuntimeError as e:
                print(f"RuntimeError during generation (likely OOM). Error: {e}")
                print("Clearing CUDA cache and recording failure for this batch to prevent crash.")
                torch.cuda.empty_cache()
                # Yield empty JSON as prediction so the metrics logic naturally fails it safely
                prediction_strs = ["{}" for _ in batch_prompts]

            for i in range(len(prediction_strs)):
                pred_str = prediction_strs[i]
                ref_str = batch_refs[i]
                split_name, filename = batch_meta[i]

                metrics = quality_metrics(pred_str, ref_str)

                total_subset_em += metrics["subset_em"]
                total_field_em += metrics["field_em"]
                processed_count += 1

                print(
                    f"  -> Processed [{split_name}] {filename} | F1={metrics['field_f1']:.2f}, EM={metrics['field_em']:.2f}")

                writer.writerow({
                    "split": split_name,
                    "filename": filename,
                    "ground_truth": ref_str,
                    "prediction": pred_str,
                    "subset_em": metrics["subset_em"],
                    "field_f1": metrics["field_f1"],
                    "field_em": metrics["field_em"]
                })

            csvfile.flush()

    emissions_kg = tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed

    # Safe division
    avg_subset_em = total_subset_em / processed_count if processed_count > 0 else 0.0
    avg_field_em = total_field_em / processed_count if processed_count > 0 else 0.0

    print("\n==========================================")
    print("RUN SUMMARY")
    print("==========================================")
    print(f"Total Documents Processed : {processed_count}")
    print(f"Average Subset EM         : {avg_subset_em:.4f}")
    print(f"Average Field EM          : {avg_field_em:.4f}")
    print(f"Energy Consumed (kWh)     : {energy_kwh:.6f}")
    print(f"Total Emissions (kg CO2)  : {emissions_kg:.6f}")

    with open(summary_csv, mode="w", newline="", encoding="utf-8") as summary_file:
        summary_writer = csv.writer(summary_file)
        summary_writer.writerow(["total_documents", "average_subset_em", "average_field_em", "energy_consumed_kwh",
                                 "total_emissions_kg_co2"])
        summary_writer.writerow([processed_count, avg_subset_em, avg_field_em, energy_kwh, emissions_kg])

    print(f"\nMain results saved to '{output_csv}'.")
    print(f"Summary metrics saved to '{summary_csv}'.")


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Run NuExtract on Kleister NDA (Multi-Image) and evaluate.")

    arg_parser.add_argument("--num_docs", type=int, default=-1,
                            help="Number of documents to process per split (-1 for all documents)")
    arg_parser.add_argument("--batch_size", type=int, default=2,
                            help="Batch size. Defaults to 1 to avoid OOM when processing full documents.")
    arg_parser.add_argument("--output_csv", type=str, default="kleister_multi_image_results.csv",
                            help="Output CSV path")
    arg_parser.add_argument("--summary_csv", type=str, default="kleister_multi_image_summary.csv",
                            help="Output path for summary metrics")
    arg_parser.add_argument("--splits", nargs='+', default=["train", "dev-0", "test-A"],
                            help="List of splits to process")

    args = arg_parser.parse_args()

    main(args.num_docs, args.batch_size, args.output_csv, args.summary_csv, args.splits)