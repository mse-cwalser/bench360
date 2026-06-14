import os
import json
import gzip
import glob
import base64
import subprocess
import urllib.request
import urllib.error
import asyncio
import time
import csv
import logging
import re
import string
from difflib import SequenceMatcher
from typing import Any, Dict, List, Union

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from openai import AsyncOpenAI
from codecarbon import EmissionsTracker

# --- Configuration ---
VLLM_API_BASE = "http://localhost:23333/v1"
VLLM_CONTAINER_NAME = "nuextract_vllm"
VRDU_BASE_PATH = "../data/vrdu"
DATASET_NAME = "registration"  # "registration" or "ad-buy"
OUTPUT_CSV = "vrdu_predictions.csv"
MAX_PAGES_PER_DOC = 5
MAX_DOCS_TO_PROCESS = 500


# --- Utilities & Text Normalization ---
def normalize_answer(s: str) -> str:
    """Standard text normalization for extraction metrics."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(str(s)))))


def safe_json_loads(s: str) -> Any:
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


# --- Metric Calculations ---
class VRDUMetrics:
    @staticmethod
    def _token_f1(gold: str, preds: List[str]) -> float:
        g = normalize_answer(gold).split()
        if not g: return 1.0 if not any(normalize_answer(p).split() for p in preds) else 0.0
        best = 0.0
        for p in preds:
            pt = normalize_answer(p).split()
            if not pt: continue
            common = set(g) & set(pt)
            pr, re_score = len(common) / len(pt), len(common) / len(g)
            f1 = (2 * pr * re_score) / (pr + re_score) if (pr + re_score) > 0 else 0.0
            best = max(best, f1)
        return best

    @staticmethod
    def compute_metrics(generated: Union[str, Dict], reference: Union[str, Dict]) -> Dict[str, float]:
        gold = safe_json_loads(reference) if isinstance(reference, str) else reference
        pred = safe_json_loads(generated) if isinstance(generated, str) else generated
        gold = gold if isinstance(gold, dict) else {}
        pred = pred if isinstance(pred, dict) else {}

        gold_fields = set(gold.keys())
        if not gold_fields:
            return {"subset_em": 0.0, "field_em": 0.0, "field_f1": 0.0, "field_substring": 0.0, "field_fuzzy": 0.0}

        per_field_em, per_field_f1, per_field_sub, per_field_fuzzy = [], [], [], []

        def to_list_of_str(v: Any) -> List[str]:
            if v is None: return []
            return [str(x) for x in v] if isinstance(v, list) else [str(v)]

        for f in gold_fields:
            gold_vals = to_list_of_str(gold.get(f, []))
            pred_vals = to_list_of_str(pred.get(f, []))

            if len(pred_vals) == 0:
                per_field_em.append(0.0)
                per_field_f1.append(0.0)
                per_field_sub.append(1.0 if len(gold_vals) == 0 else 0.0)
                per_field_fuzzy.append(1.0 if len(gold_vals) == 0 else 0.0)
                continue

            # Exact Match
            gn_em = set(normalize_answer(v) for v in gold_vals if v)
            pn_em = set(normalize_answer(v) for v in pred_vals if v)
            per_field_em.append(1.0 if gn_em == pn_em else 0.0)

            # Token F1
            if not gold_vals:
                per_field_f1.append(1.0 if not pred_vals else 0.0)
            else:
                scores = [VRDUMetrics._token_f1(gv, pred_vals) for gv in gold_vals]
                per_field_f1.append(sum(scores) / len(scores))

            # Substring Match
            gn_sub = [normalize_answer(v) for v in gold_vals if v]
            pn_sub = [normalize_answer(v) for v in pred_vals if v]
            if not gn_sub:
                per_field_sub.append(1.0 if not pn_sub else 0.0)
            else:
                per_field_sub.append(sum(1.0 if any(g in p for p in pn_sub) else 0.0 for g in gn_sub) / len(gn_sub))

            # Fuzzy Match
            if not gn_sub or not pn_sub:
                per_field_fuzzy.append(1.0 if not gn_sub and not pn_sub else 0.0)
            else:
                per_field_fuzzy.append(
                    sum(max(SequenceMatcher(None, g, p).ratio() for p in pn_sub) for g in gn_sub) / len(gn_sub))

        avg = lambda x: sum(x) / len(x) if x else 0.0
        field_em = avg(per_field_em)
        return {
            "subset_em": 1.0 if field_em == 1.0 else 0.0,
            "field_em": field_em,
            "field_f1": avg(per_field_f1),
            "field_substring": avg(per_field_sub),
            "field_fuzzy": avg(per_field_fuzzy),
        }


# --- vLLM Orchestration ---
def start_vllm_container():
    print("Executing start_vllm.sh...")
    subprocess.run(["bash", "../nuextract_kleister/start_vllm.sh"], check=True)


def stop_vllm_container():
    print("\nStopping vLLM container...")
    subprocess.run(["docker", "stop", VLLM_CONTAINER_NAME], check=False)
    print("Container stopped and removed.")


def wait_for_server(timeout_seconds: int = 600):
    url = f"{VLLM_API_BASE}/models"
    start_time = time.time()
    print("Waiting for vLLM server to load the model (this may take a few minutes)...")
    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("vLLM server is online and ready!")
                    return
        except Exception:
            pass
        time.sleep(10)
    raise TimeoutError("vLLM server failed to start within the timeout period.")


# --- Dataset Management ---
class VRDUDataset:
    def __init__(self, base_path: str, dataset_name: str):
        self.base_path = os.path.abspath(base_path)
        self._ensure_dataset()
        self.entries = self._load_entries(dataset_name)

    def _ensure_dataset(self):
        if not os.path.exists(self.base_path) or not os.listdir(self.base_path):
            print(f"VRDU Dataset not found at {self.base_path}. Downloading...")
            os.makedirs(os.path.dirname(self.base_path), exist_ok=True)
            subprocess.run(["git", "clone", "https://github.com/google-research-datasets/vrdu.git", self.base_path],
                           check=True)

    def _load_entries(self, dataset_name: str) -> List[Dict[str, Any]]:
        entries = []
        search_pattern = os.path.join(self.base_path, f"{dataset_name}-form")
        form_dirs = sorted(glob.glob(search_pattern))

        for corpus_dir in form_dirs:
            main_dir = os.path.join(corpus_dir, "main")
            if not os.path.isdir(main_dir): continue

            jsonl_path = next((os.path.join(main_dir, f) for f in ["dataset.jsonl.gz", "dataset.jsonl"] if
                               os.path.exists(os.path.join(main_dir, f))), None)
            if jsonl_path:
                opener = gzip.open if jsonl_path.endswith(".gz") else open
                with opener(jsonl_path, "rt", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            try:
                                entry = json.loads(line)
                                entry["_image_root"] = os.path.join(main_dir, "jpgs")
                                entry["_pdf_root"] = os.path.join(main_dir, "pdfs")
                                entries.append(entry)
                            except:
                                pass
        return entries

    @staticmethod
    def span_text(x) -> str:
        if isinstance(x, str): return x
        if isinstance(x, dict): return x.get("text", "")
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], str): return x[0]
        return ""

    @staticmethod
    def collapse_repeated_runs(s: str, max_k: int = 8) -> str:
        toks = s.split()
        n = len(toks)
        if n <= 1: return s
        for k in range(2, min(max_k, n) + 1):
            if n % k != 0: continue
            if toks[:n // k] * k == toks: return " ".join(toks[:n // k])
        return s

    def extract_gold_fields(self, ex: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
        ann = ex.get("annotations")
        if not ann: return {}

        out = {}
        items = ann.items() if isinstance(ann, dict) else [i for i in ann if
                                                           isinstance(i, (list, tuple)) and len(i) >= 2]

        for item in items:
            field, spans = item if isinstance(ann, list) else (item[0], item[1])
            if not isinstance(field, str): continue

            vals = []
            if isinstance(spans, list):
                target_spans = spans if spans and all(isinstance(it, (list, tuple)) for it in spans) else [spans]
                for inst in target_spans:
                    pieces = [self.span_text(p) for p in inst if self.span_text(p)]
                    if pieces:
                        s = self.collapse_repeated_runs(" ".join(pieces).strip())
                        if s: vals.append(s)

            if vals:
                if field in out:
                    out[field] = (out[field] if isinstance(out[field], list) else [out[field]]) + vals
                else:
                    out[field] = vals if len(vals) > 1 else vals[0]

        cleaned = {}
        for k, v in out.items():
            if isinstance(v, list):
                seen = []
                for s in v:
                    s2 = self.collapse_repeated_runs(" ".join(s.split()))
                    if s2 and s2 not in seen: seen.append(s2)
                if seen: cleaned[k] = seen[0] if len(seen) == 1 else seen
            else:
                cleaned[k] = self.collapse_repeated_runs(" ".join(str(v).split()))
        return cleaned

    def get_image_data_urls(self, entry: Dict[str, Any], max_pages: int = MAX_PAGES_PER_DOC) -> List[str]:
        raw_name = entry.get("filename") or entry.get("id") or ""
        folder_name = os.path.splitext(raw_name)[0]
        pdf_path = os.path.join(entry["_pdf_root"], f"{folder_name}.pdf")

        if not os.path.exists(pdf_path):
            pdf_path = os.path.join(entry["_pdf_root"], raw_name)
            if not os.path.exists(pdf_path):
                return []

        data_urls = []
        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), max_pages)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=120, alpha=False)
                png_base64 = base64.b64encode(pix.tobytes("png")).decode("utf-8")
                data_urls.append(f"data:image/png;base64,{png_base64}")
            doc.close()
        except Exception as e:
            print(f"Error rendering PDF {pdf_path}: {e}")
        return data_urls


# --- Extraction Pipeline ---
async def extract_from_images(
        client: AsyncOpenAI,
        data_urls: List[str],
        template: Dict[str, Any],
        example_image_url: str = "",
        example_output: str = ""
) -> Dict[str, Any]:
    content_payload = []

    # Insert example image
    if example_image_url and example_output:
        content_payload.append({"type": "image_url", "image_url": {"url": example_image_url}})

    # Add the actual document pages
    content_payload.extend([{"type": "image_url", "image_url": {"url": url}} for url in data_urls])

    # Setup the examples block
    examples_list = []
    if example_image_url and example_output:
        examples_list.append({"input": "<image>", "output": example_output})

    try:
        response = await client.chat.completions.create(
            model="numind/NuExtract-2.0-4B",
            temperature=0.0,
            messages=[{"role": "user", "content": content_payload}],
            extra_body={
                "chat_template_kwargs": {
                    "template": json.dumps(template, indent=4),
                    "examples": examples_list
                }
            }
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"API Error during extraction: {e}")
        return {}


def merge_extracted_jsons(json_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for j in json_list:
        for k, v in j.items():
            if not v: continue
            if k not in merged or not merged[k]:
                merged[k] = v
            elif isinstance(merged[k], list) and isinstance(v, list):
                for item in v:
                    if item not in merged[k]: merged[k].append(item)
    return merged


async def process_document(
        client: AsyncOpenAI,
        data_urls: List[str],
        template: Dict[str, Any],
        example_image_url: str = "",
        example_output: str = "",
        max_images_per_call: int = 1
) -> str:
    tasks = []
    for i in range(0, len(data_urls), max_images_per_call):
        chunk = data_urls[i:i + max_images_per_call]
        tasks.append(extract_from_images(client, chunk, template, example_image_url, example_output))

    all_extracted_jsons = await asyncio.gather(*tasks)
    final_json = merge_extracted_jsons(list(all_extracted_jsons))
    return json.dumps(final_json)


# --- Main Orchestration ---
async def run_pipeline():
    if fitz is None:
        print("PyMuPDF (fitz) is required but missing. Install via: pip install PyMuPDF")
        return

    print(f"Loading VRDU {DATASET_NAME} dataset...")
    dataset = VRDUDataset(VRDU_BASE_PATH, DATASET_NAME)

    if len(dataset.entries) < 2:
        print("Not enough documents in the dataset to create a few-shot example and evaluate.")
        return

    # --- Setup Few-Shot Example ---
    example_doc = dataset.entries[0]
    example_gold_fields = dataset.extract_gold_fields(example_doc)
    example_image_urls = dataset.get_image_data_urls(example_doc, max_pages=1)
    example_image_url = example_image_urls[0] if example_image_urls else ""
    example_output_json = json.dumps(example_gold_fields)

    print(f"Using {example_doc.get('filename', 'doc_0')} as the 1-shot example.")

    # --- Setup Evaluation Dataset ---
    documents_to_process = dataset.entries[1: MAX_DOCS_TO_PROCESS + 1]
    print(
        f"Found {len(dataset.entries)} total documents. Processing {len(documents_to_process)} documents for evaluation.")

    try:
        start_vllm_container()
        wait_for_server(timeout_seconds=600)

        client = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_API_BASE)

        print("\nStarting CodeCarbon tracker...")
        logging.getLogger("codecarbon").setLevel(logging.ERROR)
        tracker = EmissionsTracker(project_name="NuExtract_VRDU_Inference", measure_power_secs=1)
        tracker.start()

        processed_count = 0
        aggregate_metrics = {"subset_em": 0.0, "field_em": 0.0, "field_f1": 0.0, "field_substring": 0.0,
                             "field_fuzzy": 0.0}

        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "ground_truth", "prediction", "field_em", "field_f1", "field_fuzzy"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for doc in documents_to_process:
                filename = doc.get("filename") or doc.get("id", f"doc_{processed_count}")
                gold_fields = dataset.extract_gold_fields(doc)

                if not gold_fields:
                    continue

                # Dynamically construct a NuExtract template expecting lists of strings
                template = {k: ["string"] for k in gold_fields.keys()}

                data_urls = dataset.get_image_data_urls(doc)
                if not data_urls:
                    print(f"\nSkipping {filename} - No valid images/PDF found.")
                    continue

                print(f"\n[{processed_count + 1}/{len(documents_to_process)}] Processing {filename}...")

                prediction_json_str = await process_document(
                    client=client,
                    data_urls=data_urls,
                    template=template,
                    example_image_url=example_image_url,
                    example_output=example_output_json
                )

                # Compute Document Metrics
                metrics = VRDUMetrics.compute_metrics(prediction_json_str, gold_fields)
                for k in aggregate_metrics:
                    aggregate_metrics[k] += metrics[k]

                writer.writerow({
                    "filename": filename,
                    "ground_truth": json.dumps(gold_fields),
                    "prediction": prediction_json_str,
                    "field_em": f"{metrics['field_em']:.4f}",
                    "field_f1": f"{metrics['field_f1']:.4f}",
                    "field_fuzzy": f"{metrics['field_fuzzy']:.4f}",
                })
                csvfile.flush()
                processed_count += 1

        emissions_kg = tracker.stop()
        energy_kwh = tracker.final_emissions_data.energy_consumed

        print("\n==========================================")
        print("RUN SUMMARY & METRICS REPORT")
        print("==========================================")
        print(f"Total Documents Processed : {processed_count}")
        print(f"Energy Consumed (kWh)     : {energy_kwh:.6f}")
        print(f"Total Emissions (kg CO2)  : {emissions_kg:.6f}")
        print(f"Predictions saved to      : {OUTPUT_CSV}")

        if processed_count > 0:
            print("\n--- Average Quality Metrics ---")
            print(f"Subset Exact Match : {aggregate_metrics['subset_em'] / processed_count:.4f}")
            print(f"Field Exact Match  : {aggregate_metrics['field_em'] / processed_count:.4f}")
            print(f"Field Token F1     : {aggregate_metrics['field_f1'] / processed_count:.4f}")
            print(f"Field Substring    : {aggregate_metrics['field_substring'] / processed_count:.4f}")
            print(f"Field Fuzzy Match  : {aggregate_metrics['field_fuzzy'] / processed_count:.4f}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        stop_vllm_container()


if __name__ == "__main__":
    asyncio.run(run_pipeline())