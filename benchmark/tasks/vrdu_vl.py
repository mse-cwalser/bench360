import os
import json
import gzip
import glob
import random
import base64
import string
import subprocess
from thefuzz import fuzz
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Union, Optional, Literal

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from benchmark.tasks.base_task import BaseTask


class VisualInfoExtractionTask(BaseTask):
    """
    Task for field extraction on the VRDU dataset using Visual Models (LMMs).
    """
    api_type: Literal["completion", "chat_completion"] = "chat_completion"

    def __init__(self, base_path: str = "./benchmark/data/vrdu",
                 dataset_name: Optional[Literal["ad-buy", "registration"]] = "registration",
                 seed: int = 42,
                 max_fields: int = 30, max_pages: int = 5):
        super().__init__()
        random.seed(seed)
        self.base_path = os.path.abspath(base_path)
        self.max_fields = max_fields
        self.max_pages = max_pages

        # Ensure the dataset is downloaded
        self._ensure_dataset(self.base_path)

        self.entries: List[Dict[str, Any]] = []

        if dataset_name:
            search_pattern = os.path.join(self.base_path, f"{dataset_name}-form")
        else:
            search_pattern = os.path.join(self.base_path, "*-form")

        form_dirs = sorted(glob.glob(search_pattern))

        for corpus_dir in form_dirs:
            main_dir = os.path.join(corpus_dir, "main")
            if not os.path.isdir(main_dir):
                continue

            jsonl_path = self._pick_jsonl(main_dir)
            if jsonl_path:
                corpus_jpgs = os.path.join(main_dir, "jpgs")
                corpus_pdfs = os.path.join(main_dir, "pdfs")
                new_entries = self._read_jsonl(jsonl_path)

                for entry in new_entries:
                    entry["_image_root"] = corpus_jpgs
                    entry["_pdf_root"] = corpus_pdfs
                    self.entries.append(entry)

        if not self.entries:
            raise FileNotFoundError(f"No VRDU entries found matching: {search_pattern}")

    # ----------------------------
    # Dataset Auto-Download
    # ----------------------------
    def _ensure_dataset(self, path: str):
        if not os.path.exists(path) or not os.listdir(path):
            print(f"VRDU Dataset not found at {path}. Downloading from GitHub...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/google-research-datasets/vrdu.git", path],
                    check=True
                )
                print("Dataset downloaded successfully.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone the VRDU dataset. Ensure 'git' is installed: {e}")

    # ----------------------------
    # BaseTask API
    # ----------------------------
    def generate_prompts(self, num_examples: int = 100) -> Tuple[List[Any], List[str]]:
        sample = random.sample(self.entries, k=min(num_examples, len(self.entries)))
        prompts: List[Any] = []
        references: List[str] = []

        for ex in sample:
            image_paths = self._get_image_paths(ex)
            if not image_paths:
                continue

            fields_to_values = self._extract_gold_fields(ex)
            if not fields_to_values:
                continue

            trimmed_fields = dict(list(fields_to_values.items())[: self.max_fields])
            messages = self._build_visual_prompt(image_paths, list(trimmed_fields.keys()))

            prompts.append(messages)
            references.append(json.dumps(trimmed_fields, ensure_ascii=False, sort_keys=True))

        return prompts, references

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        gold_raw = self._safe_json_loads(reference)
        pred_raw = self._safe_json_loads(generated)

        gold_raw = gold_raw if isinstance(gold_raw, dict) else {}
        pred_raw = pred_raw if isinstance(pred_raw, dict) else {}

        # Normalize outputs dynamically
        gold = {k: self._normalize_value(v) for k, v in gold_raw.items()}
        pred = {k: self._normalize_value(v) for k, v in pred_raw.items()}

        tp_val = 0
        fp_val = 0
        fn_val = 0

        tp_field = 0
        fp_field = 0
        fn_field = 0

        fuzzy_scores = []

        all_keys = set(pred.keys()).union(set(gold.keys()))

        for k in all_keys:
            gen_list = pred.get(k, [])
            ref_list = gold.get(k, [])

            gen_set = set(gen_list)
            ref_set = set(ref_list)

            if not ref_set and not gen_set:
                continue

            # 1. Field-level Tracking (field_em)
            if ref_set == gen_set:
                tp_field += 1
            else:
                if not ref_set and gen_set:
                    fp_field += 1
                elif ref_set and not gen_set:
                    fn_field += 1
                else:
                    fp_field += 1
                    fn_field += 1

            # 2. Value-level Tracking (F1)
            tp_val += len(gen_set.intersection(ref_set))
            fp_val += len(gen_set - ref_set)
            fn_val += len(ref_set - gen_set)

            # 3. Fuzzy Matching Tracking
            if not ref_set or not gen_set:
                fuzzy_scores.append(0.0)
            else:
                gen_str = " ".join(sorted(gen_list))
                ref_str = " ".join(sorted(ref_list))
                score = fuzz.token_sort_ratio(gen_str, ref_str) / 100.0
                fuzzy_scores.append(score)

        if tp_field == 0 and fp_field == 0 and fn_field == 0:
            return {
                "document_f1": 1.0,
                "field_em": 1.0,
                "fuzzy_score": 1.0,
                "subset_em": 1.0
            }

        precision_val = tp_val / (tp_val + fp_val) if (tp_val + fp_val) > 0 else 0.0
        recall_val = tp_val / (tp_val + fn_val) if (tp_val + fn_val) > 0 else 0.0
        f1 = 2 * precision_val * recall_val / (precision_val + recall_val) if (precision_val + recall_val) > 0 else 0.0

        field_em = tp_field / (tp_field + fp_field) if (tp_field + fp_field) > 0 else 0.0

        avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else 0.0

        return {
            "subset_em": 1.0 if f1 == 1.0 else 0.0,
            "document_f1": f1,
            "field_em": field_em,
            "fuzzy_score": avg_fuzzy
        }

    # ----------------------------
    # Visual Prompting & Rendering
    # ----------------------------
    def _get_image_paths(self, entry: Dict[str, Any]) -> List[str]:
        raw_name = entry.get("filename") or entry.get("id") or ""
        jpg_root = entry.get("_image_root")
        pdf_root = entry.get("_pdf_root")

        if not raw_name or not jpg_root or not pdf_root:
            return []

        folder_name = os.path.splitext(raw_name)[0]
        doc_img_dir = os.path.join(jpg_root, folder_name)

        if os.path.isdir(doc_img_dir):
            image_files = sorted(glob.glob(os.path.join(doc_img_dir, "*.jpg")))
            if image_files:
                return image_files[:self.max_pages]

        pdf_path = os.path.join(pdf_root, f"{folder_name}.pdf")
        if not os.path.exists(pdf_path):
            pdf_path = os.path.join(pdf_root, raw_name)
            if not os.path.exists(pdf_path):
                print(f"[dim]⚠️  PDF not found for {folder_name}[/dim]")
                return []

        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required to render PDFs to images. Install it via: pip install PyMuPDF")

        os.makedirs(doc_img_dir, exist_ok=True)
        image_files = []

        try:
            doc = fitz.open(pdf_path)
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=120)
                img_path = os.path.join(doc_img_dir, f"page_{page_num:03d}.jpg")
                pix.save(img_path)
                image_files.append(img_path)
            doc.close()
        except Exception as e:
            print(f"[dim]⚠️  Failed to render PDF {pdf_path}: {e}[/dim]")

        return image_files

    def _encode_image_to_base64(self, image_path: str) -> str:
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_visual_prompt(self, image_paths: List[str], fields: List[str]) -> List[Dict[str, Any]]:
        fields_str = ", ".join(fields)
        user_content = [
            {"type": "text", "text": f"Extract these fields from the images: {fields_str}\nOutput JSON on one line."}
        ]

        for path in image_paths:
            base64_uri = self._encode_image_to_base64(path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": base64_uri}
            })

        return [
            {"role": "system", "content": "You are an information extraction engine. Output ONLY JSON."},
            {"role": "user", "content": user_content},
        ]

    # ----------------------------
    # Utilities
    # ----------------------------
    def _normalize_value(self, val: Any) -> List[str]:
        """Recursively normalizes strings/lists for VRDU fields."""
        if val is None:
            return []
        if isinstance(val, list):
            res = []
            for v in val:
                res.extend(self._normalize_value(v))
            return res
        if isinstance(val, (str, int, float)):
            s = str(val).upper().replace('_', ' ')
            s = s.translate(str.maketrans('', '', string.punctuation))
            s = " ".join(s.split())
            return [s]
        return []

    def _extract_gold_fields(self, ex: Dict[str, Any]) -> Dict[str, Union[str, List[str]]]:
        ann = ex.get("annotations")
        if not ann: return {}

        def spans_to_values(spans) -> List[str]:
            vals = []
            if not isinstance(spans, list): return vals
            is_nested = spans and all(isinstance(it, (list, tuple)) for it in spans)
            target_spans = spans if is_nested else [spans]

            for inst in target_spans:
                pieces = [self._span_text(p) for p in inst if self._span_text(p)]
                if pieces:
                    s = self._collapse_repeated_runs(" ".join(pieces).strip())
                    if s: vals.append(s)
            return vals

        out = {}
        items = ann.items() if isinstance(ann, dict) else [i for i in ann if
                                                           isinstance(i, (list, tuple)) and len(i) >= 2]

        for item in items:
            field, spans = item if isinstance(ann, list) else (item[0], item[1])
            if not isinstance(field, str): continue
            vals = spans_to_values(spans)
            if vals:
                if field in out:
                    out[field] = self._to_list_of_str(out[field]) + vals
                else:
                    out[field] = vals if len(vals) > 1 else vals[0]

        cleaned = {}
        for k, v in out.items():
            if isinstance(v, list):
                seen = []
                for s in v:
                    s2 = self._collapse_repeated_runs(" ".join(s.split()))
                    if s2 and s2 not in seen: seen.append(s2)
                if seen: cleaned[k] = seen[0] if len(seen) == 1 else seen
            else:
                cleaned[k] = self._collapse_repeated_runs(" ".join(str(v).split()))
        return cleaned

    @staticmethod
    def _span_text(x) -> str:
        if isinstance(x, str): return x
        if isinstance(x, dict): return x.get("text", "")
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], str): return x[0]
        return ""

    @staticmethod
    def _collapse_repeated_runs(s: str, max_k: int = 8) -> str:
        toks = s.split()
        n = len(toks)
        if n <= 1: return s
        for k in range(2, min(max_k, n) + 1):
            if n % k != 0: continue
            if toks[:n // k] * k == toks: return " ".join(toks[:n // k])
        return s

    def _pick_jsonl(self, main_dir: str) -> Optional[str]:
        for f in ["dataset.jsonl.gz", "dataset.jsonl"]:
            p = os.path.join(main_dir, f)
            if os.path.exists(p): return p
        return None

    def _read_jsonl(self, path: str) -> List[Dict[str, Any]]:
        opener = gzip.open if path.endswith(".gz") else open
        res = []
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        res.append(json.loads(line))
                    except:
                        pass
        return res

    def _safe_json_loads(self, s: str) -> Any:
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

    def _to_list_of_str(self, v: Any) -> List[str]:
        if v is None: return []
        return [str(x) for x in v] if isinstance(v, list) else [str(v)]


if __name__ == "__main__":
    print("--- Testing Visual Info Extraction Task & Pre-generating Images ---")

    try:
        task = VisualInfoExtractionTask(
            base_path="../data/vrdu",
            dataset_name="registration",
            max_pages=5
        )

        print(f"Successfully loaded {len(task.entries)} entries.")
        print("Starting PDF to JPEG conversion for all documents. This might take a minute...")

        successful_renders = 0
        for i, entry in enumerate(task.entries):
            images = task._get_image_paths(entry)
            if images:
                successful_renders += 1

            if (i + 1) % 10 == 0 or (i + 1) == len(task.entries):
                print(f"Processed {i + 1}/{len(task.entries)} documents...")

        print(f"Done! Successfully generated/verified images for {successful_renders} documents.")

        print("\n--- Testing Prompt Generation for 1 Example ---")
        prompts, references = task.generate_prompts(num_examples=1)
        if prompts:
            print("Success! Prompt format is valid.")
            print(f"Extracted {len(prompts[0][1]['content']) - 1} images for this prompt.")
            print("\nSample Reference JSON:\n", references[0])

    except Exception as e:
        print(f"Error during execution: {e}")