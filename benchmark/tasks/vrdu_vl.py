import os
import json
import gzip
import glob
import random
import base64
import subprocess
from difflib import SequenceMatcher
from typing import Any, Dict, List, Tuple, Union, Optional, Literal

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from benchmark.tasks.base_task import BaseTask
from benchmark.utils import normalize_answer


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
        gold = self._safe_json_loads(reference)
        pred = self._safe_json_loads(generated)
        gold, pred = (gold if isinstance(gold, dict) else {}), (pred if isinstance(pred, dict) else {})

        gold_fields = set(gold.keys())
        per_field_em, per_field_f1, per_field_sub, per_field_fuzzy = [], [], [], []

        for f in gold_fields:
            gold_vals = self._to_list_of_str(gold.get(f, []))
            pred_vals = self._to_list_of_str(pred.get(f, []))

            if len(pred_vals) == 0:
                per_field_em.append(0.0)
                per_field_f1.append(0.0)
                per_field_sub.append(1.0 if len(gold_vals) == 0 else 0.0)
                per_field_fuzzy.append(1.0 if len(gold_vals) == 0 else 0.0)
                continue

            per_field_em.append(self._field_exact_em(gold_vals, pred_vals))
            per_field_f1.append(self._field_token_f1(gold_vals, pred_vals))
            per_field_sub.append(self._field_substring_match(gold_vals, pred_vals))
            per_field_fuzzy.append(self._field_fuzzy_similarity(gold_vals, pred_vals))

        avg = lambda x: sum(x) / len(x) if x else 0.0
        field_em = avg(per_field_em)
        return {
            "subset_em": 1.0 if field_em == 1.0 else 0.0,
            "field_em": field_em,
            "field_f1": avg(per_field_f1),
            "field_substring": avg(per_field_sub),
            "field_fuzzy": avg(per_field_fuzzy),
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

        # 1. Check if we already rendered this PDF to JPEG
        if os.path.isdir(doc_img_dir):
            image_files = sorted(glob.glob(os.path.join(doc_img_dir, "*.jpg")))
            if image_files:
                return image_files[:self.max_pages]

        # 2. If no JPEGs exist, we need to generate them from the PDF
        pdf_path = os.path.join(pdf_root, f"{folder_name}.pdf")
        if not os.path.exists(pdf_path):
            # Fallback check just in case the raw_name includes the extension
            pdf_path = os.path.join(pdf_root, raw_name)
            if not os.path.exists(pdf_path):
                print(f"[dim]⚠️  PDF not found for {folder_name}[/dim]")
                return []

        if fitz is None:
            raise ImportError(
                "PyMuPDF (fitz) is required to render PDFs to images. Install it via: pip install PyMuPDF")

        # Create the image directory for caching
        os.makedirs(doc_img_dir, exist_ok=True)
        image_files = []

        try:
            doc = fitz.open(pdf_path)
            # Render up to max_pages
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc.load_page(page_num)
                # 150 DPI is a great sweet spot for LMMs (readable text, smaller payload)
                pix = page.get_pixmap(dpi=120)
                img_path = os.path.join(doc_img_dir, f"page_{page_num:03d}.jpg")
                pix.save(img_path)
                image_files.append(img_path)
            doc.close()
        except Exception as e:
            print(f"[dim]⚠️  Failed to render PDF {pdf_path}: {e}[/dim]")

        return image_files

    def _encode_image_to_base64(self, image_path: str) -> str:
        """Helper to convert local images to Base64 Data URIs."""
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_visual_prompt(self, image_paths: List[str], fields: List[str]) -> List[Dict[str, Any]]:
        # 1. Dynamically generate a JSON template to enforce strict schemas
        json_template = {field: None for field in fields}
        template_str = json.dumps(json_template, indent=4).replace("null", "null")

        # 2. Split the prompt into System and User roles
        system_text = "You are an expert document information extraction engine. Output ONLY JSON."

        user_text = (
            "Your task is to extract specific fields from the provided document images.\n\n"
            "### EXTRACTION RULES:\n"
            "- Output ONLY a valid JSON object. Do not include markdown code blocks (like ```json), explanations, or conversational text.\n"
            "- You must include EXACTLY the keys shown in the OUTPUT TEMPLATE below.\n"
            "- If a field is not found in the document, set its value to `null`.\n"
            "- If a field appears multiple times, use a JSON array of unique values in reading order.\n"
            "- DATES: For `file_date`, look for official 'Received' or 'Filed' stamps. Do not confuse this with signature dates.\n"
            "- ENTITIES: Context matters. `registrant_name` and `foreign_principle_name` are usually organizations. `signer_name` is an individual person.\n\n"
            "### OUTPUT TEMPLATE:\n"
            f"{template_str}\n\n"
            "Extract the data and provide the final JSON below:"
        )

        # 3. Construct the Standard OpenAI Vision format
        content_list = [{"type": "text", "text": user_text}]

        for img_path in image_paths:
            base64_uri = self._encode_image_to_base64(img_path)
            content_list.append({
                "type": "image_url",
                "image_url": {"url": base64_uri}
            })

        return [
            {"role": "system", "content": system_text},
            {"role": "user", "content": content_list}
        ]

    # ----------------------------
    # Utilities
    # ----------------------------
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

    def _token_f1(self, gold: str, preds: List[str]) -> float:
        g = normalize_answer(gold).split()
        if not g: return 1.0 if not any(normalize_answer(p).split() for p in preds) else 0.0
        best = 0.0
        for p in preds:
            pt = normalize_answer(p).split()
            if not pt: continue
            common = set(g) & set(pt)
            pr, re = len(common) / len(pt), len(common) / len(g)
            f1 = (2 * pr * re) / (pr + re) if (pr + re) > 0 else 0.0
            best = max(best, f1)
        return best

    def _field_exact_em(self, g_vals: List[str], p_vals: List[str]) -> float:
        gn = set(normalize_answer(v) for v in g_vals if v)
        pn = set(normalize_answer(v) for v in p_vals if v)
        return 1.0 if gn == pn else 0.0

    def _field_token_f1(self, g_vals: List[str], p_vals: List[str]) -> float:
        if not g_vals: return 1.0 if not p_vals else 0.0
        scores = [self._token_f1(gv, p_vals) for gv in g_vals]
        return sum(scores) / len(scores)

    def _field_substring_match(self, g_vals: List[str], p_vals: List[str]) -> float:
        gn = [normalize_answer(v) for v in g_vals if v]
        pn = [normalize_answer(v) for v in p_vals if v]
        if not gn: return 1.0 if not pn else 0.0
        return sum(1.0 if any(g in p for p in pn) else 0.0 for g in gn) / len(gn)

    def _field_fuzzy_similarity(self, g_vals: List[str], p_vals: List[str]) -> float:
        gn = [normalize_answer(v) for v in g_vals if v]
        pn = [normalize_answer(v) for v in p_vals if v]
        if not gn or not pn: return 1.0 if not gn and not pn else 0.0
        return sum(max(SequenceMatcher(None, g, p).ratio() for p in pn) for g in gn) / len(gn)


if __name__ == "__main__":
    print("--- Testing Visual Info Extraction Task & Pre-generating Images ---")

    try:
        # Instantiate the task (this will auto-download the dataset if missing)
        # Assuming you run this from the root of your project
        task = VisualInfoExtractionTask(
            base_path="../data/vrdu",
            dataset_name="registration",
            max_pages=5
        )

        print(f"Successfully loaded {len(task.entries)} entries.")
        print("Starting PDF to JPEG conversion for all documents. This might take a minute...")

        # Iterate through all entries to trigger the lazy-rendering logic for everything
        successful_renders = 0
        for i, entry in enumerate(task.entries):
            # This calls the method that checks for JPGs and generates them if missing
            images = task._get_image_paths(entry)
            if images:
                successful_renders += 1

            # Print progress every 10 documents
            if (i + 1) % 10 == 0 or (i + 1) == len(task.entries):
                print(f"Processed {i + 1}/{len(task.entries)} documents...")

        print(f"Done! Successfully generated/verified images for {successful_renders} documents.")

        # Test a single prompt generation just to verify the output format
        print("\n--- Testing Prompt Generation for 1 Example ---")
        prompts, references = task.generate_prompts(num_examples=1)
        if prompts:
            print("Success! Prompt format is valid.")
            print(f"Extracted {len(prompts[0][1]['content']) - 1} images for this prompt.")
            print("\nSample Reference JSON:\n", references[0])

    except Exception as e:
        print(f"Error during execution: {e}")