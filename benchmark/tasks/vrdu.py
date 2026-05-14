import os
import json
import gzip
import glob
import random
import subprocess
from typing import Any, Dict, List, Tuple, Union, Literal, Optional

from benchmark.tasks.base_task import BaseTask


class InfoExtractionTask(BaseTask):
    """
    Task for field extraction on the VRDU dataset.
    """
    api_type: Literal["completion", "chat_completion"] = "chat_completion"

    def __init__(self, base_path: str = "./benchmark/data/vrdu",
                 dataset_name: Optional[Literal["ad-buy", "registration"]] = "registration",
                 seed: int = 42,
                 max_fields: int = 30,
                 ocr_source: Literal["default", "ocr_deepseek", "ocr_docling", "ocr_tesseract"] = "default"):
        super().__init__()
        random.seed(seed)
        self.base_path = os.path.abspath(base_path)
        self.max_fields = max_fields
        self.ocr_source = ocr_source

        # Ensure the dataset is downloaded
        self._ensure_dataset(self.base_path)

        if dataset_name:
            search_pattern = os.path.join(self.base_path, f"{dataset_name}-form")
        else:
            search_pattern = os.path.join(self.base_path, "*-form")

        self.entries: List[Dict[str, Any]] = []

        for corpus_dir in sorted(glob.glob(search_pattern)):
            main_dir = os.path.join(corpus_dir, "main")
            if not os.path.isdir(main_dir):
                continue
            jsonl_path = self._pick_jsonl(main_dir)
            if not jsonl_path:
                continue

            # Read entries and inject the main_dir so we can find the .md files later
            entries = self._read_jsonl(jsonl_path)
            for ex in entries:
                ex["_main_dir"] = main_dir
            self.entries.extend(entries)

        if not self.entries:
            raise FileNotFoundError(f"No VRDU entries found matching: {search_pattern}")

    # ----------------------------
    # Dataset Auto-Download
    # ----------------------------
    def _ensure_dataset(self, path: str):
        """Checks if the dataset exists; if not, clones it from GitHub."""
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
            fields_to_values = self._extract_gold_fields(ex)
            if not fields_to_values:
                continue

            trimmed_fields = dict(list(fields_to_values.items())[: self.max_fields])
            ocr_text = self._extract_ocr_text(ex)

            messages = self._build_prompt(ocr_text, list(trimmed_fields.keys()))
            ref_json = json.dumps(trimmed_fields, ensure_ascii=False, sort_keys=True)

            # Extract the filename from the dataset entry
            pdf_filename = ex.get("filename") or ex.get("file_name") or "unknown_doc"

            # Pass as a dictionary so we can carry the metadata
            prompts.append({
                "messages": messages,
                "doc_name": pdf_filename
            })
            references.append(ref_json)

        return prompts, references

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        gold = self._safe_json_loads(reference)
        pred = self._safe_json_loads(generated)

        gold = gold if isinstance(gold, dict) else {}
        pred = pred if isinstance(pred, dict) else {}

        tp, fp, fn = 0, 0, 0

        # 1. Evaluate keys present in Ground Truth
        for key, gt_val in gold.items():
            if gt_val in [None, ""]: continue
            pred_val = pred.get(key)

            if pred_val in [None, ""]:
                fn += 1
            else:
                norm_gt = self._to_list_of_str(gt_val)
                norm_pred = self._to_list_of_str(pred_val)
                if sorted(norm_gt) == sorted(norm_pred):
                    tp += 1
                else:
                    fp += 1

        # 2. Handle Extra Keys in Predictions (Hallucinations)
        for pred_key in pred:
            if pred_key not in gold and pred.get(pred_key) not in [None, ""]:
                fp += 1

        # 3. Calculate F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "subset_em": 1.0 if f1 == 1.0 else 0.0,
            "field_f1": f1,
            "field_em": precision,
            "field_substring": 0.0,
            "field_fuzzy": 0.0,
        }

    # ----------------------------
    # Prompting (CHAT)
    # ----------------------------
    def _build_prompt(self, ocr_text: str, fields: List[str]) -> List[Dict[str, str]]:
        fields_str = ", ".join(fields)
        system_message = (
            "You are an information extraction engine.\n"
            "Your task is to read OCR text from a document and extract specific fields.\n"
            "You must output ONLY one JSON object, with EXACTLY the requested keys.\n"
            "Rules:\n"
            "  - The JSON must be on a single line (no line breaks or indentation).\n"
            "  - Each requested key MUST be present in the JSON.\n"
            "  - If a field appears multiple times, use a JSON array of unique values in reading order.\n"
            "  - If a field is not present in the OCR text, set its value to null.\n"
            "  - Do NOT add any keys that were not requested.\n"
            "  - Do NOT output any explanations, comments, or text outside the JSON object.\n"
        )
        user_task = (
            "Extract the requested keys from the OCR\n"
            f"OCR:\n{ocr_text}\n\n"
            f"Requested keys:\n{fields_str}\n\n"
            "Output JSON (single line, no extra text):"
        )
        return [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_task},
        ]

    # ----------------------------
    # Utilities
    # ----------------------------
    def _pick_jsonl(self, main_dir: str) -> Union[str, None]:
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

    def _extract_ocr_text(self, ex: Dict[str, Any], *, add_page_headers: bool = True) -> str:
        # --- Handle custom OCR sources (.md files) ---
        if self.ocr_source != "default":
            main_dir = ex.get("_main_dir", "")
            pdf_filename = ex.get("filename") or ex.get("file_name") or ""

            if pdf_filename:
                # Swap .pdf for .md
                base_name = os.path.splitext(pdf_filename)[0]
                md_path = os.path.join(main_dir, self.ocr_source, f"{base_name}.md")

                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    return text
                else:
                    print(f"[dim]⚠️  MD file not found: {md_path}. Falling back to default OCR.[/dim]")

        # --- Fallback to default JSON-based OCR ---
        ocr = ex.get("ocr") or {}

        # 1. Fallback for when OCR is just a raw string/primitive
        if not isinstance(ocr, dict):
            return str(ocr)

        pages = ocr.get("pages")

        # 2. Fallback for when there are no individual pages, just a bulk text block
        if not isinstance(pages, list) or not pages:
            return " ".join(str(ocr.get("text", "")).split())

        out_parts = []
        seen_doc = set()

        for pi, p in enumerate(pages, start=1):
            if not isinstance(p, dict): continue

            # Simple paragraph extractor fallback
            blocks = p.get("blocks") or p.get("paragraphs") or p.get("lines") or []
            items = [it for it in blocks if isinstance(it, dict) and it.get("text", "").strip()]
            if not items: continue

            # Read-order sort
            items = sorted(items,
                           key=lambda it: it.get("bbox", [1e9] * 4)[1] if isinstance(it.get("bbox"), list) else 1e9)

            if add_page_headers:
                out_parts.append(f"[PAGE {pi}]\n")

            buf = []
            for it in items:
                text = " ".join(it.get("text", "").split())
                if text and text.casefold() not in seen_doc:
                    seen_doc.add(text.casefold())
                    buf.append(text)

            if buf:
                out_parts.append(" ".join(buf).strip() + "\n")

        return "".join(out_parts)

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