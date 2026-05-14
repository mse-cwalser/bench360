import os
import re
import json
import lzma
import random
import subprocess
from typing import Any, Dict, List, Tuple, Literal

from benchmark.tasks.base_task import BaseTask
from benchmark.utils import normalize_answer


class KleisterNDATextTask(BaseTask):
    """
    Task for Key Information Extraction on the Kleister NDA dataset using Text-Only Models.
    Extracts: effective_date, jurisdiction, party, and term from pre-computed Markdown OCR files.
    """
    api_type: Literal["completion", "chat_completion"] = "chat_completion"

    def __init__(self,
                 base_path: str = "./benchmark/data/kleister-nda",
                 source: Literal["ocr_deepseek", "ocr_docling", "ocr_tesseract"] = "ocr_docling",
                 seed: int = 42):
        super().__init__()
        random.seed(seed)
        self.base_path = os.path.abspath(base_path)
        self.source = source
        self.md_dir = os.path.join(self.base_path, self.source)

        # Target fields specific to Kleister NDA
        self.target_fields = ["effective_date", "jurisdiction", "party", "term"]

        # Ensure the dataset is downloaded
        self._ensure_dataset(self.base_path)

        # Load the TSV entries across all splits
        self.entries: List[Dict[str, Any]] = self._load_dataset()

        if not self.entries:
            raise FileNotFoundError(f"No Kleister NDA entries found in {self.base_path}")

    # ----------------------------
    # Dataset Auto-Download
    # ----------------------------
    def _ensure_dataset(self, path: str):
        if not os.path.exists(os.path.join(path, "train")):
            print(f"Kleister NDA Dataset not found at {path}. Downloading from GitHub...")
            os.makedirs(os.path.dirname(path), exist_ok=True)
            try:
                subprocess.run(
                    ["git", "clone", "https://github.com/applicaai/kleister-nda.git", path],
                    check=True
                )
                print("Kleister NDA Dataset downloaded successfully.")
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to clone the Kleister NDA dataset. Ensure 'git' is installed: {e}")

    # ----------------------------
    # Data Loading (TSV Parsing - All Splits)
    # ----------------------------
    def _load_dataset(self) -> List[Dict[str, Any]]:
        splits = ["train", "dev-0", "test-A"]
        entries = []

        for split in splits:
            expected_path = os.path.join(self.base_path, split, "expected.tsv")
            in_path = os.path.join(self.base_path, split, "in.tsv.xz")

            filenames = []

            # 1. Read in.tsv.xz to get the actual document filenames
            if os.path.exists(in_path):
                with lzma.open(in_path, mode='rt', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        if parts:
                            filenames.append(parts[0])

            # 2. Read expected.tsv to get the ground truth keys and values
            if os.path.exists(expected_path):
                with open(expected_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if not line.strip():
                            continue

                        # Align filename from in.tsv.xz, fallback to parsing if lengths mismatch
                        filename = filenames[i] if i < len(filenames) else f"unknown_{i}.pdf"
                        if not filename.endswith('.pdf'):
                            filename += '.pdf'

                        # Fix: Kleister formats truth as key=value without quotes, using underscores for spaces
                        matches = re.findall(r'(\w+)=(\S+)', line)
                        gold_dict = {}

                        for k, v in matches:
                            # Convert the underscores back into spaces for exact-match comparisons
                            v_clean = v.replace('_', ' ')

                            if k in gold_dict:
                                if isinstance(gold_dict[k], list):
                                    gold_dict[k].append(v_clean)
                                else:
                                    gold_dict[k] = [gold_dict[k], v_clean]
                            else:
                                gold_dict[k] = v_clean

                        entries.append({
                            "filename": filename,
                            "annotations": gold_dict,
                            "split": split
                        })

        return entries

    # ----------------------------
    # BaseTask API
    # ----------------------------
    def generate_prompts(self, num_examples: int = 100) -> Tuple[List[Any], List[str]]:
        sample = random.sample(self.entries, k=min(num_examples, len(self.entries)))
        prompts: List[Any] = []
        references: List[str] = []

        for ex in sample:
            document_text = self._get_markdown_text(ex)
            if not document_text:
                continue

            gold_dict = ex.get("annotations", {})

            # ONLY select fields that are actually present in the ground truth
            present_fields = [f for f in self.target_fields if f in gold_dict]

            if not present_fields:
                continue

            trimmed_fields = {field: gold_dict[field] for field in present_fields}

            # INJECT DOCUMENT LENGTH METADATA (Words instead of pages for text models)
            trimmed_fields["__num_words__"] = len(document_text.split())

            messages = self._build_text_prompt(document_text, present_fields)

            prompts.append(messages)
            references.append(json.dumps(trimmed_fields, ensure_ascii=False, sort_keys=True))

        return prompts, references

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        from dateutil import parser

        gold = self._safe_json_loads(reference)
        pred = self._safe_json_loads(generated)

        gold = gold if isinstance(gold, dict) else {}
        pred = pred if isinstance(pred, dict) else {}

        # 1. Extract metadata and remove it so it doesn't break scoring
        num_words = gold.pop("__num_words__", 0)
        pred.pop("__num_words__", None)  # Just in case the model hallucinates it

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
                list_gt = self._to_list_of_str(gt_val)
                list_pred = self._to_list_of_str(pred_val)

                if key == "effective_date":
                    list_gt = [normalize_date(x) for x in list_gt]
                    list_pred = [normalize_date(x) for x in list_pred]

                norm_gt = [normalize_answer(x) for x in list_gt]
                norm_pred = [normalize_answer(x) for x in list_pred]

                if sorted(norm_gt) == sorted(norm_pred):
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
            "num_words": num_words  # Text-equivalent to num_pages
        }

    # ----------------------------
    # Text Prompting & Markdown Loading
    # ----------------------------
    def _get_markdown_text(self, entry: Dict[str, Any]) -> str:
        raw_name = entry.get("filename", "")
        if not raw_name:
            return ""

        # Map "filename.pdf" to "filename.md"
        md_filename = raw_name.replace(".pdf", ".md")
        md_path = os.path.join(self.md_dir, md_filename)

        if not os.path.exists(md_path):
            print(f"[dim]⚠️  Markdown file not found: {md_path}[/dim]")
            return ""

        try:
            with open(md_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            print(f"[dim]⚠️  Failed to read {md_path}: {e}[/dim]")
            return ""

    def _build_text_prompt(self, document_text: str, fields: List[str]) -> str:
        fields_str = ", ".join(fields)

        system_message = (
            "You are an information extraction engine.\n"
            "Your task is to extract specific fields from the provided Non-Disclosure Agreement (NDA) document text.\n"
            "You must output ONLY one JSON object, with EXACTLY the requested keys.\n"
            "Rules:\n"
            "  - The JSON must be on a single line (no line breaks or indentation).\n"
            "  - Each requested key MUST be present in the JSON.\n"
            "  - If a field appears multiple times (e.g., multiple parties), use a JSON array of unique values.\n"
            "  - Do NOT output any explanations, comments, or text outside the JSON object.\n\n"
            "Field Guidelines:\n"
            "  - `effective_date`: Extract the effective date. Strictly format as YYYY-MM-DD.\n"
            "  - `jurisdiction`: The state or country whose laws govern the agreement (e.g., 'New York', 'Delaware').\n"
            "  - `party`: The exact names of the companies, organizations, or individuals entering into the agreement.\n"
            "  - `term`: The duration of the agreement. Format as a number followed by the unit (e.g., '3 years', '1 year', '6 months').\n"
        )

        prompt_text = (
            f"{system_message}\n\n"
            f"--- DOCUMENT TEXT ---\n"
            f"{document_text}\n"
            f"--- END OF DOCUMENT ---\n\n"
            f"Requested keys:\n{fields_str}\n\n"
            f"Output JSON:"
        )

        return prompt_text

    # ----------------------------
    # Utilities
    # ----------------------------
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
    print("--- Testing Kleister NDA Text Task ---")
    task = KleisterNDATextTask(source="ocr_deepseek", base_path="../data/kleister-nda")
    print(f"Loaded {len(task.entries)} entries across all splits.")

    prompts, references = task.generate_prompts(num_examples=1)
    if prompts:
        print("\nSample Reference JSON:\n", references[0])
        print(f"\nPrompt Length: {len(prompts[0])} characters.")