import os
import re
import json
import glob
import lzma
import random
import base64
import subprocess
from typing import Any, Dict, List, Tuple, Union, Optional, Literal
from benchmark.utils import normalize_answer, safe_json_loads
from benchmark.kleister_utils import compute_kleister_metrics


try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from benchmark.tasks.base_task import BaseTask


class KleisterNDATask(BaseTask):
    """
    Task for Key Information Extraction on the Kleister NDA dataset using Visual Models (LMMs).
    Extracts: effective_date, jurisdiction, party, and term.
    """
    api_type: Literal["completion", "chat_completion"] = "chat_completion"

    def __init__(self, base_path: str = "./benchmark/data/kleister-nda",
                 split: Literal["train", "dev-0"] = "train",
                 seed: int = 42,
                 max_pages: int = 10):
        super().__init__()
        random.seed(seed)
        self.base_path = os.path.abspath(base_path)
        self.split = split
        self.max_pages = max_pages

        # Target fields specific to Kleister NDA
        self.target_fields = ["effective_date", "jurisdiction", "party", "term"]

        # Ensure the dataset is downloaded
        self._ensure_dataset(self.base_path)

        # Load the TSV entries
        self.entries: List[Dict[str, Any]] = self._load_dataset()

        if not self.entries:
            raise FileNotFoundError(f"No Kleister NDA entries found in {self.base_path}")

    # ----------------------------
    # Dataset Auto-Download
    # ----------------------------
    def _ensure_dataset(self, path: str):
        if not os.path.exists(os.path.join(path, "documents")):
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
    # Data Loading (TSV Parsing)
    # ----------------------------
    def _load_dataset(self) -> List[Dict[str, Any]]:
        expected_path = os.path.join(self.base_path, self.split, "expected.tsv")
        in_path = os.path.join(self.base_path, self.split, "in.tsv.xz")

        entries = []
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
                        # We keep 'v' exactly as it is, maintaining YYYY-MM-DD and {number}_{units}
                        if k in gold_dict:
                            if isinstance(gold_dict[k], list):
                                gold_dict[k].append(v)
                            else:
                                gold_dict[k] = [gold_dict[k], v]
                        else:
                            gold_dict[k] = v

                    entries.append({
                        "filename": filename,
                        "annotations": gold_dict,
                        "_pdf_root": os.path.join(self.base_path, "documents"),
                        "_image_root": os.path.join(self.base_path, "jpgs")
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
            image_paths = self._get_image_paths(ex)
            if not image_paths:
                continue

            gold_dict = ex.get("annotations", {})

            # ONLY select fields that are actually present in the ground truth
            present_fields = [f for f in self.target_fields if f in gold_dict]

            if not present_fields:
                continue

            trimmed_fields = {field: gold_dict[field] for field in present_fields}

            # INJECT DOCUMENT LENGTH METADATA
            trimmed_fields["__num_pages__"] = len(image_paths)

            messages = self._build_visual_prompt(image_paths, present_fields)

            prompts.append(messages)
            references.append(json.dumps(trimmed_fields, ensure_ascii=False, sort_keys=True))

        return prompts, references

    def quality_metrics(self, generated: str, reference: str) -> Dict[str, float]:
        gold = safe_json_loads(reference)
        pred = safe_json_loads(generated)

        gold = gold if isinstance(gold, dict) else {}
        pred = pred if isinstance(pred, dict) else {}

        # Call the extracted utility function
        return compute_kleister_metrics(gold, pred, self.target_fields)
    # ----------------------------
    # Visual Prompting & Rendering
    # ----------------------------
    def _get_image_paths(self, entry: Dict[str, Any]) -> List[str]:
        raw_name = entry.get("filename", "")
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

        # 2. If no JPEGs exist, generate them from the PDF
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
        """Helper to convert local images to Base64 Data URIs."""
        with open(image_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{encoded}"

    def _build_visual_prompt(self, image_paths: List[str], fields: List[str]) -> List[Dict[str, Any]]:
        fields_str = ", ".join(fields)

        # Dictionary containing the guidelines for all possible fields
        field_guidelines = {
            "effective_date": "`effective_date`: Extract the effective date. Strictly format as YYYY-MM-DD.",
            "jurisdiction": "`jurisdiction`: The state or country whose laws govern the agreement (e.g., 'New York', 'Delaware').",
            "party": "`party`: The exact names of the companies, organizations, or individuals entering into the agreement.",
            "term": "`term`: The duration of the agreement. Format as a number followed by the unit (e.g., '3 years', '1 year', '6 months')."
        }

        # Dynamically build the guidelines string for ONLY the requested fields
        requested_descriptions = "\n".join(
            f"  - {field_guidelines[f]}" for f in fields if f in field_guidelines
        )

        system_message = (
            "You are an information extraction engine.\n"
            "Your task is to extract specific fields from the provided Non-Disclosure Agreement (NDA) document images.\n"
            "You must output ONLY one JSON object, with EXACTLY the requested keys.\n"
            "Rules:\n"
            "  - The JSON must be on a single line (no line breaks or indentation).\n"
            "  - Each requested key MUST be present in the JSON.\n"
            "  - If a field appears multiple times (e.g., multiple parties), use a JSON array of unique values.\n"
            "  - Do NOT output any explanations, comments, or text outside the JSON object.\n\n"
            f"Field Guidelines:\n{requested_descriptions}\n"
        )

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
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_content},
        ]


if __name__ == "__main__":
    print("--- Testing Kleister NDA Task & Pre-generating Images ---")

    try:
        task = KleisterNDATask()
        print(f"Loaded {len(task.entries)} entries.")
        print("Starting PDF to JPEG conversion for all documents. This might take a minute...")

        # Iterate through all entries to trigger the lazy-rendering logic for everything
        successful_renders = 0
        for i, entry in enumerate(task.entries):
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