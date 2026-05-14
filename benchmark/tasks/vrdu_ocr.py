import os
import json
import gzip
import fitz  # PyMuPDF
from benchmark.tasks.base_task import BaseTask
from typing import Any


class VRDUOCRTask(BaseTask):
    api_type = "chat_completion"

    def __init__(self, sub_dataset: str = "ad-buy-form"):
        # Dynamically build the path based on the chosen sub-dataset
        self.dataset_dir = f"./benchmark/data/vrdu/{sub_dataset}/main/pdfs"
        self.dataset_root = os.path.dirname(os.path.normpath(self.dataset_dir))

        # Create the 'ocr' output folder specifically inside the chosen 'main' directory
        self.output_dir = os.path.join(self.dataset_root, "ocr")
        os.makedirs(self.output_dir, exist_ok=True)

        # Ensure temporary image directory exists and is unique to the sub-dataset
        self.tmp_img_dir = f"/tmp/vrdu_{sub_dataset}_ocr_frames"
        os.makedirs(self.tmp_img_dir, exist_ok=True)

        # Load valid filenames from VRDU's dataset.jsonl.gz
        self.valid_pdfs = self._get_valid_filenames()

    def _get_valid_filenames(self) -> set:
        valid_files = set()
        metadata_file = os.path.join(self.dataset_root, "dataset.jsonl.gz")

        if os.path.exists(metadata_file):
            print(f"Loading VRDU metadata from {metadata_file}...")
            with gzip.open(metadata_file, 'rt', encoding='utf-8') as f:
                for line in f:
                    try:
                        item = json.loads(line.strip())
                        filename = item.get("filename")
                        if filename and filename.endswith('.pdf'):
                            valid_files.add(filename)
                    except json.JSONDecodeError:
                        continue
        else:
            print(f"Warning: Could not find VRDU metadata at {metadata_file}.")

        if not valid_files:
            print("Warning: Defaulting to all PDFs in the documents folder.")
        else:
            print(f"Loaded {len(valid_files)} benchmark PDF filenames for VRDU.")

        return valid_files

    def generate_prompts(self, num_examples: int) -> tuple[list[dict[str, Any]], list[str]]:
        prompts = []
        refs = []

        all_files = os.listdir(self.dataset_dir)
        pdf_files = sorted([
            f for f in all_files
            if f.lower().endswith('.pdf') and (not self.valid_pdfs or f in self.valid_pdfs)
        ])

        if num_examples:
            pdf_files = pdf_files[:num_examples]

        for pdf_file in pdf_files:
            pdf_path = os.path.join(self.dataset_dir, pdf_file)
            doc = fitz.open(pdf_path)

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)

                img_path = os.path.join(self.tmp_img_dir, f"{pdf_file}_page{page_num}.png")
                pix.save(img_path)

                prompt = {
                    "messages": "<|grounding|>Convert the document to markdown.",
                    "images": [img_path]
                }
                prompts.append(prompt)
                refs.append(f"{pdf_file}:::{page_num}")

            doc.close()

        return prompts, refs

    def quality_metrics(self, generated: str, reference: str) -> dict[str, float]:
        pdf_file, page_num = reference.split(":::")

        out_name = pdf_file.replace(".pdf", ".md").replace(".PDF", ".md")
        out_path = os.path.join(self.output_dir, out_name)

        mode = "w" if page_num == "0" else "a"

        with open(out_path, mode, encoding="utf-8") as f:
            if page_num != "0":
                f.write("\n\n---\n\n")
            f.write(generated)

        tmp_img_path = os.path.join(self.tmp_img_dir, f"{pdf_file}_page{page_num}.png")
        try:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
        except OSError:
            pass

        return {"CER": 1.0}