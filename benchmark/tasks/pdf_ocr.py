import os
import lzma
import shutil
import fitz  # PyMuPDF - requires: pip install PyMuPDF Levenshtein
from benchmark.tasks.base_task import BaseTask
from typing import Any


class PDFOCRTask(BaseTask):
    # Enforce chat_completion so inference_engine_client uses the multimodal endpoint
    api_type = "chat_completion"

    def __init__(self, dataset_dir: str = "./benchmark/data/kleister-nda/documents"):
        self.dataset_dir = dataset_dir
        # Assumes the dataset root is one level up (where train, dev-0, etc. live)
        self.dataset_root = os.path.dirname(os.path.normpath(self.dataset_dir))

        # 1. Create the 'ocr' output folder
        self.output_dir = os.path.join(self.dataset_root, "deepseek_ocr")
        os.makedirs(self.output_dir, exist_ok=True)

        # Ensure temporary image directory exists to avoid crashes
        self.tmp_img_dir = "/tmp/pdf_ocr_frames"
        os.makedirs(self.tmp_img_dir, exist_ok=True)

        # 2. Load valid filenames from the nested in.tsv files (extracting if needed)
        self.valid_pdfs = self._get_valid_filenames()

    def _get_valid_filenames(self) -> set:
        """Reads the in.tsv files from the split subfolders, extracting .xz if necessary."""
        valid_files = set()

        # Based on your screenshot, these are the folders containing the splits
        split_folders = ["train", "dev-0", "test-A"]

        for folder in split_folders:
            folder_path = os.path.join(self.dataset_root, folder)
            tsv_path = os.path.join(folder_path, "in.tsv")
            xz_path = os.path.join(folder_path, "in.tsv.xz")

            # Auto-extract if in.tsv is missing but in.tsv.xz exists
            if not os.path.exists(tsv_path) and os.path.exists(xz_path):
                print(f"Extracting {xz_path} to {tsv_path}...")
                with lzma.open(xz_path, "rb") as f_in:
                    with open(tsv_path, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

            # Read the extracted (or already existing) in.tsv file
            if os.path.exists(tsv_path):
                with open(tsv_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        parts = line.strip().split('\t')
                        # The first column contains the filename (e.g., [md5].pdf)
                        if parts and parts[0].endswith('.pdf'):
                            valid_files.add(parts[0])
            else:
                print(f"Note: Neither {tsv_path} nor {xz_path} were found in {folder}.")

        if not valid_files:
            print("Warning: Could not find or parse any in.tsv files. Defaulting to all PDFs.")
        else:
            print(f"Loaded {len(valid_files)} benchmark PDF filenames from the split folders.")

        return valid_files

    def generate_prompts(self, num_examples: int) -> tuple[list[dict[str, Any]], list[str]]:
        prompts = []
        refs = []

        # 3. Filter the directory list against the valid_pdfs set
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

                # Send ONE image per prompt
                prompt = {
                    "messages": "<|grounding|>Convert the document to markdown.",
                    "images": [img_path]
                }
                prompts.append(prompt)

                # Encode the filename AND page number into the reference string
                refs.append(f"{pdf_file}:::{page_num}")

            doc.close()

        return prompts, refs

    def quality_metrics(self, generated: str, reference: str) -> dict[str, float]:
        # Decode the reference string
        pdf_file, page_num = reference.split(":::")

        out_name = pdf_file.replace(".pdf", ".md").replace(".PDF", ".md")
        out_path = os.path.join(self.output_dir, out_name)

        # If it's the first page, overwrite ("w"). Otherwise, append ("a").
        mode = "w" if page_num == "0" else "a"

        with open(out_path, mode, encoding="utf-8") as f:
            if page_num != "0":
                f.write("\n\n---\n\n")  # Add a visual divider between stitched pages
            f.write(generated)

        # Cleanup the temporary image for this specific page
        tmp_img_path = os.path.join(self.tmp_img_dir, f"{pdf_file}_page{page_num}.png")
        try:
            if os.path.exists(tmp_img_path):
                os.remove(tmp_img_path)
        except OSError:
            pass

        return {
            "CER": 1.0
        }