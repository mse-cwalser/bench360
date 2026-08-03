import os
import lzma
import shutil
import time
import csv
import argparse
import gc
from pathlib import Path

# Force PyTorch memory management optimizations
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from codecarbon import EmissionsTracker
from pdf2image import convert_from_path

# vLLM & DeepSeek
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor


def get_valid_filenames(dataset_root: Path) -> set:
    """Extracts valid PDF filenames from Kleister-NDA TSV files, decompressing if needed."""
    valid_files = set()
    split_folders = ["train", "dev-0"]
    for folder in split_folders:
        folder_path = dataset_root / folder
        tsv_path = folder_path / "in.tsv"
        xz_path = folder_path / "in.tsv.xz"

        if not tsv_path.exists() and xz_path.exists():
            print(f"Decompressing {xz_path}...")
            with lzma.open(xz_path, "rb") as f_in, open(tsv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if tsv_path.exists():
            with open(tsv_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split("\t")
                    if parts and parts[0].endswith(".pdf"):
                        valid_files.add(parts[0])
    return valid_files


def main(num_docs=None, chunk_size=10, max_seqs=16):
    current_dir = Path(__file__).resolve().parent
    current_parent = current_dir.resolve().parent
    dataset_root = current_parent / "data" / "kleister-nda"
    documents_dir = dataset_root / "documents"
    output_dir = dataset_root / "ocr_deepseek_final"
    report_csv = current_dir / "deepseek_kleister_report_2.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_pdfs = get_valid_filenames(dataset_root)
    if not valid_pdfs:
        print("❌ No valid PDFs found based on TSV files.")
        return

    all_pdfs = sorted(list(documents_dir.glob("*.pdf")))
    target_pdfs = [p for p in all_pdfs if p.name in valid_pdfs]
    total_found = len(target_pdfs)

    if num_docs is not None and num_docs < total_found:
        target_pdfs = target_pdfs[:num_docs]
        print(f"\n--- Found {total_found} matching PDFs, limiting to {num_docs} document(s) ---")
    else:
        print(f"\n--- Found {total_found} matching PDFs for DeepSeek-OCR-2 ---")

    total_documents = len(target_pdfs)
    if total_documents == 0:
        print("No documents to process. Exiting.")
        return

    print("\nInitializing DeepSeek-OCR-2 via vLLM...")
    # Initialize vLLM ONCE in the main thread
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR-2",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        gpu_memory_utilization=0.9,
        max_num_seqs=max_seqs,
    )

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    sampling_param = SamplingParams(
        temperature=0.0,
        repetition_penalty=1.05,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},
        ),
        skip_special_tokens=False,
    )

    tracker = EmissionsTracker(project_name="DeepSeek_Kleister_vLLM", measure_power_secs=0.1)
    tracker.start()
    start_time = time.time()

    processed_files = 0
    total_pages = 0

    # Process in chunks to prevent System RAM exhaustion from pdf2image
    for i in range(0, total_documents, chunk_size):
        chunk_pdfs = target_pdfs[i: i + chunk_size]
        print(f"\nProcessing chunk {i // chunk_size + 1}/{(total_documents + chunk_size - 1) // chunk_size} "
              f"({len(chunk_pdfs)} documents)...")

        model_inputs = []
        page_mappings = []  # Tracks which image belongs to which document and page

        # 1. Prepare images for the current chunk
        for pdf_path in chunk_pdfs:
            try:
                pages = convert_from_path(pdf_path, dpi=300)
                if not pages:
                    continue

                for page_idx, page_img in enumerate(pages):
                    model_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": page_img.convert("RGB")}
                    })
                    page_mappings.append({
                        "pdf_path": pdf_path,
                        "page_num": page_idx + 1,
                        "total_pages": len(pages)
                    })
            except Exception as e:
                print(f"❌ Error reading {pdf_path.name}: {e}")

        if not model_inputs:
            continue

        # 2. Bulk inference using vLLM's internal batching
        print(f"Feeding {len(model_inputs)} total pages to vLLM...")
        outputs = llm.generate(model_inputs, sampling_param, use_tqdm=True)

        # 3. Route outputs back to their respective Markdown files
        doc_outputs = {}
        for mapping, output in zip(page_mappings, outputs):
            pdf_path = mapping["pdf_path"]
            if pdf_path not in doc_outputs:
                doc_outputs[pdf_path] = []

            text_result = output.outputs[0].text
            doc_outputs[pdf_path].append((mapping["page_num"], text_result))

        # 4. Write Markdown files to disk
        for pdf_path, pages_data in doc_outputs.items():
            md_path = output_dir / f"{pdf_path.stem}.md"

            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(f"# OCR Output (DeepSeek-OCR-2): {pdf_path.name}\n\n")

                # Sort to ensure pages are written in order
                pages_data.sort(key=lambda x: x[0])
                for page_num, text in pages_data:
                    md_file.write(f"\n\n---\n## Page {page_num}\n\n")
                    md_file.write(text)

            processed_files += 1
            total_pages += len(pages_data)
            print(f"✅ Success: {pdf_path.name} ({len(pages_data)} pages)")

        # 5. Clean up RAM
        del model_inputs
        del outputs
        del doc_outputs
        del page_mappings
        gc.collect()

    emissions = tracker.stop()
    total_time = time.time() - start_time
    energy_kwh = tracker._total_energy.kWh if hasattr(tracker, "_total_energy") and tracker._total_energy else 0.0
    avg_pages = (total_pages / processed_files) if processed_files > 0 else 0

    # Write CSV Report
    with open(report_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Documents Target", total_documents])
        writer.writerow(["Documents Successfully Converted", processed_files])
        writer.writerow(["Total Pages Processed", total_pages])
        writer.writerow(["Average Pages per Document", round(avg_pages, 2)])
        writer.writerow(["Total Execution Time (s)", round(total_time, 2)])
        writer.writerow(["Energy Consumed (kWh)", f"{energy_kwh:.6f}"])
        writer.writerow(["Carbon Emissions (kg CO2eq)", f"{emissions:.6f}"])

    print(
        f"\nProcessing complete! Successfully processed {processed_files}/{total_documents} files in {total_time:.2f}s.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scale DeepSeek-OCR-2 via vLLM on Kleister-NDA dataset.")

    parser.add_argument("--num_docs", type=int, default=500,
                        help="Limit the number of documents to convert (defaults to 500).")
    parser.add_argument("--chunk_size", type=int, default=10,
                        help="Number of PDFs to process per chunk to save system RAM (defaults to 10).")
    parser.add_argument("--max_seqs", type=int, default=16,
                        help="vLLM max_num_seqs. Adjust based on your GPU VRAM (defaults to 16).")

    args = parser.parse_args()

    main(num_docs=args.num_docs, chunk_size=args.chunk_size, max_seqs=args.max_seqs)