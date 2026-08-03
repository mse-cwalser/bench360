import os
import lzma
import shutil
import time
import csv
import argparse
import gc
from pathlib import Path

# Tracking & PDF conversion
from codecarbon import EmissionsTracker
from pdf2image import convert_from_path

# vLLM & DeepSeek
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor


def get_valid_filenames(dataset_root: Path) -> set:
    """Extracts valid PDF filenames from Kleister-NDA TSV files."""
    valid_files = set()
    split_folders = ["train", "dev-0", "test-A"]
    for folder in split_folders:
        folder_path = dataset_root / folder
        tsv_path = folder_path / "in.tsv"
        xz_path = folder_path / "in.tsv.xz"

        if not tsv_path.exists() and xz_path.exists():
            print(f"Decompressing {xz_path}...")
            with lzma.open(xz_path, "rb") as f_in, open(tsv_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        if tsv_path.exists():
            with open(tsv_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('\t')
                    if parts and parts[0].endswith('.pdf'):
                        valid_files.add(parts[0])
    return valid_files


def chunker(seq, size):
    """Yields successive chunks from a sequence."""
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]


def main(num_docs=None, batch_size=5):
    current_dir = Path(__file__).resolve().parent
    current_parent = current_dir.resolve().parent
    dataset_root = current_parent / "data" / "kleister-nda"
    documents_dir = dataset_root / "documents"
    output_dir = dataset_root / "ocr_deepseek_v2"
    report_csv = current_dir / "deepseek_kleister.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    valid_pdfs = get_valid_filenames(dataset_root)
    if not valid_pdfs:
        print("No valid PDFs found based on TSV files.")
        return

    all_pdfs = sorted(list(documents_dir.glob("*.pdf")))
    target_pdfs = [p for p in all_pdfs if p.name in valid_pdfs]
    total_found = len(target_pdfs)

    # Apply document limit
    if num_docs is not None and num_docs < total_found:
        target_pdfs = target_pdfs[:num_docs]
        print(f"\n--- Found {total_found} matching PDFs, limiting processing to {num_docs} document(s) ---")
    else:
        print(f"\n--- Found {total_found} matching PDFs for DeepSeek-OCR-2 ---")

    total_documents = len(target_pdfs)

    if total_documents == 0:
        print("No documents to process. Exiting.")
        return

    print("\nInitializing DeepSeek-OCR-2 via vLLM...")
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR-2",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        gpu_memory_utilization=0.9,
        max_num_seqs=16,  # Enforced to prevent multimodal KV cache OOM
    )

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    sampling_param = SamplingParams(
        temperature=0.0,  # Enforced greedy decoding for strict OCR fidelity
        repetition_penalty=1.05,  # Penalizes the model for repeating the exact same tokens
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        ),
        skip_special_tokens=False,
    )

    print(f"Starting processing in batches of {batch_size} documents...")
    tracker = EmissionsTracker(project_name="DeepSeek_Kleister_vLLM_Batched", measure_power_secs=0.1)
    tracker.start()
    start_time = time.time()

    processed_files = 0
    total_pages = 0

    # Process documents in larger chunks to maximize vLLM throughput
    for chunk_idx, pdf_chunk in enumerate(chunker(target_pdfs, batch_size)):
        chunk_model_inputs = []
        doc_page_counts = []

        # 1. Prepare batch of images
        for pdf_path in pdf_chunk:
            try:
                # Increased DPI to 300 for accurate financial/dense text extraction
                pages = convert_from_path(pdf_path, dpi=300)
                num_pages = len(pages)
                doc_page_counts.append((pdf_path, num_pages))

                for page_img in pages:
                    chunk_model_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": page_img.convert("RGB")}
                    })
            except Exception as e:
                print(f"❌ Error reading {pdf_path.name}: {e}")
                doc_page_counts.append((pdf_path, 0))  # Mark as failed

        if not chunk_model_inputs:
            continue

        # 2. Bulk Generation (This is where vLLM works its magic)
        outputs = llm.generate(chunk_model_inputs, sampling_param, use_tqdm=False)

        # 3. Distribute outputs back to their respective Markdown files
        output_idx = 0
        for pdf_path, num_pages in doc_page_counts:
            if num_pages == 0:
                continue  # Skip failed reads

            filename = pdf_path.name
            md_path = output_dir / f"{pdf_path.stem}.md"

            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(f"# OCR Output (DeepSeek-OCR-2): {filename}\n\n")

                for page_num in range(1, num_pages + 1):
                    # Clean markdown page separators for robust downstream parsing
                    md_file.write(f"\n\n---\n## Page {page_num}\n\n")
                    md_file.write(outputs[output_idx].outputs[0].text)
                    output_idx += 1

            processed_files += 1
            total_pages += num_pages
            print(f"✅ Success: {filename} ({num_pages} pages)")

        print(f"--- Completed {processed_files}/{total_documents} documents ---")

        # Memory cleanup between chunks
        del chunk_model_inputs
        del outputs
        gc.collect()

    emissions = tracker.stop()
    total_time = time.time() - start_time
    energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') and tracker._total_energy else 0.0
    avg_pages = (total_pages / processed_files) if processed_files > 0 else 0

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

    print(f"\nProcessing complete! Successfully processed {processed_files}/{total_documents} files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR-2 on Kleister-NDA PDFs using vLLM natively with document batching."
    )

    parser.add_argument("--num_docs", type=int, default=500,
                        help="Limit the number of documents to convert (defaults to 500)")
    parser.add_argument("-b", "--batch_size", type=int, default=5,
                        help="Number of documents to send to vLLM at once (increase to speed up, decrease if OOM).")

    args = parser.parse_args()
    main(args.num_docs, args.batch_size)