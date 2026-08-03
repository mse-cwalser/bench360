import os
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


def chunker(seq, size):
    """Yields successive chunks from a sequence."""
    for pos in range(0, len(seq), size):
        yield seq[pos:pos + size]


def main(input_dir, output_dir, report_csv, num_docs=None, batch_size=5):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    report_path = Path(report_csv)

    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Gather PDF files (similar to the Tesseract VRDU script)
    all_pdfs = sorted(list(input_path.glob("*.pdf")))
    total_found = len(all_pdfs)

    if num_docs is not None and num_docs < total_found:
        target_pdfs = all_pdfs[:num_docs]
        print(f"\n--- Found {total_found} PDFs, limiting processing to {num_docs} document(s) ---")
    else:
        target_pdfs = all_pdfs
        print(f"\n--- Found {total_found} PDFs for DeepSeek-OCR-2 ---")

    total_documents = len(target_pdfs)
    if total_documents == 0:
        print(f"No documents found in {input_dir}. Exiting.")
        return

    # 2. Initialize DeepSeek-OCR-2 via vLLM
    print("\nInitializing DeepSeek-OCR-2 via vLLM...")
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR-2",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        gpu_memory_utilization=0.9,
    )

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    sampling_param = SamplingParams(
        temperature=0.0,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        ),
        skip_special_tokens=False,
    )

    # 3. Processing Loop
    print(f"Starting processing in batches of {batch_size} documents...")
    tracker = EmissionsTracker(project_name="DeepSeek_VRDU_vLLM", measure_power_secs=0.1)
    tracker.start()
    start_time = time.time()

    processed_files = 0
    total_pages = 0

    for chunk_idx, pdf_chunk in enumerate(chunker(target_pdfs, batch_size)):
        chunk_model_inputs = []
        doc_page_counts = []

        # Convert PDFs in current chunk to images
        for pdf_path in pdf_chunk:
            try:
                pages = convert_from_path(pdf_path, dpi=200) # DeepSeek performs well at 200 DPI
                num_pages = len(pages)
                doc_page_counts.append((pdf_path, num_pages))

                for page_img in pages:
                    chunk_model_inputs.append({
                        "prompt": prompt,
                        "multi_modal_data": {"image": page_img.convert("RGB")}
                    })
            except Exception as e:
                print(f"❌ Error reading {pdf_path.name}: {e}")
                doc_page_counts.append((pdf_path, 0))

        if not chunk_model_inputs:
            continue

        # Bulk Inference
        outputs = llm.generate(chunk_model_inputs, sampling_param, use_tqdm=False)

        # Distribute results to Markdown files
        output_idx = 0
        for pdf_path, num_pages in doc_page_counts:
            if num_pages == 0:
                continue

            md_path = output_path / f"{pdf_path.stem}.md"
            with open(md_path, "w", encoding="utf-8") as md_file:
                md_file.write(f"# OCR Output (DeepSeek-OCR-2): {pdf_path.name}\n\n")

                for page_num in range(1, num_pages + 1):
                    md_file.write(f"\n\n--- Page {page_num} ---\n\n")
                    md_file.write(outputs[output_idx].outputs[0].text)
                    output_idx += 1

            processed_files += 1
            total_pages += num_pages
            print(f"✅ Success: {pdf_path.name} ({num_pages} pages)")

        print(f"--- Completed {processed_files}/{total_documents} documents ---")
        gc.collect()

    # 4. Final Reporting
    emissions = tracker.stop()
    total_time = time.time() - start_time
    energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') and tracker._total_energy else 0.0
    avg_pages = (total_pages / processed_files) if processed_files > 0 else 0
    avg_time = (total_time / processed_files) if processed_files > 0 else 0

    with open(report_path, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["Metric", "Value"])
        writer.writerow(["Total Documents Target", total_documents])
        writer.writerow(["Documents Successfully Converted", processed_files])
        writer.writerow(["Total Pages Processed", total_pages])
        writer.writerow(["Average Pages per Document", round(avg_pages, 2)])
        writer.writerow(["Total Execution Time (s)", round(total_time, 2)])
        writer.writerow(["Average Time per Document (s)", round(avg_time, 2)])
        writer.writerow(["Energy Consumed (kWh)", f"{energy_kwh:.6f}"])
        writer.writerow(["Carbon Emissions (kg CO2eq)", f"{emissions:.6f}"])

    print(f"\nProcessing complete! Report saved to {report_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DeepSeek-OCR-2 on VRDU PDFs.")

    parser.add_argument("--input_dir", type=str, default="../data/vrdu/registration-form/main/pdfs",
                        help="Path to input PDFs")
    parser.add_argument("--output_dir", type=str, default="../data/vrdu/registration-form/main/deepseek_ocr",
                        help="Path to save Markdown files")
    parser.add_argument("--report", type=str, default="deepseek_vrdu_report_bs10.csv",
                        help="Path for CSV benchmark report")
    parser.add_argument("--num_docs", type=int, default=500,
                        help="Limit the number of documents to convert")
    parser.add_argument("-b", "--batch_size", type=int, default=10,
                        help="Number of documents to batch in vLLM")

    args = parser.parse_args()
    main(args.input_dir, args.output_dir, args.report, args.num_docs, args.batch_size)