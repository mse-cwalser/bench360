import os
import lzma
import shutil
import time
import csv
import argparse
import gc
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Critical environment variables for memory management
os.environ["OMP_THREAD_LIMIT"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from codecarbon import EmissionsTracker

from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.pipeline_options import ThreadedPdfPipelineOptions, RapidOcrOptions
from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend

# Global variable to hold the converter instance per process
global_converter = None


def get_valid_filenames(dataset_root: Path) -> set:
    """Extracts valid PDF filenames from Kleister-NDA TSV files, decompressing if needed."""
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


def init_worker():
    """Initializer for each worker process to set up its own Docling model in VRAM."""
    global global_converter

    # Restrict threads within the spawned process to avoid CPU thrashing
    torch.set_num_threads(1)

    accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=accelerator_options,
        generate_parsed_pages=False,
        # Batch sizes lowered to safely fit multiple worker models into GPU VRAM
        ocr_batch_size=4,
        layout_batch_size=4,
        table_batch_size=4,
        do_picture_classification=False,
        do_code_enrichment=False
    )

    pipeline_options.do_ocr = True
    pipeline_options.ocr_options = RapidOcrOptions(backend="torch")

    global_converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=PyPdfiumDocumentBackend
            )
        }
    )


def process_single_pdf(pdf_path, output_dir):
    """Worker function utilizing the isolated global converter to process PDFs."""
    global global_converter
    filename = pdf_path.name
    md_path = output_dir / f"{pdf_path.stem}.md"

    try:
        result = global_converter.convert(str(pdf_path))
        markdown_text = result.document.export_to_markdown()
        num_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 1

        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(f"# OCR Output (Docling): {filename}\n\n")
            md_file.write(markdown_text)

        # Cleanup backend resources
        try:
            if hasattr(result.input, '_backend'):
                result.input._backend.unload()
        except AttributeError:
            pass

        # Force garbage collection to keep VRAM/RAM footprints stable
        del result
        del markdown_text
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (True, num_pages, f"✅ Success: {filename} ({num_pages} pages)")
    except Exception as e:
        return (False, 0, f"❌ Error processing {filename}: {e}")


def main(max_workers, num_docs=None):
    # Enforce spawn method for clean multiprocessing with CUDA
    mp.set_start_method('spawn', force=True)

    current_dir = Path(__file__).resolve().parent
    current_parent = current_dir.resolve().parent
    dataset_root = current_parent / "data" / "kleister-nda"
    documents_dir = dataset_root / "documents"
    output_dir = dataset_root / "ocr_docling"
    report_csv = current_dir / "docling_kleister.csv"

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
        print(f"\n--- Found {total_found} matching PDFs for Docling ---")

    total_documents = len(target_pdfs)

    if total_documents == 0:
        print("No documents to process. Exiting.")
        return

    print(f"\nStarting multiprocessing benchmark with {max_workers} worker(s)...")

    tracker = EmissionsTracker(project_name="Docling_Kleister_Fast_MP", measure_power_secs=0.1)
    tracker.start()
    start_time = time.time()

    processed_files = 0
    total_pages = 0
    completed_tasks = 0

    # Launch multiprocess pool
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor:
        futures = {executor.submit(process_single_pdf, pdf, output_dir): pdf for pdf in target_pdfs}

        for future in as_completed(futures):
            completed_tasks += 1
            remaining_tasks = total_documents - completed_tasks

            success, pages, message = future.result()
            if success:
                processed_files += 1
                total_pages += pages

            print(f"[{completed_tasks}/{total_documents}] {message} | Remaining: {remaining_tasks}")

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
        description="Run faster Docling OCR on Kleister-NDA PDFs using ProcessPoolExecutor."
    )

    parser.add_argument("-w", "--workers", type=int, default=6,
                        help="Number of parallel workers. Be mindful of GPU VRAM. (defaults to 4)")

    parser.add_argument("--num_docs", type=int, default=500,
                        help="Limit the number of documents to convert (defaults to 10)")

    args = parser.parse_args()

    main(args.workers, args.num_docs)