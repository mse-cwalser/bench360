import os
import lzma
import shutil
import time
import csv
import argparse
import multiprocessing as mp
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import pytesseract
from pdf2image import convert_from_path
from codecarbon import EmissionsTracker

# Ensure Tesseract doesn't fight with Python's multiprocessing
os.environ["OMP_THREAD_LIMIT"] = "1"


def process_single_pdf(pdf_path, output_dir):
    """Worker function to process a single PDF using Tesseract."""
    filename = pdf_path.name
    md_path = output_dir / f"{pdf_path.stem}.md"

    try:
        # Convert PDF to images at 300 DPI
        pages = convert_from_path(str(pdf_path), dpi=300)
        num_pages = len(pages)
        full_text = f"# OCR Output (Tesseract): {filename}\n\n"

        # Run OCR on each page
        for page_num, page_image in enumerate(pages):
            text = pytesseract.image_to_string(page_image)
            full_text += f"## Page {page_num + 1}\n\n{text}\n\n---\n\n"

        # Save the extracted text
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(full_text)

        return (True, num_pages, f"✅ Success: {filename} ({num_pages} pages)")

    except Exception as e:
        return (False, 0, f"❌ Error on {filename}: {e}")


def get_valid_filenames(dataset_root: Path) -> set:
    """Extracts valid PDF filenames from Kleister-NDA TSV files, decompressing if needed."""
    valid_files = set()
    split_folders = ["train", "dev-0", "test-A"]

    for folder in split_folders:
        folder_path = dataset_root / folder
        tsv_path = folder_path / "in.tsv"
        xz_path = folder_path / "in.tsv.xz"

        # Decompress logic from Docling script
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


def main(max_workers, num_docs=None):
    # Use 'spawn' to be consistent with Docling's approach (safer for resource management)
    mp.set_start_method('spawn', force=True)

    # Path Setup Logic
    current_dir = Path(__file__).resolve().parent
    dataset_root = current_dir.resolve().parent / "data" / "kleister-nda"
    documents_dir = dataset_root / "documents"
    output_dir = dataset_root / "ocr_tesseract"  # Named specifically for Tesseract
    report_csv = current_dir / "kleister_tesseract.csv"

    output_dir.mkdir(parents=True, exist_ok=True)

    # File Filtering Logic
    valid_pdfs = get_valid_filenames(dataset_root)
    if not valid_pdfs:
        print(dataset_root)
        print("No valid PDFs found in manifests.")
        return

    all_pdfs = list(documents_dir.glob("*.pdf"))
    target_pdfs = [p for p in all_pdfs if p.name in valid_pdfs]
    total_found = len(target_pdfs)

    # Apply the document limit if specified
    if num_docs is not None and num_docs < total_found:
        target_pdfs = target_pdfs[:num_docs]
        print(f"\n--- Found {total_found} matching PDFs, limiting processing to {num_docs} document(s) ---")
    else:
        print(f"\n--- Found {total_found} matching PDFs for Tesseract ---")

    total_documents = len(target_pdfs)

    if total_documents == 0:
        print("No documents to process. Exiting.")
        return

    # Benchmarking & Execution
    # Initialize the energy tracker, explicitly excluding the GPU
    tracker = EmissionsTracker(
        project_name="Tesseract_Kleister_Benchmark",
        gpu_ids=[]
    )
    tracker.start()
    start_time = time.time()

    processed_files = 0
    total_pages = 0

    print(f"Starting parallel processing with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_pdf, pdf, output_dir): pdf for pdf in target_pdfs}

        for future in as_completed(futures):
            success, pages, message = future.result()
            print(message)
            if success:
                processed_files += 1
                total_pages += pages

    # Report Generation Logic
    emissions = tracker.stop()
    total_time = time.time() - start_time
    energy_kwh = tracker._total_energy.kWh if tracker._total_energy else 0.0
    avg_pages = (total_pages / processed_files) if processed_files > 0 else 0
    avg_time = (total_time / processed_files) if processed_files > 0 else 0

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

    print(f"\n✅ All jobs complete! Report saved to: {report_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parallel Tesseract OCR on Kleister-NDA PDFs and generate an energy report."
    )

    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count() or 4,
                        help="Number of parallel workers (defaults to max CPU threads)")

    parser.add_argument("--num_docs", type=int, default=500,
                        help="Limit the number of documents to convert (defaults to 10)")

    args = parser.parse_args()

    main(args.workers, args.num_docs)