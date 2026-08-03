import os
import time
import argparse
import csv
import pytesseract
from pdf2image import convert_from_path
from codecarbon import EmissionsTracker
from concurrent.futures import ProcessPoolExecutor, as_completed

# CRITICAL: Prevent Tesseract from spawning its own threads and fighting with Python multiprocessing.
# This ensures 1 worker process = 1 CPU thread.
os.environ["OMP_THREAD_LIMIT"] = "1"


def process_single_pdf(pdf_path, md_path, filename):
    """Worker function to process a single PDF. Returns (success_status, num_pages, error_message)."""
    try:
        # Convert PDF to images at 300 DPI
        pages = convert_from_path(pdf_path, dpi=300)
        num_pages = len(pages)
        full_text = f"# OCR Output: {filename}\n\n"

        # Run OCR on each page
        for page_num, page_image in enumerate(pages):
            text = pytesseract.image_to_string(page_image)
            full_text += f"## Page {page_num + 1}\n\n{text}\n\n---\n\n"

        # Save the extracted text
        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(full_text)

        return (True, num_pages, f"Successfully processed {filename} ({num_pages} pages)")

    except Exception as e:
        return (False, 0, f"Error processing {filename}: {e}")


def run_ocr_benchmark(input_folder, output_folder, report_csv, max_workers, num_docs=None):
    """Runs OCR on PDFs in parallel, saves Markdown, and generates a CSV benchmark report."""

    # Ensure the output directory exists
    os.makedirs(output_folder, exist_ok=True)

    print(f"Starting parallel benchmark with {max_workers} workers...\n"
          f"Input folder: {input_folder}\n"
          f"Output folder: {output_folder}\n"
          f"Report file: {report_csv}\n")

    # Get a list of all PDF files
    all_pdf_files = [f for f in os.listdir(input_folder) if f.lower().endswith(".pdf")]
    total_found = len(all_pdf_files)

    # Apply the document limit if specified
    if num_docs is not None and num_docs < total_found:
        pdf_files = all_pdf_files[:num_docs]
        print(f"Found {total_found} PDF document(s), limiting processing to {num_docs} document(s).\n")
    else:
        pdf_files = all_pdf_files
        print(f"Found {total_found} PDF document(s) in the input folder.\n")

    total_documents = len(pdf_files)

    if total_documents == 0:
        print("No documents to process. Exiting.")
        return

    # Initialize the energy tracker, excluding the GPU
    tracker = EmissionsTracker(
        project_name="OCR_Parallel_Benchmark",
        measure_power_secs=0.1,
        gpu_ids=[]  # This prevents CodeCarbon from tracking GPU power consumption
    )

    tracker.start()
    start_time = time.time()

    processed_files = 0
    total_pages = 0

    # Execute processes in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks to the executor
        future_to_pdf = {}
        for filename in pdf_files:
            pdf_path = os.path.join(input_folder, filename)
            md_filename = os.path.splitext(filename)[0] + ".md"
            md_path = os.path.join(output_folder, md_filename)

            # Submit task
            future = executor.submit(process_single_pdf, pdf_path, md_path, filename)
            future_to_pdf[future] = filename

        # Process results as they complete
        for future in as_completed(future_to_pdf):
            success, pages, message = future.result()
            print(message)
            if success:
                processed_files += 1
                total_pages += pages

    # Stop tracking energy and time
    emissions = tracker.stop()
    end_time = time.time()

    total_time = end_time - start_time
    energy_kwh = tracker._total_energy.kWh if tracker._total_energy else 0.0

    # Calculate averages safely
    avg_pages = (total_pages / processed_files) if processed_files > 0 else 0
    avg_time = (total_time / processed_files) if processed_files > 0 else 0

    # Generate the CSV Report
    print(f"\nGenerating CSV report at: {report_csv}")

    report_dir = os.path.dirname(report_csv)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)

    with open(report_csv, mode="w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([
            "Total Documents Processed",
            "Documents Successfully Converted",
            "Total Pages Processed",
            "Average Pages per Document",
            "Total Execution Time (seconds)",
            "Average Time per Document (seconds)",
            "Total Energy Consumed (kWh)",
            "Total Carbon Emissions (kg CO2eq)"
        ])
        writer.writerow([
            total_documents,
            processed_files,
            total_pages,
            round(avg_pages, 2),
            round(total_time, 2),
            round(avg_time, 2),
            f"{energy_kwh:.6f}",
            f"{emissions:.6f}"
        ])

    # Print Final Benchmark Results to console
    print("\n" + "=" * 45)
    print("PARALLEL BENCHMARK RESULTS (CPU & RAM ONLY)")
    print("=" * 45)
    print(f"Total Documents Target:   {total_documents}")
    print(f"Documents Converted:      {processed_files}")
    print(f"Total Pages Processed:    {total_pages}")
    print(f"Average Pages/Doc:        {avg_pages:.2f}")
    print(f"Total Time:               {total_time:.2f} seconds")
    print(f"Energy Consumed:          {energy_kwh:.6f} kWh")
    print(f"Carbon Emissions:         {emissions:.6f} kg CO2eq")
    print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run parallel Tesseract OCR on PDFs, save MD output, and generate an energy report.")

    parser.add_argument("--input_dir", type=str, default="../data/vrdu/registration-form/main/pdfs",
                        help="Path to input PDFs")
    parser.add_argument("--output_dir", type=str, default="../data/vrdu/registration-form/main/tesseract_ocr",
                        help="Path to save Markdown files")
    parser.add_argument("--report", type=str, default="tesseract_vrdu_standardized.csv",
                        help="Path for CSV benchmark report")

    parser.add_argument("-w", "--workers", type=int, default=os.cpu_count(),
                        help="Number of parallel workers (defaults to max CPU threads)")

    parser.add_argument("--num_docs", type=int, default=500,
                        help="Limit the number of documents to convert (defaults to 10)")

    args = parser.parse_args()

    run_ocr_benchmark(args.input_dir, args.output_dir, args.report, args.workers, args.num_docs)