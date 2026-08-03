import os
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

global_converter = None


def init_worker():
    global global_converter

    # Restrict threads within the spawned process
    torch.set_num_threads(1)

    # Standardized Configuration for both benchmarks
    accelerator_options = AcceleratorOptions(device=AcceleratorDevice.CUDA)

    pipeline_options = ThreadedPdfPipelineOptions(
        accelerator_options=accelerator_options,
        generate_parsed_pages=False,
        ocr_batch_size=4,  # Standardized safe batch size
        layout_batch_size=4,  # Standardized safe batch size
        table_batch_size=4,  # Standardized safe batch size
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
    global global_converter
    filename = pdf_path.name
    md_path = output_dir / f"{pdf_path.stem}.md"

    try:
        result = global_converter.convert(str(pdf_path))
        markdown_text = result.document.export_to_markdown()
        num_pages = len(result.document.pages) if hasattr(result.document, 'pages') else 1

        with open(md_path, "w", encoding="utf-8") as md_file:
            md_file.write(markdown_text)

        try:
            if hasattr(result.input, '_backend'):
                result.input._backend.unload()
        except AttributeError:
            pass

        del result
        del markdown_text
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return (True, num_pages, f"Success: {filename} ({num_pages} pages)")
    except Exception as e:
        return (False, 0, f"Error processing {filename}: {e}")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    SCRIPT_ROOT = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Docling VRDU Standardized")
    parser.add_argument("--input_dir", type=str, default="../data/vrdu/registration-form/main/pdfs")
    parser.add_argument("--output_dir", type=str, default="../data/vrdu/registration-form/main/docling_ocr")
    parser.add_argument("--report", type=str, default="docling_vrdu.csv")
    parser.add_argument("--workers", type=int, default=6, help="Standardized worker count")
    args = parser.parse_args()

    input_path = (SCRIPT_ROOT / args.input_dir).resolve()
    output_path = (SCRIPT_ROOT / args.output_dir).resolve()
    report_path = (SCRIPT_ROOT / args.report).resolve()
    output_path.mkdir(parents=True, exist_ok=True)

    target_pdfs = sorted(list(input_path.glob("*.pdf")))[:500]
    total_documents = len(target_pdfs)

    if total_documents > 0:
        tracker = EmissionsTracker(project_name="docling_vrdu_standardized", measure_power_secs=0.1)
        tracker.start()
        start_time = time.time()

        processed_files = 0
        total_pages = 0

        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker) as executor:
            futures = {executor.submit(process_single_pdf, pdf, output_path): pdf for pdf in target_pdfs}
            for future in as_completed(futures):
                success, pages, message = future.result()
                if success:
                    processed_files += 1
                    total_pages += pages

        emissions = tracker.stop()
        total_time = time.time() - start_time
        energy_kwh = tracker._total_energy.kWh if hasattr(tracker, '_total_energy') and tracker._total_energy else 0.0
        avg_pages = (total_pages / processed_files) if processed_files > 0 else 0

        with open(report_path, mode="w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Metric", "Value"])
            writer.writerow(["Total Documents Found", total_documents])
            writer.writerow(["Documents Successfully Converted", processed_files])
            writer.writerow(["Total Pages Processed", total_pages])
            writer.writerow(["Average Pages per Document", round(avg_pages, 2)])
            writer.writerow(["Total Execution Time (s)", round(total_time, 2)])
            writer.writerow(["Energy Consumed (kWh)", f"{energy_kwh:.6f}"])
            writer.writerow(["Carbon Emissions (kg CO2eq)", f"{emissions:.6f}"])