import logging
import os
import time
import json
import base64
import csv
import subprocess
import urllib.request
import urllib.error
import asyncio
from typing import Dict, Any, List

import fitz  # PyMuPDF
from openai import AsyncOpenAI  # Updated to Async client
from codecarbon import EmissionsTracker

# --- Configuration ---
VLLM_API_BASE = "http://localhost:23333/v1"
VLLM_CONTAINER_NAME = "nuextract_vllm"
JSONL_PATH = "./document.jsonl"
PDF_DIR = "../data/kleister-nda/documents"  # Assuming PDFs are stored here
OUTPUT_CSV = "kleister_predictions.csv"

# --- Example Configuration ---
EXAMPLE_PDF_NAME = "./00a1d238e37ac225b8045a97953e845d.pdf"


# --- vLLM Orchestration ---

def start_vllm_container():
    print("Executing start_vllm.sh...")
    subprocess.run(["bash", "start_vllm.sh"], check=True)


def stop_vllm_container():
    print("\nStopping vLLM container...")
    subprocess.run(["docker", "stop", VLLM_CONTAINER_NAME], check=False)
    print("Container stopped and removed.")


def wait_for_server(timeout_seconds: int = 600):
    url = f"{VLLM_API_BASE}/models"
    start_time = time.time()

    print("Waiting for vLLM server to load the model (this may take a few minutes)...")
    while time.time() - start_time < timeout_seconds:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as response:
                if response.status == 200:
                    print("vLLM server is online and ready!")
                    return
        except Exception:
            # Catch connection errors while the server is still booting
            pass
        time.sleep(10)

    raise TimeoutError("vLLM server failed to start within the timeout period.")


# --- Data Parsing ---

def load_kleister_jsonl(filepath: str) -> List[Dict[str, str]]:
    """Reads document.jsonl and structures the ground truth for later evaluation."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Clean out any injected tags if they exist (e.g., <tag>{...})
            if line.startswith("<") and ">" in line:
                line = line.split(">", 1)[-1].strip()

            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON line: {e}")
                continue

            filename = data.get('name')
            if not filename:
                continue

            gt = {}
            for ann in data.get('annotations', []):
                key = ann['key']
                val = ann['values'][0]['value']

                if key in gt:
                    if isinstance(gt[key], list):
                        gt[key].append(val)
                    else:
                        gt[key] = [gt[key], val]
                else:
                    # Force 'party' to always be a list to match our NuExtract template
                    if key == "party":
                        gt[key] = [val]
                    else:
                        gt[key] = val

            records.append({
                "filename": filename,
                "ground_truth": json.dumps(gt)
            })
    return records


# --- Document Processing ---

def pdf_to_png_data_urls(pdf_path: str, dpi: int = 100) -> List[str]:
    data_urls = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                pix = page.get_pixmap(dpi=dpi, alpha=False)
                png_bytes = pix.tobytes("png")
                png_base64 = base64.b64encode(png_bytes).decode("utf-8")
                data_urls.append(f"data:image/png;base64,{png_base64}")
    except Exception as e:
        print(f"Error rendering PDF {pdf_path}: {e}")
    return data_urls


async def extract_from_images(client: AsyncOpenAI, data_urls: List[str], template: Dict[str, Any],
                              example_image_url: str, example_output: str) -> Dict[str, Any]:
    # 1. Insert the example image at the beginning of the payload
    content_payload = [{"type": "image_url", "image_url": {"url": example_image_url}}]

    # 2. Add the actual document pages to extract from
    content_payload.extend([{"type": "image_url", "image_url": {"url": url}} for url in data_urls])

    try:
        # 3. Use await for the async API call
        response = await client.chat.completions.create(
            model="numind/NuExtract-2.0-4B",
            temperature=0.0,
            messages=[{"role": "user", "content": content_payload}],
            extra_body={
                "chat_template_kwargs": {
                    "template": json.dumps(template, indent=4),
                    "examples": [
                        {
                            "input": "<image>",
                            "output": example_output
                        }
                    ]
                }
            }
        )

        raw_output = response.choices[0].message.content
        return json.loads(raw_output)
    except Exception as e:
        print(f"API Error during extraction: {e}")
        return {}


def merge_extracted_jsons(json_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {}
    for j in json_list:
        for k, v in j.items():
            if not v:
                continue
            if k not in merged or not merged[k]:
                merged[k] = v
            elif isinstance(merged[k], list) and isinstance(v, list):
                for item in v:
                    if item not in merged[k]:
                        merged[k].append(item)
    return merged


async def process_document(client: AsyncOpenAI, pdf_path: str, template: Dict[str, Any], example_image_url: str,
                           example_output: str, max_images_per_call: int = 1) -> str:
    data_urls = pdf_to_png_data_urls(pdf_path, dpi=100)
    if not data_urls:
        return "{}"

    tasks = []
    # Build a list of asynchronous tasks instead of waiting for each one sequentially
    for i in range(0, len(data_urls), max_images_per_call):
        chunk = data_urls[i:i + max_images_per_call]
        tasks.append(
            extract_from_images(client, chunk, template, example_image_url, example_output)
        )

    # Execute all tasks concurrently and wait for them all to finish
    all_extracted_jsons = await asyncio.gather(*tasks)

    final_json = merge_extracted_jsons(list(all_extracted_jsons))
    return json.dumps(final_json)


async def process_and_save_doc(doc: Dict[str, str], client: AsyncOpenAI, template: Dict[str, Any],
                               example_image_url: str, example_output_json: str,
                               writer, csvfile, sem: asyncio.Semaphore, csv_lock: asyncio.Lock):
    filename = doc["filename"]
    ground_truth = doc["ground_truth"]
    pdf_path = os.path.join(PDF_DIR, filename)

    if not os.path.exists(pdf_path):
        print(f"\nSkipping {filename} - File not found at {pdf_path}")
        return

    # Wait for an available slot according to the semaphore limit
    async with sem:
        print(f"Processing {filename}...")
        prediction_json_str = await process_document(
            client=client,
            pdf_path=pdf_path,
            template=template,
            example_image_url=example_image_url,
            example_output=example_output_json
        )

    # Lock the CSV file so only one task can write to it at a time
    async with csv_lock:
        writer.writerow({
            "filename": filename,
            "ground_truth": ground_truth,
            "prediction": prediction_json_str
        })
        csvfile.flush()


# --- Main Pipeline ---

async def run_pipeline():
    # Define the template based on document.jsonl
    template = {
        "effective_date": "date",
        "jurisdiction": "string",
        "party": ["string"],
        "term": "string"
    }

    example_output_json = json.dumps({
        "effective_date": "2001-04-18",
        "jurisdiction": "Oregon",
        "party": ["Eric Dean Sprunk", "Nike Inc."],
        "term": ""
    })

    print(f"Parsing {JSONL_PATH}...")
    documents_to_process = load_kleister_jsonl(JSONL_PATH)
    print(f"Found {len(documents_to_process)} documents to process.")

    # Load your Example Image
    example_pdf_path = os.path.join(PDF_DIR, EXAMPLE_PDF_NAME)
    if not os.path.exists(example_pdf_path):
        print(f"Warning: Example PDF not found at {example_pdf_path}. Please update EXAMPLE_PDF_NAME.")
        example_image_url = ""
    else:
        example_image_urls = pdf_to_png_data_urls(example_pdf_path, dpi=100)
        example_image_url = example_image_urls[0] if example_image_urls else ""

    try:
        start_vllm_container()
        wait_for_server(timeout_seconds=600)

        # Initialize the Async client
        client = AsyncOpenAI(api_key="EMPTY", base_url=VLLM_API_BASE)

        print("\nStarting CodeCarbon tracker...")

        logging.getLogger("codecarbon").setLevel(logging.ERROR)

        tracker = EmissionsTracker(
            project_name="NuExtract_Kleister_Inference",
            measure_power_secs=0.1,
        )
        tracker.start()

        # Configurable concurrency limit (adjust based on L4 VRAM vs document size)
        MAX_CONCURRENT_DOCS = 10
        sem = asyncio.Semaphore(MAX_CONCURRENT_DOCS)
        csv_lock = asyncio.Lock()

        # Open CSV file to save predictions
        with open(OUTPUT_CSV, mode="w", newline="", encoding="utf-8") as csvfile:
            fieldnames = ["filename", "ground_truth", "prediction"]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            tasks = []
            for doc in documents_to_process:
                tasks.append(
                    process_and_save_doc(
                        doc=doc,
                        client=client,
                        template=template,
                        example_image_url=example_image_url,
                        example_output_json=example_output_json,
                        writer=writer,
                        csvfile=csvfile,
                        sem=sem,
                        csv_lock=csv_lock
                    )
                )

            print(f"\nLaunching {len(tasks)} document tasks with a concurrency limit of {MAX_CONCURRENT_DOCS}...")

            # Execute all tasks concurrently
            await asyncio.gather(*tasks)
            processed_count = len(tasks)

        emissions_kg = tracker.stop()
        energy_kwh = tracker.final_emissions_data.energy_consumed

        print("\n==========================================")
        print("RUN SUMMARY")
        print("==========================================")
        print(f"Total Documents Processed : {processed_count}")
        print(f"Energy Consumed (kWh)     : {energy_kwh:.6f}")
        print(f"Total Emissions (kg CO2)  : {emissions_kg:.6f}")
        print(f"Predictions saved to      : {OUTPUT_CSV}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")

    finally:
        stop_vllm_container()


if __name__ == "__main__":
    # Execute the asynchronous pipeline
    asyncio.run(run_pipeline())