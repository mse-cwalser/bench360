import time
from pathlib import Path

# PDF conversion
from pdf2image import convert_from_path

# vLLM & DeepSeek
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor


def process_single_pdf(pdf_path: Path, output_md_path: Path):
    if not pdf_path.exists():
        print(f"❌ Error: The file '{pdf_path}' does not exist.")
        return

    print(f"\nInitializing DeepSeek-OCR-2 via vLLM for {pdf_path.name}...")
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR-2",
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor],
        gpu_memory_utilization=0.9,
        max_num_seqs=16,
    )

    prompt = "<image>\n<|grounding|>Convert the document to markdown. "

    sampling_param = SamplingParams(
        temperature=0.0,
        repetition_penalty=1.05,
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        ),
        skip_special_tokens=False,
    )

    print("Converting PDF to images (300 DPI)...")
    try:
        pages = convert_from_path(pdf_path, dpi=300)
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name}: {e}")
        return

    num_pages = len(pages)
    if num_pages == 0:
        print("No pages found in the PDF. Exiting.")
        return

    print(f"Found {num_pages} pages. Preparing vLLM inputs...")
    model_inputs = []
    for page_img in pages:
        model_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": page_img.convert("RGB")}
        })

    print("Generating Markdown...")
    start_time = time.time()

    # We turn on tqdm here since it's a single run and helpful to see page-by-page progress
    outputs = llm.generate(model_inputs, sampling_param, use_tqdm=True)

    total_time = time.time() - start_time

    print(f"Saving output to {output_md_path}...")
    output_md_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_md_path, "w", encoding="utf-8") as md_file:
        md_file.write(f"# OCR Output (DeepSeek-OCR-2): {pdf_path.name}\n\n")

        for page_num in range(1, num_pages + 1):
            md_file.write(f"\n\n---\n## Page {page_num}\n\n")
            # The outputs match the order of the inputs
            md_file.write(outputs[page_num - 1].outputs[0].text)

    print(f"✅ Success! Processed {num_pages} pages in {total_time:.2f} seconds.")


if __name__ == "__main__":
    # ---------------------------------------------------------
    # SET YOUR FILE PATHS HERE
    # ---------------------------------------------------------
    INPUT_PDF_PATH = "../data/kleister-nda/documents/fbf608b62ef498171b70fb7b36be61a0.pdf"
    OUTPUT_MD_PATH = "output.md"

    input_pdf = Path(INPUT_PDF_PATH)
    output_md = Path(OUTPUT_MD_PATH)

    process_single_pdf(input_pdf, output_md)