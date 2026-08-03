import os
from pathlib import Path
from pdf2image import convert_from_path

# vLLM & DeepSeek
from vllm import LLM, SamplingParams
from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor


def test_deepseek_ocr(file_list):
    """Processes a specific list of PDF documents for testing."""

    # 1. Initialize Model
    print("\n--- Initializing DeepSeek-OCR-2 via vLLM ---")
    llm = LLM(
        model="deepseek-ai/DeepSeek-OCR-2",  # Or swap to "unsloth/DeepSeek-OCR-2"
        enable_prefix_caching=False,
        mm_processor_cache_gb=0,
        logits_processors=[NGramPerReqLogitsProcessor]
    )

    # 2. Setup the Anti-Looping Sampling Parameters
    prompt = "<image>\n<|grounding|>Convert the document to markdown. "
    sampling_param = SamplingParams(
        temperature=0.1,  # Just enough variance to break loops
        repetition_penalty=1.05,  # Heavily discourages repeating exact tokens
        frequency_penalty=0.1,  # Discourages overusing the same phrases globally
        max_tokens=8192,
        extra_args=dict(
            ngram_size=30,
            window_size=90,
            whitelist_token_ids={128821, 128822},  # <td>, </td>
        ),
        skip_special_tokens=False,
    )

    # 3. Prepare Batched Inputs
    model_inputs = []
    doc_info = []  # Tracks which output belongs to which file and page

    print(f"\n--- Preparing {len(file_list)} files for batched processing ---")
    for file_path in file_list:
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {path}")
            continue

        try:
            # Convert PDF pages to images
            pages = convert_from_path(path)
            print(f"Loaded {path.name} ({len(pages)} pages)")

            for page_num, page_img in enumerate(pages, start=1):
                model_inputs.append({
                    "prompt": prompt,
                    "multi_modal_data": {"image": page_img.convert("RGB")}
                })
                doc_info.append((path.name, page_num))
        except Exception as e:
            print(f"❌ Error reading {path}: {e}")

    if not model_inputs:
        print("No valid inputs to process. Exiting.")
        return

    # 4. Generate Output via vLLM Batching
    print(f"\n--- Running batched inference on {len(model_inputs)} total pages ---")
    outputs = llm.generate(model_inputs, sampling_param)

    # 5. Save Results to a Test Directory
    output_dir = Path("test_ocr_results")
    output_dir.mkdir(exist_ok=True)
    print(f"\n--- Saving results to ./{output_dir.name}/ ---")

    for (filename, page_num), output in zip(doc_info, outputs):
        md_filename = output_dir / f"{Path(filename).stem}_page_{page_num}.md"

        with open(md_filename, "w", encoding="utf-8") as f:
            f.write(f"\n\n")
            f.write(output.outputs[0].text)

        print(f"✅ Saved: {md_filename.name}")


if __name__ == "__main__":
    cwd = os.getcwd()
    print("Current Working Directory:", cwd)

    # Define your exact list of documents to test here
    # You can copy the path of the specific PDF that was failing previously
    test_files = [
        "../data/kleister-nda/documents/fbf608b62ef498171b70fb7b36be61a0.pdf",
        # "data/kleister-nda/documents/another_test_file.pdf"
    ]
    print(test_files[0])
    print(os.path.isfile(test_files[0]))

    if (os.path.isfile(test_files[0])):
        test_deepseek_ocr(test_files)