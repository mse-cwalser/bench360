"""
deepseek_ocr_to_due.py
----------------------
Converts DeepSeek-OCR-2 markdown output (with <|ref|>...<|/ref|> and
<|det|>[[x1,y1,x2,y2]]<|/det|> tags) into the DUE documents_content.jsonl
format (https://github.com/due-benchmark/du-schema).

Configure INPUT_DIR, OUTPUT_PATH, and optionally FILE_PATTERN at the top,
then run:
    python deepseek_ocr_to_due.py

All .md files in INPUT_DIR are processed and appended as separate JSON Lines
to OUTPUT_PATH (one record per document).

Handles:
- Multi-bbox <|det|> tags: one bbox per text line (1:1 pairing)
- Untagged text anywhere on a page → included in content.text only
- Bare page-number footer lines → stripped
- Duplicate untagged+tagged text → deduplicated
"""

import re
import json
import os
import sys
import glob

# ── Configuration ────────────────────────────────────────────────────────────
INPUT_DIR   = "../data/kleister-nda/ocr_deepseek_final"               # Folder containing DeepSeek OCR .md files
OUTPUT_PATH = "./documents_content.jsonl"  # Output file (appended to if it exists)
FILE_PATTERN = "*.md"                    # Glob pattern for input files
# ─────────────────────────────────────────────────────────────────────────────

BLOCK_RE = re.compile(
    r"<\|ref\|>(.*?)<\|/ref\|>"
    r"\s*"
    r"<\|det\|>(\[.*?\])<\|/det\|>",
    re.DOTALL,
)
PAGE_RE        = re.compile(r"---\s*\n##\s+Page\s+(\d+)", re.IGNORECASE)
SINGLE_BBOX_RE = re.compile(r"\[([^\[\]]+)\]")
PAGE_NUMBER_RE = re.compile(r"^\s*\d+\s*$", re.MULTILINE)
OCR_HEADER_RE  = re.compile(r"^#?\s*OCR Output[^\n]*\n?", re.IGNORECASE | re.MULTILINE)


def parse_bboxes(det_content: str) -> list:
    """
    Parse all bounding boxes from a <|det|>…<|/det|> tag.
    Single box:  [[308, 46, 688, 75]]           → [[308, 46, 688, 75]]
    Multi-box:   [[506,203,652,218],[506,218,…]] → [[506,203,652,218], [506,218,…]]
    When multiple bboxes are present they correspond 1:1 to the text lines.
    """
    bboxes = []
    for m in SINGLE_BBOX_RE.finditer(det_content):
        coords = [float(c.strip()) for c in m.group(1).split(",")]
        if len(coords) == 4:
            bboxes.append(coords)
    return bboxes if bboxes else [[0, 0, 0, 0]]


def clean_text(raw: str) -> str:
    text = re.sub(r"^#{1,6}\s*", "", raw, flags=re.MULTILINE)
    text = PAGE_NUMBER_RE.sub("", text)
    text = OCR_HEADER_RE.sub("", text)
    return text.strip()


def parse_deepseek_md(md_text: str) -> list:
    page_splits = list(PAGE_RE.finditer(md_text))

    if not page_splits:
        page_chunks = [(1, md_text)]
    else:
        page_chunks = []
        for idx, match in enumerate(page_splits):
            page_no = int(match.group(1))
            start = match.end()
            end = page_splits[idx + 1].start() if idx + 1 < len(page_splits) else len(md_text)
            page_chunks.append((page_no, md_text[start:end]))

    pages = []
    for page_no, chunk in page_chunks:
        tagged_blocks = []
        untagged_segments = []
        matches = list(BLOCK_RE.finditer(chunk))

        prev_end = 0
        for m in matches:
            gap = clean_text(chunk[prev_end:m.start()])
            if gap:
                untagged_segments.append(gap)
            prev_end = m.end()
        gap = clean_text(chunk[prev_end:])
        if gap:
            untagged_segments.append(gap)

        tagged_texts = set()
        for m in matches:
            block_type = m.group(1).strip()
            bboxes = parse_bboxes(m.group(2))
            next_start = chunk.find("<|ref|>", m.end())
            if next_start == -1:
                next_start = len(chunk)
            text = clean_text(chunk[m.end():next_start])
            if not text:
                continue

            text_lines = [l for l in text.splitlines() if l.strip()]
            line_bbox_pairs = []
            for i, line in enumerate(text_lines):
                bbox = bboxes[i] if i < len(bboxes) else bboxes[-1]
                line_bbox_pairs.append((bbox, line))

            tagged_blocks.append({"type": block_type, "lines": line_bbox_pairs})
            tagged_texts.add(text)

        untagged_segments = [s for s in untagged_segments if s not in tagged_texts]

        pages.append({
            "page_no": page_no,
            "tagged_blocks": tagged_blocks,
            "untagged_segments": untagged_segments,
        })

    return pages


def pages_to_due(pages: list, doc_name: str) -> dict:
    tokens = []
    positions = []
    line_structure = []
    line_positions = []
    page_structure = []
    page_positions = []
    full_text_parts = []

    for page in pages:
        page_token_start = len(tokens)
        page_bbox = None

        for seg in page["untagged_segments"]:
            full_text_parts.append(seg)

        for block in page["tagged_blocks"]:
            for bbox, line_text in block["lines"]:
                words = line_text.split()
                if not words:
                    continue
                line_token_start = len(tokens)
                for word in words:
                    tokens.append(word)
                    positions.append(bbox)
                line_structure.append([line_token_start, len(tokens)])
                line_positions.append(bbox)
                full_text_parts.append(line_text)

                if page_bbox is None:
                    page_bbox = list(bbox)
                else:
                    page_bbox[0] = min(page_bbox[0], bbox[0])
                    page_bbox[1] = min(page_bbox[1], bbox[1])
                    page_bbox[2] = max(page_bbox[2], bbox[2])
                    page_bbox[3] = max(page_bbox[3], bbox[3])

        page_token_end = len(tokens)
        if page_token_end > page_token_start:
            page_structure.append([page_token_start, page_token_end])
            page_positions.append(page_bbox or [0, 0, 1000, 1000])

    tokens_layer = {
        "doc_id": doc_name,
        "tokens": tokens,
        "positions": positions,
        "structures": {
            "pages": {
                "structure_value": page_structure,
                "positions": page_positions,
            },
            "lines": {
                "structure_value": line_structure,
                "positions": line_positions,
            },
        },
    }

    return {
        "name": doc_name,
        "contents": [
            {
                "tool_name": "other",
                "tool_version": "deepseek-ocr-2",
                "text": " ".join(full_text_parts),
                "tokens_layer": tokens_layer,
            }
        ],
    }


def derive_doc_name(md_text: str, filepath: str) -> str:
    m = re.search(r"#\s+OCR Output[^:]*:\s+(.+?)(?:\.pdf)?\s*$", md_text,
                  re.IGNORECASE | re.MULTILINE)
    return m.group(1).strip() if m else os.path.splitext(os.path.basename(filepath))[0]


def convert_file(input_path: str, doc_name: str = None) -> dict:
    with open(input_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    if not doc_name:
        doc_name = derive_doc_name(md_text, input_path)
    pages = parse_deepseek_md(md_text)
    return pages_to_due(pages, doc_name)


def convert_folder(input_dir: str, output_path: str, pattern: str = "*.md") -> None:
    files = sorted(glob.glob(os.path.join(input_dir, pattern)))
    if not files:
        print(f"No files matching '{pattern}' found in '{input_dir}'", file=sys.stderr)
        return

    processed = 0
    errors = 0
    with open(output_path, "w", encoding="utf-8") as out_f:
        for filepath in files:
            try:
                record = convert_file(filepath)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(f"  ✓ {os.path.basename(filepath)} → {record['name']}", file=sys.stderr)
                processed += 1
            except Exception as e:
                print(f"  ✗ {os.path.basename(filepath)}: {e}", file=sys.stderr)
                errors += 1

    print(f"\nDone: {processed} converted, {errors} errors → {output_path}", file=sys.stderr)


if __name__ == "__main__":
    convert_folder(INPUT_DIR, OUTPUT_PATH, FILE_PATTERN)