import re
import string
from dateutil import parser
from typing import Dict, Any, List
from thefuzz import fuzz

WORD_TO_NUM = {
    'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
    'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
    'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14,
    'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18,
    'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40,
    'fifty': 50, 'sixty': 60, 'seventy': 70, 'eighty': 80, 'ninety': 90
}

def parse_number_from_text(text: str) -> str:
    text = text.lower()
    digits = re.findall(r'\d+', text)
    if digits:
        return digits[0]

    words = re.findall(r'[a-z]+', text)
    total = 0
    current = 0
    for w in words:
        if w in WORD_TO_NUM:
            current += WORD_TO_NUM[w]
        elif w == 'hundred':
            current *= 100
        elif w == 'and':
            continue
    total += current
    return str(total) if total > 0 else ""

def normalize_jurisdiction(jurisdiction_str: str) -> str:
    return jurisdiction_str.replace("State of ","")

def normalize_term(term_str: str) -> str:
    if not term_str:
        return ""
    term_str = str(term_str).lower()
    units_match = re.search(r'(year|month|day|week|hour)s?', term_str)
    unit = units_match.group(0) if units_match else ""
    number_str = term_str[:units_match.start()] if unit else term_str
    num = parse_number_from_text(number_str)

    if num and unit:
        return f"{num}_{unit}"
    elif unit:
        return unit
    return normalize_general_attribute(term_str)

def normalize_date(date_str: str) -> str:
    if not date_str:
        return ""
    try:
        parsed = parser.parse(str(date_str), fuzzy=True)
        return parsed.strftime("%Y-%m-%d")
    except (ValueError, TypeError, OverflowError):
        return normalize_general_attribute(str(date_str))

def normalize_general_attribute(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = re.sub(r'\bincorporation\b', 'inc', s)
    s = re.sub(r'\bincorporated\b', 'inc', s)
    safe_punctuation = string.punctuation.replace('_', '')
    s = s.translate(str.maketrans('', '', safe_punctuation))
    s = re.sub(r'\s+', '_', s)
    return s

def apply_kleister_normalization(key: str, val: str) -> str:
    if not val:
        return ""
    key_lower = key.lower()
    if key_lower == "term":
        return normalize_term(val)
    elif "date" in key_lower:
        return normalize_date(val)
    elif "jurisdiction" in key_lower:
        return normalize_jurisdiction(val)
    else:
        return normalize_general_attribute(val)

def _to_list_of_str(v: Any) -> List[str]:
    """Helper to ensure values are lists of strings."""
    if v is None: return []
    return [str(x) for x in v] if isinstance(v, list) else [str(v)]

def compute_kleister_metrics(gold: Dict[str, Any], pred: Dict[str, Any], target_fields: List[str]) -> Dict[str, float]:
    """Computes F1, Precision, Recall, Exact Match, and Fuzzy scores for Kleister extractions."""

    # Extract metadata and remove it so it doesn't break scoring
    num_pages = gold.pop("__num_pages__", 0)
    pred.pop("__num_pages__", None)

    num_words = gold.pop("__num_words__", 0)
    pred.pop("__num_words__", None)

    # Global document trackers
    tp, fp, fn = 0, 0, 0

    # Entity-level trackers
    entity_tp = {k: 0 for k in target_fields}
    entity_fp = {k: 0 for k in target_fields}
    entity_fn = {k: 0 for k in target_fields}

    fuzzy_scores = []
    processed_keys = set()

    # Calculate metrics by iterating over the target fields to capture both matches and misses
    for key in target_fields:
        processed_keys.add(key)
        gt_val = gold.get(key, [])
        pred_val = pred.get(key, [])

        # Clean and normalize into lists (filtering empty strings)
        gt_val_list = [x for x in _to_list_of_str(gt_val) if x]
        pred_val_list = [x for x in _to_list_of_str(pred_val) if x]

        # Skip completely empty fields
        if not gt_val_list and not pred_val_list:
            continue

        # Strictly apply Kleister NDA normalization
        norm_gt = set(apply_kleister_normalization(key, x) for x in gt_val_list)
        norm_pred = set(apply_kleister_normalization(key, x) for x in pred_val_list)

        # INTERSECTION LOGIC: Compare values individually rather than the whole set at once
        intersection = norm_gt.intersection(norm_pred)
        false_positives = norm_pred - norm_gt
        false_negatives = norm_gt - norm_pred

        tp += len(intersection)
        fp += len(false_positives)
        fn += len(false_negatives)

        if key in entity_tp:
            entity_tp[key] += len(intersection)
            entity_fp[key] += len(false_positives)
            entity_fn[key] += len(false_negatives)

        # FUZZY MATCH LOGIC
        if not norm_gt or not norm_pred:
            # If one has values but the other is empty, the similarity is 0
            fuzzy_scores.append(0.0)
        else:
            # Join lists to a single string to compare the content
            gen_str = " ".join(sorted(list(norm_pred)))
            ref_str = " ".join(sorted(list(norm_gt)))

            # token_sort_ratio returns a score between 0 and 100, which we scale to 0.0 - 1.0
            score = fuzz.token_sort_ratio(gen_str, ref_str) / 100.0
            fuzzy_scores.append(score)

    # Penalize for predicted fields not in target_fields (e.g., hallucinated keys)
    for pred_key in pred:
        if pred_key not in processed_keys:
            pred_val_list = [x for x in _to_list_of_str(pred.get(pred_key, [])) if x]
            if pred_val_list:
                norm_pred = set(apply_kleister_normalization(pred_key, x) for x in pred_val_list)
                fp += len(norm_pred)
                if pred_key in entity_fp:
                    entity_fp[pred_key] += len(norm_pred)

    # Calculate Document-level metrics
    doc_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    doc_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    doc_f1 = 2 * (doc_precision * doc_recall) / (doc_precision + doc_recall) if (doc_precision + doc_recall) > 0 else 0.0

    # EXACT MATCH LOGIC: 1.0 if zero False Positives and zero False Negatives
    subset_em = 1.0 if (fp == 0 and fn == 0) else 0.0

    # Calculate Average Fuzzy Score
    avg_fuzzy = sum(fuzzy_scores) / len(fuzzy_scores) if fuzzy_scores else (1.0 if subset_em == 1.0 else 0.0)

    metrics = {
        "subset_em": subset_em,
        "field_f1": doc_f1,
        "field_em": subset_em,
        "avg_fuzzy_score": avg_fuzzy,
        "num_pages": num_pages,
        "num_words": num_words,
        "doc_tp": float(tp),
        "doc_fp": float(fp),
        "doc_fn": float(fn)
    }

    # Calculate Entry-level (Entity-level) metrics
    for key in target_fields:
        e_tp = entity_tp.get(key, 0)
        e_fp = entity_fp.get(key, 0)
        e_fn = entity_fn.get(key, 0)

        e_prec = e_tp / (e_tp + e_fp) if (e_tp + e_fp) > 0 else 0.0
        e_rec = e_tp / (e_tp + e_fn) if (e_tp + e_fn) > 0 else 0.0
        e_f1 = 2 * (e_prec * e_rec) / (e_prec + e_rec) if (e_prec + e_rec) > 0 else 0.0

        metrics[f"{key}_f1"] = e_f1
        metrics[f"{key}_precision"] = e_prec
        metrics[f"{key}_recall"] = e_rec
        metrics[f"{key}_tp"] = float(e_tp)
        metrics[f"{key}_fp"] = float(e_fp)
        metrics[f"{key}_fn"] = float(e_fn)

    return metrics