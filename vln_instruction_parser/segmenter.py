"""English instruction segmentation."""

import re
from typing import List, Tuple


# Split markers ordered by priority (more specific first)
SPLIT_MARKERS = [
    r"\band then\b",
    r"\bafter that\b",
    r"\bthen\b",
    r"\bnext\b",
    r"\bfinally\b",
    r"\binstead\b",
]

# Standalone transition words that should not form their own segment
TRANSITION_WORDS = {
    "then", "next", "instead", "afterward", "afterwards",
    "subsequently", "thereafter", "meanwhile", "finally",
}

# Abbreviations that end with a period but should NOT be sentence boundaries
ABBREVIATIONS = {
    "e.g", "i.e", "mr", "mrs", "dr", "st", "ave", "blvd",
    "etc", "vol", "vols", "inc", "ltd", "jr", "sr", "prof",
    "fig", "figs", "et al", "vs", "viz",
}

# Cross-sentence anaphora / back-reference markers that make safe per-sentence parsing impossible
CROSS_SENTENCE_BACKREFS = {
    "it", "there", "this", "that", "those", "same",
    "previous", "former", "latter", "do so", "doing so",
}

# Sentence-initial cross-sentence rewrite / reordering markers
CROSS_SENTENCE_REWRITE_STARTERS = {
    "instead", "rather than", "before that",
}

# Safe sequential continuers that allow per-sentence processing
SAFE_SENTENCE_CONTINUERS = {
    "then", "next", "finally", "after that",
}


def normalize_whitespace(text: str) -> str:
    """Collapse multiple whitespace characters into single spaces."""
    return " ".join(text.split())


def split_sentences_for_llm(text: str) -> List[str]:
    """
    Split text into sentences for per-sentence LLM parsing.

    Rules:
    - Split only on ., ?, ! followed by space or end-of-string.
    - Do NOT split on commas, 'then', 'before', 'instead'.
    - Protect common abbreviations (e.g., i.e., Mr., Dr., St.) from being cut.
    - Preserve all punctuation and intra-sentence structure.
    """
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    # Protect abbreviations by temporarily replacing their trailing period.
    # Use re.sub with a replacer function to avoid index corruption when
    # multiple abbreviations appear in the text.
    protected = cleaned
    placeholders: List[Tuple[str, str]] = []
    placeholder_idx = 0

    def _abbrev_replacer(match: "re.Match[str]") -> str:
        nonlocal placeholder_idx
        ph = f"\x00ABBREV{placeholder_idx}\x00"
        placeholders.append((ph, match.group(0)))
        placeholder_idx += 1
        return ph

    # Match abbreviations as whole words (case-insensitive)
    for abbrev in ABBREVIATIONS:
        pattern = rf"\b{re.escape(abbrev)}\."
        protected = re.sub(pattern, _abbrev_replacer, protected, flags=re.IGNORECASE)

    def _decimal_replacer(match: "re.Match[str]") -> str:
        nonlocal placeholder_idx
        ph = f"\x00DECIMAL{placeholder_idx}\x00"
        placeholders.append((ph, match.group(0)))
        placeholder_idx += 1
        return ph

    # Also protect decimal numbers like "3.14"
    protected = re.sub(r"\d+\.\d+", _decimal_replacer, protected)

    # Split on sentence-ending punctuation followed by space or end
    raw_parts = re.split(r'(?<=[.!?])(?:\s+|$)', protected)

    sentences = []
    for part in raw_parts:
        part = part.strip()
        if not part:
            continue
        # Restore placeholders
        for ph, original in placeholders:
            part = part.replace(ph, original)
        sentences.append(part)

    return sentences


def has_cross_sentence_dependency(sentences: List[str]) -> Tuple[bool, str]:
    """
    Detect whether a multi-sentence instruction contains anaphora or
    cross-sentence reordering that makes safe per-sentence parsing impossible.

    Returns:
        (True, reason) if a dependency is found.
        (False, "") if safe to process sentence-by-sentence.
    """
    if len(sentences) < 2:
        return False, ""

    for i, sent in enumerate(sentences[1:], start=1):
        lower = sent.lower().strip()
        # Remove leading safe continuers for checking
        stripped = lower
        for continuer in SAFE_SENTENCE_CONTINUERS:
            if stripped.startswith(continuer + " "):
                stripped = stripped[len(continuer):].strip()
                break

        # Check back-references
        for ref in CROSS_SENTENCE_BACKREFS:
            # Match as whole word or phrase
            if re.search(rf"\b{re.escape(ref)}\b", stripped):
                return True, f"cross_sentence_dependency_requires_review: backref '{ref}' in sentence {i+1}"

        # Check sentence-initial rewrite/reordering markers
        for starter in CROSS_SENTENCE_REWRITE_STARTERS:
            if lower.startswith(starter):
                return True, f"cross_sentence_dependency_requires_review: rewrite starter '{starter}' in sentence {i+1}"

    return False, ""


def segment_instruction(text: str) -> List[str]:
    """
    Split an English instruction into ordered segments by execution sequence.

    Strategy:
    1. Split by explicit temporal markers (then, and then, after that, next, finally).
    2. Split by sentence boundaries.
    3. Split by comma when comma appears between action-like clauses.
    4. Trim and filter empty segments.
    """
    cleaned = normalize_whitespace(text)
    if not cleaned:
        return []

    # Step 1: split by explicit markers (case-insensitive, keep delimiters by using capture)
    pattern = "(" + "|".join(SPLIT_MARKERS) + ")"
    parts = re.split(pattern, cleaned, flags=re.IGNORECASE)

    # Reassemble: delimiter should stay with the following clause, but at start
    segments = []
    current = ""
    for part in parts:
        if re.match(pattern, part, flags=re.IGNORECASE):
            # delimiter starts a new segment; don't keep the marker in raw_text
            if current.strip():
                segments.append(current.strip())
            current = ""
        else:
            current += part
    if current.strip():
        segments.append(current.strip())

    # Step 2: split each by sentence boundaries
    expanded = []
    for seg in segments:
        # Split on period, exclamation, question mark when followed by space or end
        sub_parts = re.split(r'(?<=[.!?])\s+', seg)
        for sp in sub_parts:
            sp = sp.strip()
            if sp:
                expanded.append(sp)

    # Step 3: split by comma when comma separates two action-like fragments
    final_segments = []
    for seg in expanded:
        # Simple heuristic: if comma is between two verb-leading clauses, split
        # Look for patterns like "..., verb..."
        comma_splits = split_by_comma_clauses(seg)
        final_segments.extend(comma_splits)

    # Clean up trailing punctuation and filter standalone transition words
    result = []
    for seg in final_segments:
        seg = seg.strip().rstrip(",.!?;")
        if seg:
            # Filter out segments that are only transition words
            stripped_lower = seg.lower().strip(".,!?;:")
            if stripped_lower not in TRANSITION_WORDS:
                result.append(seg)

    return result


def split_by_comma_clauses(text: str) -> List[str]:
    """
    Split a segment by commas or 'and' when they separate what look like
    independent action clauses.
    """
    action_starters = {
        "walk", "go", "move", "turn", "stop", "enter", "exit",
        "pass", "face", "wait",
    }

    # First split by comma
    comma_parts = text.split(",")
    all_parts = []
    for cp in comma_parts:
        cp = cp.strip()
        if not cp:
            continue
        # Strip leading "and" if present (e.g., from "..., and go ...")
        cp = re.sub(r"^and\s+", "", cp, flags=re.IGNORECASE)
        # Then split each by " and " when followed by action verb
        # Use regex to find " and " before action words
        sub_parts = re.split(r"\s+and\s+(?=(?:" + "|".join(action_starters) + r")\b)", cp, flags=re.IGNORECASE)
        all_parts.extend(sub_parts)

    if len(all_parts) <= 1:
        return [text]

    result = []
    current = all_parts[0].strip()
    for part in all_parts[1:]:
        part_stripped = part.strip()
        raw_first = part_stripped.split()[0].lower() if part_stripped.split() else ""
        first_word = raw_first.strip(".,!?;:")
        if first_word in action_starters:
            if current:
                result.append(current)
            current = part_stripped
        else:
            current = current + ", " + part_stripped if current else part_stripped
    if current:
        result.append(current)

    return result
