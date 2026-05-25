"""Batch processing for multiple instructions."""

import json
from pathlib import Path
from typing import Iterable, Dict, Any

from .core import polish


def process_lines(lines: Iterable[str], strategy: str = "default") -> list[Dict[str, Any]]:
    """
    Polish multiple instructions.

    Returns a list of dicts with keys: original, polished, strategy.
    """
    results = []
    for line in lines:
        original = line.strip()
        if not original:
            continue
        results.append({
            "original": original,
            "polished": polish(original, strategy=strategy),
            "strategy": strategy,
        })
    return results


def process_file(input_path: str, output_path: str | None = None, strategy: str = "default") -> list[Dict[str, Any]]:
    """
    Read instructions from a text file (one per line) or JSON Lines file
    and optionally write results as JSON or JSON Lines.
    """
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    # Detect JSON Lines by extension or first line
    text = path.read_text(encoding="utf-8")
    is_jsonl = path.suffix == ".jsonl" or (text.strip() and text.strip()[0] == "{")

    if is_jsonl:
        lines = [json.loads(line)["instruction"] for line in text.splitlines() if line.strip()]
    else:
        lines = text.splitlines()

    results = process_lines(lines, strategy=strategy)

    if output_path:
        out_path = Path(output_path)
        if out_path.suffix == ".jsonl":
            with out_path.open("w", encoding="utf-8") as f:
                for r in results:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        else:
            out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    return results
