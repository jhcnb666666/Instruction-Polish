"""Command-line interface for instruction-polish."""

import argparse
import json
import sys

from .core import polish, analyze, list_strategies
from .batch import process_file
from .config import load_config
from .suggest import suggest_strategy
from .utils import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Polish AI instructions")
    parser.add_argument("instruction", nargs="?", help="The instruction to polish")
    parser.add_argument(
        "--strategy",
        default=None,
        help="Polishing strategy to use",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Show available strategies and exit",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        dest="do_analyze",
        help="Analyze the instruction instead of polishing",
    )
    parser.add_argument(
        "--input-file",
        default=None,
        help="File with one instruction per line",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="JSON file to write batch results",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to JSON config file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all strategies on the given instruction",
    )
    parser.add_argument(
        "--suggest",
        action="store_true",
        help="Suggest the best strategy for the instruction",
    )
    parser.add_argument(
        "--best",
        action="store_true",
        help="Automatically polish using the suggested best strategy",
    )
    parser.add_argument(
        "--diff",
        action="store_true",
        help="Show character-level diff of changes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()
    logger = setup_logging(level=("DEBUG" if args.verbose else "INFO"))

    config = load_config(args.config)
    strategy = args.strategy or config.get("default_strategy", "default")

    if args.list_strategies:
        print("Available strategies:")
        for s in list_strategies():
            print(f"  - {s}")
        sys.exit(0)

    if args.suggest:
        if not args.instruction:
            parser.error("Provide an INSTRUCTION to get a suggestion")
        suggestion = suggest_strategy(args.instruction)
        print(json.dumps(suggestion, ensure_ascii=False, indent=2))
        sys.exit(0)

    if args.best:
        if not args.instruction:
            parser.error("Provide an INSTRUCTION to auto-polish")
        suggestion = suggest_strategy(args.instruction)
        best_strategy = suggestion["strategy"]
        polished = polish(args.instruction, strategy=best_strategy)
        print(f"# strategy: {best_strategy} (score: {suggestion['score']})")
        print(polished)
        sys.exit(0)

    if args.compare:
        if not args.instruction:
            parser.error("Provide an INSTRUCTION to compare strategies")
        print(f"Original: {args.instruction!r}\n")
        for s in list_strategies():
            print(f"[{s:12s}] {polish(args.instruction, strategy=s)!r}")
        sys.exit(0)

    if args.input_file:
        results = process_file(args.input_file, args.output_file, strategy=strategy)
        if not args.output_file:
            print(json.dumps(results, ensure_ascii=False, indent=2))
        sys.exit(0)

    if not args.instruction:
        parser.error("Provide an INSTRUCTION or use --input-file")

    if args.do_analyze:
        result = analyze(args.instruction)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        sys.exit(0)

    polished = polish(args.instruction, strategy=strategy)
    if args.diff:
        import difflib
        diff = difflib.unified_diff(
            args.instruction.splitlines(keepends=True),
            polished.splitlines(keepends=True),
            fromfile="original",
            tofile="polished",
        )
        print("".join(diff), end="")
    else:
        print(polished)


if __name__ == "__main__":
    main()
