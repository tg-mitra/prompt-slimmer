"""Command-line entry point for the prompt optimizer.

Usage:
    python main.py --input prompt.txt --output optimized.txt
    echo "your prompt" | python main.py
    python main.py -i prompt.txt --stats
"""

from __future__ import annotations

import argparse
import sys

from optimizer.pipeline import PromptOptimizer


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize a prompt using config.yml")
    parser.add_argument(
        "--input", "-i", help="Path to a file containing the prompt. Reads stdin if omitted."
    )
    parser.add_argument(
        "--output", "-o", help="Path to write the optimized prompt. Prints to stdout if omitted."
    )
    parser.add_argument(
        "--config", "-c", default=None, help="Path to config.yml (default: ./config.yml)"
    )
    parser.add_argument(
        "--stats", action="store_true", help="Print size-reduction stats to stderr"
    )
    return parser.parse_args(argv)


def _read_input(path: str | None) -> str:
    if path:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return sys.stdin.read()


def _write_output(path: str | None, text: str) -> None:
    if path:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    else:
        print(text)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prompt = _read_input(args.input)

    optimizer = PromptOptimizer(config_path=args.config)
    result = optimizer.optimize(prompt)

    _write_output(args.output, result.optimized_text)

    if args.stats:
        print(
            f"Original: {result.original_chars} chars -> Optimized: {result.optimized_chars} chars "
            f"({result.percent_saved}% reduction)",
            file=sys.stderr,
        )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
