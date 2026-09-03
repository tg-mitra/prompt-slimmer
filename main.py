"""Command-line entry point for the prompt optimizer.

Usage:
    python main.py --input prompt.txt --output optimized.txt
    echo "your prompt" | python main.py
    python main.py -i prompt.txt --stats
    python main.py --chat-history conversation.json --stats
"""

from __future__ import annotations

import argparse
import json
import sys

from optimizer.chat_history import summarize_chat_history
from optimizer.config import load_config
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
    parser.add_argument(
        "--chat-history",
        help=(
            "Path to a JSON chat-history file (see "
            "docs/chat_history_summarization.MD). Runs structured chat "
            "history summarization instead of prompt optimization; "
            "--input/--output still apply, --stats prints message/category "
            "counts instead of char counts."
        ),
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


def _run_chat_history(args: argparse.Namespace) -> int:
    history = json.loads(_read_input(args.chat_history))
    config = load_config(args.config).chat_history

    summary = summarize_chat_history(
        history,
        embedding_model=config.embedding_model,
        max_items_per_category=config.max_items_per_category,
        dedup_similarity_threshold=config.dedup_similarity_threshold,
    )

    _write_output(args.output, json.dumps(summary, indent=2))

    if args.stats:
        categorized = sum(1 for c in summary["message_classifications"] if c["categories"])
        print(
            f"Messages: {len(history.get('messages', []))} "
            f"(categorized: {categorized}) -> "
            f"goals: {len(summary['user_goals'])}, decisions: {len(summary['decisions'])}, "
            f"constraints: {len(summary['constraints'])}, "
            f"preferences: {len(summary['user_preferences'])}, "
            f"facts: {len(summary['key_facts'])}, pending: {len(summary['pending_tasks'])}",
            file=sys.stderr,
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.chat_history:
        return _run_chat_history(args)

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
