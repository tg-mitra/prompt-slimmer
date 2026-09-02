"""Rewrites verbose, comma-chained instruction sentences into a compact
bullet list, e.g. "You should classify the ticket, identify urgency,
provide a short reason, and return the result in JSON format" becomes:

    Perform the following:
    - Classify the ticket
    - Identify urgency
    - Provide a short reason
    - Return the result in JSON format
"""

from __future__ import annotations

import re

from .text_utils import is_directive_sentence, split_paragraphs, split_sentences

_LEAD_IN_RE = re.compile(r"^(?:you (?:should|must|need to)|please|kindly)\s+", re.IGNORECASE)
_AND_SPLIT_RE = re.compile(r",?\s+and\s+", re.IGNORECASE)


def _min_items_for_level(level: int) -> int:
    return max(2, 5 - level // 2)


def _split_clauses(sentence: str) -> list[str]:
    sentence = sentence.rstrip(".")
    parts = [
        p.strip()
        for chunk in sentence.split(",")
        for p in _AND_SPLIT_RE.split(chunk)
    ]
    return [p for p in parts if p]


def _looks_like_instruction_list(sentence: str) -> bool:
    return is_directive_sentence(sentence) or bool(_LEAD_IN_RE.match(sentence.strip()))


def convert_to_structure(text: str, level: int) -> str:
    if level <= 0:
        return text

    min_items = _min_items_for_level(level)
    out_paragraphs = []

    for paragraph in split_paragraphs(text):
        blocks: list[str] = []
        buffer: list[str] = []

        for sentence in split_sentences(paragraph):
            clauses = _split_clauses(sentence)
            if _looks_like_instruction_list(sentence) and len(clauses) >= min_items:
                if buffer:
                    blocks.append(" ".join(buffer))
                    buffer = []
                blocks.append(_render_bullets(clauses))
            else:
                buffer.append(sentence)

        if buffer:
            blocks.append(" ".join(buffer))
        out_paragraphs.append("\n".join(blocks))

    return "\n\n".join(out_paragraphs)


def _render_bullets(clauses: list[str]) -> str:
    first = _LEAD_IN_RE.sub("", clauses[0]).strip()
    clauses = [first, *clauses[1:]]
    lines = ["Perform the following:"]
    for clause in clauses:
        clause = clause[0].upper() + clause[1:] if clause else clause
        lines.append(f"- {clause}")
    return "\n".join(lines)
