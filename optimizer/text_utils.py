"""Text parsing utilities shared by every optimization module.

Handles two concerns that don't belong to any single technique:
  - protecting user-marked sections so no module ever touches them
  - splitting prompts into paragraphs/sentences and reassembling them
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")
_PARAGRAPH_SPLIT_RE = re.compile(r"\n\s*\n")

_DIRECTIVE_MARKERS_RE = re.compile(
    r"\b(should|must|need(?:s)? to|have to|make sure|ensure|avoid|keep|"
    r"don't|do not|never|always|please|required to|remember to|"
    r"be (?:concise|brief|short|clear|detailed|thorough))\b",
    re.IGNORECASE,
)


@dataclass
class ProtectedText:
    """A prompt with protected sections swapped out for placeholders."""

    masked_text: str
    sections: dict[str, str] = field(default_factory=dict)

    def restore(self, text: str) -> str:
        for placeholder, original in self.sections.items():
            text = text.replace(placeholder, original)
        return text


def protect_sections(text: str, start_tag: str, end_tag: str) -> ProtectedText:
    """Replace everything between start_tag/end_tag with a placeholder so
    downstream modules never see, and therefore never alter, that text."""
    pattern = re.compile(re.escape(start_tag) + r"(.*?)" + re.escape(end_tag), re.DOTALL)
    sections: dict[str, str] = {}

    def _replace(match: re.Match) -> str:
        placeholder = f"§§PROTECTED{len(sections)}§§"
        sections[placeholder] = match.group(0)
        return placeholder

    return ProtectedText(masked_text=pattern.sub(_replace, text), sections=sections)


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in _PARAGRAPH_SPLIT_RE.split(text) if p.strip()]


def split_sentences(paragraph: str) -> list[str]:
    """Dependency-free sentence splitter. Good enough for prompt text; not
    meant to handle every abbreviation edge case."""
    line = re.sub(r"\s+", " ", paragraph).strip()
    if not line:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(line) if s.strip()]


def is_directive_sentence(sentence: str) -> bool:
    """Heuristic: does this sentence read like an instruction/constraint
    rather than narrative background?"""
    return bool(_DIRECTIVE_MARKERS_RE.search(sentence))


@dataclass
class SentenceDoc:
    """A prompt broken into paragraphs of sentences, with helpers to drop
    sentences by flat index and reassemble the remaining text."""

    paragraphs: list[list[str]]

    @classmethod
    def from_text(cls, text: str) -> "SentenceDoc":
        return cls([split_sentences(p) for p in split_paragraphs(text)])

    def flatten(self) -> list[str]:
        return [s for para in self.paragraphs for s in para]

    def to_text(self, dropped: set[int] | None = None) -> str:
        dropped = dropped or set()
        flat_idx = 0
        out_paragraphs = []
        for para in self.paragraphs:
            kept = []
            for s in para:
                if flat_idx not in dropped:
                    kept.append(s)
                flat_idx += 1
            if kept:
                out_paragraphs.append(" ".join(kept))
        return "\n\n".join(out_paragraphs)
