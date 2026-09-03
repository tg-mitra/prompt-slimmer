"""Structured, non-generative summarization for agentic chat history.

Chat history is JSON (messages with role/content/timestamp), not free-form
prompt text, so this runs its own pipeline rather than joining the
text-prompt pipeline in pipeline.py. As the requirements note, sentence
embeddings + scikit-learn are good at classification, clustering, and
ranking -- not generation -- so nothing here rewrites message text. Every
field in the output is composed of near-verbatim message snippets, selected
via rule-based multi-label categorization (_CATEGORY_PATTERNS) plus an
embedding-based importance score, then deduplicated the same way
consolidate.py/dedup.py already do for prompt text.

Pipeline: messages -> embeddings -> {classify by category, score importance}
-> rank + dedupe per category -> structured JSON summary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .clustering import find_near_duplicates
from .embeddings import EmbeddingBackend

# Multi-label: a message can match more than one category. Order here also
# sets the key order of the returned summary dict.
_CATEGORY_PATTERNS: dict[str, re.Pattern] = {
    "goal": re.compile(
        r"\b(i want to|i need to|i'd like to|i would like to|my goal is|help me)\b",
        re.IGNORECASE,
    ),
    "decision": re.compile(
        r"\b(i will|i'll|we will|we'll|let's|understood|agreed|decided|"
        r"going with|will use|will compare)\b",
        re.IGNORECASE,
    ),
    "constraint": re.compile(
        r"\b(exclude|excluding|only|must not|do not|don'?t|without|never|required? to)\b",
        re.IGNORECASE,
    ),
    "preference": re.compile(
        r"\b(i prefer|i'd rather|i only want|rather than|instead of)\b",
        re.IGNORECASE,
    ),
    "fact": re.compile(
        r"(\d+(\.\d+)?%)|\b(declined|increased|decreased|grew|dropped|totaled|"
        r"reached|significant)\b|\bshows? that\b|\bfound that\b|\bresults? (?:show|indicate)\b",
        re.IGNORECASE,
    ),
    "pending_task": re.compile(
        r"\b(create|generate|build|prepare|draft|identify|break down|perform)\b",
        re.IGNORECASE,
    ),
}

_CATEGORY_OUTPUT_KEYS: dict[str, str] = {
    "goal": "user_goals",
    "decision": "decisions",
    "constraint": "constraints",
    "preference": "user_preferences",
    "fact": "key_facts",
    "pending_task": "pending_tasks",
}

# Priority order for picking sentences into the natural-language summary,
# per the requirements: "prioritize goals + constraints + decisions +
# important facts + pending tasks".
_SUMMARY_CATEGORY_ORDER = ["goal", "constraint", "preference", "decision", "fact", "pending_task"]

_TOOL_ROLES = {"tool", "function"}

_LEADING_FILLER_RE = re.compile(
    r"^(yes,?\s+|sure\.?\s+|understood,?\s+|no,?\s+|ok(ay)?,?\s+)", re.IGNORECASE
)


@dataclass
class ChatMessage:
    message_id: int | str
    role: str
    content: str
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, raw: dict) -> "ChatMessage":
        return cls(
            message_id=raw.get("message_id"),
            role=raw.get("role", "user"),
            content=(raw.get("content") or "").strip(),
            timestamp=raw.get("timestamp"),
        )


def classify_message(content: str, role: str = "user") -> list[str]:
    """Multi-label rule-based categorization, e.g. a message can be both a
    "constraint" and a "user_preference" at once."""
    categories = [name for name, pattern in _CATEGORY_PATTERNS.items() if pattern.search(content)]
    if role in _TOOL_ROLES and "fact" not in categories:
        categories.append("fact")
    return categories


def _clean_snippet(content: str) -> str:
    content = _LEADING_FILLER_RE.sub("", content.strip())
    return content[:1].upper() + content[1:] if content else content


def _score_importance(
    contents: list[str], categories: list[list[str]], backend: EmbeddingBackend
) -> np.ndarray:
    """0-1 importance per message: how central it is to the conversation
    (embedding similarity to the centroid of all messages), whether it
    matched a known category, and recency (later messages -- closer to
    "now" -- are weighted slightly higher, since agents most need recent
    state)."""
    vectors = backend.encode(contents)
    centroid = vectors.mean(axis=0, keepdims=True)
    centroid_sim = cosine_similarity(vectors, centroid).flatten()
    spread = centroid_sim.max() - centroid_sim.min()
    centroid_sim = (centroid_sim - centroid_sim.min()) / (spread or 1.0)

    n = len(contents)
    recency = np.array([i / max(n - 1, 1) for i in range(n)])
    has_category = np.array([1.0 if cats else 0.0 for cats in categories])

    importance = 0.5 * centroid_sim + 0.3 * has_category + 0.2 * recency
    return np.clip(importance, 0.0, 1.0)


def _rank_dedup_trim(
    items: list[tuple[float, str]],
    max_items: int,
    dedup_threshold: float,
    backend: EmbeddingBackend,
) -> list[str]:
    if not items:
        return []
    items = sorted(items, key=lambda pair: -pair[0])
    texts = [text for _, text in items]
    dropped = find_near_duplicates(texts, backend, dedup_threshold) if len(texts) > 1 else set()
    kept = [text for i, text in enumerate(texts) if i not in dropped]
    return kept[:max_items]


def _build_summary(
    messages: list[ChatMessage], categories: list[list[str]], importance: np.ndarray
) -> str:
    scored = list(zip(importance, messages, categories))
    picked: list[str] = []
    used_ids = set()

    for target_category in _SUMMARY_CATEGORY_ORDER:
        candidates = [
            (score, m)
            for score, m, cats in scored
            if target_category in cats and m.message_id not in used_ids
        ]
        if not candidates:
            continue
        _, best = max(candidates, key=lambda pair: pair[0])
        picked.append(_clean_snippet(best.content))
        used_ids.add(best.message_id)

    if not picked:
        _, best, _ = max(scored, key=lambda triple: triple[0])
        picked.append(_clean_snippet(best.content))

    return " ".join(s if s.endswith((".", "?", "!")) else s + "." for s in picked)


def summarize_chat_history(
    history: dict,
    embedding_model: str = "all-MiniLM-L6-v2",
    max_items_per_category: int = 5,
    dedup_similarity_threshold: float = 0.85,
    backend: EmbeddingBackend | None = None,
) -> dict:
    """Turn a JSON chat history (see docs/chat_history_summarization.MD for
    the input shape) into a structured JSON summary: a short extractive
    summary plus goals / decisions / constraints / preferences / facts /
    pending tasks, each backed by near-verbatim message snippets."""
    messages = [ChatMessage.from_dict(m) for m in history.get("messages", [])]
    messages = [m for m in messages if m.content]

    if not messages:
        return {
            "conversation_id": history.get("conversation_id"),
            "summary": "",
            "user_goals": [],
            "decisions": [],
            "constraints": [],
            "user_preferences": [],
            "key_facts": [],
            "pending_tasks": [],
            "message_classifications": [],
        }

    backend = backend or EmbeddingBackend(embedding_model)
    contents = [m.content for m in messages]
    categories = [classify_message(c, m.role) for c, m in zip(contents, messages)]
    importance = _score_importance(contents, categories, backend)

    buckets: dict[str, list[tuple[float, str]]] = {key: [] for key in _CATEGORY_OUTPUT_KEYS.values()}
    for m, cats, score in zip(messages, categories, importance):
        for cat in cats:
            buckets[_CATEGORY_OUTPUT_KEYS[cat]].append((float(score), _clean_snippet(m.content)))

    result = {
        key: _rank_dedup_trim(items, max_items_per_category, dedup_similarity_threshold, backend)
        for key, items in buckets.items()
    }

    return {
        "conversation_id": history.get("conversation_id"),
        "summary": _build_summary(messages, categories, importance),
        **result,
        "message_classifications": [
            {
                "message_id": m.message_id,
                "categories": cats,
                "importance": round(float(score), 2),
            }
            for m, cats, score in zip(messages, categories, importance)
        ],
    }
