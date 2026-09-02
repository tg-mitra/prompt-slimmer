"""Wires the individual optimization modules into a single pipeline.

Techniques are applied in the order given by config.yml's pipeline.order --
by default, safe/cheap text cleanup first, semantic/structural rewriting
last -- so that later, riskier steps operate on already-cleaned text.
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import OptimizerConfig, load_config
from .consolidate import consolidate_instructions
from .dedup import deduplicate_context
from .embeddings import EmbeddingBackend
from .filler import remove_filler_phrases
from .structure import convert_to_structure
from .summarizer import summarize_long_context
from .text_utils import protect_sections

_MODULE_FUNCS = {
    "remove_filler_phrases": lambda text, level, options, backend: remove_filler_phrases(
        text, level
    ),
    "consolidate_instructions": lambda text, level, options, backend: consolidate_instructions(
        text, level, backend
    ),
    "deduplicate_context": lambda text, level, options, backend: deduplicate_context(
        text, level, backend
    ),
    "convert_to_structure": lambda text, level, options, backend: convert_to_structure(
        text, level
    ),
    "semantic_summarization": lambda text, level, options, backend: summarize_long_context(
        text,
        level,
        min_chars_to_trigger=int(options.get("min_chars_to_trigger", 800)),
        min_retained_similarity=float(options.get("min_retained_similarity", 0.6)),
        backend=backend,
    ),
}


@dataclass
class OptimizationResult:
    optimized_text: str
    original_chars: int
    optimized_chars: int

    @property
    def chars_saved(self) -> int:
        return self.original_chars - self.optimized_chars

    @property
    def percent_saved(self) -> float:
        if self.original_chars == 0:
            return 0.0
        return round(self.chars_saved / self.original_chars * 100, 1)


class PromptOptimizer:
    def __init__(
        self, config: OptimizerConfig | None = None, config_path: str | None = None
    ):
        self.config = config or load_config(config_path)
        model = self._pick_embedding_model()
        self._backend = EmbeddingBackend(model) if model else EmbeddingBackend()

    def _pick_embedding_model(self) -> str | None:
        for module_name in self.config.pipeline_order:
            model = self.config.module(module_name).options.get("embedding_model")
            if model:
                return model
        return None

    def optimize(self, prompt: str) -> OptimizationResult:
        original_chars = len(prompt)

        protected = protect_sections(
            prompt, self.config.protected_start_tag, self.config.protected_end_tag
        )
        text = protected.masked_text

        for module_name in self.config.pipeline_order:
            module_cfg = self.config.module(module_name)
            if not module_cfg.enabled or module_cfg.level <= 0:
                continue
            func = _MODULE_FUNCS.get(module_name)
            if func is None:
                continue
            text = func(text, module_cfg.level, module_cfg.options, self._backend)

        text = protected.restore(text).strip()
        return OptimizationResult(
            optimized_text=text, original_chars=original_chars, optimized_chars=len(text)
        )


def optimize_prompt(prompt: str, config_path: str | None = None) -> str:
    """Convenience function for other applications to import directly."""
    return PromptOptimizer(config_path=config_path).optimize(prompt).optimized_text
