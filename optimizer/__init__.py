"""Modular, config-driven prompt optimization framework.

See config.yml for enabling/disabling and tuning individual techniques, and
main.py for the command-line entry point.
"""

from .pipeline import OptimizationResult, PromptOptimizer, optimize_prompt

__all__ = ["OptimizationResult", "PromptOptimizer", "optimize_prompt"]
