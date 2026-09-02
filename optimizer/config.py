"""Loads and validates config.yml, the single place users toggle each
optimization module on/off and dial its aggressiveness (0-10)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yml"


@dataclass
class ModuleConfig:
    enabled: bool
    level: int
    options: dict = field(default_factory=dict)


@dataclass
class OptimizerConfig:
    protected_start_tag: str
    protected_end_tag: str
    modules: dict[str, ModuleConfig]
    pipeline_order: list[str]

    def module(self, name: str) -> ModuleConfig:
        return self.modules.get(name, ModuleConfig(enabled=False, level=0))


def load_config(path: str | Path | None = None) -> OptimizerConfig:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    protected = raw.get("protected") or {}
    modules_raw = raw.get("modules") or {}

    modules: dict[str, ModuleConfig] = {}
    for name, cfg in modules_raw.items():
        cfg = cfg or {}
        level = int(cfg.get("level", 0))
        if not 0 <= level <= 10:
            raise ValueError(f"modules.{name}.level must be between 0 and 10, got {level}")
        options = {k: v for k, v in cfg.items() if k not in ("enabled", "level")}
        modules[name] = ModuleConfig(
            enabled=bool(cfg.get("enabled", False)), level=level, options=options
        )

    pipeline_order = (raw.get("pipeline") or {}).get("order") or list(modules.keys())

    return OptimizerConfig(
        protected_start_tag=protected.get("start_tag", "<protect>"),
        protected_end_tag=protected.get("end_tag", "</protect>"),
        modules=modules,
        pipeline_order=pipeline_order,
    )
