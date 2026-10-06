"""Experiment, dataset, model and hardware configs, with cross-reference validation."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "configs"

TRACKS = {"baseline", "trl_a", "trl_b", "trl_c", "clef_native"}
METHODS = {
    "zero_shot",
    "trl_sft",
    "trl_grpo",
    "clef_head_only",
    "clef_split_lora",
    "clef_expected_reward",
    "trl_clef_grpo",
}
PRECISIONS = {"bf16", "int8", "nf4"}


@dataclass(frozen=True)
class ExperimentConfig:
    id: str
    title: str
    article: int
    track: str
    model: str
    method: str
    precision: str
    datasets: tuple[str, ...]
    hardware: str
    seed: int
    init_from: str | None = None
    teacher: str | None = None
    gold_fraction: float = 1.0
    path: Path | None = None


def _read(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def load_models() -> dict[str, dict[str, Any]]:
    return _read(CONFIG_DIR / "models.yaml")


def load_named(kind: str) -> dict[str, dict[str, Any]]:
    return {path.stem: _read(path) for path in sorted((CONFIG_DIR / kind).glob("*.yaml"))}


def load_experiment(path: Path) -> ExperimentConfig:
    raw = _read(path)
    unknown = set(raw) - {f.name for f in fields(ExperimentConfig)}
    if unknown:
        raise ValueError(f"{path.name}: unknown keys {sorted(unknown)}")
    return ExperimentConfig(**{**raw, "datasets": tuple(raw["datasets"]), "path": path})


def load_experiments() -> dict[str, ExperimentConfig]:
    experiments = [load_experiment(p) for p in sorted((CONFIG_DIR / "experiments").glob("*.yaml"))]
    return {e.id: e for e in experiments}


def validate() -> list[str]:
    """Return a list of problems across all configs; empty means valid."""
    models, datasets, hardware = load_models(), load_named("data"), load_named("hardware")
    experiments = load_experiments()
    errors: list[str] = []
    for exp in experiments.values():
        where = exp.path.name if exp.path else exp.id
        checks = [
            (exp.path is not None and exp.path.stem.split("_")[0] == exp.id, "file name must start with id"),
            (exp.track in TRACKS, f"unknown track {exp.track!r}"),
            (exp.method in METHODS, f"unknown method {exp.method!r}"),
            (exp.precision in PRECISIONS, f"unknown precision {exp.precision!r}"),
            (exp.model in models, f"unknown model {exp.model!r}"),
            (exp.hardware in hardware, f"unknown hardware {exp.hardware!r}"),
            (1 <= exp.article <= 7, "article must be 1-7"),
            (0 < exp.gold_fraction <= 1, "gold_fraction must be in (0, 1]"),
            (exp.init_from is None or exp.init_from in experiments, f"unknown init_from {exp.init_from!r}"),
            (exp.teacher is None or exp.teacher in experiments, f"unknown teacher {exp.teacher!r}"),
        ]
        errors += [f"{where}: {message}" for ok, message in checks if not ok]
        errors += [f"{where}: unknown dataset {d!r}" for d in exp.datasets if d not in datasets]
        for ref in (exp.init_from, exp.teacher):
            if ref in experiments and experiments[ref].article > exp.article:
                errors.append(f"{where}: depends on {ref}, which is published in a later article")
    return errors
