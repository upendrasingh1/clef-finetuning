"""Validate all configs and print a summary of experiments."""

from __future__ import annotations

import sys

from clef_finetuning.config import load_experiments, validate


def main() -> int:
    if errors := validate():
        print("\n".join(errors), file=sys.stderr)
        return 1
    for exp in load_experiments().values():
        print(
            f"{exp.id:4} article {exp.article}  {exp.track:11} {exp.method:20} {exp.model:11} {exp.precision}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
