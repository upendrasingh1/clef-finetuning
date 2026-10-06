"""Run one experiment by id, e.g. `python scripts/run_experiment.py e0c`."""

from __future__ import annotations

import argparse
import sys

from clef_finetuning.config import load_experiments, validate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_id")
    args = parser.parse_args()

    if errors := validate():
        print("\n".join(errors), file=sys.stderr)
        return 1
    experiments = load_experiments()
    if args.experiment_id not in experiments:
        print(f"unknown experiment {args.experiment_id!r}; known: {', '.join(experiments)}", file=sys.stderr)
        return 1
    exp = experiments[args.experiment_id]
    print(f"{exp.id}: {exp.title} [{exp.track}/{exp.method}, {exp.model}, {exp.precision}]")
    print(f"not implemented yet: lands with article {exp.article} (method {exp.method!r})", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
