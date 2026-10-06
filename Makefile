EXPERIMENTS := e0a e0b e0c e1a e1b e2a e2b e3 e4 e5 e6 e7 e8

.PHONY: setup lint format test test-gpu validate data scoreboard serve $(EXPERIMENTS)

setup:
	uv sync

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest

test-gpu:
	uv run pytest -m gpu

validate:
	uv run python scripts/validate_configs.py

data:
	@echo "lands with article 2: scripts/download_data.py" && exit 2

scoreboard:
	@echo "lands with article 2: clef_finetuning.eval.scoreboard" && exit 2

serve:
	@echo "lands with article 6: clef_finetuning.serve.api" && exit 2

$(EXPERIMENTS):
	uv run python scripts/run_experiment.py $@
