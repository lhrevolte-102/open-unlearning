# Agent Instructions

- Before running any `python`, `pip`, `pytest`, or `uv run` command, activate the virtual environment at the repository root: `source .venv/bin/activate`.
- Assume the project uses the root `.venv` by default. If the virtual environment does not exist yet, create it and sync dependencies before running Python commands.
- If you need to switch environments in a script or shell session, always make sure the current shell has the correct virtual environment activated before continuing.
- When adding or modifying features, experiments, evaluation logic, benchmarks, or related workflows, first consult the relevant documentation under `docs/` and keep the implementation aligned with the matching guide in `README.md`.
