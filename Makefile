.PHONY: quality style

check_dirs := scripts src

quality:
	ruff check $(check_dirs) setup_data.py
	ruff format --check $(check_dirs) setup_data.py

style:
	ruff check $(check_dirs) setup_data.py --fix
	ruff format $(check_dirs) setup_data.py

test:
	CUDA_VISIBLE_DEVICES= pytest tests/
