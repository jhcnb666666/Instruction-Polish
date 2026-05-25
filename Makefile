.PHONY: install test test-vln lint clean example

install:
	pip install -e .

test:
	python -m pytest tests/ -v

test-vln:
	python -m pytest vln_instruction_parser/tests -v

lint:
	python -m flake8 src tests || true
	python -m black --check src tests || true

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build dist *.egg-info .pytest_cache

example:
	instruction-polish "can u help me write python code" --strategy clarity
	instruction-polish --input-file examples/sample_instructions.txt --strategy conciseness
