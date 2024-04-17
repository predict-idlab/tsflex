black = black tsflex tests

.PHONY: format
format:
	ruff --fix tsflex tests
	$(black)

.PHONY: lint
lint:
	poetry run ruff tsflex tests
	poetry run $(black) --check --diff
	poetry run mypy tsflex # tests

.PHONY: test
test:
	poetry run pytest --cov-report term-missing --cov=tsflex tests

.PHONY: bench
bench:
	poetry run pytest --benchmark-only

.PHONY: clean
clean:
	rm -rf `find . -name __pycache__`
	rm -rf .cache
	rm -rf .pytest_cache
	rm -rf .benchmarks
	rm -rf *.egg-info
	rm -rf .ruff_cache
	rm -f .coverage
