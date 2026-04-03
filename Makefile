.PHONY: lint typecheck test security complexity check fix

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy .

test:
	pytest

security:
	bandit -r . -ll

complexity:
	radon cc --min B .

check: lint typecheck test security complexity

fix:
	ruff format .
	ruff check --fix .
