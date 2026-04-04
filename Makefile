.PHONY: lint typecheck test security complexity check fix docker-build docker-up docker-down docker-logs docker-ps docker-recreate

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy .

test:
	pytest

security:
	bandit -r . -ll
	pip-audit --strict --skip-editable . --ignore-vuln CVE-2025-69872

complexity:
	radon cc --min B .

check: lint typecheck test security complexity

fix:
	ruff format .
	ruff check --fix .

# Docker helpers
docker-build:
	docker build -t clinical-trial-agent:local .

docker-up:
	docker-compose -f docker-compose.yml -p clinical_trial_agent up -d --build

docker-down:
	docker-compose -f docker-compose.yml -p clinical_trial_agent down --remove-orphans

docker-logs:
	docker-compose -f docker-compose.yml -p clinical_trial_agent logs -f

docker-ps:
	docker-compose -f docker-compose.yml -p clinical_trial_agent ps

docker-recreate:
	docker-compose -f docker-compose.yml -p clinical_trial_agent down --remove-orphans && \
	docker-compose -f docker-compose.yml -p clinical_trial_agent up -d --build
