.PHONY: lint typecheck test security complexity eval check fix docker-build docker-up docker-down docker-logs docker-ps docker-recreate

lint:
	uv run ruff check .
	uv run ruff format --check .

typecheck:
	uv run mypy .

test:
	uv run pytest

security:
	uv run bandit -r clinical_trial_agent agents subagents tools models -ll

complexity:
	uv run radon cc --min B .

eval:
	uv run python -m evaluation.runners.run_offline_eval

check: lint typecheck test security complexity

fix:
	uv run ruff format .
	uv run ruff check --fix .

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
