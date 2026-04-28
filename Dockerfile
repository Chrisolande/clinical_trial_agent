# syntax=docker/dockerfile:1.7
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
  PYTHONUNBUFFERED=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends \
  build-essential \
  gcc \
  libpq-dev \
  libffi-dev \
  libssl-dev \
  curl \
  git \
  ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.8.5@sha256:9ac8566d708f42bae522b050004f75ebc7c344bc726d6d4e70f1d308b18c4471 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv export --frozen --no-dev --format requirements-txt --output-file /app/requirements.txt \
  && uv pip install --system -r /app/requirements.txt \
  && rm -f /app/requirements.txt

COPY . /app

RUN --mount=type=cache,target=/root/.cache/uv \
  uv pip install --system --no-deps .

RUN useradd --create-home appuser \
  && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["clinical-trial-agent"]
CMD ["--help"]
