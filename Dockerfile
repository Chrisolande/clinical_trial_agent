# syntax=docker/dockerfile:1.7
# python:3.13-slim is intentionally not digest-pinned here because this image is
# rebuilt by scheduled CI and scanned with Trivy to pick up patched Debian layers.
FROM python:3.13-slim

LABEL org.opencontainers.image.title="clinical-trial-agent" \
  org.opencontainers.image.description="Async LangGraph clinical trial matching agent" \
  org.opencontainers.image.source="https://github.com/Chrisolande/clinical_trial_agent" \
  org.opencontainers.image.licenses="MIT"

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

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

COPY pyproject.toml uv.lock /app/

RUN --mount=type=cache,target=/root/.cache/uv \
  uv export --frozen --no-dev --no-emit-project --format requirements-txt --output-file /app/requirements.txt \
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
