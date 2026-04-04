# syntax=docker/dockerfile:1
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

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
     rustc \
     cargo \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

COPY pyproject.toml uv.lock* /app/

COPY . /app

RUN uv pip install --system --no-cache .

RUN useradd --create-home appuser \
  && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["clinical-trial-agent"]
CMD ["--help"]
