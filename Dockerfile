# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies required to build wheels and for libpq
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

# Pull the official uv binaries directly into the image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy project metadata first for better layer caching
COPY pyproject.toml uv.lock* /app/

# Copy the rest of the source code
COPY . /app

# Install the package globally using uv
RUN uv pip install --system --no-cache .

# Create an unprivileged user and ensure ownership
RUN useradd --create-home appuser \
  && chown -R appuser:appuser /app

USER appuser

ENTRYPOINT ["clinical-trial-agent"]
CMD ["--help"]
