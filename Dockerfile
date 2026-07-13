# Build stage: needs build-essential to compile any deps without a
# prebuilt wheel for this platform. Installed to /install so only the
# resulting packages (not the compiler toolchain) get copied below.
FROM python:3.12-slim AS builder

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --no-compile --prefix=/install -r requirements.txt

# Final stage: no compiler toolchain, just the installed packages + app code.
FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY --from=builder /install /usr/local

COPY . .
RUN mkdir -p data/seed

EXPOSE 8000

# Render (and most PaaS platforms) assign the actual listen port via the
# $PORT env var at runtime and route traffic/health checks to it - a
# hardcoded --port here would make the container unreachable even though
# it "runs" successfully. Falls back to 8000 for local `docker run`.
# Shell form (not exec form) so $PORT is actually expanded; `exec` hands
# off to uvicorn as PID 1 so it receives SIGTERM directly for a clean
# shutdown, instead of an intermediate shell swallowing the signal.
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
