FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --default-timeout=1000 --prefix=/install -r requirements.txt

# ── Runtime ─────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install runtime system libs and create non-root user in a single layer
RUN apt-get update && apt-get install -y --no-install-recommends libopenblas0 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 1001 ego \
    && useradd --uid 1001 --gid ego --shell /bin/bash --create-home ego

WORKDIR /app

COPY --from=builder /install /usr/local

# Copy only application source (not the whole context)
COPY api/ ./api/
COPY agents/ ./agents/
COPY core/ ./core/
COPY lib/ ./lib/
COPY scripts/ ./scripts/
COPY data/ ./data/
COPY graphs/ ./graphs/

RUN mkdir -p scratch/cache data \
    && chown -R ego:ego /app

USER ego

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

