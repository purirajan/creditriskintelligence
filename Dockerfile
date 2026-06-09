# ── Base image ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

# Metadata
LABEL maintainer="Rajan Puri <purirajan.rp@gmail.com>"
LABEL description="CreditRisk Intelligence API — Basel-aligned credit risk platform"
LABEL version="1.0.0"

# ── Environment variables ────────────────────────────────────────────────────
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=8000

# ── System dependencies ──────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ────────────────────────────────────────────────────────
WORKDIR /app

# ── Install Python dependencies ──────────────────────────────────────────────
# Copy requirements first (layer caching — only re-installs if requirements change)
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────────────────────────
COPY src/ ./src/

# ── Create non-root user (security best practice) ────────────────────────────
RUN addgroup --system appgroup && \
    adduser --system --ingroup appgroup appuser && \
    chown -R appuser:appgroup /app
USER appuser

# ── Health check ─────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# ── Expose port ──────────────────────────────────────────────────────────────
EXPOSE ${PORT}

# ── Start the API ─────────────────────────────────────────────────────────────
CMD ["uvicorn", "src.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "2", \
     "--log-level", "info"]
