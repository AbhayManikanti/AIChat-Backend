# Multi-stage Dockerfile optimized for Google Cloud Run
# Ensures proper SQLite3 version for ChromaDB compatibility

FROM python:3.9-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-slim

# Install runtime dependencies including newer SQLite3
RUN apt-get update && apt-get install -y \
    sqlite3 \
    libsqlite3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Verify SQLite3 version (should be >= 3.35.0)
RUN sqlite3 --version

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create non-root user for security (Cloud Run best practice)
RUN useradd --create-home --shell /bin/bash appuser

# Set working directory
WORKDIR /app

# Copy application files with proper ownership
COPY --chown=appuser:appuser . /app/

# Create directory for ChromaDB persistence
RUN mkdir -p /app/chroma_db && chown -R appuser:appuser /app/chroma_db

# Switch to non-root user
USER appuser

# Expose port (Cloud Run will override with PORT env var)
EXPOSE 8000

# Health check for Cloud Run readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:${PORT:-8000}/health', timeout=5)" || exit 1

# Set environment variables for Cloud Run
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Command to run the application (uses PORT env var from Cloud Run)
CMD ["python", "app.py"]