# Factor Forecasting System - Production Docker Image
# Multi-stage build for optimized production deployment

# Build stage
FROM python:3.10-slim as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Create application directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.10-slim as production

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FACTOR_ENV=production \
    FACTOR_LOG_LEVEL=INFO

# Install runtime system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    procps \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN groupadd -r factor && useradd -r -g factor factor

# Create application directory and set permissions
WORKDIR /app
RUN mkdir -p /app/data /app/outputs /app/logs && \
    chown -R factor:factor /app

# Copy Python dependencies from builder stage
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=factor:factor . .

# Install the package in development mode
RUN pip install -e .

# Create entrypoint script
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Default command is API server\n\
CMD=${1:-api}\n\
\n\
case $CMD in\n\
    "api")\n\
        echo "Starting Factor Forecasting API Server..."\n\
        exec python -m src.api.api_server --host 0.0.0.0 --port ${FACTOR_API_PORT:-8000}\n\
        ;;\n\
    "train")\n\
        echo "Starting Factor Forecasting Training..."\n\
        exec python -m src.training.train --config ${FACTOR_CONFIG:-config.yaml}\n\
        ;;\n\
    "eval")\n\
        echo "Starting Factor Forecasting Evaluation..."\n\
        exec python -m src.evaluation.evaluate --config ${FACTOR_CONFIG:-config.yaml}\n\
        ;;\n\
    "worker")\n\
        echo "Starting Factor Forecasting Worker..."\n\
        exec python -m src.api.worker\n\
        ;;\n\
    "shell")\n\
        echo "Starting interactive shell..."\n\
        exec /bin/bash\n\
        ;;\n\
    *)\n\
        echo "Unknown command: $CMD"\n\
        echo "Available commands: api, train, eval, worker, shell"\n\
        exit 1\n\
        ;;\n\
esac' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Switch to non-root user
USER factor

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${FACTOR_API_PORT:-8000}/health || exit 1

# Expose ports
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["api"]

# Labels for metadata
LABEL maintainer="Factor Forecasting Team <team@factorforecasting.com>" \
      version="2.0.0" \
      description="Professional quantitative finance system for stock return prediction" \
      org.opencontainers.image.title="Factor Forecasting System" \
      org.opencontainers.image.description="Professional quantitative finance system for stock return prediction" \
      org.opencontainers.image.version="2.0.0" \
      org.opencontainers.image.vendor="Factor Forecasting Team" \
      org.opencontainers.image.licenses="MIT"
