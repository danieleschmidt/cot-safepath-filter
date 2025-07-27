# Multi-stage Docker build for CoT SafePath Filter
# Optimized for security, performance, and minimal attack surface

# Build stage
FROM python:3.11-slim-bullseye as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Set environment variables for build
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for build
RUN useradd --create-home --shell /bin/bash app

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md LICENSE ./
COPY src/ src/

# Install Python dependencies
RUN pip install --no-cache-dir build wheel && \
    python -m build --wheel && \
    pip install dist/*.whl

# Production stage
FROM python:3.11-slim-bullseye as production

# Set build labels for metadata
LABEL org.opencontainers.image.title="CoT SafePath Filter" \
      org.opencontainers.image.description="Real-time middleware for filtering harmful chain-of-thought reasoning" \
      org.opencontainers.image.version="${VERSION}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.vendor="Terragon Labs" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.source="https://github.com/terragonlabs/cot-safepath-filter"

# Set production environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ENVIRONMENT=production \
    LOG_LEVEL=INFO \
    PORT=8080

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for production
RUN useradd --create-home --shell /bin/bash --uid 1000 app

# Set work directory
WORKDIR /app

# Copy built wheel from builder stage
COPY --from=builder /app/dist/*.whl ./

# Install the application
RUN pip install --no-cache-dir *.whl && \
    rm -f *.whl

# Copy configuration files
COPY docker-entrypoint.sh ./
RUN chmod +x docker-entrypoint.sh

# Create necessary directories
RUN mkdir -p /app/logs /app/data /app/tmp && \
    chown -R app:app /app

# Switch to non-root user
USER app

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command
CMD ["safepath-server"]

# Development stage (for development containers)
FROM python:3.11-slim-bullseye as development

# Install development dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Create user for development
RUN useradd --create-home --shell /bin/bash --uid 1000 dev

# Set work directory
WORKDIR /workspace

# Install development Python packages
RUN pip install --no-cache-dir \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pre-commit

# Switch to development user
USER dev

# Default command for development
CMD ["bash"]