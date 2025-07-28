# Base image with Python and Node.js
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye as development

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    make \
    unzip \
    wget \
    libpq-dev \
    postgresql-client \
    redis-tools \
    htop \
    tree \
    vim \
    fish \
    bash-completion \
    && rm -rf /var/lib/apt/lists/*

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y

# Install additional security tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    semgrep \
    pre-commit \
    commitizen \
    pip-audit \
    && rm -rf /root/.cache/pip

# Create workspace directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements*.txt pyproject.toml* setup.py* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi
RUN if [ -f pyproject.toml ]; then pip install --no-cache-dir -e ".[dev,test,security,performance,integrations,docs]"; fi

# Set up git safe directory and shell environment
RUN git config --global --add safe.directory /workspace && \
    echo 'alias ll="ls -alF"' >> /home/vscode/.bashrc && \
    echo 'alias la="ls -A"' >> /home/vscode/.bashrc && \
    echo 'alias l="ls -CF"' >> /home/vscode/.bashrc && \
    echo 'export PS1="\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\w\[\033[00m\]\$ "' >> /home/vscode/.bashrc

# Health check for development
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1