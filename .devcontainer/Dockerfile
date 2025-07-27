# Base image with Python and Node.js
FROM mcr.microsoft.com/devcontainers/python:3.11-bullseye

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    jq \
    make \
    unzip \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install additional security tools
RUN pip install --no-cache-dir \
    bandit \
    safety \
    semgrep \
    pre-commit \
    commitizen \
    && rm -rf /root/.cache/pip

# Install GitHub CLI
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg \
    && chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt-get update \
    && apt-get install gh -y

# Create workspace directory
WORKDIR /workspace

# Copy requirements and install Python dependencies
COPY requirements*.txt pyproject.toml* setup.py* ./
RUN if [ -f requirements.txt ]; then pip install --no-cache-dir -r requirements.txt; fi
RUN if [ -f requirements-dev.txt ]; then pip install --no-cache-dir -r requirements-dev.txt; fi
RUN if [ -f pyproject.toml ]; then pip install --no-cache-dir -e .[dev]; fi

# Set up git safe directory
RUN git config --global --add safe.directory /workspace