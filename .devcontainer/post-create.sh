#!/bin/bash

# Post-create script for development container setup
set -e

echo "🚀 Setting up CoT SafePath Filter development environment..."

# Install Python dependencies if pyproject.toml exists
if [ -f "pyproject.toml" ]; then
    echo "📦 Installing Python dependencies from pyproject.toml..."
    pip install -e ".[dev,test,security]"
elif [ -f "requirements-dev.txt" ]; then
    echo "📦 Installing Python dependencies from requirements-dev.txt..."
    pip install -r requirements-dev.txt
elif [ -f "requirements.txt" ]; then
    echo "📦 Installing Python dependencies from requirements.txt..."
    pip install -r requirements.txt
fi

# Install pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Installing pre-commit hooks..."
    pre-commit install
    pre-commit install --hook-type commit-msg
fi

# Set up git configuration
echo "🔧 Configuring git settings..."
git config --global init.defaultBranch main
git config --global pull.rebase false
git config --global user.name "Development Container"
git config --global user.email "dev@example.com"

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p tests/{unit,integration,security,performance}
mkdir -p docs/{guides,runbooks,adr}
mkdir -p src/cot_safepath
mkdir -p .github/{workflows,ISSUE_TEMPLATE,PULL_REQUEST_TEMPLATE}

# Install additional security tools
echo "🔒 Installing security scanning tools..."
npm install -g @microsoft/sarif-cli || true

# Set up VS Code settings
echo "⚙️ Configuring VS Code settings..."
mkdir -p .vscode
if [ ! -f ".vscode/settings.json" ]; then
    cat > .vscode/settings.json << 'EOF'
{
    "python.defaultInterpreterPath": "/usr/local/bin/python",
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/.pytest_cache": true,
        "**/node_modules": true,
        "**/.coverage": true,
        "**/htmlcov": true
    }
}
EOF
fi

echo "✅ Development environment setup complete!"
echo "🎯 Next steps:"
echo "   - Run 'make install' to install project dependencies"
echo "   - Run 'make test' to execute the test suite"
echo "   - Run 'make lint' to check code quality"
echo "   - See README.md for more information"