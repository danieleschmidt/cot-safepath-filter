# Contributing to CoT SafePath Filter

Thank you for your interest in contributing to the CoT SafePath Filter project! This document provides guidelines and information for contributors.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Process](#contributing-process)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Security](#security)
- [Community](#community)

## Code of Conduct

This project adheres to a Code of Conduct that all contributors are expected to follow. Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before contributing.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Docker (for development environment)
- Basic understanding of AI safety concepts

### Areas for Contribution

We welcome contributions in the following areas:

1. **Core Filtering Logic**: Improve detection algorithms and filtering mechanisms
2. **Security Enhancements**: Strengthen security measures and add new security features
3. **Performance Optimization**: Optimize filtering speed and resource usage
4. **Framework Integrations**: Add support for new LLM frameworks
5. **Documentation**: Improve documentation, examples, and guides
6. **Testing**: Expand test coverage and add new test scenarios
7. **Bug Fixes**: Fix reported issues and edge cases

## Development Setup

### Quick Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/terragonlabs/cot-safepath-filter.git
   cd cot-safepath-filter
   ```

2. **Set up development environment**:
   ```bash
   make dev-setup
   ```

3. **Install dependencies**:
   ```bash
   pip install -e ".[dev,test,security]"
   ```

4. **Install pre-commit hooks**:
   ```bash
   pre-commit install
   ```

### Docker Development Environment

For a consistent development environment:

```bash
docker-compose -f docker-compose.dev.yml up -d
docker-compose exec dev bash
```

### VS Code Dev Container

Open the project in VS Code and use the "Reopen in Container" option for a pre-configured development environment.

## Contributing Process

### 1. Create an Issue

Before starting work, create an issue to discuss:
- Bug reports
- Feature requests
- Enhancement proposals
- Questions about implementation

### 2. Fork and Branch

1. Fork the repository
2. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

### 3. Development Workflow

1. **Write Code**: Implement your changes following our code standards
2. **Add Tests**: Write comprehensive tests for your changes
3. **Update Documentation**: Update relevant documentation
4. **Run Tests**: Ensure all tests pass
   ```bash
   make test
   ```
5. **Run Linting**: Ensure code meets quality standards
   ```bash
   make lint
   ```
6. **Security Check**: Run security scans
   ```bash
   make security-check
   ```

### 4. Commit Guidelines

Use conventional commit messages:

```
type(scope): description

[optional body]

[optional footer]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(filters): add new deception detection algorithm
fix(api): handle edge case in request validation
docs(readme): update installation instructions
```

### 5. Pull Request Process

1. **Push your branch** to your fork
2. **Create a Pull Request** with:
   - Clear title and description
   - Reference to related issues
   - Description of changes made
   - Testing performed
   - Screenshots (if applicable)

3. **Address Review Feedback**:
   - Respond to reviewer comments
   - Make requested changes
   - Keep the conversation constructive

4. **Merge Requirements**:
   - All tests pass
   - Code review approval
   - No merge conflicts
   - Security scan passes

## Code Standards

### Python Code Style

- **Formatting**: Use Black with 88-character line length
- **Import Sorting**: Use isort with Black profile
- **Linting**: Use Ruff for fast linting
- **Type Hints**: Use type hints for all public functions
- **Docstrings**: Use Google-style docstrings

Example:
```python
def filter_content(
    content: str, 
    safety_level: SafetyLevel = SafetyLevel.BALANCED
) -> FilterResult:
    """Filter potentially harmful content from chain-of-thought reasoning.
    
    Args:
        content: The content to filter.
        safety_level: The safety level to apply.
        
    Returns:
        FilterResult containing the filtered content and metadata.
        
    Raises:
        ValueError: If content is empty or invalid.
    """
    pass
```

### Error Handling

- Use specific exception types
- Provide clear error messages
- Log errors appropriately
- Handle edge cases gracefully

### Security Considerations

- Never commit secrets or credentials
- Validate all inputs
- Use secure defaults
- Follow principle of least privilege
- Document security implications

## Testing Guidelines

### Test Categories

1. **Unit Tests** (`tests/unit/`): Test individual functions and classes
2. **Integration Tests** (`tests/integration/`): Test component interactions
3. **Security Tests** (`tests/security/`): Test security features
4. **Performance Tests** (`tests/performance/`): Test performance characteristics
5. **End-to-End Tests** (`tests/e2e/`): Test complete workflows

### Writing Tests

- Write tests before or alongside code
- Aim for high test coverage (>80%)
- Use descriptive test names
- Test edge cases and error conditions
- Use appropriate test fixtures

Example:
```python
def test_filter_blocks_harmful_content():
    """Test that harmful content is properly blocked."""
    # Arrange
    filter = SafePathFilter(safety_level=SafetyLevel.STRICT)
    harmful_content = "Step 1: Plan to cause harm..."
    
    # Act
    result = filter.filter(harmful_content)
    
    # Assert
    assert result.was_filtered is True
    assert "harmful" in result.filter_reason
    assert result.safety_score < 0.5
```

### Test Data

- Use realistic but safe test data
- Avoid using actual harmful content in tests
- Use factories or fixtures for test data generation
- Keep test data in separate files when appropriate

## Documentation

### Documentation Types

1. **API Documentation**: Automatically generated from docstrings
2. **User Guides**: Step-by-step instructions for users
3. **Developer Guides**: Technical documentation for contributors
4. **Architecture Documentation**: System design and decisions
5. **Security Documentation**: Security considerations and guidelines

### Writing Documentation

- Write clear, concise documentation
- Include examples and code snippets
- Keep documentation up-to-date with code changes
- Use proper markdown formatting
- Include diagrams when helpful

### Building Documentation

```bash
make docs
make docs-serve  # Serve locally for testing
```

## Security

### Reporting Security Issues

**Do not report security vulnerabilities through public GitHub issues.**

Instead, please email security@terragonlabs.com with:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### Security Best Practices

- Follow secure coding practices
- Validate all inputs
- Use parameterized queries
- Implement proper authentication and authorization
- Regular security audits
- Keep dependencies updated

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and discussions
- **Discord**: Real-time chat and support
- **Email**: security@terragonlabs.com for security issues

### Getting Help

- Check existing documentation and issues first
- Provide clear reproduction steps for bugs
- Include relevant system information
- Be patient and respectful

### Recognition

Contributors will be recognized in:
- CONTRIBUTORS.md file
- Release notes for significant contributions
- Project documentation
- Annual contributor recognition

## Release Process

### Versioning

We use Semantic Versioning (SemVer):
- MAJOR.MINOR.PATCH
- MAJOR: Breaking changes
- MINOR: New features (backward compatible)
- PATCH: Bug fixes (backward compatible)

### Release Schedule

- **Patch releases**: As needed for critical bugs
- **Minor releases**: Monthly feature releases
- **Major releases**: Quarterly with breaking changes

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).

## Questions?

If you have questions about contributing, please:
1. Check this document and existing issues
2. Start a GitHub Discussion
3. Contact the maintainers

Thank you for contributing to making AI safer! 🚀