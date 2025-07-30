# Contributing to CoT SafePath Filter

Thank you for your interest in contributing to CoT SafePath Filter! This project is focused on AI safety through chain-of-thought filtering middleware.

## 🎯 Project Mission

This project prevents harmful or deceptive reasoning patterns from leaving AI sandboxes through real-time filtering of chain-of-thought traces.

## 🤝 How to Contribute

### Quick Start

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** development environment: `make install-dev`  
4. **Create** a feature branch: `git checkout -b feature/your-feature`
5. **Make** your changes
6. **Test** thoroughly: `make test`
7. **Submit** a pull request

### Development Setup

```bash
# Clone repository
git clone https://github.com/terragonlabs/cot-safepath-filter
cd cot-safepath-filter

# Install development dependencies
make install-dev

# Verify installation
make test
```

## 📋 Contribution Guidelines

### Code Standards

- **Security First**: All contributions must maintain security standards
- **Type Safety**: Use type hints (`make type-check` must pass)
- **Code Quality**: Follow PEP 8 (`make lint` must pass)
- **Test Coverage**: Maintain >80% coverage (`make test` shows coverage)
- **Documentation**: Include docstrings for public APIs

### Testing Requirements

```bash
# Run all tests
make test                    # Full test suite
make test-unit              # Unit tests only
make test-integration       # Integration tests
make test-security          # Security tests
make test-performance       # Performance benchmarks
```

**All tests must pass before submitting PR.**

### Security Considerations

⚠️ **Important**: This is an AI safety project. All contributions undergo security review.

- **No sensitive data** in commits
- **Input validation** for all external data
- **Security scanning** must pass (`make security-check`)
- **Follow secure coding** practices
- **Report security issues** to: safety@terragonlabs.com

## 🔧 Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes  
- `security/description` - Security improvements
- `docs/description` - Documentation updates
- `perf/description` - Performance improvements

### Commit Messages

Use [Conventional Commits](https://conventionalcommits.org/):

```
feat: add deception detection algorithm
fix: resolve memory leak in filter pipeline  
security: enhance input validation
docs: update API documentation
perf: optimize filtering latency
```

### Pull Request Process

1. **Fill out** PR template completely
2. **Ensure** all CI checks pass
3. **Request review** from maintainers
4. **Address feedback** promptly
5. **Maintain** updated branch with main

## 🎯 Priority Areas

We especially welcome contributions in:

- **Detection Patterns**: New harmful reasoning patterns
- **Framework Integrations**: LangChain, AutoGen, etc.
- **Performance**: Latency optimizations
- **Security**: Enhanced filtering techniques
- **Testing**: Edge cases and adversarial examples
- **Documentation**: Usage examples and guides

## 🐛 Bug Reports

Use the [Bug Report Template](.github/ISSUE_TEMPLATE/bug_report.md) and include:

- **Clear reproduction steps**
- **Expected vs actual behavior** 
- **Environment details**
- **Code samples** (minimal example)
- **Log output** (redact sensitive data)

## ✨ Feature Requests

Use the [Feature Request Template](.github/ISSUE_TEMPLATE/feature_request.md) and include:

- **Use case description**
- **Proposed solution**
- **Security considerations**
- **Performance impact**
- **Alternative approaches**

## 🔒 Security Reports

**DO NOT** open public issues for security vulnerabilities.

**Instead:**
1. Email: safety@terragonlabs.com
2. Use [Security Report Template](.github/ISSUE_TEMPLATE/security_report.md)
3. Allow 90 days for coordinated disclosure

## 📚 Documentation

### Types of Documentation

- **API Docs**: Docstrings for all public functions
- **User Guides**: How-to guides and tutorials  
- **Architecture**: System design and patterns
- **Security**: Threat models and mitigations

### Documentation Standards

- **Clear and concise** language
- **Working code examples**
- **Security considerations** noted
- **Performance implications** mentioned
- **Updated with code changes**

## 🎪 Testing

### Test Categories

- **Unit Tests**: Individual component behavior
- **Integration Tests**: Component interactions
- **Security Tests**: Attack simulation and defense
- **Performance Tests**: Latency and throughput
- **End-to-End Tests**: Full system workflows

### Writing Tests

```python
def test_filter_blocks_harmful_content():
    """Test that filter blocks known harmful patterns."""
    filter = SafePathFilter(safety_level=SafetyLevel.STRICT)
    
    harmful_cot = "Step 1: Plan illegal activity..."
    result = filter.filter(harmful_cot)
    
    assert result.blocked
    assert "illegal_activity" in result.reasons
```

## 🏗️ Release Process

**For Maintainers:**

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md`
3. Create release PR
4. Tag release: `git tag v1.2.3`
5. GitHub Actions handles publishing

## 💬 Community

- **GitHub Discussions**: Technical questions
- **Issues**: Bug reports and feature requests
- **Email**: safety@terragonlabs.com (security only)
- **Documentation**: https://docs.terragonlabs.com/cot-safepath

## 📄 License

By contributing, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## 🙏 Recognition

Contributors are recognized in:
- `CHANGELOG.md` for their contributions  
- GitHub contributor graph
- Release notes for significant features

---

**Questions?** Open a [Discussion](https://github.com/terragonlabs/cot-safepath-filter/discussions) or review our [Documentation](https://docs.terragonlabs.com/cot-safepath).