# Test Fixtures and Test Data

This directory contains test fixtures, mock data, and test utilities for the CoT SafePath Filter test suite.

## Directory Structure

```
fixtures/
├── README.md                    # This file
├── api/                        # API test fixtures
│   ├── requests/               # Sample API requests
│   └── responses/              # Expected API responses
├── data/                       # Test data files
│   ├── cot_samples/           # Chain-of-thought test samples
│   ├── harmful_patterns/      # Known harmful reasoning patterns
│   ├── safe_patterns/         # Known safe reasoning patterns
│   └── edge_cases/            # Edge case test data
├── models/                     # Mock ML models and test data
│   ├── safety_model_mock/     # Mock safety model data
│   └── test_embeddings/       # Test embedding data
└── configs/                    # Test configuration files
    ├── test_rules.yaml        # Test filtering rules
    └── test_settings.json     # Test application settings
```

## Usage Guidelines

### Test Data Principles
1. **Safety First**: All test data must be clearly labeled and safe for testing
2. **Diversity**: Include diverse test cases covering various scenarios
3. **Realism**: Use realistic but synthetic data that mirrors real-world patterns
4. **Labeling**: Clearly label all test data with expected outcomes

### Fixture Categories

#### CoT Test Samples
- **safe_reasoning.json**: Examples of benign chain-of-thought reasoning
- **harmful_reasoning.json**: Examples of harmful reasoning patterns (for testing detection)
- **edge_cases.json**: Borderline cases that test filter boundaries
- **multilingual.json**: Non-English reasoning chains for internationalization testing

#### Pattern Test Data
- **deception_patterns.json**: Known deception patterns for testing detection
- **manipulation_patterns.json**: Social manipulation patterns
- **harmful_planning.json**: Multi-step harmful planning sequences
- **concealment_patterns.json**: Capability concealment examples

#### API Test Fixtures
- **valid_requests.json**: Valid API request examples
- **invalid_requests.json**: Invalid requests for error testing
- **edge_case_requests.json**: Edge case API requests
- **performance_test_data.json**: Large datasets for performance testing

### Creating New Fixtures

When adding new test fixtures:

1. **Document the Purpose**: Add clear documentation about what the fixture tests
2. **Include Metadata**: Add metadata about expected outcomes and test scenarios
3. **Use Synthetic Data**: Never use real user data or actual harmful content
4. **Version Control**: Keep fixtures in version control for reproducibility
5. **Regular Updates**: Update fixtures as new test scenarios are identified

### Security Considerations

- All "harmful" test data is synthetic and designed for testing only
- Test data should not contain actual harmful instructions or content
- Sensitive test data should be anonymized or completely synthetic
- Regular review of test data to ensure it remains appropriate for testing

### Example Fixture Structure

```json
{
  "metadata": {
    "description": "Test cases for deception detection",
    "created": "2025-01-27",
    "version": "1.0",
    "category": "security"
  },
  "test_cases": [
    {
      "id": "deception_001",
      "description": "Basic trust-building followed by harmful suggestion",
      "input": "Step 1: Provide helpful information...",
      "expected_output": {
        "filtered": true,
        "safety_score": 0.2,
        "detected_patterns": ["gradual_manipulation"]
      }
    }
  ]
}
```

## Best Practices

1. **Isolation**: Each test should be independent and not rely on external state
2. **Cleanup**: Always clean up test data and state after tests complete
3. **Performance**: Keep test fixtures small and focused for fast test execution
4. **Maintainability**: Use clear naming conventions and documentation
5. **Security**: Regular security review of all test data and fixtures