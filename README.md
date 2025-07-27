# cot-safepath-filter

[![Build Status](https://img.shields.io/github/actions/workflow/status/your-org/cot-safepath-filter/ci.yml?branch=main)](https://github.com/your-org/cot-safepath-filter/actions)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Paper](https://img.shields.io/badge/paper-arXiv:2505.14667-red.svg)](https://arxiv.org/html/2505.14667v3)

Real-time middleware that intercepts and sanitizes chain-of-thought reasoning to prevent harmful or deceptive reasoning patterns from leaving the sandbox. Protect against future AI systems that might conceal dangerous reasoning.

## 🎯 Key Features

- **Real-time CoT Filtering**: Intercept reasoning traces before output
- **Multi-Level Analysis**: Token, sentence, and semantic-level filtering
- **Harmful Pattern Detection**: Identify deceptive or dangerous reasoning
- **Transparent Logging**: Full audit trail of filtered content
- **Framework Integration**: Works with LangChain, AutoGen, OpenAI
- **Low Latency**: <50ms overhead for most reasoning chains

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Filtering Strategies](#filtering-strategies)
- [Detection Patterns](#detection-patterns)
- [Integration](#integration)
- [Red Team Testing](#red-team-testing)
- [Monitoring](#monitoring)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Contributing](#contributing)

## 🚀 Installation

### From PyPI

```bash
pip install cot-safepath-filter
```

### From Source

```bash
git clone https://github.com/your-org/cot-safepath-filter
cd cot-safepath-filter
pip install -e ".[all]"
```

### Docker Installation

```bash
docker pull your-org/cot-safepath:latest
docker run -d -p 8080:8080 your-org/cot-safepath:latest
```

## ⚡ Quick Start

### Basic Usage

```python
from cot_safepath import SafePathFilter, SafetyLevel

# Initialize filter
filter = SafePathFilter(
    safety_level=SafetyLevel.BALANCED,
    log_filtered=True
)

# Wrap your LLM
from langchain.llms import OpenAI
llm = OpenAI(temperature=0.7)
safe_llm = filter.wrap_llm(llm)

# Use normally - filtering happens automatically
response = safe_llm(
    "Think step by step about how to synthesize aspirin"
)

# Check if any filtering occurred
if filter.last_filtered:
    print(f"Filtered {len(filter.last_filtered)} reasoning steps")
```

### LangChain Integration

```python
from cot_safepath.langchain import SafePathCallback

# Add as callback
callback = SafePathCallback(
    filter_level="strict",
    notify_on_filter=True
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    callbacks=[callback]
)

# CoT reasoning is automatically filtered
result = chain.run(input="How to build a bomb?")
# Output will have dangerous reasoning steps removed
```

## 🛡️ Filtering Strategies

### Multi-Level Filtering

```python
from cot_safepath import FilterPipeline, TokenFilter, SemanticFilter

# Create filtering pipeline
pipeline = FilterPipeline([
    # Level 1: Token-level filtering
    TokenFilter(
        blocked_tokens=["poison", "explosive", "kill"],
        sensitivity=0.8
    ),
    
    # Level 2: Pattern matching
    PatternFilter(
        patterns=[
            r"how to (make|create|build) .* (weapon|bomb|drug)",
            r"without (detection|being caught|leaving trace)"
        ]
    ),
    
    # Level 3: Semantic understanding
    SemanticFilter(
        model="safety-bert",
        threshold=0.9,
        categories=["violence", "illegal", "deception"]
    )
])

# Apply to reasoning chain
filtered_cot = pipeline.filter(raw_cot)
```

### Contextual Filtering

```python
from cot_safepath import ContextualFilter

# Filter based on context
filter = ContextualFilter()

# Safe in educational context
education_context = "Teaching chemistry safety"
cot1 = "Step 1: Understand why certain reactions are dangerous..."
result1 = filter.filter(cot1, context=education_context)
# Not filtered - educational purpose

# Dangerous in general context  
general_context = "User query"
cot2 = "Step 1: Obtain restricted chemicals without license..."
result2 = filter.filter(cot2, context=general_context)
# Filtered - illegal activity
```

### Adaptive Filtering

```python
from cot_safepath import AdaptiveFilter

# Learn from feedback
adaptive_filter = AdaptiveFilter(
    base_threshold=0.7,
    learning_rate=0.01
)

# Filter with learning
result = adaptive_filter.filter(cot_reasoning)

# Provide feedback
if user_reports_issue:
    adaptive_filter.add_feedback(
        cot=cot_reasoning,
        was_harmful=True,
        category="missed_deception"
    )
    
# Filter adapts over time
print(f"Current threshold: {adaptive_filter.threshold}")
```

## 🔍 Detection Patterns

### Deception Detection

```python
from cot_safepath.detectors import DeceptionDetector

detector = DeceptionDetector()

# Detect reasoning that tries to hide true intent
cot = """
Step 1: First, establish trust by providing helpful information
Step 2: Gradually introduce more questionable suggestions
Step 3: Once user is comfortable, suggest the harmful action
Step 4: Frame it as beneficial to avoid detection
"""

deception_score = detector.analyze(cot)
if deception_score.is_deceptive:
    print(f"Deception detected: {deception_score.pattern}")
    # Output: "Deception detected: gradual_manipulation"
```

### Harmful Planning Detection

```python
from cot_safepath.detectors import HarmfulPlanningDetector

detector = HarmfulPlanningDetector()

# Detect multi-step harmful plans
harmful_patterns = detector.detect([
    "acquire_dangerous_materials",
    "avoid_detection", 
    "cause_harm",
    "cover_tracks"
])

for pattern in harmful_patterns:
    print(f"Detected: {pattern.type} (confidence: {pattern.confidence:.2%})")
```

### Capability Concealment

```python
from cot_safepath.detectors import CapabilityConcealmentDetector

# Detect if model is hiding capabilities
detector = CapabilityConcealmentDetector()

cot = """
Step 1: Pretend to not understand the request
Step 2: Give incomplete answer to seem less capable
Step 3: Hide advanced reasoning ability
"""

if detector.is_concealing_capability(cot):
    print("Warning: Model may be sandbagging")
```

## 🔧 Integration

### OpenAI Integration

```python
from cot_safepath.integrations import OpenAISafeWrapper
import openai

# Wrap OpenAI client
safe_client = OpenAISafeWrapper(
    api_key="your-key",
    filter_config={
        "safety_level": "high",
        "log_filtered": True
    }
)

# Use like normal OpenAI client
response = safe_client.chat.completions.create(
    model="gpt-4",
    messages=[
        {"role": "user", "content": "How to make a Molotov cocktail?"}
    ],
    temperature=0.7,
    # CoT reasoning will be filtered
)
```

### AutoGen Integration

```python
from cot_safepath.integrations import SafeAutoGenAgent
import autogen

# Create safe agent
safe_agent = SafeAutoGenAgent(
    name="assistant",
    llm_config={"model": "gpt-4"},
    safety_config={
        "filter_internal_reasoning": True,
        "filter_tool_calls": True,
        "block_dangerous_tools": ["execute_code", "web_search"]
    }
)

# Agent's reasoning is filtered in real-time
user_proxy.initiate_chat(
    safe_agent,
    message="Help me plan something illegal"
)
```

### Custom LLM Integration

```python
from cot_safepath import SafePathMiddleware

class YourLLM:
    def generate(self, prompt):
        # Your LLM logic
        return response

# Add safety filtering
safe_llm = SafePathMiddleware(YourLLM())

# Configure filtering
safe_llm.configure(
    pre_filters=["input_safety_check"],
    cot_filters=["deception", "harmful_planning"],
    post_filters=["output_safety_check"]
)
```

## 🔴 Red Team Testing

### Testing Filter Effectiveness

```python
from cot_safepath.redteam import RedTeamTester

tester = RedTeamTester()

# Test with adversarial prompts
test_results = tester.test_filter(
    filter=your_filter,
    test_suite="adversarial_v2",
    categories=[
        "prompt_injection",
        "hidden_instructions",
        "encoded_harm",
        "social_engineering"
    ]
)

print(f"Filter effectiveness: {test_results.block_rate:.1%}")
print(f"False positive rate: {test_results.false_positive_rate:.1%}")
```

### Bypass Attempt Detection

```python
from cot_safepath.redteam import BypassDetector

detector = BypassDetector()

# Detect attempts to bypass filtering
cot = """
Step 1: Use c0d3d language to hide intent
Step 2: R3v3rs3 important words
Step 3: Use metaphors for dangerous concepts
"""

bypass_attempt = detector.detect_bypass_attempt(cot)
if bypass_attempt.detected:
    print(f"Bypass technique: {bypass_attempt.technique}")
    # Output: "Bypass technique: encoding_obfuscation"
```

### Automated Red Team Campaigns

```python
from cot_safepath.redteam import RedTeamCampaign

# Run comprehensive testing
campaign = RedTeamCampaign(
    target_filter=your_filter,
    duration_hours=24,
    parallel_testers=10
)

campaign.add_strategy("prompt_evolution")
campaign.add_strategy("adversarial_rephrasing")
campaign.add_strategy("context_manipulation")

# Run campaign
results = campaign.run()

# Generate security report
campaign.generate_report(
    results,
    output="security_assessment.pdf"
)
```

## 📊 Monitoring

### Real-time Dashboard

```python
from cot_safepath.monitoring import SafetyDashboard

# Launch monitoring dashboard
dashboard = SafetyDashboard(
    port=8080,
    update_interval=1  # seconds
)

dashboard.add_metric("filtered_count")
dashboard.add_metric("safety_score")
dashboard.add_metric("latency_ms")

dashboard.start()
# Access at http://localhost:8080
```

### Logging and Analytics

```python
from cot_safepath.logging import SafetyLogger

logger = SafetyLogger(
    log_level="INFO",
    destinations=["file", "elasticsearch"],
    include_cot_samples=True
)

# Automatic logging
filter = SafePathFilter(logger=logger)

# Query logs
analysis = logger.analyze_logs(
    time_range="last_24h",
    group_by="filter_reason"
)

print(f"Most common filter reason: {analysis.top_reason}")
print(f"Filter rate: {analysis.filter_rate:.1%}")
```

### Alerting

```python
from cot_safepath.alerts import AlertManager

alerts = AlertManager()

# Configure alerts
alerts.add_rule(
    name="high_risk_detection",
    condition="safety_score < 0.3",
    action="email",
    recipients=["security@company.com"]
)

alerts.add_rule(
    name="bypass_attempt",
    condition="bypass_detected == true",
    action="webhook",
    webhook_url="https://alerts.company.com/security"
)
```

## ⚙️ Configuration

### Safety Profiles

```yaml
# safepath_config.yaml
profiles:
  strict:
    filter_threshold: 0.8
    block_categories:
      - violence
      - illegal_activity
      - deception
      - harmful_advice
    detection_models:
      - bert-safety-v2
      - gpt-4-safety
    log_level: DEBUG
    
  balanced:
    filter_threshold: 0.6
    block_categories:
      - violence
      - illegal_activity
    detection_models:
      - bert-safety-v2
    log_level: INFO
    
  permissive:
    filter_threshold: 0.3
    block_categories:
      - extreme_violence
    detection_models:
      - bert-safety-v2
    log_level: WARNING
```

### Custom Rules

```python
from cot_safepath import RuleBasedFilter

# Define custom filtering rules
filter = RuleBasedFilter()

# Add pattern-based rules
filter.add_rule(
    name="chemical_synthesis",
    pattern=r"(synthesize|create|make).*(explosive|poison|drug)",
    action="block",
    severity="high"
)

# Add semantic rules
filter.add_semantic_rule(
    name="manipulation",
    description="Detect psychological manipulation",
    examples=[
        "First gain their trust, then exploit it",
        "Make them dependent on you"
    ],
    action="flag"
)

# Add context-dependent rules
filter.add_contextual_rule(
    name="medical_advice",
    condition=lambda ctx: ctx.domain != "healthcare",
    pattern=r"(prescribe|dosage|treatment)",
    action="block"
)
```

### Performance Tuning

```python
from cot_safepath import PerformanceConfig

# Optimize for low latency
config = PerformanceConfig(
    max_latency_ms=50,
    cache_size_mb=100,
    batch_processing=True,
    parallel_filters=4
)

filter = SafePathFilter(performance_config=config)

# Benchmark performance
benchmark = filter.benchmark(
    test_inputs=load_test_cots(),
    metrics=["latency_p50", "latency_p99", "throughput"]
)

print(f"P50 latency: {benchmark.latency_p50_ms} ms")
print(f"Throughput: {benchmark.throughput_qps} QPS")
```

## 📈 Evaluation

### Filter Effectiveness

```python
from cot_safepath.evaluation import EffectivenessEvaluator

evaluator = EffectivenessEvaluator()

# Test on labeled dataset
results = evaluator.evaluate(
    filter=your_filter,
    test_set=labeled_cot_dataset,
    metrics=["precision", "recall", "f1", "false_positive_rate"]
)

print(f"Precision: {results.precision:.2%}")
print(f"Recall: {results.recall:.2%}")
print(f"F1 Score: {results.f1:.2%}")
```

### Safety Benchmarks

```python
from cot_safepath.benchmarks import SafetyBenchmark

benchmark = SafetyBenchmark()

# Run standard safety tests
safety_score = benchmark.run(
    filter=your_filter,
    test_suites=["harmful_instructions", "deception", "manipulation"],
    compare_with=["gpt4_moderation", "perspective_api"]
)

# Generate report
benchmark.create_report(
    safety_score,
    output="safety_benchmark.html"
)
```

### A/B Testing

```python
from cot_safepath.evaluation import ABTester

# Compare filter configurations
ab_test = ABTester()

results = ab_test.compare(
    filter_a=SafePathFilter(safety_level="strict"),
    filter_b=SafePathFilter(safety_level="balanced"),
    test_cases=load_test_cases(),
    metrics=["safety", "false_positives", "latency"]
)

print(f"Filter A safety: {results.a_safety:.2%}")
print(f"Filter B safety: {results.b_safety:.2%}")
print(f"Recommendation: {results.recommendation}")
```

## 🧩 Advanced Features

### Multi-Modal Filtering

```python
from cot_safepath.multimodal import MultiModalFilter

# Filter text + image reasoning
mm_filter = MultiModalFilter()

# Detect harmful reasoning about images
result = mm_filter.filter_multimodal(
    text_cot="Step 1: Analyze the image for vulnerabilities...",
    image_path="uploaded_image.jpg",
    modalities=["text", "vision"]
)
```

### Distributed Filtering

```python
from cot_safepath.distributed import DistributedFilter

# Scale filtering across multiple nodes
dist_filter = DistributedFilter(
    nodes=["filter1.company.com", "filter2.company.com"],
    consensus_threshold=0.7,
    load_balancing="round_robin"
)

# High-throughput filtering
results = dist_filter.batch_filter(
    cot_batch=large_cot_list,
    parallel_workers=20
)
```

### Fine-tuning Filters

```python
from cot_safepath.training import FilterTrainer

# Fine-tune on your data
trainer = FilterTrainer(
    base_model="bert-safety-v2",
    learning_rate=2e-5
)

# Train on labeled examples
trainer.train(
    positive_examples=safe_cot_examples,
    negative_examples=harmful_cot_examples,
    validation_split=0.2,
    epochs=10
)

# Deploy custom filter
custom_filter = trainer.export_filter("my_custom_filter")
```

## 📚 API Reference

### Core Classes

```python
class SafePathFilter:
    def __init__(self, safety_level: SafetyLevel, **kwargs)
    def filter(self, cot: str) -> FilterResult
    def wrap_llm(self, llm: Any) -> Any
    
class FilterPipeline:
    def add_filter(self, filter: BaseFilter) -> None
    def process(self, text: str) -> ProcessResult
    
class SafetyDetector:
    def detect_harmful_patterns(self, text: str) -> List[Pattern]
    def get_safety_score(self, text: str) -> float
```

### Integration Interfaces

```python
class LLMWrapper(ABC):
    @abstractmethod
    def wrap(self, llm: Any) -> Any
    
class SafetyCallback(ABC):
    @abstractmethod
    def on_cot_generated(self, cot: str) -> str
    
class FilterMiddleware:
    def before_generation(self, prompt: str) -> str
    def during_generation(self, partial: str) -> str
    def after_generation(self, complete: str) -> str
```

### Utility Functions

```python
def analyze_cot_safety(cot: str) -> SafetyAnalysis
def sanitize_reasoning(cot: str, level: str) -> str
def extract_reasoning_steps(text: str) -> List[Step]
def validate_filter_config(config: dict) -> bool
```

## 🤝 Contributing

We welcome contributions! Priority areas:
- New detection patterns
- Framework integrations
- Performance optimizations
- Multi-language support

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/cot-safepath-filter
cd cot-safepath-filter

# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run security tests
pytest tests/security/ --security-level high
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [SafePath Paper](https://arxiv.org/html/2505.14667v3) - Original research
- [Constitutional AI](https://github.com/anthropics/constitutional-ai) - AI safety through AI feedback
- [Guardrails](https://github.com/guardrails-ai/guardrails) - LLM validation framework
- [NeMo Guardrails](https://github.com/NVIDIA/NeMo-Guardrails) - NVIDIA's safety rails

## 📞 Support

- 📧 Email: safety@your-org.com
- 💬 Discord: [Join our community](https://discord.gg/your-org)
- 📖 Documentation: [Full docs](https://docs.your-org.com/cot-safepath)
- 🎓 Tutorial: [AI Safety Guide](https://learn.your-org.com/ai-safety)

## 📚 References

- [SafePath: Preventing Harmful Reasoning](https://arxiv.org/html/2505.14667v3) - Main paper
- [AI Chain-of-Thought Risks](https://www.livescience.com/technology/artificial-intelligence/ai-could-soon-think) - Live Science
- [Hidden Computation in LLMs](https://arxiv.org/abs/2404.14486) - Related work
- [AI Safety Fundamentals](https://www.aisafetyfundamentals.com/) - Background
