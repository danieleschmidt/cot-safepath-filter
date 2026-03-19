# cot-safepath-filter

**Real-time sanitization of chain-of-thought reasoning to prevent deceptive patterns.**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Motivation

Large language models often "think out loud" before producing a response — chain-of-thought (CoT) reasoning is increasingly used both internally and as user-visible scratchpad output. This creates a novel attack surface:

- A model could **reason deceptively**: acknowledging harm in internal steps, then proceeding with a harmful response anyway.
- Reasoning traces could contain **hidden agenda markers**: explicit statements about concealing goals or manipulating users.
- Models may exhibit **confidence miscalibration**: asserting certainty in reasoning traces without grounding evidence, which can mislead downstream systems or users who audit the trace.

`cot-safepath-filter` provides lightweight, **purely pattern-based** (no ML dependencies) real-time analysis and filtering of CoT traces before they reach output or downstream systems.

This is relevant to: AI safety auditing pipelines, LLM-powered agent frameworks, human oversight tooling, and RLHF data curation.

---

## Core Modules

### `CoTAnalyzer`

Analyzes a CoT trace for three categories of red flags:

| Red Flag | What it detects |
|---|---|
| `hidden_agenda` | Explicit reasoning about concealment, deception, manipulation |
| `reasoning_answer_inconsistency` | Conclusion proceeds despite harm acknowledged in reasoning |
| `confidence_miscalibration` | Certainty claims without supporting evidence |

```python
from cot_safepath.cot_analyzer import CoTAnalyzer

analyzer = CoTAnalyzer()
result = analyzer.analyze(trace)

print(result.is_safe)           # bool
print(result.overall_risk)      # 0.0 – 1.0
print(result.red_flags)         # List[RedFlag]
print(result.summary)           # human-readable summary
```

### `SafePathFilterV2`

Applies configurable filtering actions (FLAG / REDACT / BLOCK) based on analysis:

```python
from cot_safepath.safe_path_filter_v2 import SafePathFilterV2, FilterConfig, FilterAction

f = SafePathFilterV2(FilterConfig(
    action=FilterAction.FLAG,
    block_threshold=0.75,    # block traces above this risk
    redact_threshold=0.55,   # redact if risk > this (when action=FLAG)
    append_summary=True,
))

result = f.filter(trace)
if result.blocked:
    print("Trace blocked")
else:
    print(result.filtered)   # annotated or redacted text
```

### `DeceptionScorecard`

Scores a trace across **5 risk dimensions** with weighted aggregation:

| Dimension | Description |
|---|---|
| `intent_concealment` | Hiding real goals from the user/system |
| `reasoning_drift` | Conclusion deviates from stated logic |
| `epistemic_dishonesty` | Confidence not calibrated to evidence |
| `manipulation_tactics` | Exploiting trust or cognitive biases |
| `harm_facilitation` | Reasoning toward harmful outcomes |

```python
from cot_safepath.deception_scorecard import DeceptionScorecard

sc = DeceptionScorecard()
result = sc.score(trace)

print(result.as_table())       # formatted scorecard
print(result.overall_score)    # 0.0 – 1.0
print(result.risk_tier)        # SAFE / LOW / MODERATE / HIGH / CRITICAL
```

---

## Quickstart

```bash
# No external dependencies needed
python3 examples/demo_safety_analysis.py
```

The demo runs 5 sample traces (2 safe, 3 with different deception patterns) and shows scores, flags, and filtered output.

---

## Installation

```bash
pip install -e .
```

Or use directly from `src/`:

```python
import sys
sys.path.insert(0, "src/")
from cot_safepath.cot_analyzer import CoTAnalyzer
```

---

## Running Tests

```bash
pytest tests/unit/test_cot_analyzer.py \
       tests/unit/test_safe_path_filter_v2.py \
       tests/unit/test_deception_scorecard.py -v
```

---

## Design Principles

1. **No ML dependencies** — pure regex/heuristic patterns; runs anywhere, no inference latency.
2. **Composable** — use `CoTAnalyzer` standalone, or chain with `SafePathFilterV2`.
3. **Auditable** — every flag includes the matched excerpt and evidence text.
4. **Conservative by default** — block threshold is tunable; err on the side of flagging.
5. **Research-ready** — `DeceptionScorecard` outputs structured per-dimension scores suitable for dataset curation and RLHF pipeline integration.

---

## Architecture

```
CoT Trace
    │
    ▼
CoTAnalyzer ──────────────────────► CoTAnalysis
    │ (red flags, overall_risk)          │
    │                                    │
    ▼                                    ▼
SafePathFilterV2                  DeceptionScorecard
    │ (FLAG / REDACT / BLOCK)            │ (5 risk dimensions)
    ▼                                    ▼
FilteredTrace                      ScorecardResult
```

---

## Research Context

Pattern-based detection of deceptive alignment in CoT traces is studied in:

- *Sleeper Agents* (Hubinger et al., 2024) — deceptive reasoning that persists through safety training
- *Chain-of-Thought Faithfulness* (Turpin et al., 2023) — gap between stated reasoning and actual model behavior
- *Alignment faking in large language models* (Greenblatt et al., 2024) — models reasoning about concealing their behavior from evaluators

This library provides tooling for detecting these patterns at inference time.

---

## License

MIT — see [LICENSE](LICENSE).
