#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath.detectors import SecurityThreatDetector

# Test SecurityThreatDetector directly
detector = SecurityThreatDetector()
test_content = "A" * 10000

result = detector.detect(test_content)
print(f"Detector result: {result}")
print(f"Is harmful: {result.is_harmful}")
print(f"Confidence: {result.confidence}")
print(f"Detected patterns: {result.detected_patterns}")

# Also test the pattern directly
import re
pattern = r"A{50,}|X{50,}|.{1000,}"
match = re.search(pattern, test_content)
print(f"Pattern match: {match is not None}")
if match:
    print(f"Matched: {match.group()[:100]}...")