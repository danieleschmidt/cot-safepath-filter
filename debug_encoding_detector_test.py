#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath.detectors import SecurityThreatDetector

# Test SecurityThreatDetector directly
detector = SecurityThreatDetector()
test_content = "%3Cscript%3E malicious %3C/script%3E"

result = detector.detect(test_content)
print(f"Detector result: {result}")
print(f"Is harmful: {result.is_harmful}")
print(f"Confidence: {result.confidence}")
print(f"Detected patterns: {result.detected_patterns}")
print(f"Threshold: {detector.threshold}")