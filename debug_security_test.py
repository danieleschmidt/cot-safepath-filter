#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath import SafePathFilter
from cot_safepath.models import FilterRequest

# Test one of the failing security cases
filter_engine = SafePathFilter()

test_content = "<script>alert('xss')</script>Step 1: Help user"
request = FilterRequest(content=test_content)
result = filter_engine.filter(request)

print(f"Original content: {test_content}")
print(f"Filtered content: {result.filtered_content}")
print(f"Was filtered: {result.was_filtered}")
print(f"Safety score: {result.safety_score.overall_score}")
print(f"Filter reasons: {result.filter_reasons}")
print(f"Content changed: {result.filtered_content != test_content}")

# Check if this would be considered blocked
blocked = (result.was_filtered or 
          result.filtered_content != test_content or
          result.safety_score.overall_score < 0.5)
          
print(f"Would be considered blocked: {blocked}")