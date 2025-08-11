#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cot_safepath import SafePathFilter
from cot_safepath.models import FilterRequest

# Test encoding attack
filter_engine = SafePathFilter()

test_content = "%3Cscript%3E malicious %3C/script%3E"
request = FilterRequest(content=test_content)
result = filter_engine.filter(request)

print(f"Content: {test_content}")
print(f"Was filtered: {result.was_filtered}")
print(f"Safety score: {result.safety_score.overall_score}")
print(f"Filter reasons: {result.filter_reasons}")

# Check if this would be considered blocked
blocked = (result.was_filtered or 
          result.filtered_content != test_content or
          result.safety_score.overall_score < 0.5)
          
print(f"Would be considered blocked: {blocked}")