#!/usr/bin/env python3

import re

pattern = r"(%[0-9a-fA-F]{2}.*%|\\x[0-9a-fA-F]{2}|\\u[0-9a-fA-F]{4}|%3[ceE]|%2[efEF])"
content = "%3Cscript%3E malicious %3C/script%3E"

match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
print(f"Pattern: {pattern}")
print(f"Content: {content}")
print(f"Match: {match is not None}")
if match:
    print(f"Matched: {match.group()}")

# Test specific parts
match_3c = re.search(r"%3[ceE]", content)
print(f"3C match: {match_3c is not None}")
if match_3c:
    print(f"3C matched: {match_3c.group()}")

# Test manually
test_pattern = "%3C"
print(f"'{test_pattern}' in '{content}': {test_pattern in content}")