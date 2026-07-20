# -*- coding: utf-8 -*-
import re

with open('C:\\VSC_Project\\ConvoPeq\\doc\\work77\\AutoGainStagingRenewal.md', 'r', encoding='utf-8') as f:
    c = f.read()

print('Header:', c.split('\n')[0])

m = re.search(r'enum class BoundMethod.*?\};', c, re.DOTALL)
if m: print('BoundMethod:', m.group()[:200])
else: print('BoundMethod: NOT FOUND')

print('EmpiricalSafetyMarginPolicy:', 'EmpiricalSafetyMarginPolicy' in c)

# Check isValid
if 'case 1: case 2: case 4: case 8: return true' in c:
    print('isValid: simplified (OK)')
else:
    print('isValid: NOT simplified')

# Check BuildAnalysis
idx = c.find('struct BuildAnalysis')
if idx >= 0:
    snippet = c[idx:idx+900]
    if 'int resolvedOsFactor' in snippet and '// Diagnostics' not in snippet[:snippet.find('int resolvedOsFactor')]:
        print('BuildAnalysis: resolvedOsFactor in MAIN (needs fix)')
    else:
        print('BuildAnalysis: resolvedOsFactor moved to Diagnostics or removed (OK)')

# Check finite check
print('finite has resolvedOsFactor:', 'resolvedOsFactor > 0' in c)
print('Done')
