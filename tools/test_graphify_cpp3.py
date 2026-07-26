"""Inspect graphify C++ node structure."""
from graphify.extract import extract_cpp
from pathlib import Path

test_file = Path(r'C:\VSC_Project\ConvoPeq\src\CmaEsOptimizer.h')
result = extract_cpp(test_file)

# Show first 3 nodes in full detail
print('First 3 nodes detail:')
for n in result['nodes'][:3]:
    for k, v in n.items():
        print(f'  {k}: {v}')
    print('---')

# Show edges detail
print('First 3 edges detail:')
for e in result['edges'][:3]:
    for k, v in e.items():
        print(f'  {k}: {v}')
    print('---')

# Also check raw_calls
print('raw_calls count:', len(result.get('raw_calls', [])))
