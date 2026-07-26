"""Test graphify C++ extraction with detailed output."""
from graphify.extract import extract_cpp
from pathlib import Path

test_file = Path(r'C:\VSC_Project\ConvoPeq\src\CmaEsOptimizer.h')
result = extract_cpp(test_file)

print('Total nodes:', len(result['nodes']))
print('Total edges:', len(result['edges']))

# Show all nodes with their types
print('\n--- All nodes (type, name) ---')
for n in result['nodes']:
    ntype = n.get('type', '?')
    nname = n.get('name', '?')
    if ntype not in ('comment',):
        print(f'  type={ntype:20s} name={nname}')

# Show unique types
types = set(n.get('type', '?') for n in result['nodes'])
print('\nUnique node types:', sorted(types))
