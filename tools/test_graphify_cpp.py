"""Verify graphify C++ extraction works."""
from graphify.extract import extract_cpp
from pathlib import Path

test_file = Path(r'C:\VSC_Project\ConvoPeq\src\CmaEsOptimizer.h')
if test_file.exists():
    result = extract_cpp(test_file)
    print('Extraction result keys:', list(result.keys()))
    if 'nodes' in result:
        print('Nodes count:', len(result['nodes']))
        for n in result['nodes'][:10]:
            print('  ' + n.get('name', '?') + '  (' + n.get('type', '?') + ')')
    if 'edges' in result:
        print('Edges count:', len(result['edges']))
        for e in result['edges'][:5]:
            print('  ' + str(e))
else:
    print('File not found:', test_file)
