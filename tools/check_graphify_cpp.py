"""Check graphify C++ extractor registration and fix if needed."""
import sys
import os

# Read the engine source
engine_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
    'AppData', 'Roaming', 'Python', 'Python314', 'site-packages',
    'graphify', 'extractors', 'engine.py')
# Try direct path
alt_path = r'C:\Users\user\AppData\Roaming\Python\Python314\site-packages\graphify\extractors\engine.py'

path = alt_path if os.path.exists(alt_path) else engine_path
print(f"Reading: {path}")

src = open(path, encoding='utf-8').read()
lines = src.split('\n')

# Find file extension mapping
print("\n=== File extension registrations ===")
for i, line in enumerate(lines):
    stripped = line.strip()
    if any(ext in stripped for ext in ['.cpp', '.hpp', '.cc', '.cxx', '.h\"', '.c\"', '.c ']):
        if len(stripped) < 200:
            print(f"L{i}: {stripped}")

# Check LANGUAGE_EXTRACTORS or similar
print("\n=== Looking for language registry ===")
for i, line in enumerate(lines):
    stripped = line.strip()
    if any(x in stripped for x in ['LANGUAGE_EXTRACTOR', 'ext_map', 'lang_map', '_LANGUAGE_', 'extension_map', 'EXTENSION']):
        print(f"L{i}: {stripped[:200]}")

# Check if C/C++ or cpp in any mapping function
print("\n=== Checking for C/C++ in dispatch ===")
for i, line in enumerate(lines):
    stripped = line.strip()
    if any(x in stripped.lower() for x in ['.cc:', '.cpp:', '.hpp:', '.cxx:', '.h:', 'c++', 'tree_sitter_c', 'ts_language']):
        print(f"L{i}: {stripped[:200]}")

# Check init.py of extractors
init_path = os.path.join(os.path.dirname(path), '__init__.py')
if os.path.exists(init_path):
    init_src = open(init_path, encoding='utf-8').read()
    print(f"\n=== extractors/__init__.py contents ===")
    print(init_src[:2000])
else:
    print(f"\n=== extractors/__init__.py NOT FOUND at {init_path} ===")
