"""Check and fix graphify C++ extractor registration."""
import os, sys

pkg = r'C:\Users\user\AppData\Roaming\Python\Python314\site-packages\graphify'
ext_dir = os.path.join(pkg, 'extractors')

# Read base.py to find extension mapping
base_path = os.path.join(ext_dir, 'base.py')
src = open(base_path, encoding='utf-8').read()

print("=== Looking for C/C++ extension registration in base.py ===")
for i, line in enumerate(src.split('\n')):
    stripped = line.strip()
    # Look for file extension registrations
    if any(ext in stripped for ext in ["'.cpp'", "'.hpp'", "'.cc'", "'.cxx'", "'.c'", '"cpp"', '"c++"']):
        if len(stripped) < 200:
            print(f"L{i}: {stripped}")

# Check extractors/__init__.py for imports
init_path = os.path.join(ext_dir, '__init__.py')
if os.path.exists(init_path):
    init_src = open(init_path, encoding='utf-8').read()
    print("\n=== extractors/__init__.py ===")
    for i, line in enumerate(init_src.split('\n')):
        print(f"L{i}: {line}")
else:
    print(f"\n=== No __init__.py in extractors/ ===")

# Check if models.py has C++ LanguageConfig
models_path = os.path.join(ext_dir, 'models.py')
if os.path.exists(models_path):
    models_src = open(models_path, encoding='utf-8').read()
    print("\n=== Looking for C/C++ LanguageConfig in models.py ===")
    for i, line in enumerate(models_src.split('\n')):
        if 'cpp' in line.lower() or 'c++' in line.lower() or '.cpp' in line:
            print(f"L{i}: {line.strip()[:200]}")
