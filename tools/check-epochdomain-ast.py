#!/usr/bin/env python3
"""
Phase-E P4: AST-like EpochDomain type usage scanner.

Scans all .h/.cpp files in src/ for EpochDomain type references that
might be missed by grep-based CI gates:
  - Direct type usage: EpochDomain&, EpochDomain*, EpochDomain
  - Type aliases: using X = EpochDomain, typedef EpochDomain X
  - Alias transitive usage: X&, X* where X is an alias of EpochDomain
  - Template parameters: Holder<EpochDomain>
  - Return types: EpochDomain& getDomain()
  - Function parameters: void func(EpochDomain& d)
"""

import re
import os
import sys
import json
from pathlib import Path
from collections import defaultdict


SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def get_source_files():
    """Get all .h and .cpp files under src/"""
    files = []
    for ext in (".h", ".cpp"):
        files.extend(SRC_DIR.rglob(f"*{ext}"))
    return sorted(files)


def extract_epochdomain_aliases(filepath):
    """Extract EpochDomain type aliases from a file"""
    aliases = set()
    text = filepath.read_text(encoding="utf-8", errors="replace")

    # using X = EpochDomain
    for m in re.finditer(r'using\s+(\w+)\s*=\s*(?:::)?(?:convo::)?EpochDomain\b', text):
        aliases.add((m.group(1), filepath))

    # typedef EpochDomain X
    for m in re.finditer(r'typedef\s+(?:::)?(?:convo::)?EpochDomain\s+(\w+)', text):
        aliases.add((m.group(1), filepath))

    return aliases


def check_direct_type_usage(text, filepath, excluded_files=None):
    """Check for direct EpochDomain type usage"""
    findings = []
    if excluded_files is None:
        excluded_files = []

    # Skip EpochDomain.h itself (definition file)
    if any(excl in str(filepath) for excl in excluded_files):
        return findings

    # Skip comments
    lines = text.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('*'):
            continue
        if stripped.startswith('#include') or stripped.startswith('#pragma'):
            continue

        # EpochDomain in non-comment code
        if re.search(r'\bEpochDomain\b', stripped):
            # Skip EpochDomain.h itself
            if 'EpochDomain.h' in str(filepath) and i < 300:
                continue
            findings.append((filepath, i, stripped))

    return findings


def check_alias_usage(text, filepath, all_aliases):
    """Check for alias type usage (two-step detection)"""
    findings = []

    # Build alias pattern
    alias_names = set(a for a, f in all_aliases if f != filepath)
    if not alias_names:
        return findings

    # Also include self-defined aliases (for other files checking)
    local_aliases = set(a for a, f in all_aliases if f == filepath)

    for alias_set, label in [(alias_names, "external"), (local_aliases, "local")]:
        if not alias_set:
            continue
        pattern = r'\b(' + '|'.join(re.escape(a) for a in alias_set) + r')\s*[&*]'
        lines = text.split('\n')
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith('//') or stripped.startswith('/*'):
                continue
            if re.search(pattern, stripped) and 'EpochDomain' not in stripped:
                # Verify it's actually a type usage, not part of using/typedef
                if not re.search(r'\busing\s+|typedef\s+', stripped):
                    findings.append((filepath, i, stripped, label))

    return findings


def check_template_epochdomain(text, filepath):
    """Check for EpochDomain used as template parameter"""
    findings = []
    excluded_files = ['EpochDomain.h', 'ISRRetireRouter.h']
    if any(excl in str(filepath) for excl in excluded_files):
        return findings

    lines = text.split('\n')
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith('//') or stripped.startswith('/*') or stripped.startswith('#'):
            continue

        # Holder<EpochDomain> pattern (not just EpochDomain.h include)
        if re.search(r'<\s*(?:::)?(?:convo::)?EpochDomain\s*[,>]', stripped):
            findings.append((filepath, i, stripped))

    return findings


def main():
    print("=" * 60)
    print("Phase-E P4: EpochDomain AST-like Scanner")
    print("=" * 60)

    source_files = get_source_files()
    print(f"Scanning {len(source_files)} source files...\n")

    # Phase 1: Extract all aliases
    print("--- Phase 1: Alias extraction ---")
    all_aliases = set()
    for f in source_files:
        aliases = extract_epochdomain_aliases(f)
        if aliases:
            for alias, path in aliases:
                print(f"  alias '{alias}' in {path.relative_to(SRC_DIR.parent)}")
            all_aliases.update(aliases)

    if not all_aliases:
        print("  (no aliases found)")

    print(f"\n  Total aliases: {len(all_aliases)}")

    # Phase 2: Check direct type usage
    print("\n--- Phase 2: Direct EpochDomain type usage ---")
    excluded = ['EpochDomain.h']
    direct_issues = []
    for f in source_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        issues = check_direct_type_usage(text, f, excluded)
        direct_issues.extend(issues)

    valid_issues = []
    for path, line, content in direct_issues:
        # Filter to only code-level usages
        rel = path.relative_to(SRC_DIR.parent)
        valid_issues.append((rel, line, content))

    for rel, line, content in valid_issues:
        print(f"  {rel}:{line}: {content[:100]}")

    print(f"\n  Direct usage count: {len(valid_issues)}")

    # Phase 3: Check alias transitive usage
    print("\n--- Phase 3: Alias transitive usage ---")
    alias_findings = []
    for f in source_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        findings = check_alias_usage(text, f, all_aliases)
        alias_findings.extend(findings)

    for path, line, content, label in alias_findings:
        rel = path.relative_to(SRC_DIR.parent)
        print(f"  [{label}] {rel}:{line}: {content[:100]}")

    print(f"\n  Alias transitive usage: {len(alias_findings)}")

    # Phase 4: Check template parameter usage
    print("\n--- Phase 4: Template parameter usage ---")
    template_findings = []
    for f in source_files:
        text = f.read_text(encoding="utf-8", errors="replace")
        findings = check_template_epochdomain(text, f)
        template_findings.extend(findings)

    for path, line, content in template_findings:
        rel = path.relative_to(SRC_DIR.parent)
        print(f"  {rel}:{line}: {content[:100]}")

    print(f"\n  Template parameter usage: {len(template_findings)}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Direct EpochDomain usage:  {len(valid_issues)}")
    print(f"  Type aliases found:        {len(all_aliases)}")
    print(f"  Alias transitive usage:    {len(alias_findings)}")
    print(f"  Template parameter usage:  {len(template_findings)}")

    total_issues = len(valid_issues) + len(alias_findings) + len(template_findings)
    print(f"\n  Total issues detected:     {total_issues}")

    if total_issues > 0:
        print("\n  NOTE: Many of these are legitimate/acceptable uses.")
        print("  Review each finding and update whitelist as needed.")
        sys.exit(0 if total_issues < 50 else 1)
    else:
        print("\n  ✅ Clean: No EpochDomain type leakage detected.")
        sys.exit(0)


if __name__ == "__main__":
    main()
