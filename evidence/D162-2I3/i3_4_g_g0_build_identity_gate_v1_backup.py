#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_identity_gate.py — D162-2-I3-4-D (COHERENCE-1..5 gate, fail-closed)

Build-system integrity gate. Never performs an automatic clean; on any failure it
exits non-zero and instructs the caller to run an explicit clean recovery.

Usage:
  python src/tools/build_identity_gate.py --build-dir build           # stamp + gate (after configure)
  python src/tools/build_identity_gate.py --build-dir build --check   # gate only (before build)
  python src/tools/build_identity_gate.py --build-dir build --show    # print stamp

Contract implemented (D162-2-I3-4-C):
  COHERENCE-1/4: build identity (generator/compiler/version/arch/config/cmake/ninja/
                 source revision/console codepage/msvc_deps_prefix hash) is stamped at
                 configure time and re-verified before every build.
  COHERENCE-3:   any relevant .obj with #deps 0 in `ninja -t deps` fails the gate
                 (ninja itself treats #deps 0 as VALID; we override fail-closed).
  COHERENCE-5:   objects without provenance never reach the Release link.
Scope guard: only project-source TUs (src/**, tests) are gated; JUCE/third-party/
resource artifacts are excluded (D4).
"""
import argparse, hashlib, os, re, subprocess, sys, json

RELEVANT_OBJ_RE = re.compile(r'/(?:Release|Debug|RelWithDebInfo)/src/', re.I)


def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                       errors='replace', shell=False)
    return p.returncode, (p.stdout or '') + (p.stderr or '')


def console_codepage():
    rc, out = run(['cmd', '/c', 'chcp'])
    m = re.search(r':\s*(\d+)', out or '')
    return m.group(1) if m else 'unknown'


def file_sha256(path):
    try:
        return hashlib.sha256(open(path, 'rb').read()).hexdigest()
    except OSError:
        return None


def rules_prefix_sha256(build_dir):
    rules = os.path.join(build_dir, 'CMakeFiles', 'rules.ninja')
    data = open(rules, 'rb').read() if os.path.exists(rules) else b''
    m = re.search(rb'msvc_deps_prefix = ([^\r\n]*)', data)
    if not m:
        return 'no-msvc-deps-prefix'
    return hashlib.sha256(m.group(1)).hexdigest()[:16]


def cmake_cache_field(build_dir, key):
    p = os.path.join(build_dir, 'CMakeCache.txt')
    if not os.path.exists(p):
        return None
    for line in open(p, encoding='utf-8', errors='replace'):
        if line.startswith(key + ':'):
            return line.split('=', 1)[1].strip()
    return None


def gather_identity(build_dir, source_root):
    gen = cmake_cache_field(build_dir, 'CMAKE_GENERATOR') or 'unknown'
    cxx = cmake_cache_field(build_dir, 'CMAKE_CXX_COMPILER') or 'unknown'
    cc = cmake_cache_field(build_dir, 'CMAKE_C_COMPILER') or 'unknown'
    _, nv = run(['ninja', '--version'])
    rc, cv = run([cxx.strip('"'), '/?'], cwd=source_root)
    ver = 'unknown'
    m = re.search(r'Version (\d+\.\d+)', cv)
    if m:
        ver = 'MSVC ' + m.group(1)
    else:
        # file version fallback
        rc2, fv = run(['powershell', '-NoProfile', '-Command',
                       "(Get-Item -LiteralPath \"%s\").VersionInfo.FileVersion" % cxx.strip('"')])
        if fv.strip():
            ver = 'MSVC ' + fv.strip()
    # source revision: short SHA + dirty flag (NOT the generation identity itself —
    # the codepage/prefix hash is part of it; see I3-4-C)
    rc, sha = run(['git', '-C', source_root, 'rev-parse', '--short', 'HEAD'])
    rc, dirty = run(['git', '-C', source_root, 'status', '--porcelain'])
    return {
        'generator': gen,
        'compiler_id': 'MSVC',
        'compiler_path': cxx,
        'compiler_version': ver,
        'architecture': 'x64',
        'cmake_version': (run(['cmake', '--version'])[1].split('\n')[0]),
        'ninja_version': nv.strip(),
        'configuration_family': 'Debug;Release;RelWithDebInfo',
        'source_revision': (sha.strip() or 'unknown') + ('+dirty' if dirty.strip() else ''),
        'console_codepage': console_codepage(),
        'msvc_deps_prefix_sha256': rules_prefix_sha256(build_dir),
    }


def stamp_path(build_dir):
    return os.path.join(build_dir, 'CMakeFiles', '.build_identity')


def write_stamp(build_dir, ident):
    ident = dict(ident)
    ident['stamp_schema'] = 1
    with open(stamp_path(build_dir), 'w', encoding='utf-8', newline='\n') as f:
        json.dump(ident, f, indent=1, sort_keys=True)
    return ident


def load_stamp(build_dir):
    try:
        return json.load(open(stamp_path(build_dir), encoding='utf-8'))
    except Exception:
        return None


def fail(msg, recovery=''):
    print('[GATE-FAIL] ' + msg)
    if recovery:
        print('[GATE-FAIL] recovery: ' + recovery)
    print('[GATE-FAIL] refusing to continue (fail-closed; no automatic clean).')
    sys.exit(3)


def deps_gate(build_dir, config):
    """COHERENCE-3: relevant .obj with #deps 0 => fail-closed."""
    impl = os.path.join(build_dir, 'CMakeFiles', 'impl-%s.ninja' % config)
    if not os.path.exists(impl):
        fail('impl-%s.ninja not found (configure first).' % config)
    rc, out = run(['ninja', '-C', build_dir, '-f', impl, '-t', 'deps'])
    if rc != 0:
        fail('ninja -t deps failed (rc=%d).' % rc)
    bad = []
    checked = 0
    cur_obj = None
    for line in out.split('\n'):
        if ': #deps' in line:
            cur_obj = line.split(':')[0].strip()
            n = int(line.split('#deps ')[1].split(',')[0])
            if RELEVANT_OBJ_RE.search(cur_obj):
                checked += 1
                if n == 0:
                    bad.append(cur_obj)
    if bad:
        print('[GATE-FAIL] %d relevant .obj file(s) have #deps 0 (no dependency info):'
              % len(bad))
        for b in bad:
            print('           ' + b)
        print('[GATE-FAIL] These objects would silently ignore future header changes')
        print('[GATE-FAIL] (root cause of D162-2 ODR/layout mixing crash).')
        fail('dependency-information missing for %d object(s).' % len(bad),
             'run: build.bat %s clean   (explicit clean recovery)' % config)
    print('[GATE-OK ] dependency gate: %d relevant .obj checked, all have dependency info.' % checked)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-dir', default='build')
    ap.add_argument('--source-root', default=None)
    ap.add_argument('--config', default='Release')
    ap.add_argument('--check', action='store_true',
                    help='verify existing stamp + deps gate (no stamp write)')
    ap.add_argument('--show', action='store_true')
    a = ap.parse_args()

    build_dir = os.path.abspath(a.build_dir)
    source_root = a.source_root or os.path.dirname(build_dir)
    if not os.path.exists(os.path.join(build_dir, 'CMakeCache.txt')):
        print('[GATE-FAIL] %s is not a configured CMake build directory.' % build_dir)
        sys.exit(3)

    ident = gather_identity(build_dir, source_root)
    if a.show:
        print(json.dumps(ident, indent=1, sort_keys=True))
        return

    old = load_stamp(build_dir)
    if not a.check or old is None:
        # configure phase: (re)write stamp
        changed = []
        if old:
            for k in sorted(ident):
                if old.get(k) != ident[k]:
                    changed.append('%s: %r -> %r' % (k, old.get(k), ident[k]))
        write_stamp(build_dir, ident)
        if changed:
            print('[GATE-OK ] build identity stamp updated (%d field(s) changed):' % len(changed))
            for c in changed:
                print('           ' + c)
            if any('msvc_deps_prefix_sha256' in c or 'console_codepage' in c for c in changed):
                fail('identity changed in dependency-parser-sensitive fields:\n           '
                     + '\n           '.join(c for c in changed
                                            if 'prefix' in c or 'codepage' in c)
                     + '\n[GATE-FAIL] Existing .obj may have been built under a different'
                     + '\n[GATE-FAIL] /showIncludes prefix — incremental build is not trusted.',
                     'run: build.bat %s clean   (explicit clean recovery)' % a.config)
        else:
            print('[GATE-OK ] build identity stamp written (first configure).')
        deps_gate(build_dir, a.config)
        return

    # --check: before-build verification
    diffs = []
    for k in sorted(ident):
        if old.get(k) != ident[k]:
            diffs.append('%s: stamp=%r now=%r' % (k, old.get(k), ident[k]))
    if diffs:
        print('[GATE-FAIL] build identity mismatch (stamp vs current environment):')
        for d in diffs:
            print('           ' + d)
        fail('COHERENCE-4 violated.',
             'run: build.bat %s clean   (explicit clean recovery)' % a.config)
    print('[GATE-OK ] build identity matches stamp (codepage=%s, prefix=%s...).' % (
        ident['console_codepage'], ident['msvc_deps_prefix_sha256'][:8]))
    deps_gate(build_dir, a.config)


if __name__ == '__main__':
    main()
