#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""build_identity_gate.py v2 - D162-2-I3-4-G (COHERENCE-1..5 gate, fail-closed)

v2 additions (I3-4-G, on top of I3-4-D v1):
  G1  Windows SDK identity stamped at configure time and verified before every build:
      windows_sdk_dir / windows_sdk_version / windows_sdk_include_fingerprint /
      windows_sdk_fingerprint_source. The fingerprint (relpath|size|mtime_ns manifest
      over the SDK include tree) detects in-place servicing that keeps the same
      version number. SDK env is read from the vcvars-provided environment, which is
      guaranteed alive because the gate runs inside build.bat.
  G2  #deps 0 semantic classification (I3-4-F contract):
      DEPS_VALID              #deps > 0                         -> PASS
      ZERO_DEPS_SYSTEM_ONLY   #deps 0 + closure proves no       -> WARN + ALLOW
                              non-system dep (test obj only)
      ZERO_DEPS_SUSPICIOUS    #deps 0 + closure finds a         -> FAIL
                              non-system dep
      ZERO_DEPS_UNRESOLVED    #deps 0 + closure cannot resolve  -> FAIL
      production objects (inputs of the ConvoPeq_artefacts\<cfg>\ConvoPeq.exe link
      edge) with #deps 0 -> ALWAYS FAIL, no exceptions.
  G3  provenance boundaries: TU identity = full source path + edge identity (never
      basename); permission ground = transitive include closure (never a filename
      whitelist, never a bare #deps 0 value).

v1 contract unchanged:
  COHERENCE-1/4 identity stamp verified before every build (any diff -> rc=3).
  COHERENCE-3   dependency information missing -> fail-closed (now semantic).
  COHERENCE-5   no object without provenance reaches the Release link.
  Scope guard: only project-source TUs (src/**) are classified; JUCE/third-party/
  resource artifacts stay outside the gate (D4). No automatic clean ever.

Usage:
  python src/tools/build_identity_gate.py --build-dir build           # stamp + gate (after configure)
  python src/tools/build_identity_gate.py --build-dir build --check   # gate only (before build)
  python src/tools/build_identity_gate.py --build-dir build --show    # print stamp
  python src/tools/build_identity_gate.py --build-dir build --check --explain-zero-deps
                                                      # diagnostic detail, rc=0 on classify-only
"""
import argparse, hashlib, os, re, subprocess, sys, json

RELEVANT_OBJ_RE = re.compile(r'/(?:Release|Debug|RelWithDebInfo)/src/', re.I)
PRODUCTION_EXE_RE = re.compile(r'ConvoPeq_artefacts[/\\](?:Release|Debug|RelWithDebInfo)[/\\]ConvoPeq\.exe$', re.I)
SYSTEM_PATH_RE = re.compile(r'program files|microsoft visual studio', re.I)
INC_RE = re.compile(r'^\s*#\s*include\s*([<"])([^">]+)[">]', re.M)
STAMP_SCHEMA = 2
SDK_SUBTREES = ['ucrt', 'um', 'shared', 'winrt', 'cppwinrt']


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


def normalize_env_path(p):
    return p.replace('/', '\\').strip().rstrip('\\')


def windows_sdk_env():
    """The SDK MSVC actually uses = vcvars selection (I3-4-F F5 truth source)."""
    d = os.environ.get('WindowsSdkDir', '')
    v = os.environ.get('WindowsSDKVersion', '')
    if not d.strip() or not v.strip():
        return None
    return normalize_env_path(d), v.rstrip('\\').rstrip('/')


def dir_fingerprint(base, subtrees):
    """Deterministic (relpath|size|mtime_ns) manifest -> sha256[:16]. Detects
    in-place SDK servicing that keeps the same version string."""
    rows = []
    for sub in subtrees:
        d = os.path.join(base, sub)
        if not os.path.isdir(d):
            continue
        for root, dirs, files in os.walk(d):
            dirs.sort()
            for f in sorted(files):
                p = os.path.join(root, f)
                rel = os.path.relpath(p, base).lower().replace('\\', '/')
                try:
                    st = os.stat(p)
                    rows.append((rel, st.st_size, st.st_mtime_ns))
                except OSError:
                    rows.append((rel, -1, -1))
    h = hashlib.sha256()
    for r in rows:
        h.update(('%s|%d|%d\n' % r).encode('utf-8'))
    return h.hexdigest()[:16], len(rows)


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
        rc2, fv = run(['powershell', '-NoProfile', '-Command',
                       "(Get-Item -LiteralPath \"%s\").VersionInfo.FileVersion" % cxx.strip('"')])
        if fv.strip():
            ver = 'MSVC ' + fv.strip()
    rc, sha = run(['git', '-C', source_root, 'rev-parse', '--short', 'HEAD'])
    rc, dirty = run(['git', '-C', source_root, 'status', '--porcelain'])
    ident = {
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
    sdk = windows_sdk_env()
    if sdk:
        sdk_dir, sdk_ver = sdk
        fp, n = dir_fingerprint(os.path.join(sdk_dir, 'include', sdk_ver), SDK_SUBTREES)
        ident['windows_sdk_dir'] = sdk_dir
        ident['windows_sdk_version'] = sdk_ver
        ident['windows_sdk_include_fingerprint'] = fp
        ident['windows_sdk_fingerprint_source'] = 'environment'
    else:
        ident['windows_sdk_dir'] = 'unknown'
        ident['windows_sdk_version'] = 'unknown'
        ident['windows_sdk_include_fingerprint'] = 'unknown'
        ident['windows_sdk_fingerprint_source'] = 'environment'
    return ident


def stamp_path(build_dir):
    return os.path.join(build_dir, 'CMakeFiles', '.build_identity')


def write_stamp(build_dir, ident):
    ident = dict(ident)
    ident['stamp_schema'] = STAMP_SCHEMA
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


# ---------------------------------------------------------------- G3: manifest

def parse_impl_manifest(impl_path):
    """obj path (forward-slash) -> {'ins': [...], 'vars': {...}}. TU identity is the
    full edge identity (output path + inputs), never a basename."""
    edges = {}
    cur = None
    for raw in open(impl_path, 'rb').read().split(b'\n'):
        line = raw.rstrip(b'\r')
        if line.startswith(b'build '):
            head = line[6:].decode('latin-1')
            outp, _, rest = head.partition(':')
            toks = rest.split()
            inputs = []
            for t in toks[1:]:
                if t in ('|', '||'):
                    break
                inputs.append(t.replace('$:', ':'))
            cur = outp.strip().replace('\\', '/')
            edges[cur] = {'ins': inputs, 'vars': {}}
        elif line.startswith(b'  ') and cur and line.strip():
            k, _, v = line.strip().partition(b'=')
            edges[cur]['vars'][k.decode('latin-1').strip()] = v.decode('latin-1').strip()
        elif line.strip() and not line.startswith(b' '):
            cur = None
    return edges


def production_link_objs(edges):
    prods = set()
    for out, e in edges.items():
        if PRODUCTION_EXE_RE.search(out):
            for i in e['ins']:
                if i.lower().endswith('.obj'):
                    prods.add(i.replace('\\', '/'))
    return prods


# ------------------------------------------------- G2: zero-deps classification

def tu_closure(src_abs, inc_dirs, env_inc_dirs):
    """Transitive include closure (I3-4-F contract). quoted: dir(cur) then -I dirs;
    angle: -I dirs then INCLUDE env. Recurses only into non-system headers
    (mirrors ninja IsSystemInclude dropping 'program files' paths)."""
    res = {'NONSYSTEM': set(), 'UNRESOLVED': []}
    seen = set()
    stack = [src_abs]
    while stack:
        cur = stack.pop()
        key = cur.lower()
        if key in seen:
            continue
        seen.add(key)
        try:
            text = open(cur, 'rb').read().decode('utf-8', 'replace')
        except OSError:
            res['UNRESOLVED'].append((cur, '<unreadable source>'))
            continue
        for m in INC_RE.finditer(text):
            q, spec = m.group(1), m.group(2)
            cands = []
            if q == '"':
                cands.append(os.path.normpath(os.path.join(os.path.dirname(cur), spec)))
            for d in inc_dirs:
                cands.append(os.path.normpath(os.path.join(d, spec)))
            if q == '<':
                for d in env_inc_dirs:
                    cands.append(os.path.normpath(os.path.join(d, spec)))
            hit = next((c for c in cands if os.path.isfile(c)), None)
            if hit is None:
                res['UNRESOLVED'].append((spec, cur))
                continue
            if SYSTEM_PATH_RE.search(hit):
                continue
            res['NONSYSTEM'].add(hit)
            stack.append(hit)
    return res


def classify_zero_deps(build_dir, config, deps_zero, edges, verbose=False):
    """Returns (allowed, prod_zero, suspicious, unresolved) - lists of detail tuples."""
    production = production_link_objs(edges)
    env_inc = [normalize_env_path(d) for d in os.environ.get('INCLUDE', '').split(';') if d.strip()]
    allowed, prod_zero, suspicious, unresolved = [], [], [], []
    for obj in deps_zero:
        slash = obj.replace('\\', '/')
        if not RELEVANT_OBJ_RE.search(slash):
            continue
        if slash in production:
            prod_zero.append((obj, 'object is an input of the production ConvoPeq.exe link edge'))
            continue
        e = edges.get(slash)
        if e is None or not e['ins']:
            unresolved.append((obj, [('no compile edge found for object', '')]))
            continue
        src = os.path.normpath(e['ins'][0])
        inc_dirs = re.findall(r'[-/]I"?([^"\s]+)"?', e['vars'].get('INCLUDES', ''))
        cl = tu_closure(src, inc_dirs, env_inc)
        if cl['UNRESOLVED']:
            unresolved.append((obj, cl['UNRESOLVED'][:4]))
        elif cl['NONSYSTEM']:
            suspicious.append((obj, sorted(cl['NONSYSTEM'])[:4]))
        else:
            allowed.append(obj)
            if verbose:
                print('[GATE-INFO] system-only closure proven for ' + obj)
    return allowed, prod_zero, suspicious, unresolved


def deps_gate(build_dir, config, explain=False):
    """COHERENCE-3 (semantic, G2): collect #deps 0 relevant objs, classify, fail-closed.
    explain=True: diagnostic detail, always rc=0 (identity validation has already run)."""
    impl = os.path.join(build_dir, 'CMakeFiles', 'impl-%s.ninja' % config)
    if not os.path.exists(impl):
        fail('impl-%s.ninja not found (configure first).' % config)
    rc, out = run(['ninja', '-C', build_dir, '-f', impl, '-t', 'deps'])
    if rc != 0:
        fail('ninja -t deps failed (rc=%d).' % rc)
    zero = []
    checked = 0
    for line in out.split('\n'):
        if ': #deps' in line:
            cur_obj = line.split(':')[0].strip()
            n = int(line.split('#deps ')[1].split(',')[0])
            if RELEVANT_OBJ_RE.search(cur_obj):
                checked += 1
                if n == 0:
                    zero.append(cur_obj)
    edges = parse_impl_manifest(impl)
    production = production_link_objs(edges)
    env_inc = [normalize_env_path(d) for d in os.environ.get('INCLUDE', '').split(';') if d.strip()]
    allowed, prod_zero, suspicious, unresolved = [], [], [], []
    for obj in zero:
        slash = obj.replace('\\', '/')
        if not RELEVANT_OBJ_RE.search(slash):
            continue
        if slash in production:
            prod_zero.append((obj, 'object is an input of the production ConvoPeq.exe link edge'))
            continue
        e = edges.get(slash)
        if e is None or not e['ins']:
            unresolved.append((obj, [('no compile edge found for object', '')]))
            continue
        src = os.path.normpath(e['ins'][0])
        inc_dirs = re.findall(r'[-/]I"?([^"\s]+)"?', e['vars'].get('INCLUDES', ''))
        cl = tu_closure(src, inc_dirs, env_inc)
        if cl['UNRESOLVED']:
            unresolved.append((obj, cl['UNRESOLVED'][:4]))
        elif cl['NONSYSTEM']:
            suspicious.append((obj, sorted(cl['NONSYSTEM'])[:4]))
        else:
            allowed.append(obj)

    for obj in allowed:
        print('[GATE-WARN] ZERO_DEPS_SYSTEM_ONLY (test-only, no project dependency '
              'proven by closure): ' + obj)
    if explain:
        print('[GATE-INFO] zero-deps classification detail (config=%s)' % config)
        print('[GATE-INFO] relevant objs checked: %d ; #deps 0: %d' % (checked, len(zero)))
        print('[GATE-INFO] system-only allowed: %d ; production-fail: %d ; '
              'suspicious: %d ; unresolved: %d'
              % (len(allowed), len(prod_zero), len(suspicious), len(unresolved)))
        for obj, why in prod_zero:
            print('[GATE-INFO]   PRODUCTION_ZERO_DEPS (would FAIL): %s (%s)' % (obj, why))
        for obj, ex in suspicious:
            print('[GATE-INFO]   ZERO_DEPS_SUSPICIOUS (would FAIL): %s' % obj)
            for s in ex:
                print('[GATE-INFO]     non-system dep: %s' % s)
        for obj, ex in unresolved:
            print('[GATE-INFO]   ZERO_DEPS_UNRESOLVED (would FAIL): %s' % obj)
            for s, frm in ex:
                print('[GATE-INFO]     unresolved include: %s (in %s)' % (s, os.path.basename(frm)))
        print('[GATE-INFO] diagnostic mode: classification result does not affect exit code.')
        return

    if prod_zero or suspicious or unresolved:
        n = len(prod_zero) + len(suspicious) + len(unresolved)
        print('[GATE-FAIL] %d zero-deps object(s) FAILED classification (config=%s):' % (n, config))
        for obj, why in prod_zero:
            print('[GATE-FAIL]   PRODUCTION #deps 0 (always fail): %s' % obj)
        for obj, ex in suspicious:
            print('[GATE-FAIL]   ZERO_DEPS_SUSPICIOUS (project dependency present but '
                  'deps lost): %s' % obj)
            for s in ex:
                print('[GATE-FAIL]     non-system dep: %s' % s)
        for obj, ex in unresolved:
            print('[GATE-FAIL]   ZERO_DEPS_UNRESOLVED (cannot prove absence): %s' % obj)
            for s, frm in ex:
                print('[GATE-FAIL]     unresolved include: %s' % s)
        print('[GATE-FAIL] These objects would silently ignore future header changes')
        print('[GATE-FAIL] (root cause of the D162-2 ODR/layout mixing crash).')
        fail('zero-deps classification failed for %d object(s).' % n,
             'run: build.bat %s clean   (explicit clean recovery)' % config)
    print('[GATE-OK ] dependency gate: %d relevant .obj checked; %d zero-deps system-only '
          'allowed (warn); production #deps 0 = 0; suspicious = 0; unresolved = 0.'
          % (checked, len(allowed)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--build-dir', default='build')
    ap.add_argument('--source-root', default=None)
    ap.add_argument('--config', default='Release')
    ap.add_argument('--check', action='store_true',
                    help='verify existing stamp + deps gate (no stamp write)')
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--explain-zero-deps', action='store_true',
                    help='diagnostic: print zero-deps classification detail (rc=0 unless identity fails)')
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
                     + '\n[GATE-FAIL] /showIncludes prefix - incremental build is not trusted.',
                     'run: build.bat %s clean   (explicit clean recovery)' % a.config)
            if any('windows_sdk' in c for c in changed):
                old_schema = old.get('stamp_schema') if old else None
                if old_schema == STAMP_SCHEMA:
                    fail('Windows SDK identity changed since last configure:\n           '
                         + '\n           '.join(c for c in changed if 'windows_sdk' in c)
                         + '\n[GATE-FAIL] SDK headers are not tracked by ninja deps'
                         + ' (system-include class);'
                         + '\n[GATE-FAIL] existing .obj may be stale against the new SDK.',
                         'run: build.bat %s clean   (explicit clean recovery)' % a.config)
                else:
                    print('[GATE-WARN] stamp migrated schema %s -> %d; Windows SDK fields '
                          'recorded for the first time.' % (old_schema, STAMP_SCHEMA))
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
    print('[GATE-OK ] build identity matches stamp (codepage=%s, prefix=%s..., sdk=%s/%s...).' % (
        ident['console_codepage'], ident['msvc_deps_prefix_sha256'][:8],
        ident['windows_sdk_version'][:12], ident['windows_sdk_include_fingerprint'][:8]))
    deps_gate(build_dir, a.config, explain=a.explain_zero_deps)


if __name__ == '__main__':
    main()
