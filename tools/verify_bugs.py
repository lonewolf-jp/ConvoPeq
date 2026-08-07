import re, os, sys

root = 'src'
results = []

def search_pattern(pattern, desc):
    matches = []
    for r, dirs, files in os.walk(root):
        for fname in files:
            if fname.endswith(('.cpp', '.h')):
                fpath = os.path.join(r, fname)
                try:
                    with open(fpath, 'r', errors='replace') as f:
                        for i, line in enumerate(f, 1):
                            if re.search(pattern, line):
                                matches.append((fpath, i, line.rstrip()[:120]))
                except:
                    pass
    return desc, len(matches), matches[:8]

checks = [
    ('1-1_nucHCMode', r'nucH[CL]Mode'),
    ('1-3_mm256_store_pd', r'_mm256_store_pd'),
    ('1-4_inline_fastTanh', r'inline double fastTanh\('),
    ('1-5_musicalSoftClip', r'AudioEngine::DSPCore::musicalSoftClip'),
    ('1-6_ippsFFT', r'ippsFFTFwd_RToCCS_64f'),
    ('1-7_fallbackMutex', r'fallbackMutex_'),
    ('2-2_sanitizeFinite', r'sanitizeFiniteChunk'),
    ('2-3_delta_1e-6', r'delta.*1e-6'),
    ('2-4_reinterpret_double', r'reinterpret_cast<const double\*>'),
    ('2-7_atomic_DSPHandle', r'atomic<DSPHandle>'),
    ('2-9_uint64_sub', r'observeUs - matchedPublishEndUs|nowUs - cbStartUs|cbStartUs - cbPrevEndUs'),
    ('2-10_NoiseShaperType_cast', r'NoiseShaperType\)\('),
    ('3-5_volatile_sink', r'volatile.*sink'),
    ('3-6_hashCombineFloat', r'hashCombineFloat'),
    ('3-7_alignas', r'alignas'),
    ('3-8_cachedLatency', r'cachedLatency'),
    ('R-1_dryScaledL', r'dryScaledL =const'),
    ('R-6_validateAndClamp', r'validateAndClampParameters'),
    ('R-7_ConvoPeq_CPU', r'ConvoPeq - CPU'),
    ('R-8_pragma_warning', r'#pragma warning'),
    ('R-13_advancePhase', r'advancePhase'),
    ('R-14_IRDSP_cancel', r'shouldExit'),
    ('R-15_audioCallbackActiveCount', r'audioCallbackActiveCount'),
]

for desc, pattern in checks:
    label, cnt, matches = search_pattern(pattern, desc)
    print(f'[{label}] pattern={pattern}: {cnt} matches')
    for fpath, lineno, text in matches:
        print(f'  {fpath}:{lineno}: {text}')
    print()
