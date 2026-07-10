#!/usr/bin/env python3
"""work70: MKLNonUniformConvolver.cpp 計装 — SetImpulse 内 mkl_malloc → DIAG_MKL_MALLOC + allocSizes 保存 + IR_LOAD/IR_LAYOUT"""
import re

with open('src/MKLNonUniformConvolver.cpp', 'r', newline='') as f:
    content = f.read()

changes = 0

# =========================================================================
# Step 1: 全 mkl_malloc → DIAG_MKL_MALLOC (永続バッファのみ)
# =========================================================================
# Layer 永続バッファ (15箇所) — すべて DIAG_MKL_MALLOC + allocSizes 保存
layer_alloc_pairs = [
    ('l.irFreqDomain', 'irBufSize'),
    ('l.irFreqReal',   'irSoaSize'),
    ('l.irFreqImag',   'irSoaSize'),
    ('l.fdlBuf',       'fdlBufSize'),
    ('l.fdlReal',      'fdlSoaSize'),
    ('l.fdlImag',      'fdlSoaSize'),
    ('l.fftTimeBuf',   'l.fftSize'),
    ('l.fftOutBuf',    'l.fftSize'),
    ('l.prevInputBuf', 'l.partSize'),
    ('l.accumBuf',     'l.partStride'),
    ('l.accumReal',    'l.complexSize'),
    ('l.accumImag',    'l.complexSize'),
    ('l.inputAccBuf',  'l.partSize'),
    ('l.tailOutputBuf', 'l.partSize'),
]

for var, size_expr in layer_alloc_pairs:
    # Find: var = static_cast<double*>(mkl_malloc(size_expr * sizeof(double), 64));
    # This pattern appears for each Layer buffer
    old = f'{var} = static_cast<double*>(mkl_malloc('
    new = f'{var} = static_cast<double*>(DIAG_MKL_MALLOC('
    if old in content:
        # Replace just the mkl_malloc call
        content = content.replace(old, new, 1)
        changes += 1

        # Now add allocSizes saving after the line
        # Find the end of this malloc line
        idx = content.find(new)
        if idx >= 0:
            # Find the semicolon ending this statement
            eol = content.find(';\n', idx)
            if eol >= 0:
                # Create allocSizes save line
                field = var.split('.')[1]  # e.g., "irFreqDomain"
                save_line = f'#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS\nl.allocSizes.{field} = {size_expr} * sizeof(double);\n#endif\n'
                # Insert after the semicolon+newline
                insert_pos = eol + 2  # after ';\n'
                content = content[:insert_pos] + save_line + content[insert_pos:]
                changes += 1

# NUC 永続バッファ (4箇所) — DIAG_MKL_MALLOC のみ (allocSizes 保存不要)
nuc_alloc_vars = [
    'm_directIRRev',
    'm_directHistory',
    'm_directWindow',
    'm_directOutBuf',
    'm_ringBuf',
]
for var in nuc_alloc_vars:
    old = f'{var} = static_cast<double*>(mkl_malloc('
    new = f'{var} = static_cast<double*>(DIAG_MKL_MALLOC('
    if old in content:
        content = content.replace(old, new, 1)
        changes += 1

# =========================================================================
# Step 2: SetImpulse 内に IR_LOAD + IR_LAYOUT ログを追加
# =========================================================================
# Find: convo::publishAtomic(m_ready, true, std::memory_order_release);
# This is where SetImpulse succeeds. After this we add the logs.

ready_marker = 'convo::publishAtomic(m_ready, true, std::memory_order_release);'
if ready_marker in content:
    ir_logs_code = '''
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t afterMkl = convo::diag::allocatedBytes();
    const uint32_t afterLost = convo::diag::lostFreeCount();

    const int l0Part = m_numActiveLayers >= 1 ? m_layers[0].partSize : 0;
    const int l1Part = m_numActiveLayers >= 2 ? m_layers[1].partSize : 0;
    const int l2Part = m_numActiveLayers >= 3 ? m_layers[2].partSize : 0;

    // IR_LOAD
    diagLogNonRt(juce::String::formatted(
        "[IR_LOAD] NUC#%p seq=%llu irLen=%d blockSize=%d "
        "Layers=%d L0Part=%d L1Part=%d L2Part=%d "
        "directTaps=%d ringSize=%d "
        "MKL: before=%lluMB after=%lluMB delta=%lldMB "
        "lostFree=%u(+%d) live=%u",
        (void*)this,
        (unsigned long long)diagSeq,
        irLen, blockSize,
        m_numActiveLayers, l0Part, l1Part, l2Part,
        m_directTapCount, m_ringSize,
        (unsigned long long)(beforeMkl / (1024*1024)),
        (unsigned long long)(afterMkl / (1024*1024)),
        (long long)((int64_t)(afterMkl) - (int64_t)(beforeMkl)) / (1024*1024),
        (unsigned)afterLost, (int)((int32_t)(afterLost) - (int32_t)(beforeLost)),
        (unsigned)liveCount.load(std::memory_order_relaxed)));

    // IR_LAYOUT (1 回の getDiagnostics で Layer 情報 + 種別内訳を取得)
    const auto __snap = getDiagnostics();
    diagLogNonRt(juce::String::formatted(
        "[IR_LAYOUT] NUC#%p seq=%llu "
        "IRFreq=%.0fMB FDL=%.0fMB Accum=%.0fMB Tail=%.0fMB "
        "Direct=%.0fMB Ring=%.0fMB Total=%.0fMB(persistent data buffers only) | "
        "L0=%.0fMB L1=%.0fMB L2=%.0fMB",
        (void*)this,
        (unsigned long long)diagSeq,
        __snap.irFreqBytes / (1024.0*1024.0),
        __snap.fdlBytes    / (1024.0*1024.0),
        __snap.accumBytes  / (1024.0*1024.0),
        __snap.tailBytes   / (1024.0*1024.0),
        __snap.directBytes / (1024.0*1024.0),
        __snap.ringBytes   / (1024.0*1024.0),
        __snap.totalBytes() / (1024.0*1024.0),
        __snap.layerBufs[0] / (1024.0*1024.0),
        __snap.layerBufs[1] / (1024.0*1024.0),
        __snap.layerBufs[2] / (1024.0*1024.0)));
#endif

'''
    content = content.replace(ready_marker, ready_marker + ir_logs_code, 1)
    changes += 1

# =========================================================================
# Step 3: SetImpulse 先頭に diagSeq 採番 + beforeMkl/beforeLost 追加
# =========================================================================
# Find SetImpulse's first lines and add before measurements
# We need to add the diagSeq + beforeMkl/beforeLost/beforeOs BEFORE releaseAllLayers

# Insert at beginning of SetImpulse, after the pointer/release check
# Marker: the early-return check block
setimpulse_start = '''    convo::publishAtomic(m_ready, false, std::memory_order_release);

    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

    releaseAllLayers();'''

diag_prefix = '''    convo::publishAtomic(m_ready, false, std::memory_order_release);

    if (impulse == nullptr || irLen <= 0 || blockSize <= 0)
        return false;

#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    const uint64_t diagSeq = globalDiagSeq.fetch_add(1, std::memory_order_relaxed) + kDiagSeqFirstRuntime;
    const uint64_t beforeMkl = convo::diag::allocatedBytes();
    const uint32_t beforeLost = convo::diag::lostFreeCount();
#endif

    releaseAllLayers();'''

if setimpulse_start in content:
    content = content.replace(setimpulse_start, diag_prefix, 1)
    changes += 1

with open('src/MKLNonUniformConvolver.cpp', 'w', newline='') as f:
    f.write(content)

print(f'OK - {changes} changes applied')
print('DIAG_MKL_MALLOC replacements + allocSizes saves + IR_LOAD/IR_LAYOUT logs added')
