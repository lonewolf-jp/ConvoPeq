#!/usr/bin/env python3
"""Verify numerical equivalence of old vs new TanhApprox formulas."""
import math

def old_tanh(x):
    """DSPCoreDouble original TanhApprox formula"""
    if x >= 4.5: return 1.0
    if x <= -4.5: return -1.0
    x2 = x*x
    NUM_A, NUM_B, NUM_C = 10395.0, 1260.0, 21.0
    DEN_A, DEN_B, DEN_C = 10395.0, 4725.0, 210.0
    num = x * (NUM_A + x2 * (NUM_B + x2 * NUM_C))
    den = DEN_A + x2 * (DEN_B + x2 * (DEN_C + x2))
    return num / den

def new_tanh(x):
    """SoftClipPadéPolicy-based formula"""
    CT = 4.5
    NumA, NumB, NumC = 10395.0, 1260.0, 21.0
    DenA, DenB, DenC = 10395.0, 4725.0, 210.0
    if x >= CT: return 1.0
    if x <= -CT: return -1.0
    x2 = x*x
    num = x * (NumA + x2 * (NumB + x2 * NumC))
    den = DenA + x2 * (DenB + x2 * (DenC + x2))
    return num / den

def fastTanh_clip(x):
    """How fastTanV256 clips first, then computes"""
    CT = 4.5
    x_clamped = max(min(x, CT), -CT)
    x2 = x_clamped * x_clamped
    NumA, NumB, NumC = 10395.0, 1260.0, 21.0
    DenA, DenB, DenC = 10395.0, 4725.0, 210.0
    num = x_clamped * (NumA + x2 * (NumB + x2 * NumC))
    den = DenA + x2 * (DenB + x2 * (DenC + x2))
    return num / den

print("=" * 60)
print("R-3: Numerical Verification — SoftClipPadéPolicy 10395 formula")
print("=" * 60)
test_vals = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 4.5, 10.0, -0.5, -4.0, -4.5]
max_diff_old = 0.0
max_diff_clip = 0.0
for v in test_vals:
    o = old_tanh(v)
    n = new_tanh(v)
    c = fastTanh_clip(v)
    diff_old = abs(o - n)
    diff_clip = abs(o - c)
    max_diff_old = max(max_diff_old, diff_old)
    max_diff_clip = max(max_diff_clip, diff_clip)
    s_old = "OK" if diff_old < 1e-15 else "MISMATCH"
    s_clip = "OK" if diff_clip < 1e-15 else "MISMATCH"
    print(f"  x={v:6.2f}: old={o:.15f} new={n:.15f} diff(old-new)={diff_old:.2e} [{s_old}]")
    print(f"           clip-then-compute={c:.15f} diff(old-clip)={diff_clip:.2e} [{s_clip}]")
print(f"\nMax diff (old vs new, direct): {max_diff_old:.2e}  {'PASS' if max_diff_old < 1e-15 else 'FAIL'}")
print(f"Max diff (old vs new, clip-first): {max_diff_clip:.2e}  {'PASS' if max_diff_clip < 1e-15 else 'FAIL'}")

print()
print("=" * 60)
print("R-3: DefaultFastTanhPolicy 27/9 formula — coefficient decomposition check")
print("=" * 60)

def pade27_compute(x):
    """DefaultFastTanhPolicy::compute() - 27/9 Padé"""
    x2 = x*x
    return x * (27.0 + x2) / (27.0 + 9.0 * x2)

def pade27_coeff(x):
    """27/9 formula using NumA/B/C, DenA/B/C decomposition"""
    CT, NumA, NumB, NumC = 4.5, 27.0, 1.0, 0.0
    DenA, DenB, DenC = 27.0, 9.0, 0.0
    if x >= CT: return 1.0
    if x <= -CT: return -1.0
    x2 = x*x
    num = x * (NumA + x2 * (NumB + x2 * NumC))
    den = DenA + x2 * (DenB + x2 * (DenC + x2))
    return num / den

max_diff = 0.0
for v in test_vals:
    o = pade27_compute(v)
    n = pade27_coeff(v)
    diff = abs(o - n)
    max_diff = max(max_diff, diff)
    status = "OK" if diff < 1e-15 else "MISMATCH"
    print(f"  x={v:6.2f}: compute={o:.15f} coeff={n:.15f} diff={diff:.2e} [{status}]")
print(f"\nMax diff: {max_diff:.2e}  {'PASS' if max_diff < 1e-15 else 'FAIL'}")

print()
print("=" * 60)
print("Edge case: AVX2 clamp-then-compute vs scalar branch-then-compute")
print("=" * 60)
# For x > ClipThreshold: scalar returns 1.0, AVX2 clamps and computes
# Are they the same? Yes, 10395*4.5/(...) ≈ 0.99927, not exactly 1.0
# So there IS a subtle difference between branch-then-return-1.0 and clamp-then-compute
edge_vals = [4.5, 4.6, 10.0, -4.5, -4.6, -10.0]
for v in edge_vals:
    scalar = old_tanh(v)  # branch: returns ±1.0 for |x|>=4.5
    avx2 = fastTanh_clip(v)  # clamp: computes tanh(clamped(x))
    diff = abs(scalar - avx2)
    print(f"  x={v:6.2f}: scalar(branch)={scalar:.10f} avx2(clamp)={avx2:.10f}  diff={diff:.2e}")
    if diff > 1e-12:
        print(f"    *** NOTE: This difference exists in the ORIGINAL code too (pre-R-3)")
        print(f"    *** The original DSPCoreDouble softClipBlockAVX2 also clamps before computing")
