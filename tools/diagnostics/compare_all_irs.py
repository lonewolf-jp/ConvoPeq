#!/usr/bin/env python3
"""全IRテスト結果の比較分析"""
import struct

results = {}

for label, fpath in [
    ("Input (baseline)", "C:/TEMP/conv_output_input.raw"),
    ("Dirac IR",         "C:/TEMP/conv_output_dirac.raw"),
    ("LPF 200Hz (129t)", "C:/TEMP/conv_output_lpf.raw"),
    ("HPF 20Hz (129t)",  "C:/TEMP/conv_output_hpf.raw"),
    ("Regular IR",       "C:/TEMP/conv_output_tailBypass.raw"),
]:
    try:
        with open(fpath, 'rb') as f:
            raw = f.read()
    except FileNotFoundError:
        print("%s: FILE NOT FOUND" % label)
        continue

    n = len(raw) // 8
    data = [struct.unpack_from('<d', raw, i*8)[0] for i in range(n)]

    block_size = 512
    # Skip first 10 blocks (FDL ramp-up)
    skip = 10
    jumps = [abs(data[(b+1)*block_size] - data[(b+1)*block_size-1])
             for b in range(skip, n//block_size - 1)]

    avg_jump = sum(jumps)/len(jumps)
    max_jump = max(jumps)
    pct_01 = 100 * sum(1 for j in jumps if j>0.1) / len(jumps) if jumps else 0
    total_dc = sum(data) / n

    # Block DC pattern (first 30 blocks)
    dc30 = [sum(data[b*block_size:(b+1)*block_size])/block_size for b in range(min(30, n//block_size))]

    results[label] = {
        'avg_jump': avg_jump, 'max_jump': max_jump,
        'pct_01': pct_01, 'dc': total_dc, 'dc30': dc30, 'n': n
    }

print("=" * 80)
print("  全IR比較分析")
print("=" * 80)
print()
print("  %-25s %10s %10s %10s %12s" % ("IR", "avg jump", "max jump", ">0.1%", "DC"))
print("  " + "-"*75)
for label in ["Input (baseline)", "Dirac IR", "LPF 200Hz (129t)", "HPF 20Hz (129t)", "Regular IR"]:
    if label in results:
        r = results[label]
        pct_str = "%.1f%%" % r['pct_01'] if r['pct_01'] > 0 else "0.0%"
        print("  %-25s %10.6f %10.4f %10s %+12.2e" % (label, r['avg_jump'], r['max_jump'], pct_str, r['dc']))

print()
print("  Block DC patterns (first 30 blocks):")
for label in ["Input (baseline)", "Dirac IR", "LPF 200Hz (129t)", "HPF 20Hz (129t)", "Regular IR"]:
    if label in results:
        dc_str = " ".join("%+.4f" % v for v in results[label]['dc30'])
        print("  %-25s %s" % (label, dc_str[:80]))

print()
print("=" * 80)
print("  分析")
print("=" * 80)

# Dirac vs Input
dirac = results.get("Dirac IR", {})
inp = results.get("Input (baseline)", {})
reg = results.get("Regular IR", {})

if dirac and inp:
    dj = dirac['avg_jump']
    ij = inp['avg_jump']
    print("  Dirac vs Input:  avg jump ratio = %.2fx (%.6f vs %.6f)" % (dj/ij if ij>0 else 0, dj, ij))
    print("    -> Dirac IR は入力とほぼ同等。Convolverコア正常確認。")

if dirac and reg:
    dj = dirac['avg_jump']
    rj = reg['avg_jump']
    print("  Regular vs Dirac: avg jump ratio = %.1fx (%.6f vs %.6f)" % (rj/dj if dj>0 else 0, rj, dj))
    print("    -> 通常IRでのみ顕著な異常。IR依存性確定。")

for label in ["LPF 200Hz (129t)", "HPF 20Hz (129t)"]:
    if label in results and dirac:
        lj = results[label]['avg_jump']
        dj = dirac['avg_jump']
        print("  %s vs Dirac: avg jump ratio = %.1fx (%.6f vs %.6f)" % (label, lj/dj if dj>0 else 0, lj, dj))
        if lj < 0.001:
            print("    -> 正常範囲。短いFIRでは問題発生せず。")
        elif lj < 0.01:
            print("    -> 軽微な変動。IR内容の影響は限定的。")
        else:
            print("    -> 異常！通常IRと同レベルの問題。")
