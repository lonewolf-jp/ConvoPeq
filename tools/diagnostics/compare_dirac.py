"""Dirac IR vs 通常IR 比較分析"""
import struct

def analyze(path, label, skip_blocks=10):
    with open(path, 'rb') as f:
        raw = f.read()
    n = len(raw) // 8
    data = [struct.unpack_from('<d', raw, i*8)[0] for i in range(n)]

    block_size = 512
    start = skip_blocks * block_size
    end = ((n // block_size) - 1) * block_size

    # Block boundary jumps (steady state)
    jumps = [abs(data[(b+1)*block_size] - data[(b+1)*block_size-1])
             for b in range(skip_blocks, n//block_size - 1)]

    # First 30 block DC
    dc30 = [sum(data[b*block_size:(b+1)*block_size])/block_size for b in range(min(30, n//block_size))]

    print("=== %s ===" % label)
    print("  Samples: %d (%.2fs)" % (n, n/48000))
    print("  Block boundary avg jump: %.6f" % (sum(jumps)/len(jumps)))
    print("  Block boundary max jump: %.4f" % max(jumps))
    print("  Jumps > 0.1: %d/%d (%.1f%%)" % (sum(1 for j in jumps if j>0.1), len(jumps), 100*sum(1 for j in jumps if j>0.1)/len(jumps)))
    print("  Total DC: %+.6e" % (sum(data)/n))
    print("  First 30 block DC pattern:")
    print("    " + " ".join("%+.4f" % v for v in dc30))
    print()

analyze("C:/TEMP/conv_output_input.raw", "Input (test tone only)")
analyze("C:/TEMP/conv_output_dirac.raw", "Dirac IR (no tail bypass)")
analyze("C:/TEMP/conv_output_tailBypass.raw", "Regular IR (tail bypass)")
