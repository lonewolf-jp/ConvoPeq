import struct

files = [
    ("Input (test tone)", "C:/TEMP/conv_output_input.raw"),
    ("Convolver (tailBypass)", "C:/TEMP/conv_output_tailBypass.raw"),
]

for name, fpath in files:
    with open(fpath, 'rb') as f:
        raw = f.read()
    n = len(raw) // 8
    data = [struct.unpack_from('<d', raw, i * 8)[0] for i in range(n)]

    block_size = 512
    # Skip first 10 blocks (FDL ramp-up silence)
    start_block = 10
    start_sample = start_block * block_size
    samples_to_analyze = min(200 * block_size, n - start_sample)
    end_sample = start_sample + samples_to_analyze

    data_range = data[start_sample:end_sample]

    jumps = [abs(data_range[i] - data_range[i-1]) for i in range(1, len(data_range))]
    block_jumps = [abs(data[(b+1)*block_size] - data[(b+1)*block_size - 1])
                   for b in range(start_block, min(start_block + 200, n // block_size - 1))]

    print(f"=== {name} ===")
    print(f"  Samples analyzed: {len(data_range)} (from sample {start_sample})")
    print(f"  Overall avg diff (adjacent samples): {sum(jumps)/len(jumps):.6f}")
    print(f"  Block boundary avg jump: {sum(block_jumps)/len(block_jumps):.6f}")
    print(f"  Block boundary max jump: {max(block_jumps):.4f}")
    print(f"  Block boundary jumps > 0.1: {100*sum(1 for j in block_jumps if j>0.1)/len(block_jumps):.1f}%")
    print()
