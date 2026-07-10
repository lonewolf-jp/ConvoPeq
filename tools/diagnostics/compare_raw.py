import struct

paths = [
    ('bypass', 'C:/TEMP/conv_output_bypass.raw'),
    ('tail_bypass', 'C:/TEMP/conv_output_tailBypass.raw'),
    ('tail_enabled', 'C:/TEMP/conv_output_work52_fail.raw'),
]

# Load all data
all_data = {}
for name, fpath in paths:
    with open(fpath, 'rb') as f:
        raw = f.read()
    all_data[name] = [struct.unpack_from('<d', raw, i*8)[0] for i in range(len(raw)//8)]
    print(f"{name}: {len(all_data[name])} samples")

# Compare pairwise
print()
for i, (n1, _) in enumerate(paths):
    for j, (n2, _) in enumerate(paths):
        if i < j:
            d1 = all_data[n1]
            d2 = all_data[n2]
            min_len = min(len(d1), len(d2))
            diff_count = sum(1 for k in range(min_len) if d1[k] != d2[k])
            max_diff = max(abs(d1[k] - d2[k]) for k in range(min_len))
            print(f"{n1} vs {n2}: diff_count={diff_count}/{min_len}, max_diff={max_diff:.2e}")

            if diff_count == 0:
                print("  -> COMPLETELY IDENTICAL!")
            elif max_diff < 1e-15:
                print("  -> Effectively identical (rounding)")
            else:
                # Find first few differing positions
                first_diffs = [(k, d1[k], d2[k], abs(d1[k]-d2[k])) for k in range(min_len) if d1[k] != d2[k]][:5]
                for pos, v1, v2, diff in first_diffs:
                    print(f"  Diff at {pos}: {v1:.6f} vs {v2:.6f} (diff={diff:.2e})")
