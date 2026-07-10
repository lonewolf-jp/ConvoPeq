#!/usr/bin/env python3
"""通常IRファイル (impulse.wav) の詳細解析"""

import struct, math, sys

PATH = "C:/Users/user/Documents/conv_filter/impulse.wav"

with open(PATH, 'rb') as f:
    raw = f.read()

print("=" * 65)
print("  IRファイル詳細解析: impulse.wav")
print("=" * 65)
print("  ファイルサイズ:", len(raw), "bytes")
print()

# ── WAV ヘッダ解析 ──
print("--- WAV Header ---")
chunk_id = raw[0:4]
file_size = struct.unpack_from('<I', raw, 4)[0]
wave_id = raw[8:12]
print("  RIFF:", chunk_id)
print("  File size (from header):", file_size)
print("  WAVE:", wave_id)

# fmt chunk
fmt_id = raw[12:16]
fmt_size = struct.unpack_from('<I', raw, 16)[0]
audio_format = struct.unpack_from('<H', raw, 20)[0]
num_channels = struct.unpack_from('<H', raw, 22)[0]
sample_rate = struct.unpack_from('<I', raw, 24)[0]
byte_rate = struct.unpack_from('<I', raw, 28)[0]
block_align = struct.unpack_from('<H', raw, 32)[0]
bits_per_sample = struct.unpack_from('<H', raw, 34)[0]
print("  fmt chunk size:", fmt_size)
print("  Audio format:", audio_format, "(3 = IEEE float)")
print("  Channels:", num_channels)
print("  Sample rate:", sample_rate)
print("  Byte rate:", byte_rate)
print("  Block align:", block_align)
print("  Bits per sample:", bits_per_sample)
print("  Format extension size:", fmt_size - 16)

# Extended fmt info for PCMFORAMTEXTENSIBLE (format 0xFFFE)
if fmt_size > 16:
    ext_size = struct.unpack_from('<H', raw, 36)[0]
    if ext_size >= 22:
        valid_bits = struct.unpack_from('<H', raw, 38)[0]
        channel_mask = struct.unpack_from('<I', raw, 40)[0]
        sub_format = raw[44:60]
        print("  Valid bits per sample:", valid_bits)
        print("  Channel mask: 0x%08X" % channel_mask)
        print("  Subformat:", sub_format.hex())
    elif ext_size >= 0:
        print("  Extension data at offset 36, size:", ext_size)
        print("  Raw ext:", raw[36:36+ext_size].hex())

# Parse chunks more carefully
print()
print("--- Chunks ---")
pos = 12  # after RIFF header
while pos < len(raw) - 8:
    ck_id = raw[pos:pos+4]
    ck_size = struct.unpack_from('<I', raw, pos+4)[0]
    ck_name = ck_id.decode('latin-1').strip()
    if ck_name == '':
        # Skip padding bytes
        pos += 1
        continue
    print("  '%s' at %d, size=%d" % (ck_name, pos, ck_size))
    if ck_id == b'data':
        data_start = pos + 8
        data_size = ck_size
        print("    => DATA at offset %d, %d bytes" % (data_start, data_size))
        break
    pos += 8 + ck_size
    if ck_size % 2:  # word alignment
        pos += 1

# ── Raw bytes of first sample ──
print()
print("--- First Sample Raw Bytes (L) ---")
first_4_bytes = raw[data_start:data_start+4]
print("  Hex:", first_4_bytes.hex())
print("  Binary:", ' '.join(format(b, '08b') for b in first_4_bytes))
# IEEE 754 single-precision interpretation
sign_bit = (first_4_bytes[3] & 0x80) >> 7
exponent = ((first_4_bytes[3] & 0x7F) << 1) | ((first_4_bytes[2] & 0x80) >> 7)
mantissa = ((first_4_bytes[2] & 0x7F) << 16) | (first_4_bytes[1] << 8) | first_4_bytes[0]
print("  IEEE754 decomposition:")
print("    Sign:", sign_bit)
print("    Exponent bits:", exponent)
print("    Mantissa bits: 0x%06X" % mantissa)
if exponent == 0xFF and mantissa == 0:
    val_str = "-Inf" if sign_bit else "+Inf"
    print("  => %s (Infinity)" % val_str)
elif exponent == 0xFF and mantissa != 0:
    print("  => NaN (Not a Number) - mantissa=0x%06X" % mantissa)
elif exponent == 0:
    print("  => Denormalized/subnormal: %e" % struct.unpack_from('<f', first_4_bytes)[0])
else:
    val = struct.unpack_from('<f', first_4_bytes)[0]
    print("  => Normalized: %e" % val)
    decoded = ((-1)**sign_bit) * (2.0**(exponent-127)) * (1.0 + mantissa/8388608.0)
    print("  => Manual decode: %e" % decoded)

# First sample R channel
first_4_bytes_r = raw[data_start+4:data_start+8]
print()
print("--- First Sample Raw Bytes (R) ---")
print("  Hex:", first_4_bytes_r.hex())
val_r = struct.unpack_from('<f', first_4_bytes_r)[0]
print("  Value: %e" % val_r)

# ── Full IR data ──
print()
print("--- Full IR Statistics ---")
n_frames = data_size // block_align
n = min(n_frames, data_size // block_align)
ir_l = []
ir_r = []
for i in range(n):
    off = data_start + i * block_align
    l = struct.unpack_from('<f', raw, off)[0]
    r = struct.unpack_from('<f', raw, off+4)[0]
    ir_l.append(l)
    ir_r.append(r)

print("  Total frames: %d (%.3fs @ %dHz)" % (n, n/sample_rate, sample_rate))

# NaN/Inf count
nan_l = sum(1 for v in ir_l if math.isnan(v))
inf_l = sum(1 for v in ir_l if math.isinf(v))
nan_r = sum(1 for v in ir_r if math.isnan(v))
inf_r = sum(1 for v in ir_r if math.isinf(v))
print("  NaN: L=%d, R=%d" % (nan_l, nan_r))
print("  Inf: L=%d, R=%d" % (inf_l, inf_r))

# First 50 samples
print()
print("--- First 50 samples ---")
print("  %5s  %15s %15s" % ("idx", "L", "R"))
for i in range(min(50, n)):
    print("  %5d  %15.6f %15.6f" % (i, ir_l[i], ir_r[i]))

# Energy distribution: find where 90% energy is
energy_l = sum(v*v for v in ir_l if not math.isnan(v) and not math.isinf(v))
energy_r = sum(v*v for v in ir_r if not math.isnan(v) and not math.isinf(v))
cum_l = 0
cum_r = 0
t90_l = 0
t90_r = 0
for i in range(n):
    if not math.isnan(ir_l[i]) and not math.isinf(ir_l[i]):
        cum_l += ir_l[i]*ir_l[i]
    if not math.isnan(ir_r[i]) and not math.isinf(ir_r[i]):
        cum_r += ir_r[i]*ir_r[i]
    if energy_l > 0 and cum_l >= 0.9 * energy_l and t90_l == 0:
        t90_l = i
    if energy_r > 0 and cum_r >= 0.9 * energy_r and t90_r == 0:
        t90_r = i
print()
print("--- Energy Distribution ---")
print("  Total energy: L=%e, R=%e" % (energy_l, energy_r))
print("  90%% energy at sample: L=%d (%.1fms), R=%d (%.1fms)" % (t90_l, t90_l/sample_rate*1000, t90_r, t90_r/sample_rate*1000))

# Peak and RMS (excluding NaN/Inf)
valid_l = [v for v in ir_l if not math.isnan(v) and not math.isinf(v)]
valid_r = [v for v in ir_r if not math.isnan(v) and not math.isinf(v)]
print()
print("--- Statistics (valid samples only) ---")
print("  L: peak=%.4f, RMS=%.6f, DC=%+.6e" % (
    max(abs(v) for v in valid_l) if valid_l else 0,
    math.sqrt(sum(v*v for v in valid_l)/len(valid_l)) if valid_l else 0,
    sum(valid_l)/len(valid_l) if valid_l else 0
))
print("  R: peak=%.4f, RMS=%.6f, DC=%+.6e" % (
    max(abs(v) for v in valid_r) if valid_r else 0,
    math.sqrt(sum(v*v for v in valid_r)/len(valid_r)) if valid_r else 0,
    sum(valid_r)/len(valid_r) if valid_r else 0
))

# Energy in frequency bands (simple DFT approximation)
print()
print("--- Spectral analysis (first 4096 valid samples, L channel) ---")
fft_n = min(4096, len(valid_l))
window = [0.5 * (1.0 - math.cos(2.0*math.pi*i/(fft_n-1))) for i in range(fft_n)]
if len(valid_l) >= fft_n:
    chunk = valid_l[:fft_n]
    windowed = [chunk[i]*window[i] for i in range(fft_n)]
    # DFT
    spec = []
    for k in range(fft_n//2):
        re, im = 0.0, 0.0
        for t in range(fft_n):
            a = -2.0*math.pi*k*t/fft_n
            re += windowed[t]*math.cos(a)
            im += windowed[t]*math.sin(a)
        spec.append(math.sqrt(re*re+im*im))
    max_s = max(spec)
    spec_db = [20*math.log10(s/max_s+1e-12) for s in spec] if max_s > 0 else [-200]*len(spec)
    bin_hz = sample_rate / fft_n
    print("  Bin width: %.2f Hz" % bin_hz)
    print("  Dominant frequencies (>-20dB):")
    for k in range(len(spec)):
        if spec_db[k] > -20:
            print("    %6.1f Hz: %+.1f dB" % (k*bin_hz, spec_db[k]))

# Check if this is likely a room IR, synthetic IR, or filter
print()
print("--- IR Type Assessment ---")
early_count = sum(1 for v in ir_l[:50] if abs(v) > 0.01)
has_direct = any(abs(v) > 0.1 for v in ir_l[:100])
if nan_l > 0 or inf_l > 0:
    print("  ⚠️  IR contains NaN/Inf values! Likely corrupted file.")
elif early_count < 5 and has_direct:
    print("  Likely: Room Impulse Response (sparse early reflections)")
elif early_count > 10 and has_direct:
    print("  Likely: Synthetic IR or Filter")
else:
    print("  Indeterminate or near-silent IR")
