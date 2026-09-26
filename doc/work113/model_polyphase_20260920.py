#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvoPeq B-1 polyphase gain 閉形式モデル（2026-09-20・v1.8 検証用）
CustomInputOversampler.cpp の prepareStage / interpolateStage / decimateStage を
厳密に Python へ移植し、remediation_plan v1.7 §2.7 の数値契約を独立再現する。
"""
import numpy as np
from scipy.special import i0 as besselI0

FS_IN = 192000.0

def prepare_coeffs(taps, atten_db):
    """CustomInputOversampler::prepareStage (cpp:287-390) の厳密移植"""
    stage_taps = max(3, taps | 1)
    center_tap = (stage_taps - 1) // 2
    center_parity = center_tap & 1
    conv_parity = 1 - center_parity
    if atten_db > 50.0:
        beta = 0.1102 * (atten_db - 8.7)
    elif atten_db >= 21.0:
        beta = 0.5842 * (atten_db - 21.0) ** 0.4 + 0.07886 * (atten_db - 21.0)
    else:
        beta = 0.0
    i0b = besselI0(beta)
    M = center_tap
    n = np.arange(stage_taps)
    t = (n - M).astype(float)
    den = np.where(n == M, 1.0, t)
    sinc = np.where(n == M, 0.5, np.sin(np.pi * 0.5 * t) / (np.pi * den))
    frac = t / M
    window = besselI0(beta * np.sqrt(np.maximum(0.0, 1.0 - frac ** 2))) / i0b
    raw = sinc * window
    for i in range(stage_taps):
        if i != center_tap and (i & 1) == center_parity:
            raw[i] = 0.0
    s = raw.sum()
    raw *= 1.0 / s
    raw[center_tap] = 0.5
    noncenter_sum = raw.sum() - raw[center_tap]
    scale = 0.5 / noncenter_sum
    for i in range(stage_taps):
        if i != center_tap:
            raw[i] *= scale
    raw[center_tap] = 0.5
    conv_count = (stage_taps - conv_parity + 1) // 2
    conv_coeffs = np.array([raw[conv_parity + 2 * r] if conv_parity + 2 * r < stage_taps else 0.0
                            for r in range(conv_count)])
    center_delay_input = (center_tap - center_parity) // 2
    return dict(raw=raw, taps=stage_taps, center_tap=center_tap, center_parity=center_parity,
                conv_parity=conv_parity, conv_count=conv_count, conv_coeffs=conv_coeffs,
                center_coeff=raw[center_tap], center_delay_input=center_delay_input,
                atten=atten_db)

def interpolate_stage(st, x, correct_gain=False):
    """interpolateStage (cpp:492-568) の厳密等価実装（ベクトル化）
    conv_value[n] = 2·Σ_k conv_coeffs[k]·x[n−k]（np.convolve と同一の畳み込み和）
    center_value[n] = centerCoeff·x[n−centerDelayInput]（n<cdi は 0 = 新規インスタンス history=0）
    出力 parity 配置: out[2n+convParity]=conv / out[2n+centerParity]=center
    """
    n_in = len(x)
    conv = 2.0 * np.convolve(x, st['conv_coeffs'])[:n_in]
    cdi = st['center_delay_input']
    center = st['center_coeff'] * np.concatenate([np.zeros(cdi), x])[:n_in]
    if correct_gain:
        center = center * 2.0
    out = np.zeros(2 * n_in)
    out[st['conv_parity']::2] = conv
    out[st['center_parity']::2] = center
    return out

def decimate_stage(st, x2):
    """decimateStage (cpp:570-723) の厳密等価実装（ベクトル化・ポリフェーズ分解）
    y[n] = Σ_r c[r]·x2[2(n−r)−convParity] + centerCoeff·x2[2n−centerTap]
    （n < 参照開始は新規インスタンス history=0 を反映して 0）
    """
    n_out = len(x2) // 2
    u = np.concatenate([np.zeros(1 if st['conv_parity'] else 0), x2[st['conv_parity']::2]])
    y = np.convolve(u, st['conv_coeffs'])[:n_out]
    v = np.concatenate([np.zeros(st['center_tap']), x2[st['center_tap']::2]])[:n_out]
    y += st['center_coeff'] * v
    return y
def chain_up(stages, x, correct_gain=False):
    y = x
    for st in stages:
        y = interpolate_stage(st, y, correct_gain)
    return y

def chain_down(stages, y):
    for st in reversed(stages):
        y = decimate_stage(st, y)
    return y

def db(x):
    return 20.0 * np.log10(np.maximum(np.abs(x), 1e-300))

PRESETS = {
    'IIRLike': [(511, 140.0), (127, 110.0), (31, 90.0)],
    'LinearPhase': [(1023, 160.0), (255, 140.0), (63, 120.0)],
}

def bandpass_edge(f, mags):
    """0.001 起点で |H|>=-0.1dB が初めて破れる前の最大周波数（Fs_in 基準）"""
    last_ok, dropped = 0.0, False
    for i in range(len(f)):
        if f[i] > 0.4999:
            break
        if f[i] >= 0.001:
            if mags[i] >= -0.1:
                if not dropped:
                    last_ok = f[i]
            else:
                dropped = True
    return last_ok

def part1():
    out = []
    A = out.append
    A("== 係数検証（全 6 design） ==")
    all_coeffs = {}
    for pname, spec in PRESETS.items():
        for i, (taps, atten) in enumerate(spec):
            st = prepare_coeffs(taps, atten)
            all_coeffs[f'{taps}/{int(atten)}'] = st
            A(f"  {taps}t/{int(atten)}dB: FIRsum={st['raw'].sum():.15f} "
              f"center={st['center_coeff']:.15f} convSum={st['conv_coeffs'].sum():.15f} "
              f"cTap={st['center_tap']} cPar={st['center_parity']} vPar={st['conv_parity']}")
    return out, all_coeffs

def spectrum(y, pad_bits=20):
    """丸め込みのない高分解能 DTFT（zero-pad FFT）。len(y) 基準で正規化周波数"""
    Nb = 1 << pad_bits
    H = np.fft.rfft(y, Nb)
    f = np.fft.rfftfreq(Nb)
    return f, np.abs(H)

def bandpass_edge2(f, mags, lo=0.001, hi=0.4999, thresh=-0.1):
    last_ok, dropped = 0.0, False
    for i in range(len(f)):
        if f[i] > hi:
            break
        if f[i] >= lo:
            if mags[i] >= thresh:
                if not dropped:
                    last_ok = f[i]
            else:
                dropped = True
    return last_ok

def part2(all_coeffs):
    out = []
    A = out.append
    A("== 1. DC round-trip（アルゴリズム厳密シミュレーション） ==")
    for pname, spec in PRESETS.items():
        for ratio, nst in ((2, 1), (4, 2), (8, 3)):
            stages = [prepare_coeffs(t, a) for (t, a) in spec[:nst]]
            x = np.ones(4096)
            yb = chain_down(stages, chain_up(stages, x, False))
            yc = chain_down(stages, chain_up(stages, x, True))
            A(f"  {pname} r={ratio}: base={yb[2000:].mean():.12f} (0.75^{nst}={0.75**nst:.12f}) "
              f"cand={yc[2000:].mean():.12f}")

    # v1.7 §2.7.1 の 3 構成（S1=単段31/90・IIR3・LP3）+ 単段 511/140 参照
    CONFIGS = [
        ('S1   (31/90)', [(31, 90.0)]),
        ('IIR3 (511/127/31)', PRESETS['IIRLike']),
        ('LP3  (1023/255/63)', PRESETS['LinearPhase']),
        ('511/140 single', [(511, 140.0)]),
    ]
    A("\n== 2. full-chain round-trip |H| (Fs_in 基準・exact DTFT) ==")
    freq_pts = [50.0, 1000.0, 10000.0, 0.25 * FS_IN, 0.45 * FS_IN, 0.49 * FS_IN]
    for cname, spec in CONFIGS:
        nst = len(spec)
        stages = [prepare_coeffs(t, a) for (t, a) in spec]
        total_taps = sum(s['taps'] for s in stages)
        L = 2 * total_taps + 4096
        x = np.zeros(L); x[0] = 1.0
        yb = chain_down(stages, chain_up(stages, x, False))
        yc = chain_down(stages, chain_up(stages, x, True))
        eff = total_taps + 2048
        fb, mb_lin = spectrum(yb[:eff])
        fc, mc_lin = spectrum(yc[:eff])
        assert np.array_equal(fb, fc)
        f = fb
        mb, mc = 20 * np.log10(np.maximum(mb_lin, 1e-300)), 20 * np.log10(np.maximum(mc_lin, 1e-300))
        A(f"  {cname} (N={nst}):")
        for fq in freq_pts:
            ib = int(round(fq / FS_IN * len(f) * 2))
            dev = mc[ib] - mb[ib] - 20 * np.log10((4.0 / 3.0) ** nst)
            A(f"    f={fq:9.1f}Hz({fq/FS_IN:.2f}Fs): base={mb[ib]:8.4f}dB cand={mc[ib]:8.4f}dB dev={dev:+.4f}dB")
        for lo, hi in ((0.005, 0.45), (0.01, 0.30)):
            m = (f >= lo) & (f <= hi)
            A(f"    ripple[{lo},{hi}]: base={mb[m].max()-mb[m].min():.3f}dB "
              f"cand={mc[m].max()-mc[m].min():.3f}dB")
        A(f"    0.1dB edge: base={bandpass_edge2(f, mb):.4f} cand={bandpass_edge2(f, mc):.4f} Fs_in")
    return out

def part3(all_coeffs):
    out = []
    A = out.append
    A("\n== 3. per-design 絶対基準（own-rate cycles/sample・exact DTFT） ==")
    for key, st in all_coeffs.items():
        h = st['raw']
        f, mags_lin = spectrum(h)
        mags = 20 * np.log10(np.maximum(mags_lin, 1e-300))
        edge = bandpass_edge2(f, mags, lo=0.0005)
        A_target = -(st['atten'] - 3.0)
        m = (f >= 0.0005) & (f <= 0.5)
        fm, mm = f[m], mags[m]
        below = fm[mm <= A_target]
        t_end = below[0] if len(below) else 0.0
        m_stop = fm >= t_end
        stop_min = mm[m_stop].min()
        le = []
        for fq in (0.26, 0.30, 0.35, 0.45, 0.474):
            i = np.searchsorted(fm, fq)
            le.append(f"{fq}:{mm[i]:.1f}")
        A(f"  {key}: edge={edge:.4f} t_end={t_end:.4f} stopmin={stop_min:.1f}dB "
          f"floor={st['atten']-10:.0f} | " + " ".join(le))
    return out

def tone_rej(y, fhat, exclude_bw=0.002):
    """出力スペクトルで目的トーン f̂ と最大スパー（0.5−f̂ 鏡像含む）を測定"""
    f, mag = spectrum(y)
    ib = np.searchsorted(f, fhat)
    tone = mag[ib]
    # 除外帯域: DC・トーン近傍
    m = (f >= 0.0005) & (np.abs(f - fhat) > exclude_bw)
    spur_f = f[m][np.argmax(mag[m])]
    spur = mag[m].max()
    mirror_i = np.searchsorted(f, 0.5 - fhat) if fhat < 0.49 else None
    mirror = mag[mirror_i] if mirror_i else float('nan')
    return 20*np.log10(tone), 20*np.log10(spur/tone), spur_f, 20*np.log10(mirror/tone)

def d1_corrected(y, fhat, N):
    """D1 補正軸（v2.2 §2.7.3 / R8-1 authoritative）.

    up 出力 y の長さは 2N・標本レートは 2*Fs。
    入力 f̂ は up レートでは f̂/2。tone bin = f̂·N、image bin = N−f̂·N。
    """
    Y = np.abs(np.fft.fft(y))
    tone_bin = int(round(fhat * N))
    image_bin = int(round((1.0 - fhat) * N))  # = N - f̂·N
    # argmax 検証: tone 近傍 ±8 bin で最大 bin を確認
    lo = max(1, tone_bin - 8)
    hi = min(len(Y) - 1, tone_bin + 9)
    argmax_tone = int(np.argmax(Y[lo:hi])) + lo
    lo_i = max(1, image_bin - 8)
    hi_i = min(len(Y) - 1, image_bin + 9)
    argmax_image = int(np.argmax(Y[lo_i:hi_i])) + lo_i
    tone = Y[argmax_tone]
    image = Y[argmax_image]
    d1_db = 20.0 * np.log10(image / (tone + 1e-300))
    return d1_db, tone_bin, image_bin, argmax_tone, argmax_image


def d1_old_axis_discarded(y, fhat, N):
    """旧軸 D1（R7-1/R8-1 破棄済・参考のみ）. f̂ vs 0.5−f̂ を 2x レートの bin で読む誤定義."""
    Y = np.abs(np.fft.fft(y))
    ib = int(round(fhat * 2 * N)); ii = int(round((0.5 - fhat) * 2 * N))
    return 20 * np.log10(Y[ii] / (Y[ib] + 1e-300))


def part4(all_coeffs):
    out = []
    A = out.append
    A("\n== 4. image rejection 測定（3 定義・独立確定） ==")
    A("  D1 AUTHORITATIVE (corrected axis, v2.2 §2.7.3 / R8-1):")
    A("    up 単段出力（2x rate）: tone bin = f̂·N / image bin = N−f̂·N + argmax 検証")
    A("  D1 OLD AXIS (discarded, R7-1/R8-1): 旧定義 f̂ vs 0.5−f̂（v1.7 相当・解釈破棄済）")
    A("  D2: 単段 round-trip 出力（output rate）で f̂ vs 0.5−f̂")
    A("  D3: full-chain（S1/IIR3/LP3）出力の worst in-band spur")

    # D1/D2: 6 design × f̂ 7 点
    for key in ('31/90', '127/110', '511/140', '63/120', '255/140', '1023/160'):
        st = all_coeffs[key]
        r1b, r1c, r1ob, r1oc, r2b, r2c = [], [], [], [], [], []
        verify_lines = []
        for fhat in (0.05, 0.10, 0.20, 0.30, 0.35, 0.40, 0.45):
            N = 32768
            n = np.arange(N)
            x = np.sin(2 * np.pi * fhat * n)
            yb = interpolate_stage(st, x, False)
            yc = interpolate_stage(st, x, True)
            # D1 corrected: tone/image bin + argmax
            d1b, tb, ib, atb, aib = d1_corrected(yb, fhat, N)
            d1c, tc, ic, atc, aic = d1_corrected(yc, fhat, N)
            r1b.append(f"{fhat}:{d1b:.2f}")
            r1c.append(f"{fhat}:{d1c:.2f}")
            verify_lines.append(
                f"{fhat}: base tone_bin={tb} image_bin={ib} argmax_t={atb} argmax_i={aib} | "
                f"cand tone_bin={tc} image_bin={ic} argmax_t={atc} argmax_i={aic}"
            )
            # D1 old axis (discarded reference)
            r1ob.append(f"{fhat}:{d1_old_axis_discarded(yb, fhat, N):.2f}")
            r1oc.append(f"{fhat}:{d1_old_axis_discarded(yc, fhat, N):.2f}")
            # D2: 単段 round-trip（output rate・過渡トリム + Hann 窓）
            zb = decimate_stage(st, yb)
            zc = decimate_stage(st, yc)
            trim = st['taps'] + 256
            def d2(z):
                zz = z[trim:] * np.hanning(len(z) - trim)
                Z = np.abs(np.fft.fft(zz))
                ib = int(round(fhat * N)); ii = int(round((0.5 - fhat) * N))
                return 20 * np.log10(Z[ii] / (Z[ib] + 1e-300))
            r2b.append(f"{fhat}:{d2(zb):.2f}")
            r2c.append(f"{fhat}:{d2(zc):.2f}")
        A(f"  {key} D1 base : " + " ".join(r1b))
        A(f"  {key} D1 cand : " + " ".join(r1c))
        A(f"  {key} D1-ARGMAX : " + " | ".join(verify_lines))
        A(f"  {key} D1 OLD AXIS base : " + " ".join(r1ob))
        A(f"  {key} D1 OLD AXIS cand : " + " ".join(r1oc))
        A(f"  {key} D2 base : " + " ".join(r2b))
        A(f"  {key} D2 cand : " + " ".join(r2c))

    # D3: full-chain
    for cname, spec in (('S1', [(31, 90.0)]), ('IIR3', PRESETS['IIRLike']), ('LP3', PRESETS['LinearPhase'])):
        stages = [prepare_coeffs(t, a) for (t, a) in spec]
        res = []
        for fhat in (0.05, 0.10, 0.20, 0.30):
            N = 16384
            n = np.arange(N)
            x = np.sin(2 * np.pi * fhat * n)
            for label, cg in (('base', False), ('cand', True)):
                y = chain_down(stages, chain_up(stages, x, cg))
                trim = sum(s['taps'] for s in stages) + 512
                ytrim = y[trim:] * np.hanning(len(y) - trim)
                tone_db, spur_rel_db, spur_f, mirror_rel_db = tone_rej(ytrim, fhat)
                res.append(f"{fhat}:{label} spur={spur_rel_db:.1f}dB@{spur_f:.3f} mirror(0.5-f)={mirror_rel_db:.1f}dB")
        A(f"  D3 {cname}: " + " | ".join(res))
    return out

if __name__ == '__main__':
    o1, coeffs = part1()
    print("\n".join(o1))
    print("\n".join(part2(coeffs)))
    print("\n".join(part3(coeffs)))
    print("\n".join(part4(coeffs)))