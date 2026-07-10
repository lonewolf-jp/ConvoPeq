"""
compare_noiseshaper_patterns.py
================================
LatticeNoiseShaper の advanceState() における2つの state convention を比較する。

Pattern A (CMSIS-DSP型 / P7前):
  state[i] = prev_backward  (前段の backward を保存)

Pattern B (Ne10型 / P7後 / 現行):
  state[i] = nextBackward   (自段の backward を保存)

ConvoPeq 実装の processSample() / quantize() / computeFeedback() を
忠実に再現した上で、両パターンの出力差を評価する。

Usage:
  cd C:\\VSC_Project\\ConvoPeq
  python tools\\analysis\\compare_noiseshaper_patterns.py

出力:
  - 各テスト条件の数値結果
  - 再現可能な全パラメータ明示
"""

import os
import sys
import numpy as np

# ============================================================================
# 定数
# ============================================================================
KORDER = 9                         # LatticeNoiseShaper::kOrder
KNUMCHANNELS = 2                    # LatticeNoiseShaper::kNumChannels
KLATTICE_STATE_LIMIT = 2.0          # advanceState 内部 clamp 限界
KSTATE_LIMIT = 1.0e12               # clampStateSIMD 限界
KCOEFF_LIMIT = 0.85                 # clampCoeff 限界

# ConvoPeq デフォルト係数 (src/audioengine/AudioEngine.Learning.cpp:286)
DEFAULT_COEFFS = np.array([
    0.82, -0.68, 0.55, -0.43, 0.33, -0.25, 0.18, -0.12, 0.07
], dtype=np.float64)

# 評価用サンプル数
NUM_SAMPLES_SHORT = 200     # インパルス応答用
NUM_SAMPLES_LONG = 20000    # 無音/定常評価用
NUM_SAMPLES_SETTLE = 500    # 整定待ち

# RNG シード (再現性保証)
RNG_SEED = 42


# ============================================================================
# Xoshiro256++ RNG — LatticeNoiseShaper の実装と同一
# ============================================================================
class Xoshiro256State:
    """Xoshiro256++ 1.0 (周期 2^256-1) — 完全移植"""
    def __init__(self, seed_state):
        self.s = np.array(seed_state, dtype=np.uint64)

    @staticmethod
    def rotl(x, k):
        return ((x << k) | (x >> (64 - k))) & 0xFFFFFFFFFFFFFFFF

    def next(self):
        result = (self.rotl(self.s[0] + self.s[3], 23) + self.s[0]) & 0xFFFFFFFFFFFFFFFF
        t = (self.s[1] << 17) & 0xFFFFFFFFFFFFFFFF
        self.s[2] ^= self.s[0]
        self.s[3] ^= self.s[1]
        self.s[1] ^= self.s[2]
        self.s[0] ^= self.s[3]
        self.s[2] ^= t
        self.s[3] = self.rotl(self.s[3], 45)
        return result


# ============================================================================
# NoiseShaper シミュレータ — ConvoPeq 実装を忠実再現
# ============================================================================
class LatticeNoiseShaperSim:
    """
    ConvoPeq の LatticeNoiseShaper を Python で再現。

    Parameter:
      use_nextbackward: True = Pattern B (Ne10型/現行)
                        False = Pattern A (CMSIS-DSP型/P7前)
      bit_depth: 量子化ビット数 (16, 24, 32)
      dither_enabled: TPDF ディザの有無
    """
    def __init__(self, use_nextbackward: bool, bit_depth: int = 24,
                 dither_enabled: bool = True):
        self.use_nb = use_nextbackward
        self.bit_depth = bit_depth
        self.dither_enabled = dither_enabled

        # scale/invScale (LatticeNoiseShaper::prepare と同一)
        safe_bits = max(1, min(bit_depth, 32))
        self.inv_scale = np.ldexp(1.0, safe_bits - 1)
        self.scale = 1.0 / self.inv_scale

        # 状態
        self.states = np.zeros((KNUMCHANNELS, KORDER), dtype=np.float64)

        # RNG (Xoshiro256++ — LatticeNoiseShaper と同一初期値)
        self.rng_states = [
            Xoshiro256State([0x123456789ABCDEF0, 0xFEDCBA9876543210,
                             0x0123456789ABCDEF, 0xEFCDAB8967452301]),
            Xoshiro256State([0x89ABCDEF01234567, 0x76543210FEDCBA98,
                             0xABCDEF0123456789, 0x67452301EFCDAB89])
        ]

    def reset(self):
        """LatticeNoiseShaper::reset と同一"""
        self.states.fill(0.0)

    def uniform(self, ch: int) -> float:
        """LatticeNoiseShaper::uniform と同一"""
        return (self.rng_states[ch].next() >> 11) * (1.0 / 9007199254740992.0)

    def quantize(self, value: float, ch: int) -> float:
        """
        LatticeNoiseShaper::quantize と同一。
        1) クランプ to [-1.0, 1.0 - 1/invScale]
        2) TPDF dither (条件付き)
        3) round(value * invScale) / invScale
        """
        min_val = -1.0
        max_val = 1.0 - (1.0 / self.inv_scale)
        value = max(min_val, min(max_val, value))

        if self.dither_enabled:
            u1 = self.uniform(ch)
            u2 = self.uniform(ch)
            value += (u1 + u2 - 1.0) * self.scale

        # round() — SSE _mm_round_sd(_, _MM_FROUND_TO_NEAREST_INT)
        rounded = np.round(value * self.inv_scale)
        return rounded * self.scale

    def compute_feedback(self, channel_state: np.ndarray,
                         coeffs: np.ndarray) -> float:
        """
        LatticeNoiseShaper::computeFeedback と同一。
        feedback = sum(state[i] * coeffs[i]) (SIMD horizontal add)
        """
        return float(np.dot(channel_state, coeffs))

    def advance_state(self, channel_state: np.ndarray, error: float,
                      coeffs: np.ndarray):
        """
        LatticeNoiseShaper::advanceState と同一。

        Pattern A (CMSIS-DSP型 / P7前):
          state[i] = clamp(prev_backward, -2.0, 2.0)

        Pattern B (Ne10型 / P7後 / 現行):
          state[i] = clamp(nextBackward, -2.0, 2.0)
        """
        forward = error
        prev_backward = error
        for i in range(KORDER):
            backward = channel_state[i]
            next_forward = forward + coeffs[i] * backward
            next_backward = coeffs[i] * forward + backward

            if self.use_nb:
                # Pattern B (Ne10型 / P7後 / 現行)
                channel_state[i] = np.clip(next_backward,
                                           -KLATTICE_STATE_LIMIT,
                                           KLATTICE_STATE_LIMIT)
            else:
                # Pattern A (CMSIS-DSP型 / P7前)
                channel_state[i] = np.clip(prev_backward,
                                           -KLATTICE_STATE_LIMIT,
                                           KLATTICE_STATE_LIMIT)

            forward = next_forward
            prev_backward = next_backward

    def clamp_state_simd(self, channel_state: np.ndarray):
        """
        LatticeNoiseShaper::clampStateSIMD と同一。
        最終的な状態を ±1e12 にクランプ。
        """
        np.clip(channel_state, -KSTATE_LIMIT, KSTATE_LIMIT, out=channel_state)

    def process_sample(self, ch: int, x: float,
                       coeffs: np.ndarray) -> float:
        """
        LatticeNoiseShaper::processSample と同一。
        1) feedback = computeFeedback(state, coeffs)
        2) shaped = x * headroom + feedback (headroom=1.0 で固定)
        3) quantized = quantize(shaped, rng)
        4) error = quantized - shaped
        5) clamped_error = clamp(error, -2*scale, 2*scale)
        6) advanceState(state, clamped_error, coeffs)
        7) return quantized
        """
        ch_state = self.states[ch]
        feedback = self.compute_feedback(ch_state, coeffs)
        shaped = x + feedback
        quantized = self.quantize(shaped, ch)
        error = quantized - shaped
        clamped_error = np.clip(error, -2.0 * self.scale, 2.0 * self.scale)
        self.advance_state(ch_state, clamped_error, coeffs)
        return quantized

    def process_block(self, samples: np.ndarray,
                      coeffs: np.ndarray) -> np.ndarray:
        """
        ブロック処理 (片チャンネル、ch=0 固定)
        """
        self.reset()
        out = np.zeros(len(samples), dtype=np.float64)
        for n in range(len(samples)):
            out[n] = self.process_sample(0, samples[n], coeffs)
        self.clamp_state_simd(self.states[0])
        return out


# ============================================================================
# テスト信号生成
# ============================================================================
def make_impulse(length: int, amplitude: float = 1.0) -> np.ndarray:
    """インパルス信号: n=0 のみ amplitude、他は 0"""
    x = np.zeros(length)
    x[0] = amplitude
    return x

def make_silence(length: int) -> np.ndarray:
    """無音信号: 全ゼロ"""
    return np.zeros(length)

def make_sine(length: int, freq_hz: float, amplitude: float,
              sample_rate: float = 48000.0) -> np.ndarray:
    """サイン波"""
    t = np.arange(length, dtype=np.float64) / sample_rate
    return amplitude * np.sin(2.0 * np.pi * freq_hz * t)

def make_noise(length: int, amplitude: float = 1.0,
               seed: int = 0) -> np.ndarray:
    """白色雑音"""
    rng = np.random.RandomState(seed)
    return amplitude * rng.randn(length)


# ============================================================================
# 分析関数
# ============================================================================
def compute_dc(signal: np.ndarray, settle: int = 0) -> float:
    """
    DC成分: 平均値 (整定期間 settle サンプルをスキップ)
    算出式: DC = mean(signal[settle:])
    """
    return float(np.mean(signal[settle:]))

def compute_ntf(impulse_response: np.ndarray,
                sample_rate: float = 48000.0):
    """
    NTF (Noise Transfer Function):
    インパルス応答の FFT 振幅スペクトル [dB]。
    FFT 条件: length = len(impulse_response), 窓関数なし (矩形窓)
    """
    n = len(impulse_response)
    spectrum = np.fft.rfft(impulse_response)
    magnitude_db = 20.0 * np.log10(np.abs(spectrum) + 1e-30)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    return freqs, magnitude_db

def compute_psd(signal: np.ndarray, sample_rate: float = 48000.0,
                nperseg: int = 1024):
    """
    PSD (Power Spectral Density): Welch 法
    """
    from scipy import signal as sp_signal
    freqs, psd = sp_signal.welch(signal, fs=sample_rate,
                                  nperseg=nperseg,
                                  window='hann')
    psd_db = 10.0 * np.log10(psd + 1e-30)
    return freqs, psd_db

def detect_limit_cycle(signal: np.ndarray, threshold_ratio: float = 0.3):
    """
    自己相関による Limit Cycle 検出。
    算出式: 自己相関のサイドローブが中心ピークの threshold_ratio を超えるピーク数をカウント
    """
    center = len(signal) // 2
    corr = np.correlate(signal, signal, mode='same')
    corr = corr[center:]
    norm = corr[0]
    if norm == 0:
        return 0, corr
    peaks = np.sum(corr[1:] > norm * threshold_ratio)
    return peaks, corr


# ============================================================================
# メイン評価
# ============================================================================


def run_comparison(coeffs: np.ndarray, label: str,
                   coeffs_source: str = ""):
    """
    与えられた係数で Pattern A vs Pattern B を比較する。
    両Patternの結果をまとめて dict で返す。
    """
    print(f"\n{'='*72}")
    print(f"係数セット: {label}")
    if coeffs_source:
        print(f"出典: {coeffs_source}")
    print(f"係数値: {np.array2string(coeffs, precision=4, separator=', ')}")
    print(f"{'='*72}")

    results = {}

    for nb_label, use_nb in [("Pattern A (CMSIS-DSP型/P7前)", False),
                              ("Pattern B (Ne10型/P7後)", True)]:
        key = "pattern_a" if not use_nb else "pattern_b"
        print(f"\n  --- {nb_label} ---")

        # Test 1: 無音入力 (DC drift / Limit Cycle) — ディザ有り
        sim = LatticeNoiseShaperSim(use_nb, bit_depth=24, dither_enabled=True)
        silence = make_silence(NUM_SAMPLES_LONG)
        out_silence = sim.process_block(silence, coeffs)

        dc = compute_dc(out_silence, settle=NUM_SAMPLES_SETTLE)
        rms = float(np.sqrt(np.mean(out_silence[NUM_SAMPLES_SETTLE:]**2)))
        peak = float(np.max(np.abs(out_silence[NUM_SAMPLES_SETTLE:])))
        abs_mean = float(np.mean(np.abs(out_silence[NUM_SAMPLES_SETTLE:])))
        lc_peaks, _ = detect_limit_cycle(
            out_silence[NUM_SAMPLES_SETTLE:NUM_SAMPLES_SETTLE+2000]
        )

        results[f'{key}_dc'] = dc
        results[f'{key}_rms'] = rms
        results[f'{key}_peak'] = peak
        results[f'{key}_lc'] = lc_peaks

        print(f"    [Silence w/ dither] DC={dc:+.4e} RMS={rms:.4e} Peak={peak:.4e} LC_peaks={lc_peaks}")

        # Test 2: 無音 + ディザなし
        sim_nd = LatticeNoiseShaperSim(use_nb, bit_depth=24, dither_enabled=False)
        out_nd = sim_nd.process_block(silence, coeffs)
        dc_nd = compute_dc(out_nd, settle=NUM_SAMPLES_SETTLE)
        rms_nd = float(np.sqrt(np.mean(out_nd[NUM_SAMPLES_SETTLE:]**2)))
        results[f'{key}_dc_nodither'] = dc_nd
        results[f'{key}_rms_nodither'] = rms_nd
        print(f"    [Silence no dither] DC={dc_nd:+.4e} RMS={rms_nd:.4e}")

        # Test 3: インパルス応答 (NTF)
        sim2 = LatticeNoiseShaperSim(use_nb, bit_depth=24, dither_enabled=True)
        impulse = make_impulse(NUM_SAMPLES_SHORT, amplitude=1.0)
        out_impulse = sim2.process_block(impulse, coeffs)
        _, ntf_db = compute_ntf(out_impulse)
        results[f'{key}_ntf_max'] = float(np.max(ntf_db))
        results[f'{key}_ntf_mean'] = float(np.mean(ntf_db))
        print(f"    [Impulse NTF] Max={results[f'{key}_ntf_max']:.1f}dB Mean={results[f'{key}_ntf_mean']:.1f}dB")

        # Test 4: サイン波 SNR
        sim3 = LatticeNoiseShaperSim(use_nb, bit_depth=24, dither_enabled=True)
        sine = make_sine(NUM_SAMPLES_LONG, 1000.0, 0.01)
        out_sine = sim3.process_block(sine, coeffs)
        settle = NUM_SAMPLES_SETTLE
        sig_power = float(np.mean(sine[settle:]**2))
        noise = out_sine[settle:] - sine[settle:]
        noise_power = float(np.mean(noise**2))
        snr = 10.0 * np.log10(sig_power / (noise_power + 1e-30))
        results[f'{key}_snr'] = snr
        print(f"    [1kHz Sine SNR] {snr:.1f}dB")

    # 両Patternの比較結果
    if results.get('pattern_a_dc', 0) != 0 or results.get('pattern_b_dc', 0) != 0:
        dc_a = results['pattern_a_dc']
        dc_b = results['pattern_b_dc']
        dc_ratio = abs(dc_b / (dc_a + 1e-30))
        print(f"\n  >>> Pattern B/A DC比 = {dc_ratio:.1f}x")

    return results


# ============================================================================
# 条件一覧と実行
# ============================================================================
if __name__ == "__main__":
    print("=" * 72)
    print("LatticeNoiseShaper state convention 比較")
    print("=" * 72)
    print(f"""
パラメータ一覧:
  KORDER                 = {KORDER}
  KLATTICE_STATE_LIMIT   = {KLATTICE_STATE_LIMIT}
  KSTATE_LIMIT           = {KSTATE_LIMIT}
  KCOEFF_LIMIT           = {KCOEFF_LIMIT}
  量子化ビット数          = 24
  TPDFディザ              = 有効 (Test2除く)
  サンプル数(無音)        = {NUM_SAMPLES_LONG}
  サンプル数(インパルス)  = {NUM_SAMPLES_SHORT}
  整定スキップ            = {NUM_SAMPLES_SETTLE}
  RNG(Xoshiro256++)      = ConvoPeq実装と同一 (2つの固定シード)
  DC算出式               = mean(signal[settle:])
  NTF                    = |FFT(impulse_response)| [dB]
  PSD                    = Welch法 (hann窓, nperseg=1024)
  Limit Cycle            = 自己相関サイドローブピーク数
""")

    all_results = []

    # ----- テスト1: デフォルト係数 -----
    print("\n>>> テスト1: ConvoPeq デフォルト係数")
    r = run_comparison(DEFAULT_COEFFS,
                       "ConvoPeq Default",
                       "src/audioengine/AudioEngine.Learning.cpp:286")
    all_results.append(("Default", r))

    # ----- テスト2: 単純係数 (小) -----
    simple_coeffs = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001])
    print("\n>>> テスト2: 単純係数 (小)")
    r = run_comparison(simple_coeffs, "Simple Small")
    all_results.append(("Simple Small", r))

    # ----- テスト3: 単純係数 (大) -----
    large_coeffs = np.array([0.5, -0.4, 0.3, -0.3, 0.2, -0.2, 0.1, -0.1, 0.05])
    print("\n>>> テスト3: 単純係数 (中)")
    r = run_comparison(large_coeffs, "Simple Medium")
    all_results.append(("Simple Medium", r))

    # ----- テスト4: ランダム係数 (10セット) -----
    print("\n>>> テスト4: ランダム係数 10セット")
    rng = np.random.RandomState(12345)
    random_coeffs_list = []
    for t in range(10):
        k = rng.uniform(-0.85, 0.85, KORDER)
        random_coeffs_list.append(k)
        r = run_comparison(k, f"Random #{t}")
        all_results.append((f"Random #{t}", r))

    # ========================================================================
    # 集計
    # ========================================================================
    print(f"\n{'='*72}")
    print("集計: Pattern A vs Pattern B のDC比較")
    print(f"{'='*72}")

    header = f"{'Coeff Set':<20s} {'DC_A':>12s} {'DC_B':>12s} {'|B/A|比':>10s} {'RMS_A':>12s} {'RMS_B':>12s}"
    print(header)
    print("-" * len(header))

    for name, r in all_results:
        dc_a = r.get('pattern_a_dc', 0.0)
        dc_b = r.get('pattern_b_dc', 0.0)
        rms_a = r.get('pattern_a_rms', 0.0)
        rms_b = r.get('pattern_b_rms', 0.0)
        ratio = abs(dc_b / (dc_a + 1e-30))
        print(f"{name:<20s} {dc_a:>+12.4e} {dc_b:>+12.4e} {ratio:>9.1f}x {rms_a:>12.4e} {rms_b:>12.4e}")

    # ========================================================================
    # 追加テスト: ディザ有無の影響 (デフォルト係数)
    # ========================================================================
    print(f"\n{'='*72}")
    print("追加テスト: ディザ有無の影響 (デフォルト係数)")
    print(f"{'='*72}")

    for dither_on in [True, False]:
        print(f"\n  ディザ {'有効' if dither_on else '無効'}:")
        for nb_label, use_nb in [("Pattern B (Ne10型)", True),
                                  ("Pattern A (CMSIS-DSP型)", False)]:
            sim = LatticeNoiseShaperSim(use_nb, bit_depth=24,
                                         dither_enabled=dither_on)
            silence = make_silence(5000)
            out = sim.process_block(silence, DEFAULT_COEFFS)
            dc = compute_dc(out, settle=500)
            rms = float(np.sqrt(np.mean(out[500:]**2)))
            peak = float(np.max(np.abs(out[500:])))
            print(f"    {nb_label:30s}: DC={dc:+.4e} RMS={rms:.4e} Peak={peak:.4e}")

    # ========================================================================
    # 結論
    # ========================================================================
    print(f"\n{'='*72}")
    print("評価サマリー")
    print(f"{'='*72}")
    print("""
本スクリプトは以下のパラメータで完全に再現可能:
  - Python: 3.14+
  - 依存: numpy (scipy は PSD 計算のみ)
  - シード: すべて明示
  - 実装: ConvoPeq の LatticeNoiseShaper の完全な動作再現

注意:
  - ここでの結果は決定論的な数値シミュレーションであり、
    ConvoPeq 実機での動作を近似するもの。
  - CMA-ES 学習器は実 LatticeNoiseShaper インスタンスで評価するため、
    学習済み係数は Pattern B に対しても最適化されている可能性がある。
  - 最終判断には実機での NTF/PSD/DC測定と ABX 試聴が必要。
""")
