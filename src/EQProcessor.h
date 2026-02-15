//============================================================================
#pragma once
// EQProcessor.h  ── v0.1 (JUCE 8.0.12対応)
//
// 20バンドパラメトリックイコライザー処理クラス
//
// ■ スレッドセーフ設計 (Lock-free Parameter Update):
//   - パラメータ変更: UIスレッドから setBandXxx() を呼び出し、内部で新しい BandNode を作成して atomic に差し替えます (RCUパターン)。
//   - 係数更新: Audio Thread 側は atomic load で最新の BandNode を取得して処理します。ロックフリーで安全に更新が反映されます。
//   - パラメータ読み取り: std::atomic なので Audio Thread から安全に読み取れます。
//
// ■ フィルタ実装:
//   - TPT (Topology-Preserving Transform) SVF を使用
//   - 時間変化に強く、係数変調時のノイズが少ない
//   - Audio EQ Cookbook (RBJ) の式で係数を計算
//
// ■ サンプルレート変更時:
//   - prepareToPlay() で検知し、フィルタ状態をリセットする
//   - 古いサンプルレートで計算された係数を使い続けると
//     音が不安定になる
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <cstdint>
#include <complex>
#include <array>
#include <vector>

//--------------------------------------------------------------
// バンドタイプ列挙型
//--------------------------------------------------------------
enum class EQBandType
{
    LowShelf,   // ロー・シェルフ
    Peaking,    // ピーキング（中域山形）
    HighShelf,  // ハイ・シェルフ
    LowPass,    // ローパス
    HighPass    // ハイパス
};

//--------------------------------------------------------------
// チャンネルモード列挙型
//--------------------------------------------------------------
enum class EQChannelMode
{
    Stereo, // ステレオ（両方）
    Left,   // 左チャンネルのみ
    Right   // 右チャンネルのみ
};

//--------------------------------------------------------------
// 単一バンドのパラメータ
// 各フィールドが個別に atomic → オーディオスレッドで読み取り安全
//--------------------------------------------------------------
struct EQBandParams
{
    float frequency{ 1000.0f }; // 中心周波数 [Hz]
    float gain    { 0.0f   }; // ゲイン [dB]
    float q       { 0.707f }; // Q値
    bool  enabled { true   }; // 有効/無効

    EQBandParams() = default;

    EQBandParams(const EQBandParams& other)
        : frequency(other.frequency),
          gain(other.gain),
          q(other.q),
          enabled(other.enabled)
    {
    }
};

//--------------------------------------------------------------
// EQフィルタ係数 (SVF: Audio Processing)
//--------------------------------------------------------------
struct EQCoeffsSVF
{
    // Vadim Zavalishin "The Art of VA Filter Design" に基づくトポロジー保存変換係数。
    double g = 0.0, k = 0.0, a1 = 0.0, a2 = 0.0, a3 = 0.0;
    double m0 = 1.0, m1 = 0.0, m2 = 0.0;
};

//--------------------------------------------------------------
// EQフィルタ係数 (Biquad: Analysis / Plotting)
//--------------------------------------------------------------
struct EQCoeffsBiquad
{
    // 周波数応答曲線の計算（getMagnitude）に使用する。
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a0 = 1.0, a1 = 0.0, a2 = 0.0;
};

//--------------------------------------------------------------
// EQプロセッサークラス
//--------------------------------------------------------------
class EQProcessor : public juce::ChangeBroadcaster
{
public:
    static constexpr int NUM_BANDS        = 20;  // 20バンドパラメトリックEQ
    static constexpr int MAX_CHANNELS     = 2;   // ステレーオ対応

    // ── デフォルト値 ──
    static constexpr float DEFAULT_FREQS[NUM_BANDS] = {
        25.0f, 40.0f, 63.0f, 100.0f, 160.0f,
        250.0f, 400.0f, 630.0f, 1000.0f, 1600.0f,
        2500.0f, 4000.0f, 6300.0f, 10000.0f, 11000.0f,
        12500.0f, 14000.0f, 16500.0f, 18000.0f, 19500.0f
    };
    static constexpr float DEFAULT_Q = 0.707f;

    // ── AGC定数 ──
    static constexpr float AGC_ALPHA       = 0.01f; // エンベロープ追従係数 (Attack/Release)
    static constexpr float AGC_GAIN_SMOOTH = 0.05f; // ゲイン変化スムーシング
    static constexpr float AGC_MIN_GAIN    = 0.06f; // 最小ゲイン制限 (~ -24dB)
    static constexpr float AGC_MAX_GAIN    = 16.0f; // 最大ゲイン制限 (~ +24dB)

    // ── DSPパラメータ制限 ──
    static constexpr float DSP_MIN_FREQ = 10.0f;
    static constexpr float DSP_MAX_FREQ_NYQUIST_RATIO = 0.95f;
    static constexpr float DSP_MIN_Q = 0.1f;
    static constexpr float DSP_MAX_Q = 20.0f;
    static constexpr float DSP_MIN_GAIN_DB = -48.0f;
    static constexpr float DSP_MAX_GAIN_DB = 48.0f;

    EQProcessor();
    ~EQProcessor();

    //----------------------------------------------------------
    // prepareToPlay: サンプルレート・バッファサイズ変更時に呼ぶ
    // フィルタ状態をリセットし、係数再計算を強制する
    //----------------------------------------------------------
    void prepareToPlay(int sampleRate, int /*samplesPerBlock*/);

    //----------------------------------------------------------
    // process: オーディオスレッドから呼ばれる
    // ロック・new・I/O・待機（IR再ロード等）禁止
    //----------------------------------------------------------
    void process(juce::AudioBuffer<double>& buffer, int numSamples);

    //----------------------------------------------------------
    // パラメータ変更 (UIスレッドから呼ぶ)
    //----------------------------------------------------------
    void setBandFrequency(int band, float freq);
    void setBandGain    (int band, float gainDb);
    void setBandQ       (int band, float q);
    void setBandEnabled (int band, bool enabled);

    // トータルゲイン・AGC
    void setTotalGain(float gainDb);
    float getTotalGain() const;
    void setAGCEnabled(bool enabled);
    bool getAGCEnabled() const;

    // フィルタータイプ変更
    void setBandType(int band, EQBandType type);
    EQBandType getBandType(int band) const;

    // チャンネルモード変更
    void setBandChannelMode(int band, EQChannelMode mode);
    EQChannelMode getBandChannelMode(int band) const;

    //----------------------------------------------------------
    // パラメータ読み取り (UIスレッドで表示に使用)
    //----------------------------------------------------------
    EQBandParams getBandParams(int band) const;

    //----------------------------------------------------------
    // デフォルト値リセット
    //----------------------------------------------------------
    void resetToDefaults();

    //----------------------------------------------------------
    // プリセット読み込み (AudioEngine::prepareToPlayから呼ばれる)
    //----------------------------------------------------------
    void loadPreset(int index);

    //----------------------------------------------------------
    // テキストファイルからプリセット読み込み
    //----------------------------------------------------------
    bool loadFromTextFile(const juce::File& file);

    //----------------------------------------------------------
    // State Management (ZLEqualizer style)
    //----------------------------------------------------------
    juce::ValueTree getState() const;
    void setState (const juce::ValueTree& state);

    //----------------------------------------------------------
    // 係数計算ヘルパー (static public)
    // 外部からの応答曲線計算などに使用
    //----------------------------------------------------------
    static EQCoeffsSVF    calcSVFCoeffs   (EQBandType type, float freq, float gainDb, float q, int sr) noexcept;
    static EQCoeffsBiquad calcBiquadCoeffs(EQBandType type, float freq, float gainDb, float q, int sr) noexcept;

    static void validateAndClampParameters(float& freq, float& gainDb, float& q, int sr) noexcept;
    static float getMagnitudeSquared(const EQCoeffsBiquad& coeffs, float freq, float sampleRate) noexcept;
    static float getMagnitudeSquared(const EQCoeffsBiquad& coeffs, const std::complex<double>& z) noexcept;

    struct BandNode
    {
        EQCoeffsSVF coeffs;
        bool active;
        EQChannelMode mode;
    };

    struct EQState
    {
        std::array<EQBandParams, NUM_BANDS> bands;
        std::array<EQBandType, NUM_BANDS> bandTypes;
        std::array<EQChannelMode, NUM_BANDS> bandChannelModes;
        float totalGainDb = 0.0f;
        bool agcEnabled = false;
    };

private:
    //----------------------------------------------------------
    // プライベートヘルパー関数
    //----------------------------------------------------------

    // 係数更新処理
    void updateCoefficientsIfNeeded(const EQState& state);

    // サイレンス検出
    bool isBufferSilent(const juce::AudioBuffer<double>& buffer, int numSamples) const noexcept;

    // 係数計算
    std::shared_ptr<BandNode> createBandNode(int bandIndex, const EQState& state) const;
    void updateBandNode(int bandIndex);

    // スムージング処理
    std::atomic<std::shared_ptr<EQState>> currentState;

    float agcCurrentGain = 1.0f;
    float agcEnvInput    = 0.0f;
    float agcEnvOutput   = 0.0f;

    // ── パラメータ補間 (Smoothing) ──
    juce::SmoothedValue<float> smoothTotalGain;
    static constexpr double SMOOTHING_TIME_SEC = 0.05; // 50ms

    // ── 係数管理 (Atomic Swap) ──
    std::atomic<std::shared_ptr<BandNode>> bandNodes[NUM_BANDS];
    std::vector<std::shared_ptr<BandNode>> trashBin;
    juce::CriticalSection trashBinLock;

    // ── フィルタ状態 [チャンネル][バンド][z1/z2] ──
    // SVFの2つの積分器状態 (ic1eq, ic2eq)
    double filterState[MAX_CHANNELS][NUM_BANDS][2] = {};

    // ── 現在のサンプルレート ──
    // prepareToPlay で更新。Audio Thread で係数再計算に使用。
    int currentSampleRate{ 0 };

    // ── リセットフラグ ──
    std::atomic<bool> isResetting { false };

    // ── AGC適用 (Audio Thread 内で呼ばれる) ──
    void processAGC(juce::AudioBuffer<double>& buffer, int numSamples, const EQState& state);
    float calculateAGCGain(float inputEnv, float outputEnv) const noexcept;

    // ── SVF係数計算 (Private Helpers) ──
    static EQCoeffsSVF calcLowShelfSVF (double freq, double gainDb, double q, double sr) noexcept;
    static EQCoeffsSVF calcPeakingSVF  (double freq, double gainDb, double q, double sr) noexcept;
    static EQCoeffsSVF calcHighShelfSVF(double freq, double gainDb, double q, double sr) noexcept;
    static EQCoeffsSVF calcLowPassSVF  (double freq, double q, double sr) noexcept;
    static EQCoeffsSVF calcHighPassSVF (double freq, double q, double sr) noexcept;

    // ── Biquad係数計算 (Private Helpers) ──
    static EQCoeffsBiquad calcLowShelfBiquad (double freq, double gainDb, double q, double sr) noexcept;
    static EQCoeffsBiquad calcPeakingBiquad  (double freq, double gainDb, double q, double sr) noexcept;
    static EQCoeffsBiquad calcHighShelfBiquad(double freq, double gainDb, double q, double sr) noexcept;
    static EQCoeffsBiquad calcLowPassBiquad  (double freq, double q, double sr) noexcept;
    static EQCoeffsBiquad calcHighPassBiquad (double freq, double q, double sr) noexcept;
};
