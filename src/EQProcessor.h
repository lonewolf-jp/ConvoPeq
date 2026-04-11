//============================================================================
#pragma once
// EQProcessor.h ── v0.2 (JUCE 8.0.12対応)
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
//   - SVF係数: Vadim Zavalishin "The Art of VA Filter Design" の式を元に計算
//   - Biquad係数(UI描画用): Audio EQ Cookbook (RBJ) の式で計算
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
#include "AlignedAllocation.h"
#include "RefCountedDeferred.h"
#include "core/EQParameters.h"

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
// EQフィルタ係数 (Biquad: 解析/描画用)
//--------------------------------------------------------------
struct EQCoeffsBiquad
{
    // 周波数応答曲線の計算（getMagnitude）に使用する。
    double b0 = 1.0, b1 = 0.0, b2 = 0.0;
    double a0 = 1.0, a1 = 0.0, a2 = 0.0;
};

//--------------------------------------------------------------
// EQCoeffCache: 係数キャッシュ（RefCounted資源）
// v2.3 Phase 1 新規追加
//
// 複数のスナップショット間で同一EQパラメータの係数を共有し、
// CPU/メモリ効率を向上させる不変キャッシュ
//--------------------------------------------------------------
#pragma warning(push)
#pragma warning(disable : 4324)  // alignas(64) により構造体がパッドされることは意図した動作のため警告を無視

struct alignas(64) EQCoeffCache : public RefCountedDeferred<EQCoeffCache>
{
    EQCoeffsSVF coeffs[20];              // SVF係数 (20バンド)
    bool bandActive[20] = {};            // バンド有効フラグ
    int channelModes[20] = {};           // チャンネルモード (0:Stereo, 1:Left, 2:Right)
    int filterStructure = 0;             // 0:Serial, 1:Parallel

    // メタデータ
    uint64_t paramsHash = 0;             // パラメータハッシュ値
    double sampleRate = 0.0;             // サンプリングレート
    int maxBlockSize = 0;                // 最大ブロックサイズ
    uint64_t generation = 0;             // 世代番号

    // Parallel モード用バッファ（事前割り当て）
    double* parallelInputBuffer = nullptr;
    double* parallelWorkBuffer = nullptr;
    double* parallelAccumBuffer = nullptr;
    int parallelBufferSize = 0;

    EQCoeffCache() = default;
    ~EQCoeffCache();
    EQCoeffCache(const EQCoeffCache&) = delete;
    EQCoeffCache& operator=(const EQCoeffCache&) = delete;
};

#pragma warning(pop)

//--------------------------------------------------------------
// EQプロセッサークラス
//--------------------------------------------------------------
class EQProcessor : public juce::ChangeBroadcaster
{
public:
    enum class FilterStructure
    {
        Serial,
        Parallel
    };

    static constexpr int NUM_BANDS        = 20;  // 20バンドパラメトリックEQ
    static constexpr int MAX_CHANNELS     = 2;   // ステレオ対応

    // ── デフォルト値 ──
    static constexpr float DEFAULT_FREQS[NUM_BANDS] = {
        25.0f, 40.0f, 63.0f, 100.0f, 160.0f,
        250.0f, 400.0f, 630.0f, 1000.0f, 1600.0f,
        2500.0f, 4000.0f, 6300.0f, 10000.0f, 11000.0f,
        12500.0f, 14000.0f, 16500.0f, 18000.0f, 19500.0f
    };
    static constexpr float DEFAULT_Q = 0.707f;

    // ── AGC定数 ──
    static constexpr double AGC_ATTACK_TIME_SEC   = 0.1; // エンベロープ追従アタック時定数 (0.1s) - 速いアタックでトランジェントに即応
    static constexpr double AGC_RELEASE_TIME_SEC  = 2.0; // エンベロープ追従リリース時定数 (2.0s) - 緩やかなリリースでポンピング抑制
    static constexpr double AGC_SMOOTH_TIME_SEC   = 0.2; // ゲイン変化スムーシング時定数 (0.2s)
    static constexpr float AGC_MIN_GAIN    = 0.06f; // 最小ゲイン制限 (~ -24dB)
    static constexpr float AGC_MAX_GAIN    = 16.0f; // 最大ゲイン制限 (~ +24dB)

    // ── DSPパラメータ制限 ──
    static constexpr float DSP_MIN_FREQ = 20.0f;
    static constexpr float DSP_MAX_FREQ = 20000.0f;
    static constexpr float DSP_MAX_FREQ_NYQUIST_RATIO = 0.95f;
    static constexpr float DSP_MIN_Q = 0.01f;
    static constexpr float DSP_MAX_Q = 20.0f;
    static constexpr float DSP_MIN_GAIN_DB = -48.0f;
    static constexpr float DSP_MAX_GAIN_DB = 48.0f;

    EQProcessor();
    ~EQProcessor();

    //----------------------------------------------------------
    // prepareToPlay: サンプルレート・バッファサイズ変更時に呼ぶ
    // フィルタ状態をリセットし、係数再計算を強制する
    //----------------------------------------------------------
    void prepareToPlay(double sampleRate, int newMaxInternalBlockSize);

    //----------------------------------------------------------
    // process: オーディオスレッドから呼ばれる
    // ロック・new・I/O・待機（IR再ロード等）禁止
    //----------------------------------------------------------
    void process(juce::dsp::AudioBlock<double>& block);
    void process(juce::dsp::AudioBlock<double>& block,
                 const convo::EQParameters& eqParams,
                 const EQCoeffCache* coeffCache);
    void releaseResources();

    // バイパス制御
    void setBypass(bool shouldBypass) { bypassRequested.store(shouldBypass, std::memory_order_relaxed); }
    bool isBypassed() const { return bypassRequested.load(std::memory_order_relaxed); }

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

    // 追加DSP設定
    void setNonlinearSaturation(float value) noexcept;
    float getNonlinearSaturation() const noexcept;
    void setFilterStructure(FilterStructure mode) noexcept;
    FilterStructure getFilterStructure() const noexcept;

    //----------------------------------------------------------
    // v2.3 EQCoeffCache 生成インターフェース
    //----------------------------------------------------------
    static uint64_t computeParamsHash(const convo::EQParameters& params) noexcept;
    static EQCoeffCache* createCoeffCache(
        const convo::EQParameters& eqParams,
        double sampleRate,
        int maxBlockSize,
        uint64_t generation) noexcept;

    //----------------------------------------------------------
    // パラメータ読み取り (UIスレッドで表示に使用)
    //----------------------------------------------------------
    EQBandParams getBandParams(int band) const;

    //----------------------------------------------------------
    // 状態リセット
    //----------------------------------------------------------
    void reset();

    //----------------------------------------------------------
    // デフォルト値リセット
    //----------------------------------------------------------
    void resetToDefaults();

    //----------------------------------------------------------
    // 状態構造体
    //----------------------------------------------------------
    struct BandNode : public RefCountedDeferred<BandNode>
    {
        EQCoeffsSVF coeffs;
        bool active;
        EQChannelMode mode;
    };

    struct EQState : public RefCountedDeferred<EQState>
    {
        std::array<EQBandParams, NUM_BANDS> bands;
        std::array<EQBandType, NUM_BANDS> bandTypes;
        std::array<EQChannelMode, NUM_BANDS> bandChannelModes;
        float totalGainDb = 0.0f;
        bool agcEnabled = false;
        float nonlinearSaturation = 0.2f;
        int filterStructure = 0; // 0: Serial, 1: Parallel

        convo::EQParameters toEQParameters() const;

        // Explicitly define the copy constructor
        EQState() = default;

        EQState(const EQState& other)
            : bands(other.bands),
              bandTypes(other.bandTypes),
              bandChannelModes(other.bandChannelModes),
              totalGainDb(other.totalGainDb),
              agcEnabled(other.agcEnabled),
              nonlinearSaturation(other.nonlinearSaturation),
              filterStructure(other.filterStructure)
        {
        }

        // Explicitly define the move constructor
        EQState(EQState&& other)
            : bands(std::move(other.bands)),
              bandTypes(std::move(other.bandTypes)),
              bandChannelModes(std::move(other.bandChannelModes)),
              totalGainDb(other.totalGainDb),
              agcEnabled(other.agcEnabled),
              nonlinearSaturation(other.nonlinearSaturation),
              filterStructure(other.filterStructure)
        {
        }

        EQState& operator=(const EQState& other)
        {
            if (this != &other)
            {
                bands             = other.bands;
                bandTypes         = other.bandTypes;
                bandChannelModes  = other.bandChannelModes;
                totalGainDb       = other.totalGainDb;
                agcEnabled        = other.agcEnabled;
                nonlinearSaturation = other.nonlinearSaturation;
                filterStructure   = other.filterStructure;
            }
            return *this;
        }

        EQState& operator=(EQState&& other)
        {
            if (this != &other)
            {
                bands             = std::move(other.bands);
                bandTypes         = std::move(other.bandTypes);
                bandChannelModes  = std::move(other.bandChannelModes);
                totalGainDb       = other.totalGainDb;
                agcEnabled        = other.agcEnabled;
                nonlinearSaturation = other.nonlinearSaturation;
                filterStructure   = other.filterStructure;
            }
            return *this;
        }
    };

    //----------------------------------------------------------
    // 状態スナップショット取得 (AudioEngine用)
    //----------------------------------------------------------
    EQState* getEQState() const;  // 生ポインタ (所有権は共有せず、ライフタイムは currentStateRaw に依存)
    const EQState* getEQStateSnapshot() const { return getEQState(); }
    bool getAndClearPendingAGCChange() noexcept
    {
        return m_pendingAGCChange.exchange(false, std::memory_order_acq_rel);
    }

    // 他のインスタンスから状態を同期 (AudioEngine用)
    void syncStateFrom(const EQProcessor& other);
    // 個別パラメータの同期 (最適化)
    void syncBandNodeFrom(const EQProcessor& other, int bandIndex);
    void syncGlobalStateFrom(const EQProcessor& other);

    // 参照カウントを追加した状態ポインタを返す (呼び出し元は release() する責任を持つ)
    EQState* getEQStateAndAddRef() const;

    //----------------------------------------------------------
    // プリセット読み込み (AudioEngine::prepareToPlayから呼ばれる)
    //----------------------------------------------------------
    void loadPreset(int index);

    //----------------------------------------------------------
    // テキストファイルからプリセット読み込み
    //----------------------------------------------------------
    bool loadFromTextFile(const juce::File& file);

    //----------------------------------------------------------
    // 状態管理 (ZLEqualizerスタイル)
    //----------------------------------------------------------
    juce::ValueTree getState() const;
    void setState (const juce::ValueTree& state);

    void cleanup();

    //----------------------------------------------------------
    // 係数計算ヘルパー (static public)
    // 外部からの応答曲線計算などに使用
    //----------------------------------------------------------
    static EQCoeffsSVF    calcSVFCoeffs   (EQBandType type, float freq, float gainDb, float q, double sr) noexcept;
    static EQCoeffsBiquad calcBiquadCoeffs(EQBandType type, float freq, float gainDb, float q, double sr) noexcept;

    static void validateAndClampParameters(float& freq, float& gainDb, float& q, double sr) noexcept;
    static float getMagnitudeSquared(const EQCoeffsBiquad& coeffs, float freq, float sampleRate) noexcept;
    static float getMagnitudeSquared(const EQCoeffsBiquad& coeffs, const std::complex<double>& z) noexcept;


    static EQCoeffsBiquad svfToDisplayBiquad(const EQCoeffsSVF& svf) noexcept;

private:
    //----------------------------------------------------------
    // プライベートヘルパー関数
    //----------------------------------------------------------

    // 【Fix Bug #7】totalGainDbTargetとtotalGainTargetを一括で更新するヘルパー。
    // Message Thread側でdB→linear変換(std::pow)を行い、Audio Threadは linear値のみ参照する。
    inline void storeTotalGainDb(float gainDb) noexcept
    {
        totalGainDbTarget.store(gainDb, std::memory_order_relaxed);
        totalGainTarget.store(juce::Decibels::decibelsToGain<double>(static_cast<double>(gainDb)),
                              std::memory_order_relaxed);
    }

    // サイレンス検出
    bool isBufferSilent(const juce::AudioBuffer<double>& buffer, int numSamples) const noexcept;
    bool isAudioBlockSilent(const juce::dsp::AudioBlock<double>& block, int numChannels, int numSamples) const noexcept;

    // 係数計算
    BandNode* createBandNode(int bandIndex, const EQState& state) const;
    void updateBandNode(int bandIndex);

    // スムージング処理
    std::atomic<EQState*> currentStateRaw { nullptr }; // Raw pointer for Audio Thread (Lock-free)

    // ── 状態リセットフラグ (Audio Thread用) ──
    std::atomic<uint32_t> bandResetMask { 0 };
    std::atomic<bool> agcResetRequest { false };

    std::atomic<bool> agcEnabled { false };
    std::atomic<bool> m_pendingAGCChange { false };
    std::atomic<double> agcCurrentGain { 1.0 };
    std::atomic<double> agcEnvInput    { 0.0 };
    std::atomic<double> agcEnvOutput   { 0.0 };
    std::atomic<double> agcAttackCoeff { 0.0 };
    std::atomic<double> agcReleaseCoeff { 0.0 };
    std::atomic<double> agcSmoothCoeff { 0.0 };
    double cachedInputRMS = 0.0; // AGC用の入力レベルキャッシュ
    convo::ScopedAlignedPtr<double> agcAttackCoeffTable;
    convo::ScopedAlignedPtr<double> agcReleaseCoeffTable;
    convo::ScopedAlignedPtr<double> agcSmoothCoeffTable;
    int agcCoeffTableCapacity = 0;

    // ── パラメータ補間 (Smoothing) ──
    std::atomic<float>  totalGainDbTarget { 0.0f };
    // 【Fix Bug #7】totalGainDbTargetのlinear値をMessage Threadで事前計算してAudio Threadに渡す。
    // Audio Thread内でのjuce::Decibels::decibelsToGain()(= std::pow/libm)呼び出しを排除する。
    std::atomic<double> totalGainTarget   { 1.0 }; // linear gain, default 0dB = 1.0
    juce::SmoothedValue<double> smoothTotalGain;
    static constexpr double SMOOTHING_TIME_SEC = 0.05; // 50ms
    static constexpr double BYPASS_FADE_TIME_SEC = 0.005; // 5ms

    // ── 係数管理 (Atomic Swap) ──
    std::array<std::atomic<BandNode*>, NUM_BANDS> bandNodes; // Raw pointer for Audio Thread
    std::array<BandNode*, NUM_BANDS> activeBandNodes { nullptr }; // Ownership for Message Thread

    // ── フィルタ状態 [チャンネル][バンド][z1/z2] ──
    // SVFの2つの積分器状態 (ic1eq, ic2eq)
    std::array<std::array<std::array<double, 2>, NUM_BANDS>, MAX_CHANNELS> filterState{};
    std::atomic<bool> bypassRequested { false };
    std::atomic<bool> bypassed { false }; // 実効バイパス状態（フェード完了後に更新）
    juce::SmoothedValue<double> bypassFadeGain;

    // ── 現在のサンプルレート ──
    // prepareToPlay で更新。Audio Thread で係数再計算に使用。
    // 【修正】スレッド間アクセスのため std::atomic に変更
    std::atomic<double> currentSampleRate{ 0.0 };

    // ==================================================================
    // 内部最大サイズ (Audio Thread安全ガード用)
    // ==================================================================
    int maxInternalBlockSize = 0;

    // ── AGC適用 (Audio Thread 内で呼ばれる) ──
    void processAGC(juce::dsp::AudioBlock <double > & block);
    double calculateAGCGain(double inputEnv, double outputEnv) const noexcept;

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

    convo::ScopedAlignedPtr<double> scratchBuffer;
    int scratchCapacity = 0;
    convo::ScopedAlignedPtr<double> dryBypassBuffer;
    int dryBypassCapacity = 0;

    convo::ScopedAlignedPtr<double> parallelInputBuffer;
    convo::ScopedAlignedPtr<double> parallelWorkBuffer;
    convo::ScopedAlignedPtr<double> parallelAccumBuffer;
    convo::ScopedAlignedPtr<double> structureOldOutBuffer;
    convo::ScopedAlignedPtr<double> structureNewOutBuffer;
    int parallelBufferCapacity = 0;
    int structureXfadeBufferCapacity = 0;

    std::atomic<float> nonlinearSaturation { 0.2f };
    std::atomic<FilterStructure> requestedStructure { FilterStructure::Serial };
    std::atomic<FilterStructure> activeStructure { FilterStructure::Serial };
};
