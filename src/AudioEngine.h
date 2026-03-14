//============================================================================
#pragma once
// AudioEngine.h  ── v0.2 (JUCE 8.0.12対応)
//
// オーディオエンジン - AudioSource実装
//
// ■ JUCE AudioSource仕様:
//   - getNextAudioBlock(...) : オーディオ処理コールバック
//   - prepareToPlay(...) : 再生準備（バッファ確保など）
//   - releaseResources() : リソース解放（再生停止時）
//
// ■ スレッド安全性とリアルタイム制約:
//   - getNextAudioBlock: Audio Threadで実行されます。
//     - リアルタイム制約があります（ブロック不可、ロック不可、メモリ割り当て不可、IR再ロード不可）。
//   - prepareToPlay / releaseResources: Audio Thread の開始前/終了後に Message Thread から呼ばれます。
//   - パラメータ設定: Message Thread から呼ばれます。std::atomic を使用して Audio Thread と安全に同期します (RCUパターン)。
//   - readFromFifo: Message Thread (Timer) から呼ばれます。FIFOバッファからデータを取得します。
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <cstring>
#include <array>
#include <vector>
#include <juce_dsp/juce_dsp.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <immintrin.h>

#include "AlignedAllocation.h"
#include "CustomInputOversampler.h"
#include "ConvolverProcessor.h"
#include "EQProcessor.h"
#include "PsychoacousticDither.h"
#include "OutputFilter.h"

class AudioEngine : public juce::AudioSource,
                  public juce::ChangeBroadcaster,
                  private juce::ChangeListener,
                  private EQProcessor::Listener,
                  private ConvolverProcessor::Listener,
                   private juce::Timer
{
public:
    using SampleType = double; // 内部DSP精度 (JUCE推奨)

     enum class ProcessingOrder
    {
        ConvolverThenEQ,
        EQThenConvolver
    };

    enum class AnalyzerSource
    {
        Input,
        Output
    };

    enum class OversamplingType
    {
        IIR,
        LinearPhase
    };

    class Listener
    {
     public:
         virtual ~Listener() = default;
        virtual void eqSettingsChanged() = 0;
    };

    // FIFO設定
    static constexpr int FIFO_SIZE = 1048576;  // Lock-free FIFO サイズ (2^20, SAFE_MAX_BLOCK_SIZE * 8x OS をカバー)

    // ── 安全性制限 ──
    static constexpr double SAFE_MIN_SAMPLE_RATE = 8000.0;
    static constexpr double SAFE_MAX_SAMPLE_RATE = 384000.0;
    static constexpr int    SAFE_MAX_BLOCK_SIZE  = 65536; // 8x Oversampling対応のため拡張

    //----------------------------------------------------------
    // コンストラクタ
    //----------------------------------------------------------
    AudioEngine();
    ~AudioEngine() override;
    void initialize();

    //----------------------------------------------------------
    // AudioSource インターフェース
    //----------------------------------------------------------
    void prepareToPlay (int samplesPerBlockExpected, double sampleRate) override;
    void releaseResources() override;
    void getNextAudioBlock (const juce::AudioSourceChannelInfo& bufferToFill) override;
    void processBlockDouble (juce::AudioBuffer<double>& buffer);
    void changeListenerCallback(juce::ChangeBroadcaster* source) override;
    void eqBandChanged(EQProcessor* processor, int bandIndex) override;
    void eqGlobalChanged(EQProcessor* processor) override;
    void convolverParamsChanged(ConvolverProcessor* processor) override;
    void timerCallback() override;

    //----------------------------------------------------------
    // 外部インターフェース (Message Thread)
    //----------------------------------------------------------
    ConvolverProcessor& getConvolverProcessor() { return uiConvolverProcessor; }
    EQProcessor& getEQProcessor() { return uiEqProcessor; }

    double getSampleRate() const { return currentSampleRate.load(); }
    double getProcessingSampleRate() const;

    // 【Fix Bug #8】gainToDecibels (std::log10 / libm) を Audio Thread から排除。
    // Audio Thread は linear gain を inputLevelLinear / outputLevelLinear に格納し、
    // getter (UI Thread) で dB 変換する。
    float getInputLevel() const
    {
        const float linear = inputLevelLinear.load(std::memory_order_relaxed);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }
    float getOutputLevel() const
    {
        const float linear = outputLevelLinear.load(std::memory_order_relaxed);
        return (linear > LEVEL_METER_MIN_MAG)
               ? juce::Decibels::gainToDecibels(linear)
               : LEVEL_METER_MIN_DB;
    }




    int getFifoNumReady() const { return audioFifo.getNumReady(); }
    void readFromFifo(float* dest, int numSamples);
    void skipFifo(int numSamples);

    void calcEQResponseCurve(float* outMagnitudesL, float* outMagnitudesR, const std::complex<double>* zArray, int numPoints, double sampleRate);

    // パラメータ設定 (Thread-safe)
    void setEqBypassRequested (bool shouldBypass) noexcept;
    void setConvolverBypassRequested (bool shouldBypass) noexcept;

    void setConvolverUseMinPhase(bool useMinPhase);
    bool getConvolverUseMinPhase() const;

    void requestEqPreset (int presetIndex);
    void requestEqPresetFromText(const juce::File& file);
    void requestConvolverPreset (const juce::File& irFile);

    void requestLoadState (const juce::ValueTree& state);
    juce::ValueTree getCurrentState() const;

    void setProcessingOrder(ProcessingOrder order) { currentProcessingOrder.store(order); }
    ProcessingOrder getProcessingOrder() const { return currentProcessingOrder.load(); }

    void setAnalyzerSource(AnalyzerSource source) { currentAnalyzerSource.store(source); }
    AnalyzerSource getAnalyzerSource() const { return currentAnalyzerSource.load(); }
    void setAnalyzerEnabled(bool enabled) noexcept { analyzerEnabled.store(enabled, std::memory_order_release); }
    bool isAnalyzerEnabled() const noexcept { return analyzerEnabled.load(std::memory_order_acquire); }

    void setInputHeadroomDb(float db);
    float getInputHeadroomDb() const;

    void setOutputMakeupDb(float db);
    float getOutputMakeupDb() const;

    void setDitherBitDepth(int bitDepth);
    int getDitherBitDepth() const;

    void setSoftClipEnabled(bool enabled);
    bool isSoftClipEnabled() const;

    void setSaturationAmount(float amount);
    float getSaturationAmount() const;

    void setOversamplingFactor(int factor);
    int getOversamplingFactor() const;

    void setOversamplingType(OversamplingType type);
    OversamplingType getOversamplingType() const;

    // ────────────────────────────────────────────────────────────────
    // 出力周波数フィルター設定 (Thread-safe)
    //
    // convHCMode / convLCMode: ① コンボルバー最終段の場合に使用
    // eqLPFMode              : ② EQ最終段の場合に使用
    // ────────────────────────────────────────────────────────────────
    void setConvHCFilterMode(convo::HCMode mode) noexcept;
    convo::HCMode getConvHCFilterMode() const noexcept;

    void setConvLCFilterMode(convo::LCMode mode) noexcept;
    convo::LCMode getConvLCFilterMode() const noexcept;

    void setEqLPFFilterMode(convo::HCMode mode) noexcept;
    convo::HCMode getEqLPFFilterMode() const noexcept;

private:
    //==============================================================================
    // 内部クラス定義
    class UltraHighRateDCBlocker {
    private:
        double m_prev_x = 0.0;
        double m_prev_y = 0.0;
        double m_R = 0.999999; // デフォルト値

    public:
        // サンプリングレートに合わせて R を計算
        // 注意: std::exp() を使用するため Audio Thread から呼んではならない。
        //       DSPCore::prepare() (rebuildThreadLoop) からのみ呼ぶこと。
        void init(double sampleRate, double cutoffHz) noexcept
        {
            // R = exp(-2 * PI * cutoff / sampleRate)
            m_R = std::exp(-2.0 * juce::MathConstants<double>::pi * cutoffHz / sampleRate);
            // init() 後に R が NaN/Inf になることはないが念のため保護
            if (! std::isfinite(m_R) || m_R <= 0.0 || m_R >= 1.0)
                m_R = 0.999999;
            reset();
        }

        void reset() noexcept
        {
            m_prev_x = 0.0;
            m_prev_y = 0.0;
        }

        // ループフュージョン最適化用ヘルパー
        void loadState() noexcept {
            px_local = m_prev_x;
            py_local = m_prev_y;
        }

        void saveState() noexcept {
            m_prev_x = px_local;
            m_prev_y = py_local;
        }

        inline void processSample(double& sample) noexcept
        {
            const double r = m_R;
            constexpr double kDenormalThreshold = 1.0e-20;

            const double curr_x = sample;
            double curr_y = curr_x - px_local + r * py_local;

            if (!std::isfinite(curr_y) || std::abs(curr_y) < kDenormalThreshold) curr_y = 0.0;

            px_local = curr_x;
            py_local = curr_y;
            sample = curr_y;
        }


        // 64byteアライメントされたバッファを高速処理 (Audio Thread 安全)
        void process(double* data, int numSamples) noexcept
        {
            double px = m_prev_x;
            double py = m_prev_y;
            const double r = m_R;
            constexpr double kDenormalThreshold = 1.0e-20;

            int i = 0;
            const int vEnd = numSamples / 4 * 4;

            if (vEnd > 0)
            {
                const double r2 = r * r;
                const double r3 = r2 * r;
                const double r4 = r3 * r;
                const __m256d vR = _mm256_set1_pd(r);
                const __m256d vR2 = _mm256_set1_pd(r2);
                // vPrevYFactors = [R, R^2, R^3, R^4]
                const __m256d vPrevYFactors = _mm256_set_pd(r4, r3, r2, r);
                const __m256d vThresh = _mm256_set1_pd(kDenormalThreshold);
                const __m256d vInfThresh = _mm256_set1_pd(1.0e100); // Inf判定用閾値
                const __m256d vSignMask = _mm256_set1_pd(-0.0);

                for (; i < vEnd; i += 4)
                {
                    // Load x[i..i+3]
                    __m256d vx = _mm256_load_pd(data + i);

                    // Prepare [x[i-1], x[i], x[i+1], x[i+2]] using px (= x[i-1] for all i).
                    //
                    // [Bug Fix] data はインプレースで書き換えられるため、i > 0 での
                    //   _mm256_loadu_pd(data + i - 1) は data[i-1] = y[i-1] を読んでしまい、
                    //   差分式 U = x[i] - x[i-1] が U = x[i] - y[i-1] に化ける。
                    //   px は各反復末尾で x[i+3] に更新されるため、常に正しい x[i-1] を保持する。
                    //   したがって permute+blend による構築を全反復で使用する。
                    __m256d t = _mm256_permute4x64_pd(vx, _MM_SHUFFLE(2, 1, 0, 0));
                    __m256d vpx = _mm256_set1_pd(px);
                    __m256d v_prev_x = _mm256_blend_pd(t, vpx, 1); // [px, x[i], x[i+1], x[i+2]]

                    // U = x - prev_x
                    __m256d vu = _mm256_sub_pd(vx, v_prev_x);

                    // Parallel Prefix Sum
                    // S0 = U
                    // S1 = S0 + R * (S0 << 1)
                    __m256d v_shift1 = _mm256_permute4x64_pd(vu, _MM_SHUFFLE(2, 1, 0, 0));
                    v_shift1 = _mm256_blend_pd(v_shift1, _mm256_setzero_pd(), 1);
                    __m256d vs1 = _mm256_fmadd_pd(vR, v_shift1, vu);

                    // S2 = S1 + R^2 * (S1 << 2)
                    __m256d v_shift2 = _mm256_permute4x64_pd(vs1, _MM_SHUFFLE(1, 0, 0, 0));
                    v_shift2 = _mm256_blend_pd(v_shift2, _mm256_setzero_pd(), 3); // mask 0011
                    __m256d vy = _mm256_fmadd_pd(vR2, v_shift2, vs1);

                    // Add contribution from prev_y
                    __m256d vpy = _mm256_set1_pd(py);
                    vy = _mm256_fmadd_pd(vPrevYFactors, vpy, vy);

                    // Anti-Denormal & Inf/NaN Check
                    __m256d abs_y = _mm256_andnot_pd(vSignMask, vy);
                    // |y| >= denormal_thresh
                    __m256d mask = _mm256_cmp_pd(abs_y, vThresh, _CMP_GE_OQ);
                    // |y| < inf_thresh (NaNはFalseになるためここで除去される)
                    mask = _mm256_and_pd(mask, _mm256_cmp_pd(abs_y, vInfThresh, _CMP_LT_OQ));
                    vy = _mm256_and_pd(vy, mask);

                    _mm256_store_pd(data + i, vy);

                    // Update state for next iter
                    _mm_storeh_pd(&px, _mm256_extractf128_pd(vx, 1));
                    _mm_storeh_pd(&py, _mm256_extractf128_pd(vy, 1));
                }
            }

            for (; i < numSamples; ++i) {
                const double curr_x = data[i];
                double curr_y = curr_x - px + r * py;

                // Anti-Denormal: デノーマル数をゼロに落とす
                // 【堅牢性向上】NaN/Infもチェックしてゼロに丸める
                if (!std::isfinite(curr_y) || std::abs(curr_y) < kDenormalThreshold) curr_y = 0.0;

                px = curr_x;
                py = curr_y;
                data[i] = curr_y;
            }

            // 【堅牢性向上】最終的な状態変数を保存する前にサニタイズする
            // これにより、万が一 px や py が NaN/Inf になっても、次回の process() 呼び出しに
            // 不正な状態が引き継がれるのを防ぐ。
            // std::abs(NaN) < limit は false になるため、NaN は 0.0 にリセットされる。
            m_prev_x = (std::abs(px) < 1.0e15) ? px : 0.0; // 1.0e15 は Inf を捕捉するための巨大な閾値
            m_prev_y = (std::abs(py) < 1.0e15) ? py : 0.0;
        }

    private:
        // ループフュージョン用ローカル状態変数
        double px_local = 0.0;
        double py_local = 0.0;
    };

    //----------------------------------------------------------
     // DSPコア (Audio Threadで実行される処理のコンテナ)
    //----------------------------------------------------------
    struct DSPCore
    {
        struct ProcessingState
        {
             bool eqBypassed;
            bool convBypassed;
            ProcessingOrder order;
            AnalyzerSource analyzerSource;
            bool analyzerEnabled;
            bool softClipEnabled;
            float saturationAmount;
        double inputHeadroomGain;
            double outputMakeupGain;
            // 出力周波数フィルターモード
            convo::HCMode convHCMode;  // ① ハイカットモード
            convo::LCMode convLCMode;  // ① ローカットモード
            convo::HCMode eqLPFMode;   // ② EQローパスモード
        };

DSPCore();
        DSPCore(const DSPCore&) = delete;
        DSPCore& operator=(const DSPCore&) = delete;

    ~DSPCore()
    {
        // Explicitly clean up convolver resources to ensure no WDL memory is leaked,
        // especially for instances that are destroyed from the trash bin.
        convolver.forceCleanup();
    }

    void prepare(double sampleRate, int samplesPerBlock, int bitDepth, int manualOversamplingFactor, OversamplingType oversamplingType);
    void reset();
    void process(const juce::AudioSourceChannelInfo& bufferToFill, juce::AbstractFifo& audioFifo,
                 juce::AudioBuffer<float>& audioFifoBuffer, std::atomic<float>& inputLevelLinear,
                 std::atomic<float>& outputLevelLinear, const ProcessingState& state);
    void processDouble(juce::AudioBuffer<double>& buffer,
                       juce::AbstractFifo& audioFifo,
                       juce::AudioBuffer<float>& audioFifoBuffer,
                       std::atomic<float>& inputLevelLinear,
                       std::atomic<float>& outputLevelLinear,
                       const ProcessingState& state);
        ConvolverProcessor convolver;
        EQProcessor eq;
        // 【最適化】出力 / 入力 DC 除去を UltraHighRateDCBlocker (1次IIR, ブロックモード) に統一。
        // 旧 DCBlocker (4次 Butterworth, サンプル単位) は 1 サンプルあたり ~20 演算を要したが、
        // 1次 IIR は ~4 演算で済みかつ process(data, N) ブロック呼び出しによりメモリアクセスも効率化。
        // DC 除去の目的 (3Hz 以下のカット) には 1 次で十分。
        UltraHighRateDCBlocker dcBlockerL, dcBlockerR;
        UltraHighRateDCBlocker inputDCBlockerL, inputDCBlockerR;
        UltraHighRateDCBlocker osDCBlockerL, osDCBlockerR; // Oversampling後のDC除去用
        ::convo::PsychoacousticDither dither;
        // 出力周波数フィルター (① ハイカット/ローカット / ② ローパス/ハイパス)
        convo::OutputFilter outputFilter;

        CustomInputOversampler oversampling;
        size_t oversamplingFactor = 1;
        int ditherBitDepth = 0; // DSPCore内でディザリング判定に使用
        double sampleRate = 0.0;

    // 【パッチ3】MKL用rawアライメントバッファ（vector完全排除・ガイドライン厳守）
        convo::ScopedAlignedPtr<double> alignedL;
        convo::ScopedAlignedPtr<double> alignedR;
        int alignedCapacity = 0;                  // 現在確保済み容量（再確保判定用）

        int maxSamplesPerBlock = 0;               // 入力側最大ブロックサイズ (SAFE_MAX_BLOCK_SIZE)

        // ─────────────────────────────────────────────────────────────
        // 【Issue 3 修正】内部処理用最大バッファサイズ
        // 理由: Oversampling有効時（最大8x）、processSamplesUp()後の
        //      ブロックサイズがSAFE_MAX×8になるため。
        //      固定で×8確保することでRCU再構築時のresizeを完全排除。
        //      メモリ増加 ≈ 8.4MB（現代PCでは無視できるレベル）
        // ─────────────────────────────────────────────────────────────
        int maxInternalBlockSize = 0;             // OS考慮後の最大サイズ（常にSAFE_MAX×8）
        std::atomic<int> fadeInSamplesLeft {0};
        static constexpr int FADE_IN_SAMPLES = 2048; // 42ms @ 48kHz

        // Helpers
        float measureLevel (const juce::dsp::AudioBlock<const double>& block) const noexcept;
        void pushToFifo(const juce::dsp::AudioBlock<const double>& block,
                        juce::AbstractFifo& audioFifo,
                        juce::AudioBuffer<float>& audioFifoBuffer) const noexcept;
        // analyzerInputTap=true の場合、ヘッドルームゲイン適用前の raw 入力を
        // audioFifo / audioFifoBuffer にプッシュする。
        // これにより、インプットスペアナ/レベルメーターがヘッドルーム非適用の
        // "入力されたデータそのもの" を表示できる。
        float processInput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples,
                           double headroomGain,
                           bool analyzerInputTap,
                           juce::AbstractFifo& audioFifo,
                           juce::AudioBuffer<float>& audioFifoBuffer) noexcept;
        void processOutput(const juce::AudioSourceChannelInfo& bufferToFill, int numSamples) noexcept;
        float processInputDouble(const juce::AudioBuffer<double>& buffer, int numSamples,
                                 double headroomGain,
                                 bool analyzerInputTap,
                                 juce::AbstractFifo& audioFifo,
                                 juce::AudioBuffer<float>& audioFifoBuffer) noexcept;
        void processOutputDouble(juce::AudioBuffer<double>& buffer, int numSamples) noexcept;
    private:
        static double musicalSoftClip(double x, double threshold, double knee, double asymmetry) noexcept;
    };

    //----------------------------------------------------------
    // 処理チェーンコンポーネント
    //----------------------------------------------------------
     // UI/State管理用のインスタンス (Audio Threadでは使用しない)
    ConvolverProcessor  uiConvolverProcessor;
    EQProcessor uiEqProcessor;

    juce::AbstractFifo audioFifo { FIFO_SIZE };
    juce::AudioBuffer<float> audioFifoBuffer;
    juce::CriticalSection fifoReadLock;

    //----------------------------------------------------------
    // 状態管理
    //----------------------------------------------------------
    std::atomic<DSPCore*> currentDSP { nullptr }; // Raw pointer for Audio Thread (Lock-free)
    DSPCore* activeDSP = nullptr; // Ownership holder for Message Thread (Raw pointer)
    std::vector<std::pair<DSPCore*, uint32>> trashBin; // Time-based garbage collection for old DSPs
    juce::CriticalSection trashBinLock;

    std::atomic<double> currentSampleRate{48000.0};
    // 【Fix Bug #8】linear gain を格納 (dB変換はgetInputLevel/getOutputLevelで行う)
    std::atomic<float> inputLevelLinear{0.0f};
    std::atomic<float> outputLevelLinear{0.0f};
    std::atomic<int>   maxSamplesPerBlock{4096};

    std::atomic<bool> eqBypassRequested { false };
    std::atomic<bool> convBypassRequested { false };
    std::atomic<bool> eqBypassActive   { false };
    std::atomic<bool> convBypassActive { false };
    std::atomic<bool> rebuildRequested { false };
    std::atomic<ProcessingOrder> currentProcessingOrder{ProcessingOrder::ConvolverThenEQ};
    std::atomic<AnalyzerSource> currentAnalyzerSource { AnalyzerSource::Output };
    std::atomic<bool> analyzerEnabled { false };
    std::atomic<int> ditherBitDepth { 0 }; // 0 = 未初期化 (DeviceSettingsで最大値に設定される)
    std::atomic<bool> softClipEnabled { true };
    std::atomic<float> saturationAmount { 0.5f };
    std::atomic<int> manualOversamplingFactor { 0 }; // 0=Auto, 1=1x, 2=2x, 4=4x, 8=8x
    std::atomic<OversamplingType> oversamplingType { OversamplingType::IIR };
    std::atomic<float> inputHeadroomDb { -6.0f };
    std::atomic<double> inputHeadroomGain { 0.5011872336272722 }; // -6dB
    std::atomic<float> outputMakeupDb { 12.0f };
    std::atomic<double> outputMakeupGain { 3.981071705534972 }; // +12dB (unity: -6dB input headroom + -6dB IR safety margin)
    std::atomic<int> rebuildGeneration { 0 }; // 非同期リビルドの競合防止用

    // 出力周波数フィルターモード (Thread-safe)
    std::atomic<convo::HCMode> convHCFilterMode { convo::HCMode::Natural }; // ① ハイカット
    std::atomic<convo::LCMode> convLCFilterMode { convo::LCMode::Natural }; // ① ローカット
    std::atomic<convo::HCMode> eqLPFFilterMode  { convo::HCMode::Natural }; // ② EQローパス

    // dB変換時の下限値
    static constexpr float LEVEL_METER_MIN_DB  = -120.0f;
    static constexpr float LEVEL_METER_MIN_MAG = 1e-6f;

    // EQ応答曲線計算用の定数
    static constexpr float EQ_GAIN_EPSILON = 0.01f;          // ゲインがこれ以下なら無視
    static constexpr float EQ_UNITY_GAIN_EPSILON = 1.0e-5f;  // 1.0との比較用

    // EQ応答曲線計算用ワークバッファ (Message Thread/UI Threadで再利用)
    std::vector<float> eqTotalMagSqLBuffer;
    std::vector<float> eqTotalMagSqRBuffer;
    std::vector<float> eqBandMagSqBuffer;

    //----------------------------------------------------------
    // ヘルパー関数
    //----------------------------------------------------------
    // Note: This function performs memory allocation (including MKL) and other blocking operations
    // such as IR resampling. It MUST only be called from the message thread.
    // The prepareToPlay() method ensures this by using MessageManager::callAsync if necessary.
    void requestRebuild(double sampleRate, int samplesPerBlock);
    void commitNewDSP(DSPCore* newDSP, int generation);
    bool isRebuildObsolete(int generation) const { return generation != rebuildGeneration.load(); }

    // Worker thread for rebuilds
    void rebuildThreadLoop();
    std::thread rebuildThread;
    std::mutex rebuildMutex;
    std::condition_variable rebuildCV;
    std::atomic<bool> rebuildThreadShouldExit { false };
    bool hasPendingTask = false;

    struct RebuildTask {
        DSPCore* newDSP = nullptr;
        DSPCore* currentDSP = nullptr;
        double sampleRate;
        int samplesPerBlock;
        int ditherDepth;
        int manualOversamplingFactor;
        OversamplingType oversamplingType;
        int generation;
    };
    RebuildTask pendingTask;

    JUCE_DECLARE_WEAK_REFERENCEABLE(AudioEngine)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(AudioEngine)
};
