//============================================================================
#pragma once
// ConvolverProcessor.h  ── v0.3 (JUCE 8.0.12対応 / MKL NUC 統合)
//
// FFTベースコンボリューションプロセッサー
//
// ■ 信号処理フロー:
//   Input → [Convolver <-> EQ] → Output
//
// ■ 用途:
//   - ルームリバーブ（ホール、スタジオ等の空間シミュレーション）
//   - スピーカーキャビネット/マイク特性の畳み込み
//   - 位相補正/周波数特性補正
//
// ■ スレッド安全設計:
//   - loadImpulseResponse(): Message Thread で実行。バックグラウンドスレッドで読み込みを行い、完了後に atomic に差し替えます (RCU)。ロード中も音切れなく古いIRで処理を継続します。
//   - process(): Audio Thread で実行。MKLNonUniformConvolver を使用してパーティション分割畳み込みを行います。
//   - パラメータ: std::atomic でスレッドセーフ。Audio Thread内でのメモリ確保やIR再ロードは行いません。
//
// ■ エンジン:
//   Intel MKL Non-Uniform Partitioned Convolution (NUC) エンジンを使用します。
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <memory>
#include <vector>
#include <array>
#include <functional>
#include <deque>
#include "AlignedAllocation.h"
#include "MKLNonUniformConvolver.h"

class ConvolverProcessor : public juce::ChangeBroadcaster,
                           private juce::Timer
{
public:
    class Listener
    {
    public:
        virtual ~Listener() = default;
        virtual void convolverParamsChanged(ConvolverProcessor* processor) = 0;
    };

    void addListener(Listener* listener) { listeners.add(listener); }
    void removeListener(Listener* listener) { listeners.remove(listener); }

    // 波形表示の解像度
    static constexpr int WAVEFORM_POINTS = 512;

    // IR処理定数
    static constexpr int MIN_PARTITION_SIZE = 256;
    static constexpr int PARTITION_SIZE_MULTIPLIER = 2;
    static constexpr float IR_SILENCE_THRESHOLD = 1.0e-6f;
    static constexpr float MIX_MIN = 0.0f;
    static constexpr float MIX_MAX = 1.0f;
    static constexpr float SMOOTHING_TIME_MIN_SEC = 0.01f;   // 10ms
    static constexpr float SMOOTHING_TIME_MAX_SEC = 0.5f;    // 500ms
    static constexpr float SMOOTHING_TIME_DEFAULT_SEC = 0.05f; // 50ms
    static constexpr float IR_LENGTH_MIN_SEC = 0.5f;
    static constexpr float IR_LENGTH_MAX_SEC = 3.0f;
    static constexpr float IR_LENGTH_DEFAULT_SEC = 1.0f;

    // DelayLine用定数 (Audio Threadでのメモリ確保防止)
    // IRの最大長(kMaxIRCap)と最大ブロックサイズをカバーする値を設定
    // 3s @ 192kHz = 576000 samples. Next power of 2 is 1048576.
    static constexpr int MAX_IR_LATENCY = 2097152; // 2^21 (3.0s @ 384kHz = ~1.15M samples をカバー)
    // 最適なFFTパフォーマンスのために、この値は2の累乗である必要があります。
    // また、MIN_PARTITION_SIZE以上である必要があります。
    // MAX_PARTITION_SIZEもまた、maxBlockSize * oversamplingFactor以上である必要があります。
    static constexpr int MAX_BLOCK_SIZE = 524288;  // 65536 * 8 (Safe for 8x oversampling of max input block)
    static constexpr int MAX_TOTAL_DELAY = MAX_IR_LATENCY + MAX_BLOCK_SIZE;
    // リングバッファのラップアラウンドをビットマスクで高速化するため、2の累乗サイズを使用
    static constexpr int DELAY_BUFFER_SIZE = 4194304; // 2^22 (approx 4M samples > MAX_TOTAL_DELAY)
    static constexpr int DELAY_BUFFER_MASK = DELAY_BUFFER_SIZE - 1;
    static constexpr double CONVOLUTION_HEADROOM_GAIN = 1.0; // 0.0 dB (Unity Gain - Headroom is baked into IR)

    ConvolverProcessor();
    ~ConvolverProcessor();

    //----------------------------------------------------------
    // 準備（Audio Thread開始前）
    //----------------------------------------------------------
    void prepareToPlay(double sampleRate, int samplesPerBlock);
    void releaseResources();

    //----------------------------------------------------------
    // インパルス応答読み込み（Message Thread）
    //
    // 対応形式: WAV, AIFF, FLAC
    // @return true=読み込み開始成功（非同期）, false=開始失敗
    //----------------------------------------------------------
    bool loadImpulseResponse(const juce::File& irFile, bool optimizeForRealTime = false);

    //----------------------------------------------------------
    // メイン処理（Audio Thread）
    //
    //----------------------------------------------------------
    void process(juce::dsp::AudioBlock<double>& block);

    //----------------------------------------------------------
    // バイパス制御
    //----------------------------------------------------------
    void setBypass(bool shouldBypass);
    bool isBypassed() const { return bypassed.load(); }

    //----------------------------------------------------------
    // Dry/Wet Mix (0.0 = Dry only, 1.0 = Wet only)
    //----------------------------------------------------------
    void setMix(float mixAmount);
    float getMix() const;

    //----------------------------------------------------------
    // Minimum Phase Mode
    //----------------------------------------------------------
    void setUseMinPhase(bool useMinPhase);
    bool getUseMinPhase() const { return useMinPhase.load(); }

    //----------------------------------------------------------
    // Smoothing Time
    //----------------------------------------------------------
    void setSmoothingTime(float timeSec);
    float getSmoothingTime() const;

    //----------------------------------------------------------
    // IR Length
    //----------------------------------------------------------
    void setTargetIRLength(float timeSec);
    float getTargetIRLength() const;

    //----------------------------------------------------------
    // 状態リセット
    //----------------------------------------------------------
    void reset();

    //----------------------------------------------------------
    // 状態取得
    //----------------------------------------------------------
    bool isIRLoaded() const { return convolution.load() != nullptr; }
    juce::String getIRName() const { return irName; }
    int getIRLength() const { return irLength.load(std::memory_order_acquire); }
    juce::String getLastError() const { return lastError; }
    float getLoadProgress() const { return loadProgress.load(); }
    int getCurrentBufferSize() const { return currentBufferSize; }
    int getLatencySamples() const;

    //----------------------------------------------------------
    // 波形表示用データ取得
    //----------------------------------------------------------
    std::vector<float> getIRWaveform() const;

    //----------------------------------------------------------
    // 周波数特性表示用データ取得
    //----------------------------------------------------------
    std::vector<float> getIRMagnitudeSpectrum() const;
    double getIRSpectrumSampleRate() const;

    //----------------------------------------------------------
    // State Management
    //----------------------------------------------------------
    juce::ValueTree getState() const;
    void setState (const juce::ValueTree& state);

    //----------------------------------------------------------
    // リビルド (サンプルレート変更時など)
    //----------------------------------------------------------
    void rebuildAllIRs();
    void rebuildAllIRsSynchronous(std::function<bool()> shouldCancel = nullptr);

    // 他のインスタンスから状態を同期 (AudioEngine用)
    void syncStateFrom(const ConvolverProcessor& other);
    void syncParametersFrom(const ConvolverProcessor& other);
    void shareConvolutionEngineFrom(const ConvolverProcessor& other);
    void refreshLatency();

    // 【NUCエンジン構築専用エントリポイント】
    // LoaderThreadからIRデータを受け取り、メッセージスレッド上で
    // SetImpulse / DftiCommitDescriptor / mkl_malloc を安全に実行する
    void finalizeNUCEngineOnMessageThread(convo::ScopedAlignedPtr<double> irL, // This is called from LoaderThread
                                          convo::ScopedAlignedPtr<double> irR,
                                          int length,
                                          double sr,
                                          int peakDelay,
                                          int maxFFTSize,
                                          int knownBlockSize,
                                          int firstPartition,
                                          int preferredCallSize,
                                          bool isRebuild,
                                          const juce::File& irFile,
                                          double scaleFactor, // This is for newConv->init
                                          std::shared_ptr<juce::AudioBuffer<double>> loadedIR,
                                          std::shared_ptr<juce::AudioBuffer<double>> displayIR);

    // 可視化データ生成の制御 (DSP用インスタンスでは無効化してメモリを節約)
    void setVisualizationEnabled(bool enabled) { visualizationEnabled = enabled; }
    bool isVisualizationEnabled() const { return visualizationEnabled; }

    // ガベージコレクション (Message Threadから定期的に呼ぶ)
    void cleanup();
    void forceCleanup();

private:
    void timerCallback() override;
    struct StereoConvolver;
    class LoaderThread;
    void applyNewState(StereoConvolver* newConv, std::shared_ptr<juce::AudioBuffer<double>> loadedIR, double loadedSR, int targetLength, bool isRebuild, const juce::File& file, double scaleFactor, std::shared_ptr<juce::AudioBuffer<double>> displayIR);
    void handleLoadError(const juce::String& error);
    void createWaveformSnapshot (const juce::AudioBuffer<double>& irBuffer);
    void createFrequencyResponseSnapshot (const juce::AudioBuffer<double>& irBuffer, double sampleRate);
    int computeTargetIRLength(double sampleRate, int originalLength) const;

    // Stereo processing wrapper
    struct StereoConvolver
    {
        double* irData[2] = { nullptr, nullptr };

        std::array<convo::ScopedAlignedPtr<convo::MKLNonUniformConvolver>, 2> nucConvolvers;
        int irDataLength = 0;

        int latency = 0;
        int irLatency = 0; // IR由来の遅延 (ピーク位置)
        int callQuantumSamples = 0;    // Audio Thread でのNUC呼び出し量子
        int prewarmedMaxSamples = 0;   // Message Thread でプリウォーム済みの最大呼び出し長

        // Clone用に初期化パラメータを保存
        double storedSampleRate = 0.0;
        int storedMaxFFTSize = 0;
        int storedKnownBlockSize = 0;
        int storedFirstPartition = 0;
        double storedScale = 1.0;

        StereoConvolver() = default;

        ~StereoConvolver() {
            if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
            if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }
        }

        // Intrusive Reference Counting
        mutable std::atomic<int> refCount { 0 };
        void addRef() const { refCount.fetch_add(1, std::memory_order_relaxed); }
        void release() const { if (refCount.fetch_sub(1, std::memory_order_acq_rel) == 1) delete this; }

        // コピーコンストラクタは禁止 (NUCエンジンは複製コストが高く、状態を持つため)
        StereoConvolver(const StereoConvolver& other) = delete;

        // 代入演算子は禁止 (使用しないため)
        StereoConvolver& operator=(const StereoConvolver&) = delete;

        bool init(double* irL, double* irR, int length, double sr, int peakDelay, int maxFFTSize, int knownBlockSize, int firstPartition, int preferredCallSize, double scale = 1.0)
        {
            // Safety: Free existing data if init is called multiple times (Leak prevention)
            if (irData[0]) { convo::aligned_free(irData[0]); irData[0] = nullptr; }
            if (irData[1]) { convo::aligned_free(irData[1]); irData[1] = nullptr; }

            // Ownership transfer
            irData[0] = irL;
            irData[1] = irR;
            irDataLength = length;
            this->irLatency = peakDelay;
            callQuantumSamples = juce::jmax(1, preferredCallSize);
            prewarmedMaxSamples = callQuantumSamples;
            storedSampleRate = sr;
            storedMaxFFTSize = maxFFTSize;
            storedKnownBlockSize = knownBlockSize;
            storedFirstPartition = firstPartition;
            storedScale = scale;

            {
                // ── MKL Non-Uniform Partitioned Convolution (NUC) ──
                // new完全禁止 → aligned_malloc + placement new (規約準拠)
                void* rn0 = convo::aligned_malloc(sizeof(convo::MKLNonUniformConvolver), 64);
                new (rn0) convo::MKLNonUniformConvolver();
                nucConvolvers[0].reset(static_cast<convo::MKLNonUniformConvolver*>(rn0));

                void* rn1 = convo::aligned_malloc(sizeof(convo::MKLNonUniformConvolver), 64);
                new (rn1) convo::MKLNonUniformConvolver();
                nucConvolvers[1].reset(static_cast<convo::MKLNonUniformConvolver*>(rn1));

                if (nucConvolvers[0]->SetImpulse(irData[0], irDataLength, knownBlockSize, scale) &&
                    nucConvolvers[1]->SetImpulse(irData[1], irDataLength, knownBlockSize, scale))
                {
                    latency   = nucConvolvers[0]->getLatency();
                    DBG("Convolver: NUC Engine Active. Latency: " << latency << " samples");
                    return true;
                }
                // NUC セットアップ失敗
                nucConvolvers[0].reset();
                nucConvolvers[1].reset();
                return false;
            }
        }

        // Deep Copyを作成する。
        // 失敗時 (MKLメモリ確保失敗等) は nullptr を返す。呼び出し元で必ずチェックすること。
        StereoConvolver* clone() const
        {
            auto newConv = new StereoConvolver();
            if (irDataLength > 0 && irData[0] && irData[1])
            {
                // メモリを新規確保してコピー (RAIIによる例外安全性確保)
                // init() に所有権を渡すまでは unique_ptr で管理し、途中で例外が発生してもリークしないようにする
                convo::ScopedAlignedPtr<double> l(static_cast<double*>(convo::aligned_malloc(irDataLength * sizeof(double), 64)));
                convo::ScopedAlignedPtr<double> r(static_cast<double*>(convo::aligned_malloc(irDataLength * sizeof(double), 64)));

                if (l && r)
                {
                    std::memcpy(l.get(), irData[0], irDataLength * sizeof(double));
                    std::memcpy(r.get(), irData[1], irDataLength * sizeof(double));

                    // [Bug Fix] 戻り値を確認する。
                    // init() 失敗時は irData の所有権は newConv に移っているが
                    // nucConvolvers は nullptr のまま。delete して nullptr を返す。
                    if (!newConv->init(l.release(), r.release(), irDataLength, storedSampleRate, irLatency, storedMaxFFTSize, storedKnownBlockSize, storedFirstPartition, callQuantumSamples, storedScale))
                    {
                        delete newConv;
                        return nullptr;
                    }
                }
                else
                {
                    // aligned_malloc 失敗
                    delete newConv;
                    return nullptr;
                }
            }
            return newConv;
        }

        void reset();
        void process(int channel, const double* in, double* out, int numSamples);
    };

    // Note: trashBin is used to hold old Convolution objects that the Audio Thread may still be using.
    std::atomic<StereoConvolver*> convolution { nullptr }; // Raw pointer for Audio Thread (Lock-free)
    StereoConvolver* activeConvolution = nullptr; // Ownership holder for Message Thread
    std::vector<std::pair<StereoConvolver*, uint32>> trashBin; // Time-based GC
    juce::CriticalSection trashBinLock;
    std::atomic<bool> isLoading { false };
    std::atomic<bool> isRebuilding { false };
    std::unique_ptr<LoaderThread> activeLoader;
    std::deque<std::unique_ptr<LoaderThread>> loaderTrashBin;
    std::atomic<float> loadProgress { 0.0f };
    juce::String lastError;
    void setLoadingProgress(float p) { loadProgress.store(p); }

    juce::dsp::ProcessSpec currentSpec = { 48000.0, 512, 2 };

    juce::ListenerList<Listener> listeners;

    //----------------------------------------------------------
    // レイテンシー補正用ディレイ
    //----------------------------------------------------------
    // juce::dsp::DelayLine<double> delayLine; // Replaced with custom AVX2 ring buffer
    convo::ScopedAlignedPtr<double> delayBuffer[2]; // L/R separate buffers
    int delayBufferCapacity = 0;
    int delayWritePos = 0;
    juce::SmoothedValue<double> latencySmoother;
    // ドップラー効果対策: クロスフェード用
    juce::SmoothedValue<double> crossfadeGain;
    double oldDelay = 0.0;

    //----------------------------------------------------------
    // パラメータ（atomic）
    //----------------------------------------------------------
    std::atomic<bool> bypassed{false};
    std::atomic<float> mixTarget{1.0f}; // UIからのターゲット値 (0.0-1.0)
    juce::SmoothedValue<double> mixSmoother; // オーディオスレッドでの平滑化用
    std::atomic<bool> useMinPhase{false};
    std::atomic<float> targetIRLengthSec{IR_LENGTH_DEFAULT_SEC};
    std::atomic<float> smoothingTimeSec{SMOOTHING_TIME_DEFAULT_SEC};

    //----------------------------------------------------------
    // IR情報
    //----------------------------------------------------------
    juce::String irName;
    std::atomic<int> irLength { 0 };  // rebuildThread(read) と Message Thread(write) 間のデータレース防止
    // --- Visualization Data (accessed by worker and message threads) ---
    std::vector<float> irWaveform;
    std::vector<float> irMagnitudeSpectrum;
    double irSpectrumSampleRate = 0.0;
    mutable juce::CriticalSection visualizationDataLock;

    juce::File currentIrFile;
    juce::CriticalSection irFileLock;
    std::atomic<bool> currentIrOptimized { false };
    std::shared_ptr<juce::AudioBuffer<double>> originalIR; // 元IR保持 (リサンプリング/トリミング用)
    double originalIRSampleRate = 0.0;
    // MKL/AVX-512用に64byteアライメントを保証するアロケータを使用
    double currentIRScale = 1.0; // IRのスケールファクター (Auto Makeup + Safety Margin)
    convo::ScopedAlignedPtr<float> cachedFFTBuffer; // FFT計算用キャッシュ (Message Thread)
    int cachedFFTBufferCapacity = 0;
    std::atomic<double> currentSampleRate { 0.0 };

    DFTI_DESCRIPTOR_HANDLE fftHandle = nullptr;
    int fftHandleSize = 0;

    //----------------------------------------------------------
    // Dry信号バッファ（Mix用）
    //----------------------------------------------------------
    juce::AudioBuffer<double> dryBuffer;
    convo::ScopedAlignedPtr<double> dryBufferStorage[2]; // Aligned storage for dryBuffer
    int dryBufferCapacity = 0;
    juce::AudioBuffer<double> smoothingBuffer; // スムーシングゲイン計算用 (Audio Threadでのメモリ確保回避)
    convo::ScopedAlignedPtr<double> smoothingBufferStorage[2]; // Aligned storage for smoothingBuffer
    int smoothingBufferCapacity = 0;

    // クロスフェード用バッファ
    juce::AudioBuffer<double> oldDryBuffer;
    convo::ScopedAlignedPtr<double> oldDryBufferStorage[2];
    int oldDryBufferCapacity = 0;

    // Wet信号用一時バッファ (StereoConvolver::process用)
    convo::ScopedAlignedPtr<double> wetBufferStorage[2];
    int wetBufferCapacity = 0;

    //----------------------------------------------------------
    // 準備完了フラグ
    //----------------------------------------------------------
    std::atomic<bool> isPrepared { false };
    bool visualizationEnabled = true; // Default true (for UI instance)
    int currentBufferSize = 0; // prepareToPlayで更新される
    double currentSmoothingTimeSec = SMOOTHING_TIME_DEFAULT_SEC; // mixSmootherに設定されている現在の時間

    JUCE_DECLARE_WEAK_REFERENCEABLE(ConvolverProcessor)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverProcessor)

};
