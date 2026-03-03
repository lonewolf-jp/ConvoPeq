//============================================================================
#pragma once
// ConvolverProcessor.h  ── v0.2 (JUCE 8.0.12対応)
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
//   - process(): Audio Thread で実行。WDL_ConvolutionEngine を使用してパーティション分割畳み込みを行います。
//   - パラメータ: std::atomic でスレッドセーフ。Audio Thread内でのメモリ確保やIR再ロードは行いません。
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <memory>
#include <vector>
#include <array>
#include <functional>
#include "WDL/convoengine.h"
#include "AlignedAllocation.h"

class ConvolverProcessor : public juce::ChangeBroadcaster
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
    int getIRLength() const { return irLength; }
    juce::String getLastError() const { return lastError; }
    float getLoadProgress() const { return loadProgress.load(); }

    //----------------------------------------------------------
    // 波形表示用データ取得
    //----------------------------------------------------------
    const std::vector<float>& getIRWaveform() const { return irWaveform; }

    //----------------------------------------------------------
    // 周波数特性表示用データ取得
    //----------------------------------------------------------
    const std::vector<float>& getIRMagnitudeSpectrum() const { return irMagnitudeSpectrum; }
    double getIRSpectrumSampleRate() const { return irSpectrumSampleRate; }

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
    void copyConvolutionEngineFrom(const ConvolverProcessor& other);
    void shareConvolutionEngineFrom(const ConvolverProcessor& other);
    void refreshLatency();

    // 可視化データ生成の制御 (DSP用インスタンスでは無効化してメモリを節約)
    void setVisualizationEnabled(bool enabled) { visualizationEnabled = enabled; }
    bool isVisualizationEnabled() const { return visualizationEnabled; }

    // ガベージコレクション (Message Threadから定期的に呼ぶ)
    void cleanup();
    void forceCleanup();

private:
    class LoaderThread;

    //----------------------------------------------------------
    // WDL Convolution Engine
    //----------------------------------------------------------
    // Stereo processing wrapper
    struct StereoConvolver
    {
        // WDL_ConvolutionEngine_Div は Non-uniform partitioned convolution を提供し、
        // 低レイテンシー動作が可能です。
        std::array<WDL_ConvolutionEngine_Div, 2> convolvers;
        double* irData[2] = { nullptr, nullptr };
        int irDataLength = 0;

        int latency = 0;
        int irLatency = 0; // IR由来の遅延 (ピーク位置)
        int callQuantumSamples = 0;    // Audio Thread でのWDL呼び出し量子
        int prewarmedMaxSamples = 0;   // Message Thread でプリウォーム済みの最大呼び出し長

        // Clone用に初期化パラメータを保存
        double storedSampleRate = 0.0;
        int storedMaxFFTSize = 0;
        int storedKnownBlockSize = 0;
        int storedFirstPartition = 0;

        using Ptr = std::shared_ptr<StereoConvolver>;

        StereoConvolver() = default;

        ~StereoConvolver() {
            if (irData[0]) convo::aligned_free(irData[0]);
            if (irData[1]) convo::aligned_free(irData[1]);
            convolvers[0].Reset();
            convolvers[1].Reset();
        }

        // コピーコンストラクタは禁止 (WDLエンジンは複製コストが高く、状態を持つため)
        StereoConvolver(const StereoConvolver& other) = delete;

        // 代入演算子は禁止 (使用しないため)
        StereoConvolver& operator=(const StereoConvolver&) = delete;

        void init(double* irL, double* irR, int length, double sr, int peakDelay, int maxFFTSize, int knownBlockSize, int firstPartition, int preferredCallSize)
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

            WDL_ImpulseBuffer impL_stack, impR_stack;
            impL_stack.samplerate = sr;
            impR_stack.samplerate = sr;
            impL_stack.SetNumChannels(1);
            impR_stack.SetNumChannels(1);

            // LoaderThreadと同様に、Resize + memcpy パターンを使用して安全に初期化
            // これにより、WDL内部でのバッファ管理が確実になり、targetLength基準での動作が保証される
            impL_stack.impulses[0].Resize(irDataLength);
            impR_stack.impulses[0].Resize(irDataLength);

            std::memcpy(impL_stack.impulses[0].Get(), irData[0], irDataLength * sizeof(double));
            std::memcpy(impR_stack.impulses[0].Get(), irData[1], irDataLength * sizeof(double));

            // WDL_ConvolutionEngine_Div の初期化
            // latency_allowed=0 で低レイテンシーモードを有効化
            convolvers[0].SetImpulse(&impL_stack, maxFFTSize, knownBlockSize, 0, 0, firstPartition);
            convolvers[1].SetImpulse(&impR_stack, maxFFTSize, knownBlockSize, 0, 0, firstPartition);

            // WDLエンジンのレイテンシーを取得 (通常は0またはパーティションサイズ依存)
            // WDL_ConvolutionEngine::GetLatency() はサンプル数を返す
            latency = convolvers[0].GetLatency();

            // プリウォーミング (Audio Threadでの後追い確保回避)
            // WDL_ConvolutionEngine_Div::Add/Avail/Get は内部キューの拡張を伴う場合があるため、
            // Message Thread側で実運用に近い呼び出しを行って必要なバッファを先に確保しておく。
            auto prewarmConvolver = [](WDL_ConvolutionEngine_Div& engine, int callSamples)
            {
                const int warmupSamples = juce::jmax(1, callSamples);
                const int latencySamples = juce::jmax(engine.GetLatency(), warmupSamples);
                const int warmupBlocks = juce::jlimit(32, 2048,
                                                      (latencySamples + warmupSamples - 1) / warmupSamples + 64);

                for (int i = 0; i < warmupBlocks; ++i)
                {
                    engine.Add(nullptr, warmupSamples, 1);
                    const int avail = engine.Avail(warmupSamples);
                    if (avail > 0)
                    {
                        (void)engine.Get();
                        engine.Advance(avail);
                    }
                }

                // 高水位プリウォーム:
                // Avail/Get を Advance せずに回し、内部出力キューの容量を先に拡張させる。
                const int queuePrimeBlocks = juce::jlimit(8, 512, warmupBlocks / 2);
                for (int i = 0; i < queuePrimeBlocks; ++i)
                {
                    engine.Add(nullptr, warmupSamples, 1);
                    if (engine.Avail(warmupSamples) > 0)
                        (void)engine.Get();
                }

                // 末尾の残留分を完全にドレインして初期状態へ戻す
                const int maxDrainIterations = warmupBlocks + queuePrimeBlocks + 64;
                for (int i = 0; i < maxDrainIterations; ++i)
                {
                    const int avail = engine.Avail(warmupSamples);
                    if (avail <= 0)
                        break;
                    (void)engine.Get();
                    engine.Advance(avail);
                }

                // 末尾の残留分を軽くフラッシュ
                for (int i = 0; i < 8; ++i)
                {
                    const int avail = engine.Avail(warmupSamples);
                    if (avail <= 0)
                        break;
                    (void)engine.Get();
                    engine.Advance(avail);
                }
            };

            prewarmConvolver(convolvers[0], callQuantumSamples);
            prewarmConvolver(convolvers[1], callQuantumSamples);
            convolvers[0].Reset();
            convolvers[1].Reset();
        }

        // Deep Copyを作成する
        std::shared_ptr<StereoConvolver> clone() const
        {
            auto newConv = std::make_shared<StereoConvolver>();
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
                    newConv->init(l.release(), r.release(), irDataLength, storedSampleRate, irLatency, storedMaxFFTSize, storedKnownBlockSize, storedFirstPartition, callQuantumSamples);
                }
            }
            return newConv;
        }

        void reset() { convolvers[0].Reset(); convolvers[1].Reset(); }
    };

    // Note: trashBin は、Audio Thread がまだ使用している可能性のある古い Convolution オブジェクトを保持するために使用されます。
    std::atomic<StereoConvolver*> convolution { nullptr }; // Raw pointer for Audio Thread (Lock-free)
    StereoConvolver::Ptr activeConvolution; // Ownership holder for Message Thread
    std::vector<std::pair<StereoConvolver::Ptr, uint32>> trashBin; // Time-based GC
    juce::CriticalSection trashBinLock;
    std::atomic<bool> isLoading { false };
    std::atomic<bool> isRebuilding { false };
    std::unique_ptr<LoaderThread> activeLoader;
    std::vector<std::unique_ptr<LoaderThread>> loaderTrashBin;
    std::atomic<float> loadProgress { 0.0f };
    juce::String lastError;
    void setLoadingProgress(float p) { loadProgress.store(p); }

    juce::ListenerList<Listener> listeners;

    juce::dsp::ProcessSpec currentSpec = { 48000.0, 512, 2 };

    //----------------------------------------------------------
    // レイテンシー補正用ディレイ
    //----------------------------------------------------------
    // juce::dsp::DelayLine<double> delayLine; // Replaced with custom AVX2 ring buffer
    convo::ScopedAlignedPtr<double> delayBuffer[2]; // L/R separate buffers
    int delayBufferCapacity = 0;
    int delayWritePos = 0;
    juce::SmoothedValue<double> latencySmoother;

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
    int irLength = 0;
    std::vector<float> irWaveform;
    std::vector<float> irMagnitudeSpectrum;
    double irSpectrumSampleRate = 0.0;
    juce::File currentIrFile;
    juce::CriticalSection irFileLock;
    std::atomic<bool> currentIrOptimized { false };
    juce::AudioBuffer<double> originalIR; // 元IR保持 (リサンプリング/トリミング用)
    double originalIRSampleRate = 0.0;
    // MKL/AVX-512用に64byteアライメントを保証するアロケータを使用
    convo::ScopedAlignedPtr<float> cachedFFTBuffer; // FFT計算用キャッシュ (Message Thread)
    int cachedFFTBufferCapacity = 0;
    std::atomic<double> currentSampleRate { 0.0 };

#if JUCE_DSP_USE_INTEL_MKL
    DFTI_DESCRIPTOR_HANDLE fftHandle = nullptr;
    int fftHandleSize = 0;
#endif

    //----------------------------------------------------------
    // Dry信号バッファ（Mix用）
    //----------------------------------------------------------
    juce::AudioBuffer<double> dryBuffer;
    convo::ScopedAlignedPtr<double> dryBufferStorage[2]; // Aligned storage for dryBuffer
    int dryBufferCapacity = 0;
    juce::AudioBuffer<double> smoothingBuffer; // スムーシングゲイン計算用 (Audio Threadでのメモリ確保回避)
    convo::ScopedAlignedPtr<double> smoothingBufferStorage[2]; // Aligned storage for smoothingBuffer
    int smoothingBufferCapacity = 0;

    //----------------------------------------------------------
    // 準備完了フラグ
    //----------------------------------------------------------
    std::atomic<bool> isPrepared { false };
    bool visualizationEnabled = true; // Default true (for UI instance)
    int currentBufferSize = 0; // prepareToPlayで更新される
    double currentSmoothingTimeSec = SMOOTHING_TIME_DEFAULT_SEC; // mixSmootherに設定されている現在の時間

    void createWaveformSnapshot (const juce::AudioBuffer<double>& irBuffer);
    void createFrequencyResponseSnapshot (const juce::AudioBuffer<double>& irBuffer, double sampleRate);
    int computeTargetIRLength(double sampleRate, int originalLength) const;
    void applyNewState(StereoConvolver::Ptr newConv,
                      const juce::AudioBuffer<double>& loadedIR,
                      double loadedSR,
                      int targetLength,
                      bool isRebuild,
                      const juce::File& file,
                      const juce::AudioBuffer<double>& displayIR);
    void handleLoadError(const juce::String& error);

    JUCE_DECLARE_WEAK_REFERENCEABLE(ConvolverProcessor)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverProcessor)
};
