//============================================================================
#pragma once
// ConvolverProcessor.h  ── v0.1 (JUCE 8.0.12対応)
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
//   - process(): Audio Thread で実行。FFTConvolver を使用してパーティション分割畳み込みを行います。レイテンシーはブロックサイズに依存します。
//   - パラメータ: std::atomic でスレッドセーフ。Audio Thread内でのメモリ確保やIR再ロードは行いません。
//============================================================================

#include <JuceHeader.h>
#include <atomic>
#include <memory>
#include <vector>
#include <array>

#include "FFTConvolver.h"

class ConvolverProcessor : public juce::ChangeBroadcaster
{
public:
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
    // static constexpr int MAX_IR_LATENCY = 524288; // IRの最大長 (computeTargetIRLengthのkMaxIRCapと一致)
    static constexpr int MAX_IR_LATENCY = 1048576;
    static constexpr int MAX_BLOCK_SIZE = 8192;   // AudioEngineのSAFE_MAX_BLOCK_SIZEと一致
    static constexpr int MAX_TOTAL_DELAY = MAX_IR_LATENCY + MAX_BLOCK_SIZE;

    ConvolverProcessor();
    ~ConvolverProcessor();

    //----------------------------------------------------------
    // 準備（Audio Thread開始前）
    //----------------------------------------------------------
    void prepareToPlay(double sampleRate, int samplesPerBlock);

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
    // buffer: インプレース処理（入力と出力が同じバッファ）
    //----------------------------------------------------------
    void process(juce::AudioBuffer<double>& buffer, int numSamples);

    //----------------------------------------------------------
    // バイパス制御
    //----------------------------------------------------------
    void setBypass(bool shouldBypass) { bypassed.store(shouldBypass); }
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

    // 他のインスタンスから状態を同期 (AudioEngine用)
    void syncStateFrom(const ConvolverProcessor& other);

private:
    class LoaderThread;

    //----------------------------------------------------------
    // FFTConvolver Engine
    //----------------------------------------------------------
    // ステレオ処理用ラッパー (FFTConvolverはモノラルのため)
    struct StereoConvolver
    {
        std::array<fftconvolver::FFTConvolver, 2> convolvers;
        int latency = 0;
        int irLatency = 0; // IR由来の遅延 (ピーク位置)

        void init(size_t blockSize, const fftconvolver::Sample* irL, const fftconvolver::Sample* irR, size_t irLen, int peakDelay)
        {
            convolvers[0].init(blockSize, irL, irLen);
            convolvers[1].init(blockSize, irR, irLen);
            latency = static_cast<int>(blockSize);
            irLatency = peakDelay;
        }

        void reset() { convolvers[0].reset(); convolvers[1].reset(); }
    };

    // Note: trashBin is used to keep old Convolution objects alive while Audio Thread might still be using them.
    std::atomic<std::shared_ptr<StereoConvolver>> convolution;
    std::vector<std::shared_ptr<StereoConvolver>> trashBin;
    juce::CriticalSection trashBinLock;
    std::atomic<bool> isLoading { false };
    std::atomic<bool> isRebuilding { false }; // ← 追加
    std::unique_ptr<LoaderThread> activeLoader;

    juce::dsp::ProcessSpec currentSpec = { 48000.0, 512, 2 };

    //----------------------------------------------------------
    // レイテンシー補正用ディレイ
    //----------------------------------------------------------
    juce::dsp::DelayLine<double> delayLine;
    std::atomic<int> currentLatency {0};

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
    std::vector<float> cachedFFTBuffer; // FFT計算用キャッシュ (Message Thread)
    std::atomic<double> currentSampleRate { 48000.0 };

    //----------------------------------------------------------
    // Dry信号バッファ（Mix用）
    //----------------------------------------------------------
    juce::AudioBuffer<double> dryBuffer;
    juce::AudioBuffer<double> convolutionBuffer; // Convolution用 (double)

    //----------------------------------------------------------
    // 準備完了フラグ
    //----------------------------------------------------------
    std::atomic<bool> isPrepared { false };
    int currentBufferSize = 512; // prepareToPlayで更新される
    double currentSmoothingTimeSec = SMOOTHING_TIME_DEFAULT_SEC; // mixSmootherに設定されている現在の時間

    void createWaveformSnapshot (const juce::AudioBuffer<double>& irBuffer);
    void createFrequencyResponseSnapshot (const juce::AudioBuffer<double>& irBuffer, double sampleRate);
    int computeTargetIRLength(double sampleRate, int originalLength) const;
    void applyNewState(std::shared_ptr<StereoConvolver> newConv,
                      const juce::AudioBuffer<double>& loadedIR,
                      double loadedSR,
                      int targetLength,
                      bool isRebuild,
                      const juce::File& file,
                      const juce::AudioBuffer<double>& displayIR);

    JUCE_DECLARE_WEAK_REFERENCEABLE(ConvolverProcessor)
    JUCE_DECLARE_NON_COPYABLE_WITH_LEAK_DETECTOR(ConvolverProcessor)
};
