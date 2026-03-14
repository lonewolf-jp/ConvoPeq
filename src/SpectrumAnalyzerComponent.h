//============================================================================
#pragma once
// SpectrumAnalyzerComponent.h ── v0.2 (JUCE 8.0.12対応)
// スペクトラムアナライザー＋EQ応答曲線＋レベルメーター
//
// ■ 描画パイプライン設計:
//   - Timer で定期的に (~60fps) FFTデータを取得し、表示を更新する
//   - スムーシング（指数移動平均）で急激な変化を緩和
//   - 対数スケールの周波数軸で、人間の聴覚特性に合わせた表示
//   - ピーク保持: 最大値を記録し、PEAK_HOLD_FRAMES フレーム間保持
//   - EQ応答曲線: 128点で計算し、スペクトラム上に白の折れ線で描画
//   - レベルメーター: 画面右側に入出力レベルの縦バーを配置
//
// ■ スレッド安全性:
//   - timerCallback(), paint() は UI Thread のみ
//   - engine.readFromFifo() は UI Thread から呼んで OK
//   - engine.getInputLevel(), getOutputLevel(), calcEQResponseCurve()
//     も UI Thread から呼んで OK
//
// ■ 安定性ポイント:
//   - デストラクタで必ず stopTimer() を呼ぶ
//   - 全バッファサイズは固定（動的リサイズなし）
//   - paint() で out-of-bounds アクセスを防止
//============================================================================

#include <JuceHeader.h>
#include "AudioEngine.h"

class SpectrumAnalyzerComponent : public juce::Component,
                                  private juce::Timer,
                                  private juce::ChangeListener,
                                  private EQProcessor::Listener
{
public:
    explicit SpectrumAnalyzerComponent(AudioEngine& audioEngine);
    ~SpectrumAnalyzerComponent() override;

    void setAnalyzerEnabled(bool enabled);

    void paint(juce::Graphics& g) override;
    void resized() override;

    // 外部からタイマーを制御するためのラッパーメソッド
    void startAnalysis(int intervalMs) { startTimer(intervalMs); }
    void stopAnalysis()                { stopTimer(); }

private:
    AudioEngine& engine;

    // ── 定数定義 (バッファサイズ決定のために先頭に配置) ──
    static constexpr int NUM_DISPLAY_BARS = 128;
    static constexpr int NUM_FFT_POINTS  = 4096;
    static constexpr int NUM_FFT_BINS    = NUM_FFT_POINTS / 2 + 1;
    static constexpr int OVERLAP_SAMPLES = NUM_FFT_POINTS / 4;

    // ── 表示範囲 ──
    static constexpr float MIN_DB       = -80.0f;

    // ── FFT設定 ──
    static constexpr float FFT_MAGNITUDE_SCALE = 4.0f / NUM_FFT_POINTS;
    static constexpr float FFT_DISPLAY_MIN_DB = -100.0f;
    static constexpr float FFT_DISPLAY_MIN_MAG = 1e-9f;

    DFTI_DESCRIPTOR_HANDLE fftHandle = nullptr;
    juce::dsp::WindowingFunction<float> window { NUM_FFT_POINTS, juce::dsp::WindowingFunction<float>::hann };
    // MKL/AVX-512用に64byteアライメントを保証するアロケータを使用
    convo::ScopedAlignedPtr<float> fftTimeDomainBuffer;
    convo::ScopedAlignedPtr<float> fftWorkBuffer;

    // ── 表示用データバッファ ──
    std::array<float, NUM_FFT_BINS> rawBuffer;        // readFromFifo で取得したデータから計算
    std::array<float, NUM_FFT_BINS> smoothedBuffer;   // 指数移動平均で更新
    std::array<float, NUM_FFT_BINS> peakBuffer;       // ピーク保持バッファ
    std::array<double, NUM_FFT_BINS> peakHoldTime;    // 各バンドのピーク保持残り時間 (秒)
    std::array<float, NUM_DISPLAY_BARS + 1> barXCoords; // 各バーのX座標をキャッシュ

    // ── EQ応答曲線データ ──
    std::array<float, NUM_DISPLAY_BARS> eqResponseBufferL;
    std::array<float, NUM_DISPLAY_BARS> eqResponseBufferR;

    // 個別バンドの応答曲線データ (timerで計算、paintで描画)

    std::array<std::array<float, NUM_DISPLAY_BARS>, EQProcessor::NUM_BANDS> individualBandCurvesL;
    std::array<std::array<float, NUM_DISPLAY_BARS>, EQProcessor::NUM_BANDS> individualBandCurvesR;
    // 表示バーの中心周波数と、EQカーブ計算用の周波数ポイントを兼ねる
    std::array<float, NUM_DISPLAY_BARS> displayFrequencies;

    // ── 計算用キャッシュ ──
    std::array<std::complex<double>, NUM_DISPLAY_BARS> zCache; // 周波数応答計算用の複素数キャッシュ (z = e^jw)
    double cachedSampleRate = 0.0;            // zCache計算時のサンプルレート

    // ── 描画用パスキャッシュ ──
    juce::Path totalCurvePathL, totalCurvePathR;
    std::vector<juce::Path> individualCurvePathsL;
    std::vector<juce::Path> individualCurvePathsR;

    // ── スムーシング係数 ──
    static constexpr float SMOOTHING_ALPHA = 0.85f; // 60fpsに合わせて調整 (0.75 -> 0.85)

    static constexpr float MIN_FREQ_HZ = 20.0f;
    static constexpr float MAX_FREQ_HZ = 20000.0f;
    static constexpr float MAX_DB       = 20.0f;

    // ── ピーク保持設定 ──
    static constexpr double PEAK_HOLD_SEC = 1.0;
    static constexpr float PEAK_DECAY_DB_PER_SEC = 15.0f;

    // ── レベルメーターのピークホールド設定 ──
    static constexpr double LEVEL_PEAK_HOLD_SEC          = 3.0;   // ピーク保持時間 (秒)
    static constexpr float  LEVEL_PEAK_DECAY_DB_PER_SEC  = 20.0f; // 保持後の減衰速度 (dB/秒)

    // ── レベルメーターピーク値 (UI Thread のみアクセス) ──
    float  inputPeakDb         = METER_MIN_DB;
    float  outputPeakDb        = METER_MIN_DB;
    double inputPeakHoldTimer  = 0.0; // 残り保持時間 (秒)
    double outputPeakHoldTimer = 0.0;

    void resetLevelPeaks() noexcept;
    bool updateLevelPeaks(double dt) noexcept;
    void updateTimerRate();

    // ── レベルメーターの幅 ──
    static constexpr int LEVEL_METER_WIDTH = 24;  // 各バーの幅(px)

    // ── レベルメーターの表示範囲 ──
    static constexpr float METER_MIN_DB = -60.0f;
    static constexpr float METER_MAX_DB = 6.0f;

    // ── レイアウトキャッシュ ──
    juce::Rectangle<int> plotArea;

    float logMinFreq = 0.0f;
    float logMaxFreq = 0.0f;

    // ── 座標変換定数 ──
    // 低域を圧縮しすぎないための二次関数マッピング係数
    static constexpr float MAP_COEFF_A = 49.0f;
    static constexpr float MAP_COEFF_B = 2.0f;
    static constexpr float MAP_COEFF_C = 51.0f;
    static constexpr float MAP_COEFF_D = 2499.0f; // sqrt内部係数

    // ── Timer コールバック (~60fps) ──
    void changeListenerCallback (juce::ChangeBroadcaster* source) override;
    void eqBandChanged(EQProcessor* processor, int bandIndex) override;
    void eqGlobalChanged(EQProcessor* processor) override;
    void timerCallback() override;

    // ── 描画ヘルパー ──
    static float mapLogFreqToX(float t);
    static float mapXToLogFreq(float x);
    float freqToX(float freq, float plotWidth) const;
    float dbToY  (float db,   float plotHeight) const;
    juce::Colour getLevelColour(float normalizedLevel) const;
    void drawLevelMeterBar(juce::Graphics& g, const juce::Rectangle<int>& barRect, float db, float peakDb, const juce::String& title);

    // ── サブ描画メソッド ──
    void paintSpectrum   (juce::Graphics& g, const juce::Rectangle<int>& area);
    void paintEQCurve    (juce::Graphics& g, const juce::Rectangle<int>& area);
    void paintLevelMeter (juce::Graphics& g, const juce::Rectangle<int>& area);
    void paintGrid       (juce::Graphics& g, const juce::Rectangle<int>& area);

    void prepareFFT();
    void releaseFFT();

    // ── パス生成ヘルパー ──
    void updateEQData();
    void updateEQPaths();

    juce::TextButton sourceButton;
    void updateSourceButtonText();
    juce::ToggleButton analyzerEnableButton;

    // ── アンダーラン対策 ──
    int underflowCount = 0;
    static constexpr float UNDERRUN_DECAY_DB = 1.5f; // データ不足時の減衰量 (dB/frame) @ 60fps -> 90dB/s

    bool eqPathsDirty = true;
    bool eqDataDirty = false;
    bool analyzerVisualsCleared = false;

    int currentTimerHz = 0;
    double lastEqUpdateTime = 0.0;

    static constexpr int TIMER_HZ_ACTIVE = 60;
    static constexpr int TIMER_HZ_IDLE_VISIBLE = 15;
    static constexpr int TIMER_HZ_HIDDEN = 5;
    static constexpr double EQ_UPDATE_INTERVAL_SEC = 0.10;

    double lastTime = 0.0;
};
