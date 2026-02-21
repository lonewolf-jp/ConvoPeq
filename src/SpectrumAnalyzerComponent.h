//============================================================================
#pragma once
// SpectrumAnalyzerComponent.h ── v0.2 (JUCE 8.0.12対応)
// スペクトラムアナライザー＋EQ応答曲線＋レベルメーター
//
// ■ 描画パイプライン設計:
//   - Timer で定期的に (~30fps) FFTデータを取得し、表示を更新する
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

    void paint(juce::Graphics& g) override;
    void resized() override;

    // 外部からタイマーを制御するためのラッパーメソッド
    void startAnalysis(int intervalMs) { startTimer(intervalMs); }
    void stopAnalysis()                { stopTimer(); }

private:
    AudioEngine& engine;

    // ── FFT設定 ──
    static constexpr int NUM_FFT_POINTS  = 4096;
    static constexpr int NUM_FFT_BINS    = NUM_FFT_POINTS / 2 + 1;
    static constexpr int OVERLAP_SAMPLES = NUM_FFT_POINTS / 4;
    static constexpr float FFT_MAGNITUDE_SCALE = 4.0f / NUM_FFT_POINTS;
    static constexpr float FFT_DISPLAY_MIN_DB = -100.0f;
    static constexpr float FFT_DISPLAY_MIN_MAG = 1e-9f;

    juce::dsp::FFT fft { static_cast<int>(std::log2(NUM_FFT_POINTS)) };
    juce::dsp::WindowingFunction<float> window { NUM_FFT_POINTS, juce::dsp::WindowingFunction<float>::hann };
    std::vector<float> fftTimeDomainBuffer;
    std::vector<float> fftWorkBuffer;

    // ── 表示用データバッファ ──
    std::vector<float> rawBuffer;        // readFromFifo で取得したデータから計算
    std::vector<float> smoothedBuffer;   // 指数移動平均で更新
    std::vector<float> peakBuffer;       // ピーク保持バッファ
    std::vector<int>   peakHoldCounter;  // 各バンドのピーク保持残フレーム数

    // ── EQ応答曲線データ ──
    std::vector<float> eqResponseBufferL; // 要素数: NUM_DISPLAY_BARS
    std::vector<float> eqResponseBufferR; // 要素数: NUM_DISPLAY_BARS

    // 個別バンドの応答曲線データ (timerで計算、paintで描画)
    std::vector<std::vector<float>> individualBandCurvesL;
    std::vector<std::vector<float>> individualBandCurvesR;
    std::vector<float> displayFrequencies; // 表示バーに対応する周波数

    // ── 計算用キャッシュ ──
    std::vector<std::complex<double>> zCache; // 周波数応答計算用の複素数キャッシュ (z = e^jw)
    double cachedSampleRate = 0.0;            // zCache計算時のサンプルレート

    // ── 描画用パスキャッシュ ──
    juce::Path totalCurvePathL, totalCurvePathR;
    std::vector<juce::Path> individualCurvePathsL;
    std::vector<juce::Path> individualCurvePathsR;

    // ── スムーシング係数 ──
    static constexpr float SMOOTHING_ALPHA = 0.75f;

    // ── 表示範囲 ──
    static constexpr float MIN_FREQ_HZ = 20.0f;
    static constexpr float MAX_FREQ_HZ = 20000.0f;
    static constexpr float MIN_DB       = -80.0f;
    static constexpr float MAX_DB       = 6.0f;

    // ── 表示バンド数 ──
    static constexpr int NUM_DISPLAY_BARS = 128;

    // ── ピーク保持フレーム数 ──
    // 30fps で約2秒間保持
    static constexpr int PEAK_HOLD_FRAMES = 60;

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

    // ── Timer コールバック (~30fps) ──
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
    void drawLevelMeterBar(juce::Graphics& g, const juce::Rectangle<int>& barRect, float db, const juce::String& title);

    // ── サブ描画メソッド ──
    void paintSpectrum   (juce::Graphics& g, const juce::Rectangle<int>& area);
    void paintEQCurve    (juce::Graphics& g, const juce::Rectangle<int>& area);
    void paintLevelMeter (juce::Graphics& g, const juce::Rectangle<int>& area);
    void paintGrid       (juce::Graphics& g, const juce::Rectangle<int>& area);

    // ── パス生成ヘルパー ──
    void updateEQData();
    void updateEQPaths();

    juce::TextButton sourceButton;
    void updateSourceButtonText();

    // ── アンダーラン対策 ──
    int underflowCount = 0;
    static constexpr float UNDERRUN_DECAY_DB = 3.0f; // データ不足時の減衰量 (dB/frame)
};
