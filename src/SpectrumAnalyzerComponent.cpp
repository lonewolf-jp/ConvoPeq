//============================================================================
// SpectrumAnalyzerComponent.cpp ── v0.2 (JUCE 8.0.12対応)
//
// スペクトラムアナライザー＋EQ応答曲線＋レベルメーター・ピーク保持
//============================================================================
#include "SpectrumAnalyzerComponent.h"
#include <cmath>
#include <algorithm>
#include <complex>

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
SpectrumAnalyzerComponent::SpectrumAnalyzerComponent(AudioEngine& audioEngine)
    : engine(audioEngine)
{
    fftTimeDomainBuffer.resize (NUM_FFT_POINTS, 0.0f);
    fftWorkBuffer.resize (NUM_FFT_POINTS * 2, 0.0f);
    rawBuffer.assign       (NUM_FFT_BINS, MIN_DB);
    smoothedBuffer.assign  (NUM_FFT_BINS, MIN_DB);
    peakBuffer.assign      (NUM_FFT_BINS, MIN_DB);
    peakHoldCounter.assign (NUM_FFT_BINS, 0);
    eqResponseBufferL.assign(NUM_DISPLAY_BARS, 0.0f);
    eqResponseBufferR.assign(NUM_DISPLAY_BARS, 0.0f);

    individualBandCurvesL.assign(EQProcessor::NUM_BANDS, std::vector<float>(NUM_DISPLAY_BARS, MIN_DB));
    individualBandCurvesR.assign(EQProcessor::NUM_BANDS, std::vector<float>(NUM_DISPLAY_BARS, MIN_DB));
    displayFrequencies.resize(NUM_DISPLAY_BARS);
    zCache.resize(NUM_DISPLAY_BARS);

    individualCurvePathsL.resize(EQProcessor::NUM_BANDS);
    individualCurvePathsR.resize(EQProcessor::NUM_BANDS);

    // 表示用の周波数ポイントを事前に計算
    logMinFreq = std::log10(MIN_FREQ_HZ);
    logMaxFreq = std::log10(MAX_FREQ_HZ);
    for (int i = 0; i < NUM_DISPLAY_BARS; ++i)
    {
        const float xNorm = static_cast<float>(i) / static_cast<float>(NUM_DISPLAY_BARS - 1);
        const float logT  = mapXToLogFreq(xNorm);
        displayFrequencies[i] = std::pow(10.0f, logMinFreq + logT * (logMaxFreq - logMinFreq));
    }

    // Source Select Button
    sourceButton.setColour(juce::TextButton::buttonColourId, juce::Colours::darkgrey.withAlpha(0.6f));
    sourceButton.setColour(juce::TextButton::textColourOffId, juce::Colours::white.withAlpha(0.9f));
    sourceButton.onClick = [this] {
        auto newSource = (engine.getAnalyzerSource() == AudioEngine::AnalyzerSource::Input)
                         ? AudioEngine::AnalyzerSource::Output
                         : AudioEngine::AnalyzerSource::Input;
        engine.setAnalyzerSource(newSource);
        updateSourceButtonText();
    };
    updateSourceButtonText();
    addAndMakeVisible(sourceButton);

    engine.addChangeListener(this);
    engine.getEQProcessor().addChangeListener(this);
    engine.getEQProcessor().addListener(this);

    updateEQData(); // 初期状態のEQカーブを計算

    startTimerHz(60); // 60fps: UIの滑らかさとFIFO消費の安定化のため
}

//--------------------------------------------------------------
// デストラクタ
//--------------------------------------------------------------
SpectrumAnalyzerComponent::~SpectrumAnalyzerComponent()
{
    stopTimer();
    engine.removeChangeListener(this);
    engine.getEQProcessor().removeChangeListener(this);
    engine.getEQProcessor().removeListener(this);
}

//--------------------------------------------------------------
// timerCallback  ──  ~30fps で呼ばれる (UI Thread)
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::timerCallback()
{
    if (!isShowing()) return;

    // ── サンプルレート変更検知 (タイマー駆動) ──
    // デバイス変更などでサンプルレートが変わった場合に追従する
    const double currentSampleRate = engine.getProcessingSampleRate();
    if (currentSampleRate > 0.0 && std::abs(currentSampleRate - cachedSampleRate) > 1.0)
    {
        updateEQData();
    }

    // ── FFTデータの取得とスムーシング ──
    // FIFOの利用可能データ数をチェック
    const int available = engine.getFifoNumReady();
    // 十分なデータがあるか確認 (FFTサイズ分のデータが揃うまで待つことでアンダーラングリッチを防ぐ)
    const int required = OVERLAP_SAMPLES;

    // データ不足 → スキップ (データ不足の場合は何もしない)
    if (available < required)
    {
        underflowCount++;

        // 30フレーム連続でアンダーランした場合 (60fpsで0.5秒)、
        // エンジンが停止したとみなし、CPU消費を防ぐためにタイマーを停止する
        if (underflowCount > 30)
        {
            stopTimer();
            return;
        }

        // アンダーラン時は「減衰保持」 (Decay Hold)
        // 視覚的に自然にフェードアウトさせる
        for (size_t i = 0; i < smoothedBuffer.size(); ++i)
        {
            smoothedBuffer[i] -= UNDERRUN_DECAY_DB;
            if (smoothedBuffer[i] < MIN_DB) smoothedBuffer[i] = MIN_DB;

            // ピーク保持の更新 (減衰時もピークロジックを継続)
            if (peakHoldCounter[i] > 0)
            {
                --peakHoldCounter[i];
            }
            else
            {
                peakBuffer[i] = std::max(smoothedBuffer[i], peakBuffer[i] - 0.5f);
            }
            if (peakBuffer[i] < MIN_DB) peakBuffer[i] = MIN_DB;
        }

        repaint();
        return;
    }

    underflowCount = 0;

    // 1. 既存データを左にシフト (古いデータを破棄)
    std::memmove(fftTimeDomainBuffer.data(),
                 fftTimeDomainBuffer.data() + OVERLAP_SAMPLES,
                 (NUM_FFT_POINTS - OVERLAP_SAMPLES) * sizeof(float));

    // 2. FIFOから新しいデータを読み込み
    engine.readFromFifo(fftTimeDomainBuffer.data() + (NUM_FFT_POINTS - OVERLAP_SAMPLES), OVERLAP_SAMPLES);

    // 安全対策: NaN/Infチェック
    // 万が一DSP処理で不正な値が発生しても、アナライザーでクラッシュさせない
    for (auto& sample : fftTimeDomainBuffer)
    {
        if (!std::isfinite(sample))
            sample = 0.0f;
    }

    // 3. 窓関数適用とFFT実行
    std::memcpy(fftWorkBuffer.data(), fftTimeDomainBuffer.data(), NUM_FFT_POINTS * sizeof(float));
    std::fill(fftWorkBuffer.data() + NUM_FFT_POINTS, fftWorkBuffer.data() + NUM_FFT_POINTS * 2, 0.0f);
    window.multiplyWithWindowingTable(fftWorkBuffer.data(), NUM_FFT_POINTS);
    // Note: この処理は Message Thread (Timer) で実行されるため、FFT実行（performFrequencyOnlyForwardTransform）に伴う負荷や一時的なメモリ確保はAudio Threadに影響せず安全です。
    fft.performFrequencyOnlyForwardTransform(fftWorkBuffer.data());

    // 4. dB変換して出力
    const int numBins = std::min(static_cast<int>(rawBuffer.size()), NUM_FFT_BINS);
    for (int i = 0; i < numBins; ++i)
    {
        const float magnitude = fftWorkBuffer[i] * FFT_MAGNITUDE_SCALE;
        rawBuffer[i] = (magnitude > FFT_DISPLAY_MIN_MAG)
                        ? juce::Decibels::gainToDecibels(magnitude)
                        : FFT_DISPLAY_MIN_DB;
    }

    for (size_t i = 0; i < smoothedBuffer.size(); ++i)
    {
        smoothedBuffer[i] = SMOOTHING_ALPHA * smoothedBuffer[i]
                            + (1.0f - SMOOTHING_ALPHA) * rawBuffer[i];

        // ── ピーク保持の更新 ──
        if (smoothedBuffer[i] >= peakBuffer[i])
        {
            // 現在値がピーク以上なら更新し、保持カウンタをリセット
            peakBuffer[i]      = smoothedBuffer[i];
            peakHoldCounter[i] = PEAK_HOLD_FRAMES;
        }
        else
        {
            // カウンタを減らし、0になったらピークを現在値に落とす
            if (peakHoldCounter[i] > 0)
            {
                --peakHoldCounter[i];
            }
            else
            {
                // ピークも少しずつ落とす（急な落ち込みを緩和）
                peakBuffer[i] = std::max(smoothedBuffer[i],
                                            peakBuffer[i] - 0.5f); // 0.5dB/frame
            }
        }
    }

    repaint();
}

void SpectrumAnalyzerComponent::changeListenerCallback (juce::ChangeBroadcaster* source)
{
    // AudioEngine (EQProcessor, ConvolverProcessor) からの変更通知
    if (source == &engine || source == &engine.getEQProcessor())
    {
        // エンジン状態が変更された場合、アンダーランで停止していたタイマーを再開する
        if (!isTimerRunning())
        {
            startTimerHz(60);
        }
        updateEQData();

        // ソース選択ボタンの表示更新 (プリセットロード時など)
        if (source == &engine)
            updateSourceButtonText();
    }
}

void SpectrumAnalyzerComponent::eqBandChanged(EQProcessor* processor, int /*bandIndex*/)
{
    if (processor == &engine.getEQProcessor())
    {
        // エンジン状態が変更された場合、アンダーランで停止していたタイマーを再開する
        if (!isTimerRunning())
        {
            startTimerHz(60);
        }
        updateEQData();
    }
}

void SpectrumAnalyzerComponent::eqGlobalChanged(EQProcessor* processor)
{
    if (processor == &engine.getEQProcessor())
    {
        // エンジン状態が変更された場合、アンダーランで停止していたタイマーを再開する
        if (!isTimerRunning())
        {
            startTimerHz(60);
        }
        updateEQData();
    }
}

//--------------------------------------------------------------
// paint  ──  メインの描画ルーティン
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::paint(juce::Graphics& g)
{
    // Note: Direct2D (Windows) / CoreGraphics (macOS) によりハードウェアアクセラレーションが効きます。
    //----------------------------------------------------------
    // レイアウト分割:
    //   [左側: スペクトラム + EQ曲線 + グリッド]
    //   [右側: レベルメーター (入/出 2バー)]
    //----------------------------------------------------------
    const int meterTotalWidth = LEVEL_METER_WIDTH * 2 + 16; // 2バー + マージン

    auto bounds = getLocalBounds();
    auto meterArea   = bounds.removeFromRight(meterTotalWidth);
    auto specArea    = bounds;

    //----------------------------------------------------------
    // 背景とプロットエリア
    //----------------------------------------------------------
    g.setColour(juce::Colours::black.withAlpha(0.9f));
    g.fillRoundedRectangle(getLocalBounds().toFloat(), 5.0f);

    if (plotArea.getWidth() <= 0 || plotArea.getHeight() <= 0) return;

    // ── グリッド描画 ──
    paintGrid(g, plotArea);

    // ── スペクトラム棒グラフ + ピーク保持描画 ──
    paintSpectrum(g, plotArea);

    // ── EQ応答曲線描画 ──
    paintEQCurve(g, plotArea);

    // ── プロットエリアの枠線 ──
    g.setColour(juce::Colours::white.withAlpha(0.35f));
    g.drawRect(plotArea.toFloat(), 1.0f);

    // ── タイトル ──
    g.setColour(juce::Colours::white.withAlpha(0.8f));
    g.setFont(juce::FontOptions(13.0f, juce::Font::bold));
    g.drawText("Spectrum Analyzer", plotArea.withHeight(18),
               juce::Justification::centredLeft);

    // ── レベルメーター描画 ──
    paintLevelMeter(g, meterArea);
}

void SpectrumAnalyzerComponent::resized()
{
    auto bounds = getLocalBounds();
    const int meterTotalWidth = LEVEL_METER_WIDTH * 2 + 16;
    auto specArea = bounds.removeFromLeft(bounds.getWidth() - meterTotalWidth);

    const int marginL = 52;
    const int marginR = 10;
    const int marginT = 10;
    const int marginB = 28;

    plotArea = specArea.withTrimmedLeft(marginL)
                       .withTrimmedRight(marginR)
                       .withTrimmedTop(marginT)
                       .withTrimmedBottom(marginB);

    // ボタン配置 (右上の余白)
    sourceButton.setBounds(specArea.getRight() - 110, marginT, 100, 18);

    updateEQPaths();
}

//--------------------------------------------------------------
// paintGrid  ──  グリッド線と軸ラベル
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::paintGrid(juce::Graphics& g, const juce::Rectangle<int>& area)
{
    const float plotX = static_cast<float>(area.getX());
    const float plotY = static_cast<float>(area.getY());
    const float plotW = static_cast<float>(area.getWidth());
    const float plotH = static_cast<float>(area.getHeight());
    const int   marginL = area.getX();

    // ── dB グリッド（水平線） ──
    g.setFont(juce::FontOptions(10.0f));
    for (float db = MIN_DB; db <= MAX_DB; db += 20.0f)
    {
        const float y = plotY + dbToY(db, plotH);

        g.setColour(juce::Colours::grey.withAlpha(0.25f));
        g.drawHorizontalLine(static_cast<int>(y), plotX, plotX + plotW);

        g.setColour(juce::Colours::white.withAlpha(0.6f));
        juce::String label = juce::String(static_cast<int>(db)) + "dB";
        g.drawText(label, 2, static_cast<int>(y) - 6, marginL - 6, 14,
                   juce::Justification::centredLeft);
    }

    // ── 0dB 基準線（明るい色で強調） ──
    {
        const float y0 = plotY + dbToY(0.0f, plotH);
        g.setColour(juce::Colours::grey.withAlpha(0.45f));
        g.drawHorizontalLine(static_cast<int>(y0), plotX, plotX + plotW);
    }

    // ── 周波数グリッド（垂直線） ──
    // 対数スケールに合わせて 1, 2, 5 の系列でグリッドを描画
    for (float base = 10.0f; base < MAX_FREQ_HZ; base *= 10.0f)
    {
        for (int i = 1; i < 10; ++i)
        {
            float f = base * static_cast<float>(i);
            if (f < MIN_FREQ_HZ) continue;
            if (f > MAX_FREQ_HZ) break;

            const float x = plotX + freqToX(f, plotW);

            // 描画範囲内かチェック
            if (x < plotX || x > plotX + plotW) continue;

            // 1, 2, 5 はメジャーグリッド（線が濃い＋ラベルあり）
            bool isMajor = (i == 1 || i == 2 || i == 5);

            if (isMajor)
            {
                g.setColour(juce::Colours::grey.withAlpha(0.25f));
                g.drawVerticalLine(static_cast<int>(x), plotY, plotY + plotH);

                juce::String label;
                if (f >= 1000.0f)
                    label = juce::String(f / 1000.0f) + "k";
                else
                    label = juce::String(static_cast<int>(f));

                g.setColour(juce::Colours::white.withAlpha(0.6f));
                g.drawText(label,
                           static_cast<int>(x) - 14,
                           static_cast<int>(plotY + plotH) + 3,
                           28, 20,
                           juce::Justification::centred);
            }
            else
            {
                // マイナーグリッド（線が薄い）
                g.setColour(juce::Colours::grey.withAlpha(0.1f));
                g.drawVerticalLine(static_cast<int>(x), plotY, plotY + plotH);
            }
        }
    }
}

//--------------------------------------------------------------
// paintSpectrum  ──  棒グラフ + ピーク保持
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::paintSpectrum(juce::Graphics& g, const juce::Rectangle<int>& area)
{
    const float plotX = static_cast<float>(area.getX());
    const float plotY = static_cast<float>(area.getY());
    const float plotW = static_cast<float>(area.getWidth());
    const float plotH = static_cast<float>(area.getHeight());

    const double sampleRate = engine.getSampleRate();
    if (sampleRate <= 0.0) return;
    const int   halfFFT    = NUM_FFT_BINS;
    const float barWidth   = plotW / static_cast<float>(NUM_DISPLAY_BARS);

    const float binFactor = NUM_FFT_POINTS / static_cast<float>(sampleRate);
    const float nyquist = static_cast<float>(sampleRate) / 2.0f;

    for (int bar = 0; bar < NUM_DISPLAY_BARS; ++bar)
    {
        // 事前計算された周波数を使用 (pow/log計算を回避)
        // ナイキスト周波数でクランプし、FFTの有効範囲外のビンを参照しないようにする
        const float freq = std::min(displayFrequencies[bar], nyquist);

        // 周波数 → FFTビンインデックス (補間用にfloatで計算)
        float binIdx = freq * binFactor;

        int idx0 = static_cast<int>(binIdx);
        int idx1 = idx0 + 1;
        float frac = binIdx - idx0;

        idx0 = std::max(0, std::min(idx0, halfFFT - 1));
        idx1 = std::max(0, std::min(idx1, halfFFT - 1));

        // スムーシング済みのdB値 (線形補間)
        float db = smoothedBuffer[idx0] * (1.0f - frac) + smoothedBuffer[idx1] * frac;
        db = std::max(MIN_DB, std::min(MAX_DB, db));

        // 棒の高さ
        const float normalizedLevel = (db - MIN_DB) / (MAX_DB - MIN_DB);
        const float barH = std::max(0.0f, std::min(plotH, normalizedLevel * plotH));

        const float barX = plotX + static_cast<float>(bar) * barWidth;
        const float barY = plotY + plotH - barH;

        // ── 色グラデーション ──
        juce::Colour barColour = getLevelColour(normalizedLevel);

        // 棒を描画
        g.setColour(barColour);
        g.fillRect(barX + 0.5f, barY, barWidth - 1.0f, barH);

        // ── ピーク保持の描画 ──
        // ピーク値も同様に補間
        float peakDb = peakBuffer[idx0] * (1.0f - frac) + peakBuffer[idx1] * frac;
        peakDb = std::max(MIN_DB, std::min(MAX_DB, peakDb));
        const float peakNorm = (peakDb - MIN_DB) / (MAX_DB - MIN_DB);
        const float peakY    = plotY + plotH - peakNorm * plotH;

        // ピーク線: 明るい色の1px水平線
        g.setColour(barColour.brighter(0.6f).withAlpha(0.9f));
        g.fillRect(barX + 0.5f, peakY, barWidth - 1.0f, 2.0f);
    }
}

//--------------------------------------------------------------
// updateEQData  ──  EQ応答曲線とパスを再計算 (パラメータ変更時)
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::updateEQData()
{
    const double sr = engine.getProcessingSampleRate();
    if (sr > 0.0)
    {
        // サンプルレートが変更された場合のみ zCache (z = e^jw) を再計算
        if (sr != cachedSampleRate)
        {
            cachedSampleRate = sr;
            const double twoPiOverSr = 2.0 * juce::MathConstants<double>::pi / static_cast<double>(sr);
            for (int i = 0; i < NUM_DISPLAY_BARS; ++i)
            {
                double w = displayFrequencies[i] * twoPiOverSr;
                zCache[i] = std::complex<double>(std::cos(w), std::sin(w));
            }
        }

        // ── 総合EQ応答曲線の計算 ──
        // キャッシュされた zCache (複素平面上の単位円) を使用して高速化
        engine.calcEQResponseCurve(eqResponseBufferL.data(), eqResponseBufferR.data(), zCache.data(), NUM_DISPLAY_BARS, sr);

        // 線形マグニチュードをdBに変換
        for (int i = 0; i < NUM_DISPLAY_BARS; ++i)
        {
            eqResponseBufferL[i] = juce::Decibels::gainToDecibels(std::max(eqResponseBufferL[i], 1.0e-9f));
            eqResponseBufferR[i] = juce::Decibels::gainToDecibels(std::max(eqResponseBufferR[i], 1.0e-9f));
        }

        for (int b = 0; b < EQProcessor::NUM_BANDS; ++b)
        {
            const auto params = engine.getEQProcessor().getBandParams(b);
            if (!params.enabled)
            {
                std::fill(individualBandCurvesL[b].begin(), individualBandCurvesL[b].end(), 0.0f);
                std::fill(individualBandCurvesR[b].begin(), individualBandCurvesR[b].end(), 0.0f);
                continue;
            }

            EQBandType type = engine.getEQProcessor().getBandType(b);
            // グラフ描画用にBiquad係数を計算
            // 音声処理にはSVFを使用しているが、周波数応答の計算には
            // 等価な特性を持つBiquad係数を使用することで、標準的な計算式(getMagnitudeSquared)を流用している。
            EQCoeffsBiquad c = EQProcessor::calcBiquadCoeffs(type, params.frequency, params.gain, params.q, sr);
            EQChannelMode mode = engine.getEQProcessor().getBandChannelMode(b);

            for (int i = 0; i < NUM_DISPLAY_BARS; ++i)
            {
                float mag = std::sqrt(EQProcessor::getMagnitudeSquared(c, zCache[i]));
                float db = juce::Decibels::gainToDecibels(mag);
                if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Left)  individualBandCurvesL[b][i] = db; else individualBandCurvesL[b][i] = 0.0f;
                if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Right) individualBandCurvesR[b][i] = db; else individualBandCurvesR[b][i] = 0.0f;
            }
        }

    }
    else
    {
        // サンプルレートが無効な場合はクリア
        std::fill(eqResponseBufferL.begin(), eqResponseBufferL.end(), 0.0f);
        std::fill(eqResponseBufferR.begin(), eqResponseBufferR.end(), 0.0f);
    }

    updateEQPaths();
    // repaint() は timerCallback で行われるので、ここでは不要
}

//--------------------------------------------------------------
// updateEQPaths  ──  EQ曲線のパスを生成 (Timer Callback内)
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::updateEQPaths()
{
    if (plotArea.isEmpty()) return;

    const float plotX = static_cast<float>(plotArea.getX());
    const float plotY = static_cast<float>(plotArea.getY());
    const float plotW = static_cast<float>(plotArea.getWidth());
    const float plotH = static_cast<float>(plotArea.getHeight());

    auto createPath = [&](juce::Path& path, const std::vector<float>& buffer)
    {
        path.clear();
        for (int i = 0; i < NUM_DISPLAY_BARS; ++i)
        {
            const float t = static_cast<float>(i) / static_cast<float>(NUM_DISPLAY_BARS - 1);
            const float x = plotX + t * plotW;
            float db = buffer[i];
            db = std::max(MIN_DB, std::min(MAX_DB, db));
            const float y = plotY + dbToY(db, plotH);

            if (i == 0) path.startNewSubPath(x, y);
            else        path.lineTo(x, y);
        }
    };

    createPath(totalCurvePathL, eqResponseBufferL);
    createPath(totalCurvePathR, eqResponseBufferR);

    for (int b = 0; b < EQProcessor::NUM_BANDS; ++b)
    {
        createPath(individualCurvePathsL[b], individualBandCurvesL[b]);
        createPath(individualCurvePathsR[b], individualBandCurvesR[b]);
    }
}

//--------------------------------------------------------------
// paintEQCurve  ──  EQ応答曲線（白の折れ線）
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::paintEQCurve(juce::Graphics& g, const juce::Rectangle<int>& area)
{
    // デバイス未接続時や初期化中はサンプルレートが0になるため、
    // 周波数応答計算（除算）が不正になるのを防ぐ
    if (engine.getSampleRate() <= 0.0) return;

    const float plotX = static_cast<float>(area.getX());
    const float plotY = static_cast<float>(area.getY());
    const float plotW = static_cast<float>(area.getWidth());
    const float plotH = static_cast<float>(area.getHeight());

    // ── 各バンドの個別応答曲線を描画 ──
    for (int b = 0; b < EQProcessor::NUM_BANDS; ++b)
    {
        const auto params = engine.getEQProcessor().getBandParams(b);
        if (!params.enabled) continue;

        EQBandType type = engine.getEQProcessor().getBandType(b);
        if (type != EQBandType::LowPass && type != EQBandType::HighPass && std::abs(params.gain) < 0.01f) continue;

        EQChannelMode mode = engine.getEQProcessor().getBandChannelMode(b);

        if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Left)
        {
            g.setColour(juce::Colours::white.withAlpha(0.15f));
            g.strokePath(individualCurvePathsL[b], juce::PathStrokeType(1.0f));
        }

        if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Right)
        {
            g.setColour(juce::Colours::red.withAlpha(0.15f));
            g.strokePath(individualCurvePathsR[b], juce::PathStrokeType(1.0f));
        }
    }

    // ── 総合EQ応答曲線を描画 (L/R) ──
    // Right (Red)
    g.setColour(juce::Colours::red.withAlpha(0.85f));
    g.strokePath(totalCurvePathR, juce::PathStrokeType(1.5f));
    // Left (White)
    g.setColour(juce::Colours::white.withAlpha(0.85f));
    g.strokePath(totalCurvePathL, juce::PathStrokeType(1.5f));

    // ── EQ曲線のポイント（各バンドの現在の周波数位置に丸を描画） ──
    // getBandParams で現在の周波数を動的に読む（パラメータ変更後も正確に移動）
    {
        for (int b = 0; b < EQProcessor::NUM_BANDS; ++b)
        {
            const EQBandParams bp = engine.getEQProcessor().getBandParams(b);
            if (!bp.enabled) continue;

            EQChannelMode mode = engine.getEQProcessor().getBandChannelMode(b);
            const float bandFreq = bp.frequency;

            // 周波数 → X座標
            const float x = plotX + freqToX(bandFreq, plotW);

            // Note: eqResponseBuffer は displayFrequencies (対数軸等間隔) に基づいて計算されており、
            // displayFrequencies は xNorm (0.0~1.0) に対して等間隔に配置されている。
            // したがって、xNorm から直接インデックスを計算することで、曲線上の正しい位置を取得できる。
            const float xNorm = (x - plotX) / plotW;

            const int barIdx = std::max(0, std::min(NUM_DISPLAY_BARS - 1,
                                                    static_cast<int>(xNorm * static_cast<float>(NUM_DISPLAY_BARS - 1))));

            // チャンネルモードに応じて参照するバッファを変える
            float db = 0.0f;
            juce::Colour dotColor = juce::Colours::white;

            if (mode == EQChannelMode::Right) {
                db = eqResponseBufferR[barIdx];
                dotColor = juce::Colours::red;
            } else {
                db = eqResponseBufferL[barIdx]; // Stereo or Left
            }

            db = std::max(MIN_DB, std::min(MAX_DB, db));
            const float y = plotY + dbToY(db, plotH);

            // 外側の丸（白・半径4px）
            g.setColour(dotColor);
            g.fillEllipse(x - 4.0f, y - 4.0f, 8.0f, 8.0f);
            // 内側の丸（黒・半透明）
            g.setColour(juce::Colours::black.withAlpha(0.7f));
            g.fillEllipse(x - 2.0f, y - 2.0f, 4.0f, 4.0f);
        }
    }
}

//--------------------------------------------------------------
// paintLevelMeter  ──  入出力レベルメーター
//
// レイアウト:
//   [IN バー] [小マージン] [OUT バー]
//
// 各バーは縦グラデーション(緑→黄→赤)で描画し、
// 現在のレベルに応じた高さで塗り潰す。
// 数値もdBで表示する。
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::paintLevelMeter(juce::Graphics& g, const juce::Rectangle<int>& area)
{
    const int marginT = 10;
    const int marginB = 28;
    const int gap = 8;  // 2バーの間隔

    auto meterBounds = area.withTrimmedTop(marginT)
                           .withTrimmedBottom(marginB);

    if (area.getWidth() <= 0 || area.getHeight() <= 0) return;

    // 入力・出力レベルの取得
    const float inDb  = engine.getInputLevel();
    const float outDb = engine.getOutputLevel();

    // 各バーの幅
    const int barW = (meterBounds.getWidth() - gap) / 2;

    // ── IN バー ──
    auto inBarArea = meterBounds.withWidth(barW);
    // ── OUT バー ──
    auto outBarArea = meterBounds.withX(meterBounds.getX() + barW + gap).withWidth(barW);

    drawLevelMeterBar(g, inBarArea,  inDb,  "IN");
    drawLevelMeterBar(g, outBarArea, outDb, "OUT");
}

//--------------------------------------------------------------
// drawLevelMeterBar  ──  単一のレベルメーターバーを描画
//--------------------------------------------------------------
void SpectrumAnalyzerComponent::drawLevelMeterBar(juce::Graphics& g, const juce::Rectangle<int>& barRect, float db, const juce::String& title)
{
    const int labelH = 16;

    {
        // 背景(バー全体の暗い色)
        g.setColour(juce::Colours::darkgrey.withAlpha(0.6f));
        g.fillRoundedRectangle(barRect.toFloat(), 3.0f);

        // 枠線
        g.setColour(juce::Colours::grey.withAlpha(0.4f));
        g.drawRoundedRectangle(barRect.toFloat(), 3.0f, 1.0f);

        // タイトル
        g.setColour(juce::Colours::white.withAlpha(0.7f));
        g.setFont(juce::FontOptions(9.0f, juce::Font::bold));
        g.drawText(title, barRect.withHeight(labelH),
                   juce::Justification::centred);

        // レベル計算
        const float clamped = std::max(METER_MIN_DB, std::min(METER_MAX_DB, db));
        const float norm    = (clamped - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB);

        // バーの全領域（塗りつぶし最大範囲）- グラデーションの基準
        auto fullBarArea = barRect
            .withTrimmedTop(labelH)
            .withTrimmedBottom(labelH)
            .reduced(2, 0);

        // バーの描画領域（タイトル・数値表示を除いた部分）
        auto fillArea = barRect
            .withTrimmedTop(labelH)
            .withTrimmedBottom(labelH)
            .reduced(2, 0);

        if (fillArea.getHeight() <= 0) return;

        const float fillH = norm * static_cast<float>(fillArea.getHeight());
        const float fillY = static_cast<float>(fillArea.getBottom()) - fillH;

        // グラデーション塗り潰し
        // 下(緑) → 中(黄) → 上(赤) のグラデーション (メーター全体に対して固定)
        if (fillH > 0.0f)
        {
            juce::ColourGradient gradient(
                juce::Colours::green,
                static_cast<float>(fullBarArea.getX()),
                static_cast<float>(fullBarArea.getBottom()),
                juce::Colours::red,
                static_cast<float>(fullBarArea.getX()),
                static_cast<float>(fullBarArea.getY()),
                false);
            gradient.addColour(0.7, juce::Colours::yellow); // 70%付近を黄色に

            g.setGradientFill(gradient);
            g.fillRect(juce::Rectangle<float>(
                static_cast<float>(fillArea.getX()),
                fillY,
                static_cast<float>(fillArea.getWidth()),
                fillH));
        }

        // 0dB基準線（赤の水平線）
        {
            const float zeroNorm = (0.0f - METER_MIN_DB) / (METER_MAX_DB - METER_MIN_DB);
            const float zeroY    = static_cast<float>(fillArea.getBottom())
                                 - zeroNorm * static_cast<float>(fillArea.getHeight());
            g.setColour(juce::Colours::red.withAlpha(0.7f));
            g.fillRect(static_cast<float>(fillArea.getX()), zeroY,
                       static_cast<float>(fillArea.getWidth()), 1.5f);
        }

        // 数値表示（下部）
        g.setColour(juce::Colours::white.withAlpha(0.8f));
        g.setFont(juce::FontOptions(9.0f));
        juce::String dbStr = juce::String(db, 1) + "dB";
        g.drawText(dbStr, barRect.withTop(barRect.getBottom() - labelH),
                   juce::Justification::centred);
    }
}

//--------------------------------------------------------------
// freqToX  ──  周波数 → X座標 (対数スケール)
//--------------------------------------------------------------
float SpectrumAnalyzerComponent::freqToX(float freq, float plotWidth) const
{
    const float logFreq = std::log10(std::max(freq, MIN_FREQ_HZ));

    float t = (logFreq - logMinFreq) / (logMaxFreq - logMinFreq);
    t = std::max(0.0f, std::min(1.0f, t));
    return mapLogFreqToX(t) * plotWidth;
}

//--------------------------------------------------------------
// dbToY  ──  dB値 → Y座標 (線形スケール)
// db = MAX_DB → 0 (上端), db = MIN_DB → plotHeight (下端)
//--------------------------------------------------------------
float SpectrumAnalyzerComponent::dbToY(float db, float plotHeight) const
{
    return plotHeight * (1.0f - (db - MIN_DB) / (MAX_DB - MIN_DB));
}

//--------------------------------------------------------------
// 座標変換ヘルパー (static)
// 周波数軸の描画において、単純な対数スケールでは低域が密集しすぎて視認性が悪いため、
// 二次関数的なマッピング補正（Low-end expansion）を行い、低域を広げて表示します。
// これにより、20Hzから200Hzあたりの重要な帯域が見やすくなります。
//--------------------------------------------------------------
float SpectrumAnalyzerComponent::mapLogFreqToX(float t)
{
    // t: 正規化された対数周波数 (0.0 ~ 1.0)
    return (MAP_COEFF_A * t * t + MAP_COEFF_B * t) / MAP_COEFF_C;
}

float SpectrumAnalyzerComponent::mapXToLogFreq(float x)
{
    // x: 正規化されたX座標 (0.0 ~ 1.0)
    return (std::sqrt(1.0f + MAP_COEFF_D * x) - 1.0f) / MAP_COEFF_A;
}

juce::Colour SpectrumAnalyzerComponent::getLevelColour(float normalizedLevel) const
{
    if (normalizedLevel < 0.33f)
        return juce::Colours::royalblue.interpolatedWith(juce::Colours::cyan, normalizedLevel / 0.33f);
    else if (normalizedLevel < 0.66f)
        return juce::Colours::cyan.interpolatedWith(juce::Colours::yellow, (normalizedLevel - 0.33f) / 0.33f);
    else
        return juce::Colours::yellow.interpolatedWith(juce::Colours::red, (normalizedLevel - 0.66f) / 0.34f);
}

void SpectrumAnalyzerComponent::updateSourceButtonText()
{
    if (engine.getAnalyzerSource() == AudioEngine::AnalyzerSource::Input)
        sourceButton.setButtonText("Analyzer: Input");
    else
        sourceButton.setButtonText("Analyzer: Output");
}
