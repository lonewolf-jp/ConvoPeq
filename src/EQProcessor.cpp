//============================================================================
// EQProcessor.cpp  ── v0.1 (JUCE 8.0.12対応)
//
// 20バンドパラメトリックイコライザー処理実装
// 参照: https://www.w3.org/2011/audio/audio-eq-cookbook.html
//============================================================================
#include "EQProcessor.h"
#include <cmath>
#include <algorithm>
#include <complex>
#include <numeric>

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
EQProcessor::EQProcessor()
{
    // 初期係数ノードの作成
    for (int i = 0; i < NUM_BANDS; ++i)
        bandNodes[i].store(nullptr);

    resetToDefaults();
}

//--------------------------------------------------------------
EQProcessor::~EQProcessor()
{
    // メモリ解放
    // shared_ptr により自動的に解放されるため、明示的な delete は不要
}

//--------------------------------------------------------------
// デフォルト値リセット
//--------------------------------------------------------------
void EQProcessor::resetToDefaults()
{
    auto newState = std::make_shared<EQState>();

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        newState->bands[i].frequency = DEFAULT_FREQS[i];
        newState->bands[i].gain = 0.0f;
        newState->bands[i].q = DEFAULT_Q;
        newState->bands[i].enabled = true;
        newState->bandChannelModes[i] = EQChannelMode::Stereo;
    }

    // バンドタイプ初期化
    newState->bandTypes[0] = EQBandType::LowShelf;
    for (int i = 1; i < 19; ++i)
        newState->bandTypes[i] = EQBandType::Peaking;
    newState->bandTypes[19] = EQBandType::HighShelf;

    currentState.store(newState, std::memory_order_release);

    agcCurrentGain = 1.0f;
    agcEnvInput    = 0.0f;
    agcEnvOutput   = 0.0f;

    // 全バンドの係数を更新
    for (int i = 0; i < NUM_BANDS; ++i)
        updateBandNode(i);

    sendChangeMessage();
}

//--------------------------------------------------------------
// loadPreset
//--------------------------------------------------------------
void EQProcessor::loadPreset(int /*index*/)
{
    // 簡易実装: プリセットIDに関わらずデフォルトに戻す
    resetToDefaults();
    sendChangeMessage();
}

//--------------------------------------------------------------
// EqualizerAPO形式のテキストファイルからプリセットを読み込む
//--------------------------------------------------------------
bool EQProcessor::loadFromTextFile(const juce::File& file)
{
    if (!file.existsAsFile())
        return false;

    // 最初に全バンドを無効化
    for (int i = 0; i < NUM_BANDS; ++i)
        setBandEnabled(i, false);

    juce::StringArray lines;
    file.readLines(lines);

    int currentFilterIndex = 0;

    for (const auto& line : lines)
    {
        auto trimmedLine = line.trim();
        if (trimmedLine.isEmpty())
            continue;

        // Preamp行の解析
        if (trimmedLine.startsWithIgnoreCase("Preamp:"))
        {
            auto valueStr = trimmedLine.fromFirstOccurrenceOf(":", false, false).trim();
            valueStr = valueStr.upToFirstOccurrenceOf("dB", false, true).trim();
            setTotalGain(valueStr.getFloatValue());
        }
        // Filter行の解析
        else if (trimmedLine.startsWithIgnoreCase("Filter"))
        {
            if (currentFilterIndex >= NUM_BANDS)
                continue; // 最大バンド数を超えたら無視

            auto tokens = juce::StringArray::fromTokens(trimmedLine, " ", "");
            if (tokens.size() < 8) // "Filter", "1:", "ON", "TYPE", "Fc", "X", "Hz", ...
                continue;

            // "ON" or "OFF"
            bool enabled = tokens[2].equalsIgnoreCase("ON");
            setBandEnabled(currentFilterIndex, enabled);

            // フィルタータイプ
            auto typeStr = tokens[3];
            if (typeStr.equalsIgnoreCase("LSC"))
                setBandType(currentFilterIndex, EQBandType::LowShelf);
            else if (typeStr.equalsIgnoreCase("PK"))
                setBandType(currentFilterIndex, EQBandType::Peaking);
            else if (typeStr.equalsIgnoreCase("HSC"))
                setBandType(currentFilterIndex, EQBandType::HighShelf);
            else if (typeStr.equalsIgnoreCase("LP"))
                setBandType(currentFilterIndex, EQBandType::LowPass);
            else if (typeStr.equalsIgnoreCase("HP"))
                setBandType(currentFilterIndex, EQBandType::HighPass);

            // パラメータを順番に解析
            float freq = 0.0f, gain = 0.0f, q = 0.0f;

            for (int i = 4; i < tokens.size() - 1; ++i)
            {
                if (tokens[i].equalsIgnoreCase("Fc"))
                {
                    freq = tokens[i + 1].getFloatValue();
                    i++;
                }
                else if (tokens[i].equalsIgnoreCase("Gain"))
                {
                    gain = tokens[i + 1].getFloatValue();
                    i++;
                }
                else if (tokens[i].equalsIgnoreCase("Q"))
                {
                    q = tokens[i + 1].getFloatValue();
                    i++;
                }
            }

            if (freq > 0.0f)
                setBandFrequency(currentFilterIndex, freq);

            // ゲインは常に設定
            setBandGain(currentFilterIndex, gain);

            if (q > 0.0f)
                setBandQ(currentFilterIndex, q);
            else // Qが指定されていない場合（シェルビングなど）はデフォルト値
                setBandQ(currentFilterIndex, DEFAULT_Q);

            // チャンネルモードは常にステレオ
            setBandChannelMode(currentFilterIndex, EQChannelMode::Stereo);

            currentFilterIndex++;
        }
    }

    sendChangeMessage();
    return true;
}

//--------------------------------------------------------------
// State Management
//--------------------------------------------------------------
juce::ValueTree EQProcessor::getState() const
{
    auto state = currentState.load();

    juce::ValueTree v ("EQ");
    v.setProperty ("totalGain", state->totalGainDb, nullptr);
    v.setProperty ("agcEnabled", state->agcEnabled, nullptr);

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        juce::ValueTree band ("Band");
        band.setProperty ("index", i, nullptr);
        band.setProperty ("enabled", state->bands[i].enabled, nullptr);
        band.setProperty ("freq", state->bands[i].frequency, nullptr);
        band.setProperty ("gain", state->bands[i].gain, nullptr);
        band.setProperty ("q", state->bands[i].q, nullptr);
        band.setProperty ("type", (int)state->bandTypes[i], nullptr);
        band.setProperty ("channel", (int)state->bandChannelModes[i], nullptr);
        v.addChild (band, -1, nullptr);
    }
    return v;
}

void EQProcessor::setState (const juce::ValueTree& v)
{
    if (v.hasProperty ("totalGain")) setTotalGain (v.getProperty ("totalGain"));
    if (v.hasProperty ("agcEnabled")) setAGCEnabled (v.getProperty ("agcEnabled"));

    for (const auto& band : v)
    {
        if (band.hasType ("Band") && band.hasProperty ("index"))
        {
            int i = band.getProperty ("index");
            if (i >= 0 && i < NUM_BANDS)
            {
                if (band.hasProperty ("enabled")) setBandEnabled (i, band.getProperty ("enabled"));
                if (band.hasProperty ("freq"))    setBandFrequency (i, band.getProperty ("freq"));
                if (band.hasProperty ("gain"))    setBandGain (i, band.getProperty ("gain"));
                if (band.hasProperty ("q"))       setBandQ (i, band.getProperty ("q"));
                if (band.hasProperty ("type"))    setBandType (i, (EQBandType)(int)band.getProperty ("type"));
                if (band.hasProperty ("channel")) setBandChannelMode (i, (EQChannelMode)(int)band.getProperty ("channel"));
            }
        }
    }
    sendChangeMessage();
}

//--------------------------------------------------------------
// prepareToPlay
// サンプルレート変更時にフィルタ状態を全リセットする
// この関数は Audio Thread 開始前、または停止後に呼ばれる
//--------------------------------------------------------------
void EQProcessor::prepareToPlay(int sampleRate, int /*samplesPerBlock*/)
{
    currentSampleRate = sampleRate;

    auto state = currentState.load(std::memory_order_acquire);

    smoothTotalGain.reset(sampleRate, SMOOTHING_TIME_SEC);
    smoothTotalGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain(state->totalGainDb));

    // フィルタ状態を全てゼロにリセット
    // (古いサンプルレートで計算された遅延要素を使い続けると
    //  音が不安定になる)
    for (int ch = 0; ch < MAX_CHANNELS; ++ch)
        for (int band = 0; band < NUM_BANDS; ++band)
            for (int z = 0; z < 2; ++z)
                filterState[ch][band][z] = 0.0;

    agcCurrentGain = 1.0f;
    agcEnvInput    = 0.0f;
    agcEnvOutput   = 0.0f;

    // 係数を即座に再計算
    for (int i = 0; i < NUM_BANDS; ++i)
        updateBandNode(i);
}

//--------------------------------------------------------------
// パラメータ変更メソッド (UIスレッドから呼ぶ)
// 各メソッドは atomic store で値を書き込み、coeffsDirty を立てる
//--------------------------------------------------------------
void EQProcessor::setBandFrequency(int band, float freq)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].frequency = freq;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    sendChangeMessage();
}

void EQProcessor::setBandGain(int band, float gainDb)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].gain = gainDb;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    sendChangeMessage();
}

void EQProcessor::setBandQ(int band, float q)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].q = q;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    sendChangeMessage();
}

void EQProcessor::setBandEnabled(int band, bool enabled)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].enabled = enabled;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    sendChangeMessage();
}

void EQProcessor::setTotalGain(float gainDb)
{
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->totalGainDb = gainDb;
    currentState.store(newState, std::memory_order_release);
    sendChangeMessage();
}

float EQProcessor::getTotalGain() const
{
    return currentState.load(std::memory_order_acquire)->totalGainDb;
}

void EQProcessor::setAGCEnabled(bool enabled)
{
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->agcEnabled = enabled;
    currentState.store(newState, std::memory_order_release);
    sendChangeMessage();
}

bool EQProcessor::getAGCEnabled() const
{
    return currentState.load(std::memory_order_acquire)->agcEnabled;
}

void EQProcessor::setBandType(int band, EQBandType type)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bandTypes[band] = type;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    sendChangeMessage();
}

EQBandType EQProcessor::getBandType(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQBandType::Peaking;
    return currentState.load(std::memory_order_acquire)->bandTypes[band];
}

void EQProcessor::setBandChannelMode(int band, EQChannelMode mode)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bandChannelModes[band] = mode;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    sendChangeMessage();
}

EQChannelMode EQProcessor::getBandChannelMode(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQChannelMode::Stereo;
    return currentState.load(std::memory_order_acquire)->bandChannelModes[band];
}

//--------------------------------------------------------------
// パラメータ読み取り (UIスレッド用)
//--------------------------------------------------------------
EQBandParams EQProcessor::getBandParams(int band) const
{
    // band が範囲外の場合はデフォルト値を返す
    if (band < 0 || band >= NUM_BANDS) return {};
    return currentState.load(std::memory_order_acquire)->bands[band];
}

namespace
{
//--------------------------------------------------------------
// 単一チャンネル・単一バンドのフィルタ処理 (TPT SVF)
// Topology-Preserving Transform State Variable Filter
// 参照: Vadim Zavalishin "The Art of VA Filter Design"
//--------------------------------------------------------------
    inline void processBand (double* data, int numSamples,
                             const EQCoeffsSVF& c,
                             double* state)
    {
        double ic1eq = state[0];
        double ic2eq = state[1];

        const double a1 = c.a1;
        const double a2 = c.a2;
        const double a3 = c.a3;
        const double m0 = c.m0;
        const double m1 = c.m1;
        const double m2 = c.m2;

        static constexpr double DENORMAL_THRESHOLD = 1.0e-15;

        for (int n = 0; n < numSamples; ++n)
        {
            const double v0 = data[n];
            const double v3 = v0 - ic2eq;
            const double v1 = a1 * ic1eq + a2 * v3;
            const double v2 = ic2eq + a2 * ic1eq + a3 * v3;

            ic1eq = 2.0 * v1 - ic1eq;
            ic2eq = 2.0 * v2 - ic2eq;

            // 安全対策: 出力をクランプ (-10.0 ~ +10.0) して発散防止
            data[n] = juce::jlimit(-10.0, 10.0, m0 * v0 + m1 * v1 + m2 * v2);
            // Denormal対策: 出力が極小値なら0にする (Branchless optimization)
            data[n] = (std::abs(data[n]) < DENORMAL_THRESHOLD) ? 0.0 : data[n];
        }

        // Denormal対策: 状態変数が極小値ならフラッシュ (再循環防止)
        // Note: ScopedNoDenormals (DAZ/FTZ) が有効な場合でも、完全な0にならないと
        // 極小値が循環し続ける可能性があるため、明示的にフラッシュして計算負荷を抑える。
        ic1eq = (std::abs(ic1eq) < DENORMAL_THRESHOLD) ? 0.0 : ic1eq;
        ic2eq = (std::abs(ic2eq) < DENORMAL_THRESHOLD) ? 0.0 : ic2eq;

        state[0] = ic1eq;
        state[1] = ic2eq;
    }
}

//--------------------------------------------------------------
// AGCゲイン計算 (Private)
//--------------------------------------------------------------
float EQProcessor::calculateAGCGain(float inputEnv, float outputEnv) const noexcept
{
    static constexpr float MIN_ENV = 0.0001f;

    float targetGain = 1.0f;
    if (outputEnv > MIN_ENV)
    {
        targetGain = inputEnv / outputEnv;
    }
    else if (inputEnv > MIN_ENV)
    {
        targetGain = 1.0f;
    }

    return juce::jlimit(AGC_MIN_GAIN, AGC_MAX_GAIN, targetGain);
}

//--------------------------------------------------------------
// AGC処理 (Private)
//--------------------------------------------------------------
void EQProcessor::processAGC(juce::AudioBuffer<double>& buffer, int numSamples, const EQState& state)
{
    const int numChannels = std::min(buffer.getNumChannels(), MAX_CHANNELS);

    // 入力レベル計測 (RMS)
    double inputRMS = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        double rms = buffer.getRMSLevel(ch, 0, numSamples);
        if (rms > inputRMS) inputRMS = rms;
    }

    // フィルタ処理後の出力レベル計測
    double outputRMS = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        double rms = buffer.getRMSLevel(ch, 0, numSamples);
        if (rms > outputRMS) outputRMS = rms;
    }

    // 数値安定性対策: NaN/Infチェックとクランプ
    // 入力が極端に大きい場合（発振など）、エンベロープが汚染されるのを防ぐ
    static constexpr double MAX_ENV_VALUE = 1000.0; // +60dB

    if (!std::isfinite(inputRMS) || inputRMS > MAX_ENV_VALUE)   inputRMS = MAX_ENV_VALUE;
    if (!std::isfinite(outputRMS) || outputRMS > MAX_ENV_VALUE) outputRMS = MAX_ENV_VALUE;

    if (!std::isfinite(agcEnvInput))  agcEnvInput = 0.0f;
    if (!std::isfinite(agcEnvOutput)) agcEnvOutput = 0.0f;
    if (!std::isfinite(agcCurrentGain)) agcCurrentGain = 1.0f;

    // 指数移動平均 (EMA) によるエンベロープ検波
    agcEnvInput  = agcEnvInput  * (1.0f - AGC_ALPHA) + static_cast<float>(inputRMS)  * AGC_ALPHA;
    agcEnvOutput = agcEnvOutput * (1.0f - AGC_ALPHA) + static_cast<float>(outputRMS) * AGC_ALPHA;

    // ターゲットゲイン計算
    float targetGain = calculateAGCGain(agcEnvInput, agcEnvOutput);

    // ゲイン変化のスムーシング
    agcCurrentGain = agcCurrentGain * (1.0f - AGC_GAIN_SMOOTH) + targetGain * AGC_GAIN_SMOOTH;

    // ゲイン適用
    // 各チャンネルに対して明示的に適用
    for (int ch = 0; ch < numChannels; ++ch)
    {
        buffer.applyGain(ch, 0, numSamples, static_cast<double>(agcCurrentGain));
    }
}

//--------------------------------------------------------------
// サイレンス検出 (Private)
//--------------------------------------------------------------
bool EQProcessor::isBufferSilent(const juce::AudioBuffer<double>& buffer, int numSamples) const noexcept
{
    for (int ch = 0; ch < buffer.getNumChannels(); ++ch)
    {
        if (buffer.getMagnitude(ch, 0, numSamples) > 1.0e-8)
            return false;
    }
    return true;
}

//--------------------------------------------------------------
// process (Audio Thread)
// リアルタイム制約 (Real-time Constraints)
//    - メモリ確保なし (No Malloc)
//    - ロックなし (No Lock)
//    - ファイルI/Oなし (No I/O)
//    - 待機なし (No Wait): 処理落ちの原因となるため (Audio Threadでの待機は厳禁)
//    - RCU (Read-Copy-Update) パターンにより、ロックフリーで安全に係数を更新
//--------------------------------------------------------------
void EQProcessor::process(juce::AudioBuffer<double>& buffer, int numSamples)
{
    juce::ScopedNoDenormals noDenormals;

    // サイレンス最適化
    // 入力が無音の場合は処理をスキップし、CPU負荷を低減する
    // (IIRフィルタのテールはカットされるが、EQ用途では許容範囲とする)
    if (isBufferSilent(buffer, numSamples))
        return;

    auto state = currentState.load(std::memory_order_acquire); // RCU Load (Lock-free)

    const int numChannels = std::min(buffer.getNumChannels(), MAX_CHANNELS);

    // フィルタバンク適用
    for (int ch = 0; ch < numChannels; ++ch)
    {
        double* data = buffer.getWritePointer(ch);
        if (data == nullptr) continue;

        for (int bandIndex = 0; bandIndex < NUM_BANDS; ++bandIndex)
        {
            // Atomic load: 係数ポインタを安全に取得
            auto node = bandNodes[bandIndex].load(std::memory_order_acquire);
            if (!node || !node->active) continue;

            const EQChannelMode mode = node->mode;
            if (mode == EQChannelMode::Left && ch != 0) continue;
            if (mode == EQChannelMode::Right && ch != 1) continue;

            processBand(data, numSamples, node->coeffs, filterState[ch][bandIndex]);
        }
    }

    // トータルゲイン / AGC 適用
    if (state->agcEnabled)
    {
        processAGC(buffer, numSamples, *state);
    }
    else
    {
        const float startGain = smoothTotalGain.getCurrentValue();
        smoothTotalGain.skip(numSamples);
        const float endGain = smoothTotalGain.getCurrentValue();

        for (int ch = 0; ch < numChannels; ++ch)
        {
            buffer.applyGainRamp(ch, 0, numSamples,
                                 static_cast<double>(startGain),
                                 static_cast<double>(endGain));
        }
    }
}

//--------------------------------------------------------------
// BandNode作成 (Message Thread)
//--------------------------------------------------------------
std::shared_ptr<EQProcessor::BandNode> EQProcessor::createBandNode(int band, const EQState& state) const
{
    auto node = std::make_shared<BandNode>();
    const auto& params = state.bands[band];

    node->active = params.enabled;
    node->mode = state.bandChannelModes[band];
    node->coeffs = calcSVFCoeffs(state.bandTypes[band], params.frequency, params.gain, params.q, currentSampleRate);

    // 最適化: ゲインが0dB付近ならスキップ
    if (node->active) {
        EQBandType type = state.bandTypes[band];
        if ((type != EQBandType::LowPass && type != EQBandType::HighPass) && std::abs(params.gain) < 0.01f)
            node->active = false;
    }

    return node;
}

//--------------------------------------------------------------
// BandNode更新 (Message Thread)
//--------------------------------------------------------------
void EQProcessor::updateBandNode(int band)
{
    auto state = currentState.load(std::memory_order_acquire);
    auto newNode = createBandNode(band, *state);

    // Atomic Exchange: 古いノードを取得し、新しいノードをセット
    auto oldNode = bandNodes[band].exchange(newNode, std::memory_order_acq_rel);

    // 古いノードをゴミ箱へ (Audio Threadが使用中の可能性があるため即削除しない)
    const juce::ScopedLock sl(trashBinLock);
    if (oldNode)
        trashBin.push_back(oldNode);

    // ゴミ箱のサイズ制限 (メモリ肥大化防止)
    // 古いノードから削除する。十分なサイズを確保しておけば、Audio Threadが参照している可能性は極めて低い。
    if (trashBin.size() > 100)
    {
        trashBin.erase(trashBin.begin());
    }
}

//--------------------------------------------------------------
// パラメータ検証とクランプ (Helper)
//--------------------------------------------------------------
void EQProcessor::validateAndClampParameters(float& freq, float& gainDb, float& q, int sr) noexcept
{
    // 周波数をナイキスト周波数以下にクランプ
    const float nyquist = static_cast<float>(sr) * 0.5f;
    freq = juce::jlimit(DSP_MIN_FREQ, nyquist * DSP_MAX_FREQ_NYQUIST_RATIO, freq);

    // Qを安全な範囲にクランプ
    q = juce::jlimit(DSP_MIN_Q, DSP_MAX_Q, q);

    // ゲインを実用範囲にクランプ
    gainDb = juce::jlimit(DSP_MIN_GAIN_DB, DSP_MAX_GAIN_DB, gainDb);
}

//--------------------------------------------------------------
// SVF係数計算 (Audio Thread用)
//--------------------------------------------------------------
EQCoeffsSVF EQProcessor::calcSVFCoeffs(EQBandType type, float freq, float gainDb, float q, int sr) noexcept
{
    // パラメータ検証 (Parameter Validation)
    // 不正な値から保護し、安全な範囲にクランプ
    if (sr <= 0 || sr > 384000)
    {
        jassertfalse;
        sr = 48000; // デフォルト値
    }

    validateAndClampParameters(freq, gainDb, q, sr);

    const double f = static_cast<double>(freq);
    const double g = static_cast<double>(gainDb);
    const double Q = static_cast<double>(q);
    const double s = static_cast<double>(sr);

    switch (type)
    {
        case EQBandType::LowShelf:  return calcLowShelfSVF(f, g, Q, s);
        case EQBandType::Peaking:   return calcPeakingSVF(f, g, Q, s);
        case EQBandType::HighShelf: return calcHighShelfSVF(f, g, Q, s);
        case EQBandType::LowPass:   return calcLowPassSVF(f, Q, s);
        case EQBandType::HighPass:  return calcHighPassSVF(f, Q, s);
    }
    return {}; // unreachable
}

//--------------------------------------------------------------
// Biquad係数計算 (UI Thread用)
//--------------------------------------------------------------
EQCoeffsBiquad EQProcessor::calcBiquadCoeffs(EQBandType type, float freq, float gainDb, float q, int sr) noexcept
{
    // パラメータ検証 (Parameter Validation)
    if (sr <= 0 || sr > 384000)
    {
        jassertfalse;
        sr = 48000;
    }

    validateAndClampParameters(freq, gainDb, q, sr);

    const double f = static_cast<double>(freq);
    const double g = static_cast<double>(gainDb);
    const double Q = static_cast<double>(q);
    const double s = static_cast<double>(sr);

    switch (type)
    {
        case EQBandType::LowShelf:  return calcLowShelfBiquad(f, g, Q, s);
        case EQBandType::Peaking:   return calcPeakingBiquad(f, g, Q, s);
        case EQBandType::HighShelf: return calcHighShelfBiquad(f, g, Q, s);
        case EQBandType::LowPass:   return calcLowPassBiquad(f, Q, s);
        case EQBandType::HighPass:  return calcHighPassBiquad(f, Q, s);
    }
    return {};
}

//--------------------------------------------------------------
// 周波数応答（マグニチュードの二乗）計算
// sqrtを避けるため、連鎖的な計算やdB変換前の最適化に使用
//--------------------------------------------------------------
float EQProcessor::getMagnitudeSquared(const EQCoeffsBiquad& c, float freq, float sampleRate) noexcept
{
    const double w = 2.0 * juce::MathConstants<double>::pi * static_cast<double>(freq) / static_cast<double>(sampleRate);
    std::complex<double> z(std::cos(w), std::sin(w));
    return getMagnitudeSquared(c, z);
}

float EQProcessor::getMagnitudeSquared(const EQCoeffsBiquad& c, const std::complex<double>& z) noexcept
{
    std::complex<double> z2 = z * z;
    std::complex<double> num = c.b0 * z2 + c.b1 * z + c.b2;
    std::complex<double> den = c.a0 * z2 + c.a1 * z + c.a2;

    double denNorm = std::norm(den); // norm returns magnitude squared
    if (denNorm < 1e-18) return 0.0f;

    return static_cast<float>(std::norm(num) / denNorm);
}

//--------------------------------------------------------------
// SVF Implementations
//--------------------------------------------------------------
EQCoeffsSVF EQProcessor::calcLowShelfSVF(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double A = std::pow(10.0, gainDb / 40.0);
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr) / std::sqrt(A);
    const double k = 1.0 / q;

    // 除算ゼロ保護 (Division by Zero Protection)
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 1.0;
    c.m1 = k * (A - 1.0);
    c.m2 = A * A - 1.0;
    return c;
}

EQCoeffsSVF EQProcessor::calcPeakingSVF(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double A = std::pow(10.0, gainDb / 40.0);
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
    const double k = 1.0 / (q * A);

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 1.0;
    c.m1 = k * (A * A - 1.0);
    c.m2 = 0.0;
    return c;
}

EQCoeffsSVF EQProcessor::calcHighShelfSVF(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double A = std::pow(10.0, gainDb / 40.0);
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr) * std::sqrt(A);
    const double k = 1.0 / q;

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = A * A;
    c.m1 = k * (1.0 - A) * A;
    c.m2 = 1.0 - A * A;
    return c;
}

EQCoeffsSVF EQProcessor::calcLowPassSVF(double freq, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
    const double k = 1.0 / q;

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 0.0;
        c.m1 = 0.0;
        c.m2 = 1.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 0.0;
    c.m1 = 0.0;
    c.m2 = 1.0;
    return c;
}

EQCoeffsSVF EQProcessor::calcHighPassSVF(double freq, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr);
    const double k = 1.0 / q;

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0;
        c.m1 = 0.0;
        c.m2 = 0.0;
        return c;
    }

    c.a1 = 1.0 / denominator;
    c.a2 = g * c.a1;
    c.a3 = g * c.a2;
    c.m0 = 1.0;
    c.m1 = -k;
    c.m2 = -1.0;
    return c;
}

//--------------------------------------------------------------
// Biquad Implementations (Audio EQ Cookbook)
//--------------------------------------------------------------
EQCoeffsBiquad EQProcessor::calcLowShelfBiquad(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double A     = std::pow(10.0, gainDb / 40.0);
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double sqrtA = std::sqrt(A);
    const double twoSqrtAAlpha = 2.0 * sqrtA * alpha;

    c.b0 =       A * ((A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.b1 =  2.0 * A * ((A - 1.0) - (A + 1.0) * cosw0);
    c.b2 =  A * ((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha);
    c.a0 =           ((A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.a1 = -2.0     * ((A - 1.0) + (A + 1.0) * cosw0               );
    c.a2 =           ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha);
    return c;
}

EQCoeffsBiquad EQProcessor::calcPeakingBiquad(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double A     = std::pow(10.0, gainDb / 40.0);
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);

    c.b0 =  1.0 + alpha * A;
    c.b1 = -2.0 * cosw0;
    c.b2 =  1.0 - alpha * A;
    c.a0 =  1.0 + alpha / A;
    c.a1 = -2.0 * cosw0;
    c.a2 =  1.0 - alpha / A;
    return c;
}

EQCoeffsBiquad EQProcessor::calcHighShelfBiquad(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double A     = std::pow(10.0, gainDb / 40.0);
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);
    const double sqrtA = std::sqrt(A);
    const double twoSqrtAAlpha = 2.0 * sqrtA * alpha;

    c.b0 =       A * ((A + 1.0) + (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.b1 = -2.0 * A * ((A - 1.0) + (A + 1.0) * cosw0               );
    c.b2 =       A * ((A + 1.0) + (A - 1.0) * cosw0 - twoSqrtAAlpha);
    c.a0 =           ((A + 1.0) - (A - 1.0) * cosw0 + twoSqrtAAlpha);
    c.a1 =  2.0     * ((A - 1.0) - (A + 1.0) * cosw0               );
    c.a2 =           ((A + 1.0) - (A - 1.0) * cosw0 - twoSqrtAAlpha);
    return c;
}

EQCoeffsBiquad EQProcessor::calcLowPassBiquad(double freq, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);

    c.b0 =  (1.0 - cosw0) / 2.0;
    c.b1 =   1.0 - cosw0;
    c.b2 =  (1.0 - cosw0) / 2.0;
    c.a0 =   1.0 + alpha;
    c.a1 =  -2.0 * cosw0;
    c.a2 =   1.0 - alpha;
    return c;
}

EQCoeffsBiquad EQProcessor::calcHighPassBiquad(double freq, double q, double sr) noexcept
{
    EQCoeffsBiquad c;
    const double w0    = 2.0 * juce::MathConstants<double>::pi * freq / sr;
    const double cosw0 = std::cos(w0);
    const double alpha = std::sin(w0) / (2.0 * q);

    c.b0 =  (1.0 + cosw0) / 2.0;
    c.b1 = -(1.0 + cosw0);
    c.b2 =  (1.0 + cosw0) / 2.0;
    c.a0 =   1.0 + alpha;
    c.a1 =  -2.0 * cosw0;
    c.a2 =   1.0 - alpha;
    return c;
}
