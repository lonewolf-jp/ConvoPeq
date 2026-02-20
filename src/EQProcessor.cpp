//============================================================================
// EQProcessor.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// 20バンドパラメトリックイコライザー処理実装
// 参照: Vadim Zavalishin "The Art of VA Filter Design" (TPT SVF)
//       https://www.w3.org/2011/audio/audio-eq-cookbook.html (Biquad Coeffs)
//============================================================================
#include "EQProcessor.h"
#include <cmath>
#include <algorithm>
#include <complex>
#include <numeric>
#include <cstring>
#include <regex>

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
EQProcessor::EQProcessor()
{
    // 初期係数ノードの作成
    for (auto& node : bandNodes)
        node.store(nullptr);

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

    totalGainDbTarget.store(0.0f, std::memory_order_relaxed);
    agcEnabled.store(false, std::memory_order_release);
    currentState.store(newState, std::memory_order_release);

    agcCurrentGain.store(1.0, std::memory_order_relaxed);
    agcEnvInput.store(0.0, std::memory_order_relaxed);
    agcEnvOutput.store(0.0, std::memory_order_relaxed);

    // 全バンドの係数を更新
    for (int i = 0; i < NUM_BANDS; ++i)
        updateBandNode(i);

    sendChangeMessage();
}

void EQProcessor::reset()
{
    // フィルタ状態をリセット
    for (auto& channelState : filterState)
    {
        for (auto& bandState : channelState)
            bandState.fill(0.0);
    }

    agcCurrentGain.store(1.0, std::memory_order_relaxed);
    agcEnvInput.store(0.0, std::memory_order_relaxed);
    agcEnvOutput.store(0.0, std::memory_order_relaxed);

    auto state = currentState.load(std::memory_order_acquire);
    if (state)
    {
        smoothTotalGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
        totalGainDbTarget.store(state->totalGainDb, std::memory_order_relaxed);
    }
}

//--------------------------------------------------------------
// プリセット読み込み
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
    bool maxBandsWarningShown = false;

    for (auto line : lines)
    {
        // コメント除去 (# と ;)
        line = line.upToFirstOccurrenceOf("#", false, false);
        line = line.upToFirstOccurrenceOf(";", false, false);
        line = line.trim();

        if (line.isEmpty())
            continue;

        // 正規表現でトークン分割 (空白区切り)
        // std::regex(R"(\S+)") は空白以外の文字の並び（トークン）にマッチします
        std::regex tokenRegex(R"(\S+)");
        auto tokensBegin = std::sregex_iterator(line.toStdString().begin(), line.toStdString().end(), tokenRegex);
        auto tokensEnd = std::sregex_iterator();

        juce::StringArray tokens;
        for (auto i = tokensBegin; i != tokensEnd; ++i)
            tokens.add(i->str());

        if (tokens.isEmpty()) continue;

        // Preamp行の解析 (例: "Preamp: -6.0 dB")
        if (tokens[0].startsWithIgnoreCase("Preamp"))
        {
            for (int i = 1; i < tokens.size(); ++i)
            {
                // 数値を含むトークンを探す
                if (tokens[i].containsAnyOf("0123456789-."))
                {
                    float val = tokens[i].getFloatValue();
                    setTotalGain(val);
                    break;
                }
            }
        }
        // Filter行の解析 (例: "Filter 1: ON PK Fc 100 Hz Gain -3.0 dB Q 2.0")
        else if (tokens[0].startsWithIgnoreCase("Filter"))
        {
            if (currentFilterIndex >= NUM_BANDS)
            {
                if (!maxBandsWarningShown)
                {
                    maxBandsWarningShown = true;
                    // ユーザーへの警告 (Message Threadなので安全)
                    juce::MessageManager::callAsync([] {
                        juce::AlertWindow::showMessageBoxAsync(
                            juce::AlertWindow::WarningIcon,
                            "Load Preset Warning",
                            "The preset contains more bands than supported (Max 20). Extra bands were ignored."
                        );
                    });
                }
                DBG("Skipping extra band: " + line);
                continue;
            }

            // ON/OFF を探す (位置が固定でない場合に対応)
            int onOffIndex = -1;
            bool enabled = true;
            for (int i = 1; i < tokens.size(); ++i)
            {
                if (tokens[i].equalsIgnoreCase("ON")) {
                    onOffIndex = i;
                    enabled = true;
                    break;
                }
                if (tokens[i].equalsIgnoreCase("OFF")) {
                    onOffIndex = i;
                    enabled = false;
                    break;
                }
            }

            if (onOffIndex == -1)
            {
                DBG("Invalid Filter line (No ON/OFF found): " + line);
                continue;
            }

            setBandEnabled(currentFilterIndex, enabled);

            // フィルタータイプ (ON/OFFの次のトークンと仮定)
            if (onOffIndex + 1 < tokens.size())
            {
                juce::String typeStr = tokens[onOffIndex + 1];
                if (typeStr.equalsIgnoreCase("LSC"))      setBandType(currentFilterIndex, EQBandType::LowShelf);
                else if (typeStr.equalsIgnoreCase("PK"))  setBandType(currentFilterIndex, EQBandType::Peaking);
                else if (typeStr.equalsIgnoreCase("HSC")) setBandType(currentFilterIndex, EQBandType::HighShelf);
                else if (typeStr.equalsIgnoreCase("LP"))  setBandType(currentFilterIndex, EQBandType::LowPass);
                else if (typeStr.equalsIgnoreCase("HP"))  setBandType(currentFilterIndex, EQBandType::HighPass);
            }

            // パラメータ解析 (Fc, Gain, Q)
            float freq = 0.0f, gain = 0.0f, q = 0.0f;

            for (int i = onOffIndex + 2; i < tokens.size(); ++i)
            {
                // キーワードの次のトークンを値として取得
                if (tokens[i].equalsIgnoreCase("Fc") && i + 1 < tokens.size())
                {
                    freq = tokens[i + 1].getFloatValue();
                }
                else if (tokens[i].equalsIgnoreCase("Gain") && i + 1 < tokens.size())
                {
                    gain = tokens[i + 1].getFloatValue();
                }
                else if (tokens[i].equalsIgnoreCase("Q") && i + 1 < tokens.size())
                {
                    q = tokens[i + 1].getFloatValue();
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
// 状態管理
//--------------------------------------------------------------
juce::ValueTree EQProcessor::getState() const
{
    auto state = currentState.load();

    juce::ValueTree v ("EQ");
    v.setProperty ("totalGain", state->totalGainDb, nullptr);

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
    setAGCEnabled(v.getProperty("agcEnabled", false));

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

void EQProcessor::syncStateFrom(const EQProcessor& other)
{
    // Copy atomics
    totalGainDbTarget.store(other.totalGainDbTarget.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // Copy shared state
    currentState.store(other.currentState.load(std::memory_order_acquire), std::memory_order_release);

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto node = other.bandNodes[i].load(std::memory_order_acquire);
        bandNodes[i].store(node, std::memory_order_release);
    }
    agcEnabled.store(other.agcEnabled.load(std::memory_order_acquire), std::memory_order_release);

    // Note: smoothTotalGain target is updated in process() based on totalGainDbTarget
}

void EQProcessor::syncBandNodeFrom(const EQProcessor& other, int bandIndex)
{
    if (bandIndex < 0 || bandIndex >= NUM_BANDS) return;

    auto node = other.bandNodes[bandIndex].load(std::memory_order_acquire);

    // Atomic Exchange: 古いノードを取得し、新しいノードをセット
    auto oldNode = bandNodes[bandIndex].exchange(node, std::memory_order_acq_rel);

    // 古いノードをゴミ箱へ (Audio Threadでの削除防止)
    if (oldNode)
    {
        const juce::ScopedLock sl(trashBinLock);
        trashBin.push_back(std::move(oldNode));
        // ゴミ箱の掃除
        trashBin.erase(std::remove_if(trashBin.begin(), trashBin.end(), [](const auto& p) { return p.use_count() == 1; }), trashBin.end());
    }
}

void EQProcessor::syncGlobalStateFrom(const EQProcessor& other)
{
    totalGainDbTarget.store(other.totalGainDbTarget.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnabled.store(other.agcEnabled.load(std::memory_order_acquire), std::memory_order_release);
}

//--------------------------------------------------------------
// prepareToPlay
// サンプルレート変更時にフィルタ状態を全リセットする
// この関数は Audio Thread 開始前、または停止後に呼ばれる
//--------------------------------------------------------------
void EQProcessor::prepareToPlay(double sampleRate, int /*samplesPerBlock*/)
{
    const bool rateChanged = (std::abs(currentSampleRate - sampleRate) > 1.0);
    if (rateChanged)
        currentSampleRate = sampleRate;

    auto state = currentState.load(std::memory_order_acquire);

    // reset()を呼び、スムーシングの状態を初期化（現在値は0.0になる）
    smoothTotalGain.reset(sampleRate, SMOOTHING_TIME_SEC);

    // 初期化直後のフェードイン（0.0 -> Target）を防ぐため、
    // 現在値をターゲット値に即座に設定する。
    if (state)
    {
        totalGainDbTarget.store(state->totalGainDb, std::memory_order_relaxed);
        smoothTotalGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
        // ダミー呼び出し: 内部状態の確実な初期化 (メモリ確保リスクの排除)
        (void)smoothTotalGain.getNextValue();
    }

    // フィルタ状態をリセット
    for (auto& channelState : filterState)
    {
        for (auto& bandState : channelState)
            bandState.fill(0.0);
    }

    agcCurrentGain.store(1.0, std::memory_order_relaxed);
    agcEnvInput.store(0.0, std::memory_order_relaxed);
    agcEnvOutput.store(0.0, std::memory_order_relaxed);

    // 係数を即座に再計算 (レート変更時のみ)
    if (rateChanged)
    {
        for (int i = 0; i < NUM_BANDS; ++i)
            updateBandNode(i);
    }
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
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setBandGain(int band, float gainDb)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].gain = gainDb;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setBandQ(int band, float q)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].q = q;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setBandEnabled(int band, bool enabled)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].enabled = enabled;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setTotalGain(float gainDb)
{
    // パラメータを安全な範囲にクランプ
    gainDb = juce::jlimit(DSP_MIN_GAIN_DB, DSP_MAX_GAIN_DB, gainDb);

    // ✅ Atomicに保存（Audio Threadで読み取る）
    totalGainDbTarget.store(gainDb, std::memory_order_relaxed);

    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->totalGainDb = gainDb;
    currentState.store(newState, std::memory_order_release);
    listeners.call(&Listener::eqGlobalChanged, this);
}

float EQProcessor::getTotalGain() const
{
    return currentState.load(std::memory_order_acquire)->totalGainDb;
}

void EQProcessor::setAGCEnabled(bool enabled)
{
    agcEnabled.store(enabled, std::memory_order_release);
    listeners.call(&Listener::eqGlobalChanged, this);
}

bool EQProcessor::getAGCEnabled() const
{
    return agcEnabled.load(std::memory_order_acquire);
}

void EQProcessor::setBandType(int band, EQBandType type)
{
    if (band < 0 || band >= NUM_BANDS) return;

    auto oldState = currentState.load(std::memory_order_acquire);
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bandTypes[band] = type;
    currentState.store(newState, std::memory_order_release);
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
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
    listeners.call(&Listener::eqBandChanged, this, band);
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

        constexpr double DENORMAL_THRESHOLD = 1.0e-15;

        for (int n = 0; n < numSamples; ++n)
        {
            const double v0 = data[n];
            const double v3 = v0 - ic2eq;
            const double v1 = a1 * ic1eq + a2 * v3;
            const double v2 = ic2eq + a2 * ic1eq + a3 * v3;

            ic1eq = 2.0 * v1 - ic1eq;
            ic2eq = 2.0 * v2 - ic2eq;


            double output = m0 * v0 + m1 * v1 + m2 * v2;
            // 出力のNaNチェック (数値安定性のため)
            if (!std::isfinite(output))
                output = 0.0;

            // 安全対策: 出力をクランプ (-100.0 ~ +100.0) して発散防止
            data[n] = juce::jlimit(-100.0, 100.0, output);
            // Denormal対策: 出力が極小値なら0にする (Branchless optimization)
        }

        // Denormal対策 & NaNチェック
        // Note: ScopedNoDenormals (DAZ/FTZ) が有効な場合でも、完全な0にならないと
        // 極小値が循環し続ける可能性があるため、明示的にフラッシュして計算負荷を抑える。
        if (!std::isfinite(ic1eq) || std::abs(ic1eq) < DENORMAL_THRESHOLD) ic1eq = 0.0;
        if (!std::isfinite(ic2eq) || std::abs(ic2eq) < DENORMAL_THRESHOLD) ic2eq = 0.0;

        state[0] = ic1eq;
        state[1] = ic2eq;
    }
}

//--------------------------------------------------------------
// AGCゲイン計算 (Private)
//--------------------------------------------------------------
double EQProcessor::calculateAGCGain(double inputEnv, double outputEnv) const noexcept
{
    static constexpr double MIN_ENV = 0.0001;

    double targetGain = 1.0;
    if (outputEnv > MIN_ENV)
    {
        targetGain = inputEnv / outputEnv;
    }
    else if (inputEnv > MIN_ENV)
    {
        targetGain = 1.0;
    }

    return juce::jlimit(static_cast<double>(AGC_MIN_GAIN), static_cast<double>(AGC_MAX_GAIN), targetGain);
}

//--------------------------------------------------------------
// AGC処理 (Private)
//--------------------------------------------------------------
void EQProcessor::processAGC(juce::dsp::AudioBlock<double>& block)
{
    const int numChannels = std::min((int)block.getNumChannels(), MAX_CHANNELS);
    const int numSamples = (int)block.getNumSamples();

    // ✅ 事前にキャッシュされた入力レベルを使用
    double inputRMS = cachedInputRMS;

    // ✅ フィルタ処理後の出力レベル計測
    double outputRMS = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        // Manual RMS calculation for AudioBlock
        double sumSq = 0.0;
        const double* data = block.getChannelPointer(ch);
        for (int i = 0; i < numSamples; ++i)
            sumSq += data[i] * data[i];
        double rms = std::sqrt(sumSq / static_cast<double>(numSamples));

        if (rms > outputRMS) outputRMS = rms;
    }

    // 数値安定性対策: NaN/Infチェックとクランプ
    // 入力が極端に大きい場合（発振など）、エンベロープが汚染されるのを防ぐ
    static constexpr double MAX_ENV_VALUE = 1000.0; // +60dB

    if (!std::isfinite(inputRMS) || inputRMS > MAX_ENV_VALUE)   inputRMS = MAX_ENV_VALUE;
    if (!std::isfinite(outputRMS) || outputRMS > MAX_ENV_VALUE) outputRMS = MAX_ENV_VALUE;

    // Load atomics
    double envIn = agcEnvInput.load(std::memory_order_relaxed);
    double envOut = agcEnvOutput.load(std::memory_order_relaxed);
    double currentGain = agcCurrentGain.load(std::memory_order_relaxed);

    if (!std::isfinite(envIn))  envIn = 0.0;
    if (!std::isfinite(envOut)) envOut = 0.0;
    if (!std::isfinite(currentGain)) currentGain = 1.0;

    // 指数移動平均 (EMA) によるエンベロープ検波
    envIn  = envIn  * (1.0 - AGC_ALPHA) + inputRMS  * AGC_ALPHA;
    envOut = envOut * (1.0 - AGC_ALPHA) + outputRMS * AGC_ALPHA;

    // Denormal対策: 極小値をゼロにクランプ (無音時のCPU負荷対策)
    if (envIn < 1.0e-20) envIn = 0.0;
    if (envOut < 1.0e-20) envOut = 0.0;

    // ターゲットゲイン計算
    double targetGain = calculateAGCGain(envIn, envOut);

    // ゲイン変化のスムーシング
    currentGain = currentGain * (1.0 - AGC_GAIN_SMOOTH) + targetGain * AGC_GAIN_SMOOTH;

    // Store atomics
    agcEnvInput.store(envIn, std::memory_order_relaxed);
    agcEnvOutput.store(envOut, std::memory_order_relaxed);
    agcCurrentGain.store(currentGain, std::memory_order_relaxed);

    // ゲイン適用
    // 各チャンネルに対して明示的に適用
    for (int ch = 0; ch < numChannels; ++ch)
    {
        juce::FloatVectorOperations::multiply(block.getChannelPointer(ch),
                                              currentGain, numSamples);
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
void EQProcessor::process(juce::dsp::AudioBlock<double>& block)
{
    juce::ScopedNoDenormals noDenormals;
    if (bypassed.load(std::memory_order_relaxed))
        return;

    const int numSamples = (int)block.getNumSamples();
    if (numSamples <= 0) return; // ゼロサンプルガード: 処理対象がない場合は即座に戻る

    const int numChannels = std::min((int)block.getNumChannels(), MAX_CHANNELS);

    const bool isAgcEnabled = agcEnabled.load(std::memory_order_acquire);
    // ✅ フィルタ処理前に入力レベルをキャッシュ (AGCが有効な場合のみ)
    if (isAgcEnabled)
    {
        cachedInputRMS = 0.0;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            double sumSq = 0.0;
            const double* data = block.getChannelPointer(ch);
            for (int i = 0; i < numSamples; ++i)
                sumSq += data[i] * data[i];
            double rms = std::sqrt(sumSq / static_cast<double>(numSamples));
            if (rms > cachedInputRMS)
                cachedInputRMS = rms;
        }
    }

    // ── 最適化: アクティブなバンドノードを事前にスタックへロード ──
    // チャンネルごとのループ内で atomic load を繰り返すと負荷が高いため、
    // 処理開始時に一度だけロードし、shared_ptr で寿命を確保する。
    struct ActiveBandNode {
        const BandNode* node;
        int index;
    };
    std::array<ActiveBandNode, NUM_BANDS> activeBands;
    int numActiveBands = 0;

    // 処理中にノードが削除されないよう shared_ptr で保持
    std::array<std::shared_ptr<BandNode>, NUM_BANDS> keptAliveNodes;

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto node = bandNodes[i].load(std::memory_order_acquire);
        if (node && node->active)
        {
            keptAliveNodes[numActiveBands] = node;
            activeBands[numActiveBands] = { node.get(), i };
            numActiveBands++;
        }
    }

    // フィルタバンク適用
    for (int ch = 0; ch < numChannels; ++ch)
    {
        double* data = block.getChannelPointer(ch);
        if (data == nullptr) continue;

        for (int i = 0; i < numActiveBands; ++i)
        {
            const auto& band = activeBands[i];
            const EQChannelMode mode = band.node->mode;

            if (mode == EQChannelMode::Stereo ||
               (mode == EQChannelMode::Left && ch == 0) ||
               (mode == EQChannelMode::Right && ch == 1))
            {
                processBand(data, numSamples, band.node->coeffs, filterState[ch][band.index].data());
            }
        }
    }

    // トータルゲイン / AGC 適用
    if (isAgcEnabled)
    {
        processAGC(block);
    }
    else
    {
        // ✅ Audio Threadでのみ setTargetValue() を呼ぶ
        const float targetDb = totalGainDbTarget.load(std::memory_order_relaxed);
        const double targetGain = juce::Decibels::decibelsToGain<double>(static_cast<double>(targetDb));

        if (std::abs(smoothTotalGain.getTargetValue() - targetGain) > 1e-6)
        {
            smoothTotalGain.setTargetValue(targetGain);
        }

        const double startGain = smoothTotalGain.getCurrentValue();
        smoothTotalGain.skip(numSamples);
        const double endGain = smoothTotalGain.getCurrentValue();

        for (int ch = 0; ch < numChannels; ++ch)
        {
            // Manual Gain Ramp
            double* data = block.getChannelPointer(ch);
            double gain = startGain;
            const double increment = (endGain - startGain) / static_cast<double>(numSamples);

            for (int i = 0; i < numSamples; ++i)
            {
                data[i] *= gain;
                gain += increment;
            }
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
        trashBin.push_back(std::move(oldNode));

    // ゴミ箱の掃除 (Garbage Collection)
    // Audio Threadが参照していない(use_count == 1)オブジェクトのみを削除する。
    trashBin.erase(std::remove_if(trashBin.begin(), trashBin.end(),
                                   [](const auto& p) { return p.use_count() == 1; }), trashBin.end()); // ラムダ構文修正
}

//--------------------------------------------------------------
// パラメータ検証とクランプ (Helper)
//--------------------------------------------------------------
void EQProcessor::validateAndClampParameters(float& freq, float& gainDb, float& q, double sr) noexcept
{
    // 周波数をナイキスト周波数以下にクランプ
    const float nyquist = static_cast<float>(sr * 0.5);
    freq = juce::jlimit(DSP_MIN_FREQ, nyquist * DSP_MAX_FREQ_NYQUIST_RATIO, freq);

    // Qを安全な範囲にクランプ
    q = juce::jlimit(DSP_MIN_Q, DSP_MAX_Q, q);

    // ゲインを実用範囲にクランプ
    gainDb = juce::jlimit(DSP_MIN_GAIN_DB, DSP_MAX_GAIN_DB, gainDb);
}

//--------------------------------------------------------------
// SVF係数計算 (Audio Thread用)
//--------------------------------------------------------------
EQCoeffsSVF EQProcessor::calcSVFCoeffs(EQBandType type, float freq, float gainDb, float q, double sr) noexcept
{
    // パラメータ検証 (Parameter Validation)
    // 不正な値から保護し、安全な範囲にクランプ
    if (sr <= 0.0)
    {
        jassertfalse;
        // 不正なサンプルレートでは計算不能なため、バイパス係数を返す
        EQCoeffsSVF c;
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }
    validateAndClampParameters(freq, gainDb, q, sr);

    const double f = static_cast<double>(freq);
    const double g = static_cast<double>(gainDb);
    const double Q = static_cast<double>(q);
    const double s = sr;

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
EQCoeffsBiquad EQProcessor::calcBiquadCoeffs(EQBandType type, float freq, float gainDb, float q, double sr) noexcept
{
    // パラメータ検証 (Parameter Validation)
    if (sr <= 0.0 || sr > 384000.0)
    {
        jassertfalse;
        sr = 48000.0;
    }

    validateAndClampParameters(freq, gainDb, q, sr);

    const double f = static_cast<double>(freq);
    const double g = static_cast<double>(gainDb);
    const double Q = static_cast<double>(q);
    const double s = sr;

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
// SVF実装
//--------------------------------------------------------------
EQCoeffsSVF EQProcessor::calcLowShelfSVF(double freq, double gainDb, double q, double sr) noexcept
{
    EQCoeffsSVF c;
    const double A = std::pow(10.0, gainDb / 40.0);
    const double g = std::tan(juce::MathConstants<double>::pi * freq / sr) / std::sqrt(A);
    const double k = 1.0 / q;

    // NaN/Infチェック: tan()が発散した場合など
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

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

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

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

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

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

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

    // 除算ゼロ保護
    const double denominator = 1.0 + g * (g + k);
    if (std::abs(denominator) < 1.0e-15)
    {
        c.a1 = 1.0;
        c.a2 = 0.0;
        c.a3 = 0.0;
        c.m0 = 1.0; // バイパス
        c.m1 = 0.0;
        c.m2 = 0.0;
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

    // NaN/Infチェック
    if (!std::isfinite(g) || !std::isfinite(k))
    {
        // デフォルト係数（バイパス状態）を返す
        c.a1 = 1.0; c.a2 = 0.0; c.a3 = 0.0;
        c.m0 = 1.0; c.m1 = 0.0; c.m2 = 0.0;
        return c;
    }

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
// Biquad実装 (Audio EQ Cookbook)
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
