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

#if JUCE_DSP_USE_INTEL_MKL
#include <mkl.h>
#endif

#if defined(__AVX2__) || defined(__FMA__)
 #include <immintrin.h>
#endif

//--------------------------------------------------------------
// コンストラクタ
//--------------------------------------------------------------
EQProcessor::EQProcessor()
{
    // 初期係数ノードの作成
    resetToDefaults();
}

//--------------------------------------------------------------
EQProcessor::~EQProcessor()
{
    // shared_ptrが自動的にリソースを管理するため、デストラクタは空で良い
    currentState.store(nullptr, std::memory_order_release);
    for (auto& node : bandNodes) {
        node.store(nullptr, std::memory_order_release);
    }
    // activeBandNodes will be cleared automatically
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
    auto oldState = currentState.exchange(newState, std::memory_order_release);
    if (oldState) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(oldState);
    }

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
        auto stdLine = line.toStdString();
        auto tokensBegin = std::sregex_iterator(stdLine.begin(), stdLine.end(), tokenRegex);
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
    auto state = currentState.load(std::memory_order_acquire);
    if (state == nullptr) return juce::ValueTree("EQ");

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
    // UIコンポーネント(other)からの同期はMessage Threadで行う必要がある
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // アトミック変数のコピー
    totalGainDbTarget.store(other.totalGainDbTarget.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // 共有状態のコピー
    auto otherState = other.currentState.load(std::memory_order_acquire);
    auto oldState = currentState.exchange(otherState, std::memory_order_release);

    // 安全性と整合性のためにtrashBinを使用
    const juce::ScopedLock sl(trashBinLock);

    if (oldState)
    {
        stateTrashBinPending.push_back(oldState);
    }

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto node = other.activeBandNodes[i];
        bandNodes[i].store(node.get(), std::memory_order_release);
        if (activeBandNodes[i]) {
            bandNodeTrashBinPending.push_back(activeBandNodes[i]);
        }
        activeBandNodes[i] = node;
    }
    agcEnabled.store(other.agcEnabled.load(std::memory_order_acquire), std::memory_order_release);
    agcCurrentGain.store(other.agcCurrentGain.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvInput.store(other.agcEnvInput.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvOutput.store(other.agcEnvOutput.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // 注意: smoothTotalGainのターゲットは、totalGainDbTargetに基づいてprocess()内で更新されます
}

void EQProcessor::syncBandNodeFrom(const EQProcessor& other, int bandIndex)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (bandIndex < 0 || bandIndex >= NUM_BANDS) return;

    auto node = other.activeBandNodes[bandIndex];

    bandNodes[bandIndex].store(node.get(), std::memory_order_release);

    if (activeBandNodes[bandIndex])
    {
        const juce::ScopedLock sl(trashBinLock);
        bandNodeTrashBinPending.push_back(activeBandNodes[bandIndex]);
    }
    activeBandNodes[bandIndex] = node;
}

void EQProcessor::syncGlobalStateFrom(const EQProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    totalGainDbTarget.store(other.totalGainDbTarget.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnabled.store(other.agcEnabled.load(std::memory_order_acquire), std::memory_order_release);
    // AGC状態の同期
    agcCurrentGain.store(other.agcCurrentGain.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvInput.store(other.agcEnvInput.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvOutput.store(other.agcEnvOutput.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

//--------------------------------------------------------------
// prepareToPlay
// サンプルレート変更時にフィルタ状態を全リセットする
// この関数は Audio Thread 開始前、または停止後に呼ばれる
//--------------------------------------------------------------
void EQProcessor::prepareToPlay(double sampleRate, int newMaxInternalBlockSize)
{
    const bool rateChanged = (std::abs(currentSampleRate - sampleRate) > 1e-6);
    if (rateChanged)
        currentSampleRate = sampleRate;

    // ==================================================================
    // 【Issue 4 完全修正】MKLスクラッチバッファ固定最大確保
    // 変更点:
    //   1. 引数名を newMaxInternalBlockSize に変更（C4458 shadowing警告解消）
    //   2. resizeを「不足時のみ」に制限 → RCU再構築時のメモリ操作を最小化
    //   3. 明示的clear（Denormal/NaN/Inf対策）
    //   4. maxInternalBlockSize を保存（process内ガード用）
    // ==================================================================
    const int requiredSize = newMaxInternalBlockSize;   // ← 引数からローカルへ

    if (scratchBuffer.size() < static_cast<size_t>(requiredSize))
    {
        scratchBuffer.resize(static_cast<size_t>(requiredSize));

        // 明示的ゼロクリア（Denormal防止 + 再現性確保）
        if (!scratchBuffer.empty())
            juce::FloatVectorOperations::clear(scratchBuffer.data(),
                                               static_cast<int>(scratchBuffer.size()));
    }

    this->maxInternalBlockSize = requiredSize;   // メンバ変数へ代入（this->で明示）

    auto state = currentState.load(std::memory_order_acquire);

    // reset()を呼び、スムーシングの状態を初期化（現在値は0.0になる）
    smoothTotalGain.reset(sampleRate, SMOOTHING_TIME_SEC);

    // 初期化直後のフェードイン（0.0 -> Target）を防ぐため、
    // 現在値をターゲット値に即座に設定する。
    if (state)
    {
        totalGainDbTarget.store(state->totalGainDb, std::memory_order_relaxed);
        smoothTotalGain.setCurrentAndTargetValue(
            juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
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
        {
            auto loopState = currentState.load(std::memory_order_acquire);
            if (loopState)
            {
                auto newNode = createBandNode(i, *loopState);
                bandNodes[i].store(newNode.get(), std::memory_order_release);
                activeBandNodes[i] = newNode;
            }
        }
    }
    // ==================================================================
    // 【スペアナグラフ統合曲線完全同期修正】
    // prepareToPlay（RCU再構築・プリセットロード・SR変更時）に必ず
    // ChangeBroadcaster通知を発行し、SpectrumAnalyzerComponentの
    // updateEQData() / updateEQPaths() を強制実行
    // これで統合曲線が実際の設定値（Gain/Q/Freq/Type）と完全に一致
    // ==================================================================
    sendChangeMessage();
}
//--------------------------------------------------------------
// パラメータ変更メソッド (UIスレッドから呼ぶ)
// 各メソッドは atomic store で値を書き込み、coeffsDirty を立てる
//--------------------------------------------------------------
void EQProcessor::setBandFrequency(int band, float freq)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].frequency = freq;

    auto prev = currentState.exchange(newState, std::memory_order_release);
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setBandGain(int band, float gainDb)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].gain = gainDb;

    auto prev = currentState.exchange(newState, std::memory_order_release);
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setBandQ(int band, float q)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].q = q;

    auto prev = currentState.exchange(newState, std::memory_order_release);
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setBandEnabled(int band, bool enabled)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bands[band].enabled = enabled;

    auto prev = currentState.exchange(newState, std::memory_order_release);
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
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
    if (oldState == nullptr) return;
    auto newState = std::make_shared<EQState>(*oldState);
    newState->totalGainDb = gainDb;

    auto prev = currentState.exchange(newState, std::memory_order_release);
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    listeners.call(&Listener::eqGlobalChanged, this);
}

float EQProcessor::getTotalGain() const
{
    auto state = currentState.load(std::memory_order_acquire);
    if (state == nullptr) return 0.0f;
    return state->totalGainDb;
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
    if (oldState == nullptr) return;
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bandTypes[band] = type;

    auto prev = currentState.exchange(newState, std::memory_order_release);
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

EQBandType EQProcessor::getBandType(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQBandType::Peaking;
    auto state = currentState.load(std::memory_order_acquire);
    if (state == nullptr) return EQBandType::Peaking;
    return state->bandTypes[band];
}

void EQProcessor::setBandChannelMode(int band, EQChannelMode mode)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentState.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = std::make_shared<EQState>(*oldState);
    newState->bandChannelModes[band] = mode;

    auto prev = currentState.exchange(newState, std::memory_order_release);
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

EQChannelMode EQProcessor::getBandChannelMode(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQChannelMode::Stereo;
    auto state = currentState.load(std::memory_order_acquire);
    if (state == nullptr) return EQChannelMode::Stereo;
    return state->bandChannelModes[band];
}

EQProcessor::EQState::Ptr EQProcessor::getEQState() const
{
    // Ptrコンストラクタが参照カウントをインクリメントしてくれる
    return EQState::Ptr(currentState.load(std::memory_order_acquire));
}

//--------------------------------------------------------------
// パラメータ読み取り (UIスレッド用)
//--------------------------------------------------------------
EQBandParams EQProcessor::getBandParams(int band) const
{
    // band が範囲外の場合はデフォルト値を返す
    if (band < 0 || band >= NUM_BANDS) return {};
    auto state = currentState.load(std::memory_order_acquire);
    if (state == nullptr) return {};
    return state->bands[band];
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

#if defined(__AVX2__) || defined(__FMA__)
    // ── 追加: Stereo 2ch 同時処理 (SSE2 / AVX2 FMA) ──
    // L, R が完全に独立した IIR 状態を持つため、128-bit レジスタに
    // [L_value, R_value] をパックして同時演算し、メモリ帯域を節約する。
    inline void processBandStereo(double* __restrict dataL,
                                   double* __restrict dataR,
                                   int numSamples,
                                   const EQCoeffsSVF& c,
                                   double* __restrict stateL,
                                   double* __restrict stateR) noexcept
    {
        // フィルタ状態を __m128d にパック: lower=L, upper=R
        __m128d ic1eq = _mm_set_pd(stateR[0], stateL[0]);
        __m128d ic2eq = _mm_set_pd(stateR[1], stateL[1]);

        const __m128d a1  = _mm_set1_pd(c.a1);
        const __m128d a2  = _mm_set1_pd(c.a2);
        const __m128d a3  = _mm_set1_pd(c.a3);
        const __m128d m0  = _mm_set1_pd(c.m0);
        const __m128d m1  = _mm_set1_pd(c.m1);
        const __m128d m2  = _mm_set1_pd(c.m2);
        const __m128d two = _mm_set1_pd(2.0);
        const __m128d cHigh = _mm_set1_pd(100.0);
        const __m128d cLow  = _mm_set1_pd(-100.0);

        constexpr double DENORMAL_THRESHOLD = 1.0e-15;

        for (int n = 0; n < numSamples; ++n)
        {
            // L[n] と R[n] を同時ロード
            const __m128d v0 = _mm_set_pd(dataR[n], dataL[n]);

            const __m128d v3 = _mm_sub_pd(v0, ic2eq);
#if defined(__AVX2__)
            // FMA: a1*ic1eq + a2*v3
            const __m128d v1 = _mm_fmadd_pd(a1, ic1eq, _mm_mul_pd(a2, v3));
            // FMA: ic2eq + a2*ic1eq + a3*v3
            const __m128d v2 = _mm_fmadd_pd(a2, ic1eq,
                                _mm_fmadd_pd(a3, v3, ic2eq));

            ic1eq = _mm_fmsub_pd(two, v1, ic1eq);  // 2*v1 - ic1eq
            ic2eq = _mm_fmsub_pd(two, v2, ic2eq);  // 2*v2 - ic2eq
            // FMA: m0*v0 + m1*v1 + m2*v2
            __m128d output = _mm_fmadd_pd(m0, v0,
                              _mm_fmadd_pd(m1, v1,
                               _mm_mul_pd(m2, v2)));
#else
            const __m128d v1 = _mm_add_pd(_mm_mul_pd(a1, ic1eq), _mm_mul_pd(a2, v3));
            const __m128d v2 = _mm_add_pd(ic2eq, _mm_add_pd(_mm_mul_pd(a2, ic1eq), _mm_mul_pd(a3, v3)));
            ic1eq = _mm_sub_pd(_mm_mul_pd(two, v1), ic1eq);
            ic2eq = _mm_sub_pd(_mm_mul_pd(two, v2), ic2eq);
            __m128d output = _mm_add_pd(_mm_mul_pd(m0, v0), _mm_add_pd(_mm_mul_pd(m1, v1), _mm_mul_pd(m2, v2)));
#endif

            // NaN/Infチェック (isfinite): (x - x) は xがInf/NaNの時NaNになる
            const __m128d diff = _mm_sub_pd(output, output);
            const __m128d mask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
            output = _mm_and_pd(output, mask);

            // クランプ (-100, +100) で発散防止
            output = _mm_min_pd(_mm_max_pd(output, cLow), cHigh);

            // L: lower element, R: upper element（修正：unpackhi_pdの第2引数を正しく）
            dataL[n] = _mm_cvtsd_f64(output);
            __m128d hi = _mm_unpackhi_pd(output, output);
            dataR[n] = _mm_cvtsd_f64(hi);
        }

        // Denormal フラッシュ & NaN チェック (状態変数のみ) - SIMD最適化
        const __m128d denormal_threshold = _mm_set1_pd(DENORMAL_THRESHOLD);
        const __m128d sign_mask = _mm_set1_pd(-0.0);

        // --- ic1eq ---
        __m128d nan_mask1 = _mm_cmpeq_pd(ic1eq, ic1eq);
        __m128d abs_ic1eq = _mm_andnot_pd(sign_mask, ic1eq);
        __m128d denormal_mask1 = _mm_cmplt_pd(abs_ic1eq, denormal_threshold);
        __m128d valid_mask1 = _mm_andnot_pd(denormal_mask1, nan_mask1);
        ic1eq = _mm_and_pd(ic1eq, valid_mask1);

        // --- ic2eq ---
        __m128d nan_mask2 = _mm_cmpeq_pd(ic2eq, ic2eq);
        __m128d abs_ic2eq = _mm_andnot_pd(sign_mask, ic2eq);
        __m128d denormal_mask2 = _mm_cmplt_pd(abs_ic2eq, denormal_threshold);
        __m128d valid_mask2 = _mm_andnot_pd(denormal_mask2, nan_mask2);
        ic2eq = _mm_and_pd(ic2eq, valid_mask2);

        // 状態を書き戻す (L/Rチャンネルに分離)
        _mm_storeu_pd(stateL, _mm_unpacklo_pd(ic1eq, ic2eq)); // [ic1eq_L, ic2eq_L]
        _mm_storeu_pd(stateR, _mm_unpackhi_pd(ic1eq, ic2eq)); // [ic1eq_R, ic2eq_R]
    }
#endif

    // ── 追加: AVX2 Gain Ramp ──
    inline void applyGainRamp_AVX2(double* __restrict data, int numSamples,
                                     double startGain, double increment) noexcept
    {
#if defined(__AVX2__)
        // 各レーンの初期ゲイン: [g0, g0+inc, g0+2*inc, g0+3*inc]
        __m256d vGain = _mm256_set_pd(startGain + 3.0 * increment,
                                       startGain + 2.0 * increment,
                                       startGain + increment,
                                       startGain);
        const __m256d vInc4 = _mm256_set1_pd(4.0 * increment);
        const __m256d vInc16 = _mm256_set1_pd(16.0 * increment);

        int i = 0;
        const int vEnd = numSamples / 16 * 16;
        for (; i < vEnd; i += 16)
        {
            _mm_prefetch(reinterpret_cast<const char*>(data + i + 64), _MM_HINT_T0);

            // 1
            __m256d vData0 = _mm256_load_pd(data + i);
            __m256d vOut0  = _mm256_mul_pd(vData0, vGain);
            _mm256_store_pd(data + i, vOut0);
            vGain = _mm256_add_pd(vGain, vInc4);

            // 2
            __m256d vData1 = _mm256_load_pd(data + i + 4);
            __m256d vOut1  = _mm256_mul_pd(vData1, vGain);
            _mm256_store_pd(data + i + 4, vOut1);
            vGain = _mm256_add_pd(vGain, vInc4);

            // 3
            __m256d vData2 = _mm256_load_pd(data + i + 8);
            __m256d vOut2  = _mm256_mul_pd(vData2, vGain);
            _mm256_store_pd(data + i + 8, vOut2);
            vGain = _mm256_add_pd(vGain, vInc4);

            // 4
            __m256d vData3 = _mm256_load_pd(data + i + 12);
            __m256d vOut3  = _mm256_mul_pd(vData3, vGain);
            _mm256_store_pd(data + i + 12, vOut3);
            vGain = _mm256_add_pd(vGain, vInc4);
        }
        // Remaining
        for (; i < (numSamples / 4 * 4); i += 4)
        {
            __m256d vData = _mm256_load_pd(data + i);
            __m256d vOut  = _mm256_mul_pd(vData, vGain);
            _mm256_store_pd(data + i, vOut);
            vGain = _mm256_add_pd(vGain, vInc4);
        }
        // スカラー残余
        double gain = startGain + static_cast<double>(i) * increment;
        for (; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
#else
        double gain = startGain;
        for (int i = 0; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
#endif
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
        const double* data = block.getChannelPointer(ch);
        double rms = 0.0;

#if JUCE_DSP_USE_INTEL_MKL
            // MKL最適化されたRMS計算 (Norm / sqrt(N))
        std::memcpy(scratchBuffer.data(), data, static_cast<size_t>(numSamples) * sizeof(double));
        double norm = cblas_dnrm2(numSamples, scratchBuffer.data(), 1);
        rms = norm / std::sqrt(static_cast<double>(numSamples));
#else
        double sumSq = 0.0;
        for (int i = 0; i < numSamples; ++i)
            sumSq += data[i] * data[i];
        rms = std::sqrt(sumSq / static_cast<double>(numSamples));
#endif

        if (rms > outputRMS) outputRMS = rms;
    }

    // 数値安定性対策: NaN/Infチェックとクランプ
    // 入力が極端に大きい場合（発振など）、エンベロープが汚染されるのを防ぐ
    static constexpr double MAX_ENV_VALUE = 1000.0; // +60dB

    if (!std::isfinite(inputRMS) || inputRMS > MAX_ENV_VALUE)   inputRMS = MAX_ENV_VALUE;
    if (!std::isfinite(outputRMS) || outputRMS > MAX_ENV_VALUE) outputRMS = MAX_ENV_VALUE;

    // アトミック変数のロード
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
    double nextGain = currentGain * (1.0 - AGC_GAIN_SMOOTH) + targetGain * AGC_GAIN_SMOOTH;

    // アトミック変数のストア
    agcEnvInput.store(envIn, std::memory_order_relaxed);
    agcEnvOutput.store(envOut, std::memory_order_relaxed);
    agcCurrentGain.store(nextGain, std::memory_order_relaxed);

    // ゲイン適用 (ランプ: currentGain -> nextGain)
    // ブロック境界での不連続性を防ぐため、サンプル単位で補間する
    const double gainIncrement = (nextGain - currentGain) / static_cast<double>(numSamples);

    for (int ch = 0; ch < numChannels; ++ch)
    {
        double* data = block.getChannelPointer(ch);
        double g = currentGain;

        for (int i = 0; i < numSamples; ++i)
        {
            data[i] *= g;
            g += gainIncrement;
        }
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

    // ==================================================================
    // 【Issue 4 追加安全ガード】オーバーラン即検出
    // ==================================================================
    // Fix: Releaseビルドでも有効なチェックに変更
    if (numSamples <= 0 || static_cast<size_t>(numSamples) > static_cast<size_t>(maxInternalBlockSize))
    {
        jassert(numSamples > 0 && static_cast<size_t>(numSamples) <= static_cast<size_t>(maxInternalBlockSize));
        return;
    }

    const int numChannels = std::min((int)block.getNumChannels(), MAX_CHANNELS);

    const bool isAgcEnabled = agcEnabled.load(std::memory_order_acquire);
    // ✅ フィルタ処理前に入力レベルをキャッシュ (AGCが有効な場合のみ)
    if (isAgcEnabled)
    {
        cachedInputRMS = 0.0;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* data = block.getChannelPointer(ch);
            double rms = 0.0;

#if JUCE_DSP_USE_INTEL_MKL
            // MKL最適化されたRMS計算
            std::memcpy(scratchBuffer.data(), data, static_cast<size_t>(numSamples) * sizeof(double));
            double norm = cblas_dnrm2(numSamples, scratchBuffer.data(), 1);
            rms = norm / std::sqrt(static_cast<double>(numSamples));
#else
            double sumSq = 0.0;
            for (int i = 0; i < numSamples; ++i)
                sumSq += data[i] * data[i];
            rms = std::sqrt(sumSq / static_cast<double>(numSamples));
#endif

            if (rms > cachedInputRMS)
                cachedInputRMS = rms;
        }
    }

    // ── 最適化: アクティブなバンドノードを事前にスタックへロード ──
    // チャンネルごとのループ内で atomic load を繰り返すと負荷が高いため、
    // 処理開始時に一度だけロードする。
    // Note: 寿命管理は Message Thread 側の trashBin (時間差削除) により保証されるため、
    // ここでは Raw Pointer を安全に使用できる。
    struct ActiveBandNode {
        const BandNode* node;
        int index;
    };
    std::array<ActiveBandNode, NUM_BANDS> activeBands;
    int numActiveBands = 0;

    // Note: bandNodes[] contains raw pointers. We assume they are valid during this block
    // because deletion is deferred by trashBin in Message Thread.
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto* node = bandNodes[i].load(std::memory_order_acquire);
        if (node && node->active)
        {
            activeBands[numActiveBands] = { node, i };
            numActiveBands++;
        }
    }

    // フィルタバンク適用
#if defined(__AVX2__) || defined(__FMA__)
    for (int i = 0; i < numActiveBands; ++i)
    {
        const auto& band = activeBands[i];
        const EQChannelMode mode = band.node->mode;

        if (mode == EQChannelMode::Stereo && numChannels >= 2)
        {
            // L と R を SSE2 レジスタで同時処理 (最大2x スループット)
            processBandStereo(
                block.getChannelPointer(0),
                block.getChannelPointer(1),
                numSamples,
                band.node->coeffs,
                filterState[0][band.index].data(),
                filterState[1][band.index].data());
        }
        else
        {
            if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Left)
                if (numChannels > 0)
                    processBand(block.getChannelPointer(0), numSamples,
                                band.node->coeffs, filterState[0][band.index].data());
            if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Right)
                if (numChannels > 1)
                    processBand(block.getChannelPointer(1), numSamples,
                                band.node->coeffs, filterState[1][band.index].data());
        }
    }
#else
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
#endif

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

        // AGC 無効時のゲインランプ
        const double increment = (endGain - startGain) / static_cast<double>(numSamples);
        for (int ch = 0; ch < numChannels; ++ch)
            applyGainRamp_AVX2(block.getChannelPointer(ch), numSamples, startGain, increment);
    }
}


//--------------------------------------------------------------
// BandNode作成 (Message Thread)
//--------------------------------------------------------------
EQProcessor::BandNode::Ptr EQProcessor::createBandNode(int band, const EQState& state) const
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
    if (state == nullptr) return;
    auto newNode = createBandNode(band, *state);

    bandNodes[band].store(newNode.get(), std::memory_order_release);

    // 古いノードをゴミ箱へ (Audio Threadが使用中の可能性があるため即削除しない)
    const juce::ScopedLock sl(trashBinLock);
    if (activeBandNodes[band])
    {
        bandNodeTrashBinPending.push_back(activeBandNodes[band]);
    }
    activeBandNodes[band] = newNode;
}

void EQProcessor::cleanup()
{
    std::vector<BandNode::Ptr> nodesToDelete;
    std::vector<EQState::Ptr> statesToDelete;

    {
        const juce::ScopedLock sl(trashBinLock);
        const uint32 now = juce::Time::getMillisecondCounter();

        // Clean BandNodes (Time-based)
        // Audio ThreadがRaw Pointerを参照している可能性があるため、即時削除せず一定時間(2000ms)待つ
        auto nodeIt = std::remove_if(bandNodeTrashBin.begin(), bandNodeTrashBin.end(),
            [now](const auto& entry) {
                return (now >= entry.second) ? (now - entry.second > 2000)
                                             : (now + (std::numeric_limits<uint32>::max() - entry.second) > 2000);
            });

        for (auto it = nodeIt; it != bandNodeTrashBin.end(); ++it)
            nodesToDelete.push_back(it->first);
        bandNodeTrashBin.erase(nodeIt, bandNodeTrashBin.end());

        // Move pending to main trash with timestamp
        for (const auto& ptr : bandNodeTrashBinPending)
            bandNodeTrashBin.push_back({ptr, now});
        bandNodeTrashBinPending.clear();

        // Clean States
        auto stateIt = std::remove_if(stateTrashBin.begin(), stateTrashBin.end(), [](const auto& p) { return p.use_count() == 1; });
        statesToDelete.insert(statesToDelete.end(), std::make_move_iterator(stateIt), std::make_move_iterator(stateTrashBin.end()));
        stateTrashBin.erase(stateIt, stateTrashBin.end());

        stateTrashBin.insert(stateTrashBin.end(), stateTrashBinPending.begin(), stateTrashBinPending.end());
        stateTrashBinPending.clear();
    }

    nodesToDelete.clear();
    statesToDelete.clear();
}

// --------------------------------------------------------------
// パラメータ検証とクランプ (Helper)
//--------------------------------------------------------------
void EQProcessor::validateAndClampParameters(float& freq, float& gainDb, float& q, double sr) noexcept
{
    // 周波数をナイキスト周波数以下にクランプ
    const float nyquist = static_cast<float>(sr * 0.5);
    const float maxFreq = std::min(DSP_MAX_FREQ, nyquist * DSP_MAX_FREQ_NYQUIST_RATIO);
    freq = juce::jlimit(DSP_MIN_FREQ, maxFreq, freq);

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

    double denNorm = std::norm(den); // normはマグニチュードの二乗を返す
    if (denNorm < 1e-18) return 0.0f;

    return static_cast<float>(std::norm(num) / denNorm);
}

//--------------------------------------------------------------
// SVF係数から等価 Biquad 係数を計算 (UI表示用)
//--------------------------------------------------------------
EQCoeffsBiquad EQProcessor::svfToDisplayBiquad(const EQCoeffsSVF& svf) noexcept
{
    EQCoeffsBiquad bq;
    const double a1 = svf.a1, a2 = svf.a2, a3 = svf.a3;
    const double m0 = svf.m0, m1 = svf.m1, m2 = svf.m2;

    if (a1 < 1e-15) { bq.b0 = 1.0; bq.a0 = 1.0; return bq; }

    const double g2  = a3 / a1;
    const double g   = a2 / a1;
    const double gk  = (1.0 - a1 - a3) / a1;

    bq.a0 =  1.0 + gk + g2;
    bq.a1 = -2.0 + 2.0 * g2;
    bq.a2 =  1.0 - gk + g2;

    bq.b0 = m0 * (1.0 + gk + g2) + m1 * g + m2 * g2;
    bq.b1 = -2.0 * m0 + 2.0 * (m0 + m2) * g2;
    bq.b2 = m0 * (1.0 - gk + g2) - m1 * g + m2 * g2;

    return bq;
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

// --------------------------------------------------------------
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
