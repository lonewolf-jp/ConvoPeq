//============================================================================
// EQProcessor.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// 20バンドパラメトリックイコライザー処理実装
// 参照: Vadim Zavalishin "The Art of VA Filter Design" (TPT SVF)
//       https://www.w3.org/2011/audio/audio-eq-cookbook.html (Biquad Coeffs)
//============================================================================
#include "EQProcessor.h"
#include "DspNumericPolicy.h"
#include <cmath>
#include <cstdint>
#include <algorithm>
#include <complex>
#include <numeric>
#include <cstring>
#include <regex>

#if defined(__AVX2__) || defined(__FMA__)
 #include <immintrin.h>
#endif

static inline double calculateRMS(const double* data, int numSamples) noexcept
{
    if (data == nullptr || numSamples <= 0)
        return 0.0;

    // cblas_dnrm2 computes the Euclidean norm (L2 norm) of a vector:
    // sqrt(x[0]^2 + x[1]^2 + ... + x[n-1]^2)
    const double norm = cblas_dnrm2(numSamples, data, 1);

    // RMS = norm / sqrt(n)
    // Audio Thread内でのlibm呼び出しを避けるため、SSE2命令で平方根を計算する
    __m128d n = _mm_set_sd(static_cast<double>(numSamples));
    __m128d sqrtN = _mm_sqrt_sd(n, n);
    double denom;
    _mm_store_sd(&denom, sqrtN);
    return (denom > 0.0) ? (norm / denom) : 0.0;
}

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
    forceCleanup();
    // shared_ptrが自動的にリソースを管理するため、デストラクタは空で良い
    currentStateRaw.store(nullptr, std::memory_order_release);
    if (activeState) { activeState->release(); activeState = nullptr; }
    for (auto& node : bandNodes) {
        node.store(nullptr, std::memory_order_release);
    }
    // activeBandNodes will be cleared automatically
    for (auto* node : activeBandNodes) if (node) node->release();

    releaseResources();
}

void EQProcessor::releaseResources()
{
    forceCleanup();
    scratchBuffer.reset();
    scratchCapacity = 0;
}

//--------------------------------------------------------------
// デフォルト値リセット
//--------------------------------------------------------------
void EQProcessor::resetToDefaults()
{
    auto newState = new EQState();
    newState->addRef();

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

    storeTotalGainDb(0.0f);
    agcEnabled.store(false, std::memory_order_release);

    auto oldState = activeState;
    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);

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

    // 全状態のリセットを予約
    bandResetMask.store(0xFFFFFFFF, std::memory_order_relaxed);
    agcResetRequest.store(true, std::memory_order_relaxed);

    sendChangeMessage();
}

void EQProcessor::reset()
{
    // フィルタ状態をリセット (memsetで高速化)
    std::memset(filterState.data(), 0, sizeof(filterState));

    agcCurrentGain.store(1.0, std::memory_order_relaxed);
    agcEnvInput.store(0.0, std::memory_order_relaxed);
    agcEnvOutput.store(0.0, std::memory_order_relaxed);

    auto state = currentStateRaw.load(std::memory_order_acquire);
    if (state)
    {
        smoothTotalGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
        storeTotalGainDb(state->totalGainDb);
    }

    bandResetMask.store(0, std::memory_order_relaxed);
    agcResetRequest.store(false, std::memory_order_relaxed);

    const bool requestedBypass = bypassRequested.load(std::memory_order_relaxed);
    bypassed.store(requestedBypass, std::memory_order_relaxed);
    bypassFadeGain.setCurrentAndTargetValue(requestedBypass ? 0.0 : 1.0);
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

    // 最初に全バンドをリセット (無効化 + パラメータ初期化)
    // これにより、ファイルに記述されていないバンドが前の設定(Channel Mode等)を引き継ぐのを防ぐ
    for (int i = 0; i < NUM_BANDS; ++i)
    {
        setBandEnabled(i, false);
        setBandChannelMode(i, EQChannelMode::Stereo);
        setBandGain(i, 0.0f);
    }

    juce::StringArray lines;
    file.readLines(lines);

    int currentFilterIndex = 0;
    bool maxBandsWarningShown = false;
    EQChannelMode currentChannelMode = EQChannelMode::Stereo;

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
        // Channel行の解析 (例: "Channel: L", "Channel: R", "Channel: L R")
        else if (tokens[0].startsWithIgnoreCase("Channel"))
        {
            bool hasL = false;
            bool hasR = false;

            for (const auto& token : tokens)
            {
                juce::String t = token;
                if (t.startsWithIgnoreCase("Channel"))
                    t = t.substring(7); // "Channel" length

                t = t.removeCharacters(":,");

                if (t.equalsIgnoreCase("L") || t.equalsIgnoreCase("Left"))
                    hasL = true;
                else if (t.equalsIgnoreCase("R") || t.equalsIgnoreCase("Right"))
                    hasR = true;
            }

            if (hasL && hasR)      currentChannelMode = EQChannelMode::Stereo;
            else if (hasL)         currentChannelMode = EQChannelMode::Left;
            else if (hasR)         currentChannelMode = EQChannelMode::Right;
            else                   currentChannelMode = EQChannelMode::Stereo;
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
                    const auto showMaxBandsWarning = [] {
                        juce::NativeMessageBox::showAsync(
                            juce::MessageBoxOptions()
                                .withIconType(juce::MessageBoxIconType::WarningIcon)
                                .withTitle("Load Preset Warning")
                                .withMessage("The preset contains more bands than supported (Max 20). Extra bands were ignored.")
                                .withButton("OK"),
                            nullptr);
                    };

                    const bool queued = juce::MessageManager::callAsync(showMaxBandsWarning);
                    if (!queued)
                    {
                        juce::MessageManagerLock mmLock;
                        if (mmLock.lockWasGained())
                            showMaxBandsWarning();
                    }
                }
                DBG("Skipping extra band: " + line);
                continue;
            }

            // デフォルト値
            bool enabled = true;
            bool typeFound = false;
            float freq = 0.0f;
            float gain = 0.0f;
            float q = DEFAULT_Q;
            bool qFound = false;

            // トークン解析 (順序非依存)
            for (int i = 1; i < tokens.size(); ++i)
            {
                const juce::String& t = tokens[i];

                // ON/OFF
                if (t.equalsIgnoreCase("ON")) {
                    enabled = true;
                    continue;
                }
                if (t.equalsIgnoreCase("OFF")) {
                    enabled = false;
                    continue;
                }

                // Filter Type
                if (!typeFound)
                {
                    if (t.equalsIgnoreCase("LSC") || t.equalsIgnoreCase("LowShelf")) {
                        setBandType(currentFilterIndex, EQBandType::LowShelf);
                        typeFound = true;
                        continue;
                    }
                    if (t.equalsIgnoreCase("PK") || t.equalsIgnoreCase("Peaking")) {
                        setBandType(currentFilterIndex, EQBandType::Peaking);
                        typeFound = true;
                        continue;
                    }
                    if (t.equalsIgnoreCase("HSC") || t.equalsIgnoreCase("HighShelf")) {
                        setBandType(currentFilterIndex, EQBandType::HighShelf);
                        typeFound = true;
                        continue;
                    }
                    if (t.equalsIgnoreCase("LP") || t.equalsIgnoreCase("LowPass")) {
                        setBandType(currentFilterIndex, EQBandType::LowPass);
                        typeFound = true;
                        continue;
                    }
                    if (t.equalsIgnoreCase("HP") || t.equalsIgnoreCase("HighPass")) {
                        setBandType(currentFilterIndex, EQBandType::HighPass);
                        typeFound = true;
                        continue;
                    }
                }

                // Parameters
                if (i + 1 < tokens.size())
                {
                    if (t.equalsIgnoreCase("Fc")) {
                        freq = tokens[i + 1].getFloatValue();
                    }
                    else if (t.equalsIgnoreCase("Gain")) {
                        gain = tokens[i + 1].getFloatValue();
                    }
                    else if (t.equalsIgnoreCase("Q")) {
                        q = tokens[i + 1].getFloatValue();
                        qFound = true;
                    }
                }
            }

            // タイプが見つからなかった場合のデフォルト (Peaking)
            if (!typeFound)
                setBandType(currentFilterIndex, EQBandType::Peaking);

            setBandEnabled(currentFilterIndex, enabled);

            if (freq > 0.0f)
                setBandFrequency(currentFilterIndex, freq);

            setBandGain(currentFilterIndex, gain);

            if (qFound && q > 0.0f)
                setBandQ(currentFilterIndex, q);
            else
                setBandQ(currentFilterIndex, DEFAULT_Q);

            // チャンネルモード設定
            setBandChannelMode(currentFilterIndex, currentChannelMode);

            currentFilterIndex++;
        }
    }

    // プリセットロード時は全状態をリセットしてクリーンな状態にする
    bandResetMask.store(0xFFFFFFFF, std::memory_order_relaxed);

    sendChangeMessage();
    return true;
}

//--------------------------------------------------------------
// 状態管理
//--------------------------------------------------------------
juce::ValueTree EQProcessor::getState() const
{
    auto state = activeState;
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

    // 状態ロード時は全リセット
    bandResetMask.store(0xFFFFFFFF, std::memory_order_relaxed);
    agcResetRequest.store(true, std::memory_order_relaxed);
    sendChangeMessage();
}

void EQProcessor::syncStateFrom(const EQProcessor& other)
{
    // UIコンポーネント(other)からの同期はMessage Threadで行う必要がある
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    // アトミック変数のコピー
    storeTotalGainDb(other.totalGainDbTarget.load(std::memory_order_relaxed));

    // 共有状態のコピー
    auto otherState = other.activeState;
    auto oldState = activeState;
    activeState = otherState;
    if (activeState) activeState->addRef();
    currentStateRaw.store(activeState, std::memory_order_release);

    // 安全性と整合性のためにtrashBinを使用
    const juce::ScopedLock sl(trashBinLock); // This lock is for the trash bin, not the state itself

    if (oldState)
    {
        stateTrashBinPending.push_back(oldState);
    }

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto node = other.activeBandNodes[i];
        if (node) node->addRef();
        bandNodes[i].store(node, std::memory_order_release);
        if (activeBandNodes[i]) {
            bandNodeTrashBinPending.push_back(activeBandNodes[i]);
        }
        activeBandNodes[i] = node;
    }
    agcEnabled.store(other.agcEnabled.load(std::memory_order_acquire), std::memory_order_release);
    agcCurrentGain.store(other.agcCurrentGain.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvInput.store(other.agcEnvInput.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvOutput.store(other.agcEnvOutput.load(std::memory_order_relaxed), std::memory_order_relaxed);

    // 注意: smoothTotalGainのターゲットは、totalGainTarget(linear値)に基づいてprocess()内で更新されます。
    // totalGainTargetはtotalGainDbTargetから storeTotalGainDb() を通じて同期されます。
}

void EQProcessor::syncBandNodeFrom(const EQProcessor& other, int bandIndex)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (bandIndex < 0 || bandIndex >= NUM_BANDS) return;

    auto node = other.activeBandNodes[bandIndex];

    if (node) node->addRef();
    bandNodes[bandIndex].store(node, std::memory_order_release);

    if (activeBandNodes[bandIndex])
    {
        const juce::ScopedLock sl(trashBinLock);
        bandNodeTrashBinPending.push_back(activeBandNodes[bandIndex]);
    }
    activeBandNodes[bandIndex] = node;
}

void EQProcessor::syncGlobalStateFrom(const EQProcessor& other)
{
    // Note: This method can be called from the rebuild thread (Worker Thread),
    // so we cannot assert isThisTheMessageThread(). The operations are atomic and thread-safe.

    storeTotalGainDb(other.totalGainDbTarget.load(std::memory_order_relaxed));
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
    // 【修正】atomic load して比較
    const bool rateChanged = (std::abs(currentSampleRate.load(std::memory_order_relaxed) - sampleRate) > 1e-6);
    if (rateChanged)
        // 【修正】atomic store して書き込み
        currentSampleRate.store(sampleRate, std::memory_order_relaxed);

    // process() 内のバッファガードに使用
    const int requiredSize = newMaxInternalBlockSize;
    this->maxInternalBlockSize = requiredSize;

    const int required = juce::nextPowerOfTwo(requiredSize) * 8; // MKL/EQ用目安
    if (scratchCapacity < required)
    {
        scratchBuffer.reset(static_cast<double*>(convo::aligned_malloc(required * sizeof(double), 64)));
        scratchCapacity = required;
        juce::FloatVectorOperations::clear(scratchBuffer.get(), required); // 念のためゼロクリア
    }
    // This can be called from a worker thread, so use the raw pointer.
    auto state = currentStateRaw.load(std::memory_order_acquire);

    // reset()を呼び、スムーシングの状態を初期化（現在値は0.0になる）
    smoothTotalGain.reset(sampleRate, SMOOTHING_TIME_SEC);
    bypassFadeGain.reset(sampleRate, BYPASS_FADE_TIME_SEC);

    // 初期化直後のフェードイン（0.0 -> Target）を防ぐため、
    // 現在値をターゲット値に即座に設定する。
    if (state)
    {
        storeTotalGainDb(state->totalGainDb);
        smoothTotalGain.setCurrentAndTargetValue(
            juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
    }

    // フィルタ状態をリセット (memsetで高速化)
    std::memset(filterState.data(), 0, sizeof(filterState));

    agcCurrentGain.store(1.0, std::memory_order_relaxed);
    agcEnvInput.store(0.0, std::memory_order_relaxed);
    agcEnvOutput.store(0.0, std::memory_order_relaxed);

    // フラグもクリア
    bandResetMask.store(0, std::memory_order_relaxed);
    agcResetRequest.store(false, std::memory_order_relaxed);

    const bool requestedBypass = bypassRequested.load(std::memory_order_relaxed);
    bypassed.store(requestedBypass, std::memory_order_relaxed);
    bypassFadeGain.setCurrentAndTargetValue(requestedBypass ? 0.0 : 1.0);

    // 係数を即座に再計算 (レート変更時のみ)
    if (rateChanged)
    {
        for (int i = 0; i < NUM_BANDS; ++i)
        {
            auto loopState = currentStateRaw.load(std::memory_order_acquire);
            if (loopState)
            {
                auto newNode = createBandNode(i, *loopState);
                bandNodes[i].store(newNode, std::memory_order_release);

                const juce::ScopedLock sl(trashBinLock);
                if (activeBandNodes[i])
                    bandNodeTrashBinPending.push_back(activeBandNodes[i]);

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
    auto oldState = activeState;
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->addRef();
    newState->bands[band].frequency = freq;

    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);
    auto prev = oldState;
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
    auto oldState = activeState;
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->addRef();
    newState->bands[band].gain = gainDb;

    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);
    auto prev = oldState;
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
    auto oldState = activeState;
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->addRef();
    newState->bands[band].q = q;

    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);
    auto prev = oldState;
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
    auto oldState = activeState;
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->addRef();
    newState->bands[band].enabled = enabled;

    // 有効化時はステートをリセットして、古い状態（ポップノイズの原因）を排除する
    if (enabled)
        bandResetMask.fetch_or(1 << band, std::memory_order_relaxed);

    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);
    auto prev = oldState;
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
    storeTotalGainDb(gainDb);

    auto oldState = activeState;
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->addRef();
    newState->totalGainDb = gainDb;

    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);
    auto prev = oldState;
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    listeners.call(&Listener::eqGlobalChanged, this);
}

void EQProcessor::setAGCEnabled(bool enabled)
{
    agcEnabled.store(enabled, std::memory_order_release);
    if (enabled)
        agcResetRequest.store(true, std::memory_order_relaxed);
    listeners.call(&Listener::eqGlobalChanged, this);
}

bool EQProcessor::getAGCEnabled() const
{
    return agcEnabled.load(std::memory_order_acquire);
}

void EQProcessor::setBandType(int band, EQBandType type)
{
    if (band < 0 || band >= NUM_BANDS) return;

    auto oldState = activeState;
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->addRef();
    newState->bandTypes[band] = type;

    // フィルタタイプ変更時はトポロジーが変わるためリセット必須
    bandResetMask.fetch_or(1 << band, std::memory_order_relaxed);

    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);
    auto prev = oldState;
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

void EQProcessor::setBandChannelMode(int band, EQChannelMode mode)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = activeState;
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->addRef();
    newState->bandChannelModes[band] = mode;

    // チャンネルモード変更時もリセット推奨
    bandResetMask.fetch_or(1 << band, std::memory_order_relaxed);

    activeState = newState;
    currentStateRaw.store(newState, std::memory_order_release);
    auto prev = oldState;
    if (prev) {
        const juce::ScopedLock sl(trashBinLock);
        stateTrashBinPending.push_back(prev);
    }
    updateBandNode(band);
    listeners.call(&Listener::eqBandChanged, this, band);
}

EQBandParams EQProcessor::getBandParams(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return {};
    auto state = activeState;
    if (state == nullptr) return {};
    return state->bands[band];
}

EQProcessor::EQState* EQProcessor::getEQState() const
{
    return activeState;
}

float EQProcessor::getTotalGain() const
{
    auto state = activeState;
    if (state == nullptr) return 0.0f;
    return state->totalGainDb;
}

EQBandType EQProcessor::getBandType(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQBandType::Peaking;
    auto state = activeState;
    if (state == nullptr) return EQBandType::Peaking;
    return state->bandTypes[band];
}

EQChannelMode EQProcessor::getBandChannelMode(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQChannelMode::Stereo;
    auto state = activeState;
    if (state == nullptr) return EQChannelMode::Stereo;
    return state->bandChannelModes[band];
}

namespace
{
    inline double absNoLibm(double value) noexcept
    {
        union { double d; uint64_t u; } v { value };
        v.u &= 0x7FFFFFFFFFFFFFFFULL;
        return v.d;
    }

    inline bool isFiniteNoLibm(double value) noexcept
    {
        const __m128d v = _mm_set1_pd(value);
        const __m128d diff = _mm_sub_pd(v, v);
        const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
        return _mm_movemask_pd(finiteMask) == 0x3;
    }

    inline bool isFiniteAndAbsInRangeMask(double value, double minAbsInclusive, double maxAbsExclusive) noexcept
    {
        const __m128d v = _mm_set1_pd(value);
        const __m128d diff = _mm_sub_pd(v, v);
        const __m128d finiteMask = _mm_cmpeq_pd(diff, _mm_setzero_pd());

        const __m128d signMask = _mm_set1_pd(-0.0);
        const __m128d absV = _mm_andnot_pd(signMask, v);
        const __m128d minV = _mm_set1_pd(minAbsInclusive);
        const __m128d maxV = _mm_set1_pd(maxAbsExclusive);
        const __m128d geMinMask = _mm_cmpge_pd(absV, minV);
        const __m128d ltMaxMask = _mm_cmplt_pd(absV, maxV);

        const __m128d validMask = _mm_and_pd(finiteMask, _mm_and_pd(geMinMask, ltMaxMask));
        return _mm_movemask_pd(validMask) == 0x3;
    }

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

        constexpr double DENORMAL_THRESHOLD = convo::numeric_policy::kDenormThresholdAudioState;

        for (int n = 0; n < numSamples; ++n)
        {
            const double v0 = data[n];
            const double v3 = v0 - ic2eq;
            const double v1 = a1 * ic1eq + a2 * v3;
            const double v2 = ic2eq + a2 * ic1eq + a3 * v3;

            ic1eq = 2.0 * v1 - ic1eq;
            ic2eq = 2.0 * v2 - ic2eq;

            double output = m0 * v0 + m1 * v1 + m2 * v2;

            // NaN/Infチェックとクランプを追加 (processBandStereoと一貫性を保つ)
            if (!isFiniteAndAbsInRangeMask(output, 0.0, 1.0e15))
                output = 0.0;

            // 出力もクランプして発散を防ぐ
            data[n] = std::clamp(output, -100.0, 100.0);

            // 状態変数が Inf/NaN に発散した場合は即座にリセットして次サンプルへの伝播を遮断する。
            // FTZ/DAZ 有効下でも Inf は flush されないため、この防衛は省略できない。
            // 条件式 !(abs < 閾値) は NaN に対しても true を返すため NaN/Inf を一括で捕捉する。
            // ループ内で状態変数のみをチェックする (出力データへのクランプより低コスト)。
            if (!isFiniteAndAbsInRangeMask(ic1eq, 0.0, 1.0e15)) ic1eq = 0.0;
            if (!isFiniteAndAbsInRangeMask(ic2eq, 0.0, 1.0e15)) ic2eq = 0.0;
        }

        // Denormal対策 & NaNチェック
        // Note: ScopedNoDenormals (DAZ/FTZ) が有効な場合でも、完全な0にならないと
        // 極小値が循環し続ける可能性があるため、明示的にフラッシュして計算負荷を抑える。
        if (!isFiniteAndAbsInRangeMask(ic1eq, DENORMAL_THRESHOLD, 1.0e15)) ic1eq = 0.0;
        if (!isFiniteAndAbsInRangeMask(ic2eq, DENORMAL_THRESHOLD, 1.0e15)) ic2eq = 0.0;

        state[0] = ic1eq;
        state[1] = ic2eq;
    }

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

        constexpr double DENORMAL_THRESHOLD = convo::numeric_policy::kDenormThresholdAudioState;

        for (int n = 0; n < numSamples; ++n)
        {
            // 【提案2強化版】AVX2相当のprefetch + 128byte先読み（L1キャッシュ最適化）
            // 最新コードのprocessBandStereoに適合した形で追加（__m128dベースを維持しつつ帯域向上）
            if (n + 8 < numSamples)
            {
                _mm_prefetch(reinterpret_cast<const char*>(dataL + n + 8), _MM_HINT_T0);
                _mm_prefetch(reinterpret_cast<const char*>(dataR + n + 8), _MM_HINT_T0);
            }

            // L[n] と R[n] を同時ロード
            const __m128d v0 = _mm_set_pd(dataR[n], dataL[n]);

            const __m128d v3 = _mm_sub_pd(v0, ic2eq);
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

            // NaN/Infチェック (isfinite): (x - x) は xがInf/NaNの時NaNになる
            const __m128d diff = _mm_sub_pd(output, output);
            const __m128d mask = _mm_cmpeq_pd(diff, _mm_setzero_pd());
            output = _mm_and_pd(output, mask);

            // クランプ (-100, +100) で発散防止
            output = _mm_min_pd(_mm_max_pd(output, cLow), cHigh);

            // L: lower element, R: upper element
            _mm_store_sd(&dataL[n], output);
            _mm_storeh_pd(&dataR[n], output);
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

    // ── 追加: AVX2 Gain Ramp ──
    inline void applyGainRamp_AVX2(double* __restrict data, int numSamples,
                                     double startGain, double increment) noexcept
    {
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
            __m256d vData0 = _mm256_loadu_pd(data + i);
            __m256d vOut0  = _mm256_mul_pd(vData0, vGain);
            _mm256_storeu_pd(data + i, vOut0);
            __m256d vGain1 = _mm256_add_pd(vGain, vInc4);

            // 2
            __m256d vData1 = _mm256_loadu_pd(data + i + 4);
            __m256d vOut1  = _mm256_mul_pd(vData1, vGain1);
            _mm256_storeu_pd(data + i + 4, vOut1);
            __m256d vGain2 = _mm256_add_pd(vGain1, vInc4);

            // 3
            __m256d vData2 = _mm256_loadu_pd(data + i + 8);
            __m256d vOut2  = _mm256_mul_pd(vData2, vGain2);
            _mm256_storeu_pd(data + i + 8, vOut2);
            __m256d vGain3 = _mm256_add_pd(vGain2, vInc4);

            // 4
            __m256d vData3 = _mm256_loadu_pd(data + i + 12);
            __m256d vOut3  = _mm256_mul_pd(vData3, vGain3);
            _mm256_storeu_pd(data + i + 12, vOut3);

            vGain = _mm256_add_pd(vGain, vInc16);
        }
        // Remaining
        for (; i < (numSamples / 4 * 4); i += 4)
        {
            __m256d vData = _mm256_loadu_pd(data + i);
            __m256d vOut  = _mm256_mul_pd(vData, vGain);
            _mm256_storeu_pd(data + i, vOut);
            vGain = _mm256_add_pd(vGain, vInc4);
        }
        // スカラー残余
        double gain = startGain + static_cast<double>(i) * increment;
        for (; i < numSamples; ++i) { data[i] *= gain; gain += increment; }
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

    // 【修正】サンプルレートを一度だけ atomic load してローカル変数に格納
    // 複数回の load() 呼び出しを回避し、Audio Thread 負荷を最小化
    const double sr = currentSampleRate.load(std::memory_order_relaxed);

    double inputRMS = cachedInputRMS;

    // ✅ フィルタ処理後の出力レベル計測
    double outputRMS = 0.0;
    for (int ch = 0; ch < numChannels; ++ch)
    {
        const double* data = block.getChannelPointer(ch);
        const double rms = calculateRMS(data, numSamples);

        if (rms > outputRMS) outputRMS = rms;
    }

    // 数値安定性対策: NaN/Infチェックとクランプ
    // 入力が極端に大きい場合（発振など）、エンベロープが汚染されるのを防ぐ
    static constexpr double MAX_ENV_VALUE = 1000.0; // +60dB

    if (!isFiniteNoLibm(inputRMS) || inputRMS > MAX_ENV_VALUE)   inputRMS = MAX_ENV_VALUE;
    if (!isFiniteNoLibm(outputRMS) || outputRMS > MAX_ENV_VALUE) outputRMS = MAX_ENV_VALUE;

    // アトミック変数のロード
    double envIn = agcEnvInput.load(std::memory_order_relaxed);
    double envOut = agcEnvOutput.load(std::memory_order_relaxed);
    double currentGain = agcCurrentGain.load(std::memory_order_relaxed);

    if (!isFiniteNoLibm(envIn))  envIn = 0.0;
    if (!isFiniteNoLibm(envOut)) envOut = 0.0;
    if (!isFiniteNoLibm(currentGain)) currentGain = 1.0;

    // ブロック長とサンプリングレートに基づいて係数を動的に計算
    // alpha = dt / (dt + tau)  (dt: samples, tau: samples)
    const double alphaAttack  = (double)numSamples / ((double)numSamples + sr * AGC_ATTACK_TIME_SEC);
    const double alphaRelease = (double)numSamples / ((double)numSamples + sr * AGC_RELEASE_TIME_SEC);
    const double smoothAlpha  = (double)numSamples / ((double)numSamples + sr * AGC_SMOOTH_TIME_SEC);

    // 指数移動平均 (EMA) によるエンベロープ検波 (アシンメトリック: アタック速い / リリース遅い)
    const double inputAlpha  = (inputRMS  > envIn)  ? alphaAttack : alphaRelease;
    const double outputAlpha = (outputRMS > envOut) ? alphaAttack : alphaRelease;
    envIn  = envIn  * (1.0 - inputAlpha)  + inputRMS  * inputAlpha;
    envOut = envOut * (1.0 - outputAlpha) + outputRMS * outputAlpha;

    // Denormal対策: 極小値をゼロにクランプ (無音時のCPU負荷対策)
    if (envIn < convo::numeric_policy::kDenormThresholdAudioState) envIn = 0.0;
    if (envOut < convo::numeric_policy::kDenormThresholdAudioState) envOut = 0.0;

    // ターゲットゲイン計算
    double targetGain = calculateAGCGain(envIn, envOut);

    // ゲイン変化のスムーシング
    double nextGain = currentGain * (1.0 - smoothAlpha) + targetGain * smoothAlpha;

    // アトミック変数のストア
    agcEnvInput.store(envIn, std::memory_order_relaxed);
    agcEnvOutput.store(envOut, std::memory_order_relaxed);
    agcCurrentGain.store(nextGain, std::memory_order_relaxed);

    // ゲイン適用 (ランプ: currentGain -> nextGain)
    // ブロック境界での不連続性を防ぐため、サンプル単位で補間する
    const double gainIncrement = (nextGain - currentGain) / static_cast<double>(numSamples);

    for (int ch = 0; ch < numChannels; ++ch)
        applyGainRamp_AVX2(block.getChannelPointer(ch), numSamples, currentGain, gainIncrement);
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
    // Audio Thread 入口で MXCSR の FTZ/DAZ を関数スコープで保証する。
    // 呼び出し元設定に依存せず、EQ 単体でもデノーマル起因の負荷増大を防ぐ。
    juce::ScopedNoDenormals noDenormals;

    const bool requestedBypass = bypassRequested.load(std::memory_order_relaxed);
    bool effectiveBypass = bypassed.load(std::memory_order_relaxed);

    const double targetBypassFade = requestedBypass ? 0.0 : 1.0;
    if (absNoLibm(bypassFadeGain.getTargetValue() - targetBypassFade) > 1.0e-12)
    {
        if (!requestedBypass && effectiveBypass)
            bypassed.store(false, std::memory_order_relaxed);
        bypassFadeGain.setTargetValue(targetBypassFade);
        effectiveBypass = bypassed.load(std::memory_order_relaxed);
    }

    const bool bypassTransitionActive = bypassFadeGain.isSmoothing();
    if (requestedBypass && effectiveBypass && !bypassTransitionActive)
        return;

    const int numSamples = (int)block.getNumSamples();

    // ==================================================================
    // 【Issue 4 追加安全ガード】オーバーラン即検出（Audio Thread安全版）
    // ==================================================================
    // パッチ2: jassert削除 + 安全クリア（ガイドライン厳守）
    if (numSamples <= 0 || static_cast<size_t>(numSamples) > static_cast<size_t>(maxInternalBlockSize))
    {
        // バッファオーバーラン時は安全にゼロクリアして早期リターン
        for (int ch = 0; ch < (int)block.getNumChannels(); ++ch)
            juce::FloatVectorOperations::clear(block.getChannelPointer(ch), numSamples);
        return;
    }

    const int numChannels = std::min((int)block.getNumChannels(), MAX_CHANNELS);

    double* dryCopyBase = nullptr;
    if (bypassTransitionActive)
    {
        const int requiredDrySamples = numSamples * numChannels;
        if (scratchBuffer && requiredDrySamples <= scratchCapacity)
        {
            dryCopyBase = scratchBuffer.get();
            for (int ch = 0; ch < numChannels; ++ch)
                std::memcpy(dryCopyBase + (ch * numSamples),
                            block.getChannelPointer(ch),
                            sizeof(double) * static_cast<size_t>(numSamples));
        }
    }

    // ── State Reset Handling ──
    // パラメータ変更に伴う状態リセット要求を処理
    if (agcResetRequest.exchange(false, std::memory_order_relaxed))
    {
        agcCurrentGain.store(1.0, std::memory_order_relaxed);
        agcEnvInput.store(0.0, std::memory_order_relaxed);
        agcEnvOutput.store(0.0, std::memory_order_relaxed);
    }

    uint32_t mask = bandResetMask.exchange(0, std::memory_order_relaxed);
    if (mask != 0)
    {
        // 最適化: 全バンドリセットの場合は memset で一括クリア
        if (mask == 0xFFFFFFFF)
        {
            std::memset(filterState.data(), 0, sizeof(filterState));
        }
        else
        {
            for (int i = 0; i < NUM_BANDS; ++i)
            {
                if (mask & (1u << i))
                    for (int ch = 0; ch < MAX_CHANNELS; ++ch)
                        std::memset(filterState[ch][i].data(), 0, sizeof(double) * 2);
            }
        }
    }

    const bool isAgcEnabled = agcEnabled.load(std::memory_order_acquire);
    // ✅ フィルタ処理前に入力レベルをキャッシュ (AGCが有効な場合のみ)
    if (isAgcEnabled)
    {
        cachedInputRMS = 0.0;
        for (int ch = 0; ch < numChannels; ++ch)
        {
            const double* data = block.getChannelPointer(ch);
            const double rms = calculateRMS(data, numSamples);

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
    // C-1改善対応: 係数変化時はクロスフェード処理を実施
    for (int i = 0; i < numActiveBands; ++i)
    {
        const auto& band = activeBands[i];
        const EQChannelMode mode = band.node->mode;

        // 【C-1改善】係数が急激に変わった場合のみ 2倍の処理コスト発生
        //（通常はこのブランチは実行されず、追加コストなし）
        if (band.node->coeffsChanged && scratchBuffer && scratchCapacity >= numSamples * MAX_CHANNELS)
        {
            // ── クロスフェード処理 ──
            if (mode == EQChannelMode::Stereo && numChannels >= 2)
            {
                // L/R 両チャンネルのスクラッチ領域を確保
                double* scratchL = scratchBuffer.get();
                double* scratchR = scratchBuffer.get() + numSamples;

                // 入力をスクラッチにコピー
                std::copy(block.getChannelPointer(0), block.getChannelPointer(0) + numSamples, scratchL);
                std::copy(block.getChannelPointer(1), block.getChannelPointer(1) + numSamples, scratchR);

                // 状態を保存（旧係数用）
                double prevStateL[2] = { filterState[0][band.index][0], filterState[0][band.index][1] };
                double prevStateR[2] = { filterState[1][band.index][0], filterState[1][band.index][1] };

                // ① スクラッチに旧係数で処理実行
                processBandStereo(scratchL, scratchR, numSamples,
                                 band.node->prevCoeffs,
                                 prevStateL, prevStateR);

                // ② 元バッファに新係数で処理実行（状態は自動的に更新）
                processBandStereo(block.getChannelPointer(0), block.getChannelPointer(1), numSamples,
                                 band.node->coeffs,
                                 filterState[0][band.index].data(),
                                 filterState[1][band.index].data());

                // ③ リニアクロスフェード: scratchBuffer (旧) → data (新)
                const double step = 1.0 / static_cast<double>(numSamples);
                for (int n = 0; n < numSamples; ++n)
                {
                    const double t = (n + 1.0) * step;
                    const double w_old = 1.0 - t;
                    const double w_new = t;
                    block.getChannelPointer(0)[n] = scratchL[n] * w_old + block.getChannelPointer(0)[n] * w_new;
                    block.getChannelPointer(1)[n] = scratchR[n] * w_old + block.getChannelPointer(1)[n] * w_new;
                }
            }
            else
            {
                // モノラル処理（L/R個別）
                if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Left)
                    if (numChannels > 0)
                    {
                        double* data = block.getChannelPointer(0);
                        std::copy(data, data + numSamples, scratchBuffer.get());

                        double prevState[2] = { filterState[0][band.index][0], filterState[0][band.index][1] };
                        processBand(scratchBuffer.get(), numSamples, band.node->prevCoeffs, prevState);
                        processBand(data, numSamples, band.node->coeffs, filterState[0][band.index].data());

                        const double step = 1.0 / static_cast<double>(numSamples);
                        for (int n = 0; n < numSamples; ++n)
                        {
                            const double t = (n + 1.0) * step;
                            data[n] = scratchBuffer[n] * (1.0 - t) + data[n] * t;
                        }
                    }
                if (mode == EQChannelMode::Stereo || mode == EQChannelMode::Right)
                    if (numChannels > 1)
                    {
                        double* data = block.getChannelPointer(1);
                        std::copy(data, data + numSamples, scratchBuffer.get());

                        double prevState[2] = { filterState[1][band.index][0], filterState[1][band.index][1] };
                        processBand(scratchBuffer.get(), numSamples, band.node->prevCoeffs, prevState);
                        processBand(data, numSamples, band.node->coeffs, filterState[1][band.index].data());

                        const double step = 1.0 / static_cast<double>(numSamples);
                        for (int n = 0; n < numSamples; ++n)
                        {
                            const double t = (n + 1.0) * step;
                            data[n] = scratchBuffer[n] * (1.0 - t) + data[n] * t;
                        }
                    }
            }
        }
        else
        {
            // 係数が変わらない場合は通常のプロセッシング（無視できるコスト）
            if (mode == EQChannelMode::Stereo && numChannels >= 2)
            {
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
    }

    // トータルゲイン / AGC 適用
    if (isAgcEnabled)
    {
        processAGC(block);
    }
    else
    {
        // 【Fix Bug #7】Audio Thread内でのlibm呼び出し禁止。
        // juce::Decibels::decibelsToGain() は内部で std::pow() (libm) を呼ぶため
        // Audio Thread 内での使用は規約違反である。
        // 対策: ゲイン値はMessage Thread側でdBからlinearに変換し、
        // totalGainTarget (atomic<double>) で事前に渡す。
        // Audio Threadでは atomic load のみ行う。
        const double targetGain = totalGainTarget.load(std::memory_order_relaxed);

        if (absNoLibm(smoothTotalGain.getTargetValue() - targetGain) > 1e-6)
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

    if (bypassTransitionActive && dryCopyBase != nullptr)
    {
        for (int sampleIndex = 0; sampleIndex < numSamples; ++sampleIndex)
        {
            const double wetGain = bypassFadeGain.getNextValue();
            const double dryGain = 1.0 - wetGain;

            for (int ch = 0; ch < numChannels; ++ch)
            {
                double* wetPtr = block.getChannelPointer(ch);
                const double dryValue = dryCopyBase[ch * numSamples + sampleIndex];
                wetPtr[sampleIndex] = wetPtr[sampleIndex] * wetGain + dryValue * dryGain;
            }
        }

        if (!bypassFadeGain.isSmoothing())
        {
            if (requestedBypass)
            {
                bypassed.store(true, std::memory_order_relaxed);
                bypassFadeGain.setCurrentAndTargetValue(0.0);
            }
            else
            {
                bypassed.store(false, std::memory_order_relaxed);
                bypassFadeGain.setCurrentAndTargetValue(1.0);
            }
        }
    }
}


//--------------------------------------------------------------
// BandNode作成 (Message Thread)
//--------------------------------------------------------------
EQProcessor::BandNode* EQProcessor::createBandNode(int band, const EQState& state) const
{
    auto node = new BandNode();
    node->addRef();
    const auto& params = state.bands[band];

    node->active = params.enabled;
    node->mode = state.bandChannelModes[band];
    // 【修正】atomic load して calcSVFCoeffs に渡す
    node->coeffs = calcSVFCoeffs(state.bandTypes[band], params.frequency, params.gain, params.q,
                                 currentSampleRate.load(std::memory_order_relaxed));

    // 最適化: ゲインが0dB付近ならスキップ
    if (node->active) {
        EQBandType type = state.bandTypes[band];
        if ((type != EQBandType::LowPass && type != EQBandType::HighPass) && std::abs(params.gain) < 0.01f)
            node->active = false;
    }

    return node;
}

//--------------------------------------------------------------
// SVF係数比較ヘルパー (Message Thread内での使用)
//--------------------------------------------------------------
static inline bool areCoeffsEqual(const EQCoeffsSVF& a, const EQCoeffsSVF& b) noexcept
{
    static constexpr double COEFF_EPSILON = 1e-10;
    return std::abs(a.g - b.g) < COEFF_EPSILON &&
           std::abs(a.k - b.k) < COEFF_EPSILON &&
           std::abs(a.a1 - b.a1) < COEFF_EPSILON &&
           std::abs(a.a2 - b.a2) < COEFF_EPSILON &&
           std::abs(a.a3 - b.a3) < COEFF_EPSILON &&
           std::abs(a.m0 - b.m0) < COEFF_EPSILON &&
           std::abs(a.m1 - b.m1) < COEFF_EPSILON &&
           std::abs(a.m2 - b.m2) < COEFF_EPSILON;
}

//--------------------------------------------------------------
// BandNode更新 (Message Thread)
// C-1 改善: 係数変化検出とクロスフェード準備
//--------------------------------------------------------------
void EQProcessor::updateBandNode(int band)
{
    auto state = activeState;
    if (state == nullptr) return;
    auto newNode = createBandNode(band, *state);

    // 係数が実際に変更されたかを判定
    const BandNode* oldNode = activeBandNodes[band];
    if (oldNode && !areCoeffsEqual(oldNode->coeffs, newNode->coeffs))
    {
        // 係数が変わる場合: 前回の係数を保持し、クロスフェードフラグを立てる
        newNode->prevCoeffs = oldNode->coeffs;
        newNode->coeffsChanged = true;
    }
    else
    {
        // 係数が変わらない場合: 前回の係数を新しい係数で初期化（初回時など）
        newNode->prevCoeffs = newNode->coeffs;
        newNode->coeffsChanged = false;
    }

    bandNodes[band].store(newNode, std::memory_order_release);

    // 古いノードをゴミ箱へ (Audio Threadが使用中の可能性があるため即削除しない)
    const juce::ScopedLock sl(trashBinLock);
    if (oldNode)
    {
        bandNodeTrashBinPending.push_back(const_cast<BandNode*>(oldNode));
    }
    activeBandNodes[band] = newNode;
}

void EQProcessor::cleanup()
{
    std::vector<BandNode*> nodesToDelete;
    std::vector<EQState*> statesToDelete;

    {
        const juce::ScopedLock sl(trashBinLock);
        const uint32 now = juce::Time::getMillisecondCounter();

        // Clean BandNodes (Time-based)
        // Audio ThreadがRaw Pointerを参照している可能性があるため、即時削除せず一定時間(10000ms)待つ
        for (auto it = bandNodeTrashBin.begin(); it != bandNodeTrashBin.end(); )
        {
            if ((now - it->second) > 10000)
            {
                nodesToDelete.push_back(it->first);
                it = bandNodeTrashBin.erase(it);
            }
            else { ++it; }
        }

        // Move pending to main trash with timestamp
        for (const auto& ptr : bandNodeTrashBinPending)
            bandNodeTrashBin.push_back({ptr, now});
        bandNodeTrashBinPending.clear();

        // Clean States (use_count==1 で解放)
        for (auto it = stateTrashBin.begin(); it != stateTrashBin.end(); )
        {
            if ((*it)->refCount.load() == 1)
            {
                statesToDelete.push_back(*it);
                it = stateTrashBin.erase(it);
            }
            else { ++it; }
        }

        stateTrashBin.insert(stateTrashBin.end(), stateTrashBinPending.begin(), stateTrashBinPending.end());
        stateTrashBinPending.clear();

        // 安全性優先: use_count == 1 の条件を満たすもののみ回収する。
        // サイズ超過のみを理由に強制解放すると、Audio Thread 参照中オブジェクトの
        // 早期解放リスクがあるため行わない。
    }

    for (auto* p : nodesToDelete) p->release();
    for (auto* p : statesToDelete) p->release();
}

void EQProcessor::forceCleanup()
{
    std::vector<std::pair<BandNode*, uint32>> nodesToDelete;
    std::vector<BandNode*> nodesPendingToDelete;
    std::vector<EQState*> statesToDelete;
    std::vector<EQState*> statesPendingToDelete;

    {
        const juce::ScopedLock sl(trashBinLock);
        nodesToDelete.swap(bandNodeTrashBin);
        nodesPendingToDelete.swap(bandNodeTrashBinPending);
        statesToDelete.swap(stateTrashBin);
        statesPendingToDelete.swap(stateTrashBinPending);
    }

    for (auto& entry : nodesToDelete) entry.first->release();
    for (auto* p : nodesPendingToDelete) p->release();
    for (auto* p : statesToDelete) p->release();
    for (auto* p : statesPendingToDelete) p->release();
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
    // [FIX] 上限チェックを削除: 以前の sr > 384000.0 は誤り。
    // オーバーサンプリング使用時 (例: 96kHz × 8 = 768kHz) にsr=48000へフォールバックし、
    // zCache (処理srベース) とbiquad係数 (48kHzベース) が不整合になり、
    // 表示曲線のピーク周波数が最大16倍ずれる不具合を引き起こしていた。
    // (例: 100Hz設定 → 1600Hz表示、ユーザー報告: 25-100Hzブーストが800Hz付近に表示)
    // Biquad計算式はsr > 0.0であれば任意のサンプルレートで数学的に正確に動作する。
    if (sr <= 0.0 || !std::isfinite(sr))
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
