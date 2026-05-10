//============================================================================
// EQProcessor.Core.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// ライフサイクル、リソース管理、状態I/O, getters
//============================================================================
#include "EQProcessor.h"
#include <cmath>
#include <cstring>
#include <regex>
#include "core/EpochManager.h"
#include "core/EBRQueue.h"
#include "core/RCUReader.h"

//============================================================================
// Destruction handlers for EBR (Epoch-Based Reclamation)
//============================================================================
static void retireEQState(EQProcessor::EQState* state) {
    if (state) state->release();
}

static void retireBandNode(EQProcessor::BandNode* node) {
    if (node) node->release();
}

//============================================================================
// コンストラクタ
//============================================================================
EQProcessor::EQProcessor()
{
    // 初期係数ノードの作成
    resetToDefaults();
}

//============================================================================
// デストラクタ
//============================================================================
EQProcessor::~EQProcessor()
{
    juce::Logger::writeToLog("[DIAG EQProcessor] ~EQProcessor: enter");
    if (auto* oldState = currentStateRaw.exchange(nullptr, std::memory_order_acq_rel))
        retireEQState(oldState);

    for (auto& node : bandNodes) {
        if (auto* n = node.exchange(nullptr, std::memory_order_release))
            retireBandNode(n);
    }

    for (auto*& node : activeBandNodes) {
        if (node) {
            retireBandNode(node);
            node = nullptr;
        }
    }

    releaseResources();
    juce::Logger::writeToLog("[DIAG EQProcessor] ~EQProcessor: exit");
}

//============================================================================
// リソース解放 (Message Thread)
//============================================================================
void EQProcessor::releaseResources()
{
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before scratchBuffer.reset");
    scratchBuffer.reset();
    scratchCapacity = 0;
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: after scratchBuffer.reset");

    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before dryBypassBuffer.reset");
    dryBypassBuffer.reset();
    dryBypassCapacity = 0;
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: after dryBypassBuffer.reset");

    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before parallelInputBuffer.reset");
    parallelInputBuffer.reset();
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: after parallelInputBuffer.reset");

    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before parallelWorkBuffer.reset");
    parallelWorkBuffer.reset();
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: after parallelWorkBuffer.reset");

    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before parallelAccumBuffer.reset");
    parallelAccumBuffer.reset();
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: after parallelAccumBuffer.reset");

    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before structureOldOutBuffer.reset");
    structureOldOutBuffer.reset();
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: after structureOldOutBuffer.reset");

    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: before structureNewOutBuffer.reset");
    structureNewOutBuffer.reset();
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: after structureNewOutBuffer.reset");

    parallelBufferCapacity = 0;
    structureXfadeBufferCapacity = 0;
    juce::Logger::writeToLog("[DIAG EQProcessor] releaseResources: end");
}

//============================================================================
// デフォルト値リセット
//============================================================================
void EQProcessor::resetToDefaults()
{
    auto newState = new EQState();

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
    nonlinearSaturation.store(0.2f, std::memory_order_relaxed);
    requestedStructure.store(FilterStructure::Serial, std::memory_order_relaxed);
    activeStructure.store(FilterStructure::Serial, std::memory_order_relaxed);
    newState->agcEnabled = false;
    newState->nonlinearSaturation = 0.2f;
    newState->filterStructure = 0;

    auto oldState = currentStateRaw.exchange(newState, std::memory_order_acq_rel);

    if (oldState) {
        retireEQState(oldState);
    }
    convo::EpochManager::instance().advanceEpoch();

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

//============================================================================
// フィルタ状態リセット (Audio Thread)
//============================================================================
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

//============================================================================
// プリセット読み込み (simplified)
//============================================================================
void EQProcessor::loadPreset(int /*index*/)
{
    resetToDefaults();
    sendChangeMessage();
}

bool EQProcessor::loadFromTextFile(const juce::File& file)
{
    if (!file.existsAsFile())
        return false;

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
        line = line.upToFirstOccurrenceOf("#", false, false);
        line = line.upToFirstOccurrenceOf(";", false, false);
        line = line.trim();

        if (line.isEmpty())
            continue;

        std::regex tokenRegex(R"(\S+)");
        auto stdLine = line.toStdString();
        auto tokensBegin = std::sregex_iterator(stdLine.begin(), stdLine.end(), tokenRegex);
        auto tokensEnd = std::sregex_iterator();

        juce::StringArray tokens;
        for (auto i = tokensBegin; i != tokensEnd; ++i)
            tokens.add(i->str());

        if (tokens.isEmpty()) continue;

        if (tokens[0].startsWithIgnoreCase("Preamp"))
        {
            for (int i = 1; i < tokens.size(); ++i)
            {
                if (tokens[i].containsAnyOf("0123456789-."))
                {
                    const float val = tokens[i].getFloatValue();
                    setTotalGain(val);
                    break;
                }
            }
        }
        else if (tokens[0].startsWithIgnoreCase("Channel"))
        {
            bool hasL = false;
            bool hasR = false;

            for (const auto& token : tokens)
            {
                juce::String t = token;
                if (t.startsWithIgnoreCase("Channel"))
                    t = t.substring(7);

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
        else if (tokens[0].startsWithIgnoreCase("Filter"))
        {
            if (currentFilterIndex >= NUM_BANDS)
            {
                if (!maxBandsWarningShown)
                {
                    maxBandsWarningShown = true;
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

            bool enabled = true;
            bool typeFound = false;
            float freq = 0.0f;
            float gain = 0.0f;
            float q = DEFAULT_Q;
            bool qFound = false;

            for (int i = 1; i < tokens.size(); ++i)
            {
                const juce::String& t = tokens[i];

                if (t.equalsIgnoreCase("ON")) {
                    enabled = true;
                    continue;
                }
                if (t.equalsIgnoreCase("OFF")) {
                    enabled = false;
                    continue;
                }

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

            setBandChannelMode(currentFilterIndex, currentChannelMode);
            currentFilterIndex++;
        }
    }

    bandResetMask.store(0xFFFFFFFF, std::memory_order_relaxed);
    sendChangeMessage();
    return true;
}

//============================================================================
// 状態取得
//============================================================================
juce::ValueTree EQProcessor::getState() const
{
    auto state = currentStateRaw.load(std::memory_order_acquire);
    if (state == nullptr) return juce::ValueTree("EQ");

    juce::ValueTree v ("EQ");
    v.setProperty ("totalGain", state->totalGainDb, nullptr);
    v.setProperty("agcEnabled", agcEnabled.load(std::memory_order_acquire), nullptr);
    v.setProperty("nonlinearSaturation", nonlinearSaturation.load(std::memory_order_relaxed), nullptr);
    v.setProperty("filterStructure",
                  static_cast<int>(requestedStructure.load(std::memory_order_relaxed)),
                  nullptr);

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

//============================================================================
// 状態設定
//============================================================================
void EQProcessor::setState (const juce::ValueTree& v)
{
    if (v.hasProperty ("totalGain")) setTotalGain (v.getProperty ("totalGain"));
    setAGCEnabled(v.getProperty("agcEnabled", false));
    if (v.hasProperty("nonlinearSaturation"))
        setNonlinearSaturation(static_cast<float>(v.getProperty("nonlinearSaturation")));
    if (v.hasProperty("filterStructure"))
        setFilterStructure(static_cast<FilterStructure>(static_cast<int>(v.getProperty("filterStructure"))));

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
    activeStructure.store(requestedStructure.load(std::memory_order_relaxed), std::memory_order_relaxed);
    sendChangeMessage();
}

//============================================================================
// 状態同期（他のプロセッサから）
//============================================================================
void EQProcessor::syncStateFrom(const EQProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    storeTotalGainDb(other.totalGainDbTarget.load(std::memory_order_relaxed));

    auto otherState = other.currentStateRaw.load(std::memory_order_acquire);
    auto oldState = currentStateRaw.exchange(otherState, std::memory_order_acq_rel);

    if (oldState)
    {
        retireEQState(oldState);
    }
    convo::EpochManager::instance().advanceEpoch();

    for (int i = 0; i < NUM_BANDS; ++i)
    {
        auto node = other.activeBandNodes[i];
        bandNodes[i].store(node, std::memory_order_release);
        if (activeBandNodes[i]) {
            retireBandNode(activeBandNodes[i]);
        }
        activeBandNodes[i] = node;
    }
    convo::EpochManager::instance().advanceEpoch();
    agcEnabled.store(other.agcEnabled.load(std::memory_order_acquire), std::memory_order_release);
    agcCurrentGain.store(other.agcCurrentGain.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvInput.store(other.agcEnvInput.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvOutput.store(other.agcEnvOutput.load(std::memory_order_relaxed), std::memory_order_relaxed);
    nonlinearSaturation.store(other.nonlinearSaturation.load(std::memory_order_relaxed), std::memory_order_relaxed);
    requestedStructure.store(other.requestedStructure.load(std::memory_order_relaxed), std::memory_order_relaxed);
    activeStructure.store(requestedStructure.load(std::memory_order_relaxed), std::memory_order_relaxed);

    bandResetMask.store(other.bandResetMask.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcResetRequest.store(other.agcResetRequest.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

//============================================================================
// 単一バンド同期
//============================================================================
void EQProcessor::syncBandNodeFrom(const EQProcessor& other, int bandIndex)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (bandIndex < 0 || bandIndex >= NUM_BANDS) return;

    auto node = other.activeBandNodes[bandIndex];

    if (node) {
        // EBR: lifetime managed by RCU
    }
    bandNodes[bandIndex].store(node, std::memory_order_release);

    if (activeBandNodes[bandIndex])
    {
        retireBandNode(activeBandNodes[bandIndex]);
    }
    activeBandNodes[bandIndex] = node;
}

//============================================================================
// グローバル状態同期 (Worker Threadからも安全)
//============================================================================
void EQProcessor::syncGlobalStateFrom(const EQProcessor& other)
{
    storeTotalGainDb(other.totalGainDbTarget.load(std::memory_order_relaxed));
    agcEnabled.store(other.agcEnabled.load(std::memory_order_acquire), std::memory_order_release);
    agcCurrentGain.store(other.agcCurrentGain.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvInput.store(other.agcEnvInput.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcEnvOutput.store(other.agcEnvOutput.load(std::memory_order_relaxed), std::memory_order_relaxed);
    nonlinearSaturation.store(other.nonlinearSaturation.load(std::memory_order_relaxed), std::memory_order_relaxed);
    requestedStructure.store(other.requestedStructure.load(std::memory_order_relaxed), std::memory_order_relaxed);

    bandResetMask.store(other.bandResetMask.load(std::memory_order_relaxed), std::memory_order_relaxed);
    agcResetRequest.store(other.agcResetRequest.load(std::memory_order_relaxed), std::memory_order_relaxed);
}

//============================================================================
// メモリ事前確保 & 係数再計算
//============================================================================
void EQProcessor::prepareToPlay(double sampleRate, int newMaxInternalBlockSize)
{
    const bool rateChanged = (std::abs(currentSampleRate.load(std::memory_order_relaxed) - sampleRate) > 1e-6);
    if (rateChanged)
        currentSampleRate.store(sampleRate, std::memory_order_relaxed);

    const int requiredSize = newMaxInternalBlockSize;
    this->maxInternalBlockSize = requiredSize;

    const int required = juce::nextPowerOfTwo(requiredSize) * 8;
    if (scratchCapacity < required)
    {
        scratchBuffer.reset(static_cast<double*>(convo::aligned_malloc(required * sizeof(double), 64)));
        scratchCapacity = required;
        juce::FloatVectorOperations::clear(scratchBuffer.get(), required);
    }

    const int channelRequired = juce::nextPowerOfTwo(requiredSize) * MAX_CHANNELS;
    if (dryBypassCapacity < channelRequired)
    {
        dryBypassBuffer.reset(static_cast<double*>(convo::aligned_malloc(channelRequired * sizeof(double), 64)));
        dryBypassCapacity = channelRequired;
        juce::FloatVectorOperations::clear(dryBypassBuffer.get(), channelRequired);
    }

    if (parallelBufferCapacity < channelRequired)
    {
        parallelInputBuffer.reset(static_cast<double*>(convo::aligned_malloc(channelRequired * sizeof(double), 64)));
        parallelWorkBuffer.reset(static_cast<double*>(convo::aligned_malloc(channelRequired * sizeof(double), 64)));
        parallelAccumBuffer.reset(static_cast<double*>(convo::aligned_malloc(channelRequired * sizeof(double), 64)));
        parallelBufferCapacity = channelRequired;
        juce::FloatVectorOperations::clear(parallelInputBuffer.get(), channelRequired);
        juce::FloatVectorOperations::clear(parallelWorkBuffer.get(), channelRequired);
        juce::FloatVectorOperations::clear(parallelAccumBuffer.get(), channelRequired);
    }

    if (structureXfadeBufferCapacity < channelRequired)
    {
        structureOldOutBuffer.reset(static_cast<double*>(convo::aligned_malloc(channelRequired * sizeof(double), 64)));
        structureNewOutBuffer.reset(static_cast<double*>(convo::aligned_malloc(channelRequired * sizeof(double), 64)));
        structureXfadeBufferCapacity = channelRequired;
        juce::FloatVectorOperations::clear(structureOldOutBuffer.get(), channelRequired);
        juce::FloatVectorOperations::clear(structureNewOutBuffer.get(), channelRequired);
    }

    if (newMaxInternalBlockSize > 0 && agcCoeffTableCapacity < (newMaxInternalBlockSize + 1))
    {
        agcAttackCoeffTable.reset(static_cast<double*>(convo::aligned_malloc((newMaxInternalBlockSize + 1) * sizeof(double), 64)));
        agcReleaseCoeffTable.reset(static_cast<double*>(convo::aligned_malloc((newMaxInternalBlockSize + 1) * sizeof(double), 64)));
        agcSmoothCoeffTable.reset(static_cast<double*>(convo::aligned_malloc((newMaxInternalBlockSize + 1) * sizeof(double), 64)));

        if (agcAttackCoeffTable && agcReleaseCoeffTable && agcSmoothCoeffTable)
            agcCoeffTableCapacity = newMaxInternalBlockSize + 1;
        else
            agcCoeffTableCapacity = 0;
    }

    auto state = currentStateRaw.load(std::memory_order_acquire);

    smoothTotalGain.reset(sampleRate, SMOOTHING_TIME_SEC);
    bypassFadeGain.reset(sampleRate, BYPASS_FADE_TIME_SEC);

    if (state)
    {
        storeTotalGainDb(state->totalGainDb);
        smoothTotalGain.setCurrentAndTargetValue(
            juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
    }

    std::memset(filterState.data(), 0, sizeof(filterState));

    const double sr = sampleRate;
    agcAttackCoeff.store(std::exp(-1.0 / (sr * AGC_ATTACK_TIME_SEC)), std::memory_order_relaxed);
    agcReleaseCoeff.store(std::exp(-1.0 / (sr * AGC_RELEASE_TIME_SEC)), std::memory_order_relaxed);
    agcSmoothCoeff.store(std::exp(-1.0 / (sr * AGC_SMOOTH_TIME_SEC)), std::memory_order_relaxed);

    if (agcCoeffTableCapacity > 0 && agcAttackCoeffTable && agcReleaseCoeffTable && agcSmoothCoeffTable)
    {
        for (int i = 0; i < agcCoeffTableCapacity; ++i)
        {
            const double n = static_cast<double>(i);
            agcAttackCoeffTable[i]  = 1.0 - std::exp(-n / (sr * AGC_ATTACK_TIME_SEC));
            agcReleaseCoeffTable[i] = 1.0 - std::exp(-n / (sr * AGC_RELEASE_TIME_SEC));
            agcSmoothCoeffTable[i]  = 1.0 - std::exp(-n / (sr * AGC_SMOOTH_TIME_SEC));
        }
    }

    agcCurrentGain.store(1.0, std::memory_order_relaxed);
    agcEnvInput.store(0.0, std::memory_order_relaxed);
    agcEnvOutput.store(0.0, std::memory_order_relaxed);

    bandResetMask.store(0, std::memory_order_relaxed);
    agcResetRequest.store(false, std::memory_order_relaxed);
    activeStructure.store(requestedStructure.load(std::memory_order_relaxed), std::memory_order_relaxed);

    const bool requestedBypass = bypassRequested.load(std::memory_order_relaxed);
    bypassed.store(requestedBypass, std::memory_order_relaxed);
    bypassFadeGain.setCurrentAndTargetValue(requestedBypass ? 0.0 : 1.0);

    if (rateChanged)
    {
        for (int i = 0; i < NUM_BANDS; ++i)
        {
            auto loopState = currentStateRaw.load(std::memory_order_acquire);
            if (loopState)
            {
                auto newNode = createBandNode(i, *loopState);
                auto oldNode = bandNodes[i].exchange(newNode, std::memory_order_release);

                if (activeBandNodes[i])
                    retireBandNode(activeBandNodes[i]);

                activeBandNodes[i] = newNode;
            }
        }
    }

    sendChangeMessage();
}

//============================================================================
// Getters & helpers
//============================================================================
convo::EQParameters EQProcessor::EQState::toEQParameters() const
{
    convo::EQParameters params;
    for (int i = 0; i < EQProcessor::NUM_BANDS; ++i)
    {
        params.bands[i].frequency = bands[i].frequency;
        params.bands[i].gain = bands[i].gain;
        params.bands[i].q = bands[i].q;
        params.bands[i].enabled = bands[i].enabled;
        params.bands[i].type = static_cast<int>(bandTypes[i]);
        params.bands[i].channelMode = static_cast<int>(bandChannelModes[i]);
    }
    params.totalGainDb = totalGainDb;
    params.agcEnabled = agcEnabled;
    params.nonlinearSaturation = nonlinearSaturation;
    params.filterStructure = filterStructure;
    return params;
}

EQProcessor::EQState* EQProcessor::getEQState() const
{
    return currentStateRaw.load(std::memory_order_acquire);
}

float EQProcessor::getTotalGain() const
{
    auto state = currentStateRaw.load(std::memory_order_acquire);
    if (state == nullptr) return 0.0f;
    return state->totalGainDb;
}

EQBandType EQProcessor::getBandType(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQBandType::Peaking;
    auto state = currentStateRaw.load(std::memory_order_acquire);
    if (state == nullptr) return EQBandType::Peaking;
    return state->bandTypes[band];
}

EQChannelMode EQProcessor::getBandChannelMode(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQChannelMode::Stereo;
    auto state = currentStateRaw.load(std::memory_order_acquire);
    if (state == nullptr) return EQChannelMode::Stereo;
    return state->bandChannelModes[band];
}

void EQProcessor::cleanup()
{
    // EBR管理されているため、個別クリーンアップ不要
}

EQBandParams EQProcessor::getBandParams(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return {};
    auto state = currentStateRaw.load(std::memory_order_acquire);
    if (state == nullptr) return {};
    return state->bands[band];
}
