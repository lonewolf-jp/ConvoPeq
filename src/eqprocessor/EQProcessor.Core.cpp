//============================================================================
// EQProcessor.Core.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// ライフサイクル、リソース管理、状態I/O, getters
//============================================================================
#include "EQProcessor.h"
#include <cmath>
#include <cstring>
#include <regex>
#include "core/EpochDomain.h"
#include "core/RCUReader.h"
#include "audioengine/ISRRuntimePublicationCoordinator.h"

#include "audioengine/AtomicAccess.h"

//============================================================================
// Destruction handlers for EBR (Epoch-Based Reclamation)
// L5: epoch-only 削除。RefCountedDeferred の二重ライフタイムモデルを廃止。
//============================================================================
namespace
{
void deleteEQStatePtr(void* p) noexcept { delete static_cast<EQProcessor::EQState*>(p); }
void deleteBandNodePtr(void* p) noexcept { delete static_cast<EQProcessor::BandNode*>(p); }
}

bool EQProcessor::enqueueDeferredDeleteWithFallback(void* ptr,
                                                    void (*deleter)(void*),
                                                    uint64_t epoch) noexcept
{
    if (ptr == nullptr || deleter == nullptr)
        return true;

    // Retire authority: route through coordinator if available
    // [Phase-A] 単一回試行 + drop. retryループは撤廃 (P0-5).
    if (m_retireCoordinator != nullptr)
    {
        const uint64_t retireEpoch = (epoch != 0) ? epoch : m_epochDomain.currentEpoch();
        // ★ FIX: reinterpret_cast<ISRRetireRouter&>(m_epochDomain) は UB である。
        //   ISRRetireRouter と EpochDomain は共通基底 (IEpochProvider) を持つが
        //   直接の継承関係になく、メモリレイアウトが異なるため、
        //   enqueueRetire() 内の epochDomain_ メンバがガベージになる。
        //   正しい対策: スタック上に ISRRetireRouter を構築する。
        convo::isr::ISRRetireRouter stackRouter(m_epochDomain);
        auto result = m_retireCoordinator->enqueueRetire(
            convo::isr::RetireAuthority::Granted,
            stackRouter,
            ptr, deleter, retireEpoch);
        if (result == convo::isr::RetireEnqueueResult::Success)
            return true;

        // [P0-5] enqueue failure -> drop + telemetry (RT-safe).
        // Non-RT 側の定期的な reclaim が backlog を消化することを期待。
        return false;
    }

    // Fallback: direct EpochDomain path (backward compat before coordinator is set)
    // [Phase-B] coordinator 常時設定確認後、この経路は削除.
#pragma warning(push)
#pragma warning(disable : 4996)
    const bool ok = m_epochDomain.enqueueRetire(ptr, deleter,
        (epoch != 0) ? epoch : m_epochDomain.currentEpoch());
#pragma warning(pop)
    if (!ok)
    {
        // [P0-5] drop + telemetry. retry/reclaim/advanceEpoch は行わない.
        return false;
    }
    return true;
}
// [P1-14] 保留中の advanceEpoch を一括実行.
// パラメータ変更毎の advanceEpoch を遅延させ、本関数で1回に集約する.
void EQProcessor::flushPendingEpochAdvance() noexcept
{
    if (convo::exchangeAtomic(m_epochAdvancePending, false, std::memory_order_acq_rel))
    {
        m_epochDomain.publishEpoch();
    }
}
void EQProcessor::retireEQStateDeferred(EQState* state) noexcept
{
    if (state == nullptr)
        return;

    const uint64_t epoch = m_epochDomain.currentEpoch();
    enqueueDeferredDeleteWithFallback(state, deleteEQStatePtr, epoch);
}

void EQProcessor::retireBandNodeDeferred(BandNode* node) noexcept
{
    if (node == nullptr)
        return;

    const uint64_t epoch = m_epochDomain.currentEpoch();
    enqueueDeferredDeleteWithFallback(node, deleteBandNodePtr, epoch);
}

//============================================================================
// コンストラクタ
//============================================================================
EQProcessor::EQProcessor()
{
    juce::Logger::writeToLog("[EQ_CTOR] enter");
    // MSVC では std::atomic<uintptr_t> の default 初期化子 (= default) は
    // トリビアル型の場合はゼロ初期化しないため、aligned_malloc 経由で
    // DSPCore が構築されると bandNodeBits に未初期化ゴミが残る。
    // 明示的に全エントリをゼロクリアする。
    for (auto& b : bandNodeBits)
        convo::publishAtomic(b, static_cast<std::uintptr_t>(0), std::memory_order_relaxed);
    juce::Logger::writeToLog("[EQ_CTOR] bandNodeBits zeroed, calling resetToDefaults");

    // 初期係数ノードの作成
    resetToDefaults();
    juce::Logger::writeToLog("[EQ_CTOR] exit");
}

//============================================================================
// デストラクタ
//============================================================================
EQProcessor::~EQProcessor()
{
    juce::Logger::writeToLog("[DIAG EQProcessor] ~EQProcessor: enter");
    if (auto* oldState = exchangeCurrentState(nullptr, std::memory_order_acq_rel)) // acq_rel: acquire で先行 exchangeCurrentState/publishCurrentState と HB; release で後続観測者と HB
        retireEQStateDeferred(oldState);

    for (auto& nodeBits : bandNodeBits) {
        const auto bits = convo::exchangeAtomic(nodeBits, static_cast<std::uintptr_t>(0), std::memory_order_release); // release: デストラクタ後の観測者に対して null 書き込みを公知。acquire 不要 — デストラクタは排他的所有権を持つ
        if (auto* n = fromBandNodeBits(bits))
            retireBandNodeDeferred(n);
    }

    for (auto& node : activeBandNodes) {
        node = nullptr;
    }

    // 退役キューを強制 drain して可能な限り回収する。
    m_epochDomain.tryReclaim();
    m_epochDomain.drainAll();
    m_epochDomain.tryReclaim();

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
    // [P1-14] 保留中の advanceEpoch を一括実行
    flushPendingEpochAdvance();
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
    convo::publishAtomic(agcEnabled, false, std::memory_order_release);             // release: getAGCEnabled acquire と HB
    convo::publishAtomic(nonlinearSaturation, 0.2f, std::memory_order_release);     // release: getNonlinearSaturation acquire と HB
    convo::publishAtomic(requestedStructure, FilterStructure::Serial, std::memory_order_release); // release: getFilterStructure acquire と HB
    convo::publishAtomic(activeStructure, FilterStructure::Serial, std::memory_order_release);    // release: prepareToPlay の acquire と HB
    newState->agcEnabled = false;
    newState->nonlinearSaturation = 0.2f;
    newState->filterStructure = 0;

    auto oldState = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で先行 load と HB; release で後続 loadCurrentState acquire と HB

    if (oldState) {
        retireEQStateDeferred(oldState);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred

    convo::publishAtomic(agcCurrentGain, 1.0, std::memory_order_release); // release: Processing.cpp の acquire と HB し AGC 初期化を公知
    convo::publishAtomic(agcEnvInput, 0.0, std::memory_order_release);    // release: 同上
    convo::publishAtomic(agcEnvOutput, 0.0, std::memory_order_release);   // release: 同上

    // 全バンドの係数を更新
    for (int i = 0; i < NUM_BANDS; ++i)
        updateBandNode(i);

    // 全状態のリセットを予約
    requestAllBandReset();
    requestAgcReset();

    sendChangeMessage();
}

//============================================================================
// フィルタ状態リセット (Audio Thread)
//============================================================================
void EQProcessor::reset()
{
    // フィルタ状態をリセット (memsetで高速化)
    std::memset(filterState.data(), 0, sizeof(filterState));

    convo::publishAtomic(agcCurrentGain, 1.0, std::memory_order_release); // release: Processing.cpp の agcCurrentGain acquire と HB し初期値を公知
    convo::publishAtomic(agcEnvInput, 0.0, std::memory_order_release);    // release: Processing.cpp の acquire と HB
    convo::publishAtomic(agcEnvOutput, 0.0, std::memory_order_release);   // release: Processing.cpp の acquire と HB

    auto state = loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (state)
    {
        smoothTotalGain.setCurrentAndTargetValue(juce::Decibels::decibelsToGain<double>(static_cast<double>(state->totalGainDb)));
        storeTotalGainDb(state->totalGainDb);
    }

    convo::publishAtomic(bandResetPacked, static_cast<std::uint64_t>(0), std::memory_order_release); // release: Processing.cpp の bandResetPacked acquire と HB しリセット完了を公知
    convo::publishAtomic(agcResetSerial, static_cast<std::uint64_t>(0), std::memory_order_release);  // release: Processing.cpp の agcResetSerial acquire と HB
    rtDeferredBandResetMask = 0;
    rtSeenBandResetSerial = 0;
    rtSeenAgcResetSerial = 0;

    const bool requestedBypass = convo::consumeAtomic(bypassRequested, std::memory_order_acquire); // acquire: setBypass の publishAtomic release と HB
    convo::publishAtomic(bypassed, requestedBypass, std::memory_order_release);                    // release: Processing.cpp の bypassed acquire と HB
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

    requestAllBandReset();
    sendChangeMessage();
    return true;
}

//============================================================================
// 状態取得
//============================================================================
juce::ValueTree EQProcessor::getState() const
{
    auto state = loadCurrentState(std::memory_order_acquire); // acquire: publishCurrentState/exchangeCurrentState の release/acq_rel と HB
    if (state == nullptr) return juce::ValueTree("EQ");

    juce::ValueTree v ("EQ");
    v.setProperty ("totalGain", state->totalGainDb, nullptr);
    v.setProperty("agcEnabled", convo::consumeAtomic(agcEnabled, std::memory_order_acquire), nullptr);                    // acquire: setAGCEnabled の publishAtomic release と HB
    v.setProperty("nonlinearSaturation", convo::consumeAtomic(nonlinearSaturation, std::memory_order_acquire), nullptr);  // acquire: setNonlinearSaturation の publishAtomic release と HB
    v.setProperty("filterStructure",
                  static_cast<int>(convo::consumeAtomic(requestedStructure, std::memory_order_acquire)), // acquire: setFilterStructure の publishAtomic release と HB
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
    requestAllBandReset();
    requestAgcReset();
    convo::publishAtomic(activeStructure,
                         convo::consumeAtomic(requestedStructure, std::memory_order_acquire), // acquire: setFilterStructure の release と HB
                         std::memory_order_release); // release: prepareToPlay/Processing の activeStructure acquire と HB

    sendChangeMessage();
}

//============================================================================
// 状態同期（他のプロセッサから）
//============================================================================
void EQProcessor::syncStateFrom(const EQProcessor& other)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());
    auto otherState = other.loadCurrentState(std::memory_order_acquire); // acquire: other の exchangeCurrentState/publishCurrentState と HB
    if (otherState == nullptr)
        return;

    auto* clonedState = new EQState(*otherState);
    auto oldState = exchangeCurrentState(clonedState, std::memory_order_acq_rel); // acq_rel: acquire で先行 load と HB; release で後続 loadCurrentState acquire と HB

    if (oldState)
    {
        retireEQStateDeferred(oldState);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred

    for (int i = 0; i < NUM_BANDS; ++i)
        updateBandNode(i);

    const double syncedAgcCurrentGain = convo::consumeAtomic(other.agcCurrentGain, std::memory_order_acquire); // acquire: other の publishAtomic release と HB
    const double syncedAgcEnvInput = convo::consumeAtomic(other.agcEnvInput, std::memory_order_acquire);       // acquire: 同上
    const double syncedAgcEnvOutput = convo::consumeAtomic(other.agcEnvOutput, std::memory_order_acquire);     // acquire: 同上
    const FilterStructure syncedStructure = static_cast<FilterStructure>(clonedState->filterStructure);
    const bool syncedBypassed = convo::consumeAtomic(other.bypassed, std::memory_order_acquire);                      // acquire: other の bypassed publishAtomic release と HB
    const std::uint64_t syncedBandResetPacked = convo::consumeAtomic(other.bandResetPacked, std::memory_order_acquire); // acquire: other の requestBandReset/publishAtomic acq_rel/release と HB
    const std::uint32_t syncedBandResetMask = bandResetMaskFromPacked(syncedBandResetPacked);
    const std::uint64_t syncedBandResetSerial = static_cast<std::uint64_t>(bandResetSerialFromPacked(syncedBandResetPacked));
    const std::uint64_t syncedAgcResetSerial = convo::consumeAtomic(other.agcResetSerial, std::memory_order_acquire); // acquire: other の requestAgcReset acq_rel と HB

    // Avoid publication-domain split here: sync uses immutable state swap plus RT-local shadow updates.
    bypassFadeGain.setCurrentAndTargetValue(syncedBypassed ? 0.0 : 1.0);
    smoothTotalGain.setCurrentAndTargetValue(
        juce::Decibels::decibelsToGain<double>(static_cast<double>(clonedState->totalGainDb)));

    rtBypassedShadow = syncedBypassed;
    rtActiveStructureShadow = syncedStructure;
    rtAgcCurrentGainShadow = syncedAgcCurrentGain;
    rtAgcEnvInputShadow = syncedAgcEnvInput;
    rtAgcEnvOutputShadow = syncedAgcEnvOutput;

    rtDeferredBandResetMask = syncedBandResetMask;
    rtSeenBandResetSerial = syncedBandResetSerial;
    rtSeenAgcResetSerial = syncedAgcResetSerial;

}

//============================================================================
// 単一バンド同期
//============================================================================
void EQProcessor::syncBandNodeFrom(const EQProcessor& other, int bandIndex)
{
    jassert (juce::MessageManager::getInstance()->isThisTheMessageThread());

    if (bandIndex < 0 || bandIndex >= NUM_BANDS) return;

    const auto* otherState = other.loadCurrentState(std::memory_order_acquire); // acquire: other の exchangeCurrentState/publishCurrentState と HB
    if (otherState == nullptr)
        return;

    auto* newNode = createBandNode(bandIndex, *otherState);
    auto* oldNode = exchangeBandNode(bandIndex, newNode, std::memory_order_acq_rel); // acq_rel: acquire で先行 load と HB; release で後続 loadBandNode acquire と HB

    activeBandNodes[bandIndex] = newNode;

    if (oldNode)
        retireBandNodeDeferred(oldNode);

    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
}

//============================================================================
// グローバル状態同期 (Worker Threadからも安全)
//============================================================================
void EQProcessor::syncGlobalStateFrom(const EQProcessor& other)
{
    const auto* otherState = other.loadCurrentState(std::memory_order_acquire); // acquire: other の exchangeCurrentState/publishCurrentState と HB
    const double syncedAgcCurrentGain = convo::consumeAtomic(other.agcCurrentGain, std::memory_order_acquire); // acquire: other の publishAtomic release と HB
    const double syncedAgcEnvInput = convo::consumeAtomic(other.agcEnvInput, std::memory_order_acquire);       // acquire: 同上
    const double syncedAgcEnvOutput = convo::consumeAtomic(other.agcEnvOutput, std::memory_order_acquire);     // acquire: 同上
    const FilterStructure syncedStructure = (otherState != nullptr)
        ? static_cast<FilterStructure>(otherState->filterStructure)
        : convo::consumeAtomic(other.requestedStructure, std::memory_order_acquire); // acquire: setFilterStructure の release と HB (フォールバック)
    const bool syncedBypassed = convo::consumeAtomic(other.bypassed, std::memory_order_acquire);                      // acquire: other の bypassed publishAtomic release と HB
    const std::uint64_t syncedBandResetPacked = convo::consumeAtomic(other.bandResetPacked, std::memory_order_acquire); // acquire: other の requestBandReset acq_rel と HB
    const std::uint32_t syncedBandResetMask = bandResetMaskFromPacked(syncedBandResetPacked);
    const std::uint64_t syncedBandResetSerial = static_cast<std::uint64_t>(bandResetSerialFromPacked(syncedBandResetPacked));
    const std::uint64_t syncedAgcResetSerial = convo::consumeAtomic(other.agcResetSerial, std::memory_order_acquire); // acquire: other の requestAgcReset acq_rel と HB

    // Keep sync as shadow-state update to prevent multi-atomic partial visibility.
    bypassFadeGain.setCurrentAndTargetValue(syncedBypassed ? 0.0 : 1.0);

    const float syncedTotalGainDb = (otherState != nullptr)
        ? otherState->totalGainDb
        : convo::consumeAtomic(other.totalGainDbTarget, std::memory_order_acquire); // acquire: other.storeTotalGainDb の publishAtomic release と HB
    smoothTotalGain.setCurrentAndTargetValue(
        juce::Decibels::decibelsToGain<double>(static_cast<double>(syncedTotalGainDb)));

    rtBypassedShadow = syncedBypassed;
    rtActiveStructureShadow = syncedStructure;
    rtAgcCurrentGainShadow = syncedAgcCurrentGain;
    rtAgcEnvInputShadow = syncedAgcEnvInput;
    rtAgcEnvOutputShadow = syncedAgcEnvOutput;

    rtDeferredBandResetMask = syncedBandResetMask;
    rtSeenBandResetSerial = syncedBandResetSerial;
    rtSeenAgcResetSerial = syncedAgcResetSerial;

}

//============================================================================
// メモリ事前確保 & 係数再計算
//============================================================================
void EQProcessor::prepareToPlay(double sampleRate, int newMaxInternalBlockSize)
{
    juce::Logger::writeToLog("[EQ_PREPARE] enter: sr=" + juce::String(sampleRate) + " block=" + juce::String(newMaxInternalBlockSize));

    const bool rateChanged = (std::abs(convo::consumeAtomic(currentSampleRate, std::memory_order_acquire) - sampleRate) > 1e-6); // acquire: setSampleRate の publishAtomic release と HB
    juce::Logger::writeToLog("[EQ_PREPARE] rateChanged=" + juce::String(static_cast<int>(rateChanged)));
    if (rateChanged)
        convo::publishAtomic(currentSampleRate, sampleRate, std::memory_order_release); // release: 次回 prepareToPlay/setSampleRate の acquire と HB

    const int requiredSize = newMaxInternalBlockSize;
    this->maxInternalBlockSize = requiredSize;

    const int required = juce::nextPowerOfTwo(requiredSize) * 8;
    juce::Logger::writeToLog("[EQ_PREPARE] scratch: required=" + juce::String(required) + " capacity=" + juce::String(scratchCapacity));
    if (scratchCapacity < required)
    {
        scratchBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(required));
        scratchCapacity = required;
        juce::FloatVectorOperations::clear(scratchBuffer.get(), required);
        juce::Logger::writeToLog("[EQ_PREPARE] scratch allocated");
    }

    const int channelRequired = juce::nextPowerOfTwo(requiredSize) * MAX_CHANNELS;
    juce::Logger::writeToLog("[EQ_PREPARE] channel: required=" + juce::String(channelRequired) + " dryCap=" + juce::String(dryBypassCapacity));
    if (dryBypassCapacity < channelRequired)
    {
        dryBypassBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(channelRequired));
        dryBypassCapacity = channelRequired;
        juce::FloatVectorOperations::clear(dryBypassBuffer.get(), channelRequired);
        juce::Logger::writeToLog("[EQ_PREPARE] dryBypass allocated");
    }

    if (parallelBufferCapacity < channelRequired)
    {
        parallelInputBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(channelRequired));
        parallelWorkBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(channelRequired));
        parallelAccumBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(channelRequired));
        parallelBufferCapacity = channelRequired;
        juce::FloatVectorOperations::clear(parallelInputBuffer.get(), channelRequired);
        juce::FloatVectorOperations::clear(parallelWorkBuffer.get(), channelRequired);
        juce::FloatVectorOperations::clear(parallelAccumBuffer.get(), channelRequired);
        juce::Logger::writeToLog("[EQ_PREPARE] parallel buffers allocated");
    }

    if (structureXfadeBufferCapacity < channelRequired)
    {
        structureOldOutBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(channelRequired));
        structureNewOutBuffer = convo::makeAlignedArray<double>(static_cast<size_t>(channelRequired));
        structureXfadeBufferCapacity = channelRequired;
        juce::FloatVectorOperations::clear(structureOldOutBuffer.get(), channelRequired);
        juce::FloatVectorOperations::clear(structureNewOutBuffer.get(), channelRequired);
        juce::Logger::writeToLog("[EQ_PREPARE] xfade buffers allocated");
    }

    if (newMaxInternalBlockSize > 0 && agcCoeffTableCapacity < (newMaxInternalBlockSize + 1))
    {
        agcAttackCoeffTable = convo::makeAlignedArray<double>(static_cast<size_t>(newMaxInternalBlockSize + 1));
        agcReleaseCoeffTable = convo::makeAlignedArray<double>(static_cast<size_t>(newMaxInternalBlockSize + 1));
        agcSmoothCoeffTable = convo::makeAlignedArray<double>(static_cast<size_t>(newMaxInternalBlockSize + 1));
        juce::Logger::writeToLog("[EQ_PREPARE] agc tables allocated");

        if (agcAttackCoeffTable && agcReleaseCoeffTable && agcSmoothCoeffTable)
            agcCoeffTableCapacity = newMaxInternalBlockSize + 1;
        else
            agcCoeffTableCapacity = 0;
    }

    auto state = loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB

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
    convo::publishAtomic(agcAttackCoeff,  std::exp(-1.0 / (sr * AGC_ATTACK_TIME_SEC)),  std::memory_order_release); // release: Processing.cpp の agcAttackCoeff acquire と HB
    convo::publishAtomic(agcReleaseCoeff, std::exp(-1.0 / (sr * AGC_RELEASE_TIME_SEC)), std::memory_order_release); // release: Processing.cpp の agcReleaseCoeff acquire と HB
    convo::publishAtomic(agcSmoothCoeff,  std::exp(-1.0 / (sr * AGC_SMOOTH_TIME_SEC)),  std::memory_order_release); // release: Processing.cpp の agcSmoothCoeff acquire と HB

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

    convo::publishAtomic(agcCurrentGain, 1.0, std::memory_order_release); // release: Processing.cpp の agcCurrentGain acquire と HB
    convo::publishAtomic(agcEnvInput, 0.0, std::memory_order_release);    // release: Processing.cpp の acquire と HB
    convo::publishAtomic(agcEnvOutput, 0.0, std::memory_order_release);   // release: Processing.cpp の acquire と HB

    convo::publishAtomic(bandResetPacked, static_cast<std::uint64_t>(0), std::memory_order_release);  // release: Processing.cpp の bandResetPacked acquire と HB
    convo::publishAtomic(agcResetSerial, static_cast<std::uint64_t>(0), std::memory_order_release);   // release: Processing.cpp の agcResetSerial acquire と HB
    rtDeferredBandResetMask = 0;
    rtSeenBandResetSerial = 0;
    rtSeenAgcResetSerial = 0;
    convo::publishAtomic(activeStructure,
                         convo::consumeAtomic(requestedStructure, std::memory_order_acquire), // acquire: setFilterStructure の release と HB
                         std::memory_order_release); // release: Processing.cpp の activeStructure acquire と HB

    const bool requestedBypass = convo::consumeAtomic(bypassRequested, std::memory_order_acquire); // acquire: setBypass の publishAtomic release と HB
    convo::publishAtomic(bypassed, requestedBypass, std::memory_order_release);                    // release: Processing.cpp の bypassed acquire と HB
    bypassFadeGain.setCurrentAndTargetValue(requestedBypass ? 0.0 : 1.0);

    if (rateChanged)
    {
        for (int i = 0; i < NUM_BANDS; ++i)
        {
            auto loopState = loadCurrentState(std::memory_order_acquire); // acquire: 直前の exchangeCurrentState acq_rel と HB
            if (loopState)
            {
                auto newNode = createBandNode(i, *loopState);
                auto oldNode = exchangeBandNode(i, newNode, std::memory_order_acq_rel); // acq_rel: acquire で先行 load と HB; release で後続 loadBandNode acquire と HB
                if (oldNode)
                    retireBandNodeDeferred(oldNode);
                activeBandNodes[i] = newNode;
            }
        }
    }

    // [P1-14] 保留中の advanceEpoch を一括実行
    flushPendingEpochAdvance();

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
    return loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB
}

float EQProcessor::getTotalGain() const
{
    auto state = loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (state == nullptr) return 0.0f;
    return state->totalGainDb;
}

EQBandType EQProcessor::getBandType(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQBandType::Peaking;
    auto state = loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (state == nullptr) return EQBandType::Peaking;
    return state->bandTypes[band];
}

EQChannelMode EQProcessor::getBandChannelMode(int band) const
{
    if (band < 0 || band >= NUM_BANDS) return EQChannelMode::Stereo;
    auto state = loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB
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
    auto state = loadCurrentState(std::memory_order_acquire); // acquire: exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (state == nullptr) return {};
    return state->bands[band];
}
