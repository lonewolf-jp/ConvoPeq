//============================================================================
// EQProcessor.Parameters.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// パラメータセッター・ゲッター
//============================================================================
#include "EQProcessor.h"
#include "core/EpochManager.h"

static void retireEQState(EQProcessor::EQState* state)
{
    if (state) state->release();
}

//============================================================================
// パラメータ変更メソッド (UIスレッドから呼ぶ)
// 各メソッドは atomic store で値を書き込み、coeffsDirty を立てる
//============================================================================

//--------------------------------------------------------------
// 周波数変更
//--------------------------------------------------------------
void EQProcessor::setBandFrequency(int band, float freq)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].frequency = freq;

    auto prev = currentStateRaw.exchange(newState, std::memory_order_acq_rel);
    if (prev) {
        retireEQState(prev);
    }
    convo::EpochManager::instance().advanceEpoch();
    updateBandNode(band);
}

//--------------------------------------------------------------
// ゲイン変更
//--------------------------------------------------------------
void EQProcessor::setBandGain(int band, float gainDb)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].gain = gainDb;

    auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
    if (prev) {
        retireEQState(prev);
    }
    convo::EpochManager::instance().advanceEpoch();
    updateBandNode(band);
}

//--------------------------------------------------------------
// Q値変更
//--------------------------------------------------------------
void EQProcessor::setBandQ(int band, float q)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].q = q;

    auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
    if (prev) {
        retireEQState(prev);
    }
    convo::EpochManager::instance().advanceEpoch();
    updateBandNode(band);
}

//--------------------------------------------------------------
// バンド有効化
//--------------------------------------------------------------
void EQProcessor::setBandEnabled(int band, bool enabled)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].enabled = enabled;

    // 有効化時はステートをリセットして、古い状態（ポップノイズの原因）を排除する
    if (enabled)
        bandResetMask.fetch_or(1 << band, std::memory_order_relaxed);

    auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
    if (prev) {
        retireEQState(prev);
    }
    convo::EpochManager::instance().advanceEpoch();
    updateBandNode(band);
}

//--------------------------------------------------------------
// 全体ゲイン変更
//--------------------------------------------------------------
void EQProcessor::setTotalGain(float gainDb)
{
    // パラメータを安全な範囲にクランプ
    gainDb = juce::jlimit(DSP_MIN_GAIN_DB, DSP_MAX_GAIN_DB, gainDb);

    // ✅ Atomicに保存（Audio Threadで読み取る）
    storeTotalGainDb(gainDb);

    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->totalGainDb = gainDb;

    auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
    if (prev) {
        retireEQState(prev);
    }
    convo::EpochManager::instance().advanceEpoch();
}

//--------------------------------------------------------------
// AGC有効化
//--------------------------------------------------------------
void EQProcessor::setAGCEnabled(bool enabled)
{
    agcEnabled.store(enabled, std::memory_order_release);
    m_pendingAGCChange.store(true, std::memory_order_release);

    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState != nullptr)
    {
        auto newState = new EQState(*oldState);
        newState->agcEnabled = enabled;
        auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
        if (prev)
            retireEQState(prev);
        convo::EpochManager::instance().advanceEpoch();
    }

    if (enabled)
        agcResetRequest.store(true, std::memory_order_relaxed);
}

//--------------------------------------------------------------
// AGC状態取得
//--------------------------------------------------------------
bool EQProcessor::getAGCEnabled() const
{
    return agcEnabled.load(std::memory_order_acquire);
}

//--------------------------------------------------------------
// フィルタタイプ変更
//--------------------------------------------------------------
void EQProcessor::setBandType(int band, EQBandType type)
{
    if (band < 0 || band >= NUM_BANDS) return;

    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bandTypes[band] = type;

    // フィルタタイプ変更時はトポロジーが変わるためリセット必須
    bandResetMask.fetch_or(1 << band, std::memory_order_relaxed);

    auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
    if (prev) {
        retireEQState(prev);
    }
    convo::EpochManager::instance().advanceEpoch();
    updateBandNode(band);
}

//--------------------------------------------------------------
// チャンネルモード変更
//--------------------------------------------------------------
void EQProcessor::setBandChannelMode(int band, EQChannelMode mode)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bandChannelModes[band] = mode;

    // チャンネルモード変更時もリセット推奨
    bandResetMask.fetch_or(1 << band, std::memory_order_relaxed);

    auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
    if (prev) {
        retireEQState(prev);
    }
    convo::EpochManager::instance().advanceEpoch();
    updateBandNode(band);
}

//--------------------------------------------------------------
// 非線形飽和度設定
//--------------------------------------------------------------
void EQProcessor::setNonlinearSaturation(float value) noexcept
{
    const float clamped = juce::jlimit(0.0f, 1.0f, value);
    const float previous = nonlinearSaturation.load(std::memory_order_relaxed);
    if (std::abs(static_cast<double>(clamped - previous)) < 1.0e-9)
        return;

    nonlinearSaturation.store(clamped, std::memory_order_relaxed);

    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState != nullptr)
    {
        auto newState = new EQState(*oldState);
        newState->nonlinearSaturation = clamped;
        auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
        if (prev)
            retireEQState(prev);
        convo::EpochManager::instance().advanceEpoch();
    }
}

//--------------------------------------------------------------
// 非線形飽和度取得
//--------------------------------------------------------------
float EQProcessor::getNonlinearSaturation() const noexcept
{
    return nonlinearSaturation.load(std::memory_order_relaxed);
}

//--------------------------------------------------------------
// フィルタ構造設定 (Serial/Parallel)
//--------------------------------------------------------------
void EQProcessor::setFilterStructure(FilterStructure mode) noexcept
{
    if (mode != FilterStructure::Serial && mode != FilterStructure::Parallel)
        mode = FilterStructure::Serial;

    const FilterStructure previous = requestedStructure.load(std::memory_order_relaxed);
    if (previous == mode)
        return;

    requestedStructure.store(mode, std::memory_order_relaxed);

    auto oldState = currentStateRaw.load(std::memory_order_acquire);
    if (oldState != nullptr)
    {
        auto newState = new EQState(*oldState);
        newState->filterStructure = (mode == FilterStructure::Parallel) ? 1 : 0;
        auto prev = currentStateRaw.exchange(newState, std::memory_order_release);
        if (prev)
            retireEQState(prev);
        convo::EpochManager::instance().advanceEpoch();
    }
}

//--------------------------------------------------------------
// フィルタ構造取得
//--------------------------------------------------------------
EQProcessor::FilterStructure EQProcessor::getFilterStructure() const noexcept
{
    return requestedStructure.load(std::memory_order_relaxed);
}
