//============================================================================
// EQProcessor.Parameters.cpp  ── v0.2 (JUCE 8.0.12対応)
//
// パラメータセッター・ゲッター
//============================================================================
#include "EQProcessor.h"
#include "core/EpochDomain.h"

#include "audioengine/AtomicAccess.h"

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
    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].frequency = freq;

    auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
    if (prev) {
        retireEQStateDeferred(prev);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    updateBandNode(band);
}

//--------------------------------------------------------------
// ゲイン変更
//--------------------------------------------------------------
void EQProcessor::setBandGain(int band, float gainDb)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].gain = gainDb;

    auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
    if (prev) {
        retireEQStateDeferred(prev);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    updateBandNode(band);
}

//--------------------------------------------------------------
// Q値変更
//--------------------------------------------------------------
void EQProcessor::setBandQ(int band, float q)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].q = q;

    auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
    if (prev) {
        retireEQStateDeferred(prev);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    updateBandNode(band);
}

//--------------------------------------------------------------
// バンド有効化
//--------------------------------------------------------------
void EQProcessor::setBandEnabled(int band, bool enabled)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bands[band].enabled = enabled;

    // 有効化時はステートをリセットして、古い状態（ポップノイズの原因）を排除する
    if (enabled)
        requestBandReset(static_cast<uint32_t>(1u << band));

    auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
    if (prev) {
        retireEQStateDeferred(prev);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
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

    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->totalGainDb = gainDb;

    auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
    if (prev) {
        retireEQStateDeferred(prev);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
}

//--------------------------------------------------------------
// AGC有効化
//--------------------------------------------------------------
void EQProcessor::setAGCEnabled(bool enabled)
{
    convo::publishAtomic(agcEnabled, enabled, std::memory_order_release);        // release: getAGCEnabled/Processing 側の acquire と HB
    convo::publishAtomic(m_pendingAGCChange, true, std::memory_order_release);   // release: getAndClearPendingAGCChange の acq_rel acquire と HB

    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState != nullptr)
    {
        auto newState = new EQState(*oldState);
        newState->agcEnabled = enabled;
        auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
        if (prev)
            retireEQStateDeferred(prev);
        convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    }

    if (enabled)
        requestAgcReset();
}

//--------------------------------------------------------------
// AGC状態取得
//--------------------------------------------------------------
bool EQProcessor::getAGCEnabled() const
{
    return convo::consumeAtomic(agcEnabled, std::memory_order_acquire); // acquire: setAGCEnabled の publishAtomic release と HB
}

//--------------------------------------------------------------
// フィルタタイプ変更
//--------------------------------------------------------------
void EQProcessor::setBandType(int band, EQBandType type)
{
    if (band < 0 || band >= NUM_BANDS) return;

    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bandTypes[band] = type;

    // フィルタタイプ変更時はトポロジーが変わるためリセット必須
    requestBandReset(static_cast<uint32_t>(1u << band));

    auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
    if (prev) {
        retireEQStateDeferred(prev);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    updateBandNode(band);
}

//--------------------------------------------------------------
// チャンネルモード変更
//--------------------------------------------------------------
void EQProcessor::setBandChannelMode(int band, EQChannelMode mode)
{
    if (band < 0 || band >= NUM_BANDS) return;
    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState == nullptr) return;
    auto newState = new EQState(*oldState);
    newState->bandChannelModes[band] = mode;

    // チャンネルモード変更時もリセット推奨
    requestBandReset(static_cast<uint32_t>(1u << band));

    auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
    if (prev) {
        retireEQStateDeferred(prev);
    }
    convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    updateBandNode(band);
}

//--------------------------------------------------------------
// 非線形飽和度設定
//--------------------------------------------------------------
void EQProcessor::setNonlinearSaturation(float value) noexcept
{
    const float clamped = juce::jlimit(0.0f, 1.0f, value);
    const float previous = convo::consumeAtomic(nonlinearSaturation, std::memory_order_acquire); // acquire: 先行 publishAtomic release と HB
    if (std::abs(static_cast<double>(clamped - previous)) < 1.0e-9)
        return;

    convo::publishAtomic(nonlinearSaturation, clamped, std::memory_order_release); // release: getNonlinearSaturation/getState の acquire と HB

    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState != nullptr)
    {
        auto newState = new EQState(*oldState);
        newState->nonlinearSaturation = clamped;
        auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
        if (prev)
            retireEQStateDeferred(prev);
        convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    }
}

//--------------------------------------------------------------
// 非線形飽和度取得
//--------------------------------------------------------------
float EQProcessor::getNonlinearSaturation() const noexcept
{
    return convo::consumeAtomic(nonlinearSaturation, std::memory_order_acquire); // acquire: setNonlinearSaturation の publishAtomic release と HB
}

//--------------------------------------------------------------
// フィルタ構造設定 (Serial/Parallel)
//--------------------------------------------------------------
void EQProcessor::setFilterStructure(FilterStructure mode) noexcept
{
    if (mode != FilterStructure::Serial && mode != FilterStructure::Parallel)
        mode = FilterStructure::Serial;

    const FilterStructure previous = convo::consumeAtomic(requestedStructure, std::memory_order_acquire); // acquire: 先行 publishAtomic release と HB
    if (previous == mode)
        return;

    convo::publishAtomic(requestedStructure, mode, std::memory_order_release); // release: getFilterStructure/getState の acquire と HB

    auto oldState = loadCurrentState(std::memory_order_acquire); // acquire: 先行 exchangeCurrentState/publishCurrentState の release/acq_rel と HB
    if (oldState != nullptr)
    {
        auto newState = new EQState(*oldState);
        newState->filterStructure = (mode == FilterStructure::Parallel) ? 1 : 0;
        auto prev = exchangeCurrentState(newState, std::memory_order_acq_rel); // acq_rel: acquire で旧状態読取と HB; release で後続 loadCurrentState acquire と HB
        if (prev)
            retireEQStateDeferred(prev);
        convo::publishAtomic(m_epochAdvancePending, true, std::memory_order_release); // [P1-14] deferred
    }
}

//--------------------------------------------------------------
// フィルタ構造取得
//--------------------------------------------------------------
EQProcessor::FilterStructure EQProcessor::getFilterStructure() const noexcept
{
    return convo::consumeAtomic(requestedStructure, std::memory_order_acquire); // acquire: setFilterStructure の publishAtomic release と HB
}
