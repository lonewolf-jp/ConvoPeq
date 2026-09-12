// NUPCTestAccess.h
// Gardner Null Test v2.9 (Step 3) 専用の Friend Test Access。
//
// MKLNonUniformConvolver は private メンバ（m_layers / m_numActiveLayers / m_tailLayerGain）を
// 持つため、テストから到達するには friend 宣言（MKLNonUniformConvolver.h private: 直後）が必要。
// 本クラスは DeferredPublicationTestAccess.h と同じ friend + static accessor パターンを踏襲する。
//
// ★ Practical Stable ISR Bridge Runtime の Observer 原則:
//   - Observer は metrics / logging / telemetry のみ許可
//   - publish / retire / crossfade 変更 / ownership 取得 / 状態変更 は禁止
//   → 本クラスは**純粋な読み取り getter のみ**を提供する。
//     setter・cursor 変更・sync・reset・publish・retire は一切提供しない。

#pragma once

#include <cstdint>

#include "MKLNonUniformConvolver.h"

namespace convo
{

struct NUPCTestAccess final
{
    // ── topology ──
    static int numActiveLayers (const MKLNonUniformConvolver& c) noexcept { return c.m_numActiveLayers; }
    static int layerPartSize        (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].partSize; }
    static int layerNumPartsIR      (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].numPartsIR; }
    static int layerNumParts        (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].numParts; }
    static int layerPartsPerCallback(const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].partsPerCallback; }
    static bool layerIsImmediate    (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].isImmediate; }

    // ── B13 遅延補償構成（測定対象: assert してはならない） ──
    static int layerOutputDelaySamples (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].outputDelaySamples; }
    static int layerDelayLineCapacity  (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].delayLineCapacity; }

    // ── tail gain（M1 reference 用の実測値） ──
    static double layerTailGain (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_tailLayerGain[li]; }
    static bool   tailEnabled   (const MKLNonUniformConvolver& c) noexcept { return c.m_tailEnabled; }

    // ── B13 cursor（write / read-anchor event 観測用） ──
    static std::uint64_t layerDelayWriteCursor (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].delayWriteCursor; }
    static std::uint64_t layerDelayReadCursor  (const MKLNonUniformConvolver& c, int li) noexcept { return c.m_layers[li].delayReadCursor; }

    // ── B13 Policy R（stream clock / I2 safety counter — 読み取りのみ） ──
    // ★ ISR Bridge 原則: atomic は wrapper 経由のみ（raw .load()/.store() 不使用）。
    static std::uint64_t outputSamplesProcessed (const MKLNonUniformConvolver& c) noexcept { return c.m_outputSamplesProcessed; }
    static std::uint32_t delayI2ViolationCount  (const MKLNonUniformConvolver& c) noexcept { return convo::consumeAtomic (c.m_delayI2ViolationCount, std::memory_order_acquire); }
};

} // namespace convo
