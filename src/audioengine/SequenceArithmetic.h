//============================================================================
// SequenceArithmetic.h — dash2 §1.6.1 (Phase H) modular sequence arithmetic
//
// PublicationSequenceId / PublicationEpoch は uint64_t の単調増加カウンタ。
// 単純な `a < b` 比較は wraparound（2^64 値到達時）で壊れるため、modulo 2^64
// の比較で仕様化する（REPAIR_PLAN2-dash2.md §1.6.1 / Appendix E）。
//
//   正規化: dist(a, b) = (b - a) mod 2^64（unsigned wrap で定義）
//   isBefore(a, b)     : a は b より真に前（b が a から forward half 内）
//   isAfter(a, b)      : a は b より真に後
//   isAtOrBefore(a, b) : a は b と等しいか前（modular <=）
//   isCompleted(seq, watermark): seq は watermark に到達済み（isAtOrBefore(seq, watermark)）
//
// ★ プラン式との関係（off-by-one 補正）:
//   プラン §1.6.1 は isBefore(a,b) = ((b - a) < UINT64_MAX / 2) と定義する。
//   この式は (1) a == b も true にする（at-or-before 意味論）(2) UINT64_MAX が
//   奇数であるため境界で off-by-one（dist == 2^63-1 が before に入らない）。
//   本ヘッダでは意味論を厳密化し、
//     - isBefore を strict（a != b を含む）とし、
//     - 等値含む「到達済み」は isAtOrBefore / isCompleted に分離、
//     - しきい値は RFC 1982 serial number arithmetic と同一の `< 2^63`（= modulo
//       2^64 のちょうど半円。antipode のみ曖昧）とした。
//   非 wrap 値（seq/epoch は 1 ずつ増加・差分が 2^63 に達することはない）では
//   いずれも `</<=` と完全に等価（semantics-preserving hardening — Appendix E）。
//
// ■ 適用 semantic domain（2026-08-15 verify-before-implement 確認済み）:
//   本 primitive 群は「単調 serial カウンタ（RFC 1982 serial number）の比較」を仕様化する。
//   「型（uint64_t）が同じだから比較規則も同じ」ではなく、**両適用箇所が同一 semantic
//   domain であることをデータフローで確認**した上で共有している:
//
//   1. PublishReceiptWaiter（completion watermark）:
//      completion seqId は publication sequenceId そのもの（同一カウンタ）。
//      commitRuntimePublication() → seqId = world->publication.sequenceId
//      → intent.sequenceId → executePublish → onPublishCommitted(intent.sequenceId)
//      → notifyPublishReceipt → complete(seqId)。
//   2. RuntimeIntentCoordinator::commit()（monotonicity）:
//      新 publish の publication.sequenceId / publication.epoch を直前 published world
//      のそれらと比較。sequenceId は上記と同一カウンタ。epoch は並列の単調 serial
//      （publish ごとに同時に増加）で同一の serial-number semantics。
//
//   したがって両サイトは同じ「monotonic serial の比較」semantics を要求する。ただし
//   **操作ごとに正しい primitive を使い分ける**:
//     - watermark 進行 / monotonicity 検証 = isAfter（strict 増加）: complete()・commit()
//     - waitFor の完了判定 = isCompleted（at-or-before）: waitFor()
//   この使い分けは「同じ counter でも、要求する関係（strict 増加 vs 到達済み）が異なる」
//   ことを反映しており、単純な `>/<=` の置き換えではない。
//
// ■ 使用箇所:
//   - PublishReceiptWaiter::complete / waitFor（AudioEngine.h）: watermark 比較
//   - RuntimeIntentCoordinator::commit() monotonicity（ISRRuntimePublicationCoordinator.cpp）:
//     sequenceId / epoch 比較
//
// ■ sparse completion（§1.5 completedThrough_ + completedOutOfOrder_）は
//   MPSC completion / parallel publish 許容時のみ必要。現状は PublishExecutor sole
//   gateway + FIFO のため INV-X2-6（contiguous completion）を維持する — 本ヘッダは
//   その前提を壊さない（現状は実装不要・将来保留）。
//============================================================================
#pragma once

#include <cstdint>

namespace convo::isr {

// modulo 2^64 のちょうど半円（= 2^63）。isBefore/isAfter のしきい値。
// RFC 1982 serial number arithmetic と同一の定義（antipode はどちらでもない）。
inline constexpr std::uint64_t kSeqHalfModulus = std::uint64_t{1} << 63;

// Modular forward distance: (b - a) mod 2^64（unsigned wrap で定義）
[[nodiscard]] constexpr std::uint64_t seqDistance(std::uint64_t a, std::uint64_t b) noexcept
{
    return b - a;
}

// a は b より真に前（strict before）。
//   true  ⇔  a != b かつ dist(a,b) < 2^63（b は a から forward half 内）
//   偽    ⇔  a == b（等値は前でも後でもない）または dist(a,b) >= 2^63（antipode 含む）
[[nodiscard]] constexpr bool isBefore(std::uint64_t a, std::uint64_t b) noexcept
{
    return a != b && seqDistance(a, b) < kSeqHalfModulus;
}

// a は b より真に後（strict after）。
[[nodiscard]] constexpr bool isAfter(std::uint64_t a, std::uint64_t b) noexcept
{
    return isBefore(b, a);
}

// a は b と等しいか前（modular at-or-before / <=）。
[[nodiscard]] constexpr bool isAtOrBefore(std::uint64_t a, std::uint64_t b) noexcept
{
    return a == b || isBefore(a, b);
}

// seq が完成した（watermark が seq に到達済み）。PublishReceiptWaiter::waitFor の述語。
[[nodiscard]] constexpr bool isCompleted(std::uint64_t seq, std::uint64_t watermark) noexcept
{
    return isAtOrBefore(seq, watermark);
}

} // namespace convo::isr
