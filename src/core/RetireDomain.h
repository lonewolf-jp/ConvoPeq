#pragma once

//==============================================================================
// RetireDomain.h — 退役ドメインの識別子（domain tag 設計 v2.9 §2.2 / §2.3）
//
// 依存は <cstdint> のみに閉じる。DeferredDeletionQueue.h（entry への刻印）と
// IRetireProvider.h（retireDomainId() の戻り型）の双方から参照されるため、
// どちらか一方に定義するとインクルードが循環する。独立ヘッダに置く。
//
//   Unstamped : 刻印されていない（enqueue 時の刻印欠落＝バグ）。生成側は付けない。
//   Unknown   : 刻印済みだがドメインを特定できない（意図的）。Unstamped と混同しない。
//   EngineISR : AudioEngine の epoch ドメイン
//   EQ        : EQProcessor の epoch ドメイン
//   Count     : sentinel（配列サイズ用）。ドメイン値としては使わない。
//==============================================================================

#include <cstdint>

namespace convo {

enum class RetireDomainId : std::uint8_t
{
    Unstamped = 0,
    Unknown   = 1,
    EngineISR = 2,
    EQ        = 3,
    Count     = 4
};

// 配列サイズ用（std::size_t を持ち込まないため int で定義する）
inline constexpr int kRetireDomainCount = static_cast<int>(RetireDomainId::Count);

// JSON 出力用の名前（設計差分 §5.1 の id と一致させる）
inline const char* retireDomainName(RetireDomainId id) noexcept
{
    switch (id)
    {
        case RetireDomainId::Unstamped: return "Unstamped";
        case RetireDomainId::Unknown:   return "Unknown";
        case RetireDomainId::EngineISR: return "EngineISR";
        case RetireDomainId::EQ:        return "EQ";
        case RetireDomainId::Count:     break;
    }

    return "Invalid";
}

} // namespace convo
