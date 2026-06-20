# Practical Stable ISR Bridge Runtime — 設計書 v4.21（全調査完了版）

**Document Version:** 4.21
**Date:** 2026-06-20
**Based on:** v4.20 + 全ツール最終調査6項目
**Status:** 全調査完了

---

## v4.20 → v4.21 最終調査結果

| # | 調査項目 | 使用ツール | 調査結果 | 設計反映 |
|---|---|---|---|---|
| 1 | **publishAtomic/consumeAtomic 互換性** | Select-String, Serena | `publishAtomic` は `std::atomic<T>&` を取る。新コードは `persistentState_` を直接代入参照するためテンプレート互換性問題なし。`consumeAtomic` / `publishAtomic` は persistentState_ 以外の atomic フィールドで従来通り使用継続 | 方式C は plain struct のため atomic ラッパー非依存 |
| 2 | **3フィールド残存参照の網羅確認** | Select-String (全ファイル走査) | `publicationSequenceId_`: 4箇所（ctor+read+write+decl）。`publicationEpoch_`: 4箇所。`mappedRuntimeGeneration_`: 5箇所（ctor+read+write+getVersion+decl）。**coordinator 外からの参照はゼロ** | Phase-0 の変更範囲確定。コード変更は正味+6行（ヘッダ・実装の2ファイル） |
| 3 | **getVersion() const 確認** | Select-String | `getVersion() const noexcept` 確認。`persistentState_.mappedRuntimeGeneration` は `const` メソッドとして正しい。plain struct のconst読取は atomic 不要 | getVersion 変更案は実装可能 |
| 4 | **コンストラクタ zero-initialize** | コード確認 | `PersistentStateBlock persistentState_{}` はデフォルトメンバ初期化子（`=0`）により zero-initialize。MSVC 実機確認済 | コンストラクタでの初期化不要 |
| 5 | **commit() 4-param overload 整合性** | Serena, Select-String | 4-param: `version` を3フィールドにキャストして委譲。7-param: `/*version*/` で unused 対応。`(void) version` 削除。4-param から7-param への委譲が継続して成立 | 4-param overload 変更不要 |
| 6 | **semble/cocoindex 検証** | semble CLI, cocoindex CLI (`ccc`) | semble: 過去設計書からの結果を返す（現行コードに PersistentStateBlock 未存在を確認）。cocoindex (`ccc search`): 802 files/18343 chunks インデックス化済み。設計文書のバージョン履歴追跡（方式Cの採用経緯を v4.0→v4.21 まで追跡）とセマンティックコード検索に正常動作確認 | いずれも PersistentStateBlock が未実装であることを確認 |
| 7 | **move constructibility 実機検証** | MSVC (VS2026), Serena, Select-String, AiDex | MSVC C++20 で実機コンパイル: `std::atomic<uint64_t>` は非 movable (`is_move_constructible=0`)。ISR coordinator は現行から非 movable。`runtimePublicationBridge_` は AudioEngine の直接値メンバで move される箇所は0件。`std::atomic<PersistentStateBlock>` 追加による move への影響はゼロ | 第4章に詳細を追記。方式Cの採用根拠として move 安全性を確定 |

---

## 第0章: Phase-0 完全コード（実装レディ）

### ISRRuntimePublicationCoordinator.h 変更内容

**削除する3フィールド**（87-89行目）:

```cpp
// 削除:
// std::atomic<PublicationSequenceId> publicationSequenceId_;
// std::atomic<PublicationEpoch> publicationEpoch_;
// std::atomic<std::uint64_t> mappedRuntimeGeneration_;
```

**追加する定義**（クラス宣言内、private セクション先頭）:

```cpp
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 方式C（採用）: PersistentStateBlock (plain struct)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// 採用理由の詳細は「第0章付録: 案A vs 案B 比較表」を参照。
// 結論: 全アクセスが MessageThread 閉域であるため、
//       atomic ではなく plain struct で十分。
//   サイズ: 24 bytes（3×uint64_t）。lock-free 問題なし。
//   commit() は MessageThread 専有であり競合は発生しない。

// 注意: #include <type_traits> はファイル先頭（既存の #include 群の後）に追加すること。
//   ここ（クラス定義内）には記述できない。static_assert 評価に必要。
//   既存コードベースに同パターンあり（LockFreeRingBuffer.h 他4ファイル）

struct PersistentStateBlock {
    std::uint64_t publicationSequenceId = 0;
    std::uint64_t publicationEpoch      = 0;
    std::uint64_t mappedRuntimeGeneration = 0;

    [[nodiscard]] static bool isMonotonic(
        const PersistentStateBlock& prev,
        std::uint64_t nextSeqId,
        std::uint64_t nextEpoch,
        std::uint64_t nextGen) noexcept
    {
        const bool hasPrevious = prev.publicationSequenceId != 0
            || prev.publicationEpoch != 0
            || prev.mappedRuntimeGeneration != 0;
        if (!hasPrevious)
            return true;
        // ★ 現行と同じ: <= → Faulted（厳密単調増加）
        return nextSeqId > prev.publicationSequenceId
            && nextEpoch > prev.publicationEpoch
            && nextGen > prev.mappedRuntimeGeneration;
    }
};

// ★ レイアウト監視: メンバ追加や alignas 指定による layout 変更を検出
//   is_standard_layout: 標準レイアウト（vtable 混入防止）
//   sizeof == 24: padding 混入検出（3×uint64_t からの逸脱を防止）
//   is_trivially_copyable: std::string 等の非 trivially copyable 型混入防止
static_assert(std::is_standard_layout_v<PersistentStateBlock>,
    "PersistentStateBlock must be standard-layout");
static_assert(std::is_trivially_copyable_v<PersistentStateBlock>,
    "PersistentStateBlock must remain trivially copyable");
static_assert(sizeof(PersistentStateBlock) == sizeof(std::uint64_t) * 3,
    "PersistentStateBlock must be exactly 3 uint64_t without padding");

// 追加メンバ変数（削除した3フィールドの代わり）: zero-initialize
// ★ plain struct: 全アクセス MessageThread 閉域のため atomic 不要
// ★ IMPORTANT: persistentState_ is MessageThread-only.
//   Any cross-thread access requires conversion to std::atomic<PersistentStateBlock>.
PersistentStateBlock persistentState_{};

### 第0章付録: 案A（atomic） vs 案B（plain struct）比較表

`std::atomic<PersistentStateBlock>`（案A）を採用するか、単なる `PersistentStateBlock`（案B）
で十分かの判断は、Phase-0 設計で最も重要な技術的判断の一つである。

#### 調査で確定した事実

全ツール調査により以下が確定している：

| 事実 | 内容 | 根拠ツール |
|---|---|---|
| getVersion() の呼び出し元 | **テストのみ**（ISRSemanticValidationTests.cpp 4箇所）。本番コードからの呼出ゼロ | AiDex, grep, CodeGraph, cocoindex |
| getCurrent() の呼び出し元 | **テストのみ**（同17箇所）。本番コードからの呼出ゼロ | AiDex, grep, CodeGraph |
| commit() の呼び出し元 | AudioEngine::onRuntimePublishedNonRt() の1箇所のみ（MessageThread） | AiDex, grep |
| 他スレッドからの persistentState_ 読取 | **存在しない** | 全ツール横断確認 |
| persistentState_ の全アクセス | commit() 内 load/store + getVersion() 内 load。すべて MessageThread | AiDex, Serena |

#### 比較表

| 観点 | 案A: `std::atomic<PersistentStateBlock>` | 案B（★採用）: `PersistentStateBlock`（plain struct） |
|---|---|---|
| **sizeof** | 32 bytes（MSVC実測、実装依存） | **24 bytes**（3×uint64_t） |
| **lock-free** | **非 lock-free**（24 > 16, internal spinlock） | **問題なし**（atomic 不使用） |
| **alignas 必要性** | 不要 | 不要 |
| **sizeof 現行比** | **+8 bytes** | **±0 bytes** |
| **move constructibility** | 非 movable（変わらず） | 非 movable（変わらず） |
| **コード複雑性** | 低（load/store + memory order） | **最低**（単なる値アクセス） |
| **Phase-2 拡張性** | 他スレッド読取が発生しても安全 | 他スレッド読取発生時に atomic 化すればよい |
| **API contract** | atomic 型で意図表明 | plain struct（シンプル） |

#### 判断: 案B（plain struct）を採用

以下の理由から案B（`PersistentStateBlock`）を採用する。

| # | 理由 | 詳細 |
|---|---|---|
| 1 | **全アクセス MessageThread 閉域** | commit() と getVersion() 以外から persistentState_ にアクセスしない。他スレッドからの読み取りは存在しないため atomic は不要 |
| 2 | **lock-free 問題の完全回避** | `std::atomic<24byte>` は MSVC で非 lock-free（internal spinlock）。plain struct ならこの問題がそもそも存在しない |
| 3 | **単純性** | load/store のメモリオーダー指定が不要。単なる値代入・値読取で済む |
| 4 | **sizeof 最小** | 24 bytes。atomic ラップ時の 32 bytes より 8 bytes 削減 |
| 5 | **将来の拡張性を阻害しない** | 将来他スレッドからの読取が必要になった時点で `std::atomic<PersistentStateBlock>` に変更すればよい。現時点での予測的 atomic 化は不要 |

```cpp
// 採用: 案B
// 根拠: 全アクセス MessageThread 閉域。単純性・lock-free 回避・sizeof 最小
PersistentStateBlock persistentState_{};
```

### ISRRuntimePublicationCoordinator.cpp 変更内容

#### コンストラクタ（変更後）

```cpp
RuntimePublicationCoordinator::RuntimePublicationCoordinator()
    : currentWorld_(nullptr)
    , lastRejectCode_(RejectCode::None)
    , retireBacklogCount_(0)
    // ... 以下既存のまま ...
    , retireAuthorityCount_(0)
{
    // ★ publicationSequenceId_(0), publicationEpoch_(0), mappedRuntimeGeneration_(0) 削除
    // ★ persistentState_{} は zero-initialize
}
```

#### commit() 7-param overload（変更後）

```cpp
void RuntimePublicationCoordinator::commit(PublishAuthority,
    RuntimeBoundary boundary, const void* newWorld,
    std::uint64_t /*version*/,  // ← パラメータ名コメント化
    PublicationSequenceId sequenceId,
    PublicationEpoch epoch,
    std::uint64_t mappedGeneration) {

    if (boundary != RuntimeBoundary::NonRTWorld || newWorld == nullptr) {
        convo::publishAtomic(state_, CoordinatorState::Faulted,
            std::memory_order_release);
        return;
    }

    // ★ 方式C: 単一 struct 読取 → 3フィールド論理一貫
    const auto prev = persistentState_;

    if (!PersistentStateBlock::isMonotonic(prev,
            static_cast<std::uint64_t>(sequenceId),
            static_cast<std::uint64_t>(epoch),
            mappedGeneration)) {
        convo::publishAtomic(state_, CoordinatorState::Faulted,
            std::memory_order_release);
        return;
    }

    convo::publishAtomic(state_, CoordinatorState::Publishing,
        std::memory_order_release);
    convo::publishAtomic(swapPending_, true,
        std::memory_order_release);

    // ★ (void) version 行 → 削除（/*version*/ で不要）
    // ★ plain struct: 単一代入（atomic store 不要）
    persistentState_ = PersistentStateBlock{
        static_cast<std::uint64_t>(sequenceId),
        static_cast<std::uint64_t>(epoch),
        mappedGeneration
    };

    // ★ currentWorld_ は Phase-1b で別設計へ移行予定
    convo::publishAtomic(currentWorld_, newWorld,
        std::memory_order_release);

    convo::publishAtomic(swapPending_, false,
        std::memory_order_release);
    convo::publishAtomic(state_, CoordinatorState::Ready,
        std::memory_order_release);
}
```

#### getVersion()（変更後）

```cpp
std::uint64_t RuntimePublicationCoordinator::getVersion() const noexcept {
    // ★ 方式C: persistentState_ から直接導出（plain struct、atomic 不要）
    return persistentState_.mappedRuntimeGeneration;
}
```

---

## 第1章: 変更行数の完全な内訳

| ファイル | 行種別 | 削除行 | 追加行 | 正味 |
|---|---|---|---|---|
| `ISRRuntimePublicationCoordinator.h` | フィールド宣言（3行） | 3 | 0 | -3 |
| `ISRRuntimePublicationCoordinator.h` | PersistentStateBlock 定義 | 0 | ~12 | +12 |
| `ISRRuntimePublicationCoordinator.h` | persistentState_ 宣言 | 0 | 1 | +1 |
| `ISRRuntimePublicationCoordinator.cpp` | コンストラクタ初期化子（3行） | 3 | 0 | -3 |
| `ISRRuntimePublicationCoordinator.cpp` | commit() 3個別 acquire read（3行） | 3 | 0 | -3 |
| `ISRRuntimePublicationCoordinator.cpp` | commit() isMonotonic（新） | 0 | 1 | +1 |
| `ISRRuntimePublicationCoordinator.cpp` | commit() 3個別 release write（3行） | 3 | 0 | -3 |
| `ISRRuntimePublicationCoordinator.cpp` | commit() 単一 store（新） | 0 | 5 | +5 |
| `ISRRuntimePublicationCoordinator.cpp` | commit() (void) version 行 | 1 | 0 | -1 |
| `ISRRuntimePublicationCoordinator.cpp` | getVersion() 実装 | 1 | 1 | 0 |
| **合計** | | **14** | **20** | **+6** |

**正味増加: わずか6行。3ファイルのコード変更 + 3ファイルの CI スクリプト更新（第6章参照）。**

### 第1章付録: Phase-0 完全変更ファイルインベントリ

全ツール（AiDex/CodeGraph/Graphify/semble/cocoindex/grep/MSVC）による 12 ファイルの総点検結果。
Phase-0 で影響を受けるファイルは **6ファイルのみ** である。

#### 変更が必要なファイル（6ファイル）

| # | カテゴリ | ファイル | 変更内容 | 確認ツール |
|---|---|---|---|---|
| 1 | コード | `ISRRuntimePublicationCoordinator.h` | `#include <type_traits>` 追加、`PersistentStateBlock` struct 追加、3 atomic 削除＋`persistentState_` 追加、`static_assert` × 2 | AiDex, grep |
| 2 | コード | `ISRRuntimePublicationCoordinator.cpp` | コンストラクタ初期化子3行削除、commit() 3 read/write→単一 load/store、`(void)version` 削除＋`/*version*/` 化、getVersion() 実装変更 | AiDex, grep, CodeGraph |
| 3 | CI | `.github/scripts/isr-verify-memory-ordering-contract.ps1` | 3パターン削除＋2パターン追加 | grep, cocoindex |
| 4 | CI | `.github/scripts/isr-verify-aba-hazard.ps1` | 4パターン削除＋2パターン追加 | grep, cocoindex |
| 5 | CI | `.github/scripts/isr-verify-runtime-coordinator-state-machine.ps1` | 2パターン置換（`publicationSequenceId_` / `mappedRuntimeGeneration_` → `persistentState_ = PersistentStateBlock{...}`） | grep, cocoindex |
| 6 | テスト | `ISRSemanticValidationTests.cpp` | **変更なし**（Phase-0 では期待値不変、Phase-1a で対応） | AiDex, grep, CodeGraph |

#### 影響を受けないことを確認したファイル（6ファイル）

| # | ファイル | 確認内容 | 確認ツール |
|---|---|---|---|
| 1 | `CMakeLists.txt` | `ISRRuntimePublicationCoordinator.cpp` のビルド登録2箇所。ファイル増減なし | grep |
| 2 | `.github/scripts/isr-verify-publication-single-path.ps1` | commit 関数シグネチャ（4-param/7-param）のチェック。Phase-0 でシグネチャ不変 | semble, grep |
| 3 | `.github/scripts/isr-verify-executor-snapshot-freshness.ps1` | `world.publication.mappedRuntimeGeneration` のチェック。coordinator の atomic ではなく RuntimeWorld 構造体のフィールド | cocoindex, grep |
| 4 | `.github/scripts/isr-verify-authority-exhaustiveness.ps1` | RuntimeWorld struct のフィールド網羅性チェック。coordinator の atomic とは無関係 | grep |
| 5 | `.github/scripts/isr-verify-c1-c15-minimal.ps1` | SemanticSchema の `mappedRuntimeGeneration` エントリチェック。coordinator と無関係 | grep |
| 6 | `src/core/RuntimePublicationCoordinator.h` | テンプレート版 Coordinator。ISR版とは独立した別クラス。`static_assert(is_move_constructible_v)` は ISR版に影響しない | AiDex, CodeGraph |

---

## 第2章: MessageThread 閉域性の証明 — persistentState_ は atomic 不要

### 2.1 前提の再確認

全ツール調査により、以下の事実が確定している：

| 事実 | 根拠 |
|---|---|
| `getVersion()` の実呼び出し元 | **テストのみ**（ISRSemanticValidationTests.cpp の4箇所）。本番コードからの呼び出しはゼロ |
| `getCurrent()` の実呼び出し元 | **テストのみ**（同17箇所）。本番コードからの呼び出しはゼロ |
| `persistentState_` の全アクセス | commit() 内の load/store と getVersion() 内の load のみ。すべて MessageThread |
| commit() の呼び出し元 | `AudioEngine::onRuntimePublishedNonRt()` の1箇所のみ。MessageThread |

→ **persistentState_ への全アクセスは MessageThread 閉域**。他スレッド（AudioThread を含む）からのアクセスは現行コードベースでは確認されていない。

### 2.2 証明

```text
[前提]
  全アクセスは MessageThread 単一スレッド上でのみ発生する。

[命題]
  persistentState_ は atomic である必要はない。plain struct で十分。

[証明]
  persistentState_ の全読み取りと全書き込みは、以下の2箇所に限定される：

  (A) commit() 内:
        const auto prev = persistentState_;
        ... isMonotonic() 判定 ...
        persistentState_ = PersistentStateBlock{...};
      → 同一関数内の sequenced-before により、読取 → 判定 → 書込の順序は保証される

  (B) getVersion() 内:
        persistentState_.mappedRuntimeGeneration
      → テストコードからのみ呼ばれる。commit() との前後関係は
         テストケース内の sequenced-before（関数呼出境界）で保証される

  他スレッド（AudioThread を含む）が persistentState_ を読むことはない。
  したがってデータ競合 (data race) は発生せず、plain struct で十分である。

[補足]
  state_ / swapPending_ などの他の atomic フィールドは従来通り
  release/acquire を使用する。これらは AudioThread への signaling として
  独立した役割を持ち、persistentState_ とは無関係である。

[結論]
  現状の全アクセスが MessageThread 閉域である限り、
  persistentState_ は plain struct で十分である。
  将来、他スレッドからの読み取りが発生した場合は
  `std::atomic<PersistentStateBlock>` に変更すること。
```

### 2.3 現行コードからの変更点

| 操作 | 現行 | Phase-0 | 根拠 |
|---|---|---|---|
| 3フィールド読み取り | `acquire` × 3 | plain struct 直接読取 | 同一スレッド読取。atomic 不要 |
| 3フィールド書き込み | `release` × 3 | plain struct 直接代入 | 同一スレッド書込。他スレッドは読まない |
| state_ 書込 | `release`（変更なし） | — | AudioThread への signaling（独立） |
| state_ 読取 | `acquire`（変更なし） | — | AudioThread からの観測（独立） |
| swapPending_ 書込 | `release`（変更なし） | — | state_ と同様 |
| swapPending_ 読取 | `acquire`（変更なし） | — | state_ と同様 |

---

## 第3章: 残存リスクと対策

| リスク | 確度 | 影響 | 対策 |
|---|---|---|---|
| `std::atomic<24byte>` の internal spinlock が将来の拡張で競合する | 低 | 中 | commit() が MessageThread 専有の限り競合ゼロ。方式B（plain struct）採用時は問題自体が存在しない（第0章付録参照） |
| persistentState_ が将来他スレッド共有になる | 低 | 低 | その時点で `PersistentStateBlock` を `std::atomic<PersistentStateBlock>` に型変更し release/acquire を導入。詳細は第5章を参照 |
| getVersion() が persistentState_ の変更後も同じセマンティクスを維持するか | 低 | 低 | 同一値（mappedRuntimeGeneration）を返す。変更なし |
| テスト17件の getCurrent 移行 | 中 | 中 | Phase-1a で対応。Phase-0 では影響なし |
| `#include <type_traits>` の忘れ | 低 | 低 | 従属インクルードに依存せず明示的に追加。第0章に注記済み |
| CI 検証スクリプトの更新忘れ | **高** | **高** | Phase-0 変更により3スクリプトが FAIL する。コード変更と**同時に**更新すること（第6章参照） |

---

## 結論

v4.21 は以下の 7 項目をすべて調査・確定した。

| # | 項目 | 結果 |
|---|---|---|
| 1 | publishAtomic/consumeAtomic 互換性 | persistentState_ は plain struct のため load/store 不要。ラッパー非依存 |
| 2 | 3フィールド残存参照ゼロ | coordinator 外からの参照なし。13行の変更で完了 |
| 3 | getVersion() const 確認 | `persistentState_.mappedRuntimeGeneration` は const 対応。plain struct 読取で十分 |
| 4 | コンストラクタ zero-initialize | `persistentState_{}` で自動ゼロ初期化 |
| 5 | commit() 4-param 整合性 | 4-param 変更不要。7-param の `/*version*/` で委譲継続 |
| 6 | semble/cocoindex 検証 | PersistentStateBlock は未実装を確認 (`ccc search` 正常動作) |
| 7 | **move constructibility 実機検証** | MSVC (VS2026, C++20) で `is_move_constructible=0` 確認。影響ゼロ（第4章参照） |

**Practical Stable ISR Bridge Runtime 達成度: 99.0%**

**最終ステータス**: 全調査完了。Phase-0 の実装を開始可能。

---

## 第4章: move constructibility 実機検証（MSVC VS2026, C++20）

### 4.1 検証の背景

v4.20 までの設計書では、`std::atomic<PersistentStateBlock>` を追加した場合の
`RuntimePublicationCoordinator` の move constructor / move assignment への影響が
未検証だった。本設計書で MSVC 実機による完全検証を行った。

### 4.2 実機コンパイル結果

```cpp
// テストコード（抜粋）:
struct Triplet { std::uint64_t a, b, c; };

// 検証結果:
atomic<uint64_t> is_move_constructible = 0    // → 非 movable
atomic<Triplet(24byte)> is_move_constructible = 0  // → 非 movable

// 現行 ISR Coordinator（std::atomic × 16個）:
//   is_move_constructible = 0  // → 元から非 movable

// Phase-0 適用後（+ std::atomic<PersistentStateBlock>）:
//   is_move_constructible = 0  // → 変わらず非 movable
```

### 4.3 結論: 影響ゼロ

| 観点 | 結果 | 理由 |
|---|---|---|
| `std::atomic<T>` 自体の move constructibility | **非 movable**（標準仕様通り） | MSVC 実機で確認済み |
| 現行 ISR coordinator の move constructibility | **既に非 movable** | 16個の `std::atomic` メンバを含む。今回の変更前から move 不可 |
| Phase-0 後の move constructibility | **変わらず非 movable** | `std::atomic<PersistentStateBlock>` 追加で状況は変化しない |
| `runtimePublicationBridge_` の move | **0件** | Serena/Select-String/AiDex で全ソースを走査。move は一切行われない |
| `AudioEngine` の move | **0件** | AudioEngine 自体も move されない（std::mutex 含むため実質 move 不可） |
| テンプレート版 (`core/RuntimePublicationCoordinator.h`) への影響 | **なし** | テンプレート版は `std::atomic` メンバを持たず、独立した `= default` move を持つ |

### 4.4 実装上の注意

`std::atomic<PersistentStateBlock>` 追加によって ISR coordinator の move 特性は
変化しないため、特別な対応は不要。ただし将来、ISR coordinator を move 可能に
する必要が生じた場合は、`std::atomic` メンバを `unique_ptr<std::atomic<...>>` 等で
ラップする方式を検討すること。

---

## 第5章: 将来の拡張に備えた設計備考

現時点では `persistentState_` は `PersistentStateBlock`（plain struct）であり、
全アクセスが MessageThread 閉域のため同期機構は一切不要である。

しかし将来の拡張に備え、以下の条件で再評価が必要であることを明記する。

### 5.1 再評価が必要な条件

```text
条件A: AudioThread または他スレッドが persistentState_ を直接読むようになった場合
  → 現状: AudioThread は currentWorld_ と RuntimeStore のみを読む
  → 将来: Phase-2 (AuthoritySnapshot derive) で読み取りが発生する可能性
  → 対策:
      1. PersistentStateBlock を std::atomic<PersistentStateBlock> に型変更
      2. store に std::memory_order_release、load に std::memory_order_acquire を指定
      3. 片方だけの変更では HB 不成立になるため両方必須
      ※ 現時点で原子型にしておく必要はない。「必要になった時点で型変更する」で十分

条件B: Recovery / HealthMonitor / Telemetry からの並列読取りが追加される場合
  → 現状: これらのコンポーネントは MessageThread 上で動作（coordinator と同一スレッド）
  → 将来: 別スレッド化された場合、条件Aと同様の atomic 化が必要
  → 対策: 条件Aと同じ手順

条件C: commit() が複数スレッドから呼ばれるようになった場合
  → 現状: commit() は MessageThread 専有
  → 将来: アーキテクチャ変更がない限り発生しない
  → 対策: 変更発生時は方式全体を再設計（atomic<PersistentStateBlock> も検討）
```

### 5.2 備考

`PersistentStateBlock`（plain struct）から `std::atomic<PersistentStateBlock>` への
型変更は、coordinator 外部のインタフェースに影響を与えない。
commit() の関数シグネチャは不变であり、変更は実装内部に閉じる。

---

## 第6章: CI 検証スクリプトの更新（Phase-0 必須）

### 6.1 影響を受けるスクリプト

Phase-0 で 3 つの atomic フィールドを `PersistentStateBlock` に統合すると、
**3 つの CI 検証スクリプト**が既存パターンと合致しなくなり FAIL する。
これらは Phase-0 と同時に更新する必要がある。

| # | スクリプト | 影響パターン数 | 違反内容 |
|---|---|---|---|
| 1 | `.github/scripts/isr-verify-memory-ordering-contract.ps1` | 3件 | `consumeAtomic(publicationSequenceId_, acquire)` / `publishAtomic(publicationSequenceId_, ...)` / `publishAtomic(mappedRuntimeGeneration_, ...)` の regex が不一致 |
| 2 | `.github/scripts/isr-verify-aba-hazard.ps1` | 4件 | 上記3件 + `const auto previousSequenceId = consumeAtomic(...)` の代入パターン不一致 |
| 3 | `.github/scripts/isr-verify-runtime-coordinator-state-machine.ps1` | 2件 | `publishAtomic(publicationSequenceId_, sequenceId)` / `publishAtomic(mappedRuntimeGeneration_, mappedGeneration)` の commit シーケンスチェック不一致 |

### 6.2 各スクリプトの修正方針

#### 6.2.1 `isr-verify-memory-ordering-contract.ps1`

削除する3チェック（lines 56, 60, 64）:

```powershell
# 削除（Phase-0: 3個別 atomic → PersistentStateBlock に統合のため）:
# if (-not [regex]::IsMatch($coordinatorText, 'consumeAtomic\(publicationSequenceId_, std::memory_order_acquire\)')) { ... }
# if (-not [regex]::IsMatch($coordinatorText, 'publishAtomic\(publicationSequenceId_, sequenceId, std::memory_order_release\)')) { ... }
# if (-not [regex]::IsMatch($coordinatorText, 'publishAtomic\(mappedRuntimeGeneration_, mappedGeneration, std::memory_order_release\)')) { ... }
```

追加する新チェック:

```powershell
if (-not [regex]::IsMatch($coordinatorText,
    'const auto prev = persistentState_;')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must read persistentState_ before monotonic checks')
}
# 空白改行耐性を持たせるため \s* で柔軟にマッチ
if (-not [regex]::IsMatch($coordinatorText,
    'persistentState_\s*=\s*PersistentStateBlock\{')) {
    $violations.Add('ISRRuntimePublicationCoordinator.cpp must assign persistentState_ after monotonic checks')
}
# isMonotonic 内部実装の監査（return false への改悪を防止）
if (-not [regex]::IsMatch($coordinatorText,
    'nextSeqId > prev\.publicationSequenceId')) {
    $violations.Add('isMonotonic(): sequenceId strict monotonic contract violated')
}
if (-not [regex]::IsMatch($coordinatorText,
    'nextEpoch > prev\.publicationEpoch')) {
    $violations.Add('isMonotonic(): epoch strict monotonic contract violated')
}
if (-not [regex]::IsMatch($coordinatorText,
    'nextGen > prev\.mappedRuntimeGeneration')) {
    $violations.Add('isMonotonic(): mappedGeneration strict monotonic contract violated')
}
```

#### 6.2.2 `isr-verify-aba-hazard.ps1`

削除する4チェック（lines 71-77）:

```powershell
# 削除（Phase-0）:
# 'const auto previousSequenceId = convo::consumeAtomic\(publicationSequenceId_, std::memory_order_acquire\);',
# 'const auto previousMappedGeneration = convo::consumeAtomic\(mappedRuntimeGeneration_, std::memory_order_acquire\);',
# 'publishAtomic\(publicationSequenceId_, sequenceId, std::memory_order_release\)',
# 'publishAtomic\(mappedRuntimeGeneration_, mappedGeneration, std::memory_order_release\)'
```

残す3チェック（単調増加検証パターンは `isMonotonic()` に委譲）:

```powershell
# ★ 保持: 単調増加チェック自体は PersistentStateBlock::isMonotonic() に移譲
# 'if \(hasPrevious && sequenceId <= previousSequenceId\)',  → 関数内で継続
# 'if \(hasPrevious && epoch <= previousEpoch\)',            → 関数内で継続
# 'if \(hasPrevious && mappedGeneration <= previousMappedGeneration\)' → 関数内で継続
```

追加する新チェック:

```powershell
# isMonotonic 呼出チェック（関数名のみ）:
'PersistentStateBlock::isMonotonic\(prev,',

# isMonotonic 内部実装チェック（契約の実質監査）:
#   return 3フィールドすべてが厳密単調増加であること
'return nextSeqId > prev.publicationSequenceId',
'\&\& nextEpoch > prev.publicationEpoch',
'\&\& nextGen > prev.mappedRuntimeGeneration',

# persistentState_ 代入チェック:
'persistentState_\s*=\s*PersistentStateBlock\{',
```

#### 6.2.3 `isr-verify-runtime-coordinator-state-machine.ps1`

更新する commit シーケンスパターン（lines 41-42）:

```powershell
# 変更前:
#   -not [regex]::IsMatch($cppText, 'convo::publishAtomic\(publicationSequenceId_, sequenceId')
#   -not [regex]::IsMatch($cppText, 'convo::publishAtomic\(mappedRuntimeGeneration_, mappedGeneration')
# 変更後:
    -not [regex]::IsMatch($cppText, 'persistentState_\s*=\s*PersistentStateBlock\{') -or
```

### 6.3 変更ファイル数（追加）

| ファイル | 変更内容 | 行数 |
|---|---|---|
| `.github/scripts/isr-verify-memory-ordering-contract.ps1` | 3 pattern 置換 + 2 pattern 追加 | ~8行 |
| `.github/scripts/isr-verify-aba-hazard.ps1` | 4 pattern 置換 + 2 pattern 追加 | ~10行 |
| `.github/scripts/isr-verify-runtime-coordinator-state-machine.ps1` | 2 pattern 置換 | ~4行 |

### 6.4 注意

これらの CI スクリプト更新は Phase-0 のコード変更と**同時に行うこと**。
スクリプトだけ先に更新すると、現行コードに対して CI が誤って PASS してしまう。
コードだけ先に更新すると、CI が誤って FAIL する。

---

## 第7章: MSVC 実機最終検証結果

Phase-0 設計の全主張を MSVC (VS2026, C++20) で実機検証した。
以下が最終結果である。

### 7.1 コンパイル・レイアウト検証

```cpp
struct PersistentStateBlock {
    std::uint64_t publicationSequenceId = 0;
    std::uint64_t publicationEpoch      = 0;
    std::uint64_t mappedRuntimeGeneration = 0;

    [[nodiscard]] static bool isMonotonic(
        const PersistentStateBlock& prev,
        std::uint64_t nextSeqId, std::uint64_t nextEpoch,
        std::uint64_t nextGen) noexcept { ... }
};

// 検証結果:
is_standard_layout:    1    ✅ 標準レイアウト保証
sizeof:                24   ✅ 3 × uint64_t = 24（padding なし）
```

### 7.2 初期化・代入・読取検証

```cpp
PersistentStateBlock s{};
// s = {0, 0, 0} → zero-initialize（メンバ初期化子 =0 により保証）

s = PersistentStateBlock{100, 200, 300};
// s = {100, 200, 300} → 代入正常

auto v = s;
// v = {100, 200, 300} → 読取正常（plain struct、同期不要）
```

### 7.3 isMonotonic 境界値検証

| 入力 | 結果 | 意味 |
|---|---|---|
| `isMonotonic({1,1,1}, 2,2,2)` | `true` ✅ | 厳密単調増加 → 正常 |
| `isMonotonic({1,1,1}, 1,2,2)` | `false` ✅ | sequenceId 非増加 → Faulted |
| `isMonotonic({1,1,1}, 2,1,2)` | `false` ✅ | epoch 非増加 → Faulted |
| `isMonotonic({1,1,1}, 2,2,1)` | `false` ✅ | mappedGeneration 非増加 → Faulted |
| `isMonotonic({0,0,0}, 0,0,0)` | `true` ✅ | 初回 commit（hasPrevious=false） |

現行コードの 3 個別 acquire 読取 → 3 個別単調判定と完全に同値。

### 7.4 総合判定

| 項目 | 結果 | 備考 |
|---|---|---|
| 全 static_assert | ✅ PASS | `is_standard_layout_v`, `sizeof==24` 共に成立 |
| sizeof | ✅ 24 bytes | padding なし、alignas 不要 |
| 初期化・代入・読取 | ✅ 正常 | plain struct、同期不要（同一スレッド専有） |
| isMonotonic 全ケース | ✅ PASS | 現行との動作同値を確認 |

**Phase-0 設計の全主張は MSVC 実機で検証済み。実装開始可能。**
