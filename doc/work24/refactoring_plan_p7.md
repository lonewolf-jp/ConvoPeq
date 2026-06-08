# 改修計画書: Practical Stable ISR Bridge Runtime — 未達6領域の改修

> **ベース文書**: `doc/work24/notfinished7.md`
> **検証対象**: ConvoPeq 最新ソースコード（2026-06-08時点）
> **レビュー日**: 2026-06-08
> **ステータス**: レビュー済み — フィードバック反映版 v2（7項目の修正完了）

---

## 目次

1. [検証サマリー](#1-検証サマリー)
2. [優先順位とフェーズ分け](#2-優先順位とフェーズ分け)
3. [Phase A-1: Quarantine実体化](#3-phase-a-1-quarantine実体化)
4. [Phase A-2: Shutdown完全閉包](#4-phase-a-2-shutdown完全閉包)
5. [Phase B-1: Reclaim完結保証](#5-phase-b-1-reclaim完結保証)
6. [Phase B-2: Overflow自己防衛](#6-phase-b-2-overflow自己防衛)
7. [Phase C-1: Generation 64bit化](#7-phase-c-1-generation-64bit化)
8. [Phase C-2: Deferred Publish改善](#8-phase-c-2-deferred-publish改善)
9. [テスト計画](#9-テスト計画)
10. [実装順序と依存関係](#10-実装順序と依存関係)

---

## 1. 検証サマリー

### 検証方法

以下のツールをすべて使用し、現状コードを多角的に検証した：

| ツール | 用途 | 結果 |
| :--- | :--- | :--- |
| `grep` / `Select-String` | キーワード横断検索 | 全対象ファイルを網羅 |
| `ccc search` (cocoindex-code) | ASTベースセマンティック検索 | generation, shutdown関連を確認 |
| `graphify query` (DeepSeek) | 知識グラフ解析 | DSPQuarantineManager/ShutdownRuntime間の関連を可視化 |
| `semble` (CLI) | 参照解析・呼び出し元検出 | quarantine呼び出し元の有無を確認 |
| `codegraph query` (CodeGraph MCP) | モジュール構造解析 | インデックス0件（差分なし） |
| 直接ファイル読み取り | コード詳細確認 | 全該当ファイルを精読 |

### 現状評価

| # | 領域 | notfinished7評価 | コード検証結果 | 乖離 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Quarantine | 60% | **25%** — `dspQuarantineManager_` は宣言のみで未使用。`DSPHandleRuntime::quarantine()` も未呼び出し。実質的に `RetireRuntimeEx::quarantine()` (lane管理)しか動作していない | **大** |
| 2 | Shutdown完全閉包 | 85% | **70%** — ShutdownFSM (ShutdownPhase) は存在するが `PublicationAdmission::evaluate()` は AudioEngine::lifecycleState を見ており、ShutdownRuntime::phase_ との連携が不十分。`notifyTransitionComplete` に shutdown ガードがない。DrainAudit は存在しない | **中** |
| 3 | Reclaim完結保証 | 75% | **65%** — `isFullyDrained()` は存在するが pendingPublication / deferredPublish / activeCrossfade / pendingDeletion の個別追跡がない。全slotのlifecycle監査ダンプ不在 | **中** |
| 4 | Overflow自己防衛 | 65% | **70%** — `evaluateRetirePressureLevelNoRt` / `applyRetirePressurePolicyNoRt` は実装済み。既存 level (0-3) で十分機能。overflow 検出 → throttle の結合が弱い | **小** |
| 5 | Generation安全性 | 70% | **70%** — `uint32_t` のまま。`onRuntimeRetiredNonRt()` で world->generation (uint64_t) を 32bit に truncate している箇所あり | **小** |
| 6 | Deferred Publish改善 | 60% | **60%** — `std::optional<PublishRequest>` 単一保持のまま。sequence番号によるstaleness検出がない | **一致** |

### 重大発見: デッドコード/未使用資産

1. **`AudioEngine::dspQuarantineManager_`** — メンバ変数として宣言されている (`AudioEngine.h:3443`) が、**どの .cpp からも使用されていない**。`quarantineHandle()`, `reclaimSlot()`, `isQuarantined()` は実装されているが呼び出し元ゼロ。
2. **`DSPHandleRuntime::quarantine(DSPHandle)`** — 実装は存在する (`ISRDSPHandle.cpp:112`) が、**呼び出し元が存在しない**。デッドコード。
3. **`DSPQuarantineManager` クラス** — `std::vector<std::atomic<bool>> quarantineFlags_` のみ。reason/timestamp/generation の管理がなく、「隔離」ではなく「フラグ設定」に留まる。

---

## 2. 優先順位とフェーズ分け

実運用クラッシュ/メモリリークに直結するものから着手する。

```text
Phase A（最優先）: Quarantine実体化 + Shutdown完全閉包
Phase B（次優先）: Generation 64bit化 + Reclaim完結保証
Phase C（計画的）: Overflow自己防衛 + Deferred Publish改善
```

### 採用方針

| 項目 | 判断 | 理由 |
| :--- | :--- | :--- |
| QuarantineManager統合 | 修正して採用 | 未使用の `DSPQuarantineManager` と `DSPHandleRuntime::quarantine()` を接続。`create()` 改修は不要（現状で Quarantined スロットは再利用されない） |
| Shutdown閉包 | そのまま採用 | ただし Authority は `ShutdownRuntime` に一本化。`AudioEngine::isShutdownInProgress()` は委譲する。DrainAudit の完了条件から quarantineResident は除外 |
| Reclaim保証 | 修正して採用 | ReclaimTicket/カウンタ追跡は採用せず、`emitRetireTrace()` による slot 単位の監査ダンプを追加 |
| Overflow防衛 | 修正して採用 | RuntimePressureState 列挙型は追加せず、既存 int level を維持。コメント強化と overflow→throttle 結合のみ |
| Generation64bit | そのまま採用 | 事前に全影響型（DSPHandle/RetireIntent/CrossfadeRecord/serialization/trace）の棚卸し必須 |
| Deferred Publish | 修正して採用 | FIFOリング化は採用せず、sequence 番号による stale discard 方式を採用 |

---

## 3. Phase A-1: Quarantine実体化

### 3.1 現状

```cpp
ISRDSPQuarantine.h:
  class DSPQuarantineManager {
      std::vector<std::atomic<bool>> quarantineFlags_;  // ← フラグのみ
  };

AudioEngine.h:3443:
  DSPQuarantineManager dspQuarantineManager_;  // ← 宣言のみ。未使用
```

- `DSPHandleRuntime::resolve()` は slot state が `Quarantined` の場合は `{nullptr,false,false}` を返す（これは正しい）
- `DSPHandleRuntime::create()` は `state == Reclaimed` のみチェックしているが、`Quarantined` は `Reclaimed` ではないため、**現状でも Quarantined スロットは再利用されない**（初版計画書の「再利用可能」は誤りであった）
- `DSPHandleRuntime::quarantine()` は実装済みだが誰も呼んでいない（デッドコード）
- `dspQuarantineManager_` は AudioEngine のメンバとして宣言されているが誰も使っていない（デッドメンバ変数）
- `RetireRuntimeEx::quarantine(slot)` は呼ばれている（Commit.cpp L536, L556）が、これは lane 管理（RetireLane::Quarantine への遷移）であり、DSPHandle の隔離とは別系統

### 3.2 改修内容

#### 3.2.1 QuarantineEntry 構造体の追加

**対象ファイル**: `src/audioengine/ISRDSPQuarantine.h`

```cpp
enum class QuarantineReason {
    GenerationMismatch,
    ResolveFailure,
    PublishViolation,
    CrossfadeViolation,
    ShutdownViolation,
    RetireDeferralTimeout,
    Unknown
};

struct QuarantineEntry {
    uint32_t slot;                    // 対象スロット
    uint64_t generation;              // uint64_t (C-1 先行対応)
    QuarantineReason reason;          // 隔離理由
    uint64_t quarantineEpoch;         // 隔離時のエポック
    uint64_t quarantineTimestampUs;   // 隔離時のタイムスタンプ(us)
    uint32_t detailCode;              // 詳細コード（custom message 代替。enum にない理由をコード化）
    bool reclaimAllowed;              // 強制解放許可フラグ
};
```

#### 3.2.2 DSPQuarantineManager の拡張

**対象ファイル**: `src/audioengine/ISRDSPQuarantine.h`, `ISRDSPQuarantine.cpp`

`std::vector<std::atomic<bool>>` から固定長ベクタによる構造化管理へ移行。slot 数は固定（MAX_DSP_SLOTS=256）であり、lookup は slot 指定のため hash 不要。Practical Stable 思想に合致する `std::vector<std::optional<QuarantineEntry>>` を採用する：

```cpp
class DSPQuarantineManager {
public:
    explicit DSPQuarantineManager(std::size_t maxSlots = 256);

    // 隔離: slot + generation(最初からuint64_t, C-1先行対応) + reason を記録
    // ★ residentCount_ の二重加算防止のため、既存エントリの有無を確認してからインクリメント
    void quarantineHandle(uint32_t slot, uint64_t generation,
                          QuarantineReason reason);

    // スロット解放（隔離解除）
    // ★★★ generation 一致確認を追加。generation が異なる場合は削除しない。
    //     理由: slot=7, generation=100 で quarantine 後、generation=101 で再利用されると
    //     reclaimSlot(7) だけで新しい隔離情報まで消せる。
    //     generation 一致時のみ削除することで、誤った隔離情報削除を防止する。
    void reclaimSlot(uint32_t slot, uint64_t generation);

    // ★★★ 隔離状態の判定は DSPQuarantineManager が Authority（3.5 参照）。
    //    quarantineHandle() の成功が隔離成立条件であり、その後に DSPHandleRuntime の
    //    slot state が従属的に更新される。状態確認は DSPHandleRuntime::resolve() 経由。
    //    isQuarantined() は提供しない。

    // 隔離エントリ情報取得（調査用）
    std::optional<QuarantineEntry> getEntry(uint32_t slot) const;

    // 全隔離エントリ数
    size_t residentCount() const noexcept;

    // 最長 quarantine 経過時間（秒）。全 Entry の timestampUs と現在時刻の差の最大値を返す。
    // TTL超過検出用。長時間運用で quarantine が永久残留するのを防止する監査。
    // 閾値例: 10分 → Critical Quarantine Leak 警告、30分 → slot枯渇リスク警告、60分 → 強制ログ
    uint64_t getMaxEntryAgeSec() const noexcept;

    // ★ 強制解放は ShutdownPhase::ReclaimComplete 以降のみ許可
    //    → destroyForShutdown() として rename（forceReclaim は危険のため廃止）
    bool destroyForShutdown(uint32_t slot);

private:
    // ★★★ 設計変更: RT アクセスと NonRT 監査を分離する。
    //    従来案は単一 Entry 構造体（atomic active + 残りは NonRT専用）だったが、
    //    「active フラグ」と「Entry の有無」が二重の真実（divergence source）となるリスクがある。
    //    そのため以下に分割する：
    //
    //    RT側: std::array<std::atomic<bool>, kMaxSlots>
    //      隔離中か否かの二値情報のみ。RT スレッドから lock-free で読み取り可能。
    //      書き込みは NonRT からの publishAtomic のみ。
    //
    //    NonRT側: std::vector<QuarantineEntry>
    //      隔離理由/世代/タイムスタンプを含む完全な監査情報。historical vector。
    //      RT からはアクセス不可。NonRT のみが読み書きする。
    //
    //    この分離により、RT の二値判定と NonRT の詳細監査が独立し、
    //    「active と Entry が一致しない」という divergence が原理的に発生しない。
    static constexpr size_t kMaxSlots = 256;

    // RT側: 隔離中フラグ bitset（atomic read only）
    std::array<std::atomic<bool>, kMaxSlots> quarantineActiveFlags_{};

    // NonRT側: 監査記録ベクタ（historical。エントリは削除されず追記される）
    struct Entry {
        uint64_t timestampUs;             // 隔離時のタイムスタンプ(us)
        uint64_t generation;              // 隔離時点の世代番号
        QuarantineReason reason;          // 隔離理由
        uint32_t slot;                    // 対象スロット
        bool resolved;                    // true=隔離解除済み（reclaim/destroy完了）
    };
    // ★★★ auditLog_ は追記専用 vector だが、長期運用でメモリ増加が懸念される。
    //     以下の compaction 戦略を採用する：
    //     1. 全エントリの resolved==true を確認（隔離解除済み）
    //     2. resolved かつ一定数（例: 1024エントリ）を超えた場合、先頭から
    //        resolved エントリを削除（ring buffer 動作に近い）
    //     3. 未解決（resolved==false）のエントリは絶対に削除しない
    //     これにより未解決の隔離情報を保持しつつ、メモリ増加を抑制する。
    std::vector<Entry> auditLog_;         // 追記専用。参照は slot index ではなく線形スキャン

    // ★★★ quarantine TTL監査: getMaxEntryAgeSec() は全 active Entry の最大経過時間を返す。
    //    長時間運用で quarantine が永久残留するのを防止するための監査指標。
    //    閾値例:
    //      >10分 → Warning: Quarantine Leak Detected（診断ログ出力）
    //      >30分 → Critical: Quarantine Slot Exhaustion Risk（全slotダンプ推奨）
    //      >60分 → Fatal: Quarantine Overflow（強制ログ + 調査必須）
    //    本メソッドは DSPQuarantineManager の getMaxEntryAgeSec() として実装。
    //    collectDrainAudit() から定期的に呼ばれる。
};
```

#### 3.2.3 DSPHandleRuntime::quarantine() と DSPQuarantineManager の接続

**重要**: `DSPHandleRuntime` に `DSPQuarantineManager` への参照を注入する必要はない。代わりに **AudioEngine レベル**で接続する。

現在の `DSPHandleRuntime::quarantine()` は slot state を `Quarantined` に設定する。これで `resolve()` 経由のアクセスはブロックされる。

**不足しているのは「隔離理由の記録」と「調査可能性」のみ**である。

したがって、以下の方針とする：

1. `DSPHandleRuntime::quarantine()` はそのまま維持（slot state 設定）
2. `DSPQuarantineManager` は隔離理由/世代/タイムスタンプの記録専用
3. 接続は **呼び出し側（AudioEngine）** で両方を呼ぶ

**呼び出し経路 — 集約関数 `quarantineSlot()` 経由（3.5 参照）**:

現状の `retireRuntimeEx_.quarantine(pendingSlot)` 呼び出し箇所（`AudioEngine.Commit.cpp` L536, L556）は、新設する集約関数 `AudioEngine::quarantineSlot()` に置き換える。この関数は3系統の隔離（QuarantineManager → DSPHandleRuntime → RetireRuntimeEx）を1トランザクションとして実行する：

```cpp
// AudioEngine.Commit.cpp — 変更後
// ★★★ 単一の quarantineSlot() に集約（3系統の隔離を1トランザクション化）
//     従来の3段階呼び出し（retireRuntimeEx_.quarantine + dspHandleRuntime_.quarantine
//     + dspQuarantineManager_.quarantineHandle）は quarantineSlot() に統合。
quarantineSlot(pendingSlot, pending.generation,
               convo::isr::QuarantineReason::RetireDeferralTimeout);
```

これにより以下の統一経路が完成する：

```text
AudioEngine
  └── quarantineSlot(slot, generation, reason)

```text
AudioEngine
  └── quarantineSlot(slot, generation, reason)   ← ★★★ 集約関数
       ├── RetireRuntimeEx::quarantine(slot)      → lane管理
       ├── DSPHandleRuntime::quarantine(handle)   → slot state（Quarantined）
       └── DSPQuarantineManager::quarantineHandle() → 監査記録
```

**★★★ CRITICAL: quarantineSlot() 集約関数**

上記3系統（RetireRuntimeEx / DSPHandleRuntime / DSPQuarantineManager）を別々の箇所から呼ばせると、以下の非同期状態が発生する：

| RetireRuntimeEx | DSPHandleRuntime | 問題 |
| :--- | :--- | :--- |
| Quarantine | Active | RetireLane のみ隔離。DSP へのアクセスは継続可能 |
| Retired | Quarantined | DSP は隔離済みだが RetireLane が通常経路を継続 |

上記の非同期状態は、いずれかの quarantine 呼び出しが失敗した場合に発生する。

**対策**: `AudioEngine` に以下の集約関数 `quarantineSlot()` を追加し、3系統の quarantine を1つのトランザクションとして実行する：

```cpp
// AudioEngine.h — 追加メソッド
// quarantineSlot: 3系統の隔離を1つのトランザクションとして実行
// RetireRuntimeEx + DSPHandleRuntime + DSPQuarantineManager の3系統すべてを
// 同一関数内で呼び出し、いずれかが失敗した場合は即座に証跡を残す。
// これにより以下の非同期状態を防止する：
//   - RetireLane=Quarantine かつ DSPState=Active
//   - RetireLane=Retired かつ DSPState=Quarantined
bool quarantineSlot(uint32_t slot, uint64_t generation,
                    convo::isr::QuarantineReason reason) noexcept
{
    ASSERT_NON_RT_THREAD();

    // 1. RetireLane の quarantine 遷移
    retireRuntimeEx_.quarantine(slot);

    // 2. DSPHandle の state を Quarantined に設定（generation 一致不要）
    dspHandleRuntime_.quarantineSlot(slot);

    // 3. 隔離理由を記録
    dspQuarantineManager_.quarantineHandle(slot, generation, reason);

    // ★ 証跡: 全3系統完了後に emitOwnershipTrace で状態確認
    //    実際のトレース出力は開発時のみ有効（リリースビルドでは条件付き）
#ifdef CONVOPEQ_DEBUG
    emitOwnershipTrace(evidenceRoot / "quarantine_trace.json");
#endif

    return true;
}
```

**注意**: `quarantineSlot()` は常に3系統すべてを実行する。部分的失敗（たとえば `quarantineSlot()` 内の1系統が no-op）が発生した場合でも、証跡トレースにより事後調査が可能。`DSPHandleRuntime::quarantine(DSPHandle)`（generation 一致が必要な厳格パス）は維持するが、当面 `quarantineSlot()` 経路が優先される。

### 3.3 create() は改修不要

検証の結果、`DSPHandleRuntime::create()` は `state == Reclaimed` のみを再利用条件としており、`Quarantined` 状態のスロットは触らない。したがって初版計画書にあった create() 改修は **不要**。

### 3.4 resolve() も現状で十分

`DSPHandleRuntime::resolve()` は既に `state == Quarantined` の場合に `{nullptr, false, false}` を返す。`DSPQuarantineManager` の参照を resolve に注入する必要はない。

### 3.5 Authority の一本化 — QuarantineManager を隔離判定の Authority とする

**★★★ 変更: DSPQuarantineManager を隔離判定の唯一の Authority とする。**

従来計画では `DSPHandleRuntime` を Authority としていたが、実運用では以下の問題がある：

1. `DSPHandleRuntime::quarantine()` が成功しても `dspQuarantineManager_.quarantineHandle()` が失敗すると、DSPState のみ Quarantined で監査情報がない状態になる
2. 逆に `quarantineHandle()` が成功しても `DSPHandleRuntime::quarantine()` が失敗すると、監査情報のみ存在し DSPState が Quarantined にならない
3. 二重管理により「どちらが正しい状態か」の判断が運用時に困難

**修正方針**: `DSPQuarantineManager` を隔離状態の唯一の truth store（source of truth）とする。`DSPHandleRuntime` と `RetireRuntimeEx` は projection（派生状態）であり、truth store の変更を契機に更新されるが、truth の決定権は持たない。

**★★★ 1 truth + 2 projections モデル**:

```text
DSPQuarantineManager ← 唯一の truth store
    ├── DSPHandleRuntime → projection（resolve() 経由の読み取り専用）
    └── RetireRuntimeEx → projection（lane管理の自動反映）
```

```cpp
// ★ quarantineSlot() の実装 — 1 truth + 2 projections
bool AudioEngine::quarantineSlot(uint32_t slot, uint64_t generation,
                                convo::isr::QuarantineReason reason) noexcept
{
    ASSERT_NON_RT_THREAD();

    // ★ Step 1: Truth store 更新（唯一の隔離判定）
    dspQuarantineManager_.quarantineHandle(slot, generation, reason);

    // ★ Step 2: Truth 確認（quarantineHandle が no-op だった場合は隔離しない）
    const auto entry = dspQuarantineManager_.getEntry(slot);
    // ★ Entry の resolved が false かつ quarantineActiveFlags_ が true なら隔離成立
    const bool flagActive = convo::consumeAtomic(
        dspQuarantineManager_.quarantineActiveFlags_[slot], std::memory_order_acquire);
    if (!flagActive)
        return false;

    // ★ Step 3: Projection 更新（truth を反映。独立した状態変更ではない）
    dspHandleRuntime_.quarantineSlot(slot);   // projection: resolve() が Quarantined を返すように
    retireRuntimeEx_.quarantine(slot);        // projection: RetireLane が Quarantine に

    return true;
}
```

**重要な設計判断**: `DSPHandleRuntime` と `RetireRuntimeEx` の quarantine 状態は `DSPQuarantineManager` の truth から導出される。万が一 projection 更新が失敗しても、truth store が正しければ後続の定期同期または次回 quarantineSlot() 呼び出しで修正される。

**resolve() の変更**: `DSPHandleRuntime::resolve()` は従来 slot state のみを参照していたが、truth store との整合性確認を追加する：

```cpp
// ISRDSPHandle.cpp — resolve() の拡張
// ★ state==Quarantined の場合でも、QuarantineManager の truth と整合性確認
ResolvedDSP DSPHandleRuntime::resolve(DSPHandle handle) const noexcept
{
    // ...既存の generation/state チェック...
    if (state == DSPState::Quarantined) {
        // QuarantineManager の truth と整合性を確認
        // 不整合が検出された場合はログ出力し、state を補正
        const auto qEntry = /* QuarantineManager 参照 */;
        if (!qEntry || !qEntry->resolved) {
            // truth が存在しない → projection の誤り。state を元に戻す
            // ただし shutdown 中は補正しない（破壊リスク回避）
            if (!engine.isShutdownInProgress())
                restoreSlotState(handle.slot, DSPState::Active);
        }
        return { nullptr, false, false };
    }
    // ...
}
```

**注意**: この truth + projection モデルでは、`DSPQuarantineManager` の `quarantineActiveFlags_` が唯一的な隔離判定の基準となる。`DSPHandleRuntime::resolve()` はこのフラグを直接確認するか、slot state と cross-reference することで二重真実状態を防止する。

**DSPQuarantineManager の API 変更**: `isQuarantined()` は提供しない方針を維持。隔離状態の問い合わせは従来通り `DSPHandleRuntime::resolve()` 経由とするが、`getEntry(slot)` で隔離情報の有無を確認可能。

**二重状態の防止効果**:

| 状態 | DDRuntime state | QuarantineMgr Entry | 意味 |
| :--- | :--- | :--- | :--- |
| 正常 | != Quarantined | なし | OK |
| 隔離成立 | Quarantined | あり | ✅ 正しい状態 |
| (不一致) | Quarantined | なし | quarantineSlot の Entry 作成が失敗（ログ推奨） |
| (不一致) | != Quarantined | あり | reclaimSlot で削除忘れ（ログ推奨） |

**注意**: 上記の不一致状態は `quarantineSlot()` 集約関数内で Entry 作成を先に行うことでほぼ発生しない。不一致が検出された場合はログ出力し、後続の定期監査で自動修復される。

`reclaimSlot()` は両方を同時に解除する設計とする：

**重要: 解除順序は必ず `DSPHandleRuntime` → `DSPQuarantineManager` とする。**
`DSPHandleRuntime` が state 変更の Authority であり、`DSPQuarantineManager` は監査情報削除の従属動作。
QuarantineManager のデータに依存して DSPHandleRuntime の操作を決定してはならない。

```cpp
// ★ forceReclaim() は危険であり廃止。代わりに shutdown 専用の destroyQuarantineSlot() を追加。
//    通常の reclaim(DSPHandle) は generation 一致を前提とするため、
//    Quarantine 経由では slot が再利用済み（generation 更新済み）の可能性がある。
//    その場合は old generation での reclaim が失敗する。
//    そのため destroyQuarantineSlot は state==Quarantined の表明（assertion）を入れ、
//    原因不明のまま slot が reuse される経路を完全に断つ。
//    shutdown の ShutdownPhase::ReclaimComplete 以降でのみ呼び出し可能。
void DSPHandleRuntime::destroyQuarantineSlot(
    uint32_t slot, uint64_t expectedGeneration) noexcept
{
    if (slot >= MAX_DSP_SLOTS) return;

    // ★ generation 保護: expectedGeneration が指定されている場合、
    //    現在の registry_[slot].generation と一致することを確認する。
    //    これにより誤った slot 解放（generation 更新後の再利用スロット等）を検出する。
    //    shutdown 専用のパスであっても、万一 generation が進んでいる slot を
    //    誤って解放するリスクを防止するための保険。
    if (expectedGeneration != 0) {
        const auto currentGen = convo::consumeAtomic(
            registry_[slot].generation, std::memory_order_acquire);
        if (currentGen != expectedGeneration)
            return;
    }

    // ★ state==Quarantined を表明（安全策: 隔離状態でない slot を誤って解放しない）
    //    通常運用では destroyQuarantineSlot の呼び出しは「隔離を確認してから解放」の順序が必須。
    //    assert により開発時に違反を検出し、隔離不明のまま slot が reuse される経路を完全防止する。
    const auto prevState = convo::consumeAtomic(registry_[slot].state, ...);
    assert(prevState == DSPState::Quarantined);
    if (prevState != DSPState::Quarantined) return;

    // ★★ CRITICAL: Active/Fading/Crossfade 保護 + DestroyPending 2段階化
    //    destroyQuarantineSlot は shutdown 専用パスだが、以下の TOCTOU 競合が理論上成立する：
    //      Thread A: チェック → instance = nullptr
    //      Thread B: crossfade 開始（チェックと代入の間に割り込む）
    //    → instance=nullptr 後に activeHandle へ昇格すると Use After Free 相当になる。
    //
    //    これを防止するため、以下の2段階化を行う：
    //
    //    Phase 1: 状態チェック + DestroyPending マーク
    //      - getActiveRuntimeDSPHandle() がこの slot を保持していない
    //      - getFadingRuntimeDSPHandle() がこの slot を保持していない
    //      - isSlotInCrossfade(slot) がこの slot を含む crossfade がない
    //      上記すべて false の場合のみ state を DestroyPending に遷移
    //    Phase 2: DestroyPending 確認後、instance 解放
    //      - active/fading/crossfade 側は DestroyPending を拒否する
    //      - 確認後、instance=nullptr + state=Reclaimed
    //
    //    ★ DSPState::DestroyPending の追加が必要（ISRDSPHandle.h の enum）
    const bool activeHandleMatch = (getActiveRuntimeDSPHandle().slot == slot);
    const bool fadingHandleMatch = (getFadingRuntimeDSPHandle().slot == slot);
    const bool inCrossfade = isSlotInCrossfade(slot);

    if (activeHandleMatch || fadingHandleMatch || inCrossfade)
        return;

    // Phase 1: DestroyPending マーク
    // ★ CAS により、他スレッドが state を変更していないことを確認
    auto expected = convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire);
    while (expected == DSPState::Quarantined) {
        if (convo::compareExchangeAtomic(registry_[slot].state,
                                         expected, DSPState::DestroyPending,
                                         std::memory_order_acq_rel,
                                         std::memory_order_acquire))
            break;
    }
    if (expected != DSPState::Quarantined)
        return;

    // Phase 2: 念のため再確認（lock-free。実際の競合は CAS が保証）
    const auto finalState = convo::consumeAtomic(registry_[slot].state, std::memory_order_acquire);
    assert(finalState == DSPState::DestroyPending);

    registry_[slot].instance = nullptr;
    convo::publishAtomic(registry_[slot].state, DSPState::Reclaimed, ...);
}

// AudioEngine 側の呼び出し（shutdown 専用。通常運用では使わない）
void AudioEngine::destroyQuarantineSlotForShutdown(uint32_t slot, uint64_t generation) {
    dspHandleRuntime_.destroyQuarantineSlot(slot, generation);
    dspQuarantineManager_.destroyForShutdown(slot);
}
```

### 3.6 期待効果

- 隔離が「フラグ」から「理由＋世代＋タイムスタンプを持つ管理構造」に昇格
- デッドコードだった `DSPHandleRuntime::quarantine()` が実際に動作
- デッドメンバ変数だった `dspQuarantineManager_` が実使用される
- Authority が `DSPHandleRuntime` に明確化（二重管理防止）
- `DSPQuarantineManager` は純粋な監査役に限定
- 管理者が隔離理由を確認可能（`getEntry()`）
- `destroyQuarantineSlot()` による shutdown 専用解放パス（通常運用では使わない）
- forceReclaim() は廃止。原因不明のまま slot が reuse される経路を完全排除

---

## 4. Phase A-2: Shutdown完全閉包

### 4.1 現状

- `ShutdownRuntime` クラスは存在し、`ShutdownPhase`（Running〜ShutdownComplete）を管理
- `isShutdownInProgress()` は **2系統** で異なる実装:
  - `AudioEngine::isShutdownInProgress()`: `lifecycleState` (EngineLifecycleState::Releasing/Destroyed) を参照
  - `ShutdownRuntime::isShutdownInProgress()`: `phase_ != Running && phase_ != ShutdownComplete` を参照
- `PublicationAdmission::evaluate()` は `engine.isShutdownInProgress()` を呼ぶ（AudioEngine側）
- `RuntimePublicationOrchestrator::notifyTransitionComplete()` に shutdown ガードなし
- シャットダウン完了時の DrainAudit 不在
- Shutdown FSM と EBR/Retire の最終収束確認がない

### 4.2 改修内容

#### 4.2.1 Shutdown Authority — OR 判定を永久維持

**完全委譲は行わない。** 実運用の破綻耐性を優先し、`EngineLifecycleState` と `ShutdownRuntime` の OR 判定を永久維持する。`AudioEngine::isShutdownInProgress()` は以下を OR で結合する：

**対象ファイル**: `src/audioengine/AudioEngine.h`, `ISRShutdown.h`

```cpp
// AudioEngine.h — Phase A-2 以降（永久 OR 判定）
[[nodiscard]] bool isShutdownInProgress() const noexcept
{
    // ★ OR 判定を永久維持する。数学的な一本化より実運用の破綻耐性を優先。
    //   EngineLifecycleState と ShutdownPhase の間には移行期間の乖離が発生し得るため、
    //   両方を確認する二重保護を常に維持する。
    //   ShutdownRuntime のみへの完全委譲は行わない。
    const auto lifecycleState = consumeAtomic(lifecycleState, std::memory_order_acquire);
    const bool lifecycleShutdown = (lifecycleState == EngineLifecycleState::Releasing
                                 || lifecycleState == EngineLifecycleState::Destroyed);
    return lifecycleShutdown || shutdownRuntime_.isShutdownInProgress();
}
```

**設計判断**: 完全委譲（`ShutdownRuntime` のみ）への移行は行わない。`EngineLifecycleState` は `Releasing`/`Destroyed` を、`ShutdownRuntime` は `ShutdownPhase` を管理しており、これらは異なるライフサイクル視点である。両者を OR で結合することで「どちらかが shutdown 状態なら shutdown とみなす」安全側の判定を恒久的に維持する。これにより `EngineLifecycleState = Releasing` かつ `ShutdownRuntime = Running` のような過渡期乖離があっても publish が通るリスクを防止する。

**AudioEngine one-shot lifecycle に関する注記**:

`EngineLifecycleState::Destroyed` は終端状態であり、AudioEngine の再初期化（同一インスタンスの再利用）はサポートしない。そのため `isShutdownInProgress()` は `Destroyed` 到達後永久に `true` を返す。将来 AudioEngine の再初期化が必要になった場合は、`lifecycleState` のリセット機構と `isShutdownInProgress()` の復帰条件を別途設計すること。

**OR 永久維持の副作用と対策**:

`return lifecycleShutdown || shutdownRuntime_.isShutdownInProgress()` の OR 判定は、`Destroyed` 到達後永久に `true` を返す。これは「shutdown 中に誤った publish を通さない」という安全側の設計だが、以下の副作用がある：

1. Engine のリセット再利用（hot reload / plugin reinit）が事実上不可能
2. テスト harness の再初期化が `isShutdownInProgress()==true` により阻害される
3. 長期間稼働後の lifecycle state が不可逆になる

**推奨ガード（将来対応）**: 将来リセット再利用が必要になった場合は、OR 条件に `Resetting` のゲーティングを追加する：

```cpp
// 将来のリセット再利用対応（Phase D 以降）
// ★ EngineLifecycleState::Resetting を enum に追加した場合の OR 条件
const bool lifecycleShutdown = (lifecycleState == EngineLifecycleState::Releasing
                             || lifecycleState == EngineLifecycleState::Destroyed);
const bool isResetting = (lifecycleState == EngineLifecycleState::Resetting);
return (lifecycleShutdown || shutdownRuntime_.isShutdownInProgress())
    && !isResetting;
```

ただし現段階では `Resetting` の追加は行わない。AudioEngine の one-shot lifecycle を前提とし、将来の要件変更時に対応する。

**★★★ Shutdown watchdog**: `EpochSettled` 到達後、300秒以上 `ShutdownComplete` に進まない場合の監査機構を追加する。AudioEngine の Timer または監査スレッドで定期実行する。300秒以上残留した quarantine を検出した場合、強制解放は行わずログ出力のみとする。Practical Stable の思想では「強制破壊より停止してログ」が安全である。

```cpp
// AudioEngine の Timer または定期監査パス — Shutdown watchdog 実装例
if (shutdownRuntime_.getPhase() >= convo::isr::ShutdownPhase::EpochSettled
    && shutdownRuntime_.getPhase() < convo::isr::ShutdownPhase::ShutdownComplete)
{
    const auto audit = collectDrainAudit();
    if (audit.quarantineResident > 0 && audit.maxQuarantineAgeSec > 300) {
        juce::Logger::writeToLog(
            "[ISR][ShutdownWatchdog] Quarantine leak detected: "
            + juce::String(static_cast<int64>(audit.quarantineResident))
            + " entries, max age=" + juce::String(static_cast<int64>(audit.maxQuarantineAgeSec)) + "s");
        emitOwnershipTrace(evidenceRoot / "shutdown_watchdog_quarantine_leak.json");
    }
}
```

また `EngineLifecycleState` と `ShutdownPhase` の OR 判定は永久維持する。両者を一本化しない理由：

- `lifecycleState` は AudioEngine 全体の状態（高レベル）
- `ShutdownPhase` は ISR Runtime の shutdown FSM 進行度（低レベル）

これらは異なるライフサイクル視点であり、移行期間に乖離が発生し得る。OR 判定により「どちらかが shutdown 状態なら shutdown とみなす」安全側を維持する。

**PublicationAdmission::evaluate()** は変更不要（`engine.isShutdownInProgress()` の実装変更のみ）。

#### 4.2.2 notifyTransitionComplete に shutdown ガード追加

**対象ファイル**: `src/audioengine/RuntimePublicationOrchestrator.cpp`

```cpp
void RuntimePublicationOrchestrator::notifyTransitionComplete(
    AudioEngine::DSPCore* currentAfterFade) noexcept
{
    if (currentAfterFade == nullptr)
        return;

    transition_.onTransitionComplete(currentAfterFade);

    // shutdown 中は deferred 再投入をキャンセル（残留タスク防止）
    if (engine_.isShutdownInProgress()) {
        if (hasDeferred_) {
            deferredRequest_.reset();
            hasDeferred_ = false;
        }
        return;
    }

    // [PR-7] クロスフェード完了後、deferred publish request を再試行する。
    if (hasDeferred_) {
        auto deferredReq = consumeDeferredRequest();
        if (deferredReq.has_value())
            submitPublishRequest(std::move(*deferredReq));
    }
}
```

#### 4.2.3 RuntimeDrainAudit 構造体の追加

**新規ファイル**: `src/audioengine/RuntimeDrainAudit.h`

```cpp
#pragma once
#include <cstdint>

namespace convo::isr {

// RuntimeDrainAudit: Shutdown 完了条件の監査構造体。
// isAllZero() が完了条件。quarantineResident は監査項目であって完了条件ではない。
//
// ■ 完了条件に含めるもの
//   pendingRetire  — RetireRuntime に未処理の retire intent がない
//   routerPendingRetire — ISRRetireRouter に滞留中の retire item 数
//   activeCrossfade — 進行中のクロスフェードがない
//   deferredPublish — 未投入の deferred publish がない
//   pendingPublication — RuntimePublicationCoordinator に pending publish がない
//   maxDeferredAgeMs — deferred publish 最長滞留時間（オプション．F-8.4.6参照）
//
// ■ 監査のみ（完了条件にしない）
//   quarantineResident — 隔離保留中のエントリ数。shutdown 完了を妨げない
//   oldestPendingAgeMs — 最長滞留時間。shutdown 完了条件にはしないが、
//                        設計段階から構造体に含めておく
//                        （AudioEngine::oldestPendingAge_ を consumeAtomic 読み取り。新規API不要）
struct RuntimeDrainAudit {
    uint64_t pendingPublication;
    uint64_t pendingRetire;
    uint64_t activeCrossfadeCount;   // ★ 常に件数表現（A-2は0/1、将来は0..N）。isPending()のboolではない。
    uint64_t routerPendingRetire;    // ISRRetireRouter 滞留 item 数（pendingRetire とは別段階。重複計上に注意）
    uint64_t deferredPublish;
    uint64_t maxDeferredAgeMs;       // deferred publish 最長滞留時間（0=滞留なし）
    uint64_t quarantineResident;   // 監査のみ
    uint64_t oldestPendingAgeMs;   // 監査のみ（AudioEngine::oldestPendingAge_ 参照。新規API不要）
    uint64_t maxQuarantineAgeSec;  // 最長 quarantine 経過時間（秒）。TTL超過検出用（F-8.4.6参照）

    // ★ isAllZero() は監査ログ出力専用。shutdown 完了判定の authority にはしない。
    //    実際の shutdown 完了判定は既存の isFullyDrained() ロジックを維持する
    //    （RuntimePublicationCoordinator::isFullyDrained() + !hasDeferredCommit の既存2重チェック）。
    //    isAllZero() は ReleaseResources 内のログ出力や emitRetireTrace の発火条件としてのみ使用する。
    //    これにより shutdown 判定と監査表示を分離し、どちらかの変更が他方に影響するのを防止する。
    //
    // ★ activeCrossfadeCount は isAllZero() に含める。
    //    理由: notifyTransitionComplete は crossfade 完了後に呼ばれるため、
    //    activeCrossfade > 0 は transition 未完了を意味する。空の shutdown では
    //    crossfade が発生しないため activeCrossfadeCount==0 が正常。
    //    crossfade が完了しないまま isAllZero()==true になることはない（安全側）。
    //
    // ★ 含めないもの:
    //    quarantineResident — 隔離保留中のエントリ。shutdown 完了を妨げない。
    //    oldestPendingAgeMs — 監視情報。完了条件にすると古い pending が復旧を阻害する。
    //    routerPendingRetire — ISRRetireRouter 滞留 item 数。
    //        pendingRetire（RetireRuntime）とは別段階の滞留。
    //        ただし同一 retire 経路を2段階で計上している可能性があるため、
    //        両方が同時に非ゼロの場合、同じ1件が2箇所でカウントされている可能性を考慮すること。
    //        隔離済み DSP 由来なら shutdown 完了を妨げない。
        // ★★★ BlockingReason: shutdown 完了を阻害している主要因を特定する
    //     getPrimaryBlockingReason() は isAllZero()==false の場合に原因特定を支援する。
    //     ログ出力だけでなくプログラムからも参照可能にすることで、
    //     運用障害解析が劇的に容易になる。
    enum class BlockingReason : uint8_t {
        None,
        PendingPublication,
        PendingRetire,
        ActiveCrossfade,
        DeferredPublish,
        QuarantineResident,
        RouterPendingRetire,
        Unknown
    };

    [[nodiscard]] BlockingReason getPrimaryBlockingReason() const noexcept {
        if (pendingPublication > 0)    return BlockingReason::PendingPublication;
        if (pendingRetire > 0)         return BlockingReason::PendingRetire;
        if (activeCrossfadeCount > 0)  return BlockingReason::ActiveCrossfade;
        if (deferredPublish > 0)       return BlockingReason::DeferredPublish;
        if (quarantineResident > 0)    return BlockingReason::QuarantineResident;
        if (routerPendingRetire > 0)   return BlockingReason::RouterPendingRetire;
        return BlockingReason::Unknown;
    }

    // ★ isAllZero() が false かつ getPrimaryBlockingReason() から原因が特定できない場合は
    //    oldestPendingAgeMs / maxQuarantineAgeSec / maxDeferredAgeMs の値も併せて確認すること。
    bool isAllZero() const noexcept {
        return pendingPublication == 0
            && pendingRetire == 0
            && activeCrossfadeCount == 0
            && deferredPublish == 0;
    }
};

} // namespace convo::isr
```

#### 4.2.6 activeCrossfade 取得 — 既存 API を利用（新規追加不要）

DrainAudit の `activeCrossfade` は新規 API を追加せず、**既存の `CrossfadeAuthorityRuntime` を利用する**。

**判断理由**:

- `CrossfadeAuthorityRuntime::getActiveCrossfades()` が既に存在し、crossfade の一覧を vector で返す
- `CrossfadeRuntime::isPending()` が既に存在し、crossfade の有無を二値判定可能
- 新たに `DSPTransition::activeTransitionCount()` を追加すると、既存の CrossfadeAuthority との二重管理リスクが生じる
- 実運用監査に必要なのは「滞留 crossfade が存在するか」の有無情報であり、`isPending()` で十分

```cpp
// DrainAudit での利用 — 新規 API 不要
.activeCrossfade = crossfadeRuntime_.isPending() ? 1u : 0u,
// より詳細が必要な場合は CrossfadeAuthorityRuntime の getActiveCrossfades() を利用
// auto records = crossfadeAuthorityRuntime_.getActiveCrossfades();
// .activeCrossfadeCount = records.size(),
```

`getActiveCrossfades()` は内部で `std::vector` の確保を行うが、DrainAudit は NonRT スレッドで呼ばれるため問題ない。

**注意**: `activeCrossfade` は DrainAudit の完了条件（`isAllZero() const`）に**含める**。理由: `notifyTransitionComplete` は crossfade 完了後に呼ばれるため、`activeCrossfade > 0` は transition 未完了を意味する。空の shutdown では crossfade が発生しないため `activeCrossfadeCount==0` が正常。crossfade が完了しないまま `isAllZero()==true` になることはない。

#### 4.2.7 isFullyDrained() の強化

**対象ファイル**: `src/audioengine/AudioEngine.Threading.cpp`

```cpp
bool AudioEngine::isFullyDrained() noexcept
{
    // ★★★ isAllZero() は監査ログ専用。shutdown 完了判定の authority にはしない。
    //     既存の RuntimePublicationCoordinator::isFullyDrained() + !hasDeferredCommit を維持する。
    //     collectDrainAudit() は ReleaseResources 内のログ出力および emitRetireTrace の
    //     発火条件としてのみ使用する。
    //     これにより shutdown 判定（isFullyDrained）と監査表示（DrainAudit）を分離し、
    //     どちらかの変更が他方に影響するのを防止する。
    return runtimePublicationBridge_.isFullyDrained() && !hasDeferredCommit;
}

// ★★★ collectDrainAudit() は EpochSettled phase 到達後にのみ呼び出すこと。
//     EpochSettled 未到達で DrainAudit を取得すると、進行中の publish/retire/crossfade
//     により各カウンタが snapshot 時点で変動中であり、一貫性のない監査結果になる。
//     特に shutdown 完了判定前に DrainAudit を参照する場合、以下の順序を厳守：
//       1. ShutdownRuntime::advancePhase() → EpochSettled
//       2. collectDrainAudit() → 監査結果取得
//       3. isFullyDrained() または audit 内容の判断
//     これにより snapshot 取得中の状態変化リスクを最小化する。
//     各フィールドは独立した atomic 読み取りであり、全体の atomic スナップショットではない点に注意。
RuntimeDrainAudit AudioEngine::collectDrainAudit() noexcept
{
    return RuntimeDrainAudit{
        .pendingPublication = /* RuntimePublicationCoordinator に publicationBacklogCount() getter 追加 */,
        .pendingRetire = retireRuntime_.pendingIntentCount(),
        .activeCrossfadeCount = crossfadeRuntime_.isPending() ? 1u : 0u,  // 4.2.6 判断: 新規API不要
        .routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount()),
        .maxDeferredAgeMs = runtimeOrchestrator_->getMaxDeferredAgeMs(),
        .deferredPublish = runtimeOrchestrator_->hasDeferredRequest() ? 1u : 0u,
        .quarantineResident = dspQuarantineManager_.residentCount(),  // 監査のみ
        // ★ retireRuntimeEx_.oldestPendingAgeMs() は不要（Appendix B-3）。
        //    既存の AudioEngine::oldestPendingAge_ を consumeAtomic で読み取る。
        .oldestPendingAgeMs = static_cast<uint64_t>(
            convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire))
    };
}
```

**完了条件の明確化**: `pendingPublication` は `isAllZero()` の完了条件に含める。`// TODO` で放置せず、Phase A-2 内で必ず `RuntimePublicationCoordinator::getPublicationBacklogCount()` の実測値で埋めること。A-2 の完了条件に「全フィールドが実測値であること」を含める。`collectDrainAudit()` は監査ログ専用であり、shutdown 完了判定には使用しない。実際の shutdown 完了は既存 `isFullyDrained()` のロジック（`RuntimePublicationCoordinator::isFullyDrained()` + `!hasDeferredCommit`）を維持する。

**RuntimePublicationSnapshot**: 実装の容易さと将来の拡張性を考慮し、以下のスナップショット構造体を推奨：

```cpp
// RuntimePublicationSnapshot: DrainAudit が Coordinator から取得する値の構造化
// TODO を残さず Phase A-2 内で実装完了すること
struct RuntimePublicationSnapshot {
    uint64_t publicationBacklog;   // setPublicationBacklogCount() に対応
    uint64_t retireBacklog;        // setRetireBacklogCount() に対応
    uint64_t pendingIntent;        // setPendingIntentCount() に対応
    uint64_t reclaimInFlight;      // getReclaimInFlightCount() で取得可
};
```

各値は Phase A-2.6 で追加する getter から取得し、`collectDrainAudit()` 内で `RuntimePublicationSnapshot` に格納して `RuntimeDrainAudit` へマッピングする。

#### 4.2.8 Shutdown最終段での監査とレポート

#### 4.2.5 Shutdown最終段での監査とレポート

**対象ファイル**: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp`

```cpp
// 既存の releaseResources 内の shutdown 完了判定部を強化
// ★ emitRetireTrace は EpochSettled 以降で呼ぶ（別スレッドの lane 更新中との競合防止）
const auto shutdownPhase = shutdownRuntime_.getPhase();
const bool traceSafe = (shutdownPhase >= convo::isr::ShutdownPhase::EpochSettled);

const auto audit = collectDrainAudit();
if (!drainedWithinBudget || !audit.isAllZero()) {
    juce::Logger::writeToLog(
        "[ISR][Shutdown] Drain incomplete: "
        "pendingPub=" + String(static_cast<int64>(audit.pendingPublication)) +
        " pendingRetire=" + String(static_cast<int64>(audit.pendingRetire)) +
        " crossfade=" + String(static_cast<int64>(audit.activeCrossfadeCount)) +
        " routerPendingRetire=" + String(static_cast<int64>(audit.routerPendingRetire)) +
        " maxDeferredAgeMs=" + String(static_cast<int64>(audit.maxDeferredAgeMs)) +
        " deferred=" + String(static_cast<int64>(audit.deferredPublish)) +
        " quarantine=" + String(static_cast<int64>(audit.quarantineResident)) +
        " oldestAgeMs=" + String(static_cast<int64>(audit.oldestPendingAgeMs)) +
        " (observation only)");

    // 異常時のみ retire trace を出力（正常 shutdown では出力しない）
    // 出力ファイルは毎回上書き（_last.json 方式）でディレクトリ肥大化防止
    // ★ emitRetireTrace は完全 noexcept とする（shutdown パスでの例外投擲を防止）
    if (traceSafe)
        retireRuntimeEx_.emitRetireTrace(evidenceRoot / "retire_trace_shutdown_last.json");
}

// quarantine resident が残っている場合は警告を出力
if (audit.quarantineResident > 0) {
    juce::Logger::writeToLog(
        "[ISR][Shutdown] Drain complete but quarantine residents remain: "
        + String(static_cast<int64>(audit.quarantineResident)));
}
```

#### 4.2.9 ShutdownPhase 拡張は Phase D（将来）へ延期

**判断**: 実運用の安定性向上効果が小さいため、ShutdownPhase 拡張は後回しとする。

- 本当に必要なのは notifyTransitionComplete shutdown guard + DrainAudit + authority 統一であり、新 phase 追加は必須ではない
- 新 phase 追加による switch 漏れリスクを現時点で負う必要はない
- Phase A-2 の完了条件に ShutdownPhase 拡張は含めない

ただし実装前に switch 網羅性監査だけは実施しておく。全 `switch(ShutdownPhase)` 箇所を検索し、以下のいずれかのパターンであることを確認する：

```cpp
// 実施（事前調査のみ。Phase A-2 では enum 変更しない）
Select-String -Pattern 'ShutdownPhase' -Path src/**/*
```

1. `default:` で未列挙の値に対応している
2. 全列挙子を網羅している（`case Running:` ... `case ShutdownComplete:`）

**ShutdownPhase 拡張は Phase D（将来）へ延期**
enum 変更は行わない。`isShutdownInProgress()` のロジック変更と `notifyTransitionComplete` shutdown guard のみで Phase A-2 は完了する。

### 4.3 期待効果

- Shutdown 判定を OR で永久維持（破綻耐性優先。完全委譲は行わない）
- `notifyTransitionComplete` の shutdown ガードにより、shutdown 開始後の Deferred Publish 再投入を完全防止
- 5項目の DrainAudit で「何が残っているか」を可視化（quarantineResident は監査のみ）
- crossfade 滞留は既存 `CrossfadeRuntime::isPending()` / `CrossfadeAuthorityRuntime::getActiveCrossfades()` で実測（新規 API 不要）
- DrainAudit の全フィールドが TODO ではなく実測値で埋まる
- Shutdown 不完全時のログ出力でデバッグ性向上

#### 4.2.9 Shutdown 3-phase モデル — publish 抑止の明確化

**問題**: 現在の ShutdownPhase は7段階で細粒度だが、`PublicationAdmission::evaluate()` の shutdown チェック段階が不明確。以下の ordering undefined が発生しうる：

1. Shutdown 開始 → deferred publish reset
2. crossfade completion callback → publish 再実行
3. publish が通る（shutdown 開始直後の過渡期）

**対策**: ShutdownPhase を以下の3層に分類し、`evaluate()` のチェックを Draining 相当以降に固定する：

| 層 | 該当 Phase | 許可される動作 |
| :--- | :--- | :--- |
| **Active** | Running | 全動作許可 |
| **Draining** | AudioStopped / ObserverDrained / RetireClosed / EpochSettled | **publish 禁止** / retire 許可 / crossfade 完了待ち |
| **Finalizing** | ReclaimComplete / ShutdownComplete | 全イベント無効 / メモリ解放のみ |

```cpp
// PublicationAdmission::evaluate() — 追加チェック
// Draining 以降は publish を完全拒否。shutdown 開始後の publish 再実行を防止
if (engine.shutdownPhase() >= ShutdownPhase::ObserverDrained)
    return Decision::RejectedShutdown;
```

**注意**: ShutdownPhase 列挙子は変更しない。上記の3層分類は動作許可の論理的なガイドラインであり、enum 追加は行わない。

---

## 5. Phase B-1: Reclaim完結保証

### 5.1 現状

- `RetireRuntimeEx` に `RetireLifecycleState`（6段階）と `RetireLane`（5段階）が存在
- `lifecycleStateOf(slot)` で各 slot の状態を取得可能
- `laneOf(slot)` で各 slot の lane を取得可能
- `lifecycleCounters_` で各状態の累積カウントを追跡中
- しかし「現時点でどの slot がどの段階で止まっているか」のスナップショットダンプが存在しない
- `retiredWorldCount_` はカウントのみ。個別スロットの滞留可視化がない

### 5.2 改修内容

#### 5.2.1 ReclaimTicket は採用しない

初版計画書で提案した `ReclaimTicket`（slot/generation/retireCompleted/epochSettled/reclaimExecuted の構造体）は **過剰設計** である。`RetireRuntimeEx` は既に slot 単位の state と lane を追跡しているため、別途 ticket 構造を追加すると二重管理になる。

カウンタベースの `lifecyclePendingCount()` も不十分（どの slot が止まったかが分からない）。

#### 5.2.2 emitRetireTrace() によるスナップショットダンプ

**対象ファイル**: `src/audioengine/ISRRetireRuntimeEx.h`, `ISRRetireRuntimeEx.cpp`

slot 単位の現在状態を JSON ダンプする関数を追加：

```cpp
// ISRRetireRuntimeEx.h — 追加メソッド
// emitRetireTrace: 全 slot の現在ライフサイクル状態を JSON 出力
// 調査用。shutdown 直前や定期監査で呼ぶ。
// emitRetireTrace: 全 slot の現在ライフサイクル状態を JSON 出力
    // ★ 完全 noexcept（shutdown パスでの例外投擲を防止するため）
    //    std::ofstream のエラーは無視する
    void emitRetireTrace(const std::filesystem::path& outputPath) const noexcept;
```

**I/O 分離方針**: 付録D-2 の検証により、`DebugRuntime` に `enqueueJsonOutput()` は**存在しない**ことが確認された。そのため以下の方針とする：

1. **Shutdown 時**: `std::ofstream` 直接出力（ブロックリスク低。他スレッド停止済みのため）
2. **定期監査時**: `std::ofstream` 直接出力＋位相制限（`EpochSettled` 以降のみ）。Monitor 監査パスではログ肥大化防止のため `oldestPendingAge_ > 30000ms` の異常時のみ出力
3. **DebugRuntime 非同期キュー**: 将来の対応とする。`DebugRuntime` に `enqueueJsonOutput()` を追加する場合は、shutdown 後も生存し queue flush が保証されることを確認した上で移行する

```cpp
// ★ Shutdown 時は std::ofstream 直接出力
//    DebugRuntime::enqueueJsonOutput() は未実装のため、非同期委譲は行わない。
//    shutdown 時は他スレッドが停止しているためブロックリスクは低い。
void RetireRuntimeEx::emitRetireTrace(const std::filesystem::path& outputPath) const noexcept
{
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        return;

    // ... JSON 内容生成（従来通り）...
    // （ofstream 直接書き込み）
}
```

```cpp
void RetireRuntimeEx::emitRetireTrace(const std::filesystem::path& outputPath) const
{
    std::error_code ec;
    std::filesystem::create_directories(outputPath.parent_path(), ec);

    std::ofstream file(outputPath, std::ios::binary | std::ios::trunc);
    if (!file.is_open())
        return;

    file << "{\n  \"schema\": \"retire_trace_v1\",\n";
    file << "  \"totalSlots\": " << kMaxSlots << ",\n";
    file << "  \"slots\": [\n";

    bool first = true;
    for (std::size_t i = 0; i < kMaxSlots; ++i) {
        const auto lane = laneOf(static_cast<uint32_t>(i));
        const auto lifecycle = lifecycleStateOf(static_cast<uint32_t>(i));

        // RTIntent 以外（何らかの処理中）のスロットのみ出力
        if (lane == RetireLane::RTIntent && lifecycle == RetireLifecycleState::Visible)
            continue;

        if (!first)
            file << ",\n";
        first = false;

        file << "    { \"slot\": " << i
             << ", \"lane\": \"" << laneName(lane)
             << "\", \"lifecycle\": \"" << lifecycleStateName(lifecycle)
             << "\" }";
    }

    file << "\n  ],\n";
    file << "  \"counts\": {\n";
    // ... lane と lifecycle のカウンタ出力
    file << "  }\n}\n";
}
```

#### 5.2.3 滞留時間監視: 既存 oldestPendingAge_ の活用

`emitRetireTrace()` は「観測」であって「保証」ではない。滞留の異常を検出するため、`AudioEngine` に **既に存在する** `oldestPendingAge_`（`std::atomic<double>`）を DrainAudit から参照する。

**新規 API は不要**: `AudioEngine::oldestPendingAge_` は `AudioEngine.Commit.cpp` の `onRuntimeRetiredNonRt()` 内で更新済みであり、`RetireRuntimeEx` に新たな API を追加する必要はない。

**DrainAudit との統合**:

```cpp
// AudioEngine::collectDrainAudit() 内 — 監査項目として追加
.oldestPendingAgeMs = static_cast<uint64_t>(
    convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire)),
```

**閾値監視例**（実装時に定数定義）:

```
pending > 0 かつ oldestPendingAge > 30000ms → 異常とみなして診断ログ
```

#### 5.2.4 DrainAudit での活用

`isFullyDrained()` では以下で可視化する：

```cpp
// DrainAudit の routerPendingRetire として RetireRouter 滞留 item 数を計上
// ★ pendingRetire（RetireRuntime pendingIntentCount）とは別段階。
//   同一経路を2段階で計上している可能性があるため、診断時は両方の値を参照すること。
// ★★★ ReclaimAuditSnapshot: emitRetireTrace() のファイル出力だけでは不十分なため、
//    メモリ上のスナップショット構造体も併用する。
//    emitRetireTrace() は shutdown 時の単発調査用。
//    ReclaimAuditSnapshot は定期監査で常時利用可能。
//
//    ReclaimAuditSnapshot 構造体（定期監査用）:
//    struct ReclaimAuditSnapshot {
//        uint64_t generation;           // 現在の world generation
//        uint32_t activeSlots;          // アクティブスロット数
//        uint32_t quarantinedSlots;     // 隔離中スロット数
//        uint32_t pendingRetire;        // RetireRuntime pending intent 数
//        uint32_t routerPendingRetire;  // ISRRetireRouter 滞留 item 数
//        uint64_t oldestPendingAgeMs;   // 最長滞留時間
//    };
//    RetireRuntimeEx::collectReclaimAudit() として実装し、DrainAudit と統合する。
//    これにより emitRetireTrace() のファイル I/O に依存しない、プログラムから参照可能な
//    状態スナップショットが得られる。RT 検証不能問題を緩和する。
.routerPendingRetire = static_cast<uint64_t>(m_retireRouter->pendingRetireCount()),
```

また、Shutdown 直前に `emitRetireTrace()` を呼び出して証跡を残す：

```cpp
// AudioEngine.Processing.ReleaseResources.cpp
retireRuntimeEx_.emitRetireTrace(evidenceRoot / "retire_trace_shutdown.json");
```

**滞留時にも emitRetireTrace を出力（条件付き）**: Shutdown 失敗時だけでなく、`oldestPendingAge_ > 30000ms` の滞留異常が検出された場合も証跡を残す。これにより Shutdown 前に問題を検出可能：

```cpp
// Timer または定期監査パス
// ★ 常時出力せず、異常時のみ出力することで長期運用のログ肥大化を防止
const auto age = convo::consumeAtomic(oldestPendingAge_, std::memory_order_acquire);
if (age > 30000.0) {
    retireRuntimeEx_.emitRetireTrace(evidenceRoot / "retire_trace_stall.json");
}
```

**emitRetireTrace 出力の基本方針**:

- 正常 shutdown では emitRetireTrace を出力しない（`audit.isAllZero()` の場合はスキップ）
- 異常滞留（`oldestPendingAge_ > 30000ms`）時のみ出力
- 正常 shutdown でも `audit.isAllZero()` が false の場合のみ出力
- 出力ファイルは毎回上書き（`retire_trace_shutdown_last.json` 方式）とし、連続生成によるディレクトリ肥大化を防止
- これにより長期運用で retire トレースファイルが肥大化するのを防止

### 5.3 期待効果

- 新規データ構造の追加最小化（既存の `lifecycleStateOf()` / `laneOf()` を活用）
- Shutdown 直前に全 slot の状態スナップショットを取得可能
- **`oldestPendingAgeMs()` により、滞留の「観測」から「異常検出」へ昇格**
- pending>0 かつ oldestPendingAge>30s を異常判定可能
- 「どの slot がどの段階で止まっているか」を可視化
- 既存の `laneCounters_` / `lifecycleCounters_` と併用可能

---

## 6. Phase B-2: Overflow自己防衛

### 6.1 現状

- `RetireRuntime` に `overflowCount_` / `droppedIntentCount_` あり。計測のみ
- `evaluateRetirePressureLevelNoRt()` / `applyRetirePressurePolicyNoRt()` が存在し、4段階（0-3）の圧力レベルを管理
  - level 0: 正常
  - level 1: mild（coalescing active）
  - level 2: medium（throttle active）
  - level 3: severe（strict admission + protective mode）
- level 判定は比率ベース（`retireDepth * 100 / hwm`）
- `retirePressurePublicationThrottleActive_` が level>=2 で true → `PublicationAdmission` で `RejectedPressure`
- overflow 検出そのものから throttle への直接結合がない

### 6.2 改修内容

#### 6.2.1 RuntimePressureState 列挙型は追加しない

既存の `int retirePressureLevel`（0-3）で十分。新たな列挙型を追加すると既存の level 表現との二重管理になるため、**追加しない**。

代わりに、既存の level 定義にコメントを追加して意図を明確化する：

```cpp
// AudioEngine.Retire.cpp — コメント強化（コード変更なし）
// Retire pressure level semantics:
//   level 0: Normal — no restriction
//   level 1: Mild   — coalescing active (>=75% of HWM)
//   level 2: Medium — throttle publication active (>=90% of HWM)
//   level 3: Severe — strict admission + protective mode (>=95% of HWM)
```

#### 6.2.2 Overflow 検出 → drainDeferredRetireQueues での throttle 結合

RT スレッド制約のため、`emitRetireIntent()` 内での直接 throttle 適用は行わない。代わりに NonRT の `drainDeferredRetireQueues()` 内で overflow 状態をチェックし、強制的に severe pressure 扱いにする：

**対象ファイル**: `src/audioengine/AudioEngine.Retire.cpp`

```cpp
// drainDeferredRetireQueues() 内 — 既存の evaluateRetirePressureLevelNoRt 呼び出し直後に追加
// ★ 累積 droppedIntentCount ではなく、前回との差分（delta）を使用する
//    累積値だと一度発生した overflow が永遠に severe を継続させる
const uint64_t droppedTotal = retireRuntime_.droppedIntentCount();
const uint64_t prevDropped = convo::exchangeAtomic(prevDroppedSnapshot_, droppedTotal, ...);
const uint64_t droppedDelta = (droppedTotal > prevDropped) ? (droppedTotal - prevDropped) : 0;

// Overflow 発生中は合成圧力レベルを計算する
// （直接フラグを上書きせず、既存 policy の level に補正をかける）
const bool overflowActive = (droppedDelta > 0);
int overflowLevel = overflowActive ? 3 : 0;  // overflow 時は severe

// ★ 既存 evaluateRetirePressureLevelNoRt() の結果を上書きせず合成する
//    effectiveLevel = max(normalLevel, overflowLevel)
//    これにより overflow 終了後は通常の比率ベース評価に自動復帰する
const auto normalLevel = convo::consumeAtomic(retirePressureLevel_, ...);
const int effectiveLevel = std::max(static_cast<int>(normalLevel), overflowLevel);

// effectiveLevel に基づいて各フラグを設定
// （直接フラグ上書きではなく、既存 applyRetirePressurePolicyNoRt() と同様の分岐）
convo::publishAtomic(retirePressurePublicationThrottleActive_, effectiveLevel >= 2, ...);
convo::publishAtomic(retirePressureAdmissionStrict_, effectiveLevel >= 3, ...);
convo::publishAtomic(retireProtectiveModeActive_, effectiveLevel >= 3, ...);
```

**この合成は `applyRetirePressurePolicyNoRt()` の直後に行うこと**。
`applyRetirePressurePolicyNoRt()` が通常の比率ベースで level を設定した後、overflow 状態に応じて effectiveLevel を合成する。
overflow 終了後は `overflowLevel=0` となるため `effectiveLevel = normalLevel` に自動復帰し、
直接フラグ上書きのようなラッチ状態は発生しない。

**重要**: Overflow 時は **Publication のみ抑制** し、**Retire 処理は常に継続** する。overflow により retire まで停止するとキュー滞留が悪化する。`retirePressureAdmissionStrict_` は publication の admission にのみ適用され、`drainDeferredRetireQueues()` や retire 経路には影響させない。

#### 6.2.3 連続 Overflow 検出: overflowStartTimestamp_ の追加

**対象ファイル**: `src/audioengine/ISRRetire.h`

一時的なスパイクと慢性的な詰まりを区別するため、`RetireRuntime` に overflow 継続時間を追跡するカウンタを追加する。

**設計判断**: `overflowScore_`（overflow→+4, success→-1）は採用しない。理由：

- スコアの物理的意味が不明瞭（継続時間なのか回数なのか）
- RT/NonRT 混在環境での atomic 整合性コスト
- 実運用では「何回起きたか」より「どれだけ長く続いたか」が重要（付録F-4参照）

```cpp
// ISRRetire.h — 追加メンバ
std::atomic<uint64_t> lastOverflowTicks_{0};       // 最終 overflow 発生時刻
std::atomic<uint64_t> overflowStartTimestamp_{0};  // overflow 継続開始時刻（0=非overflow状態）

// ★★★ overflowWindowCounter_ — 30秒窓の overflow 回数カウンタ
//     overflowStartTimestamp_ では検出できない振動パターン
//     （98%→100%→98%→100% の繰り返し）を検出するため。
//     drainDeferredRetireQueues() 内で30秒ごとにスナップショット取得後リセット。
//     RT スレッドからの atomic fetchAdd のみ許可。
std::atomic<uint64_t> overflowWindowCounter_{0};

// ★★★ lastOverflowWindowResetTicks_ — 最終 window カウンタリセット時刻
//     drainDeferredRetireQueues() 内で管理（NonRT専用）。
uint64_t lastOverflowWindowResetTicks_{0};

// ★★★ lastOverflowWindowCount_ — 直前30秒間の overflow 回数（監査用）
std::atomic<uint64_t> lastOverflowWindowCount_{0};
```

**emitRetireIntent() 内で更新**:

```cpp
void RetireRuntime::emitRetireIntent(const RetireIntent& intent) noexcept
{
    uint64_t tail = convo::consumeAtomic(retireIntentTail_, ...);
    uint64_t nextTail = (tail + 1) % RETIRE_INTENT_QUEUE_SIZE;
    uint64_t head = convo::consumeAtomic(retireIntentHead_, ...);

    if (nextTail == head) {
        (void)convo::fetchAddAtomic(overflowCount_, uint64_t{1}, ...);
        (void)convo::fetchAddAtomic(droppedIntentCount_, uint64_t{1}, ...);
        return;
    }
    // ...
}
```

**drainDeferredRetireQueues() での利用（overflowStartTimestamp_ ベース）**:

```cpp
// ★ overflowScore 方式は採用しない（意味不明瞭・チューニング地獄の原因）。
//    overflowStartTimestamp_ による継続時間追跡に加え、
//    振動パターン（98%→100%→98%→100%）を検出するため
//    overflowWindowCounter で30秒窓の overflow 回数も監視する。
const uint64_t overflowStart = convo::consumeAtomic(
    retireRuntime_.overflowStartTimestamp_, std::memory_order_acquire);

// ★ 継続時間ベースの判定（連続 overflow 検出）
bool chronicByDuration = false;
if (overflowStart != 0) {
    const uint64_t now = getCurrentTicks();  // システム時刻
    const uint64_t overflowDurationMs = (now - overflowStart) / 1000;  // us→ms

    // ★ overflow 継続 > 5秒 を慢性的詰まりと判定
    //    一時的スパイク（<1秒）は通常 throttle (level 2) に委ねる
    chronicByDuration = (overflowDurationMs > 5000);
}

// ★★★ overflowWindowCounter の固定閾値(>100回/30秒)は
//    サンプリングレート(48/96/192kHz)やブロックサイズ(128/256/1024sample)
//    に依存して意味が変わるため、単純な回数ではなく overflowRate に変換する。
//    overflowRate = windowOverflows / 30sec で 1秒あたりの overflow 率を算出。
//    これによりサンプリングレートやブロックサイズに依存しない汎用的な判定が可能。
//    閾値は経験則: 3回/sec を超えると慢性的詰まりとみなす（4回/sec で100%到達相当）。
//    実運用の特性に合わせて調整可能。
const uint64_t windowOverflows = retireRuntime_.overflowWindowCounter();
constexpr uint64_t kWindowDurationSec = 30;
const double overflowRate = static_cast<double>(windowOverflows) / kWindowDurationSec;
// ★★★ 閾値は constexpr ではなく RuntimeConfig で調整可能にする
//     サンプリングレート(44.1/48/96/192kHz)やブロックサイズ(64/128/256/512/1024sample)
//     に依存して適切な値が変わるため、固定 constexpr は実運用で不適切。
//     デフォルト 3.0 回/sec をベースに、AudioEngine 初期化時に RuntimeConfig から
//     読み込む。constexpr の場合はビルド時固定になり環境適応ができない。
//     実装:
//       double overflowRateThreshold = engine_.getRuntimeConfig().overflowRateThreshold;
//       if (overflowRateThreshold <= 0.0) overflowRateThreshold = 3.0;  // fallback
const double overflowRateThreshold = engine_.getRuntimeConfig().overflowRateThreshold > 0.0
    ? engine_.getRuntimeConfig().overflowRateThreshold
    : 3.0;  // デフォルトフォールバック
const bool chronicByFrequency = (overflowRate > overflowRateThreshold);
    convo::publishAtomic(retireProtectiveModeActive_, true, ...);
}
```

**overflowStartTimestamp_ の更新（emitRetireIntent 内）**:

overflow 発生時は `emitRetireIntent()` 内で `overflowStartTimestamp_` と `overflowWindowCounter_` を更新する：

```cpp
// emitRetireIntent() 内 — キュー溢れ検出時の処理
if (nextTail == head) {
    (void)convo::fetchAddAtomic(overflowCount_, uint64_t{1}, ...);
    (void)convo::fetchAddAtomic(droppedIntentCount_, uint64_t{1}, ...);

    // ★ overflowStartTimestamp_ を初回のみ設定（2回目以降は上書きしない）
    uint64_t expected = 0;
    convo::compareExchangeAtomic(overflowStartTimestamp_, expected,
        getCurrentTicks(), std::memory_order_release);

    // ★★★ overflowWindowCounter_ をインクリメント
    //     このカウンタは drainDeferredRetireQueues() 内で定期的にリセットされる
    //     （30秒窓方式）。overflow のたびに +1 されるため、
    //     振動パターンでも蓄積される。
    (void)convo::fetchAddAtomic(overflowWindowCounter_, uint64_t{1},
        std::memory_order_release);
    return;
}
// ★ success（キュー空きあり）: overflow が継続中ならタイムスタンプをリセット
//    overflow が終了した場合のみ 0 に戻す
uint64_t prevStart = convo::exchangeAtomic(overflowStartTimestamp_, uint64_t{0},
    std::memory_order_release);
```

**drainDeferredRetireQueues() での window リセット**:

`overflowWindowCounter_` は drainDeferredRetireQueues() の先頭で定期的にスナップショットを取得し、リセットする（30秒窓方式）：

```cpp
// drainDeferredRetireQueues() 先頭 — 30秒窓の window カウンタ管理
const uint64_t nowTick = getCurrentTicks();
const uint64_t elapsedFromLastReset = (nowTick - lastOverflowWindowResetTicks_);

if (elapsedFromLastReset > 30'000'000) {  // 30秒経過でリセット
    // 直前30秒間の overflow 回数を取得
    const uint64_t windowOverflows = convo::exchangeAtomic(
        overflowWindowCounter_, uint64_t{0}, std::memory_order_acq_rel);
    convo::publishAtomic(lastOverflowWindowCount_, windowOverflows,
        std::memory_order_release);
    convo::publishAtomic(lastOverflowWindowResetTicks_, nowTick,
        std::memory_order_release);
}
```

これにより以下を区別できる：

| 状態 | droppedDelta | overflowDuration | overflowRate (回/sec) | 対応 |
| :--- | :--- | :--- | :--- | :--- |
| 一時的スパイク | 1〜2 | 短時間（<1秒） | 低（<0.3） | 通常 throttle (level 2) |
| 連続スパイク（振動） | 継続的 | 短い（successでリセット） | 高（>3） | 強制 level 3 (protective) |
| 慢性的詰まり | 継続的 | 長時間（>5秒） | 高（>3） | 強制 level 3 (protective) |

**補足**: duration + overflowRate の二重判定により、連続 overflow が成功で途切れる振動パターンでも chronic 判定が可能。overflowRate（回/秒）に変換することでサンプリングレートやブロックサイズに依存しない汎用的な判定が可能。閾値は RuntimeConfig で調整可能にすることで、環境適応性を確保する。

### 6.3 期待効果

- 新たなデータ型を追加せず、既存の `int` レベル体系を維持
- Overflow 検出と throttle の直接結合により、キュー溢れからの回復が確実に
- **`overflowStartTimestamp_` による継続時間追跡で一時的スパイクと慢性的詰まりを区別可能**
- **`overflowWindowCounter_` + overflowRate による30秒窓の回数集計で振動パターン（98%→100%→98%→100%）を検出可能**
- **overflowRate（回/秒）に変換することでサンプリングレート（48/96/192kHz）やブロックサイズ（128/256/1024sample）に依存しない汎用的な閾値を実現**
- 継続時間 + overflowRate の二重判定により、実運用の多様なパターンに対応
- RT スレッド制約を尊重（NonRT 側でのみ throttle 操作）
- overflowScore のような意味不明瞭なスコアリングを排除し、実運用で判断可能な継続時間ベースに統一

---

## 7. Phase C-1: Generation 64bit化

### 7.1 現状

- `DSPHandle::generation`: `uint32_t`
- `DSPRegistrySlot::generation`: `std::atomic<uint32_t>`
- `RetireIntent::generation`: `uint32_t`
- `onRuntimeRetiredNonRt()` で `world->generation`（`uint64_t`）を 32bit に truncate:

  ```cpp
  uint32_t generation = static_cast<uint32_t>(world->generation & 0xFFFFFFFFu);
  ```

### 7.2 改修内容

#### 7.2.1 型変更一覧

変更前に以下の全出現箇所を棚卸しする：

| # | ファイル | シンボル | 現状 | 変更後 |
| :--- | :--- | :--- | :--- | :--- |
| 1 | `ISRDSPHandle.h` | `DSPHandle::generation` | `uint32_t` | `uint64_t` |
| 2 | `ISRDSPHandle.h` | `DSPRegistrySlot::generation` | `std::atomic<uint32_t>` | `std::atomic<uint64_t>` |
| 3 | `ISRRetire.h` | `RetireIntent::generation` | `uint32_t` | `uint64_t` |
| 4 | `ISRRetireRuntimeEx.h` | `emitIntent()` の generation 引数 | `uint32_t` | `uint64_t` |
| 5 | `ISRDSPQuarantine.h` | `QuarantineEntry::generation`（追加予定） | — | `uint64_t` |
| 5b | `ISRRetire.h` | `acknowledgeGeneration_[]` | `std::atomic<uint32_t>` (array) | `std::atomic<uint64_t>` (array) |
| 6 | `ISRDSPHandle.h` | `CrossfadeRecord` | `DSPHandle fromHandle` / `DSPHandle toHandle` に generation 内包 | 必須棚卸し |
| 7 | `ISRDSPHandle.h` | `CrossfadeAuthorityRuntime` | `records_` が `CrossfadeRecord` の vector | 必須棚卸し（C-1 完了後に型変更漏れがないか確認） |
| 8 | `ISRDSPHandle.h` | `CrossfadeId` | generation を内包しないが、trace 出力で使用 | 棚卸し対象 |
| 9 | 全 trace 出力 | `emitOwnershipTrace()` / `emitRetireTrace()` / `emitShutdownTrace()` | generation を JSON 出力している箇所すべて | 棚卸し対象（64bit 書式への影響確認） |
| 10 | `AudioEngine.h` | `activeRuntimeDSPHandle_` | `DSPHandle` を保持（generation 内包） | **CRITICAL: 1箇所でも uint32_t が残ると generation mismatch → resolve失敗 → 偽Quarantine になる** |
| 11 | `AudioEngine.h` | `fadingRuntimeDSPHandle_` | `DSPHandle` を保持（generation 内包） | **CRITICAL: 同上。全ハンドルの generation 型が一致していることを確認** |
| 12 | `ISRDSPHandle.h` | `DSPHandle` 比較演算子 | `operator==` / `operator!=` の generation 比較 | **64bit 比較に自動追従するが、キャストが挟まっていないか確認** |

#### 7.2.2 truncate の削除

**対象ファイル**: `src/audioengine/AudioEngine.Commit.cpp`

```cpp
// 変更前:
uint32_t generation = static_cast<uint32_t>(world->generation & 0xFFFFFFFFu);
if (generation == 0u)
    generation = 1u;

// 変更後:
uint64_t generation = world->generation;
if (generation == 0u)
    generation = 1u;
```

#### 7.2.3 影響調査が必要な箇所（変更前に確認）

- **`DSPHandle` の比較演算**: `operator==` / `operator!=` は generation の一致を確認。64bit 化で問題なし
- **`RetireIntent` のソート**: `dequeuePendingRetireIntents()` 内で `generation` による `stable_sort` がある。64bit 化で問題なし
- **`emitIntent()` の generation==0 判定**: `ISRRetireRuntimeEx.cpp` の `emitIntent()` 内で `generation == 0u` を quarantine 判定に使用。64bit 化後も同じロジックで動作
- **シリアライゼーション**: JSON trace export で generation を出力している箇所。`uint64_t` への書式変更が必要になる場合がある（`"generation": 12345` は問題なし）
- **テストコード**: generation を `uint32_t` にキャストしている箇所がないか確認
- **`PublicationSequenceId`**: C-2 の stale discard で使用するため、型定義と全出現箇所を確認。`lastCommittedPublicationSequence_` が `uint64_t` 相当であれば問題なし
- **★★★ CRITICAL: `activeRuntimeDSPHandle_` / `fadingRuntimeDSPHandle_` / `CrossfadeRecord` / `CrossfadeAuthorityRuntime`**:
   1箇所でも `uint32_t` の生成残骸があると generation mismatch → resolve() が generation 不一致を検出 → 偽の Quarantined 状態になる。
   全ハンドル・全レコードの generation 型が `uint64_t` に統一されていることを、Select-String + 目視の二重監査で確認すること。
   `DSPHandleRuntime::create()` の `generation + 1` も uint64_t になることを確認。

#### 7.2.5 generation 全出現棚卸し（C-1 開始前の必須作業）

C-1 実装着手前に、`generation` / `Generation` の全出現箇所を網羅的に棚卸しする。以下のツールを併用：

```powershell
# Select-String で全出現箇所を列挙
Select-String -Pattern "generation|Generation" -Path src/**/*.h,src/**/*.cpp > generation_audit.txt

# semble でセマンティック検索
semble search "generation uint32_t atomic" src --content code

# ccc で補完
ccc search "generation"
```

**特に注意すべきパターン**:

- `uint32_t` で generation を保持している箇所（型変更対象）
- `unordered_map` / `map` のキーとして generation を使用している箇所（64bit化で hash 変更）
- `CrossfadeRecord`, `TransitionRecord`, `PublicationRecord` 等のレコード構造体内の generation
- generation を `uint32_t` にキャストしている箇所（truncation 検出）

#### 7.2.4 atomic lock-free 性の事前確認

**実装前に必ず確認**。`std::atomic<uint32_t>` → `std::atomic<uint64_t>` への変更により、lock-free 性がプラットフォーム依存で変わる可能性がある：

```cpp
// x64 (MSVC) では std::atomic<uint64_t> は常に lock-free
// 念のため static_assert で確認を推奨
static_assert(std::atomic<uint64_t>::is_always_lock_free,
    "atomic<uint64_t> must be lock-free on x64");
```

x64/MSVC では `uint64_t` の atomic 操作は常に lock-free である。しかし、ARM/ARM64 クロスコンパイルや将来のプラットフォーム変更を考慮し、該当 atomic 変数すべてに `static_assert` または実行時チェックを追加することを推奨する。

**確認対象**: `DSPRegistrySlot::generation`, `DSPHandle` 比較時の generation 読み取り, `RetireIntent` キュー内の generation アクセス

#### 7.2.5 Generation wrapper 型の推奨

**問題**: 生の `uint64_t` を generation として使用すると、以下のリスクがある：

1. 型混在事故（`uint32_t` 残存箇所との意図しない変換）
2. `map` / `unordered_map` のキーとして使用された場合の暗黙変換
3. log 圧縮時の narrowing 検出不能

**推奨**: `Generation` wrapper 型を導入し、生の整数型での generation 受け渡しを禁止する：

```cpp
// ISRRuntimeSemanticSchema.h — 追加
struct Generation {
    uint64_t value{0};

    bool operator==(const Generation& other) const noexcept { return value == other.value; }
    bool operator!=(const Generation& other) const noexcept { return value != other.value; }
    bool isNull() const noexcept { return value == 0; }

    // DSPHandle の generation 比較で使用するためのキャスト（明示的）
    explicit operator uint64_t() const noexcept { return value; }
};

// std::hash 特殊化（unordered_map キーとして使用可能）
template<>
struct std::hash<Generation> {
    std::size_t operator()(const Generation& g) const noexcept {
        return std::hash<uint64_t>{}(g.value);
    }
};
```

**変更対象**: `DSPHandle::generation`, `DSPRegistrySlot::generation`, `RetireIntent::generation`, `RetireRuntimeEx::emitIntent()` の generation 引数, `DSPQuarantineManager::quarantineHandle()` の generation 引数, `acknowledgeGeneration_` の全出現箇所を `uint32_t` / `uint64_t` から `Generation` に置き換える。

**注意**: Primitive `uint64_t` は `uint32_t` との暗黙変換を許容するため、コードレビューで見落としやすい。`Generation` wrapper 型にすることでコンパイル時に型不一致を検出できる。これは Phase C-1 の完了条件に含める。

### 7.3 期待効果

- 事実上 generation wrap を無視可能（`uint64_t` の wrap には約5.8億年）
- `world->generation`（uint64_t）の truncate 削除により情報損失ゼロ
- **Generation wrapper 型により型混在事故をコンパイル時に検出可能**

---

## 8. Phase C-2: Deferred Publish改善

### 8.1 現状

```cpp
// RuntimePublicationOrchestrator.h
std::optional<PublicationAdmission::PublishRequest> deferredRequest_;
bool hasDeferred_ = false;

void enqueueDeferred(const PublicationAdmission::PublishRequest& req) noexcept
{
    deferredRequest_ = req;  // ← 上書き。最新1件のみ保持
    hasDeferred_ = true;
}
```

### 8.2 改修方針

**FIFOリングバッファは採用しない。** ConvoPeq の ISR Runtime は「Latest World Wins」思想であり、古い publish 要求を順番に消化する FIFO モデルは相性が悪い。不要な Crossfade/Retire/Publish を増やすだけである。

代わりに **sequence 番号による stale discard** を導入する。

### 8.3 改修内容

#### 8.3.1 DeferredPublishSlot の導入

**対象ファイル**: `src/audioengine/RuntimePublicationOrchestrator.h`

```cpp
struct DeferredGuard {
    uint64_t generation;  // enqueue 時点の rebuildRequestGeneration
    PublicationSequenceId sequence;  // enqueue 時点の lastCommittedPublicationSequence
};

// ★★★ DiscardReason: deferred publish が破棄された理由を分類する
//     最低限の分類を持つことで、運用時の原因分析が容易になる。
enum class DiscardReason : uint8_t {
    None,              // 未破棄
    ShutdownDiscard,   // shutdown による強制消去
    StaleDiscard,      // stale 判定（generation/sequence 不一致）
    SupersededDiscard  // より新しい publish で上書き
};

struct DeferredPublishSlot {
    PublicationAdmission::PublishRequest request;
    // ★★★ DeferredGuard で generation と sequence を統一的に管理する。
    //     片方だけの比較（generation のみ / sequence のみ）は安全性が不十分。
    //     generation と sequence は別系統のため分岐可能。両方を同時にチェックすることで
    //     stale discard の信頼性が向上する。
    DeferredGuard guard;
    DiscardReason lastDiscardReason{DiscardReason::None};  // ★ 最終破棄理由
};

// std::optional<PublishRequest> → DeferredPublishSlot へ置換
// （依然1件保持。リング化はしない）
std::optional<DeferredPublishSlot> deferredSlot_;
```

#### 8.3.2 enqueueDeferred の改善

**対象ファイル**: `src/audioengine/RuntimePublicationOrchestrator.cpp`

```cpp
void RuntimePublicationOrchestrator::enqueueDeferred(
    const PublicationAdmission::PublishRequest& req) noexcept
{
    deferredSlot_ = DeferredPublishSlot{
        req,
        DeferredGuard{
            .generation = req.generation,  // enqueue 時点の rebuildRequestGeneration
            .sequence = engine_.getLastCommittedPublicationSequence()  // グローバル sequence
        }
    };
    hasDeferred_ = true;
}
```

**重要**: `publicationSequenceAtEnqueue` は `RuntimePublicationOrchestrator` ローカルの単調カウンタではなく、**`AudioEngine::lastCommittedPublicationSequence_` のスナップショット** である。これにより異なる sequence space の比較（ローカル vs グローバル）という論理的不整合を解消する。

#### 8.3.2.5 Shutdown 時の Deferred Publish 強制消去

shutdown 開始時に `deferredSlot_` が残っていると、`notifyTransitionComplete` の shutdown guard で return されるため再投入は防止される。しかし `deferredSlot_` のメモリが解放されず、DrainAudit の `deferredPublish` が非ゼロになり続ける。

**対策**: ShutdownRuntime の advance 系パス（`AudioStopped` 以降）で以下の強制消去を実行する：

```cpp
// RuntimePublicationOrchestrator.h — 追加メソッド
// shutdown 時に deferred publish を強制消去する。
// stale discard より強力で、shutdown authority が deferred publish を
// 完全にクリアすることを保証する。
void clearDeferredForShutdown() noexcept
{
    if (hasDeferred_) {
        deferredSlot_.reset();
        hasDeferred_ = false;
    }
}
```

**呼び出し箇所**: `ShutdownRuntime::advancePhase()` または `AudioEngine` の shutdown 経路で `ShutdownPhase::AudioStopped` 遷移後に `runtimeOrchestrator_->clearDeferredForShutdown()` を呼ぶ。これにより `deferredPublish=0` が保証される。

**注意**: 通常運用時（非 shutdown）の deferred publish は従来通り `notifyTransitionComplete` 内で処理される。`clearDeferredForShutdown()` は shutdown 専用パスであり、通常運用では呼ばない。

#### 8.3.3 notifyTransitionComplete での stale discard

**対象ファイル**: `src/audioengine/RuntimePublicationOrchestrator.cpp`

```cpp
void RuntimePublicationOrchestrator::notifyTransitionComplete(
    AudioEngine::DSPCore* currentAfterFade) noexcept
{
    // ... existing code ...

    // shutdown 中は deferred 再投入をキャンセル
    if (engine_.isShutdownInProgress()) {
        if (hasDeferred_) {
            deferredSlot_.reset();
            hasDeferred_ = false;
        }
        return;
    }

    // クロスフェード完了後、deferred publish request を再試行する
    if (hasDeferred_ && deferredSlot_.has_value())
    {
        const auto& deferred = *deferredSlot_;

        // stale discard（二重検査）:
        // 1. generation 検査: enqueue 時点の generation が現在と異なれば stale
        //    ★ C-1 (Generation 64bit化) との整合性のため uint64_t
        const uint64_t currentGen = convo::consumeAtomic(
            engine_.rebuildRequestGeneration, std::memory_order_acquire);
        if (deferred.requestGeneration != 0 && deferred.requestGeneration != currentGen) {
            deferredSlot_.reset();
            hasDeferred_ = false;
            return;
        }

        // 2. publication sequence 検査:
        //    enqueue 時点の publicationSequence が現在より古ければ
        //    より新しい publish が成功しているため破棄
        const auto currentPubSeq = engine_.getLastCommittedPublicationSequence();
        if (deferred.publicationSequenceAtEnqueue < currentPubSeq) {
            deferredSlot_.reset();
            hasDeferred_ = false;
            return;
        }

        // 有効な deferred → submit
        // ★ submitPublishRequest() 内部で PublicationAdmission::evaluate()
        //   が再実行されるため、更なる安全層として機能する。
        auto req = deferred.request;
        deferredSlot_.reset();
        hasDeferred_ = false;
        submitPublishRequest(req);
    }
}
```

#### 8.3.4 consumeDeferredRequest の更新

```cpp
std::optional<PublicationAdmission::PublishRequest>
RuntimePublicationOrchestrator::consumeDeferredRequest() noexcept
{
    if (!hasDeferred_ || !deferredSlot_.has_value())
        return std::nullopt;

    auto req = deferredSlot_->request;
    deferredSlot_.reset();
    hasDeferred_ = false;
    return req;
}
```

### 8.4 publicationSequence の提供

`getLastCommittedPublicationSequence()` は既に `AudioEngine` に存在する：

```cpp
// AudioEngine.h（既存）
[[nodiscard]] convo::isr::PublicationSequenceId
getLastCommittedPublicationSequence() const noexcept
{
    return consumeAtomic(lastCommittedPublicationSequence_, std::memory_order_acquire);
}
```

`DeferredPublishSlot::publicationSequenceAtEnqueue` はこの値のスナップショットであり、同じ sequence space で比較される。

### 8.5 期待効果

- 常に最新1件のみ保持（現状維持、複雑化しない）
- **`publicationSequenceAtEnqueue` はグローバル sequence のスナップショット**。ローカルカウンタとの混同を解消
- stale discard により、deferred の enqueue 後に新しい publish が成功していた場合に無駄な再投入を防止
- stale discard を通り抜けても **`submitPublishRequest()` → `PublicationAdmission::evaluate()` の再実行** により、generation staleness や shutdown 状態は防護される
- 「Latest World Wins」思想との整合性を維持
- FIFO リングバッファのような副作用（不要な Crossfade/Retire/Publish の増加）がない

---

## 9. テスト計画

### 9.1 既存テストの確認

不足しているテスト：

- ❌ `DSPQuarantineManager` の単体テスト
- ❌ `ShutdownRuntime` のFSM遷移テスト
- ❌ `DSPHandleRuntime::quarantine()` 呼び出しテスト
- ❌ Shutdown 中の deferred 再投入防止テスト
- ❌ Overflow 時の圧力ポリシーテスト

### 9.2 追加すべきテスト

| テストケース | 種別 | 対象Phase |
| :--- | :--- | :--- |
| `QuarantineEntryTests` — 隔離/解放/世代照合 | Unit | A-1 |
| `QuarantineManagerIntegrationTests` — DSPHandleRuntimeとの統合 | Integration | A-1 |
| `ShutdownFSMTransitionTests` — 全phase遷移の正当性 | Unit | A-2 |
| `ShutdownPublishRejectTests` — shutdown中のpublish拒否 | Integration | A-2 |
| `ShutdownDrainAuditTests` — 完全閉包監査（quarantine除外確認含む） | Integration | A-2 |
| `DeferredPublishStaleDiscardTests` — stale discard による再投入防止 | Integration | C-2 |
| `DeferredPublishShutdownGuardTests` — shutdown中のdeferredキャンセル | Integration | A-2/C-2 |
| `OverflowPressurePolicyTests` — 溢れ→throttle→復帰（継続時間ベース） | Integration | C-1 |
| `OverflowOscillationDetectionTests` — 振動パターン（98%→100%繰り返し）検出 | Integration | C-1 |
| `Generation64bitConversionTests` — 型変更後の動作確認 | Unit | B-1 |
| `RetireTraceEmissionTests` — emitRetireTrace の出力検証 | Unit | B-2 |

---

## 10. 実装順序と依存関係

```
Phase A-1 (Quarantine実体化)
  ├── A-1.1: QuarantineEntry 構造体追加
  ├── A-1.2: DSPQuarantineManager 拡張（フラグ→構造体）
  ├── A-1.3: DSPHandleRuntime::quarantine() 呼び出し経路実装
  │          （AudioEngine.Commit.cpp の retireRuntimeEx_.quarantine() 箇所に追加）
  ├── A-1.4: DSPQuarantineManager 呼び出し経路実装
  │          （同上箇所に dspQuarantineManager_.quarantineHandle() 追加）
  └── A-1.5: Test: QuarantineEntryTests + QuarantineManagerIntegrationTests

Phase A-2 (Shutdown完全閉包)
  ├── [pre] switch(ShutdownPhase) 全出現箇所の網羅性監査（調査のみ。enum変更なし）
  │         （ShutdownPhase 拡張は Phase D へ延期）
  ├── A-2.1: AudioEngine::isShutdownInProgress() を OR 判定に変更（永久維持）
  ├── A-2.2: notifyTransitionComplete shutdown ガード追加
  ├── A-2.3: RuntimeDrainAudit 構造体追加（oldestPendingAgeMs 含む。AudioEngine::oldestPendingAge_ 参照）
  ├── A-2.4: activeCrossfade 実測 — 既存 CrossfadeRuntime::isPending() を利用（新規API追加不要）
  ├── A-2.5: RuntimePublicationCoordinator に getter 群追加
  │          （publicationBacklogCount() / pendingIntentCount() / retireBacklogCount() 等）
  ├── A-2.6: isFullyDrained() 強化 + collectDrainAudit()
  │          （pendingPublication は実数 getter で取得。TODO 禁止。完了条件に含める）
  ├── A-2.7: ReleaseResources の DrainAudit 統合
  │          （EpochSettled 到達後に collectDrainAudit() を呼ぶ。isAllZero() 時のみ emitRetireTrace 出力）
  └── A-2.8: Test: ShutdownFSMTransitionTests + ShutdownPublishRejectTests
             + ShutdownDrainAuditTests

Phase B-1 (Generation 64bit化)
  ├── B-1.1: generation 全出現箇所の事前棚卸し
  │          （DSPHandle/RetireIntent/DSPRegistrySlot/CrossfadeRecord/
  │           PublicationSequenceId/serialization/trace export/テストコード）
  │          ※ Select-String/semble/ccc を併用。unordered_map のキーも確認
  ├── B-1.2: atomic<uint64_t> lock-free static_assert 確認
  ├── B-1.3: DSPHandle::generation uint32_t→uint64_t
  ├── B-1.4: RetireIntent::generation uint32_t→uint64_t
  ├── B-1.5: DSPRegistrySlot::generation uint32_t→uint64_t
  ├── B-1.6: onRuntimeRetiredNonRt() の truncate 削除
  ├── B-1.7: emitIntent() の引数型変更
  └── B-1.8: Test: Generation64bitConversionTests

Phase B-2 (Reclaim完結保証)
  ├── B-2.1: RetireRuntimeEx::emitRetireTrace() 追加（条件付き出力。正常時は出力しない）
  ├── B-2.2: DrainAudit に既存 oldestPendingAge_ を統合（新規API不要）
  └── B-2.3: Test: RetireTraceEmissionTests

Phase C-1 (Overflow自己防衛)
  ├── C-1.1: lastOverflowTicks_ + overflowStartTimestamp_ 追加（RetireRuntime）
  │          （overflow 継続時間追跡。overflowScore_ は採用せず）
  │           overflowDurationMs = now - overflowStartTimestamp_ で算出）
  ├── C-1.2: overflowWindowCounter_ + lastOverflowWindowResetTicks_ + lastOverflowWindowCount_ 追加
  │          （30秒窓の overflow 回数集計。振動パターン検出用）
  │           emitRetireIntent 内で fetchAddAtomic、drainDeferredRetireQueues 内で30秒ごとにリセット）
  ├── C-1.3: drainDeferredRetireQueues() に overflow→throttle結合（duration + windowCounter の二重判定）
  └── C-1.4: Test: OverflowPressurePolicyTests（振動パターンテストを含む）

Phase C-2 (Deferred Publish改善)
  ├── C-2.1: DeferredPublishSlot 構造体追加（publicationSequenceAtEnqueue）
  ├── C-2.2: enqueueDeferred で global sequence スナップショットを記録
  ├── C-2.3: notifyTransitionComplete の stale discard 実装（sequence space 比較）
  └── C-2.4: Test: DeferredPublishStaleDiscardTests
             + DeferredPublishShutdownGuardTests
```

### 依存関係グラフ

```texttext
A-1 ──→ A-2 ──→ B-1 ──→ B-2 ──→ C-1 ──→ C-2
                      ↑                    ↑
                      └── (並行可能) ────────┘
```

- A-1 と A-2 は A-2 の DrainAudit が A-1 の quarantine 機能を参照するため逐次
- B-1 と B-2 は並行実装可能（独立したコンポーネント）
- C-1 と C-2 は C-2 が C-1 で追加予定の `PublicationSequenceId` を使用するため逐次推奨
- 全 Phase を通じて、各ステップ完了後に `Build_CMakeTools` + `Strict Atomic Dot-Call Scan` + `RunCtest_CMakeTools` で検証

---

## 付録A: 検証ツール活用ガイド

各フェーズの実装前/実装後に使用すべき検証ツール：

| 目的 | ツール | 使用方法 |
| :--- | :--- | :--- |
| 静的解析 | `codegraph query` | モジュール構造の変化を確認 |
| 参照解析 | `semble <path> --ref <symbol> --content code` | 呼び出し元/被呼び出しを確認 |
| セマンティック検索 | `ccc search <query>` | 影響範囲の横断調査 |
| 知識グラフ | `graphify query "..." --backend deepseek` | アーキテクチャ変更前後の構造比較 |
| リンター | `Build_CMakeTools` | ビルドエラー確認 |
| 規約遵守 | `Strict Atomic Dot-Call Scan` | Atomic dot-call 規約違反検出 |
| 回帰テスト | `RunCtest_CMakeTools` | 既存テストの回帰確認 |

---

## 付録B: コード検証による未確定事項の確定結果

Phase A-2 / B-1 / B-2 / C-2 の設計に関して、現行ソースコードの詳細検証により以下の未確定事項を確定した。

### B-1. activeCrossfade の実測 API

**結論**: **新規 API 追加は不要。既存 `CrossfadeRuntime::isPending()` を DrainAudit で利用する。**

| コンポーネント | 既存API | 用途 | 採用 |
| :--- | :--- | :--- | :--- |
| `CrossfadeRuntime` | `isPending()` | クロスフェード実行中か否かの二値判定 | **DrainAudit で採用** |
| `CrossfadeAuthorityRuntime` | `getActiveCrossfades()` | 全 active crossfade 一覧取得 | 必要に応じて利用（vector 確保あり） |
| `AudioEngine` | `activeCrossfadeId_` | CrossfadeId の有無 | ReleaseResources.cpp で既に使用中 |

**判断理由**: DrainAudit の目的は「shutdown 完了判定」であり、正確な件数管理ではなく crossfade 残存の有無が分かれば十分。新たな API を追加すると既存 CrossfadeAuthority との二重管理リスクが生じる。`DSPTransition::activeTransitionCount()` は追加しない。

### B-2. pendingPublication の実測値

**結論**: `RuntimePublicationCoordinator::isFullyDrained()` が以下の全 backlog をチェック済み：

| カウンタ | 型 | 役割 |
| :--- | :--- | :--- |
| `swapPending_` | `atomic<bool>` | commit 途中 |
| `retireBacklogCount_` | atomic<uint64_t> | retire backlog |
| `publicationBacklogCount_` | atomic<uint64_t> | publication backlog |
| `pendingIntentCount_` | atomic<uint64_t> | pending intent |
| `fallbackBacklogCount_` | atomic<uint64_t> | fallback backlog |
| `reclaimInFlightCount_` | atomic<uint64_t> | reclaim 実行中 |
| `deferredRetireResidencyCount_` | atomic<uint64_t> | deferred retire |

**推奨**: `AudioEngine::collectDrainAudit()` では `runtimePublicationBridge_.isFullyDrained()` をそのまま利用する。個別カウンタ値が必要な場合は `runtimePublicationBridge_.getReclaimInFlightCount()` 等を個別取得。

### B-3. oldestPendingAgeMs() の実装

**結論**: `AudioEngine::oldestPendingAge_` (`std::atomic<double>`) が **既に存在する**。`AudioEngine.Commit.cpp` の `onRuntimeRetiredNonRt()` 内で更新されている。新規に `RetireRuntimeEx` へ追加する必要はない。

**推奨**: DrainAudit では `convo::consumeAtomic(engine_.oldestPendingAge_, ...)` を読み取る。`oldestPendingAgeMs()` の新規追加は不要。ただし `oldestPendingAge_` は `onRuntimeRetiredNonRt()` でのみ更新される点に注意（deferred reclaim での滞留は別途監視が必要な場合あり）。

### B-4. prevDroppedSnapshot_ の有無

**結論**: **存在しない**。新規追加が必要。追加先候補：

| 候補 | 利点 | 欠点 |
| :--- | :--- | :--- |
| `AudioEngine` のメンバ | DrainAudit に近い | 既存クラスの肥大化 |
| `RetireRuntime` のメンバ | overflow 管理に自然 | 参照経路が増える |
| `AudioEngine.Retire.cpp` の static 変数 | 最小変更 | テスタビリティ低下 |

**推奨**: `AudioEngine` に `prevDroppedSnapshot_` を追加し、`drainDeferredRetireQueues()` 内で管理する。

### B-5. overflowScore の有無（再検討結果は付録F-4参照）

**結論**: **存在しない**。当初計画では `overflowScore_`（overflow→+4, success→-1, 上限16）を追加予定だったが、**採用しない**。代わりに `overflowStartTimestamp_` のみ追加する。理由：

- スコアの物理的意味が不明瞭（継続時間なのか回数なのか）
- RT/NonRT 混在環境での atomic 整合性コスト
- 実運用では「3回の overflow 回数」より「30秒間の overflow 継続時間」の方が危険シグナルとして価値がある
- `lastOverflowTicks_`：最終 overflow 発生時刻（監査用）
- `overflowStartTimestamp_`：overflow 継続時間追跡用（0=非overflow状態）

### C-1. ShutdownPhase switch 網羅性

**結論**: `ISRShutdown.cpp` に2つの switch 文が存在：

| switch 文 | default: の有無 | 新列挙子追加時のリスク |
| :--- | :--- | :--- |
| `advancePhase()` | **あり**（`ShutdownComplete` 後） | 安全。ただし新列挙子の case 追加を忘れるとコンパイル警告なしで無視される |
| `emitShutdownTrace()` | **なし** | **危険。未ハンドルの新列挙子があると無視され phaseName が "Running" 固定になる** |
| `transitionTo()` | switch不使用 (int比較) | 安全 |
| `isShutdownInProgress()` | switch不使用 | 安全 |

**推奨**: Phase A-2 の拡張前に、`emitShutdownTrace()` に `default:` ケースを追加する。または全列挙子を明示してコンパイラの C4062 警告が有効になるようにする。

### C-2. DSPHandleRuntime::quarantine() 呼び出し元

**結論**: **定義のみで呼び出し元ゼロ**（デッドコード）。Phase A-1 で `dspHandleRuntime_.quarantine(dspHandle)` の呼び出し経路を新規実装する必要がある。

### C-3. isShutdownInProgress 判定の最終方針

**結論**: OR 判定を永久維持する（完全委譲は行わない）。

`AudioEngine::isShutdownInProgress()` は以下を OR で判定する：

```cpp
return lifecycleShutdown || shutdownRuntime_.isShutdownInProgress();
```

ここで `lifecycleShutdown` は `lifecycleState == Releasing || lifecycleState == Destroyed` である。

**設計判断**:

- `EngineLifecycleState` と `ShutdownPhase` は異なるライフサイクル視点であり、移行期間に乖離が発生し得る
- OR 判定により「どちらかが shutdown 状態なら shutdown とみなす」安全側の保護を常に維持
- 数学的な一本化（ShutdownRuntime のみへの完全委譲）は破綻耐性の観点から行わない
- この方針は Phase A-2 以降も恒久的に維持する

### C-4. `isFullyDrained()` の現状

**結論**: 以下の2重チェックが既に存在する：

1. `AudioEngine::isFullyDrained()` → `runtimePublicationBridge_.isFullyDrained()` + `!hasDeferredCommit`
2. `RuntimePublicationCoordinator::isFullyDrained()` → 7個の atomic カウンタチェック

**推奨**: Phase A-2 ではこれらを `collectDrainAudit()` でラップし、`RuntimeDrainAudit` 構造体に各値をマッピングする。既存ロジックを書き換えるのではなく上書きする形とする。

---

## 付録C: 実装検証による追加確定事項（2026-06-08）

v5 策定後、現行ソースコードの追加検証（grep/ccc/semble/CodeGraph/Graphify/ファイル精読）により以下の事項を確定した。

### C-1. DSPHandleRuntime::destroyQuarantineSlot() の実装確認

**結論**: `forceReclaim()` は危険のため廃止し、代わりに shutdown 専用の `destroyQuarantineSlot()` を新規実装する。`DSPRegistrySlot` 構造体は以下を持ち、`destroyQuarantineSlot()` は generation を参照せずに強制解放できる：

```cpp
struct DSPRegistrySlot {
    std::atomic<uint32_t> generation;  // ★ C-1 で uint64_t 化予定
    void*                 instance;    // destroyQuarantineSlot では nullptr 設定
    std::atomic<DSPState> state;       // destroyQuarantineSlot では Reclaimed 設定
};
```

実装イメージ：

```cpp
void DSPHandleRuntime::destroyQuarantineSlot(uint32_t slot) noexcept {
    if (slot >= MAX_DSP_SLOTS) return;
    const auto prevState = convo::consumeAtomic(registry_[slot].state,
                                                std::memory_order_acquire);
    assert(prevState == DSPState::Quarantined);
    if (prevState != DSPState::Quarantined) return;
    registry_[slot].instance = nullptr;
    convo::publishAtomic(registry_[slot].state, DSPState::Reclaimed,
                         std::memory_order_release);
}
```

`forceReclaim()` と異なる点：

1. 関数名を `destroyQuarantineSlot` に変更（forceReclaim の危険性を連想させない）
2. state==Quarantined の assertion を追加（隔離状態でない slot を誤って解放しない）
3. ShutdownPhase::ReclaimComplete 以降でのみ呼び出し可能（通常運用では使わない）
4. `destroyForShutdown()` が DSPQuarantineManager 側の対応関数

### C-2. RuntimePublicationCoordinator の不足 getter 一覧

**結論**: 以下の getter が不足している。Phase A-2.6 で追加する：

| 不足 getter | 対応する private member | 型 |
| :--- | :--- | :--- |
| `getPublicationBacklogCount()` | `publicationBacklogCount_` | `std::atomic<uint64_t>` |
| `getPendingIntentCount()` | `pendingIntentCount_` | `std::atomic<uint64_t>` |
| `getRetireBacklogCount()` | `retireBacklogCount_` | `std::atomic<uint64_t>` |
| `getFallbackBacklogCount()` | `fallbackBacklogCount_` | `std::atomic<uint64_t>` |
| `getDeferredRetireResidencyCount()` | `deferredRetireResidencyCount_` | `std::atomic<uint64_t>` |

既存: `getReclaimInFlightCount()`（唯一の getter）、`isSwapPending()`（bool getter）

これらの getter は `convo::consumeAtomic(member_, std::memory_order_acquire)` を返すのみで、setter と対称的な実装となる。

### C-3. isShutdownInProgress() 呼び出し元の完全分類

**結論**: 以下の3系統が存在。Phase A-2 の段階的移行で対応：

| 系統 | 呼び出しパス | 箇所数 | 備考 |
| :--- | :--- | :--- | :--- |
| A | `AudioEngine::isShutdownInProgress()` | **15箇所以上** | Commit/Processing/RebuildDispatch/Retire/Snapshot/Timer |
| B | `engine_->isShutdownInProgress()` | 1箇所 | AudioEngine.h:2666（内部ヘルパー） |
| C | `shutdownRuntime_.isShutdownInProgress()` | 1箇所 | Commit.cpp:402（onRuntimeRetiredNonRt内） |

段階的移行の完了条件：

1. `AudioEngine::isShutdownInProgress()` を `lifecycleShutdown || shutdownRuntime_.isShutdownInProgress()` に変更
2. 系統Aの呼び出し元は自動的に新しい実装を参照
3. 系統Cは既に `ShutdownRuntime` を直接参照。問題なし
4. 全15箇所の呼び出し結果が期待通りか CI/テストで確認後、最終的に `lifecycleShutdown` 部分を削除して委譲完了

### C-5. PublicationSequenceId の型定義

**結論**: `ISRRuntimeSemanticSchema.h:195` で定義：

```cpp
using PublicationSequenceId = std::uint64_t;
```

C-2 での使用に問題なし。`DeferredPublishSlot::publicationSequenceAtEnqueue` はこの型を使用する。

### C-6. atomic<uint64_t> の lock-free 性確認

**結論**: `std::atomic<uint32_t>` → `std::atomic<uint64_t>` 変更の lock-free 性は、x64 (MSVC) では常に保証される。以下の static_assert を該当 atomic 変数定義箇所に追加することを推奨：

```cpp
// DSPRegistrySlot の generation 変更箇所
static_assert(std::atomic<uint64_t>::is_always_lock_free,
    "atomic<uint64_t> must be lock-free on x64 platform");
```

**確認対象 atomic 変数一覧**:

| 変数 | ファイル | 現状 | 変更後 |
| :--- | :--- | :--- | :--- |
| `DSPRegistrySlot::generation` | ISRDSPHandle.h | `std::atomic<uint32_t>` | `std::atomic<uint64_t>` |
| `RetireIntent::generation` | ISRRetire.h | `uint32_t` (非atomic) | `uint64_t` (非atomic) |
| `DSPHandle::generation` | ISRDSPHandle.h | `uint32_t` (非atomic) | `uint64_t` (非atomic) |

非atomicの `uint32_t` → `uint64_t` 変更は lock-free 問題とは無関係。`std::atomic` ラッパーの型変更のみが lock-free 確認対象。

### C-7. CrossfadeRecord の generation 影響

**結論**: `CrossfadeRecord` は以下のメンバを持ち、`DSPHandle fromHandle` / `DSPHandle toHandle` を内包する：

```cpp
struct CrossfadeRecord {
    CrossfadeId id;
    DSPHandle   fromHandle;   // ← generation を含む
    DSPHandle   toHandle;     // ← generation を含む
    uint64_t    startEpoch;
    bool        active;
};
```

C-1 (Generation 64bit化) では `DSPHandle::generation` の型変更に伴い、`CrossfadeRecord` 内の `fromHandle`/`toHandle` も自動的に 64bit 化される。個別の対応は不要（`DSPHandle` の型変更に追従する）。

`CrossfadeAuthorityRuntime::records_` (`std::vector<CrossfadeRecord>`) も同様。**明示的な棚卸し対象とするが、個別コード変更は不要**（型変更のみで対応可能）。

---

## 付録D: 追加検証による未確定事項の確定（2026-06-08 最終版）

v7 策定後、全ツール（grep/ccc/semble/CodeGraph/Graphify/ファイル精読）を使用した追加検証により以下の事項を確定した。

### D-1. activeCrossfade 実測 API の最終判断

**結論**: **新規追加は不要。既存 `CrossfadeRuntime::isPending()` を利用する。**

`DSPTransition` クラスに `activeTransitionCount()` / `hasActiveTransition()` は存在しない。当初計画では Phase B-1 で追加予定だったが、以下の理由で **新規追加を行わない**：

1. `CrossfadeRuntime::isPending()` が既に `AudioEngine.h` 内で使用されている（L2156, L2268）
2. `CrossfadeAuthorityRuntime::getActiveCrossfades()` も存在し、必要なら詳細取得可能
3. 新たな API を追加すると既存 CrossfadeAuthority との二重管理リスクが生じる
4. DrainAudit の `activeCrossfade` は完了条件に含めない設計（4.2.4 定義済み）であり、二値情報（有無）で十分

**実装**: DrainAudit では `crossfadeRuntime_.isPending() ? 1u : 0u` を使用する。

### D-2. DebugRuntime の非同期 I/O キュー

**結論**: **存在しない**。`DebugRuntime` クラス（`ISRDebugRuntime.h`）には `emitCIArtifacts()`, `emitHBTrace()`, `recordHBEdge()` のみ。`enqueueJsonOutput()` や非同期書き込みキューは未実装。

**影響**: 計画書の「emitRetireTrace は DebugRuntime 非同期キュー経由」案は、DebugRuntime に `enqueueJsonOutput()` を追加する前提となる。代替案として以下を推奨：

| 方式 | メリット | デメリット |
| :--- | :--- | :--- |
| DebugRuntime に enqueueJsonOutput 追加 | 設計として一貫性あり | 新規API追加が必要 |
| std::ofstream 直接出力（位相制限付き） | 実装が単純 | 定期監査時のブロックリスク |
| 別スレッドのファイル書き込みキュー | 完全非同期 | 設計が大掛かり |

**推奨**: Phase B-1 では `std::ofstream` 直接出力＋位相制限（`EpochSettled` 以降のみ）を維持。`DebugRuntime::enqueueJsonOutput()` の追加は将来対応とする。理由：shutdown 時は既に他スレッドが停止しているためブロックリスクは低い。

### D-3. Coordinator 不足 getter の確定

**結論**: 以下の5個の getter が不足していることを確認：

| getter | 対応 member | 現状 |
| :--- | :--- | :--- |
| `getPublicationBacklogCount()` | `publicationBacklogCount_` | ❌ setter のみ |
| `getPendingIntentCount()` | `pendingIntentCount_` | ❌ setter のみ |
| `getRetireBacklogCount()` | `retireBacklogCount_` | ❌ setter のみ |
| `getFallbackBacklogCount()` | `fallbackBacklogCount_` | ❌ setter のみ |
| `getDeferredRetireResidencyCount()` | `deferredRetireResidencyCount_` | ❌ setter のみ |

既存の getter: `isSwapPending()`, `getReclaimInFlightCount()`。Phase A-2.6 で追加が必要。

### D-4. laneName / lifecycleStateName ヘルパーの有無

**結論**: **存在しない**。`ISRRetireRuntimeEx.cpp` 内に `laneName()` および `lifecycleStateName()` 関数の定義はない。一方、`epochModeName()` や `lifecycleFromRaw()` 等の類似ヘルパーは存在する。

**影響**: 計画書の `emitRetireTrace()` で `laneName(lane)` および `lifecycleStateName(lifecycle)` を参照している。Phase B-1.1 でこれらのヘルパー関数を追加する必要がある。`epochModeName()` と同様のパターンで実装可能。

### D-5. generation truncation の残存確認

**結論**: `AudioEngine.Commit.cpp:410` で以下が確認された（計画書の指摘通り）：

```cpp
std::uint32_t generation = static_cast<std::uint32_t>(world->generation & 0xFFFFFFFFu);
```

この truncation は Phase B-1.6 で削除対象。`world->generation` は `uint64_t` であり、32bit への切り詰めにより generation wrap が理論上発生し得る。

### D-6. DSPQuarantineManager の現在の generation 型

**結論**: `ISRDSPQuarantine.h` の `quarantineHandle()` は現在 `std::uint32_t generation` を受け取っている。**Phase A-1 で既に `uint64_t` に変更する（計画書反映済み）**。これにより C-1 での再修正を防止する。

### D-7. Trace 出力関数の棚卸し

結論: 以下の trace 出力関数が generation を JSON 出力している。全件が Generation64 棚卸し対象：

| 関数 | ファイル | 出力方式 |
| :--- | :--- | :--- |
| `emitOwnershipTrace()` | `ISRDSPHandle.cpp` | `std::ofstream` |
| `emitRetireTimeline()` | `ISRRetireRuntimeEx.cpp` | `std::ofstream` |
| `emitShutdownTrace()` | `ISRShutdown.cpp` | `std::ofstream` |
| `emitCIArtifacts()` | `ISRDebugRuntime.cpp` | 内部処理 |
| `emitHBTrace()` | `ISRDebugRuntime.cpp` | 内部処理 |

特に `emitShutdownTrace()` は Generation64 の影響を受けない（ShutdownPhase のみ出力）。`emitOwnershipTrace()` と `emitRetireTimeline()` は generation 値を JSON に書き込むため、uint64_t 対応の確認が必要。

### D-8. isFullyDrained() の現状（最終確認）

**結論**: 2重チェックを確認：

1. `AudioEngine::isFullyDrained()` → `runtimePublicationBridge_.isFullyDrained()` + `!hasDeferredCommit`
2. `RuntimePublicationCoordinator::isFullyDrained()` → 7個の atomic チェック（`swapPending_`, `retireBacklogCount_`, `publicationBacklogCount_`, `pendingIntentCount_`, `fallbackBacklogCount_`, `reclaimInFlightCount_`, `deferredRetireResidencyCount_`）

Phase A-2 では `collectDrainAudit()` でラップする方針は妥当。`runtimePublicationBridge_.isFullyDrained()` の bool 値に加えて、個別カウンタ値を getter 経由で `RuntimeDrainAudit` にマッピングする。

### D-9. 計画書と実コードの乖離一覧

計画書の記述と現行コードの間に乖離がないか確認した結果：

| 計画書の記述 | 現行コード | 乖離 |
| :--- | :--- | :--- |
| DSPHandleRuntime::quarantine() は未呼び出し（デッドコード） | `ISRDSPHandle.cpp:112` に定義のみ。呼び出し元なし | ✅ 一致 |
| dspQuarantineManager_ は宣言のみで未使用 | `AudioEngine.h:3443` に宣言のみ | ✅ 一致 |
| DSPHandle::generation は uint32_t | `ISRDSPHandle.h:24` で `uint32_t` | ✅ 一致 |
| RetireIntent::generation は uint32_t | `ISRRetire.h:20` で `uint32_t` | ✅ 一致 |
| 現状の deferred publish は std::optional 単一保持 | `RuntimePublicationOrchestrator.h:48` で `std::optional<PublishRequest>` | ✅ 一致 |
| isShutdownInProgress は AudioEngine と ShutdownRuntime の2系統 | `AudioEngine.h:1014`（lifecycleState） + `ISRShutdown.h:49`（ShutdownPhase） | ✅ 一致 |
| oldestPendingAge_ が AudioEngine に存在 | `AudioEngine.h:1571` で `std::atomic<double>` | ✅ 一致 |
| CrossfadeRuntime::isPending() が存在 | `CrossfadeRuntime.h:55` で実装 | ✅ 一致 |
| CrossfadeAuthorityRuntime::getActiveCrossfades() が存在 | `ISRDSPHandle.h:171` で宣言、`ISRDSPHandle.cpp:168` で実装 | ✅ 一致 |
| PublicationSequenceId = uint64_t | `ISRRuntimeSemanticSchema.h:195` で `using` | ✅ 一致 |

**総評**: 計画書と現行コードの間に重大な乖離はない。全ての記述が現行コードと整合している。

---

*初版 (v1) からの主な変更点:*

1. *Quarantine: create() 改修不要の判断。`DSPHandleRuntime::quarantine()` と `DSPQuarantineManager` を AudioEngine で接続*
2. *Shutdown: 二重判定排除。Authority を `ShutdownRuntime` に一本化。`quarantineResident` を完了条件から除外*
3. *Reclaim: ReclaimTicket/カウンタ追跡を破棄。`emitRetireTrace()` によるスナップショットダンプへ変更*
4. *Overflow: `RuntimePressureState` 列挙型追加を撤回。既存 int level 維持 + コメント強化*
5. *Deferred Publish: FIFOリングバッファを撤回。sequence 番号による stale discard 方式へ変更*

*v2 での修正点（レビューフィードバック反映 2026-06-08）:*

1. *① quarantineHandle の generation 引数を最初から uint64_t 化（Phase C-1 先行対応）*
2. *② forceReclaim() を廃止。代わりに shutdown 専用の destroyQuarantineSlot() + destroyForShutdown() を追加*
3. *③ Shutdown 判定は OR 判定を永久維持（完全委譲は行わない）。EngineLifecycleState と ShutdownPhase の二重保護を常時維持*
4. *④ activeTransitionCount() 新規追加を撤回。既存 CrossfadeRuntime::isPending() を利用*
5. *⑤ emitRetireTrace() は条件付き出力（正常 shutdown では出力しない）。長期運用のログ肥大化防止*
6. *⑦ overflowScore に加えて overflowStartTimestamp_ を追加。overflow 継続時間（秒単位）を監査項目化*
7. *⑧ DSPQuarantineManager から isQuarantined() を削除（Authority である DSPHandleRuntime のみが状態判定を行う）*
8. *⑨ RuntimeDrainAudit::activeCrossfade → activeCrossfadeCount にリネーム（bool→countの一貫性確保）*
9. *⑩ Overflow 制御: 直接フラグ上書きから effectiveLevel=max(normal,overflow) の合成方式に変更（ラッチ化防止）*
10. *⑪ ShutdownPhase 拡張（ShutdownRequested）を Phase D（将来）へ延期。A-2 では enum 変更しない*
11. *⑫ QuarantineManager::Entry のアクセスモデルを明記（NonRT専用。RTからのreadはactive atomicのみ）*
12. *⑬ emitRetireTrace 出力ファイルを毎回上書き方式（retire_trace_shutdown_last.json）に変更。ディレクトリ肥大化防止*
13. *⑭ pendingPublication の完了条件化を明記。A-2 完了条件に「全フィールド実測値」を含める*

---

## 付録E: 全ツール横断調査による最終確定事項（2026-06-08）

v2 最終版策定後、全ツール（grep/Select-String/Serena MCP/CodeGraph MCP/ccc/semble/Graphify/ファイル精読）を使用した徹底調査により、以下の事項を確定した。

### E-1. AudioEngine::isShutdownInProgress() の現状

**結論**: `EngineLifecycleState` のみを参照。`ShutdownRuntime` は未参照。

```cpp
// AudioEngine.h:1014-1021
[[nodiscard]] bool isShutdownInProgress() const noexcept {
    const auto state = consumeAtomic(lifecycleState, std::memory_order_acquire);
    return state == EngineLifecycleState::Releasing
        || state == EngineLifecycleState::Destroyed;
}
```

呼び出し元は **20箇所以上**（Commit/Processing/RebuildDispatch/Retire/Snapshot/Timer等）。一方 `Commit.cpp:402` のみ `shutdownRuntime_.isShutdownInProgress()` を直接参照している。

**計画との整合**: ✅ 一致。OR 判定への変更が必要。

### E-2. PublicationAdmission::evaluate() の shutdown 参照経路

**結論**: `engine.isShutdownInProgress()` を呼んでいる（`PublicationAdmission.cpp:11`）。AudioEngine の実装変更（OR 判定追加）のみで自動的に保護強化される。

### E-3. DSPHandleRuntime::quarantine() のデッドコード確定

**結論**: ✅ 定義のみ (`ISRDSPHandle.cpp:112-117`)。.cpp 内のどの関数からも呼ばれていない。grep/Serena/semble の全ツールで呼び出し元ゼロを確認。

### E-4. dspQuarantineManager_ のデッドメンバ確定

**結論**: ✅ `AudioEngine.h:3443` で宣言されているが、全 .cpp ファイルで参照ゼロ。grep/Serena で確認。

### E-5. CrossfadeAuthorityRuntime::getActiveCrossfades() の完全シグネチャ

```cpp
// ISRDSPHandle.h:171
std::vector<CrossfadeRecord> getActiveCrossfades() const noexcept;
```

内部で `std::vector<CrossfadeRecord>` を構築するため、NonRT スレッドからのみ呼び出し可能。DrainAudit での利用に問題なし。

### E-6. CrossfadeRuntime::isPending() の実装

```cpp
// CrossfadeRuntime.h:63-64
[[nodiscard]] bool isPending() const noexcept {
    return convo::consumeAtomic(pending_, std::memory_order_acquire);
}
```

✅ `pending_` atomic を読むのみ。lock-free、RT スレッドからも安全に呼び出し可能。

### E-7. DSPRegistrySlot の完全定義

```cpp
// ISRDSPHandle.h:90-95
struct DSPRegistrySlot {
    std::atomic<uint32_t> generation;  // C-1 で uint64_t 化予定
    void*                 instance;
    std::atomic<DSPState> state;
};
```

`static_assert(std::atomic<uint64_t>::is_always_lock_free)` の追加が必要。

### E-8. DSPHandle の完全定義

```cpp
// ISRDSPHandle.h:20-23
struct DSPHandle {
    uint32_t slot;       // レジストリスロット番号
    uint32_t generation; // 世代番号（C-1 で uint64_t 化予定）
};
```

CrossfadeRecord（`fromHandle`/`toHandle`）が DSPHandle を含むため、型変更のみで自動追従する。

### E-9. RetireIntent の generation 型確認

```cpp
// ISRRetire.h:20-24
struct RetireIntent {
    uint32_t dspSlot;
    uint32_t generation;  // C-1 で uint64_t 化予定
    uint64_t retireEpoch;
    bool isValid;
};
```

### E-10. 既存 generation truncation の完全確認

```cpp
// AudioEngine.Commit.cpp:410-413
std::uint32_t generation = static_cast<std::uint32_t>(world->generation & 0xFFFFFFFFu);
if (generation == 0u)
    generation = 1u;
```

`world->generation` は `uint64_t`。32bit への truncation により `generation == 0u → 1u` の置換が `0x1_0000_0000` 相当の上位ビットのみ立っている場合にも発生する。

### E-11. emitIntent() の generation 比較

```cpp
// ISRRetireRuntimeEx.cpp:160-170
const auto lane = (generation == 0u) ? RetireLane::Quarantine : RetireLane::RTIntent;
```

C-1 後は `generation == 0ull` への変更が必要。

### E-12. generation 全出現箇所（Phase C-1 棚卸し）

grep/ccc/semble による横断調査の結果、以下が Phase C-1 の変更対象：

| シンボル | ファイル | 現状 | 変更後 |
| :--- | :--- | :--- | :--- |
| `DSPHandle::generation` | `ISRDSPHandle.h:23` | `uint32_t` | `uint64_t` |
| `DSPRegistrySlot::generation` | `ISRDSPHandle.h:92` | `std::atomic<uint32_t>` | `std::atomic<uint64_t>` |
| `RetireIntent::generation` | `ISRRetire.h:22` | `uint32_t` | `uint64_t` |
| `acknowledgeGeneration_[]` | `ISRRetire.h:56` | `std::atomic<uint32_t>` (array) | `std::atomic<uint64_t>` (array) |
| `RetireRuntimeEx::emitIntent()` 引数 | `ISRRetireRuntimeEx.h:42` | `std::uint32_t generation` | `std::uint64_t` |
| `DSPQuarantineManager::quarantineHandle()` 引数 | `ISRDSPQuarantine.h:11` | `std::uint32_t generation` | `std::uint64_t` (A-1 で先行対応) |
| `Commit.cpp:410` truncation | `AudioEngine.Commit.cpp:410` | `static_cast<uint32_t>(...)` | 削除 |
| `generation == 0u` 比較 | `ISRRetireRuntimeEx.cpp:165` | `0u` | `0ull` |

### E-13. oldestPendingAge_ の完全な更新パス

```cpp
// AudioEngine.Commit.cpp:483-491
convo::publishAtomic(oldestPendingAge_, std::max(0.0, nowMs - firstSeen), ...); // 滞留中
convo::publishAtomic(oldestPendingAge_, 0.0, ...);  // 滞留解消
const double oldestPendingAgeMs = convo::consumeAtomic(oldestPendingAge_, ...); // 読み取り
```

`oldestPendingAge_` は `AudioEngine.Commit.cpp` の `onRuntimeRetiredNonRt()` 内（NonRT パス）でのみ更新される。DrainAudit からの読み取りにも問題なし。

### E-14. コードグラフ（CodeGraph）インデックス状況

CodeGraph Incremental Index 実行結果:

- Entities: 0, Relations: 0, Files Indexed: 0（前回と差分なしのため）
- Communities: 52, データベースサイズ: 11868 entities
- ツール使用前のインデックス整備は完了

### E-15. 全ツール調査結果サマリー

| 調査項目 | ツール | 確定結果 | 計画との整合 |
| :--- | :--- | :--- | :--- |
| `DSPHandleRuntime::quarantine()` 呼び出し元 | grep/Serena/semble | **呼び出し元ゼロ**（デッドコード） | ✅ 一致 |
| `dspQuarantineManager_` 使用箇所 | grep/Serena | **全 .cpp で参照ゼロ**（デッドメンバ） | ✅ 一致 |
| `isShutdownInProgress()` 実装 | grep/ファイル精読 | `EngineLifecycleState` のみ参照。`ShutdownRuntime` 未参照 | ✅ OR化が必要 |
| `oldestPendingAge_` 更新パス | grep/ファイル精読 | `AudioEngine.Commit.cpp:483-491` NonRT パス | ✅ 既存活用可能 |
| `CrossfadeRuntime::isPending()` | grep/ファイル精読 | `bool` を返す atomic 読み取り | ✅ DrainAudit で利用 |
| `CrossfadeAuthorityRuntime::getActiveCrossfades()` | grep/ファイル精読 | `vector<CrossfadeRecord>` 返却 | ✅ NonRT で利用可 |
| `laneName()`/`lifecycleStateName()` | grep | **存在しない** | ✅ B-1 で追加必要 |
| `emitRetireTimeline()` generation 出力 | grep/ファイル精読 | generation の JSON 出力はなし（lane counterのみ） | ✅ C-1 棚卸し対象外 |
| `PublicationAdmission::evaluate()` の shutdown 参照 | grep/ファイル精読 | `engine.isShutdownInProgress()` 呼び出し | ✅ OR化で自動保護 |
| `RetireRuntimeEx` API 一覧 | grep/ファイル精読 | `laneOf()`, `lifecycleStateOf()` 存在 | ✅ emitRetireTrace 実装可能 |
| generation truncation | grep/semble/ccc | `Commit.cpp:410` に実在 | ✅ B-1 で削除予定 |
| DSPQuarantineManager 現状 | grep/ファイル精読 | `vector<atomic<bool>>` のみ。reason/timestampなし | ✅ A-1 で拡張予定 |

## 付録F: 第3ラウンド追加検証による確定事項（2026-06-08）

全6ツール（grep/Select-String/Serena MCP/CodeGraph MCP/ccc/semble/Graphify）を使用した第3ラウンドの追加検証により、以下の事項を確定した。

### F-1. DSPHandle のハッシュ/コンテナ使用状況

**結論**: `std::hash<DSPHandle>`, `std::unordered_map<DSPHandle, ...>`, `std::unordered_set<DSPHandle>` は**存在しない**。grep/ccc/semble の全ツールで確認。Phase C-1 (Generation64bit化) での影響はない。

### F-2. DSPHandleRuntime の quarantine 方式

**現状**: `DSPHandleRuntime::quarantine(DSPHandle handle)` は generation 一致を要求する。`pending.generation`（`onRuntimeRetiredNonRt()` 時点の値）と `registry_[slot].generation`（現在値）が乖離した場合、quarantine が何も行わない（generation mismatch による no-op）。

**リスク**: retire→crossfade→epoch待ち→quarantine の経路で generation 更新が発生し得る。

**推奨変更**: `quarantine(DSPHandle)` の代わりに `quarantineSlot(uint32_t slot)` を追加。generation 一致を要求せず、slot の state のみを `Quarantined` に設定。これにより generation 乖離による quarantine 漏れを防止する。

```cpp
// ISRDSPHandle.h — 追加メソッド
// Slot 直接 quarantine: generation 一致を要求しない
// Handle ベース quarantine が generation mismatch で no-op になるのを防止
void quarantineSlot(uint32_t slot) noexcept;
// 既存の quarantine(DSPHandle) は維持（generation 一致が必要な厳格な隔離）
```

```cpp
// ISRDSPHandle.cpp — 実装
void DSPHandleRuntime::quarantineSlot(uint32_t slot) noexcept
{
    if (slot >= MAX_DSP_SLOTS) return;
    convo::publishAtomic(registry_[slot].state, DSPState::Quarantined,
                         std::memory_order_release);
}
```

### F-3. EngineLifecycleState 全参照の棚卸し

全参照箇所（18箇所）を特定：

| ファイル | 行 | 参照内容 | OR化影響 |
| :--- | :--- | :--- | :--- |
| `AudioEngine.h:1019-1020` | isShutdownInProgress | `Releasing/Destroyed`参照 | ✅ OR化で変更 |
| `AudioEngine.CtorDtor.cpp:46` | デストラクタ | `lifecycleState=Releasing`設定 | ❌ 設定側、影響なし |
| `AudioEngine.CtorDtor.cpp:141` | デストラクタ | `lifecycleState=Destroyed`設定 | ❌ 設定側、影響なし |
| `AudioEngine.Processing.AudioBlock.cpp:18` | processBlock | `!=Prepared`で早期return | ❌ 別系統、影響なし |
| `AudioEngine.Processing.BlockDouble.cpp:18` | processBlock | `!=Prepared`で早期return | ❌ 別系統、影響なし |
| `AudioEngine.Processing.PrepareToPlay.cpp:36` | prepareToPlay | `lifecycleState=Unprepared`設定 | ❌ 設定側、影響なし |
| `AudioEngine.Processing.PrepareToPlay.cpp:42-44` | prepareToPlay | state遷移前チェック | ❌ 別系統、影響なし |
| `AudioEngine.Processing.PrepareToPlay.cpp:52` | prepareToPlay | `state=Preparing`設定 | ❌ 設定側、影響なし |
| `AudioEngine.Processing.PrepareToPlay.cpp:195` | prepareToPlay | `state=Prepared`設定 | ❌ 設定側、影響なし |
| `AudioEngine.Processing.ReleaseResources.cpp:24-44` | releaseResources | state遷移チェック+Releasing設定 | ❌ 設定側、影響なし |
| `AudioEngine.Processing.ReleaseResources.cpp:236` | releaseResources | `state=Unprepared`設定 | ❌ 設定側、影響なし |

**結論**: isShutdownInProgress() 内の `EngineLifecycleState` 参照のみが OR 化の影響を受ける。他の参照は設定側または別目的（Prepared/Preparing チェック）であり、shutdown 判定とは無関係。OR 化による副作用はない。

### F-4. overflowScore 方式の再検討

**問題点**: 当初計画の `overflowScore_`（overflow→+4, success→-1, 上限16）は、overflow/success 交互パターン（+4-1+4-1...）でも徐々に蓄積するため、理論上は慢性的詰まりを検出可能。しかし以下の課題がある：

1. スコアの物理的意味が不明瞭（継続時間なのか回数なのか）
2. RT/NonRT 混在環境での atomic 整合性コスト
3. 単純な `droppedDelta > 0` の方が実運用で安定

**推奨変更**: `overflowScore_` を削除し、代わりに以下を採用：

1. **`droppedDelta`**（前回監視時点との差分）による overflow 検出（既存案通り）
2. **`overflowStartTimestamp`** による継続時間追跡（既存案通り）
3. **`overflowWindowCount`**（直近N秒間の overflow 回数）は追加せず、`overflowDurationMs` で代替

理由：実運用では「何回起きたか」より「どれだけ長く続いたか」が重要であり、継続時間監視で十分な実用耐性が得られる。

**計画書の修正**: ISRRetire.h の追加メンバを以下に変更：

```cpp
// ISRRetire.h — 追加メンバ（overflowScore_ は削除）
std::atomic<uint64_t> lastOverflowTicks_{0};       // 最終 overflow 発生時刻
std::atomic<uint64_t> overflowStartTimestamp_{0};  // overflow 継続開始時刻（0=非overflow）
```

### F-5. emitRetireTrace() の同期問題

**問題点**: `emitRetireTrace()` が Runtime 稼働中に呼ばれた場合、`laneOf(slot)` と `lifecycleStateOf(slot)` の間に原子性がなく、矛盾した状態（例: lane=Epoch かつ lifecycle=Reclaimed）が出力される可能性がある。

**推奨変更**: emitRetireTrace() は **Shutdown path のみ** で使用する。稼働中の定期監査用途には使用しない。その理由として以下を明記する：

- Shutdown 時は他スレッドが停止しているため矛盾が発生しない
- 稼働中に slot 状態の一貫性が必要な場合は、事前に全 slot の atomic スナップショットを取得してから出力する設計にする

### F-6. DSPHandleRuntime::quarantineSlot の組み込み

**計画書 3.2.3 の修正**: `DSPHandleRuntime::quarantine(DSPHandle)` の代わりに `DSPHandleRuntime::quarantineSlot(uint32_t slot)` を使用する。

```cpp
// AudioEngine.Commit.cpp — 変更後
// 1. RetireLane の quarantine 遷移（既存）
retireRuntimeEx_.quarantine(pendingSlot);

// 2. Slot 直接 quarantine（新規: generation 一致不要）
dspHandleRuntime_.quarantineSlot(pendingSlot);

// 3. 隔離理由を記録（新規）
dspQuarantineManager_.quarantineHandle(
    pendingSlot, generation,  // ← この generation は Commit.cpp のローカル変数
    convo::isr::QuarantineReason::RetireDeferralTimeout);
```

**注意**: `quarantineSlot()` は generation 一致を要求しないため、より確実に隔離できる。一方、`DSPHandleRuntime::quarantine(DSPHandle)` は維持（generation 一致が必要な厳格な隔離パスとして残すが当面未使用）。

### F-7. 変更履歴

1. *⑮ quarantine 方式を Handle ベースから Slot ベース（quarantineSlot）に変更（generation 乖離による quarantine 漏れ防止）*
2. *⑯ overflowScore_ を削除。継続時間追跡（overflowStartTimestamp_）のみに統一*
3. *⑰ overflowScore 関連の記述を全削除し、droppedDelta + overflowStartTimestamp のみの設計に統一*
4. *⑱ EngineLifecycleState 全参照箇所を棚卸し（18箇所）。isShutdownInProgress()以外はOR化の影響なしを確認*
5. *⑲ `hash<DSPHandle>` / `unordered_map<DSPHandle>` / `unordered_set<DSPHandle>` の不在を確認（Generation64bit化に影響なし）*
6. *⑳ emitRetireTrace() を Shutdown path のみに制限（稼働中の原子性問題を回避）*
7. *㉝ **Generation Dependency Audit** の独立実施を Phase C-1 完了条件に追加*
   - **3軸監査**: `generation`（全メンバ変数）, `uint32_t`（全 typedef/using）, `hash`（全 std::hash 特殊化 + unordered_map/unordered_set キー型）
   - **補助構造体**: `struct Key { uint32_t generation; }` のような generation を含む補助構造体も検出
   - **ツール**: `grep -rn generation src/`, `grep -rn 'uint32_t' src/`, `ccc search 'hash.*generation'`, `ccc search 'struct.*generation'`
   - **完了条件**: 「全 generation 型の uint64_t 化完了 + 全補助構造体の対応完了 + 全 hash 特殊化の変更完了」

### F-8. 最終ラウンド追加確定事項（2026-06-08）

#### F-8.6 第5ラウンド全ツール調査による確定事項（2026-06-08）

grep/Select-String/Serena MCP/CodeGraph MCP/ccc/semble/Graphify/ファイル精読を駆使した第5ラウンド調査により、以下の全未確定事項を確定した。

##### F-8.6.1 DSPState enum — DestroyPending は未実装

**結論**: 現行 `DSPState` 列挙型には `DestroyPending` は**存在しない**。現在の列挙子は `Constructing` / `Active` / `CrossfadingIn` / `CrossfadingOut` / `Retired` / `Quarantined` / `Reclaimed` の7種。計画書の destroyQuarantineSlot 2段階化に伴い、`DestroyPending` の追加が必須。

##### F-8.6.2 DSPQuarantineManager 現状 — フラグのみ、全拡張未実装

**結論**: 現行 `DSPQuarantineManager` は以下を持つ：

- `quarantineHandle(uint32_t slot, uint32_t)` — generation 引数は**未使用**（unnamed parameter）。単に `vector<atomic<bool>>` のフラグを true に設定
- `reclaimSlot(uint32_t slot)` — generation 引数なし。単にフラグを false に設定
- `isQuarantined(uint32_t slot)` — 存在する（計画書では削除予定）

以下のメソッドは**いずれも存在しない**（計画書で追加予定の新規API）：

| メソッド | 計画書 | 現状 |
| :--- | :--- | :--- |
| `getEntry(slot)` | A-1 | ❌ 未実装 |
| `residentCount()` | A-1 | ❌ 未実装 |
| `destroyForShutdown(slot)` | A-1 | ❌ 未実装 |
| `getMaxEntryAgeSec()` | 監査TTL | ❌ 未実装 |
| `Entry` 構造体（generation/reason/timestamp） | A-1 | ❌ 未実装 |

`dspQuarantineManager_` は `AudioEngine.h` で宣言されているが、全 .cpp ファイルで**参照ゼロ**（デッドメンバ変数）。

##### F-8.6.3 DSPHandleRuntime 現状 — quarantineSlot/isSlotInCrossfade/destroyQuarantineSlot 未実装

**結論**: `DSPHandleRuntime` に以下のメソッドは**存在しない**：

- `quarantineSlot(uint32_t slot)` — 計画書 A-1 / F-6 で追加予定
- `isSlotInCrossfade(uint32_t slot)` — 計画書で destroyQuarantineSlot の事前チェック用に追加予定
- `destroyQuarantineSlot(slot, generation)` — 計画書で追加予定

`DSPHandleRuntime::quarantine(DSPHandle)` は**定義済みだが呼び出し元ゼロ**（デッドコード）。`DSPHandleRuntime::getActiveRuntimeDSPHandle()` と `getFadingRuntimeDSPHandle()` は実装済み。

##### F-8.6.4 RuntimePublicationOrchestrator 現状

**結論**: 現行 `RuntimePublicationOrchestrator` は以下を持つ：

- `std::optional<PublishRequest> deferredRequest_`（`DeferredPublishSlot` ではない）
- `bool hasDeferred_`
- `hasDeferredRequest()` と `consumeDeferredRequest()`（inline実装済み）

以下のメソッドは**存在しない**：

| メソッド | 計画書 | 現状 |
| :--- | :--- | :--- |
| `DeferredPublishSlot` 構造体 | C-2 | ❌ 未実装 |
| `clearDeferredForShutdown()` | shutdown強制消去 | ❌ 未実装 |
| `maxDeferredAgeMs_` / `getMaxDeferredAgeMs()` | 監査 | ❌ 未実装 |
| `deferredOverwriteCount_` | 監査 | ❌ 未実装 |

`notifyTransitionComplete()` には shutdown ガードが**ない**（計画書 A-2.2 の指摘通り）。

##### F-8.6.5 RetireRuntime 現状

**結論**: 現行 `RetireRuntime` は `overflowCount_` / `droppedIntentCount_` のみを持つ。以下は**存在しない**：

- `overflowStartTimestamp_` / `lastOverflowTicks_`（C-1 追加予定）
- `overflowWindowCounter_` / `lastOverflowWindowResetTicks_` / `lastOverflowWindowCount_`（C-1 追加予定）
- `overflowWindowCounter()` ゲッター

##### F-8.6.6 RuntimePublicationCoordinator — getter 群未実装

**結論**: 以下の getter は**存在しない**（Appendix C-2 で追加予定）：

- `getPublicationBacklogCount()`
- `getPendingIntentCount()`
- `getRetireBacklogCount()`
- `getFallbackBacklogCount()`
- `getDeferredRetireResidencyCount()`

既存は `getReclaimInFlightCount()` / `isSwapPending()` / `isFullyDrained()` のみ。

##### F-8.6.7 RuntimeConfig — 未実装

**結論**: `RuntimeConfig` 構造体は現行コードのいずれの .h/.cpp にも**存在しない**。計画書で overflow 閾値の RuntimeConfig 化を提案しているが、実際の config 機構は Phase C-1 で新規追加が必要。

##### F-8.6.8 Generation Dependency Audit — 補助構造体の検出結果

grep/ccc/semble による横断調査の結果、以下の generation 関連補助構造体を検出：

| 構造体/変数 | ファイル | 型 | 備考 |
| :--- | :--- | :--- | :--- |
| `DSPHandle::generation` | `ISRDSPHandle.h:23` | `uint32_t` | ★ Phase C-1 変更対象 |
| `DSPRegistrySlot::generation` | `ISRDSPHandle.h:92` | `std::atomic<uint32_t>` | ★ Phase C-1 変更対象 |
| `RetireIntent::generation` | `ISRRetire.h:22` | `uint32_t` | ★ Phase C-1 変更対象 |
| `acknowledgeGeneration_` | `ISRRetire.h:56` | `std::atomic<uint32_t>[]` | ★ Phase C-1 変更対象（要確認） |
| `emitIntent()` generation 引数 | `ISRRetireRuntimeEx.h:42` | `std::uint32_t` | ★ Phase C-1 変更対象 |
| `quarantineHandle()` generation 引数 | `ISRDSPQuarantine.h:11` | `std::uint32_t` | ★ A-1/C-1 変更対象 |
| `AdaptiveCoeffBankSlot::generation` | `AudioEngine.h:1845` | `std::atomic<uint32_t>` | ISR generation とは別系統。**変更不要**だが Generation Dependency Audit の検出対象 |
| `adaptiveCoeffGeneration` | `AudioEngine.h:350,557,682` | `uint32_t` | ISR generation とは別系統。変更不要 |
| `adaptiveCoeffGeneration` | `ISRRuntimeSemanticSchema.h:339` | `std::uint32_t` | ISR generation とは別系統。変更不要 |
| `adaptiveCoeffGeneration` | `RuntimeTransition.h:58` | `std::uint32_t` | ISR generation とは別系統。変更不要 |
| `std::hash<DSPHandle>` | 全局 | — | **存在しない** ✅ |
| `CrossfadeRecord::fromHandle/toHandle` | `ISRDSPHandle.h:80-81` | `DSPHandle` | `uint64_t` 化に自動追従 ✅ |
| `activeRuntimeDSPHandle_` | `ISRDSPHandle.h:145` | `std::atomic<DSPHandle>` | `uint64_t` 化に自動追従 ✅ |
| `fadingRuntimeDSPHandle_` | `ISRDSPHandle.h:146` | `std::atomic<DSPHandle>` | `uint64_t` 化に自動追従 ✅ |

##### F-8.6.9 ShutdownRuntime — advancePhase は deferred を消去しない

**結論**: `ISRShutdown.cpp` の `advancePhase()` は `RuntimePublicationOrchestrator` への参照を持たず、deferred publish の消去を行わない。計画書の `clearDeferredForShutdown()` は新規実装が必要。

##### F-8.6.10 調査サマリーテーブル

| 調査項目 | ツール | 確定結果 | 計画書との整合 |
| :--- | :--- | :--- | :--- |
| `DestroyPending` の有無 | grep/ccc | ❌ 未実装 | ✅ A-1 で追加予定 |
| `DSPQuarantineManager` の現状 | grep/Serena/ccc | フラグのみ。全拡張未実装 | ✅ 全て計画通りの追加予定 |
| `dspQuarantineManager_` 使用状況 | grep/Serena | 全 .cpp で**参照ゼロ** | ✅ デッドメンバ確定 |
| `DSPHandleRuntime::quarantine(DSPHandle)`呼び出し元 | grep/ccc/semble | **呼び出し元ゼロ** | ✅ デッドコード確定 |
| `DSPHandleRuntime::quarantineSlot()` | grep | ❌ 未実装 | ✅ F-6 で追加予定 |
| `DSPHandleRuntime::isSlotInCrossfade()` | grep | ❌ 未実装 | ⚠️ 本調査で新規発見。A-1 前提タスク |
| `RuntimePublicationOrchestrator` deferred 現状 | grep/ファイル精読 | `optional<PublishRequest>`のみ | ✅ 全拡張は計画通り |
| `clearDeferredForShutdown()` | grep | ❌ 未実装 | ✅ 新規指摘で追加 |
| `maxDeferredAgeMs_` | grep | ❌ 未実装 | ✅ 新規指摘で追加 |
| `deferredOverwriteCount_` | grep | ❌ 未実装 | ✅ F-8.1 で追加予定 |
| `RuntimePublicationCoordinator` getter 群 | grep | ❌ 未実装（5種） | ✅ C-2 で追加予定 |
| `RetireRuntime` overflow 追加メンバ | grep | ❌ 未実装（5種） | ✅ C-1 で追加予定 |
| `RuntimeConfig` | grep | ❌ 未実装 | ⚠️ 本調査で確認。Phase C-1 で新規追加必要 |
| `std::hash<DSPHandle>` 特殊化 | grep/ccc/semble | **存在しない** ✅ | ✅ 影響なし確認 |
| generation 補助構造体 | grep/ccc | `adaptiveCoeffGeneration` は別系統 | ⚠️ Generation Dependency Audit 対象だが変更不要 |
| ISR generation 型変更対象 | grep/ccc | 6箇所（DSPHandle/RegistrySlot/RetireIntent/acknowledgeGeneration/emitIntent/quarantineHandle） | ✅ Phase C-1 で変更予定 |
| ShutdownRuntime advancePhase | ファイル精読 | deferred 消去を行わない | ✅ clearDeferredForShutdown 追加予定 |
| `notifyTransitionComplete` shutdown ガード | ファイル精読 | **存在しない** | ✅ A-2.2 で追加予定 |
| **Authority 集約の論点** | 全体設計レビュー | 現在3系統（DSPHandleRuntime/QuarantineManager/RetireRuntimeEx）に分散 | ⚠️ 将来 RetireRuntime への単一集約を検討（F-8.7 参照） |

#### F-8.7 Authority 集約の論点 — RetireRuntime を唯一の状態管理主体とする提案

**背景**: 現在の計画では状態管理 Authority が以下の3系統に分散している：

| 主体 | 管理対象 | アクセス層 |
| :--- | :--- | :--- |
| `DSPHandleRuntime` | `DSPState`（slot state） | RT/NonRT |
| `DSPQuarantineManager` | quarantine 隔離フラグ + 監査記録 | NonRT（RTはatomic read only） |
| `RetireRuntimeEx` | `RetireLane` / `RetireLifecycleState` | NonRT |

この分散により、以下の subtle race が理論上成立する：

1. `DSPHandleRuntime` が quarantine を設定
2. 直後に `RetireRuntimeEx` が同一 slot を別 lane に遷移
3. `DSPQuarantineManager` の隔離フラグが未設定

**Single Authority Principle の提案**: 長期的には `RetireRuntimeEx` を唯一の state authority とし、`DSPHandleRuntime` と `DSPQuarantineManager` を derived state / projection に変更する：

```text
現在:
  DSPHandleRuntime (DSPState)
  QuarantineManager (quarantine flag)
  RetireRuntimeEx (lane / lifecycle)

将来:
  RetireRuntimeEx (sole state authority)
    ├── DSPHandleRuntime → projection (resolve() 経由の読み取り専用)
    └── QuarantineManager → derived state (RetireLane::Quarantine から自動判定)
```

**ただし現段階での変更は推奨しない。** 理由：

- この変更は ISR Runtime の core architecture に涉及する大規模リファクタリング
- 現在の3系統分離でも実運用上の問題は確認されていない
- Phase A-1/A-2/B-1/C-1/C-2 の完了後に、余剰設計として評価するのが妥当

現時点では「Authority 集約の方向性」を認識しておき、Phase D（将来）で検討する。

#### F-8.1 deferredOverwriteCount_ の追加

**RuntimePublicationOrchestrator** に監査用カウンタ `deferredOverwriteCount_` を追加する。`enqueueDeferred()` が上書きされるたびにインクリメント：

```cpp
// RuntimePublicationOrchestrator.h — 追加メンバ
std::atomic<uint64_t> deferredOverwriteCount_{0};
```

```cpp
// enqueueDeferred() 内 — 上書きカウント
void enqueueDeferred(const PublicationAdmission::PublishRequest& req) noexcept
{
    if (hasDeferred_)
        convo::fetchAddAtomic(deferredOverwriteCount_, uint64_t{1}, ...);
    deferredRequest_ = req;
    hasDeferred_ = true;
}
```

shutdown trace または DrainAudit に `deferredOverwriteCount` を出力することで、deferred publish の輻輳状況を把握可能。これにより「何回上書きされたか」の可視性が向上する。

**さらに `maxDeferredAgeMs`（最長滞留時間）も併用することで、「上書きが多いのか」「クロスフェードが終わらないのか」の原因分析が可能になる**。

- `deferredOverwriteCount` が高く `maxDeferredAgeMs` が低い → publish 要求が頻繁に更新されている（設計上の制約内）
- `deferredOverwriteCount` が低く `maxDeferredAgeMs` が高い → 一度 enqueue された publish が長時間滞留（problematic。crossfade が完了しない等）

`RuntimePublicationOrchestrator` に以下のメンバを追加する：

```cpp
// RuntimePublicationOrchestrator.h — 追加メンバ
// enqueueDeferred 時に現在時刻を記録。consumeDeferredRequest 時に経過時間を更新。
std::atomic<uint64_t> maxDeferredAgeMs_{0};
uint64_t deferredEnqueueTimestampUs_{0};

// getMaxDeferredAgeMs — DrainAudit からの参照用
[[nodiscard]] uint64_t getMaxDeferredAgeMs() const noexcept {
    return convo::consumeAtomic(maxDeferredAgeMs_, std::memory_order_acquire);
}
```

```cpp
// enqueueDeferred() 内
void enqueueDeferred(const PublicationAdmission::PublishRequest& req) noexcept
{
    if (hasDeferred_) {
        convo::fetchAddAtomic(deferredOverwriteCount_, uint64_t{1}, ...);
        // 上書き時は滞留時間を maxDeferredAgeMs に反映
        const auto now = getCurrentTicks();  // 現在時刻(us)
        const uint64_t ageMs = (now - deferredEnqueueTimestampUs_) / 1000;
        uint64_t currentMax = convo::consumeAtomic(maxDeferredAgeMs_, ...);
        while (ageMs > currentMax) {
            if (convo::compareExchangeAtomic(maxDeferredAgeMs_, currentMax, ageMs, ...))
                break;
        }
    }
    deferredEnqueueTimestampUs_ = getCurrentTicks();
    deferredRequest_ = req;
    hasDeferred_ = true;
}
```

#### F-8.2 atomic<uint64_t> lock-free の事前確認

現行コードベースには `std::atomic<uint64_t>::is_always_lock_free` の static_assert は**存在しない**。Phase C-1 の実装前に以下の static_assert を追加することを必須確認事項とする：

```cpp
// 以下の atomic<uint64_t> 変更箇所に static_assert を追加
static_assert(std::atomic<uint64_t>::is_always_lock_free,
    "atomic<uint64_t> must be lock-free on x64 for ISR Runtime");
```

**確認対象変数**:

| 変数 | ファイル | 変更後 |
| :--- | :--- | :--- |
| `DSPRegistrySlot::generation` | `ISRDSPHandle.h:92` | `std::atomic<uint64_t>` |
| `DSPHandle::generation` | `ISRDSPHandle.h:23` | `uint64_t`（非atomic、static_assert不要） |
| `RetireIntent::generation` | `ISRRetire.h:22` | `uint64_t`（非atomic、static_assert不要） |

x64 (MSVC) では `std::atomic<uint64_t>` は常に lock-free。ARM32/古いAtom/組み込み環境を考慮した保険として追加する。

#### F-8.3 変更履歴

1. *㉑ deferredOverwriteCount_ 追加（RuntimePublicationOrchestrator）。deferred publish の上書き回数を監査可能に*
2. *㉒ atomic<uint64_t> lock-free static_assert の追加を必須確認事項として指定*
3. *㉓ Deferred Publish stale discard を二重検査化（generation一致+publicationSequence比較）*
4. *㉔ isAllZero() を診断専用に変更（shutdown完了判定のauthorityから除外）*
5. *㉕ emitRetireTrace() を完全 noexcept 化*
6. *㉖ Overflow 時は Publication のみ抑制。Retire は常に継続することを明記*

#### F-8.4 第4ラウンド調査による新規確定事項（2026-06-08）

この節は、grep/Select-String/Serena MCP/CodeGraph MCP/ccc/semble/Graphify/ファイル精読を使用した第4ラウンド調査の結果として確定した事項を記載する。

##### F-8.4.1 `hasDeferredCommit` は既に実装されている

**★ 訂正（第6ラウンド調査で確定）**: `hasDeferredCommit` は現行 ConvoPeq ソースコードに**既に実装されている**。

```cpp
// AudioEngine.Threading.cpp:25 — 既存実装
const bool hasDeferredCommit = (runtimeOrchestrator_ != nullptr
                             && runtimeOrchestrator_->hasDeferredRequest());

// AudioEngine.Threading.cpp:35 — isFullyDrained() で使用
return !hasDeferredCommit && runtimePublicationBridge_.isFullyDrained();
```

`hasDeferredCommit` はメンバ変数ではなく `isFullyDrained()` 内のローカル変数として存在する。`runtimeOrchestrator_->hasDeferredRequest()` を呼び出し、その結果を `!hasDeferredCommit` として `isFullyDrained()` の条件に使用している。また `setPendingIntentCount()` と `setPublicationBacklogCount()` にも反映される。

**計画書との整合**: ✅ 既存の実装は計画書の A-2.6 と完全に一致する。`isFullyDrained()` のロジック変更は不要。A-2.6 では `collectDrainAudit()` による監査ログ追加のみ行えばよい。

##### F-8.4.2 `DSPHandleRuntime::crossfadeRecords_` と `CrossfadeAuthorityRuntime::records_` は別管理

**結論**: `DSPHandleRuntime` と `CrossfadeAuthorityRuntime` は**独立した** crossfade 記録ベクタを持つ：

```cpp
// ISRDSPHandle.h:148 — DSPHandleRuntime の crossfadeRecords_
std::vector<CrossfadeRecord> crossfadeRecords_;

// ISRDSPHandle.h:177 — CrossfadeAuthorityRuntime の records_
std::vector<CrossfadeRecord> records_;
```

**影響**: 計画書 3.5 の `destroyQuarantineSlot()` 内で `hasCrossfadeInvolving(slot)` を呼ぶ設計は、`CrossfadeAuthorityRuntime::hasCrossfadeInvolving(DSPHandle)` を `DSPHandleRuntime` から呼び出せない。`destroyQuarantineSlot()` は `DSPHandleRuntime` のメソッドであり、`CrossfadeAuthorityRuntime` への参照を持たない。

**修正案**: `DSPHandleRuntime` に独自の crossfade 関与チェックを追加：

```cpp
// ISRDSPHandle.h — DSPHandleRuntime に追加
// destroyQuarantineSlot 用: slot が crossfade に関与しているか確認
bool isSlotInCrossfade(uint32_t slot) const noexcept
{
    for (const auto& record : crossfadeRecords_) {
        if (record.active && (record.fromHandle.slot == slot || record.toHandle.slot == slot))
            return true;
    }
    return false;
}
```

`destroyQuarantineSlot()` 内のチェックを以下に修正：

```cpp
const bool inCrossfade = isSlotInCrossfade(slot);  // DSPHandleRuntime の crossfadeRecords_ を使用
```

##### F-8.4.3 `world->generation` のスロットエンコーディング

**結論**: `AudioEngine.Commit.cpp` の `onRuntimeRetiredNonRt()` では `world->generation` からスロット番号を抽出している：

```cpp
// AudioEngine.Commit.cpp:409-410
const std::uint32_t slot = static_cast<std::uint32_t>(world->generation % 256u);
std::uint32_t generation = static_cast<std::uint32_t>(world->generation & 0xFFFFFFFFu);
```

`world->generation` のフォーマット：

- Bits 0-7: slot number (0-255)
- Bits 8+: 実質的な generation カウンタ

**影響**:

1. Phase C-1 で truncation を削除すると、`DSPHandle::generation` が full 64-bit の composite value になる
2. 現状でも `RetireIntent::generation`（truncated）と `DSPRegistrySlot::generation`（create() のインクリメント）は異なる空間
3. Phase C-1 では両者の generation 形式を一致させる必要がある

**必須チェック項目**:

```cpp
// resolve() で比較される generation が同じ空間であることを確認
// create() → registry_[slot].generation = gen;  // 単純インクリメントカウンタ
// onRuntimeRetiredNonRt() → intent.generation = world->generation & 0xFFFFFFFFu;  // composite
// resolve() → currentGen (from registry_) vs handle.generation
// ↑ これらが異なる空間の場合、resolve() が常に stale を返す
```

##### F-8.4.4 現状実装の完全確認

| 項目 | 現状 | 計画との整合 |
| :--- | :--- | :--- |
| `RuntimePublicationOrchestrator::deferredRequest_` | `std::optional<PublishRequest>` + `bool hasDeferred_` | ✅ C-2 で DeferredPublishSlot へ置換予定 |
| `notifyTransitionComplete()` shutdown ガード | **存在しない** | ✅ A-2.2 で追加予定 |
| `emitShutdownTrace()` switch | `default:` なし。全7列挙子明示 | ✅ Phase D 対応で問題なし |
| `DSPHandleRuntime::quarantineSlot()` | **存在しない**（F-6 で追加予定） | ✅ A-1 で追加予定 |
| `DSPHandleRuntime::isSlotInCrossfade()` | **存在しない**（F-8.4.2 で追加が必要） | ⚠️ 新規。destroyQuarantineSlot 実装前に追加必要 |

##### F-8.4.5 変更履歴

1. *㉗ `hasDeferredCommit` の実装を確認（AudioEngine.Threading.cpp:25 ローカル変数）。計画書の記述と一致 ✅*
2. *㉘ crossfade 記録の二重管理を確認。destroyQuarantineSlot 内で isSlotInCrossfade() を使用する設計に修正*
3. *㉙ world->generation のスロットエンコーディングを確認。Phase C-1 必須チェックとして generation 形式一致確認を追加*
4. *㉚ deferredRequest_ の現状実装確認（計画書記述と一致）*
5. *㉛ emitShutdownTrace() switch 網羅性確認（Phase D 対応で問題なし）*
6. *㉜ `DSPHandleRuntime::isSlotInCrossfade()` の追加を A-1 の前提タスクとして明記*
