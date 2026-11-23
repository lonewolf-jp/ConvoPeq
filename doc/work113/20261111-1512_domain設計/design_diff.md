# domain tag 設計差分 v2.9（実装前・Owner レビュー用）

Owner レビュー 4 回目（2026-11-11、v2.7 に対する指摘 1 点）を反映した版である。設計（B-1）は変えない。
v2.7 のプローブは `jassertfalse` を無条件の `__debugbreak()` として写しており、JUCE の実際の定義
（`JUCE_LOG_CURRENT_ASSERTION` の後に `juce_isRunningUnderDebugger()` が true のときだけブレークする）と
異なっていた。v2.8 で **JUCE 8.0.12 の実物ヘッダを include して実マクロを使う**形に直し、
v2.9 で **5 条件の行列**（ビルド構成 × デバッガ × ガード）で「ブレークを決めるのはビルド構成ではなく JUCE のガード」であることを実測した。
デバッガ設定の根拠には Microsoft Learn の `sx` コマンド資料を引用し、実測と文献の区別を明記した。**製品コードは一切変更していない**（HEAD `c94d3767` に未コミットの Gate1／段階①／段階③が載った状態のまま）。

## §0 版管理と結論

### 0.1 版履歴

| 版 | 日時（JST） | 内容 |
|---|---|---|
| v1 | 2026-11-11 15:12 | 初版（棚卸し・データ構造・伝播・集計・テスト） |
| v2 | 2026-11-11 15:40 | Owner 追加条件 8 点を反映（`Release` の意味、`Unknown` の運用、JSON 不変条件の表化、統計量の定義、`DeletionEntry` の検証順序、`PublishReason` の責任分担） |
| v2.1 | 2026-11-11 15:59 | §12「照合用ソース引用（20 件）」を追加 |
| v2.2 | 2026-11-11 18:08 | Owner 指摘 6 点＋版管理 2 点を修正 |
| v2.3 | 2026-11-11 18:40 | `RetireDomainId` の配置・`Unknown`/`Unstamped` の区別・生成箇所の明示化 |
| v2.4 | 2026-11-11 19:20 | 生成箇所 35 件への訂正・`providerAbsent` の再定義・JSON 例の不変条件整合・版/実施状況の更新 |
| v2.5 | 2026-11-11 19:55 | Owner レビュー 2 点（§10 の既定値説明の訂正・`EpochDomain` の値域制約）を反映 |
| v2.6 | 2026-11-11 21:30 | `EpochDomain` ctor の Release 契約を B-1（全ビルドで不正値を採用しない＋観測カウンタ）に確定・§9 の v2.4 表記を修正 |
| v2.7 | 2026-11-11 22:05 | §7.2 #15 を 15-a（共通変換ロジック）／15-b（Release の実 ctor）／15-c（Debug の実 ctor と加算順序）に分割・証跡の区別を明記・MANIFEST のサイズとソース取得経路を修正 |
| v2.8 | 2026-11-11 23:10 | 15-c の方法を JUCE の実マクロと実条件（`juce_isRunningUnderDebugger()`）に合わせて修正。デバッガ非接続と接続の 2 条件で実測 |
| **v2.9** | **2026-11-11 23:50** | **本版。5 条件の行列（Debug／Release × デバッガ有無 × ガード強制）を追加し、ブレークの可否が JUCE のガードで決まることを実測。`sxd` の根拠を Microsoft Learn で補い、実測と文献を区別** |

### 0.2 v2.9 の修正対応表（Owner レビュー 2026-11-11・4 回目と、その追加検証）

| # | 指摘・追加確認 | 対応 | 反映先 |
|---|---|---|---|
| 1 | v2.7 の `assert_probe.cpp` は `jassertfalse` を無条件の `__debugbreak()` として写しており、JUCE の実定義（`juce_isRunningUnderDebugger()` が true のときだけブレーク）と異なる。15-c の手順もその条件に合わせる必要がある | JUCE 実物ヘッダの実マクロに置換。15-c を 15-c-1（非接続）／15-c-2（接続）に分割 | **§3.2**・**§7.2** |
| 2 | 追加確認（v2.8 の実測を補強）: ブレークの可否が「ビルド構成」ではなく「JUCE のガード」で決まることの直接確認と、デバッガ設定の根拠 | 5 条件の行列を実測（A〜E）。`sxd` の意味を Microsoft Learn で確認し、実測と文献を区別して記載 | **§3.2**（条件行列）・**§7.2**（15-c-2 の注記） |

設計（B-1）と 15-a／15-b／15-c-1／15-c-2 の分担は v2.8 のままで、変更しない。v2.9 は**証跡の補強**である。v2.7 のプローブは「SEH で捕捉できること」の部分実証として残し、15-c の根拠には使わない。

### 0.3 結論（方針は不変）

1. 退役ドメインは「**退役キューを所有する epoch ドメイン**」。実測の棚卸しでは **EngineISR** と **EQ** の 2 つだけ（§1）。
2. 帰属は **enqueue 時に provider が刻印**（`EpochDomain` が自ドメイン ID を持つ）。reclaim 時にスレッド・型・経路から推測しない。
3. `DeletionEntryType`（Generic/World）とは独立した `RetireDomainId` を新設し、`Unknown` を必ず用意する。
4. 集計は `retireLatencyClassified.domains[]` に追加し、`Σ domains == totals` をテストで固定する。`retireLatencyD1` は不変。
5. publish 理由は案2（引数化・デフォルト引数なし）。`Reconfigure` は識別不能のため enum に含めない。

## §1 棚卸し（実測・2026-11-11 18:08 JST）

### 1.1 退役キューを所有するドメイン

| 実体 | 宣言 | 退役キュー | 受付 |
|---|---|---|---|
| `AudioEngine::m_epochDomain` | `src/audioengine/AudioEngine.h:5455` | あり | `m_retireRouter`（`AudioEngine.CtorDtor.cpp:62`） |
| `EQProcessor::m_epochDomain` | `src/eqprocessor/EQProcessor.h:612` | あり | `m_ownedRetireRouter`（`EQProcessor.h:618`） |
| `ConvolverProcessor::m_epochDomain` | `src/ConvolverProcessor.h:1468` | 型としては持つが**未使用** | なし（`runtimeRcuReader` の reader 専用。`ConvolverProcessor.h:1470`） |

- `ConvolverProcessor` の退役は `getRcuProvider()`（＝`AudioEngine`）経由（`src/convolver/ConvolverProcessor.Lifecycle.cpp:59, 72`）→ **EngineISR** に帰属。
- `EpochControl` の slot retire（`src/audioengine/ISRRetire.h:125` / `ISRRetireRuntimeEx.cpp:184`）は `RetireLatencyRecorder` を**使わない**（grep 0 件）→ 本設計の対象外（§1.5）。

### 1.2 producer の棚卸し（グループ名とサイト数を分離）

**定義**: producer サイト = 退役権限に所有権を渡す**呼出し式**（`enqueueWithRetry` / `enqueueRetire` / `coordinator.enqueueRetire`）。
実装内部の再帰呼出し・ヘルパ内部の呼出しは数えない。１式が複数ポインタを渡す場合も 1 と数える。

#### A. 権限への直接呼出し（ドメインが決まる点）—— **13 サイト**

| # | グループ | 位置 | 呼出し | 期待ドメイン |
|---|---|---|---|---|
| A1 | EQ | `EQProcessor.Core.cpp:60` | `m_retireCoordinator->enqueueRetire(Granted, m_ownedRetireRouter, …)` | **EQ** |
| A2 | EQ | `EQProcessor.Core.cpp:73` | `m_ownedRetireRouter.enqueueWithRetry(…)`（QueueFull 時） | **EQ** |
| A3 | Engine | `AudioEngine.h:4716` | `m_retireRouter->enqueueWithRetry(…)`（`enqueueDeferredDeleteNonRtWithResult` の内部） | **EngineISR** |
| A4 | Engine | `AudioEngine.Retire.cpp:32` | `m_retireRouter->enqueueRetire(…)` | **EngineISR** |
| A5 | Engine | `DSPLifetimeManager.cpp:58` | `router_->enqueueWithRetry(…)` | **EngineISR** |
| A6 | Engine | `DSPLifetimeManager.cpp:121` | 同上 | **EngineISR** |
| A7 | Engine | `DSPLifetimeManager.cpp:155` | 同上 | **EngineISR** |
| A8 | Engine | `SnapshotCoordinator.h:95` | `enqueueWithRetry(*m_epochProvider, …)` | **EngineISR** |
| A9 | Engine | `SnapshotCoordinator.h:108` | 同上 | **EngineISR** |
| A10 | Engine | `SnapshotCoordinator.h:192` | 同上 | **EngineISR** |
| A11 | Engine | `SnapshotCoordinator.h:200` | 同上 | **EngineISR** |
| A12 | Engine | `SnapshotCoordinator.cpp:85` | 同上 | **EngineISR** |
| A13 | Engine | `SnapshotCoordinator.cpp:169` | 同上 | **EngineISR** |

- 内訳: **EQ 2 ・ Engine 11**。
- `RuntimeIntentCoordinator::enqueueRetire`（`ISRRuntimePublicationCoordinator.cpp:168`）は `router.enqueueWithRetry` への**転送実装**であり、
  ドメインは**引数で渡された router**で決まる。したがって独立の producer として数えない（A1 の実装体）。
- `SnapshotCoordinator.h:166,171` はヘルパ `enqueueWithRetry` 内部の再試行呼出しであり、ヘルパの呼出し元（A8–A13）で数える。

#### B. 上流の producer（ドメインを決めない＝A3 に集約）—— **8 サイト**

| グループ | サイト数 | 位置 | 経由先 |
|---|---|---|---|
| Convolver（IRState / StereoConvolver） | 2 | `ConvolverProcessor.Lifecycle.cpp:59, 72` | `AudioEngine::enqueueDeferredDeleteNonRt` → A3 |
| Cache（CacheMap） | 2 | `AudioEngine.Cache.cpp:16, 40` | 同上 → A3 |
| World（release） | 1 | `AudioEngine.Processing.ReleaseResources.cpp:719` | 同上 → A3 |
| DSP handle（Orchestrator） | 3 | `RuntimePublicationOrchestrator.cpp:200, 273, 853` | `DSPLifetimeManager` → A5–A7 |

#### C. 対象外（死活の確認済み）

| 機構 | 位置 | 理由 |
|---|---|---|
| slot retire（`ISRRetire`） | `AudioEngine.Commit.cpp:624` `worldAuthority_.lifetime().enqueueRetire(pendingSlot)` | 引数が `slot`。`EpochControl::enqueueRetire`（`ISRRetireRuntimeEx.cpp:184`）であり、`RetireLatencyRecorder` を使わない |
| レガシー `DeletionQueue` | `src/core/DeletionQueue.cpp:19, 44` | 到達しない（§1.4） |

- 合計: **A 13 + B 8 = 21 サイト**。このうちドメインを決めるのは A のみ。
- 以前の「producer 17 サイト」は、A と B を混ぜ、またグループとサイトを同じに数えていたため誤り。後の版（v2.2）で定義を明記して修正した。

### 1.3 完了経路（`recordComplete` の**全 11 サイト**）

現行ソースの `RetireLatencyRecorder::instance().recordComplete(` を全件列挙した（テストを除く）。

| # | 位置 | 関数 | Path | ドメインの取得方法（設計） | 状態 |
|---|---|---|---|---|---|
| 1 | `DeferredDeletionQueue.h:169` | `reclaim()` | `Queue` | `entry.domainId` | 稼働 |
| 2 | `DeferredDeletionQueue.h:224` | `drainAllUnsafe()` | `DrainAll` | `entry.domainId` | 稼働（shutdown） |
| 3 | `ISRRetireRouter.cpp:211` | `TerminalReclaimAuthority::drain` | `Drain` | `router.retireDomainId()` | 稼働 |
| 4 | `ISRRetireRouter.cpp:249` | `TerminalReclaimAuthority::drainAll`（retryable=Destroyed） | `DrainAll` | 同上 | 稼働（shutdown） |
| 5 | `ISRRetireRouter.cpp:275` | `TerminalReclaimAuthority::drainAll`（deleter） | `DrainAll` | 同上 | 稼働（shutdown） |
| 6 | `ISRRetireRouter.cpp:767` | Router 同期破棄（deleter） | `Sync` | 同上 | 稼働 |
| 7 | `ISRRetireRouter.cpp:813` | Router 同期破棄（retryable） | `Sync` | 同上 | 稼働 |
| 8 | `RetireQuarantineStore.h:139` | `drain()` | `Quarantine` | 引数で受け取る（router が渡す） | 稼働 |
| 9 | `RetireQuarantineStore.h:179` | `drainAllUnsafe()` | `DrainAll` | 同上 | 稼働（shutdown） |
| 10 | `DeletionQueue.cpp:19` | legacy `enqueue` オーバーフロー | `QueueOverflow` | — | **未使用（§1.4）** |
| 11 | `DeletionQueue.cpp:44` | legacy `reclaim()` | `Queue` | — | **未使用（§1.4）** |

- **重要**: Terminal / Quarantine のエントリは `DeletionEntry` を持たない（`TerminalReclaimAuthority` の内部エントリ・`RetireQuarantineStore` の配列）。したがって `entry.domainId` は使えず、**router のドメインを引数で渡す**（§3.3）。v2.1 の §1.3 はこの区別を落としていた（v2.2 で修正）。
- 稼働サイトは **9**（Queue 1 / DrainAll 4 / Drain 1 / Sync 2 / Quarantine 1）。

### 1.4 レガシー `src/core/DeletionQueue.{h,cpp}` の扱い（対象外）

- `DeletionQueue` は `src/core/SnapshotRetireManager.h:49` の `DeletionQueue m_queue;` からのみ使われる。
- **`SnapshotRetireManager` のインスタンスは現行ツリーに存在しない**（`SnapshotRetireManager` の出現は同ヘッダ内の定義・特殊メンバ宣言のみ。`.cpp` なし・生成箇所なし）。
- したがって `DeletionQueue.cpp` の `recordComplete`（#10/#11）は**現時点で到達しない**。
- 設計判断: **現行ツリーでは未使用のレガシー経路として対象外**とする。将来 `SnapshotRetireManager` を有効化する場合は、**その時点でドメイン帰属（provider 刻印 or 明示 `Unknown`）を設計してから計測対象に加える**。無条件に計測へ混入させない。
- 実装時の注意: `DeletionQueue` を削除しない（範囲外）。コメントで「対象外」と明記するに留める。

### 1.5 対象外（明示）

| 機構 | 理由 |
|---|---|
| `EpochControl`（`ISRRetire.h:125` の slot retire） | `RetireLatencyRecorder` を使わない（grep 0 件） |
| `src/core/DeletionQueue.cpp`（レガシー） | 到達しない（§1.4） |
| `WorldLifetimeRecorder`（world 寿命） | 別系統のレコーダ。`retireLatencyClassified` の対象外 |

## §2 データ構造

### 2.1 `DeletionEntry`（条件付きフィールドを明示）

```cpp
struct DeletionEntry
{
    void* ptr = nullptr;                                    // 8
    void (*deleter)(void*) = nullptr;                        // 8
    uint64_t epoch = 0;                                      // 8
    DeletionEntryType type = DeletionEntryType::Generic;      // 1
    RetireDomainId domainId{RetireDomainId::Unstamped};        // 1  ★追加（既定 = 未刻印）
    uint64_t publicationSequenceId{0};                        // 8
    uint64_t generation{0};                                   // 8
    uint64_t firstScanTsNs{0};                                // 8  （段階③）
#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS
    size_t objectBytes{0};                                   // 8  ★このフィールドは条件付き
#endif
};
```

- `objectBytes` は **`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` 有効時のみ**存在する。したがって
  **サイズ・アラインメントの比較は「同一ビルド設定同士」で行う**。
- 実装時の比較条件:

| 構成 | 比較する値 |
|---|---|
| 診断 **有効** | 変更前 `sizeof`/`alignof` と変更後（`domainId` 追加後） |
| 診断 **無効** | 同上（`objectBytes` が無い構成でも別途比較） |

- 実装順序（パディング前提にしない）: ① `domainId` の underlying type を `std::uint8_t` に固定 → ② 変更前の `sizeof`/`alignof` を実測して記録 → ③ `domainId` を追加 → ④ 同一構成で再実測して比較 → ⑤ 変化した場合は影響（キュー 4096 要素のメモリ寄与・コピー量・`constexpr` 依存）を評価してから判断。
- `static_assert` で固定する項目: `sizeof(RetireDomainId) == 1`、`std::is_trivially_copyable_v<DeletionEntry>`、`std::is_trivially_destructible_v<DeletionEntry>`、および上記で確定した `sizeof`/`alignof`（構成別に `#if` で分ける）。

### 2.2 `RetireDomainId`（v2.3: `Unstamped` を分離）

```cpp
enum class RetireDomainId : std::uint8_t
{
    Unstamped = 0,   // 刻印されていない（enqueue 時の刻印欠落）—— 設計上 0 が期待値
    Unknown   = 1,   // 刻印済みだがドメインを特定できない（意図的）
    EngineISR = 2,   // AudioEngine::m_epochDomain / m_retireRouter
    EQ        = 3,   // EQProcessor::m_epochDomain / m_ownedRetireRouter
    Count     = 4
};
```

| 値 | 意味 | 誰が付けるか |
|---|---|---|
| `Unstamped` | 刻印が欠落している（バグ） | `DeletionEntry` の既定値のみ。**生成側からは絶対に付けない** |
| `Unknown` | 刻印済みだが帰属を特定できない（意図的な未特定） | Convolver の reader 専用 `EpochDomain`、テスト用 `EpochDomain`、provider が未対応の場合 |
| `EngineISR` / `EQ` | 特定の退役ドメイン | Engine / EQ の `EpochDomain` |

- 以前の誤り: v2.2 では `Unknown` を未刻印の代表値にしていたため、**意図的な Unknown も `stampMissing` に計上**される矛盾があった。v2.3 で分離した。
- `DeletionEntryType`（Generic/World）とは用途が違う（寿命管理の種別 vs 退役ドメイン）。統合しない。
- **`Count` は sentinel（配列サイズ用）であり、ドメイン値としては使わない**。`EpochDomain` の ctor は §3.2 の `isAssignableDomain()` が false の値
  （`Unstamped` / `Count` / 5 以上）を**全ビルドで採用しない**（`Unknown` に落として計数する。Debug では加えて `jassertfalse` で停止）。
  許容は `Unknown` / `EngineISR` / `EQ` の 3 値のみである。

### 2.3 `RetireDomainId` の定義位置と include 依存（v2.3 で新設）

- 新設ヘッダ `src/core/RetireDomain.h`。**include は `<cstdint>` のみ**で、`convo` の他ヘッダを一切引かない。
- 依存方向（一方通）:

```text
src/core/RetireDomain.h          (enum RetireDomainId)
        ^                     ^
        |                     |
src/DeferredDeletionQueue.h   src/core/IRetireProvider.h
        ^                     ^
        |                     |
   (entry.domainId)      (retireDomainId() の戻り型)
```

- 現行の include 関係（実測）: `IRetireProvider.h:5` が `#include "../DeferredDeletionQueue.h"`（`DeletionEntryType` のため）。
  したがって `RetireDomainId` を `IRetireProvider.h` 側に定義すると `DeferredDeletionQueue.h` → `IRetireProvider.h` の逆依存が必要になり**循環する**。
  独立ヘッダに置くことでこの循環が生じない。
- `DeferredDeletionQueue.h` は既に `audioengine/ISRRetireLatency.h` を include しているため、新たな依存は 1 本（`RetireDomain.h`）増えるだけである。
- 代替案（`DeferredDeletionQueue.h` 側に定義）は採用しない: ドライバリ（診断・レコーダ）がキューヘッダを引き込む必要が生じる。

## §3 伝播

### 3.1 インターフェース（`provider()` は使わない）

現行の `ISRRetireRouter` は `public convo::IEpochProvider, public convo::IRetireRouter` を継承し、`IEpochProvider` は
`IReaderEpochProvider` / `IPublicationProvider` / `IRetireProvider` を継承する（`src/audioengine/ISRRetireRouter.h:205`、`src/core/IEpochProvider.h:42`）。
実 provider は **private メンバ `IEpochProvider* provider_`**（`ISRRetireRouter.h:474`、ctor で設定: `ISRRetireRouter.cpp:330-333`）であり、**公開 `provider()` アクセサは存在しない**。

そこで **`IRetireProvider` に `retireDomainId()` を追加し、`EpochDomain` と `ISRRetireRouter` の両方が override する**。

```cpp
// src/core/IRetireProvider.h（既定は Unknown — 刻印漏れを可視化する）
virtual RetireDomainId retireDomainId() const noexcept { return RetireDomainId::Unknown; }

// EpochDomain（自分のドメインを返す）
RetireDomainId retireDomainId() const noexcept override { return domainId_; }

// ISRRetireRouter（provider へ委譲。provider() を公開しない）
[[nodiscard]] RetireDomainId retireDomainId() const noexcept override
{
    return provider_ != nullptr ? provider_->retireDomainId() : RetireDomainId::Unknown;
}
```

- Router が override しないと、`IEpochProvider&` 経由の問い合わせが既定の `Unknown` に落ちる（Owner 指摘 1 のとおり）。**必ず override する**。
- `provider_` は ctor でのみ設定され、以後不変（`provider_ =` の出現は ctor の 1 回のみ）。したがって委譲結果は安定する。
- provider の内部ポインタは公開しない。

### 3.2 enqueue 時の刻印と `EpochDomain` の生成箇所（v2.3）

```cpp
// 生成側に必ず意図を書かせる（デフォルト引数なし）
explicit EpochDomain(RetireDomainId domain) noexcept
    : domainId_(sanitizeDomain(domain)), globalEpoch(1)
{
    if (!isAssignableDomain(domain))
    {
        // 不正値は全ビルドで採用しない。Debug では更に停止して検出する。
        invalidDomainAssignments_.fetch_add(1, std::memory_order_relaxed);
        jassertfalse;
    }
}

// 契約の判定子（constexpr・全ビルド共通）
static constexpr bool isAssignableDomain(RetireDomainId domain) noexcept
{
    return domain == RetireDomainId::Unknown
        || domain == RetireDomainId::EngineISR
        || domain == RetireDomainId::EQ;
}

// 不正値は決して採用しない（全ビルド）。採用値は Unknown（= providerUnknown に計上）
static constexpr RetireDomainId sanitizeDomain(RetireDomainId domain) noexcept
{
    return isAssignableDomain(domain) ? domain : RetireDomainId::Unknown;
}

RetireDomainId domainId() const noexcept { return domainId_; }
static std::uint32_t invalidDomainAssignments() noexcept;   // 両ビルドで観測できる

// DeferredDeletionQueue::enqueue は domain を受け取り entry に書く（以降不変）
entry.domainId = domainId;
```

#### ctor の契約（v2.6 で確定）

- **全ビルド（Release を含む）**: ctor は `isAssignableDomain()` が false の値を**採用しない**。`domainId()` は必ず `Unknown` を返すため、`EngineISR` / `EQ` への誤帰属は起こらない。発生回数は `invalidDomainAssignments()`（`std::atomic<std::uint32_t>`）が数える。
- **Debug の追加検出**: `jassertfalse` によりその場で停止する。`jassert` は Release で無効なので、**契約の本体は `sanitizeDomain()` とカウンタ**であり、assertion は補助である。v2.5 までの「`jassert` だけで拒否する」記述は、Release では何も起きない点で不正確だった。
- **`std::abort()` を採らない理由**: ctor は `AudioEngine` / `EQProcessor` の初期化時に走る。Release で `abort` すると起動不能になり、原因の特定が難しくなる。加えて in-process のテストでは観測できず、死亡テストを別途用意する必要がある。不正値を受け入れず、観測可能にする方が実運用に合う。
- 不正値が入り得るのは `static_cast<RetireDomainId>` を伴うコードだけであり、本番 3 箇所はいずれも許可値のリテラルを渡す（下の棚卸し）。したがってカウンタは本番で 0 のままである。
- カウンタは診断用で、JSON へは出さない（§8 の非目標）。必要になった時点で別途追加する。
- メンバは `domainId_` を `globalEpoch` より先に宣言する（初期化子の順序警告を出さない）。

#### 証跡の区別（v2.8）

| 証跡 | 対象 | 何を言えるか |
|---|---|---|
| v2.6 の `contract_check.cpp` | 共通変換ロジックのみ（`jassertfalse` を含まない） | `sanitizeDomain()` のフォールバック・カウンタ加算・許可値の採用。**Debug の実 ctor は検証していない**（#15-a 相当） |
| v2.7 の `assert_probe.cpp` | 無条件 `__debugbreak()` を書いた再現コード | `__debugbreak()` が `EXCEPTION_BREAKPOINT` を出し、`__try`／`__except` と「捕捉後のカウンタ読み」が成立すること。**JUCE の条件付きブレークの実証ではない**（部分実証としてのみ有効） |
| v2.8 の `assert_probe.cpp` | **JUCE 8.0.12 の実物ヘッダ**（`JUCE/modules/juce_core/system/juce_PlatformDefs.h`）の `jassertfalse` | `JUCE_LOG_CURRENT_ASSERTION` が走ること、`juce_isRunningUnderDebugger()` が false ならブレークしないこと、true ならブレークして `EXCEPTION_BREAKPOINT` になること、いずれもカウンタが +1 であること |
| v2.9 の 5 条件の行列（同じ `assert_probe.cpp`） | JUCE 実マクロの条件依存性 | ビルド構成を変えずにガードだけ true にするとブレークが起きること（C）、Release ではガードが true でもブレークしないこと（E） |
| 実装後の #15-b／#15-c | 製品の `EpochDomain` | Release のフォールバックと、Debug の assertion 発火＋加算順序 |

v2.8 のプローブが JUCE から借りていないのは 2 つの関数の実装だけである。`juce_isRunningUnderDebugger()` は JUCE の Windows 実装（`juce_Threads_windows.cpp:264`）と同じ `IsDebuggerPresent() != FALSE` を返し、`logAssertion()` は呼ばれた事実を記録する。マクロ本体と分岐条件は JUCE の実物をそのまま使う。
したがって製品 ctor に対する #15-c も、実装後に**同じ 2 条件**（デバッガ非接続／接続）で実施する。

#### 条件行列（v2.9 の実測）

| # | ビルド | デバッガ | `juce_isRunningUnderDebugger()` | `jassertfalse` のブレーク | 結果 |
|---|---|---|---|---|---|
| A | Debug | なし | false | しない（ログのみ） | PASS |
| B | Debug | あり（最小デバッガ） | true | する（`EXCEPTION_BREAKPOINT` 0x80000003 を `__except` が捕捉） | PASS |
| C | Debug | なし（`DOMAIN_PROBE_FORCE_DEBUGGER=1` でガードだけ true） | true | する | PASS |
| D | Release | なし | false | しない（ログも出ない） | PASS |
| E | Release | あり（最小デバッガ） | true（ただし `JUCE_DEBUG=0` でブレーク分岐自体が無い） | しない（ログも出ない） | PASS |

- **A と C の差はビルド構成ではなく JUCE のガード**である（同じ Debug バイナリ系列で、ガードだけを true にするとブレークする）。v2.7 の「Debug なら無条件にブレーク」という前提は成り立たない。
- **E** は、非 Debug の `jassertfalse`（`:193`）にブレークが無いことの確認である。デバッガを接続してもブレークしない。
- いずれも `invalidDomainAssignments()` は +1 で、加算は assertion より前に行われている。

#### 生成箇所の全棚卸し（実測）

| 種別 | 件数 | 生成側 | 付ける値 |
|---|---|---|---|
| 本番 | 3 | `AudioEngine.h:5455` | `RetireDomainId::EngineISR` |
| 本番 | — | `EQProcessor.h:622`（`make_unique<convo::EpochDomain>`） | `RetireDomainId::EQ` |
| 本番 | — | `ConvolverProcessor.h:1468` | `RetireDomainId::Unknown`（reader 専用・退役キュー不使用であることを明示） |
| テスト | 35 | 下記の内訳 | `RetireDomainId::Unknown`（テストドメインは本番帰属を持たないため意図的に Unknown） |

テスト 35 件の内訳（実測、`make_unique` 形式を含む）:

| ファイル | 件数 | 形式 |
|---|---|---|
| `STG11EQRetireTests.cpp` | 2 | 変数宣言（:330, :331） |
| `V11ReaderReacquireTests.cpp` | 9 | 変数宣言 |
| `D8_2_B_2_Tests.cpp` | 4 | 変数宣言 |
| `invariant_INV3_INV5.cpp` | 3 | 変数宣言 |
| `ScopedEpochReaderTests.cpp` | 4 | 変数宣言 |
| `STG10ReaderQuarantineTests.cpp` | 4 | 変数宣言 |
| `StuckReaderFallbackDrainTests.cpp` | 6 | 変数宣言 |
| **`STG11D2SnapshotRetireTests.cpp`** | **3** | **`std::make_unique<convo::EpochDomain>()`（:181 / :254 / :313）** |
| 合計 | **35** | — |

- 上記 3 件は `checkTD21QFullEscalatesToEmergency` / `checkTD22EFullEscalatesToTerminal` / `checkTD23ShutdownDrainCompletes` の中で生成される。
  **変数宣言だけの grep では漏れる**ため、v2.4 では `make_unique` 形式も含めて数えた。

- テスト 32 箇所は `convo::EpochDomain x;` の形であるため、引数必須化により**コンパイルエラーで全件が漏れなく検出される**。
- `Unstamped` は**生成側から一切付けない**。付くのは `DeletionEntry` の既定値だけであり、それが見つかったら刻印漏れである。
- 検証（ソースレベル）: 実装後に `grep -rn "EpochDomain [a-z_]*;" src --include=*.h --include=*.cpp` が **0**、
  `grep -rn "deferredDeletionQueue\.enqueue" src` が `EpochDomain.h` 以外で **0** であることを確認する。

### 3.3 完了経路ごとのドメイン取得（§1.3 に対応）

| 経路 | 取得方法 | 変更点 |
|---|---|---|
| `Queue` / `DrainAll`（`DeferredDeletionQueue`） | `entry.domainId` | entry にフィールド追加＋enqueue で刻印 |
| `Sync`（`ISRRetireRouter.cpp:767, 813`） | `retireDomainId()` | Router 内なので `this->retireDomainId()` |
| `Drain`（`ISRRetireRouter.cpp:211`） | 同上 | `TerminalReclaimAuthority::drain(minReader, isOlderFn)` に **`RetireDomainId` を追加**し、router が渡す |
| `DrainAll`（`ISRRetireRouter.cpp:249, 275`） | 同上 | `TerminalReclaimAuthority::drainAll(RetireDomainId)` |
| `Quarantine`（`RetireQuarantineStore.h:139`） | 同上 | `drain(minReader, isOlderFn, RetireDomainId)` |
| `DrainAll`（`RetireQuarantineStore.h:179`） | 同上 | `drainAllUnsafe(RetireDomainId)` |

- 実シグネチャ（現行）: `TerminalReclaimAuthority::drain(uint64_t, const std::function<bool(uint64_t,uint64_t)>&)`（`ISRRetireRouter.cpp:107`）、`drainAll()`（`:223`）、`tryReclaim(uint64_t, const std::function<…>&)`（`:302`）、`RetireQuarantineStore::drain(uint64_t, const std::function<…>&)`（`RetireQuarantineStore.h:102`）、`drainAllUnsafe()`（`:155`）。
- 引数を追加するのは**完了記録のためだけ**であり、reclaim 条件・順序・所有権は変更しない。
- **`provider_ == nullptr` は現行 API では到達不能**である。`ISRRetireRouter` は `explicit ISRRetireRouter(IEpochProvider& provider, …)`（参照引数）で
  `provider_(&provider)` と初期化し、コピー/ムーブは禁止、`provider_` を null にする公開 API もない（§4.2）。
  したがって、**通常の原因は `providerUnknown` に一本化**し、`providerAbsent` は**防御的・到達不能のカウンタ**として位置づける（常に 0）。

### 3.4 Convolver: reader ドメインと退役ドメインは別（安全性の前提も明記）

- Convolver のプロセスは `convo::RCUReaderGuard guard(runtimeRcuReader);`（`src/convolver/ConvolverProcessor.Runtime.cpp:225`）で**Convolver 自身の reader ドメイン**を保護する。
- 退役（IRState / StereoConvolver）は `getRcuProvider()` 経由で **Engine の退役ドメイン**に入る。
- さらに、Convolver の `process()` は **Engine の audio callback の reader 区間内**で呼ばれる（`AudioEngine.Processing.BlockDouble.cpp:158` の `RCUReaderGuard rcuGuard(audioThreadRcuReader)` のスコープ内で `convolverRt().process(...)` を実行: `AudioEngine.Processing.DSPCoreDouble.cpp:383, 447`）。
- つまり「Engine reader が Convolver 処理区間も覆う」ため、Engine ドメインに置かれた退役は Convolver の処理中に解放されない。**帰属（EngineISR）と安全性（reader 区間）の両方をこの二重の保護で説明できる**。

## §4 `Unknown` の運用

### 4.1 不変条件

| 不変条件 | 内容 |
|---|---|
| 集計から除外しない | `domains[]` の要素として必ず出す |
| 振り替えない | `EngineISR` / `EQ` に加算しない |
| 検出可能 | `unknownRatio` と `unknownReasons` で監視 |
| 経路別に期待値を持つ | **全体のゼロ和で代用しない**（§7.1） |

### 4.2 原因の判定規則（v2.4: `providerAbsent` は防御的・到達不能）

| 完了経路 | 条件 | 記録されるドメイン | 加算するカウンタ |
|---|---|---|---|
| `Queue` / `DrainAll`（`DeferredDeletionQueue`） | `entry.domainId == Unstamped` | `Unstamped` | **`stampMissing`** |
| `Queue` / `DrainAll`（`DeferredDeletionQueue`） | `entry.domainId == Unknown` | `Unknown` | **`providerUnknown`** |
| `Sync` / `Drain` / `DrainAll` / `Quarantine`（router 経由） | provider が `Unknown` を返す | `Unknown` | **`providerUnknown`** |
| 同上 | `provider_ == nullptr`（**到達不能・防御的**） | `Unknown` | `providerAbsent`（常に 0） |
| 全経路 | `EngineISR` / `EQ` | その値 | なし |

- **`providerAbsent` は現行 Router API では発生しない**。`ISRRetireRouter` の ctor は `IEpochProvider&`（参照）で `provider_(&provider)`、
  コピー/ムーブ禁止、`provider_` を null にする公開 API なし。したがって通常の原因は `providerUnknown` に一本化する。
- `providerAbsent` を有効な計測対象にするには、**Router の provider を nullable にする API 設計変更**が別途必要になる（本案では行わない）。
- ※ `EpochDomain` が意図的に `Unknown` で生成された場合は、キュー経由でも `providerUnknown` に計上する。
  つまり **`Unknown` は常に `providerUnknown`、`Unstamped` は常に `stampMissing`** に対応する（意味の一意性）。

#### 不変式（テストで固定）

```text
domains[Unstamped].count == stampMissing
domains[Unknown].count   == providerUnknown + providerAbsent   // providerAbsent は通常 0
stampMissing == 0                      // 設計上の期待値（本番・テストとも）
providerAbsent == 0                    // 現行 API では到達不能（非ゼロなら API 変更が入ったサイン）
Σ unknowns == domains[Unstamped].count + domains[Unknown].count
```

### 4.3 JSON 上の配置（v2.4: 例示値を不変式に合わせる）

- `unknownRatio` / `stampMissingRatio` と `unknownReasons` は、**それぞれのドメイン行の中にのみ**出力する。
  - `domains[Unknown].unknownRatio`、`domains[Unknown].unknownReasons { providerUnknown, providerAbsent }`
  - `domains[Unstamped].stampMissingRatio`、`domains[Unstamped].unknownReasons { stampMissing }`
- 分母はどちらも `totals.completions`。分母 0 のときは `0.0` を出力し、`…Defined: false` を併記する。
- **例示は不変式 `count = classified + unclassified + excluded` を満たす値にする**（v2.3 の例は `count: 0` に対して `unclassified: 2, excluded: 3` で不成立だった）。

```json
{ "id": "Unknown",
  "count": 5, "classified": 0, "unclassified": 2, "excluded": 3, "consistencyErrors": 0,
  "unknownRatio": 0.0212, "unknownRatioDefined": true,
  "unknownReasons": { "providerUnknown": 5, "providerAbsent": 0 },
  "paths": [ … ] },
{ "id": "Unstamped",
  "count": 0, "classified": 0, "unclassified": 0, "excluded": 0, "consistencyErrors": 0,
  "stampMissingRatio": 0.0, "stampMissingRatioDefined": true,
  "unknownReasons": { "stampMissing": 0 },
  "paths": [ … ] }
```

- 上の例では `count(5) = 0 + 2 + 3` が成立し、`unknownReasons` の和（5）は `domains[Unknown].count`（5）と一致する。

## §5 JSON と不変条件

### 5.1 スキーマ（v2.3: 4 ドメイン行）

```json
"retireLatencyClassified": {
  "note": "diagnostic only; the verdict uses retireLatencyD1.runtime and is unchanged",
  "thresholdNs": 50000000,
  "required": 59,
  "paths":  [ … 既存（全ドメイン合算）… ],
  "totals": { … 既存（合算）… },
  "domains": [
    { "id": "EngineISR", "count": 0, "…": 0, "paths": [ … ] },
    { "id": "EQ",        "count": 0, "…": 0, "paths": [ … ] },
    { "id": "Unknown",   "count": 0, "unknownRatio": 0.0, "unknownRatioDefined": true,
      "unknownReasons": { "providerUnknown": 0, "providerAbsent": 0 }, "paths": [ … ] },   // providerAbsent は常に 0
    { "id": "Unstamped", "count": 0, "stampMissingRatio": 0.0, "stampMissingRatioDefined": true,
      "unknownReasons": { "stampMissing": 0 }, "paths": [ … ] }
  ],
  "countsConsistent": true,
  "domainSumsConsistent": true
}
```

- 4 行は**常に出力**する（件数 0 でも省略しない）。`paths` / `totals` は従来どおりの合算で、キーも値も変えない。

### 5.2 不変条件（Owner 提示の表＋追加）

| 不変条件 | 期待する関係 |
|---|---|
| ドメイン別件数 | `Σ domains[].count == totals.count`（`count = classified + unclassified + excluded`） |
| ドメイン別 classified | `Σ domains[].classified == totals.classified` |
| ドメイン別 unclassified | `Σ domains[].unclassified == totals.unclassified` |
| ドメイン別 excluded | `Σ domains[].excluded == totals.excluded` |
| ドメイン別 consistencyErrors | `Σ domains[].consistencyErrors == totals.consistencyErrors` |
| パス別の成分件数 | `Σ domains[].paths[p].<component>.count == paths[p].<component>.count`（3 成分） |
| 母数分離 | `classified + unclassified + excluded == completions` |
| 原因の分離 | `domains[Unstamped].count == stampMissing`、`domains[Unknown].count == providerUnknown + providerAbsent`（`providerAbsent` は通常 0） |
| 刻印漏れの不在 | `stampMissing == 0`（本番・テストとも。非ゼロなら実装バグ） |
| 防御カウンタの不在 | `providerAbsent == 0`（現行 API では到達不能。非ゼロなら Router の provider API が変わったサイン） |

※ 最後の式は「Unknown に落ちた観測は必ずどちらかの原因に計上される」ことを表す。成立しない場合は実装漏れ。

### 5.3 統計量の定義

| 量 | 定義 |
|---|---|
| `sumNs` / `count` / `over50ms` | 加算可。上記のゼロ和が成立する |
| `maxNs` | **加算不可**（ドメイン別最大はそのドメイン内の最大。`Σ domains[].maxNs ≠ totals.maxNs`）。不変条件に含めない |
| 平均 | 出力しない（必要なら `sumNs / count`、件数 0 は未定義） |
| パーセンタイル | **出力しない**（ドメイン別はサンプルが薄く信頼できない） |

## §6 publish 理由タグ（案2）

### 6.1 棚卸し（現行）

| 呼出し元 | 現状 | 案2 の理由 |
|---|---|---|
| `EQProcessor::prepareToPlay` | `setPublishReasonForNextFlush(Prepare)` + `flushPendingEpochAdvance()` | `Prepare` |
| `EQProcessor::releaseResources` | 同上（Release、唯一の flush） | `Release` |
| `EQProcessor::driveRetireReclaimNonRt`（タイマー） | タグを使わず `countPublishReason(TimerPending)` | `TimerPending`（flush を呼ばない） |
| その他 | なし | `Explicit`（将来の直呼び） |

### 6.2 変更内容

```cpp
enum class PublishReason : std::uint8_t { Prepare, Release, TimerPending, Explicit };
void flushPendingEpochAdvance(PublishReason reason) noexcept;   // デフォルト引数なし
```

- `m_publishReasonTag` と `setPublishReasonForNextFlush()` を削除（タグ書き込みと消費の分離をなくす）。
- `countPublishReason(reason)` は flush 成功時に 1 回。publish 条件・順序・reclaim は不変。
- `retireDriverStats()` の `publishOther` → `publishExplicit` へ改名（JSON には未出力のため外部互換に影響なし）。
- 段階①の契約テストは「タグ無し flush = other」を「`Explicit` を明示して flush」に置換。**別スレッドから setter/prepare/release を呼ぶ競合テストは維持**（理由が引数なので混ざらない）。

### 6.3 意味の明記

- **`Release` は「`releaseResources()` による呼出し」を意味し、終了処理（shutdown）を意味しない**。終了時も同じ関数が呼ばれるが区別できないため同じ値になる。
- **`TimerPending` の責任分担**: ドライバの直接計数は「タイマーが pending を消費して publish した回数」、`flushPendingEpochAdvance(reason)` の理由は「どの入口が flush を呼んだか」。ドライバは flush を呼ばないため二重計数しない（テストで固定）。

## §7 検証計画

### 7.1 経路別の期待ドメイン（サイト数つき・§1.2/1.3 と対応）

| 対象 | サイト数 | 期待ドメイン | 期待 `Unknown` |
|---|---|---|---|
| §1.2 A1–A2（EQ） | 2 | `EQ` | 0（`stampMissing` も 0） |
| §1.2 A3–A13（Engine） | 11 | `EngineISR` | 0 |
| §1.2 B（上流 4 グループ） | 8 | A3/A5–A7 経由で `EngineISR` | 0 |
| §1.3 #1–#2（queue: Queue / DrainAll） | 2 | エントリのドメイン | 0 |
| §1.3 #3–#7（router: Drain / DrainAll / Sync） | 5 | router の provider | 0 |
| §1.3 #8–#9（quarantine） | 2 | 同上 | 0 |
| テスト用スタブ（provider が `Unknown` を返す） | — | `Unknown` | 発生する（`providerUnknown`） |
| provider 不在（`provider_ == nullptr`） | — | `Unknown`（到達不能） | 常に 0（`providerAbsent` は防御的） |
| §1.3 #10–#11（レガシー） / §1.2 C | 2 + 1 | 対象外（観測されないこと） | — |

- 検証は「**A 13 サイトごと**」と「**完了記録 9 実動サイトごと**」の 2 軸で行う。合計のゼロ和で代用しない。

### 7.2 検証項目（15 項目）

> 数の対応: v2.2 の 10 項目 → v2.3 で #13（未刻印の不在）・#14（生成箇所の明示化）を追加し 12 項目 →
> v2.5 で #15（ctor の値域制約）を追加し **15 項目**。v2.7 で #15 を 15-a／15-b／15-c に分け、v2.8 で 15-c を 15-c-1（デバッガ非接続）／15-c-2（デバッガ接続）に分けた（項目数は 15 のまま）。
> Owner 承認範囲の「14 項目」は v2.4 時点の番号である。
> **#15 の 3 副検証がそろって初めて #15 の完了**であり、15-a だけでは #15 を満たさない。

| # | 検証 | 合格条件 |
|---|---|---|
| 1 | 刻印（サイト別） | A1–A13 の 13 サイトそれぞれが期待ドメインになる（全体のゼロ和で代用しない） |
| 2 | 推測禁止 | reclaim 前後で `entry.domainId` が不変、observation と一致 |
| 3 | Router API | `ISRRetireRouter::retireDomainId()` が provider の値を返す（provider null なら `Unknown`）。`provider()` を公開しない |
| 4 | 完了経路の網羅 | 9 稼働サイトすべてでドメインが付く。Terminal/Quarantine は引数経由で正しい値 |
| 5 | レガシー対象外 | `SnapshotRetireManager` 未使用であること（生成箇所なし）を確認し、`DeletionQueue` 由来の観測が出ない |
| 6 | 合算整合 | §5.2 の全式（ゼロ和・パス別成分・母数分離）が成立 |
| 7 | 原因の分離 | スタブ（provider が `Unknown` を返す）で `providerUnknown` >0・`stampMissing` 0・`providerAbsent` 0。`providerAbsent` は到達不能なので、検証は「0 であること」のみ |
| 8 | `Unknown` の非混入 | `Unknown` の値が `EngineISR`/`EQ` に加算されない（意図的に Unknown を作って確認） |
| 9 | 既存互換 | `retireLatencyD1` のキー構造比較＋`evaluateRetireDelay` の verdict 不変 |
| 10 | サイズ・型 | §2.1 の 5 項目を**診断有効・無効の両構成**で確認 |
| 11 | `PublishReason` | 4 値の付与が呼出し元と一致。ドライバ直接計数と二重計数しない |
| 12 | 実測 | 3 シナリオ再実行。eq の 11.9 s `driverWait` が `EngineISR` に帰属するかを確定 |
| 13 | 未刻印の不在 | `stampMissing == 0`、`domains[Unstamped].count == 0`（本番・テストとも） |
| 14 | 生成箇所の明示化 | `EpochDomain x;`（デフォルト生成）が `src` に 0 件、`deferredDeletionQueue.enqueue` が `EpochDomain.h` 以外で 0 件 |
| 15-a | 値域制約・共通変換ロジック（両構成） | `isAssignableDomain()` が `Unknown` / `EngineISR` / `EQ` で true、`Unstamped` / `Count` / `static_cast<RetireDomainId>(5)` / `static_cast<RetireDomainId>(255)` で false。`sanitizeDomain()` が不正値を `Unknown` に落とす。**ctor の実動作は対象にしない** |
| 15-b | Release の実 ctor | 製品の `EpochDomain` を Release で構成し、不正値（`Unstamped` / `Count` / 5 / 255）を渡すと `domainId() == Unknown` になり、`invalidDomainAssignments()` が 1 件ずつ増える。停止は起きない |
| 15-c-1 | Debug・**デバッガ非接続**の実 ctor | 製品の `EpochDomain` を Debug で構成し、不正値を渡すと `jassertfalse` は **`JUCE_LOG_CURRENT_ASSERTION` だけを行い、ブレークしない**（JUCE の実定義どおり）。`domainId() == Unknown`、`invalidDomainAssignments()` が +1、例外は発生しない |
| 15-c-2 | Debug・**デバッガ接続**の実 ctor（assertion と加算順序） | `juce_isRunningUnderDebugger()` が true になる条件（デバッガ接続）で同じ構成を行い、`jassertfalse` が `JUCE_BREAK_IN_DEBUGGER`（Windows では `__debugbreak()`）を実行して `EXCEPTION_BREAKPOINT`（0x80000003）を発生させること。デバッガが初回例外をアプリへ渡す設定（後述）で `__try`／`__except` が捕捉でき、**捕捉後の `invalidDomainAssignments()` が +1**（＝加算がブレークより前）であること |

> **15-c-2 の前提**: デバッガ非接続では JUCE はブレークしないため、`__try`／`__except` で捕捉しようとしても例外は出ない（15-c-1 の結果になる）。したがって 15-c-2 は必ずデバッガ接続で実行する。
> 手順は、① Debug ビルドのテストをデバッガから起動する（`IsDebuggerPresent()` が true になる）→ ② 初回例外 `0x80000003` を**アプリへ渡す**設定にする → ③ `__try { EpochDomain bad(static_cast<RetireDomainId>(5)); } __except (EXCEPTION_EXECUTE_HANDLER) { ... }` で捕捉し、`GetExceptionCode()` とカウンタを読む。
> デバッガ側の設定: WinDbg／cdb は `sxd 80000003` のうえ `g`（初回例外をアプリへ渡す）。Visual Studio は「例外設定」で Win32 Exceptions `0x80000003` の中断を外し、「Just My Code」を無効にする。既定のままだとデバッガが先に止まり、アプリの `__except` へ回らない。
> `sxd` の意味は Microsoft Learn「sx, sxd, sxe, sxi, sxn, sxr, sx- (Set Exceptions)」の **Second chance break (Disabled)** で、初回例外ではデバッガが止まらずアプリのハンドラへ渡る（`https://learn.microsoft.com/windows-hardware/drivers/debuggercmds/sx--sxd--sxe--sxi--sxn--sxr--sx---set-exceptions-`）。
> 本環境には cdb／WinDbg が無いため、**実測は同じ条件を再現する最小デバッガ（`mini_debugger.cpp`）で行い**、WinDbg／Visual Studio の設定は文献に基づく記載である（未実測）。
> 本設計の証跡では、同じ条件を再現する最小デバッガ（`mini_debugger.cpp`、`DEBUG_ONLY_THIS_PROCESS` で起動し、ローダの初期ブレークは `DBG_CONTINUE`、2 回目以降は `DBG_EXCEPTION_NOT_HANDLED` でアプリへ渡す）で 15-c-2 を実測した。
> 本番 3 箇所の生成値が許可値で `invalidDomainAssignments() == 0` であることは 15-b の一部として確認する。

## §8 非目標・不変条件

- 遅延そのものは改善しない（帰属を可能にするだけ）。
- 判定系（`evaluateRetireDelay`・50 ms・`N_req=59`・verdict）は変更しない。
- 観測できない情報を推測で補完しない（`Unknown` を残す）。
- `invalidDomainAssignments()` はテストと診断のためのカウンタで、JSON へは出さない（v2.6）。
- `retireLatencyD1` と `worldLifetime` は不変。
- レガシー `DeletionQueue` の削除・有効化は行わない。
- commit／merge／release は含まない。

## §9 実施順序

| 順序 | 作業 | 状態 |
|---|---|---|
| 1 | 本設計差分 v2.9 の確定 | 提出済み・レビュー待ち（v2.7 の指摘 1 点＋条件行列の実測を反映済み） |
| 2 | domain tag 実装＋§7.2 の **15 項目** | **実装は承認済み（2026-11-11）。15-c の方法を Owner が確認した後に着手** |
| 3 | publish 理由タグ案 2（§6） | **実装完了（2026-11-11：ctest 60/60・アプリ 3 シナリオ実測済み）** |
| 4 | Reader レジストリ統合 | 未承認 |
| 5 | ASan | 未実施 |
| 6 | commit／merge／release | 未承認 |

## §10 リスクと代替案

| リスク | 対策 |
|---|---|
| 刻印漏れ | 既定は `Unstamped`（未刻印の印）なので、刻印されないままキューに入った entry を見逃さずに検出できる（`stampMissing` > 0 で検出）。一方、意図的な `Unknown` は `providerUnknown` として別に計上され、誤帰属はしない |
| Router の override 忘れ | コンパイルで防げないため、テスト #3 で `retireDomainId()` の委譲を直接検証 |
| Terminal/Quarantine への引数追加漏れ | 既定値を**設けない**（`RetireDomainId` を必須引数に）→ 呼出し漏れをコンパイルで検出 |
| ドメイン追加 | `RetireDomainId` に値を追加し JSON の `domains[].id` を追加するだけ（既存キーは不変） |
| ctor に不正値が渡る（Release） | 全ビルドで `sanitizeDomain()` により `Unknown` へ落とし、その値を採用しない。`invalidDomainAssignments()` で発生を観測でき、Debug では `jassertfalse` で停止する。不正値が `EngineISR` / `EQ` へ誤帰属することはない |
| 代替案（router が ID を持ち provider へ引数で渡す） | 全 provider 実装・13 サイトに引数追加が必要で誤りの余地が増える。provider 刻印＋router 委譲を優先 |

## §11 成果物と取得経路

- 本設計: `doc/work113/20261111-1512_domain設計/design_diff.md`
- 取得経路: `doc/work113/20261111-1512_domain設計/MANIFEST.md`（実パス・サイズ・共有リンク）
- 証跡（v2.6 の検証）: `doc/work113/20261111-2130_domain設計26/evidence/`（`contract_check.cpp`・ビルド実行ログ・ソース現状）
- 証跡（v2.7 の部分実証）: `doc/work113/20261111-2205_domain設計27/evidence/`（無条件 `__debugbreak()` のプローブ。15-c の根拠には使わない）
- 証跡（v2.8 の 15-c 実測）: `doc/work113/20261111-2310_domain設計28/evidence/`（JUCE 実ヘッダを使うプローブ・最小デバッガ・3 条件の実行ログ）
- 証跡（v2.9 の条件行列）: `doc/work113/20261111-2350_domain設計29/evidence/`（5 条件のビルドと実行ログ・プローブ・最小デバッガ）
- コード変更: **なし**（本稿は設計のみ）

## §12 照合用のソース引用（2026-11-11 時点・実測）

`grep -n` の実測行。行番号は変わり得るため、照合は**引用文字列**で行う。

| 位置 | 引用（実ファイル） | 何を示すか |
|---|---|---|
| `src/audioengine/AudioEngine.h:5455` | `convo::EpochDomain m_epochDomain;` | Engine の epoch ドメイン実体 |
| `src/audioengine/AudioEngine.CtorDtor.cpp:62` | `m_retireRouter = std::make_unique<convo::isr::ISRRetireRouter>(m_epochDomain, &worldRetirementReference_);` | Engine router は Engine のドメインに束縛 |
| `src/eqprocessor/EQProcessor.h:612` | `std::unique_ptr<convo::EpochDomain> m_epochDomain { std::make_unique<convo::EpochDomain>() };` | EQ の epoch ドメイン実体 |
| `src/eqprocessor/EQProcessor.h:618` | `convo::isr::ISRRetireRouter m_ownedRetireRouter { *m_epochDomain };` | EQ 専用 router は EQ のドメインに束縛 |
| `src/ConvolverProcessor.h:1468` | `convo::EpochDomain m_epochDomain;` | Convolver の epoch ドメイン実体 |
| `src/ConvolverProcessor.h:1470` | `convo::RCUReader runtimeRcuReader { m_epochDomain };` | Convolver のドメインは reader 専用 |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp:59` | `provider->enqueueDeferredDeleteNonRt(oldState, deleter);` | IRState 退役は Engine 経由 |
| `src/convolver/ConvolverProcessor.Lifecycle.cpp:72` | `provider->enqueueDeferredDeleteNonRt(sc, destroyStereoConvolver);` | StereoConvolver 退役は Engine 経由 |
| `src/convolver/ConvolverProcessor.Runtime.cpp:225` | `convo::RCUReaderGuard guard(runtimeRcuReader);` | Convolver 自身の reader 区間 |
| `src/audioengine/AudioEngine.Processing.BlockDouble.cpp:158` | `const convo::RCUReaderGuard rcuGuard(audioThreadRcuReader` | Engine reader が audio callback を覆う |
| `src/audioengine/AudioEngine.Processing.DSPCoreDouble.cpp:383` | `convolverRt().process(processBlock, diagSink());` | Convolver 処理は Engine reader 区間内 |
| `src/audioengine/ISRRetireRouter.h:205` | `class ISRRetireRouter : public convo::IEpochProvider,` | Router は IEpochProvider を継承 |
| `src/audioengine/ISRRetireRouter.h:474` | `convo::IEpochProvider* provider_ = nullptr;` | provider は private メンバ（公開アクセサなし） |
| `src/core/IEpochProvider.h:42` | `class IEpochProvider : public IReaderEpochProvider,` | IEpochProvider は IRetireProvider を含む |
| `src/audioengine/ISRRetireRouter.cpp:107` | `void TerminalReclaimAuthority::drain(uint64_t minReaderEpoch,` | Terminal drain の現行シグネチャ |
| `src/audioengine/ISRRetireRouter.cpp:223` | `void TerminalReclaimAuthority::drainAll() noexcept` | Terminal drainAll の現行シグネチャ |
| `src/audioengine/RetireQuarantineStore.h:102` | `void drain(uint64_t minReaderEpoch,` | Quarantine drain の現行シグネチャ |
| `src/audioengine/RetireQuarantineStore.h:155` | `void drainAllUnsafe() noexcept` | Quarantine drainAll の現行シグネチャ |
| `src/DeferredDeletionQueue.h:42` | `struct DeletionEntry {` | キュー entry の定義 |
| `src/DeferredDeletionQueue.h:169` | `convo::isr::RetireLatencyRecorder::instance().recordComplete (` | queue 経路の完了記録 |
| `src/audioengine/ISRRetireLatency.h:312` | `void recordComplete (const void* object, Path path, std::uint64_t firstScanTsNs = 0) noexcept` | 現行シグネチャ |
| `src/core/SnapshotRetireManager.h:49` | `DeletionQueue m_queue;` | レガシー queue の唯一の保持者（インスタンス生成なし） |
| `src/core/IRetireProvider.h:5` | `#include "../DeferredDeletionQueue.h"` | `RetireDomainId` をここに定義すると循環する根拠 |
| `src/core/EpochDomain.h:671` | `DeferredDeletionQueue deferredDeletionQueue;` | キューの実体は `EpochDomain` が保有 |
| `src/eqprocessor/EQProcessor.h:622` | `std::unique_ptr<convo::EpochDomain> m_epochDomain { std::make_unique<convo::EpochDomain>() };` | EQ の生成箇所（v2.3 で明示引数化） |
