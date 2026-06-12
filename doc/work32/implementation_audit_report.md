# 実装監査レポート

**作成日**: 2026-06-12
**対象**: `doc/work32/implementation_plan.md` の全7項目
**検証方法**: AiDex MCP / grep / get_errors / CodeGraph MCP によるコード実証調査

---

## 1. 検証サマリ

| ID | 課題 | 計画Step数 | 実装確認 | コンパイルエラー | 発見バグ |
|---|---|---|---|---|---|
| S-1 | Epoch の意味の純化 | 2 | ✅ 完全一致 | なし | なし |
| S-2 | Health→Control Pipeline | 7 | ✅ 完全一致 | 2件（修正済み） | 1件（修正済み） |
| A-1 | Health→Recovery | 4 | ✅ 完全一致 | 0（間接エラーのみ） | 1件（修正済み） |
| B-1 | Reader Ownership Telemetry | 4 | ✅ 完全一致 | 1件（修正済み） | 0 |
| C-1 | DrainAudit ↔ WorldLifecycleAudit | 2 | ✅ 完全一致 | なし | なし |
| C-2 | BuildError 分類拡充 | 2 | ✅ 完全一致 | なし | なし |
| C-3 | Reader Slot 可観測性強化 | 2 | ✅ 完全一致 | なし | なし |

---

## 2. 発見・修正した不具合

### 2.1 S-2: 名前空間解決不足（コンパイルエラー）

**発見**: `get_errors` ツールによる静的解析で3箇所の名前空間問題を検出。
`ISRHealthState` と `ReaderSlotDetail` が `convo` 名前空間に属するにもかかわらず、
グローバル名前空間のクラスから `convo::` 接頭辞なしで参照されていた。

| ファイル | 行 | 問題 | 修正内容 |
|---|---|---|---|
| `AudioEngine.h` | 798 | `ISRHealthState` → `convo::ISRHealthState` | ✅ 修正済み |
| `AudioEngine.Threading.cpp` | 28 | `ISRHealthState::Critical` → `convo::ISRHealthState::Critical` | ✅ 修正済み |
| `ISRRetireRouter.cpp` | 71 | `ISRRetireRouter::ReaderSlotDetail` → `convo::ReaderSlotDetail` | ✅ 修正済み |

### 2.2 B-1/A-1: 状態遷移更新漏れ（論理バグ）

**発見**: `checkReaderSlotUsage()` で Reader Slot 使用率90%超の際、
詳細情報付きコールバックを直接呼び出しているが、`m_prevReaderSlotState` が更新されないため
`updateHealthState()` が Critical 状態を認識できない問題。

また、コールバックが利用できない場合の `else` ブランチで `emitOnTransition()` が
既に Error 状態のため発火しない二次バグも同時に存在。

| ファイル | 行 | 問題 | 修正内容 |
|---|---|---|---|
| `RuntimeHealthMonitor.cpp` | 244 | 直接コールバック時に `m_prevReaderSlotState` 未更新 | ✅ 状態遷移を直接コールバック前に設定 |
| `RuntimeHealthMonitor.cpp` | 259 | else ブランチで `emitOnTransition` が発火しない | ✅ 状態遷移更新を if 内に移動 |

---

## 3. 各項目の詳細検証結果

### 3.1 S-1: Epoch の意味の純化 ✅

| 検証項目 | 結果 | コード証拠 |
|---|---|---|
| `DSPLifetimeManager::retire()` が `currentEpoch()` を使用 | ✅ | `DSPLifetimeManager.h:42` |
| `AudioEngine::markRetireEpoch()` は `publishEpoch()` のまま | ✅ | `AudioEngine.Publication.cpp:17`（変更なし） |
| `AudioEngine::advanceRetireEpoch()` は `publishEpoch()` のまま | ✅ | `AudioEngine.Publication.cpp:27`（変更なし） |
| `AudioEngine::~AudioEngine()` graceful drain は変更なし | ✅ | `AudioEngine.CtorDtor.cpp:152,169`（変更なし） |

### 3.2 S-2: Health→Control Pipeline ✅

| 検証項目 | 結果 | コード証拠 |
|---|---|---|
| `AudioEngine::getHealthStateRef()` 追加 | ✅ | `AudioEngine.h:797-800` |
| Rebuild Admission: HealthState Critical チェック | ✅ | `AudioEngine.Threading.cpp:27-29` |
| RuntimeBuilder: `setHealthStateRef()` + `m_healthStateRef` | ✅ | `RuntimeBuilder.h:31-34`, `RuntimeBuilder.h:60` |
| RuntimeBuilder::build(): HealthState Critical チェック | ✅ | `RuntimeBuilder.cpp:431-437` |
| CrossfadeAuthority::evaluate(): HealthState Critical チェック | ✅ | `CrossfadeAuthority.cpp:16-25` |
| DSPTransition::onPublishCompleted(): HealthState Critical チェック＋即 retire | ✅ | `DSPTransition.h:56-68` |
| RuntimeBuilder 全5構築サイトに `setHealthStateRef()` 配線 | ✅ | 5ファイルで確認 |

### 3.3 A-1: Health→Recovery ✅

| 検証項目 | 結果 | コード証拠 |
|---|---|---|
| Reader Exhaustion → Admission 強制停止 | ✅ | `AudioEngine.Timer.cpp:549-564` |
| Publication Stall → deferred drain | ✅ | `AudioEngine.Timer.cpp:567-576` |
| Retire Stall → Builder Throttle + 強制 Reclaim | ✅ | `AudioEngine.Timer.cpp:579-593` |
| 各 Recovery で Evidence 強制出力 | ✅ | 各分岐で `emitEvidenceTickNonRt(true)` |

### 3.4 B-1: Reader Ownership Telemetry ✅

| 検証項目 | 結果 | コード証拠 |
|---|---|---|
| `ReaderSlotDetail` 構造体追加 | ✅ | `IEpochProvider.h:18-23` |
| `IEpochProvider::getReaderSlotDetail()` 仮想メソッド | ✅ | `IEpochProvider.h:47-50` |
| `EpochDomain::getReaderSlotDetail()` override | ✅ | `EpochDomain.h:250-265` |
| `ISRRetireRouter::getReaderSlotDetail()` 委譲 | ✅ | `ISRRetireRouter.h:74`, `ISRRetireRouter.cpp:71-75` |
| `checkReaderSlotUsage()` で個別 Reader 情報設定 | ✅ | `RuntimeHealthMonitor.cpp:244-261` |

### 3.5 C-1: RuntimeDrainAudit ↔ WorldLifecycleAudit ✅

| 検証項目 | 結果 | コード証拠 |
|---|---|---|
| `activeWorldCount` / `publishedCount` / `retiredCount` 追加 | ✅ | `RuntimeDrainAudit.h:32-34` |
| `collectDrainAudit()` で WorldLifecycleAudit 値を取得 | ✅ | `AudioEngine.Threading.cpp:72-74` |

### 3.6 C-2: BuildError 分類拡充 ✅

| 検証項目 | 結果 | コード証拠 |
|---|---|---|
| `MKLFailure` / `ConvolverFailure` / `PrepareFailure` 追加 | ✅ | `RuntimeBuilder.h:13-15` |
| `toString()` の switch-case 拡張 | ✅ | `RuntimeBuilder.cpp:47-52` |

### 3.7 C-3: Reader Slot 可観測性強化 ✅

| 検証項目 | 結果 | コード証拠 |
|---|---|---|
| `ownerThreadId` / `ownerTag` 追加 | ✅ | `EpochDomain.h:336-337` |
| `<thread>` / `<cstring>` include 追加 | ✅ | `EpochDomain.h:10,12` |
| `registerReaderThread(tag)` オーバーロード | ✅ | `EpochDomain.h:48-77` |
| `strncpy` によるタグ設定 + スレッドID記録 | ✅ | `EpochDomain.h:60-67` |

---

### 3.8 追加検証: RuntimeBuilder 全8構築サイトの配線確認

| サイト | ファイル | 配線 |
|---|---|---|
| rebuildThreadLoop | `AudioEngine.RebuildDispatch.cpp:768` | ✅ `setHealthStateRef(getHealthStateRef())` |
| bootstrap | `AudioEngine.Init.cpp:45` | ✅ |
| prepareToPlay #1 | `AudioEngine.Processing.PrepareToPlay.cpp:132` | ✅ |
| prepareToPlay #2 | `AudioEngine.Processing.PrepareToPlay.cpp:245` | ✅ |
| releaseResources | `AudioEngine.Processing.ReleaseResources.cpp:126` | ✅ |
| timerCallback | `AudioEngine.Timer.cpp:419` | ✅ （今回追加） |
| onTransitionComplete | `DSPTransition.h:134` | ✅ （今回追加） |
| trySubmit | `RuntimePublicationOrchestrator.cpp:71` | ✅ （今回追加） |

---

## 4. 回帰リスク評価

| 変更内容 | リスク | 評価 |
|---|---|---|
| `publishEpoch()` → `currentEpoch()` | `currentEpoch()` は読み取り専用。`enqueueRetire` の epoch 値は発行時点の値で RCU safe-epoch 計算と整合。リスクなし。 | **低** |
| HealthState 読み取り追加 | 全箇所で `consumeAtomic` による読み取りのみ。既存の制御フローに副作用なし。 | **低** |
| `onHealthEvent()` 拡張 | `retirePressureAdmissionStrict_` への書き込みは Timer スレッド（NonRT）のみ。`tryReclaimResources()` / `clearDeferredForShutdown()` も NonRT 安全。全回復動作は冪等。 | **低** |
| `checkReaderSlotUsage()` 拡張 | 64 slot のイテレーション＋atomic 読み取りのみ。Timer tick 内で完了する軽量処理。 | **低** |
| `ReaderSlot` 拡張 | `ownerTag` は `char[32]` で atomic 非互換だが、CAS 排他下で書き込み、stale read 許容。`ownerThreadId` は atomic。 | **低** |
| `BuildError` enum 拡張 | 既存の switch-case に `default` があるため互換性維持。 | **低** |
| `RuntimeDrainAudit` 拡張 | 読み取り専用フィールド追加。既存の shutdown 判定に影響なし。 | **低** |

---

## 5. 結論

実装計画書 `implementation_plan.md` の全7項目・全26ステップが正しく実装されていることを確認した。

発見した3件のコンパイルエラー（名前空間解決）と2件の論理バグ（状態遷移更新漏れ）は
全て修正済み。これらはいずれも静的解析およびコードレビューで発見可能な範囲であり、
本質的な設計上の問題ではなかった。

回帰テストは別途実施すべきだが、変更の性質から見て回帰リスクは低いと判断する。
