# Phase D2-0 事前監査 — PrepareResult / BuildError 伝播の必要性検証

- **日付**: 2026-08-19
- **対象**: Phase D2「`PrepareResult`（status + subsystem + retryability）ヒエラルキー導入」の必要性検証（`REPAIR_PLAN2-dash2.md §1.8.9 Phase 2` / `§1.8.11`）
- **目的**: 現在の `BuildError` 伝播（`build()` → `BuildResult` → caller の `runtime == nullptr` 検知 → classifyBuildError）が十分かどうかを caller 単位で追跡し、「何の情報が実際に失われているのか」を証明する。chain のどこにも semantic loss がなければ NO-GO で閉じる
- **判定**: **NO-GO（現時点では実装しない）** — 条件付き GO（将来 convolver/prepare の実 failure が観測可能になり、subsystem 別の retry 判定が必要になる設計確定時）
- **ゲート**: `PrepareResult` の production implementation は変更しない（本監査では production code を一切変更しない）

---

## 1. Phase D 仕様（D2 PrepareResult 提案の定義）

### 1.1 dash2 §1.8.9 Phase 2（REPAIR_PLAN2-dash2.md:1001-1018）— 将来 wiring

D2 の核心は **§1.8.9 Phase 2 項目 8**:

> `DSPCore::prepare()` の戻り値を `void` → ステータス型（`PrepareResult` with status + subsystem + retryability）

Phase 2 の全体項目（7-15）:

| # | 内容 | 実装状況 |
| --- | --- | --- |
| 7 | `BuildFailureClass` / `RetryDisposition` enum を RuntimeBuilder.h に追加 | ✅ 実装済み（`FailureClassification` / `RetryDisposition` として） |
| 8 | `DSPCore::prepare()` の戻り値を `void` → `PrepareResult`（status + subsystem + retryability） | ❌ **未実装（本監査の対象）** |
| 9 | `ConvolverProcessor::applyBuildSnapshot()` / `transferIRStateFrom()` にステータス/エラー伝播を追加 | ❌ 未実装 |
| 10 | build() の try ブロック内でステータスチェック → `MKLFailure` / `ConvolverFailure` / `PrepareFailure` をセット | ❌ 未実装 |
| 11 | `BuildError → BuildFailureClass → RetryDisposition` 変換関数 | ✅ 実装済み（`classifyBuildError` / `kBuildErrorDefaultTable`） |
| 12 | caller の failure check を `runtime == nullptr` or `error != None` に強化 | ⚠️ 部分的（現行は `runtime == nullptr` のみ） |
| 13 | WarmupFailed call-site ごとに context-dependent retry | ✅ 実装済み（site3: `shouldRetryWarmupFailure` / site2: `settlePendingRecoveryAdmission(true)` / site1: `continue`） |
| 14 | `kMaxRecoveryConsecutiveFailures=4` による infinite retry loop prevention を維持 | ✅ 実装済み |
| 15 | `buildErrorCount_` telemetry カウンタを AudioEngine に追加 | ❌ 未実装 |

### 1.2 §1.8.11 影響範囲（dash2:1119）

> **フェーズ 2**: RuntimeBuilder.h/cpp（BuildFailureClass + RetryDisposition + PrepareResult）, AudioEngine.h（prepare ステータス）, AudioEngine.Processing.DSPCoreLifecycle.cpp（prepare ステータス返却）, ConvolverProcessor.h/cpp（ステータス伝播）, RebuildDispatch.cpp（caller failure check 強化 + WarmupFailed context-dependent retry）, AudioEngine.h（buildErrorCount_ 追加） — 影響: **中**（error handling パスの変更）。リスク: **中**（retry ポリシーの不整合による無限リトライ）

## 2. prepare / build API 完全列挙（subsystem 失敗モード）

| # | API | 戻り値 | 失敗モード | 備考 |
| --- | --- | --- | --- | --- |
| 1 | `DSPCore::prepare(...)`（AudioEngine.h:869 / DSPCoreLifecycle.cpp:72） | **void** | なし（サブシステムは全て void。allocation 失敗は `std::bad_alloc` 例外 → build() の catch で `ResourceUnavailable` に変換） | Non-RT（Message/Worker）経路。RAII で commit 前に完了。RT 例外・RT allocation の対象外 |
| 2 | 内部サブシステム prepare（10+）: `ramp.prepare()` / `oversampling.prepare()` / `convolverState->prepare()` / `eqState->prepare()` / `dcBlockers().init()` / noiseShaper prepare / `outputFilter.prepare()` / `truePeakDetector.prepare()` / `loudnessMeter.prepare()` / `peakLimiter.prepare()` | **全て void** | 失敗を返さない。内部失敗は log のみ（juce::Logger）または黙殺 | **§1.8.12 項目1 調査完了: ステータス型導入には全サブシステムの戻り値変更が必要** |
| 3 | `RuntimeBuilder::build(...)`（RuntimeBuilder.cpp:428-469） | `BuildResult { runtime, error, prepared }` | `InvalidInput`（入力不正）/ `ResourceUnavailable`（bad_alloc）/ `InternalError`（catch...）/ `None`（成功） | try/catch は **常に有効**（production code、diagnostics guard なし — §1.8.12 項目6） |
| 4 | `RuntimeBuilder::validateWarmup(...)`（RuntimeBuilder.cpp:471-479） | `BuildError` | `WarmupFailed`（isIRLoaded && !isIRFinalized）/ `None` | |
| 5 | `buildRuntimePublishWorld(...)`（RuntimeBuilder.cpp:179-426） | `worldOwner`（失敗時 nullptr） | BuildError を**使用しない**。内部 try/catch で nullptr | **§1.8.12 項目5: build()（DSPCore 構築）と buildRuntimePublishWorld()（World 組立）は責務分離が正しい。将来の分離は不要** |
| 6 | `applyBuildSnapshot(...)`（ConvolverProcessor.h:477,1132） | **void** | 例外なし。ScopedLock + publishAtomic のみ（§1.8.12 項目2） | ステータスチェック不能 |
| 7 | `transferIRStateFrom(...)`（ConvolverProcessor） | **void** | IR AudioBuffer コピー。成功/失敗を `juce::Logger` でログ（`[CONV_IR] transferIRStateFrom: ...`）のみ（§1.8.12 項目2） | 戻り値なし |
| 8 | `publishRuntimeProcessSnapshot()`（ConvolverProcessor.h:1027） | **void** | 常に成功（allocation なし）。エラー伝播パスなし（§1.8.12 項目4） | |
| 9 | `ConvolverProcessor::prepareToPlay`（Lifecycle.cpp:276-295） | — | `newConv->init(...)` が **bool** を返す。false → `writeToLog("NUC re-init failed...")` + **既存 engine 保持**（サイレント fallback） | **UI プロセッサー経路。build() チェーン外**（DSPCore::prepare とは別 convolver）。可用性優先の意図的設計 |
| 10 | `OversamplingPolicy::resolve(buildInput)` | `{ supported, resolvedOsFactor }` | 非対応 OS 因子 → factor=0 | prepare 内で唯一 status を持つ |

## 3. エラー伝播機構の完全列挙

| # | 機構 | 使用箇所 | 伝播先 |
| --- | --- | --- | --- |
| 1 | `BuildResult.error`（BuildError enum） | `build()` | caller の `runtime == nullptr` 検知後、toString / classifyBuildError 用 |
| 2 | `BuildError` 戻り値 | `validateWarmup()` | caller の `!= BuildError::None` チェック |
| 3 | `runtime == nullptr` 検知 | **全 3 call site** | build 失敗の一次検知。error フィールドはログ/分類のみ |
| 4 | try/catch（`std::bad_alloc` → ResourceUnavailable, `...` → InternalError） | `build()` | 常に有効（production） |
| 5 | bool 戻り値 | `newConv->init()` | false → log + 既存 engine 保持（サイレント fallback） |
| 6 | void + log-only | prepare サブシステム / applyBuildSnapshot / transferIRStateFrom / publishRuntimeProcessSnapshot | 失敗はログのみ（黙殺） |
| 7 | `classifyBuildError()` → `BuildOutcome{ classification, retry }` | **site 3 のみ**（main rebuild, RebuildDispatch.cpp:1091-1102） | diagLog（error + classification + retry） |
| 8 | 例外の RT 境界伝播 | — | **なし**（prepare は Non-RT。RT thread は build/prepare 結果を参照しない） |

## 4. production callers 完全列挙（failure semantics の caller 単位追跡）

`rebuildThreadLoop` 内の 3 call site（`AudioEngine.RebuildDispatch.cpp`）:

| Site | 場所 | コンテキスト | build 失敗時 | warmup 失敗時 | retry 方針 |
| --- | --- | --- | --- | --- | --- |
| 1 | :941-956 | transient recovery（recoveryIntentQueue_ 消費） | `runtime == nullptr` → diagLog(toString) + `continue`（discard） | `!= None` → destroyDSPCoreNode + `continue` | **常に discard** |
| 2 | :1015-1049 | durable recovery（PendingRecoveryAdmission 消費） | `runtime == nullptr` → diagLog + `settlePendingRecoveryAdmission(true)` + 連続失敗カウンタ | `!= None` → destroy + `settle(true)` + カウンタ | **常に retry**（+ kMaxRecoveryConsecutiveFailures=4） |
| 3 | :1087-1102 | main rebuild（task snapshot） | `runtime == nullptr` → **classifyBuildError(buildResult.error)** + diagLog（error+classification+retry）+ `continue` | `!= None` → `shouldRetryWarmupFailure`（isLoadingIR check） | classification をログ記録（retry 実適用は D-5 将来拡張） |

**重要な観察**:

- 3 site 全てが **`runtime == nullptr` で失敗を一次検知**し、error フィールドは toString / classifyBuildError のログ用
- site 3 は既に `classifyBuildError`（§1.8.8.2 return-code 配線）を適用済み
- WarmupFailed の retryability は call-site 依存（site3: isLoadingIR / site2: 常に retry / site1: 常に discard）— **既に context-dependent 実装済み**（§1.8.5.2 設計どおり）
- `buildRuntimePublishWorld` は publish 経路で呼ばれ、失敗時 nullptr（内部扱い）

## 5. BuildError 分類との接続（実装済み）

`RuntimeBuilder.h` に **完全実装済み**:

- `BuildError`（8値）/ `FailureClassification`（Permanent/Transient/Infrastructure/Fatal）/ `RetryDisposition`（NoRetry/RetryBackoff/RetryImmediate）/ `BuildOutcome` / `BuildResult`
- `kBuildErrorDefaultTable`（8エントリ、`static_assert` で網羅性検証）
- `kBuildErrorNames`（8エントリ、`static_assert`）
- `classifyBuildError` / `classifyBuildErrorToString`（bounds-check 付き table lookup）

**実テーブルの分類**（§1.8.5.2 設計の Inspect とは異なり、既に実用的な分類が付与済み）:

| BuildError | classification | retry | build() で返り得るか |
| --- | --- | --- | --- |
| None | Permanent | NoRetry | ✅（成功） |
| InvalidInput | Permanent | NoRetry | ✅ |
| ResourceUnavailable | Transient | RetryBackoff | ✅ |
| MKLFailure | Fatal | NoRetry | ❌（enum のみ） |
| ConvolverFailure | Infrastructure | RetryBackoff | ❌（enum のみ） |
| PrepareFailure | Infrastructure | RetryBackoff | ❌（enum のみ） |
| WarmupFailed | Transient | RetryImmediate | ✅（validateWarmup） |
| InternalError | Fatal | NoRetry | ✅ |

**§1.8.5.3 の semantic mismatch（設計レベル）**: MKLFailure/ConvolverFailure/PrepareFailure は「一時的（retry）」設計だが実コードでは生成されない（enum+toString のみの保険分類）。**ただしこれらはどのコードパスからも生成されないため、ランタイムの semantic loss は存在しない**。mismatch は将来 wiring 時の設計整合性の問題であり、現時点の failure 挙動に影響しない。

## 6. Realtime boundary

- `DSPCore::prepare()` は **Non-RT（Message/Worker）** 経路で実行（DSPCoreLifecycle.cpp:86-91 コメント明記）。RT 例外禁止・RT allocation 禁止の対象外
- RT thread は build/prepare の結果を**参照しない**。完成した DSPCore のみを immutable publish 経由で受け取る
- → **RT 境界での semantic loss は構造的に不可能**

## 7. Runtime publication との関係

- チェーン: **Prepare → Build → Publish**
  - `build()` = DSPCore 構築（invalid input / alloc / internal を BuildError で分類可能）
  - `buildRuntimePublishWorld()` = World 組立（責務分離済み。failure は nullptr、InternalError 相当 — §1.8.12 項目5）
- build 失敗時: caller は **publish しない**（site1/3 は `continue`、site2 は `settle(true)` = DurablePending へ戻す）。前回 runtime は immutable publish により**保持**される
- プレースホルダー DSP セマンティクス: build 失敗時は新 World をコミットせず、既存 runtime / プレースホルダーで継続
- → **build 失敗 → 前回 runtime 保持の chain に情報喪失なし**

## 8. 既存 invariant

| # | invariant | 根拠 |
| --- | --- | --- |
| 1 | build 失敗時の runtime 所有権: DSPCore は try 内の `aligned_make_unique`（RAII）で確保、成功時のみ `release()` | RuntimeBuilder.cpp:440-452。失敗時はデストラクタで自動解放（リークなし） |
| 2 | 前回 runtime の保持: build 失敗で publish しない | immutable publish パターン。caller 3 site すべて |
| 3 | プレースホルダー DSP: build 失敗時、新 World をコミットしない | site1/2/3 の `continue` / `settle` |
| 4 | 未コミット DSP の破棄: warmup 失敗時 `destroyDSPCoreNode` してから continue | RebuildDispatch.cpp:970-978, 1037-1045（監査指摘で追加済み） |
| 5 | Recovery: quarantined DSP 除外、IR は UI convolver から `transferIRStateFrom` で転送 | site1/2 の buildSource 値コピー |
| 6 | durable admission の lease: build 成功 → `settle(false)` / transient → `settle(true)`（retry を構造的に保証） | site2 |

## 9. 既存テスト

| テスト | BuildError 関連 | 内容 |
| --- | --- | --- |
| `BuildInputSemanticContractTests.cpp` | `BuildResult build(...)` シグネチャ・`applyBuildSnapshot(...)` をソース契約として固定 | sealed snapshot フロー。**BuildError 値・WarmupFailed・classifyBuildError は未参照** |
| `RuntimeWorldAuthorityProjectionTests.cpp` | RuntimeBuilder.cpp をソース読取 | World authority 投影 |
| `PublishPipelineIntegrationTests` / `SoakPublishIntegrationTests` / `WorldRetirementMeasurementTests` | RuntimeBuilder を使用 | 成功パスの実 World ビルド |
| **BuildError 分類テスト** | **なし** | **toString / classifyBuildError / WarmupFailed / PrepareResult を参照するテストは 0 件**（AiDex 検索 0 ヒット — §1.8.6 / §1.8.11 と一致） |

**現状**: BuildError taxonomy（8値）は enum+table の静的検証のみで、ランタイム挙動のテストが存在しない。

## 10. D2 実装 diff（PrepareResult 導入のコスト）

§1.8.9 Phase 2（項目 8-10, 12, 15）+ §1.8.11 影響範囲:

- `RuntimeBuilder.h/cpp`: `PrepareResult` 型追加 + build() のステータスチェック（項目10）
- `AudioEngine.h` / `AudioEngine.Processing.DSPCoreLifecycle.cpp`: `prepare()` の void → status 化
- `ConvolverProcessor.h/cpp`: `applyBuildSnapshot()` / `transferIRStateFrom()` の戻り値追加
- `RebuildDispatch.cpp`: caller failure check 強化（`runtime == nullptr || error != None`）
- `AudioEngine.h`: `buildErrorCount_` telemetry

**前提条件（§1.8.12 項目1・2・3 調査完了）**: この diff を意味あるものにするには、**先に 10+ の prepare サブシステム全てが status を返すよう変更**が必要（現在全て void）。加えて convolver の `init()` / `transferIRStateFrom()` にも戻り値追加が必要。**これは error handling パス全体の侵入的変更**であり、現時点で生成される failure は全て既存機構で正しく伝播している。

## 11. 実際の利益（semantic loss 分析 — 監査の核心）

**chain: `prepare failure → build classification → BuildError → caller handling → publication / fallback / 前回 runtime 保持`**

| chain 区間 | 現状 | semantic loss |
| --- | --- | --- |
| prepare failure | サブシステムは全て void。failure モード = bad_alloc（→ ResourceUnavailable に変換済み）or 黙殺。**伝播すべき status がそもそも存在しない** | **なし**（生成されない情報は失われない。PrepareResult は全サブシステムの status 化を前提にした先回り） |
| build classification | `InvalidInput` / `ResourceUnavailable` / `InternalError` / `None` を正しく返す | **なし** |
| BuildError → caller | 3 site 全て `runtime == nullptr` で検知し、コンテキスト別に discard / retry / classify を正しく適用。site 3 は `classifyBuildError` 適用済み | **なし** |
| publication / fallback | build 失敗 → publish しない。前回 runtime 保持。durable recovery は retry。プレースホルダー継続 | **なし** |

**結論: この chain のどこにも semantic loss は存在しない。**

- 現行 failure モード（InvalidInput / ResourceUnavailable / InternalError / WarmupFailed）は全て既存機構で正しく分類・伝播・処理される
- `MKLFailure` / `ConvolverFailure` / `PrepareFailure` は**どのコードパスからも生成されない**（保険分類）。このため §1.8.5.3 の「設計上の一時的 vs 実装上の InternalError 丸め」の不一致は**設計レベルで休眠状態**であり、ランタイムの情報喪失を引き起こさない
- PrepareResult の利益は全て**投機的**（将来 convolver/prepare failure の subsystem 別分類）であり、その前提（サブシステムの status 化）は現状存在しない

## 12. GO/NO-GO 判定

### 判定: **NO-GO（現時点では実装しない）**

**理由**:

1. 現在の `BuildError` + `BuildResult` + `runtime == nullptr` 検知 + `classifyBuildError`（site 3）の伝播は、**全ての現行 failure モードに対して十分**
2. chain のどこにも semantic loss がなく、PrepareResult が回収する情報が現状存在しない
3. PrepareResult 導入は 10+ サブシステムの status 化（§1.8.12 項目1）を前提とした**大規模侵入的変更**で、現行の利益ゼロ・リスク中
4. BuildError 分類機構（table / static_assert / classifyBuildError）は既に実装済みであり、D2 の核心は「prepare の status 化」のみに残っているが、それは先回りの保険に過ぎない

### 条件付き GO（将来トリガー）

**IF**（いずれか）:

- runtime build 経路（build() → DSPCore::prepare）で convolver/prepare の**実 failure が観測可能**になる（例: `newConv->init()` の bool を convolver 構築パスで `ConvolverFailure` に wiring、または prepare サブシステムが実 status を返すようになる）
- かつ、その failure に**subsystem 別 / retryability 別の区別が必要**になる設計が確定する

**THEN**: フル `PrepareResult` ヒエラルキーではなく**最小 wiring** で対応:

- 実際に失敗する関数（convolver init / prepare subsystem）に status 伝播を追加（§1.8.9 項目9相当）
- caller failure check を `runtime == nullptr || error != None` に強化（項目12相当）
- `buildErrorCount_` telemetry 追加（項目15相当）

（これは D3/D4 の convolver failure wiring 領域であり、D2 の PrepareResult 全体導入は不要）

### 推奨アクション（production code 変更なし）

1. **`BuildErrorClassificationTests.cpp` 追加**（Phase 1 safety hardening / D1 領域、D2 ではない）: 全 8 enum 値の `toString` / `classifyBuildError` 網羅性をテスト。§1.8.6 のテストゼロ問題（AiDex 0 ヒット）を解消
2. **§1.8.5.3 の semantic mismatch を「受け入れ済み設計（休眠）」として明文化**: enum-only 値が将来 wiring される時点で解消すれば良い（現行 failure 挙動への影響なし）
3. **将来の convolver failure wiring 時は最小 status-code 伝播を優先**し、フル PrepareResult ヒエラルキーは導入しない

---

## 付録: 監査のスコープ

- **変更した production code**: なし（本監査は audit-only）
- **ゲート**: `PrepareResult` の production implementation は変更しない（遵守）
- **参照**: `REPAIR_PLAN2-dash2.md §1.8`（Phase D 仕様 / §1.8.5.2 / §1.8.5.3 / §1.8.6 / §1.8.9 / §1.8.11 / §1.8.12）、`RuntimeBuilder.h`（BuildError 分類）、`RuntimeBuilder.cpp`（build/validateWarmup）、`AudioEngine.RebuildDispatch.cpp`（3 call site）、`AudioEngine.Processing.DSPCoreLifecycle.cpp`（prepare）、`ConvolverProcessor.Lifecycle.cpp`（newConv->init() fallback）、`BuildInputSemanticContractTests.cpp`
