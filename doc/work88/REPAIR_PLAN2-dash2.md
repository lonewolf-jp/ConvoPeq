# REPAIR_PLAN2-dash2 将来対応・一部実装の詳細設計

**★ 作成日:** 2026-08-11
**★ 対象:** `doc/work88/REPAIR_PLAN2-dash.md`（本実装は完了済み）で「将来対応」「一部実装」とされた項目の**整理・詳細設計**
**★ 前提:** REPAIR_PLAN2-dash.md の本実装（P2-1〜P2-4 / X1〜X6 / X4-B / X3-R4 Phase 7）は完了済み・ctest 28/28 PASS・CI 検証 PASS
**★ normative source:** REPAIR_PLAN2-dash.md（現行版・A-2.42 以降 + §5/§6 の「現在形」記述）＋実コード照合

---

## 実装状況サマリ

| 分類 | 項目 | 現状 |
|---|---|---|
| 将来対応 | 1.1 R1: recoveryIntentQueue_ の MPSC 化 | Phase 5 将来拡張（未実装） |
| 将来対応 | 1.2 Recovery coalesce（マージ）実装 | 四次レビュー NO-GO → 別タスク（P3） |
| 将来対応 | 1.3 §4.3: ConvolverProcessor の LinearRamp 分離 | 対象外・文書化（未実装） |
| 将来対応 | 1.4 isFullyDrained の他カウンタ実測上書きの全廃 | 別タスク（P2 範囲外） |
| 将来対応 | 1.5 PublishReceiptWaiter の sparse completion | 将来 MPSC completion 許容時のみ |
| 将来対応 | 1.6 X2 の wraparound / out-of-order テスト | 将来 sparse 化時のみ追加 |
| 将来対応 | 1.7 X4-B 案2（currentWorld_ 廃止） | 将来タスク（read-source singularization） |
| 将来対応 | 1.8 BuildError 保険分類 | enum + toString のみ・catch 拡張未実装（work32 Step 3 未対応）。静的検証・テストなし |
| 将来対応 | 1.9 初回 publish 前 quarantine の無駄な起床 | Phase 5 最適化候補 |
| 一部実装 | 2.1 R4: retire 順序逆転の完全解消 | runtime 経路対応済み・完全解消は保留 |
| 一部実装 | 2.2 shutdown lifetime contract の明文化 | R4 詳細設計で ShutdownQuiescenceProof 確定済み |

---

# 1. 将来対応（Phase 5 以降・将来タスク）

## 1.1 R1: recoveryIntentQueue_ の MPSC 化

**現状（実コード照合）**
- `recoveryIntentQueue_` は `LockFreeRingBuffer<RecoveryIntent, 256>`（SPSC、ISRRuntimePublicationCoordinator.h:433-434）
- Producer は **CoordinatorLoop 単一スレッド**（`submitRecoveryRequest` → intentQueue_ → processIntent → RecoveryIntentHandler → `submitRecoveryIntent`）
- `popRecoveryRequest`（Builder Work Queue 消費）も単一スレッド

**課題 / 動機**
- 将来 Timer 等から直接 `submitRecoveryRequest` を呼ぶ場合、**MPSC 化が必要**
- 現状は SPSC 成立のため**緊急性なし**（Phase 5 将来拡張）

**設計方針**
```
LockFreeRingBuffer<RecoveryIntent, 256>  →  MpscBoundedRing<RecoveryIntent, 256>
Producer 側 submitRecoveryRequest:  §1.1（REPAIR_PLAN2-dash）の reservation→push→rollback に変更
Consumer 側 popRecoveryRequest:     pop 成功時 fetchSub（pendingIntentCount_）
```

**実装手順**
1. `recoveryIntentQueue_` の型を `MpscBoundedRing<RecoveryIntent, 256>` へ置換
2. `submitRecoveryRequest` に reservation→push→rollback（pendingIntentCount_ 管理）を追加
3. `popRecoveryRequest` で pop 成功時 fetchSub
4. テスト追加: 2 Producer からの並行 enqueue / queue full → rollback / 重複なし / underflow なし

**Acceptance Criteria**
- MPSC 2 Producer で全件 enqueue・**重複なし・underflow なし**（reservation-before-push の happens-before チェーン）
- `isFullyDrained`（pendingIntentCount_ + queue emptiness）が整合
- 既存 Recovery テスト（ISRSemanticValidationTests / AudioEngineHarness 統合）全 PASS

**影響範囲 / リスク**
- 影響: `ISRRuntimePublicationCoordinator.h`（recoveryIntentQueue_ 型）、`submitRecoveryRequest`、`popRecoveryRequest`、テスト
- リスク: **中**（MPSC 化に伴う memory_order / Vyukov bounded queue の producer hole の正しさ）。MpscBoundedRing は P2-3 でテスト固定済み

---

## 1.2 Recovery coalesce（マージ）実装

**現状**
- REPAIR_PLAN2-dash.md §2.2 で **四次レビュー NO-GO**（`lastRecoveryHandle_` tracking は**正当な Recovery を silent loss** するため）
- 現行は「1 logical Recovery admission = exactly one reservation（INV-X1-5）」「durable は transport と二重計上しない（INV-X1-6）」で成立

**課題 / 動機**
- 同一 handle の重複 Recovery が発生した場合の**安全なマージ**（reservation 増加なし・正当な Recovery を喪失しない）
- coalesce 実装は**別タスク（P3）**として記録

**設計方針（NO-GO 理由を踏まえた安全設計）**
```
現方式（単一 last handle tracking）は採用しない（NO-GO の再発防止）
→ PendingRecoveryAdmission に「同一 handle の重複」を検出し、最新 buildSource を採用
→ マージ時は reservation を増やさない（INV-X1-5）
→ hasDuplicates API を新設し「保留中全マージ」へ拡張
```

**実装手順**
1. `PendingRecoveryAdmission` に同一 handle の重複検出ロジックを追加
2. 重複時: buildSource を最新化・reservation は増やさない（INV-X1-5 / INV-X1-6）
3. `hasDuplicates` API を新設
4. テスト追加: Recovery A,A coalesce / queue full → durable / 二重計上なし / 正当な Recovery が喪失されない

**Acceptance Criteria**
- **INV-X1-5**: 1 logical Recovery admission = exactly one reservation（coalesce で reservation 増加なし）
- **INV-X1-6**: durable Recovery は transport residency と二重計上しない
- A(G10) / B(G10) / C(G11) → G11 のみが必要（obsolete policy 再利用）

**影響範囲 / リスク**
- 影響: `PendingRecoveryAdmission`、`submitRecoveryRequest`、RebuildDispatch recovery
- リスク: **高**（coalesce が正当な Recovery を喪失しないことの保証が困難 → NO-GO 理由）。実装時は徹底的なテストが必要

---

## 1.3 §4.3: ConvolverProcessor の LinearRamp 分離

**現状（実コード照合）**
- `ConvolverProcessor.h:910,935,945` の `latencySmoother` / `crossfadeGain` / `mixSmoother` は、CrossfadeRuntime の `gain_` / `dryScaleGain_` とは**別個**の LinearRamp
- 設計判断として**対象外・文書化**（四次レビュー承認）

**課題 / 動機**
- 将来の**独立 RT-safety 検証**（各 smoother の RT スレッド所有権・atomic 性）

**設計方針**
- 対象外（文書化）を維持しつつ、将来各 smoother の read/write 箇所を棚卸しして独立検証

**実装手順**
1. `ConvolverProcessor` の 3 つの smoother（latencySmoother / crossfadeGain / mixSmoother）の read/write 箇所を棚卸し
2. 各 smoother が RT スレッドのみで write されることを検証
3. 非 RT からの write がある場合のみ修正（RT 安全性の担保）

**Acceptance Criteria**
- 各 smoother が **RT スレッドのみで write** される（非 RT write なし）
- 既存 ConvolverProcessor テスト・オーディオパス検証（HeadlessAudioPathVerification）全 PASS

**影響範囲 / リスク**
- 影響: `ConvolverProcessor.h`
- リスク: **低**（対象外・文書化。機能変更なし）

---

## 1.4 isFullyDrained の他カウンタ実測上書きの全廃

**現状（実コード照合）**
- `AudioEngine::isFullyDrained`（Threading.cpp:117,131）が、`fallbackBacklog` / `retireBacklog` / `deferredRetire` / `quarantineResident` を**外部 setter で実測上書き**している
- REPAIR_PLAN2-dash.md §1.2（:447）で「実測上書きの全廃は別タスク（P2 範囲を超える）」と記録

**課題 / 動機**
- 上書きが「drain 判定の正しさ」を担保している面があり、**廃止には全カウンタの正確な増減管理が必要**
- Coordinator 内部の純粋 accounting に一本化したい

**設計方針**
```
外部 setter（setFallbackBacklogCount / setRetireBacklogCount / setDeferredRetireResidencyCount /
quarantineResidentCount 上書き）を廃止
→ isFullyDrained で実測値（queue size / DSPQuarantineManager::residentCount）を直接判定
```

**実装手順**
1. 各カウンタの増減箇所を棚卸し（push/pop の実測）
2. `isFullyDrained` で実測値を直接参照（queue size / resident count）
3. 外部 setter を廃止
4. テスト: isFullyDrained が実測と整合 / ctest 全 PASS

**Acceptance Criteria**
- 外部 setter が**廃止**される（コード参照 0）
- `isFullyDrained` が実測 queue/resident と一致（X5/X6 のカウンタと整合）
- `isr-verify-backlog-specfixed-residual.ps1` が PASS

**影響範囲 / リスク**
- 影響: `AudioEngine.Threading.cpp`、`AudioEngine.Commit.cpp`、CI スクリプト
- リスク: **中**（上書き廃止による drain 判定の過検出/未検出。X5/X6 でカウンタが正確化されているため、廃止可能な基盤は整っている）

---

## 1.5 PublishReceiptWaiter の sparse completion

**現状（実コード照合）**
- `PublishReceiptWaiter`（AudioEngine.h:3631）は **high-water mark（`lastCompleted_`）+ contiguous completion**（INV-X2-6: completion order == publication sequence order）
- `complete()` は `if (seqId > lastCompleted_)` で monotonic watermark（INV-X2-4: stale completion は上書きしない）

**課題 / 動機**
- 将来 **MPSC completion / parallel publish / async completion** を許す場合、contiguous 前提が壊れる
- 現在は PublishExecutor sole gateway + FIFO で**不要**

**設計方針**
```
completedThrough_（contiguous frontier）+ completedOutOfOrder_（sparse set）の二重構造
waitFor(seq) は frontier と sparse を併用
INV-X2-5（sole completion writer）は維持
```

**実装手順**
1. `completedThrough_` / `completedOutOfOrder_` を導入
2. `complete()` を CAS max + sparse set 更新に変更
3. `waitFor()` を frontier + sparse 併用に変更
4. テスト追加: out-of-order / duplicate / wraparound

**Acceptance Criteria**
- out-of-order completion でも `waitFor(seq)` が正しく完了
- 既存 contiguous テスト（PublishPipelineIntegrationTests / PublishCompletionMonotonicity）維持
- INV-X2-6 の architectural test（PublishExecutor 以外から completion が発生しない）維持

**影響範囲 / リスク**
- 影響: `AudioEngine.h`（PublishReceiptWaiter）、`commitRuntimePublication`（waitForPublishReceipt）
- リスク: **高**（completion 意味論の変更。contiguous 前提が architecture に埋め込まれている。**現状は実装不要**）

---

## 1.6 X2 の wraparound / out-of-order テスト

**現状**
- INV-X2-6（completion order == publication sequence order）を architectural test で固定
- 正常系 10→11→12（FIFO completion invariant）を検証
- out-of-order は **PublishExecutor sole gateway である限り不要**（:1844）

**課題 / 動機**
- 将来 sparse completion（1.5）を導入する場合に備え、テストを準備しておく

**設計方針**
- 現状は architectural invariant 固定を維持
- **sparse 化（1.5）と同時に**、11→10（out-of-order）/ duplicate / wraparound（UINT64_MAX 近傍）のテストを追加

**実装手順**
1. 現状維持（INV-X2-6 の architectural test）
2. 1.5（sparse completion）実装時にテスト追加
3. 追加テスト: 10→11→12（正常）/ 11→10（out-of-order）/ 10→10（duplicate）/ UINT64_MAX 近傍（wraparound）

**Acceptance Criteria**
- 正常系テスト維持
- sparse 化時に out-of-order / duplicate / wraparound テストが全 PASS

**影響範囲 / リスク**
- 影響: テストのみ
- リスク: **低**（現状維持・テスト準備）

---

## 1.7 X4-B 案2（currentWorld_ 廃止 = read-source singularization）

**現状**
- `currentWorld_`（ISR メタデータ observation alias）と `runtimeStore.current`（物理 publication source）の **dual-pointer**
- X4-B は **write authority singularization** に限定（案1）
- dual-pointer は「暫定正常状態」として明示（INV-X4-6: publish 完了後は同一 PublicationIdentity を保証）

**課題 / 動機**
- read-source singularization（currentWorld_ 廃止・読み取りを全て runtimeStore.current に一本化）
- ISR commit の意味論変更を伴うため**独立タスク**（A-2.20 に記録）

**設計方針**
```
ISR 側 read（currentEpoch / sequence / version）を RuntimeState::publication から直接導出
（FUTURE-4 で persistentState_ 削除済み → currentWorld_ も削除可能な方向）
commit() の currentWorld_ 更新を廃止し、メタデータ source を runtimeStore.current に一本化
```

**実装手順**
1. ISR read（currentEpoch / sequence / version）を `runtimeStore.current` 経由に変更
2. `commit()` の `currentWorld_` 更新を廃止
3. INV-X4-6 の検証を identity 一致に一本化
4. テスト: Test 9（dual-pointer identity）/ read 経路が runtimeStore のみ / INV-X4-A/B/C 充足

**Acceptance Criteria**
- `currentWorld_` が**廃止**される
- 全 read が `runtimeStore.current`（INV-X4-B）のみ
- INV-X4-A / INV-X4-C（RT API は currentWorld_ から ownership/lifetime を導出しない）充足
- Test 9（PublicationIdentity 一致）・Test 10（INV-X4-7）維持

**影響範囲 / リスク**
- 影響: `RuntimeWorldAuthority`、`ISRRuntimePublicationCoordinator`、`RuntimeIntentCoordinator`、read path 全箇所（~13 箇所）
- リスク: **高**（ISR commit 意味論変更・大規模リファクタ）。**dual-pointer は「暫定正常状態」として許容済みのため、実装は任意・将来タスク**

---

## 1.8 BuildError 保険分類

### 1.8.1 補足: C-2 起源（work32 implementation_plan.md §7 / implementation_audit_report.md §3.6）

`MKLFailure` / `ConvolverFailure` / `PrepareFailure` は **work32 タスク C-2「BuildError 分類拡充」** によって enum と `toString` に追加された。C-2 は 4 ステップの計画だったが、checklist.md:80-81 によれば**Step 1（enum 拡張）と Step 2（toString 拡張）のみ実装・確認済み**。

| Step | 内容 | 実装状況 | 備考 |
|---|---|---|---|
| 1 | enum 拡張（MKLFailure / ConvolverFailure / PrepareFailure） | ✅ 完了 | RuntimeBuilder.h:107-113 |
| 2 | toString() switch-case 拡張 | ✅ 完了 | RuntimeBuilder.cpp:53-72 |
| 3 | build() catch 拡張（`catch(mkl::exception)→MKLFailure` 等） | ❌ **未実装** | work32 plan:803-835 に仕様記載ありが、実装はされていない |
| 4 | rebuildThreadLoop telemetry（buildErrorCount_） | ❌ **未実装** | work32 plan:842-861 は optional。`buildErrorCount_` フィールドも存在しない |

**監査の範囲不足**: work32/implementation_audit_report.md:102-107 は C-2 について**enum 追加 + toString 拡張の 2 項目のみ**を検証した（コード証拠: RuntimeBuilder.h:13-15, RuntimeBuilder.cpp:47-52）。**Step 3 の catch 拡張は監査対象外**。現行コードでも `mkl::exception` 型は存在しない（AiDex 検索: 0 ヒット）。

**監査の行番号不一致**: work32 監査は RuntimeBuilder.h:13-15 / RuntimeBuilder.cpp:47-52 を参照するが、現行コードは v9.5 リオーガナイゼーション後に RuntimeBuilder.h:107-116 / RuntimeBuilder.cpp:53-76 に移動している。監査は旧版（v9.5 以前）の行番号で実施された。

### 1.8.2 実コード照合 — BuildError の完全な分類

#### 1.8.2.1 Enum 定義（RuntimeBuilder.h:107-116）

```cpp
enum class BuildError {
    None,
    InvalidInput,
    ResourceUnavailable,
    MKLFailure,          // ★ C-2: MKL 初期化・FFT 計画失敗
    ConvolverFailure,    // ★ C-2: Convolver Build 失敗
    PrepareFailure,      // ★ C-2: DSPCore::prepare() 失敗
    WarmupFailed,
    InternalError
};
```

8 値すべて。`toString`（RuntimeBuilder.cpp:53-76）は**全 8 値に対応済み**。`switch` に `default:` はなく、フォールバックは `switch` の外に `return "Unknown"`（:75）がある（work32 監査行144 の「`default` がある」という主張は**不正確** — `default:` ラベルは存在しない）。

#### 1.8.2.2 build() 関数が実際に返す値（RuntimeBuilder.cpp:428-469）

`build()` は **try/catch** で囲まれた 3 段階の検証を行う:

| ステップ | コード | スロー可能性 | 現状のマッピング |
|---|---|---|---|
| Input validation | :433-435 (`sampleRate <= 0 \|\| blockSize <= 0`) | なし | `InvalidInput` ✅ |
| try ブロック | :441-457 | | |
| - `aligned_make_unique<DSPCore>()` | :443 | `std::bad_alloc` | `ResourceUnavailable` (:461) ✅ |
| - `convolverRt().setVisualizationEnabled(false)` | :444 | **not noexcept**（ConvolverProcessor.h:533, trivial inline） | `catch(...)` → `InternalError` |
| - `convolverRt().applyBuildSnapshot(snapshot)` | :445 | **not noexcept**（ConvolverProcessor.h:477）| `catch(...)` → `InternalError` |
| - `convolverRt().transferIRStateFrom(...)` | :447 | **not noexcept**（ConvolverProcessor.h:1132） | `catch(...)` → `InternalError` |
| - `runtime->prepare(...)` | :448-454 | **not noexcept**（AudioEngine.h:866） | `catch(...)` → `InternalError` |
| catch `std::bad_alloc` | :459-462 | | `ResourceUnavailable` ✅ |
| catch `...` | :464-467 | | `InternalError` ✅ |

**重要な設計的ギャップ — Catch-based なアプローチは根本的に不適**:
build() の try ブロック内で呼び出される convolver/prepare 操作は**すべてステータスコードベースのエラー処理**を使用する:

- `ConvolverProcessor::prepareToPlay()`（ConvolverProcessor.Lifecycle.cpp:211）: `jassert` + ステータスコードベースの初期化。C++ 例外をスローしない（`GlobalGuard` で ReaderLock を取得し、`fftHandle.reset()`、`juce::ScopedLock` を使用）。
- `ConvolverProcessor::applyBuildSnapshot()`（ConvolverProcessor.StateAndUI.cpp:271）: `juce::ScopedLock` + atomic publish。`publishRuntimeProcessSnapshot()` 内でのエラーは `bool` 戻り値や `jassert` で処理される可能性があるが、例外はスローしない。
- `DSPCore::prepare()`（AudioEngine.Processing.DSPCoreLifecycle.cpp:72）: `oversampling.prepare()`、`convolverState->prepare()`（内部で `ConvolverProcessor::prepareToPlay`）、`eqState->prepare()`、`dither.prepare()` を呼ぶ。メモリ確保失敗以外はステータスコード/`jassert` で処理される。
- MKL: `MklFftEvaluator.h` はコメントに「[v2.1] MKL DFTI を Intel IPP に換装」（:13）とあり、IPP の `IppStatus` コード（`ippsFFTInit_R_64f` 等）を使用。`DftiHandle.h` は `DftiFreeDescriptor`（C API）を RAII でラップ。**MKL は C++ 例外をスローしない** — ステータスコードベース。

したがって、work32 Step 3 の `catch(const mkl::exception&) → MKLFailure` は**アーキテクチャ的に不可能**。`mkl::exception` は存在しない上、MKL/IPP は例外をスローしない。同様に `ConvolverFailure`/`PrepareFailure` も例外ベースで捕捉できない。**正しい wiring アプローチは return-code / status-code チェックである**（1.8.4 参照）。

#### 1.8.2.3 validateWarmup() が実際に返す値（RuntimeBuilder.cpp:471-479）

```cpp
if (runtime.convolverRt().isIRLoaded() && !runtime.convolverRt().isIRFinalized())
    return BuildError::WarmupFailed;  // :476
return BuildError::None;  // :478
```

`WarmupFailed` / `None` の 2 値のみ。`convolverRt().isIRLoaded()` / `isIRFinalized()` / `isLoadingIR()` はすべて `noexcept`（AudioEngine.h:910, ConvolverProcessor.h）。

#### 1.8.2.4 実際に返る値の総計

`build()` + `validateWarmup()` の組み合わせで返り得る値:

| BuildError | build() | validateWarmup() | 実コード照合 |
|---|---|---|---|
| `None` | ✓ (implicit, :455-457) | ✓ (:478) | ✅ |
| `InvalidInput` | ✓ (:435) | ✗ | ✅ |
| `ResourceUnavailable` | ✓ (:461) | ✗ | ✅ |
| `WarmupFailed` | ✗ | ✓ (:476) | ✅ |
| `InternalError` | ✓ (:466) | ✗ | ✅ |
| `MKLFailure` | ✗ | ✗ | ✅ (未使用) |
| `ConvolverFailure` | ✗ | ✗ | ✅ (未使用) |
| `PrepareFailure` | ✗ | ✗ | ✅ (未使用) |

**結論**: 実際に返るのは `None` / `InvalidInput` / `ResourceUnavailable` / `WarmupFailed` / `InternalError` の 5 値。`MKLFailure` / `ConvolverFailure` / `PrepareFailure` は enum 定義 + `toString` のみで** build パスでは未使用**（保険分類）。

### 1.8.3 Build パスアーキテクチャ — 2 系統の分離

BuildError は**build() / validateWarmup() の 2 関数でのみ**使用される。メインの publish パス `buildRuntimePublishWorld()`（RuntimeBuilder.h:134-196, RuntimeBuilder.cpp:178-426）は **BuildError を一切使用しない**。

| 関数 | ファイル | noexcept | BuildError 使用 | 呼び出し元 |
|---|---|---|---|---|
| `build()` | h:206-207, cpp:428-469 | ✅ | ✅（BuildResult.error） | RebuildDispatch.cpp:941, :1015, :1087 の 3 サイトのみ |
| `validateWarmup()` | h:209, cpp:471-479 | ✅ | ✅（直接返り値） | RebuildDispatch.cpp:955, :1032, :1143 の 3 サイトのみ |
| `buildRuntimePublishWorld()` | h:134-137, cpp:178-426 | ✅ | ❌（別経路） | PrepareToPlay.cpp:149,271; ReleaseResources.cpp:169; Transition.cpp:22; Timer.cpp:912; Init.cpp:53; etc. |

`buildRuntimePublishWorld` は `engine.makeEngineRuntimeState()`（cpp:208）で DSPCore を間接的に作成するが、**build() を呼ばない**（cpp:428 は独立した関数定義）。失敗時は世界が nullptr になるか `jassert` で停止する（noexcept であるため例外はスローされない）。したがって **BuildError の影響範囲は rebuildThreadLoop（RebuildDispatch.cpp:897-1162）に限定**される。

### 1.8.4 3 箇所の build() 呼び出しサイトと caller パターン

すべて `AudioEngine.RebuildDispatch.cpp` の `rebuildThreadLoop` 内。**すべて `runtime == nullptr`（ポインタの null 判定）で失敗を検知し、`error` フィールドは toString でログ出力のみ**（`error != None` ではない）。

| サイト | 行 | パス | failure check | error 使用 | リトライ方針 |
|---|---|---|---|---|---|
| 1 | :941-948 | transient recovery | `recoveryResult.runtime == nullptr` | `toString()` ログのみ | `continue`（破棄） |
| 2 | :1015-1027 | durable recovery | `recoveryResult.runtime == nullptr` | `toString()` ログのみ | `settlePendingRecoveryAdmission(true)` → retry（上限: `kMaxRecoveryConsecutiveFailures=4`） |
| 3 | :1087-1098 | main rebuild task | `buildResult.runtime == nullptr` | `toString()` ログのみ | `continue`（破棄） |

validateWarmup() の 3 サイトも同様のパターン:

| サイト | 行 | パス | failure check | error 使用 | リトライ方針 |
|---|---|---|---|---|---|
| 1 | :955-971 | transient recovery warmup | `!= BuildError::None` | `toString()` ログ + destroyDSP | `continue`（破棄） |
| 2 | :1032-1049 | durable recovery warmup | `!= BuildError::None` | `toString()` ログ + destroyDSP + retry | `settlePendingRecoveryAdmission(true)` → retry（上限: `kMaxRecoveryConsecutiveFailures=4`） |
| 3 | :1143-1162 | main rebuild warmup | `!= BuildError::None` | `toString()` ログ + `shouldRetryWarmupFailure()` | `shouldRetryWarmupFailure(dsp)` → `isLoadingIR()` で判定。retryable なら `submitRebuildIntent(Structural)` |

`shouldRetryWarmupFailure()`（RebuildDispatch.cpp:78-81）: `return dsp.convolverRt().isLoadingIR();` — IR ロード中であれば retryable。

### 1.8.5 再試行方針（REPAIR_PLAN2-dash.md:1602-1613 との整合）

REPAIR_PLAN2-dash.md の BuildError × retry マトリクス（十八次別視点11 / 別視点13）と実コードの整合:

| BuildError | 分類 | retry 方針 (code) | 実装状況 |
|---|---|---|---|
| `InvalidInput` | **永続** | Discarded（state clear） | ✅ build(:435) で返り、caller は `continue` |
| `ResourceUnavailable` | **一時的** | Building → DurablePending（retry） | ✅ build(:461) で返り、durable recovery は retry（:1015-1048） |
| `WarmupFailed` | **一時的** | Building → DurablePending（retry） | ✅ validateWarmup(:476) で返り、main task は `shouldRetryWarmupFailure` で retryable 判定 |
| `InternalError` | **永続** | Discarded（state clear） | ✅ build(:466) で返り、caller は `continue` |
| `MKLFailure` | **一時的** (設計済み) | Building → DurablePending（retry） | ❌ enum のみ。build() では InternalError に丸められる |
| `ConvolverFailure` | **一時的** (2026-08-10 確定) | Building → DurablePending（retry） | ❌ enum のみ。build() では InternalError に丸められる |
| `PrepareFailure` | **一時的** | Building → DurablePending（retry） | ❌ enum のみ。build() では InternalError に丸められる |

**設計整合の問題**: REPAIR_PLAN2-dash.md:1602-1613 は MKLFailure/ConvolverFailure/PrepareFailure を**「一時的（retry）」** として設計しているが、実コードではこれらは**永続の InternalError に丸められており retry 対象外**になっている。保険分類として「一時的」を想定しているのに、実際の failure パスでは「永続（Discarded）」になっている**意味論の不一致**が存在する。将来 wiring する際はこの不一致を解消する必要がある。

### 1.8.6 コンパイル時安全機構の欠如

| メカニズム | 状態 | 備考 |
|---|---|---|
| `-Wswitch` / `-Wswitch-enum` | ❌ 未指定 | CMakeLists.txt:1264 `/W4` は MSVC C4061（enum switch 未網羅）を含むが、gcc/clang は `-Wall -Wextra`（:1327）のみで `-Wswitch-enum` は未指定 |
| `static_assert` enum↔toString 網羅性 | ❌ なし | RuntimeBuilder.cpp に static_assert は 0 件 |
| clang-tidy `enum-ast-visibility` / `switch-default` | ❌ OFF | CMakeLists.txt:40-57: `CONVOPEQ_ENABLE_CLANG_TIDY OFF`。CI 環境（`CONVO_CI_BUILD`）では ON だが、ローカル開発では無効 |
| `return "Unknown"` フォールバック | ⚠️ あるが `default:` なし | switch の外にフォールバック return はあるが、`default:` ラベルはない。新規 enum 値追加時、コンパイラは C4061 で警告する可能性があるが静的解析ツール（clang-tidy）未設定の環境では静静に "Unknown" が返る |
| **テストによる enum↔toString 一致検証** | ❌ **なし** | **すべてのテストファイルに BuildError/toString 参照ゼロ**（AiDex 検索: 0 ヒット）。ISRSemanticValidationTests.cpp（941行）は ISR closure/payload セマンティクスのみをテストし、BuildError は未テスト |

**work32 監査の不正確さ**（implementation_audit_report.md:144）:
> `BuildError` enum 拡張 | 既存の switch-case に `default` があるため互換性を維持。| 低

この主張は**不正確**。toString の switch には `default:` ラベルがない（`switch` の後に `return "Unknown"` があるだけ）。しかし実質的には「未処理値が暗黙に `Unknown` になる」ため、**ランタイム安全性は確保されているが、**静的検出はない。

### 1.8.7 現状 / 課題 / 動機

**現状**
- `MKLFailure` / `ConvolverFailure` / `PrepareFailure` は enum 定義 + `toString` のみで build パス未使用（保険分類）
- `build()` が実際に返すのは `InvalidInput`（:435）/ `ResourceUnavailable`（:461）/ `InternalError`（:466）/ `None`（implicit success）、`validateWarmup()` が返すのは `WarmupFailed`（:476）/ `None`（:478）
- work32 C-2 Step 3（catch 拡張）は**未実装**。`mkl::exception` は存在しない。MKL/IPP はステータスコードベースで例外をスローしない
- caller は `runtime == nullptr` で失敗検知。`error` フィードはログ出力のみ
- buildRuntimePublishWorld（メイン publish パス）は BuildError を全く使用しない

**課題 / 動機**
- 将来 convolver build / prepare 失敗を**適切に分類**（transient/DurablePending）して retry できるようにする
- 現状では `InternalError`（永続/Discarded）に丸められており、一時的な convolver/prepare 問題で正当な Recovery が喪失される（REPAIR_PLAN2-dash.md:1614 の「安全側は一時的」設計方針と矛盾）
- `toString` の enum 網羅性に対する**静的検証ギャップ**（テストなし・static_assert なし）

### 1.8.8 設計方針 — 将来 wiring の正しいアプローチ

#### 1.8.8.1 Catch-based アプローチは不適（work32 Step 3 は破棄すべき）

work32 plan:803-835 の Step 3 は `catch(const mkl::exception&) → MKLFailure` を提案するが、これは**実装不可能**：
1. `mkl::exception` は oneMKL C++ wrappers にのみ存在し、本プロジェクトは **C API**（`mkl.h`, `mkl_dfti.h`）を使用
2. MKL / IPP は **ステータスコード** (`IppStatus`, `DFTI_STATUS`) でエラーを返す。C++ 例外をスローしない
3. `ConvolverProcessor::prepareToPlay()` は `jassert` + ステータスコードで処理

#### 1.8.8.2 正しい wiring アプローチ — Return-code チェック

`build()` の try ブロック内で呼び出される各操作の**戻り値/ステータスをチェック**し、対応する BuildError を設定する:

```cpp
// 将来の build() 内 (案)
BuildResult result {};

if (in.sampleRate <= 0.0 || in.blockSize <= 0) {
    result.error = BuildError::InvalidInput;
    return result;
}

try {
    runtime = convo::aligned_make_unique<AudioEngine::DSPCore>();
    runtime->convolverRt().setVisualizationEnabled(false);
    runtime->convolverRt().applyBuildSnapshot(convolverBuildSnapshot);
    runtime->convolverRt().transferIRStateFrom(engine.getConvolverProcessor());

    // ★ 将来: prepare() の戻り値/ステータスをチェック
    // prepare() は現在 void だが、将来的に Status を返すように変更、
    // または prepare() 内で ConvolverFailure/PrepareFailure/MKLFailure を
    // internal ステータスとして設定し、build() に伝播する
    auto prepareStatus = runtime->prepareWithStatus(...);
    if (prepareStatus == PrepareStatus::ConvolverInitFailed) {
        result.error = BuildError::ConvolverFailure;
        return result;
    }
    if (prepareStatus == PrepareStatus::MKLFailed) {
        result.error = BuildError::MKLFailure;
        return result;
    }
    if (prepareStatus == PrepareStatus::ResourceExhausted) {
        result.error = BuildError::PrepareFailure;
        return result;
    }

    result.runtime = runtime.release();
    result.prepared = true;
    return result;
}
catch (const std::bad_alloc&) {
    result.error = BuildError::ResourceUnavailable;
    return result;
}
catch (...) {
    result.error = BuildError::InternalError;
    return result;
}
```

**前提条件**: `DSPCore::prepare()` をステータスコードを返すバージョンに変更する必要がある（現状は `void`）。また、`applyBuildSnapshot`/`transferIRStateFrom` にステータスチェックを追加する。

#### 1.8.8.3 Retry ポリシーの整合

将来 wiring する際は REPAIR_PLAN2-dash.md:1602-1613 の分類に合わせる:
- **Transient**（Building → DurablePending, retry）: `ResourceUnavailable`, `MKLFailure`, `ConvolverFailure`, `PrepareFailure`, `WarmupFailed`
- **Permanent**（Discarded）: `InvalidInput`, `InternalError`

現在の caller パターン（`runtime == nullptr` チェック）は**変更不要** — insurance 分類を wiring する場合でも、`prepare()`/`convolverRt()` 操作が失敗した場合は `runtime` が nullptr にならない可能性がある（部分的に初期化された DSPCore）。この場合は `result.error != None` もチェックする必要がある:

```cpp
// 将来の caller (案) — error チェックを追加
if (buildResult.runtime == nullptr || buildResult.error != BuildError::None) {
    diagLog("...build failed error=" + toString(buildResult.error));
    if (buildResult.error == BuildError::InvalidInput ||
        buildResult.error == BuildError::InternalError) {
        // permanent → discard
        continue;  // or settle(false)
    }
    // transient → retry (DurablePending)
    runtimePublicationBridge_.settlePendingRecoveryAdmission(true);
    continue;
}
```

### 1.8.9 実装手順

**フェーズ 1（今すぐ可能 — safety hardening）**
1. `toString()` に `default:` ケースを追加し、未知の enum 値を「検出可能」にする（`return "Unknown"` を `default: return "Unknown-!!";` などの目印付きに変更）
2. `static_assert` による enum↔toString 網羅性のコンパイル時検証を追加（詳細は 1.8.10）
3. **テスト追加**: `BuildErrorToStringTests.cpp` — 全 8 enum 値の `toString()` 戻り値を検証

**フェーズ 2（将来 wiring — insurance 分類の活用）**
4. `DSPCore::prepare()` の戻り値を `void` → ステータス型（例: `PrepareStatus`）に変更
5. `ConvolverProcessor::applyBuildSnapshot()` と `transferIRStateFrom()` にステータス/エラー伝播を追加
6. build() の try ブロック内でステータスチェック → `MKLFailure`/`ConvolverFailure`/`PrepareFailure` をセット
7. caller（RebuildDispatch.cpp:941/1015/1087）の failure check を `runtime == nullptr || error != None` に強化
8. retry policy の適用: transient は DurablePending, permanent は Discarded
9. `buildErrorCount_` telemetry カウンタを AudioEngine に追加（work32 Step 4）

### 1.8.10 static_assert による enum↔toString 網羅性検証

toString に対するコンパイル時網羅性検証の実装案（RuntimeBuilder.cpp の `toString` の後ろに追加）:

```cpp
// ★ C-2: toString の enum 網羅性をコンパイル時検証
// 新規 enum 値を追加した場合、この static_assert がコンパイルエラーになる。
// （switch は default を持たないため、新規値は "Unknown" に落ちる静的デグレードを防止）
#define BUILD_ERROR_ENUM_COUNT 8
static_assert(static_cast<int>(BuildError::None) + 1 == 1, "");  // 0
static_assert(static_cast<int>(BuildError::InvalidInput) + 1 == 2, "");
static_assert(static_cast<int>(BuildError::ResourceUnavailable) + 1 == 3, "");
static_assert(static_cast<int>(BuildError::MKLFailure) + 1 == 4, "");
static_assert(static_cast<int>(BuildError::ConvolverFailure) + 1 == 5, "");
static_assert(static_cast<int>(BuildError::PrepareFailure) + 1 == 6, "");
static_assert(static_cast<int>(BuildError::WarmupFailed) + 1 == 7, "");
static_assert(static_cast<int>(BuildError::InternalError) + 1 == 8, "");
static_assert(static_cast<int>(BuildError::InternalError) == BUILD_ERROR_ENUM_COUNT - 1,
              "BuildError enum count mismatch — update toString switch");
```

### 1.8.11 Test 設計

**現状**: テストファイルに `BuildError` または `toString` の参照は**ゼロ**（AiDex 検索: 0 ヒット）。ISRSemanticValidationTests.cpp（941行）は ISR closure/payload セマンティクスのみをテスト。

**提案するテストファイル**: `src/tests/BuildErrorClassificationTests.cpp`

```cpp
#include <catch2/catch.hpp>  // or JUCE UnitTest
#include "RuntimeBuilder.h"

TEST_CASE("BuildError::toString covers all enum values") {
    using namespace convo;
    REQUIRE(std::string(toString(BuildError::None)) == "None");
    REQUIRE(std::string(toString(BuildError::InvalidInput)) == "InvalidInput");
    REQUIRE(std::string(toString(BuildError::ResourceUnavailable)) == "ResourceUnavailable");
    REQUIRE(std::string(toString(BuildError::MKLFailure)) == "MKLFailure");
    REQUIRE(std::string(toString(BuildError::ConvolverFailure)) == "ConvolverFailure");
    REQUIRE(std::string(toString(BuildError::PrepareFailure)) == "PrepareFailure");
    REQUIRE(std::string(toString(BuildError::WarmupFailed)) == "WarmupFailed");
    REQUIRE(std::string(toString(BuildError::InternalError)) == "InternalError");
}
```

**Acceptance Criteria**
- `BuildError::toString` が全 8 enum 値に対応済み ✅（実コード照合済み）
- 新規 enum 値追加時にコンパイルエラーになる `static_assert` を追加する
- `BuildErrorClassificationTests.cpp` で toString 一貫性をテストする
- 既存 ctest 28/28 PASS（影響なし）

**影響範囲 / リスク**
- **フェーズ 1**: RuntimeBuilder.cpp (static_assert 追加), 新規テストファイル — 影響: なし。リスク: **低**
- **フェーズ 2**: RuntimeBuilder.cpp (build() の catch 拡張), AudioEngine.h (prepare ステータス), AudioEngine.Processing.DSPCoreLifecycle.cpp (prepare ステータス返却), ConvolverProcessor.h/cpp (ステータス伝播), RebuildDispatch.cpp (caller failure check 強化), AudioEngine.h (buildErrorCount_ 追加) — 影響: **中**（error handling パスの変更）。リスク: **中**（retry ポリシーの不整合による無限リトライ / 回復不能エラーの永久リトライ）

### 1.8.12 未解決課題 / 今後の調査

| # | 課題 | 調査方法 | ステータス |
|---|---|---|---|
| 1 | `DSPCore::prepare()` 内部の convolver/eq/dither/oversampling 各サブシステムの失敗モードを列挙 | ConvolverProcessor.Lifecycle.cpp:211, AudioEngine.Processing.DSPCoreLifecycle.cpp:72 を解析 | **要調査** — prepare のステータス型設計に必要 |
| 2 | `applyBuildSnapshot()` / `transferIRStateFrom()` の失敗を検知する方法 | ConvolverProcessor.StateAndUI.cpp:271, ConvolverProcessor.h:1132 を解析 | **要調査** |
| 3 | `MKLNonUniformConvolver` が MKL を使用する箇所とステータスコードの取得方法 | MKLNonUniformConvolver.cpp/h を解析 | **要調査** — MKLFailure wiring に必要 |
| 4 | `publishRuntimeProcessSnapshot()` の戻り値/エラー伝播 | ConvolverProcessor.StateAndUI.cpp:286 付近 | **要調査** |
| 5 | `buildRuntimePublishWorld` の failure ハンドリング — なぜ BuildError を使わないのか | buildRuntimePublishWorld (cpp:178-426) の設計意図を確認 | **要調査** — build() vs buildRuntimePublishWorld() の責務分離 |
| 6 | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` が有効な場合の diagLog パス | RebuildDispatch.cpp:888,1073,1104,1138,1213 | 現在 build() は `diagLog` を使用しない（build() には `#if CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS` ガードなし） |

---

## 1.9 初回 publish 前 quarantine の無駄な起床の最適化

**現状（実コード照合）**
- 初回 publish 前の quarantine で `submitRecoveryIntent` が `recoveryPending = true` + `notify` を実行し、**無意味な起床**が発生（RebuildDispatch / RuntimePublicationCoordinator）

**課題 / 動機**
- 機能上の問題はないが、**無意味な RebuildThread 起床**による CPU 無駄（Phase 5 最適化候補）

**設計方針**
- `submitRecoveryIntent` が**実際に durable recovery が存在する場合のみ** `recoveryPending` を set + notify するように変更

**実装手順**
1. `submitRecoveryIntent` の recoveryPending set 条件を精査（durable admission の有無を確認）
2. durable recovery が存在しない場合の early return を追加
3. テスト: 無意味な起床の削減 / 機能回帰なし（既存 Recovery テスト）

**Acceptance Criteria**
- 初回 publish 前の無意味な起床が発生しない（recoveryPending の誤 set なし）
- 既存 Recovery / soak テスト全 PASS

**影響範囲 / リスク**
- 影響: `submitRecoveryIntent`、RebuildDispatch
- リスク: **低**（最適化のみ・機能変更なし）

---

# 2. 一部実装（保留・部分対応）

## 2.1 R4: retire 順序逆転の完全解消

**現状（実コード照合）**
- **runtime 経路は対応済み**: `retireDSPHandleForRuntime` / `retireByHandle` は `requestReclaimHandle` 経由に一本化
- `shutdownReclaim` は **X3-R4 Phase 7 で完全削除済み**（AC-R4-1 call sites==0 / AC-R4-2 symbol absent）
- `reclaim(ReclaimMode, ...)` は RuntimeEBR / ShutdownQuiescent の2モードで一本化（ReclaimAuthority）

**残る課題**
- **retire 順序逆転**（後から発行された retire が先に処理される可能性）の**完全解消は保留**
- ただし quarantine fallback で **UAF / リークは排除済み**

**設計方針**
- R4 詳細設計（REPAIR_PLAN2-dash.md §6.3 末尾・Phase 0-7）に基づき、retire の epoch 順序保証（FIFO）を強化

**実装手順**
1. retire の epoch 順序保証（FIFO）を強化（enqueue 順と epoch 順の整合）
2. `drainDeferredRetireQueues` の順序検証（pendingReclaimHandles_ の再試行機構と整合）
3. テスト追加: retire 順序逆転の回帰テスト（AC-R4-T1〜T7 を拡張）

**Acceptance Criteria**
- retire が epoch 順に処理される（順序逆転なし）
- quarantine fallback でリークなし（既存）
- **AC-R4-1〜10 全充足**（shutdownReclaim 0 / ReclaimAuthority 一本化 / epoch unsafe は pendingReclaimHandles_ に残る / Faulted で pending を clear しない）

**影響範囲 / リスク**
- 影響: `ISRRetireRouter`、`AudioEngine.Retire.cpp`（drainDeferredRetireQueues）、`ReleaseResources.cpp`
- リスク: **中**（retire 順序の変更は epoch 安全性に影響。runtime 経路は既に対応済みのため、残余は順序保証の強化）

---

## 2.2 shutdown lifetime contract の明文化

**現状**
- REPAIR_PLAN2-dash.md §4（:837-846）で「shutdown lifetime contract の明文化」を**将来タスクとして記録**
- R4 詳細設計で **ShutdownQuiescenceProof**（admissionClosed / producersJoined / coordinatorStopped / builderStopped / audioStopped / readerRegistrationClosed / readersZero / epochSettled）が**答えとして確定済み**
- `readerRegistrationClosed`（EpochDomain.h）は実装済み（X3）・`reclaim(ShutdownQuiescent)` の precondition は実装済み

**課題 / 動機**
- shutdown 側の「本当に reader が存在しないこと」の**保証をコード上の契約として明文化**
- AC-X3-11〜18 を満たす（ReclaimPermit の生成を ShutdownRuntime のみに限定等）

**設計方針**
```
ShutdownQuiescenceProof を独立オブジェクト化（valid() は完全条件）
ReclaimPermit は ShutdownRuntime のみが生成（caller cannot manufacture — AC-X3-11）
shutdownPhase >= Destroy は ShutdownQuiescent reclaim の証明にならない（AC-X3-12）
```

**実装手順**
1. `ShutdownQuiescenceProof` の `valid()` を完全条件（上記8項目）で実装
2. `ReclaimPermit` の生成を ShutdownRuntime のみに限定（friend class・AC-X3-11）
3. `registerReaderThread()` が閉鎖後 -1 を返す（INV-X3-4・AC-X3-15）— 実装済み
4. CacheMap / ReleaseResources は ShutdownRuntime の Permit 経由で reclaim（AC-X3-12）
5. テスト: AC-R4-T1〜T7 / AC-X3-11〜18

**Acceptance Criteria**
- **AC-X3-11〜18 全充足**（ReclaimPermit 生成限定 / phase 依存禁止 / readerRegistrationClosed 証明 / pendingReclaimHandles empty は producer join 後 / registerReaderThread 失敗 / 二重 retire 防止 / Audio Thread から不可 / physical delete と独立）
- `isFullyDrained` が proof と整合

**影響範囲 / リスク**
- 影響: `ShutdownRuntime`、`RuntimeIntentCoordinator`、`CacheMap`（AudioEngine.h）、`ReleaseResources.cpp`
- リスク: **中**（shutdown 意味論の厳密化。**readerRegistrationClosed は実装済み**のため、残余は Proof オブジェクトの完全化と Permit 導入）

---

# 3. 実装順序（推奨）

```
Phase A（低リスク・独立）:  1.9（無駄な起床最適化）→ 1.8-F1（static_assert + toString test）→ 1.8（保険分類・現状維持）→ 1.3（LinearRamp 文書化）
  ※ 1.8-F1 = フェーズ1 safety hardening（static_assert + test）。1.8 本体（insurance wiring）は Phase B 以降。
Phase B（中リスク）:        2.2（shutdown lifetime contract 明文化・Proof 完全化）→ 1.4（isFullyDrained 上書き全廃）
Phase C（高リスク・大規模）: 2.1（R4 retire 順序逆転の完全解消）→ 1.1（R1 MPSC 化）
Phase D（将来・任意）:      1.5+1.6（sparse completion + テスト）→ 1.7（X4-B 案2 currentWorld_ 廃止）
Phase E（別タスク P3）:     1.2（Recovery coalesce）
```

- **各 Phase でビルド・ctest を通過**させる（rollback point 確保）
- Phase C 以降は**高い影響範囲**のため、実施前に現行 REPAIR_PLAN2-dash.md の該当設計（§5 R1/R4・§6.2 sparse・§6.4-X4 案2）を再確認する

---

# 4. 参考: 実装済みだが計画書表記が古い項目（REPAIR_PLAN2-dash.md に 2026-08-11 追記済み）

| 項目 | 計画書の古い表記 | 実際の状態 |
|---|---|---|
| §4.5（bootstrap jassert） | 「実装は別途」 | ✅ 実装済み（:988 に追記） |
| X1〜X6 | 「実装未着手」 | ✅ 全実装済み（:1061 に追記） |
| X4 read API | 「未実装（X4-B-9 新設予定）」 | ✅ 実装済み（:1066 等に追記） |
| §4.4（BlockDouble finalizeCrossfadeMixPath） | 「対象外・実測後判断」 | ✅ R6 で対応済み（`finalizeCrossfadeMixPath(dsp, fading, true)`）※計画書は対象外表記のまま |
| shutdownReclaim 二系統 | 「別タスク」 | ✅ X3-R4 Phase 7 で削除済み |
| EpochDomain readerRegistrationClosed | 「ガード未実装」 | ✅ 実装済み（:1958 に追記） |

---

## 補足
- 本ドキュメント（dash2）の項目は、いずれも**設計・アーキテクチャ上の将来拡張・最適化**であり、現在の ISR パイプラインの正しさ（ctest 28/28・CI 検証 PASS）には影響しません。
- 各項目の実装着手時は、本ドキュメントの Acceptance Criteria をテストとして固定してから実施してください（REPAIR_PLAN2-dash.md の Phase 0 方針と同一）。
