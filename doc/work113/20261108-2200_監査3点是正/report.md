# 外部監査 3 点の残課題に対する調査・是正

- 実施: 2026-11-08 22:00
- 基準リビジョン: `main` = `06b0bcbe`（V0.6.12）＋本作業
- 対象: Q2/Q5/Q6（Shutdown Quiescence）、`loaderGeneration_`、`FrozenRuntimeWorld`、関連ドキュメント
- 方針: 監査レポートやコメントを根拠にせず、現行ソース・呼出経路・ビルド構成・テスト登録のみで判断した

## 1. 結論

対象 4 項目すべてに、実装または契約の欠陥があり、いずれも是正した。中心は「証明の条件が caller の自己申告に依存していた」点である。Q2 は authority 側の join 記録との AND に、Q5 は観測束（新設の `EpochQuiescenceEvidence`）と generation 一致の検証に、Q6 は authority 内部カウンタの直接参照に置き換えた。あわせて reclaim 認可の失敗を Release ビルドでも観測できるようにした。検証の過程で、ハーネスだけ `JUCE_DSP_USE_INTEL_MKL` が未定義であることによる allocator の ODR 不整合（既存バグ）も見つかったため、併せて修正した。

| 項目 | 変更前 | 変更後 | 残る限界 |
|---|---|---|---|
| Q2 AllProducersJoined | `obs.allProducersJoined = true` 固定 | RebuildThread / CoordinatorLoop の join 実測 **AND** `joinProducers()` 成功（`producersJoined_`） | 観測と authority の二重化で、片方だけの偽装は防げるが両方の同時偽装は型では防げない |
| Q5 EpochSettled | `obs.epochSettled = true` 固定。証拠型は存在しなかった | `EpochQuiescenceEvidence`（7 フィールド）を実観測で生成し、`quiescent()` と generation 一致を authority が検証。Proof に保存 | epoch 前進の停止そのものは証明しない（publisher 不在が前提） |
| Q6 postStopEnqueue == 0 | `obs.postStopEnqueueZero = true` 固定。カウンタは private で読み出し不可 | `postStopEnqueueCount()` を公開し、`tryMakeQuiescenceProof` が内部で判定（observation から削除） | 0 超で Faulted へ昇格する仕様（H.11.11.4）は未実装 |
| `loaderGeneration_` | 宣言と destructor の加算のみ。読み出し 0 | 削除。currency は `convolverStateGeneration`、寿命は join + WeakReference が担保すると明文化 | 実 LoaderThread の競合を engine レベルで再現するテストは未追加 |
| `FrozenRuntimeWorld` | `releaseState()` の位置づけが不明瞭（seal を前提と読める記述） | release は所有権移譲のみ、seal は publish transaction 内、強制点は commit と明文化（挙動不変） | `RuntimeState` は public フィールドのため、seal 中の直接書き込みは型で禁止できない |
| アロケータ選択（調査中に発見） | ハーネスだけ `JUCE_DSP_USE_INTEL_MKL` 未定義。`DiagnosticsConfig.h` が `defined()` 判定のため、TU ごとに mkl / system が混在（ODR 違反 → AV） | ハーネスに `JUCE_DSP_USE_INTEL_MKL=1` を付与し、判定を値ベースへ修正 | 他の将来ターゲットが同種の定義漏れをしない保証は CMake のレビュー依存 |
| 失敗の観測性 | `jassert(reclaimed)` のみ（Release で消える） | `noteReclaimProofFailure()` の累計を `collectResult` と `shutdown_trace.json` に出力 | 失敗時の追加動作（再試行・Faulted 遷移）は未定義のまま |

## 2. 調査結果

### 2.1 Q2（AllProducersJoined）

**固定 true の代入があった。** `AudioEngine.h` の `tryShutdownQuiescentReclaim` が `obs.allProducersJoined = true` を代入しており、producer の join 状態を一切観測していなかった。`ShutdownRuntime::tryMakeQuiescenceProof` も observation の bool をそのまま Q2 に採用していたため、join が未完了でも Proof が成立し得る構造だった。

根拠となる実装は次のとおり。

- `AudioEngine.Processing.ReleaseResources.cpp:237` で `shutdownCoordinatorLoop()`、`:238` で `stopRebuildThread()`、`:250` で `joinProducers()` の再試行ループ。順序としては join 済みだが、証明側はその事実を参照していなかった。
- `AudioEngine.RebuildDispatch.cpp:815-836` の `stopRebuildThread()` は `join()` の後にのみ `rebuildThreadStopped` を publish する（R8 の契約）。
- `ISRCoordinatorLoop.cpp:24` の `stopLoop()` は `stopThread(2000)` の結果を捨てていた。

発生条件は「shutdown 経路で producer が join し切っていない場合」だが、変更前は条件自体が存在せず常に成立扱いだった。したがって quiescence 証明の 1 条件が無条件に真という状態で、安全性の主張としては過大だった。

**是正。** 二重の evidence にした。呼出側は `rebuildThreadStopped`（join 後 publish）と新設の `coordinatorLoopStopped`（`stopLoop()` の join 結果）を実測し、`ShutdownRuntime` 側は `joinProducers()` の Closing→Closed 成功時のみ `producersJoined_` を立てて、両者の AND で Q2 を判定する。`closeAdmission()` は新しい shutdown transaction の開始点として `producersJoined_` を false に戻す。片方だけの自己申告では Proof が出ない。

### 2.2 Q5（EpochSettled）

**証拠型そのものが無かった。** `obs.epochSettled = true` の固定代入で、`EpochQuiescenceEvidence` という型はリポジトリに存在しなかった（コメント内の言及のみ）。H.11.13.3 が要求する「epoch / reader registration / active reader を一つの transaction で観測する」構造は未実装で、Q5 は事実上「shutdown が進んだから true」という仮定だった。

**是正。** `ISRLifetimeProof.h` に `EpochQuiescenceEvidence` を追加した。フィールドは `captured` / `epoch` / `epochGeneration` / `readerRegistrationGeneration` / `readerRegistrationClosed` / `activeReaders` / `minReaderEpoch` で、`quiescent()` が「観測済み + 登録閉鎖 + active reader ゼロ」を要求する。

観測点は `AudioEngine::captureEpochQuiescenceEvidence()`（`AudioEngine.h`）。`m_epochDomain` と `m_retireRouter` から取得し、router が無い場合は `captured=false` のまま返して Q5 を成立させない（Q4 の null 既定より厳しい）。`tryMakeQuiescenceProof` は `quiescent()` に加えて、evidence の generation が identity に束縛する `observation.epochGeneration` / `readerRegistrationGeneration` と一致することを要求する。成立した evidence は Proof に保存し、`epochEvidence()` で事後監査できる。

### 2.3 Q6（postStopEnqueue == 0）

**カウンタに読み出し口が無かった。** `obs.postStopEnqueueZero = true` の固定代入。実カウンタ `sh6PostStopEnqueueCount_` は `ISRShutdown.h` の private にあり、`markPostStopEnqueue()` で増えるが、公開アクセサが無く Proof から参照できなかった。

**是正。** `postStopEnqueueCount()` を公開し、`tryMakeQuiescenceProof` が `postStopEnqueueCount() == 0` を直接判定する。observation の `postStopEnqueueZero` フィールドは削除した（caller が値を操作する経路を無くすため）。これで Q6 は「caller の申告」ではなく「authority が持つ唯一のカウンタ」で決まる。

### 2.4 失敗時の動作と観測性

**失敗が Release ビルドで消えていた。** `ReleaseResources.cpp` の reclaim 呼び出しは `jassert(reclaimed)` で結果を捨てていた。Release ビルドでは `jassert` が消えるため、Proof 不成立や Permit 拒否が記録に残らなかった。

**是正。** `ShutdownRuntime::noteReclaimProofFailure()` を追加し、`active` / `fading` の両経路で失敗を計上する。`ShutdownResult::reclaimProofFailureCount` と `shutdown_trace.json` の `reclaimProofFailureCount` に出力する。動作は従来どおり継続する。物理破棄は `DSPLifetimeManager::retireDetachedCoreForShutdown` が handle 台帳経由で行うため、reclaim を skip しても UAF にはならない（`DSPLifetimeManager.cpp:139-165` を確認）。

### 2.5 `loaderGeneration_`

**死んだ状態だった。** 全参照は `ConvolverProcessor.h` の宣言と `ConvolverProcessor.Lifecycle.cpp` の destructor での加算のみで、読み出しが 1 箇所も無かった。destructor のコメント「Generation を進めて全 LoaderThread を無効化」は成立しておらず、owner-validity を何も保証していなかった。

実際の契約は別に成立している。

- currency: `ConvolverProcessor.LoadPipeline.cpp:59` / `:253` の `bumpGeneration()` と、`:223`（キャッシュ削除判定）・`:297`（変換キャンセル）・`:404`（`applyComputedIR`）・`:827`（`executePendingCommit`）の `isCurrentGeneration`、`ConvolverProcessor.StateAndUI.cpp:1078`（state 更新）、`LoaderThread.cpp:642-654`（リサンプル）。
- owner lifetime: `forceCleanup()` の `stopThread(500)`（`StateAndUI.cpp:1045-1052`）と、`callAsync` の `WeakReference`。

**是正。** `loaderGeneration_` を削除し、destructor とメンバーのコメントを実際の機構に置き換えた。`LoaderThread.cpp` の B-6 コメントも、ループ内に世代チェックが無いこと・正当性は commit 直前ガードが持つこと・旧コメントの参照先 `REPAIR_PLAN` は現存しないことを明記した。

### 2.6 `FrozenRuntimeWorld`

**契約の記述が実装と逆だった。** ヘッダーと ARCHITECTURE は「Builder 境界で freeze された world だけが publish に進む」と読める記述になっていたが、実経路では `releaseState()` の時点で world は未 seal で、seal は publish transaction の中で行われる。実際に `releaseState()` へ「seal 済みのみ移譲」という前提条件を入れて検証したところ、D12-1-C（正当な dither 変更の publish）が失敗した。`[TEMP] releaseState sealed=0` の実測で、publish 経路の最初の release が未 seal であることを確認した。

不変性保証を 4 層に分けた実態は次のとおり。

1. 型レベル: `get()` / `operator*` / `operator->` は const のみ。ただし `releaseState()` が非 const ポインタを返すため、保証は移譲まで。
2. 規約 + 実行時ゲート: publish transaction 内で seal し、commit 側（`AudioEngine.Commit.cpp:242` の `isFrozen()` と `:247` の `isSealedRecursively()`）が検証する。未 seal の world は publish で拒否される。
3. 実行時 sealing: `SealedObject` のフラグは論理的な不変条件で、メモリ保護ではない。`RuntimeState` のフィールドは public で、seal 中の直接書き込みを型で止められない。唯一の意図的な seal 後書き込みは `RuntimeWorldAuthority::publish` の publication metadata bake（seal-before-bake 順序）。
4. テスト: release 順序（未 seal でも移譲されること、移譲後の seal が効くこと）を D12-6 で固定した。

**是正。** `releaseState()` を `.cpp` に移し、**挙動は変えず**に「所有権移譲のみ・seal は publish transaction 側」という契約をコメントと ARCHITECTURE に明記した。前提条件を追加する案は実測で正当な publish を壊すことが分かったため採用しない。代わりに、同じ変更を将来入れようとした場合に D12-1-C と D12-6 が落ちる形で回帰を検出できるようにした。

### 2.7 ドキュメント

陳腐化していたのは `ARCHITECTURE.md:212`（「Currently type-only」）、`:236`（「Phase-4 frozen world concept」）、`:585`（Layer 8 の型説明）、`ISRLifetimeProof.h` の冒頭、`ISRShutdown.h:252-254`（「型のみ・production 未接続」）、`LoaderThread.cpp` の B-6、`ConvolverProcessor.h` の `loaderGeneration_` 説明、`ConvolverProcessor.Lifecycle.cpp` の destructor コメント。すべて現行実装に合わせた。

`evidence/D101-26-*.md` などは当時の状態を記録した歴史的証跡であり、遡って書き換えない（判断として明記する）。`.auto/` 配下の生成物も同様に対象外とした。

### 2.8 ビルド構成の不整合（D12-6 の検証中に発見）

**既存の不整合があった。** ハーネスで `RuntimeState` を破棄すると 0xC0000005 で落ちた。D12-6 を実装して初めて、`FrozenRuntimeWorld` が所有権を保持したまま破棄する経路が実行されたことで表面化した。

切り分けは次のとおり。

- `RuntimeState::createForTest()`（`std::make_unique` + `delete`）は正常。
- `createForBuilder()`（aligned）+ 明示デストラクタ + `aligned_free` で再現。`sizeof` / `alignof` は両 TU で一致（1256 / 8）。
- 同一 TU 内で確保→破棄しても再現し、別 TU では再現しない（TU 依存）。

原因は `JUCE_DSP_USE_INTEL_MKL` の定義状態の不一致だった。`build/CMakeFiles/impl-Release.ninja` の `DEFINES` を比較すると、`ConvoPeq` は `-DJUCE_DSP_USE_INTEL_MKL=1` を持つが `AudioEngineHarness` は持たない。`JUCE/modules/juce_dsp/juce_dsp.h` は未定義時に `#define JUCE_DSP_USE_INTEL_MKL 0` を行うため、`AudioEngine.h` を先に include する TU では「定義済み（値 0）」、`AlignedAllocation.h` を先に include する `FrozenRuntimeWorld.cpp` では「未定義」になる。`DiagnosticsConfig.h` が `#if defined(JUCE_DSP_USE_INTEL_MKL)` で分岐していたため、前者は `mkl_malloc`/`mkl_free`、後者は `system_aligned_malloc`/`_aligned_free` を選び、inline の `convo::aligned_free` が TU 間で異なる定義になった（ODR 違反）。リンカが一方を選ぶため、確保と解放のペアが食い違って AV になる。

**是正。** ハーネスへ `JUCE_DSP_USE_INTEL_MKL=1` を追加し（アプリ・ISR テストと同一条件）、`DiagnosticsConfig.h` の判定を `#if defined(...) && JUCE_DSP_USE_INTEL_MKL` に変更した。アプリは 1 定義のため挙動は変わらない。非 MKL ビルドでは従来 mkl 経路に入り得たため、この修正で一貫する。

この不整合はハーネス全体の停止（`[EQ_DIRECT]` 付近の AV）と同じ原因であり、修正後に既定スイートが先へ進むことを確認した（4.2 参照）。

## 3. 変更内容

### 3.1 変更ファイル

| 種別 | ファイル | 内容 |
|---|---|---|
| 本番 | `src/audioengine/ISRLifetimeProof.h` | `EpochQuiescenceEvidence` 追加、Proof に evidence 保存、冒頭コメント更新 |
| 本番 | `src/audioengine/ISRShutdown.h` | observation から Q5 bool / Q6 bool を削除、`producersJoined` / `postStopEnqueueCount` / `noteReclaimProofFailure` / `reclaimProofFailureCount` 追加、`ShutdownResult` に失敗回数 |
| 本番 | `src/audioengine/ISRShutdown.cpp` | Q2/Q5/Q6 の判定、join/close での evidence 更新、accessor、collectResult と trace 出力 |
| 本番 | `src/audioengine/AudioEngine.h` | Q2 の実測、Q5 evidence の観測点、G19/G20 を evidence と同一スナップショットに、`coordinatorLoopStopped` |
| 本番 | `src/audioengine/AudioEngine.Threading.cpp` | CoordinatorLoop の join 結果を publish |
| 本番 | `src/audioengine/ISRCoordinatorLoop.{h,cpp}` | `stopLoop()` が join 成否を返す |
| 本番 | `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` | reclaim 失敗の計上と診断ログ |
| 本番 | `src/audioengine/FrozenRuntimeWorld.{h,cpp}` | `releaseState()` を .cpp へ移し、release / seal / commit の順序契約を明文化（挙動不変） |
| 本番 | `src/ConvolverProcessor.h` / `src/convolver/ConvolverProcessor.Lifecycle.cpp` | `loaderGeneration_` 削除とコメント是正 |
| 本番 | `src/convolver/ConvolverProcessor.LoaderThread.cpp` | B-6 コメント是正 |
| 本番 | `src/DiagnosticsConfig.h` | allocator 選択を「定義の有無」から「値」へ修正（ODR 回避） |
| テスト | `src/tests/invariant_INV3_INV5.cpp` | Q2/Q5/Q6 の実状態テスト、失敗観測テスト、共通 observation ビルダー |
| テスト | `src/tests/AudioEngineHarness/STG11D12StateIntegerGuardTests.cpp` | D12-6（release 順序と seal の強制点） |
| テスト | `src/tests/LoaderCurrencyContractTests.cpp`（新規） | GenerationManager の stale 契約 |
| ビルド | `CMakeLists.txt` | `LoaderCurrencyContractTests` ターゲットと `add_test`、`AudioEngineHarness` への `JUCE_DSP_USE_INTEL_MKL=1`（MKL 検出時のみ） |
| テスト | `src/tests/AudioEngineHarness/PublishPipelineIntegrationTests.cpp` | 限定実行入口（`CONVOPEQ_STG11D12_ONLY` / `CONVOPEQ_D12_FROZEN_ONLY`） |
| 文書 | `ARCHITECTURE.md` | 3 箇所の記述を実装に合わせて更新 |

### 3.2 変更したクラス・関数

- `convo::isr::EpochQuiescenceEvidence`（新規）
- `convo::isr::ShutdownQuiescenceProof::epochEvidence()`
- `convo::isr::ShutdownRuntime::producersJoined()` / `postStopEnqueueCount()` / `noteReclaimProofFailure()` / `reclaimProofFailureCount()`
- `convo::isr::ShutdownRuntime::tryMakeQuiescenceProof()`（Q2/Q5/Q6 の判定）
- `convo::isr::ShutdownRuntime::joinProducers()` / `closeAdmission()`（evidence の更新）
- `convo::isr::CoordinatorLoop::stopLoop()`（戻り値）
- `AudioEngine::captureEpochQuiescenceEvidence()` / `tryShutdownQuiescentReclaim()` / `shutdownCoordinatorLoop()` / `startCoordinatorLoop()`
- `convo::FrozenRuntimeWorld::releaseState()`

### 3.3 削除した状態・処理

- `ConvolverProcessor::loaderGeneration_`（宣言・destructor の加算・コメント）
- `QuiescenceObservation::epochSettled`（bool）
- `QuiescenceObservation::postStopEnqueueZero`（bool）
- `ReleaseResources` の `jassert(reclaimed)` / `juce::ignoreUnused(reclaimed)`

## 4. テスト・ビルド

### 4.1 追加・変更したテスト

| テスト | 検証内容 | 実状態の構成方法 |
|---|---|---|
| `testQ2ProducerJoinAuthorityGate` | join 前は Proof が出ない / join 後は出る / join 後でも観測 false なら出ない | 実 `ShutdownRuntime` の FSM 遷移（closeAdmission → joinProducers） |
| `testQ5EpochEvidenceRealDomain` | reader active では Proof が出ない / exit + registration close + epoch 前進で成立 / evidence と observation の矛盾は拒否 | 実 `EpochDomain` に reader を登録・enter・exit し、実 `ISRRetireRouter` から activeReaders を取得 |
| `testQ6PostStopEnqueueBlocksProof` | 0 件なら成立 / `markPostStopEnqueue()` 後は不成立 / Permit も発行できない | 実カウンタを実 API で増やす |
| `testReclaimProofFailureObservability` | 失敗回数が accessor と `collectResult` に反映 | 実メソッド呼び出し |
| `checkD12T6FrozenWorldReleaseOrder` | 未 seal でも release できる（実経路の順序）/ 移譲後の seal が効く / seal 済みも移譲できる / 二重 seal は冪等 | 実 `RuntimeState`（aligned 確保、Builder と同じ経路） |
| `LoaderCurrencyContractTests`（3 ケース） | bump で旧世代が無効化 / 並行 bump で勝者は 1 つ / 実行中タスクが新要求で stale になる | 実 `GenerationManager` |

既存テストは observation の型変更に追随させた（`makeQuiescentObservation` ヘルパーへ集約）。Q5 の否定ケースは「bool を false にする」形式を廃止し、evidence の欠測・reader 残存・generation 不一致で構成している。

### 4.2 実行結果

| 対象 | 実行方法 | 結果 |
|---|---|---|
| `invariant_INV3_INV5Tests` | `build\Release\invariant_INV3_INV5Tests.exe` | ALL TESTS PASSED（exit 0） |
| `LoaderCurrencyContractTests` | `build\Release\LoaderCurrencyContractTests.exe` | 3 ケース PASS（exit 0） |
| `AudioEngineHarness`（D12 限定） | `CONVOPEQ_STG11D12_ONLY=1` | D12-1-A/B/C・D12-3・D12-4・D12-6 PASS（exit 0） |
| `AudioEngineHarness`（D12-6 限定） | `CONVOPEQ_D12_FROZEN_ONLY=1` | PASS（exit 0） |
| `AudioEngineHarness`（全体） | `ctest --test-dir build -C Release -R ^AudioEngineHarness$` | **1/1 Passed（321.82 秒）** |
| `ConvoPeq`（MSVC Release） | `cmake --build build --config Release --target ConvoPeq` | 成功（exit 0） |

補足が 2 点ある。

第一に、アロケータ修正の前はハーネスの既定スイートが 26 行で 0xC0000005 に落ちていた。修正後は 726 行まで完走し、ctest でも pass した。今回の調査で見つかった既存不具合であり、是正対象 4 項目とは独立に修正した。

第二に、exe を直接実行すると `CONVOPEQ_SOURCE_ROOT` が未設定のため D18-D（ソース構造チェック）だけが FAIL する。ctest は `ENVIRONMENT "CONVOPEQ_SOURCE_ROOT=..."` を設定するため pass する。テスト側の環境依存であり、今回の変更による回帰ではない（ログに `D18-D: CONVOPEQ_SOURCE_ROOT is not set` を確認）。

### 4.3 未実施の検証

- 実 `LoaderThread` を走らせた非同期競合テスト（engine 起動と MKL 初期化が必要なため、機構テストと静的検証で代替）。
- `FrozenRuntimeWorld` の publish 後の変更検出テスト（型で禁止できないため、commit 時の seal 検証と契約記述で代替）。
- 実機のシャットダウンでの `reclaimProofFailureCount` 採取（`shutdown_trace.json` の実測は未実施）。
- allocator の ODR 不整合そのものを検出するテストは無い。症状（D12-6 の破棄とハーネス全体の完走）で間接的に確認した。

## 5. 残課題

| 項目 | 内容 | 理由 | 安全性への影響 | 追加作業 |
|---|---|---|---|---|
| Q3/Q4 の authority 直読み | Q3（registration closed）と Q4（active readers）は依然 observation 経由 | `EpochDomain` / `ISRRetireRouter` は AudioEngine 所有で、ShutdownRuntime へ参照を渡す設計変更が必要 | caller が同時に偽装すれば Proof が成立し得る（Q2/Q6 は authority で防止済み） | ShutdownRuntime へ読み取り専用の観測インターフェースを注入する |
| Q6 > 0 の Faulted 昇格 | H.11.11.4 の「postStopEnqueue > 0 → Faulted」は未実装 |  shutdown 完了判定の意味を変えるため、別判断が必要 | 現状は Proof 不成立で reclaim を拒否する（安全側） | `ShutdownBlockingReason` の追加と遷移設計 |
| seal の強制力 | `RuntimeState` の public フィールドは seal 中でも直接書き換え可能。release に seal 前提条件は置けない（実測で publish が壊れる） | カプセル化の大規模変更が必要 / publish 順序の変更が必要 | コード規約違反時のみ。commit 時の sealed 検証では検出できない | フィールドを private 化し setter に `assertMutable()` を入れる（別作業） |
| loader の実時間競合テスト | 実 LoaderThread の競合は未試験 | engine 起動が必要 | commit 直前ガードの静的確認に留まる | AudioEngineHarness に非同期 IR 差し替えテストを追加 |
| Coordinator join 失敗時 | `stopThread(2000)` が失敗すると Q2 が false になり reclaim を拒否するが、`unique_ptr` reset は既存挙動のまま | 既存の JUCE スレッド破棄契約 | join できていないスレッドを破棄するリスクは従来どおり | `stopLoop()` の失敗を上位で扱う設計（別作業） |

保証できること: Q2/Q6 は authority の状態のみで判定され、caller の observation だけでは成立しない。Q5 は観測束と generation の一致が無ければ成立しない。reclaim の失敗は Release でも記録される。FrozenRuntimeWorld の publish 境界は seal を要求する。

保証できないこと: 4 条件（Q3/Q4 を含む）すべての同時偽装、seal 中の直接書き込み、実時間の並行実行での安全性。これらは「テストが通ったから安全」とは主張しない。

## 6. 検証コマンド

```
# 判定ロジック
grep -n "const bool q2\|const bool q5\|const bool q6" src/audioengine/ISRShutdown.cpp
grep -n "captureEpochQuiescenceEvidence\|coordinatorLoopStopped" src/audioengine/AudioEngine.h

# 死んだ状態の消滅
grep -rn "loaderGeneration_" src        # コメントとテストのみ

# seal 前提条件
grep -n "releaseState" -A 10 src/audioengine/FrozenRuntimeWorld.cpp

# テスト
build\Release\invariant_INV3_INV5Tests.exe
build\Release\LoaderCurrencyContractTests.exe
build\Release\AudioEngineHarness.exe
# ハーネスの限定実行（engine を起動しない D12-6 のみ）
set CONVOPEQ_D12_FROZEN_ONLY=1 && build\Release\AudioEngineHarness.exe
# アロケータ定義の一貫性（ターゲットごとの JUCE_DSP_USE_INTEL_MKL）
grep -c 'JUCE_DSP_USE_INTEL_MKL=1' build/CMakeFiles/impl-Release.ninja
```

## 7. 付録: 変更の意図（要約）

監査の指摘は「コード自身の自己申告が実装に追随していない」ことを突いていた。今回の是正は、自己申告を実装に合わせる作業に留めず、自己申告自体を減らす方向で進めた。Q2/Q6 は authority の状態を直接読み、Q5 は観測束で受けて generation を照合し、reclaim の成否は assertion ではなくカウンタに残した。次に同じ箇所を読む人は、コメントとコードのどちらからでも同じ契約を読み取れる。
