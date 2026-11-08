# 外部監査の 3 指摘（未接続・概念段階の証拠）の実態調査

- 実施: 2026-11-08 20:40
- 対象: `main` = `8825d741`（V0.6.12）
- 調査方法: ソース全文の参照関係、CMake の本番ターゲット定義、ビルド済みオブジェクトのシンボル、CI ワークフローの実行範囲
- コード変更: なし（本報告のみ）

## 結論

指摘 3 点のうち 2 点は「コード内の自己申告が実装に追随していない」ことによる誤読で、事実として誤りである。残る 1 点は「実装の場所が違う」ことによる誤読で、正当性は別機構が担保している。ただし 3 点すべてに、指摘の趣旨——証拠が自己申告と文書に依存している——が当たる実質的な残課題がある。監査の推定手順は妥当で、結論だけが現行コードとずれている。

| 指摘 | 一次判定 | 実態 | 残る実質課題 |
|---|---|---|---|
| `ISRLifetimeProof.h` は「型のみ・production 未接続」 | **不成立** | Phase A2 で本番のシャットダウン経路に接続済み。型だけでなく生成・消費まで実装されている | 8 つの quiescence 条件のうち 3 つが `true` 固定。証拠の 37.5% が仮定 |
| ARCHITECTURE §3.2 が `FrozenRuntimeWorld` を「Phase-4 frozen world concept」と記す | **不成立**（ただし文書の欠陥） | publish 境界の RAII wrapper として本番経路で構築・消費されている | 表の記述が役割を説明しておらず、未使用と読める。型レベルの不変性は Coordinator 経路には及ばない |
| `ConvolverProcessor.LoaderThread.cpp` の「B-6: generation check は実装待ち」 | **一部妥当** | 陳腐化したコメント。stale 上書きを防ぐ世代ガードはコード全体で 8 箇所ある | ロードループ内に世代チェックが無い。`loaderGeneration_` は書き込み専用。参照先文書が存在しない |

## 1. `ISRLifetimeProof` は型だけではない

### 1.1 本番のシャットダウン経路に接続されている

`~AudioEngine` から `releaseResources()` に入り、次の順で Proof → Permit → reclaim が実行される。

| 段階 | 位置 | 内容 |
|---|---|---|
| admission 閉鎖 | `AudioEngine.Processing.ReleaseResources.cpp:109` | `shutdownRuntime_.closeAdmission()` |
| producer join | 同 `:250` | `while (!shutdownRuntime_.joinProducers())` の再試行ループ |
| reclaim 要求 | 同 `:534` / `:541` | `tryShutdownQuiescentReclaim(activeHandle)` / `(fadingHandle)` |
| 観測値の収集 | `AudioEngine.h:4901-4912` | `QuiescenceObservation` に Q0〜Q7 と世代を格納 |
| Proof 生成 | `AudioEngine.h:4914` → `ISRShutdown.cpp:350` | 全 Q 成立時のみ `ShutdownQuiescenceProof` を返す |
| Permit 発行 | `AudioEngine.h:4918` → `ISRShutdown.cpp:401` | Proof の identity を束縛した `ReclaimPermit` |
| 認可と消費 | `AudioEngine.h:4928` → `ISRRuntimePublicationCoordinator.cpp:775` | identity 照合 → `permit.consume()` → `retire` / `reclaim` |

型が定義だけである可能性は、次の 3 点で排除できる。

1. `ISRShutdown.cpp` は本番ターゲットのソース一覧に含まれる（`CMakeLists.txt:1512`、`target_sources(ConvoPeq ...)` は `:1630`）。
2. 本番ビルドのオブジェクトに生成関数の実体がある。`llvm-nm --demangle build-icx/CMakeFiles/ConvoPeq.dir/Release/src/audioengine/ISRShutdown.cpp.obj` が `T`（定義済み）として `ShutdownRuntime::tryMakeQuiescenceProof` と `ShutdownRuntime::tryMakeReclaimPermit` を返す。
3. 専用テストが存在し、CI でも実行される。`src/tests/invariant_INV3_INV5.cpp` が G13（Q 条件）・G21（Permit の単回使用）などを検証し、`CMakeLists.txt:1050` で `add_test(NAME InvariantINV3INV5 ...)` に登録されている。MSVC の ASan ジョブは全構成をビルドしたうえで `ctest`（`.github/workflows/sanitizer-ci.yml:195`）を実行するため、この登録名も対象に入る（`CONVOPEQ_ENABLE_ISR_TESTS` の既定は ON）。直近の PR #197 でも同ジョブは pass している。

### 1.2 「型のみ」と読める状態が残っている理由

コメントと文書が Phase A1 の時点で止まっている。同じヘッダー内で記述が矛盾しているため、コードを読んだ監査者が古い側を引用するのは自然な流れである。

- `src/audioengine/ISRLifetimeProof.h:12-13` — 「現在は『型のみ・production 未接続』。production reclaim の接続は Phase A2 で行う」
- `src/audioengine/ISRShutdown.h:252-254` — 「ShutdownQuiescenceProof / ReclaimPermit は『型のみ・production 未接続』」
- `src/audioengine/ISRShutdown.h:286` — 「`AudioEngine::tryShutdownQuiescentReclaim` … から production で呼出・接続済み」（**同じヘッダーの 34 行下で接続済みと書いている**）
- `ARCHITECTURE.md:212` — 「Currently type-only; production reclaim connection is a later phase.」

さらに監査は「CMakeLists 精読範囲でもテストリンク一覧に現れない」としているが、`ISRLifetimeProof.h` はヘッダーなのでリンク一覧に現れないのが正常である。実装を持つ `ISRShutdown.cpp` はテストターゲット（`CMakeLists.txt:228`）にも本番（`:1512`）にも明記されている。探索対象をヘッダー名にしたことが見落としの原因である。

### 1.3 ただし証拠の質には実質的な穴がある

接続は事実だが、Proof が検証する 8 条件のうち 3 条件は定数をそのまま渡している。`AudioEngine.h:4904-4909` を読むと、Q2・Q5・Q6 に `true` が直接代入されている。

| 条件 | 供給元 | 評価 |
|---|---|---|
| Q0 予約ゼロ | `shutdownRuntime_.outstanding() == 0` | 実測 |
| Q1 admission 閉鎖 | `admissionState()`（`ISRShutdown.cpp:356`） | 実測 |
| Q2 producer join 完了 | `true` 固定（`AudioEngine.h:4904`） | 仮定。ただし Q0 と Q1 の成立から導ける（`joinProducers()` は `outstanding()==0 && Closing` を要求）ため、実質は冗長 |
| Q3 reader 登録閉鎖 | `m_epochDomain.readerRegistrationClosed()` | 実測 |
| Q4 active reader ゼロ | `m_retireRouter->activeReaderCount() == 0`（router が null なら `true`） | 実測（null 時のみ仮定） |
| Q5 epoch settled | `true` 固定（`AudioEngine.h:4908`） | 仮定。`EpochQuiescenceEvidence` という型は存在せず、コメントにしか現れない |
| Q6 postStopEnqueue ゼロ | `true` 固定（`AudioEngine.h:4909`） | 仮定。`ShutdownRuntime` 自身が `sh6PostStopEnqueueCount_`（`ISRShutdown.h:345`）を持ち、`collectResult`（`ISRShutdown.cpp:178`）で読めるのに使っていない |
| Q7 再起動なし | `!shutdownRuntime_.isAdmissionOpen()` | 実測 |

Q5 と Q6 は「測れるのに測っていない」または「測る対象が未定義」であり、ここは監査の懸念が正しく当たる部分である。加えて `ReleaseResources.cpp:535` / `:542` の `jassert(reclaimed)` は Release ビルドで消えるため、reclaim の失敗が観測されずに進む。証明機構は動いているが、証明の内容の一部が空欄というのが正確な状態である。

## 2. `FrozenRuntimeWorld` は概念ではなく publish 境界の実装

### 2.1 本番の publish 経路で構築される

`RuntimePublicationOrchestrator.cpp:297` が Builder の返した world を `FrozenRuntimeWorld` で包み、`PublicationExecutor::publish` に渡す。Executor は `frozen->releaseState()`（`PublicationExecutor.cpp:37`）で `RuntimeState*` を取り出し、`commitRuntimePublication` に渡す。`FrozenRuntimeWorld` のデストラクタは所有権が残っている場合に `unseal()` を呼ぶ（`FrozenRuntimeWorld.cpp`）。`CMakeLists.txt:1527` で本番ソースに含まれる。

したがって「概念段階」ではない。publish のたびに構築・消費される実装である。

### 2.2 ただし型レベルの保証範囲は名前より狭い

ヘッダー自身が制約を書いている。C++ の `operator->` は生ポインタの組み込み演算子を経由しないため、`consumeWorldHandle` が返す `const FrozenRuntimeWorld*` に対して `world->field` が書けない。このため PublicationCoordinator の World 型パラメータとしては使えず、Coordinator は `RuntimeState` を直接扱う（`FrozenRuntimeWorld.h` 冒頭のコメント）。

つまり `FrozenRuntimeWorld` が保証するのは Builder 境界の RAII と const アクセスまでで、Coordinator 以降の不変性は `sealRecursively()` と規約に依存する。監査が「概念」と読んだのは、`ARCHITECTURE.md:236` の記述が「Phase-4 frozen world concept」の一語しかなく、この役割と制約を書いていないためである。これはコードの欠陥というより文書の欠陥である。

## 3. B-6 の世代チェックは「ループ内には無い」が「正当性は別にある」

### 3.1 コメントの参照先が解決できない

`ConvolverProcessor.LoaderThread.cpp:43-44` は「B-6: Generation check is pending implementation in the load loop. See REPAIR_PLAN section B-6」と書く。しかし `REPAIR_PLAN` という名称のファイルはリポジトリに存在しない（`doc/work52/repair_plan.md` などはあるが B-6 の項目を持たない）。`doc/bug.md` の B-6 は `prepareToPlay` のロールバックで、別物である。「B-6」は旧バグ一覧（`doc/bug.md:100`）、D101-31-B（`AudioEngine.h:4903`）、work88（`AudioEngine.Init.cpp:63` の X4-B-6）、work92（`MKLNonUniformConvolver.h:380`）、本ローダー、RuntimeHealthMonitor（`5-VI-B §B-6`）で共用されており、識別子として機能していない。

### 3.2 stale 上書きを防ぐガードは 8 箇所に実在する

| 位置 | 役割 |
|---|---|
| `ConvolverProcessor.LoadPipeline.cpp:59`、`:253` | 要求開始時に `bumpGeneration()` で世代を確定 |
| 同 `:223` | キャッシュ削除の安全性判定（使用中エントリの保護） |
| 同 `:297` | 変換のキャンセル述語（`!isCurrentGeneration(generation)`） |
| 同 `:404` | `applyComputedIR` の入口判定 |
| 同 `:827` | `executePendingCommit` の commit 直前判定 |
| `ConvolverProcessor.LoaderThread.cpp:642-654` | リサンプル中と直後の自世代照合 |
| `ConvolverProcessor.StateAndUI.cpp:1078` | `updateConvolverState` の stale 破棄 |

同期・非同期のどちらの commit も `executePendingCommit` を通る（`LoadPipeline.cpp:761`、`:778`）ため、古い結果が新しい要求を上書きする経路は静的には成立しない。設計意図もコメントに明記されている——「cancellation は最適化、commit guard が正当性」（`LoadPipeline.cpp:56-58`）。

### 3.3 それでも残る 3 つの実欠陥

第一に、ロードループの各段（`doLoadIRStep` / `doTrimStep` / `doTransformStep` / `doBuildStep`）に世代チェックが無い。`LoaderThread` は要求時の世代を `requestGenerationId` として保持するが、参照するのはリサンプル処理だけである（`LoaderThread.cpp:642-654`）。各段の中断判定は `externalCancellationCheck`（同 `:110-116`）＝ `threadShouldExit` であり、世代は見ない。

`requestAsyncLoad` は新しい要求のたびに旧ローダーへ `signalThreadShouldExit` を送る（`LoadPipeline.cpp:50-54`）ため、通常は次のチェックポイントで止まる。しかし `loadIR()` は世代だけを進めてローダーに通知しない（`LoadPipeline.cpp:253`）。この経路で世代が進むと、旧ローダーは停止要求を受け取らないまま読み込み・トリム・位相変換・NUC エンジン初期化（`finalizeNUCEngineOnMessageThread`、同 `:663`）まで進み、commit 直前の世代判定で捨てられる。誤った状態にはならないが、作業と一時メモリの浪費になる。

第二に、`loaderGeneration_`（`ConvolverProcessor.h:996`）が書き込み専用である。デストラクタで加算するだけで（`ConvolverProcessor.Lifecycle.cpp:109`）、読み出す箇所が 1 つもない。コメントは「Loader は次回起動時に終了」と説明するが、その判定は存在しない。所有者寿命は実際には `forceCleanup()` の `stopThread(500)`（`ConvolverProcessor.StateAndUI.cpp:1045-1052`）と `WeakReference` による `callAsync` で守られている。宣言だけが残った死んだ機構である。

第三に、停止した理由がログから区別できない。`threadShouldExit` は新しい要求による差し替えでも所有者の破棄でも同じように立つため、読み込みが途中で終わったときに、取り消されたのか破棄されたのかを診断情報から復元できない。

## 4. 監査の判定手順について

3 点はいずれも「コード自身の自己申告」を根拠に選ばれている。自己申告が古い場合に誤る構造で、実際に 2 点がそうなった。一方で、指摘の奥にある「証拠が自己申告に依存している」という問題意識は、Q2/Q5/Q6 の固定値、`loaderGeneration_` の死んだ宣言、`jassert` 依存の失敗観測という形で実在する。監査の結論を否定しても、指摘の方向は妥当である。

なお同節には Timer Thread と RetryScheduler の内部矛盾の指摘もあるが、本調査の範囲外であり、判定していない。

## 5. 修正提案

優先度は、誤読の再発防止を上位に置いた。

| 優先 | 対象 | 方針 | 規模 |
|---|---|---|---|
| P1 | `ISRLifetimeProof.h:12-13`、`ISRShutdown.h:252-254`、`ARCHITECTURE.md:212` | 「Phase A1 で型を導入し、Phase A2 で接続済み」と現状を書き換える。接続点のファイル名を併記する | 3 箇所のコメント |
| P1 | `ARCHITECTURE.md:236` | 「Phase-4 frozen world concept」を、Builder 境界の RAII wrapper であること、Coordinator は `RuntimeState` を直接扱うことの 2 点に置き換える | 1 行 |
| P1 | `AudioEngine.h:4909`（Q6） | `ShutdownRuntime` に `postStopEnqueueCount()` を公開し、実測値で判定する。`sh6PostStopEnqueueCount_` は既にあるため追加コストは小さい | アクセサ 1 つ＋1 行 |
| P2 | `AudioEngine.h:4908`（Q5）、`:4904`（Q2） | Q5 は証拠型を作るか、仮定であることをコメントで明示する。Q2 は Q0・Q1 から導出できると注記する | コメントまたは型 1 つ |
| P2 | `LoaderThread.cpp:43-44` | 実装済みの内容（どの世代ガードがどこにあるか）に書き換え、参照先の無い `REPAIR_PLAN` を削除する | 2 行 |
| P2 | `ConvolverProcessor.h:996` / `Lifecycle.cpp:109` | `loaderGeneration_` を削除するか、各ステップ先頭で `requestGenerationId` と併せて判定する形で接続する | 数行 |
| P3 | `LoaderThread.cpp` の `stepOnce` | 各ステップ先頭に `owner.convolverStateGeneration.isCurrentGeneration(requestGenerationId)` を置き、stale な作業を早期に打ち切る | 1 箇所 |
| P3 | `ReleaseResources.cpp:535`、`:542` | `jassert` だけでなく、失敗回数を診断カウンタに記録して Release でも観測できるようにする | 数行 |

## 6. 検証の再現手順

```
# 本番接続の確認（ソース）
grep -rn 'tryShutdownQuiescentReclaim' src/audioengine
grep -n 'ISRShutdown.cpp' CMakeLists.txt        # 228（テスト）, 1512（本番）
grep -n 'target_sources(ConvoPeq' CMakeLists.txt

# 本番オブジェクトのシンボル確認
"C:\Program Files\LLVM\bin\llvm-nm.exe" --demangle \
  build-icx/CMakeFiles/ConvoPeq.dir/Release/src/audioengine/ISRShutdown.cpp.obj \
  | grep -i 'QuiescenceProof\|ReclaimPermit'

# Q 条件の供給（実測 / 固定値）
sed -n '4896,4915p' src/audioengine/AudioEngine.h

# FrozenRuntimeWorld の構築点
grep -rn 'aligned_make_unique<convo::FrozenRuntimeWorld>' src

# 世代ガードの分布
grep -n 'isCurrentGeneration' src/convolver/*.cpp

# 死んだ世代（書き込みのみ）
grep -rn 'loaderGeneration_' src

# CI の ctest 範囲
grep -n 'ctest' .github/workflows/sanitizer-ci.yml
grep -n 'InvariantINV3INV5' CMakeLists.txt
```

## 7. 調査の限界

静的調査に限定した。実機のシャットダウンで Proof が成立したかは、`evidence/shutdown_trace.json` の採取では確認していない。stale ローダーが捨てる作業量（NUC 初期化を含む）も実測していない。Q5 の `EpochQuiescenceEvidence` は仕様書（H.11.13.3）側の定義を確認していないため、未実装なのか別名で存在するのかは断定していない。

## 8. 変更ファイル

| 種別 | ファイル |
|---|---|
| 報告 | 本ファイルのみ（コード・既存文書の変更なし） |
