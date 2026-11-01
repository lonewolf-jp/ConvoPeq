# A. Debug assert 違反の原因分析（2026-11-01）

- 対象: `src/audioengine/ISRDSPHandle.cpp:529` の `assert(prevState == DSPState::Quarantined)`
- 制約: **修正は未承認**（Owner 判断）。本書は原因と修正案の比較までで、コードは変更していない。
- 基準: `main` = `fd92c6e7`

## 1. 症状

`build-diag` の **Debug** 実行で次の assert が発火し、プロセスが abort する（exit code 3）。

```
Assertion failed: prevState == DSPState::Quarantined,
  file src\audioengine\ISRDSPHandle.cpp, line 529
```

Release / RelWithDebInfo では `NDEBUG` により assert が消えるため abort せず、関数は `false` を返す。

## 2. 該当コードの構造

`DSPHandleRuntime::destroyQuarantineSlot(slot, expectedGeneration)` は次の順で判定する。

| 順 | 処理 | 失敗時の戻り値 |
|---|---|---|
| 1 | `slot >= MAX_DSP_SLOTS` | `false` |
| 2 | generation 照合（`expectedGeneration != 0` のときのみ） | `false` |
| 3 | **state を読む → `assert(prevState == Quarantined)`** | （Debug は abort） |
| 4 | state != Quarantined なら | `false` |
| 5 | active / fading / crossfade の照合 | `false` |
| 6 | pin gate（`closePinsForDestroy` → `pinnedCount != 0` なら deferred 計上） | `false` |
| 7 | **CAS で `Quarantined → DestroyPending`** | CAS 失敗なら `false` |
| 8 | `instance = nullptr` → state `Reclaimed` → フリーリストへ push | `true` |

**要点**: 3 の assert は「読んだ時点の観測」に対する検査であり、**7 の CAS より前**にある。
一方、実際の状態遷移を排他しているのは **7 の CAS** である。この順序のため、
「観測した時点では Quarantined だったが、CAS の前に他スレッドが遷移させた」場合に
**assert だけが発火し、遷移は正しく 1 回に保たれる**という状態になる。

## 3. 呼び出し元とスレッド

`destroyQuarantineSlot` を呼ぶ製品コードは 3 系統ある。

| # | 呼び出し元 | 関数 | 実行スレッド（推定） |
|---|---|---|---|
| 1 | `AudioEngine.Commit.cpp:682` | `AudioEngine::onRuntimeRetiredNonRt()` | commit 経路（`commitRuntimePublication` の呼び出し元に依存。`PrepareToPlay` / `ReleaseResources` / `Timer` から到達） |
| 2 | `AudioEngine.Processing.ReleaseResources.cpp:468` | `releaseResources`（shutdown） | shutdown 実行スレッド |
| 3 | `ISRDSPHandle.cpp:606`（`retryDeferredQuarantineDestroy` 内） | `AudioEngine::tryReclaimResources()`（`Retire.cpp:55`）、`drainDeferredRetireQueues()`（`Retire.cpp:89`） | タイマーコールバック（`Timer.cpp:2018/2034`）、`ConvolverProcessor.Lifecycle.cpp:226`、`AudioEngine.h:5624` |

**競合の型**: 系統 1（commit）と系統 3（reclaim tick）が**別スレッドで同一 slot に対して並行に走る**。

```
スレッド X (commit)                   スレッド Y (reclaim tick)
------------------------------------  ------------------------------------
quarantine フラグ・lane・grace を確認
                                       retryDeferredQuarantineDestroy()
                                         → destroyQuarantineSlot(slot, 0)
                                           state=Quarantined を観測
                                           assert 通過
                                           CAS 成功 → Reclaimed
destroyQuarantineSlot(slot, 0)
  state を読む → すでに Reclaimed
  assert(prevState == Quarantined)  → **発火（Debug）**
  （Release では false を返して安全に終了）
```

系統 1 の手前には `dspQuarantineManager_.isActive(qslot)` と
`laneOf(qslot) == Quarantine` と grace 完了の確認があるが、いずれも**確認と破棄の間の排他ではない**。
したがって「確認は通ったが、その直後に系統 3 が破棄した」という順序が成立する。

### 3.1 スレッド証拠（追記・2026-11-01 17:20）

「別スレッドで並行する」ことを、コード上の経路で裏付けた。

| 系統 | 実行スレッドの根拠 | 証拠 |
|---|---|---|
| reclaim（系統 3） | JUCE Timer のコールバック＝メッセージスレッド | `AudioEngine.Timer.cpp:2018` / `:2034`（`tryReclaimResources()`）、`ConvolverProcessor.Lifecycle.cpp:226`（`provider->tryReclaimResources()`） |
| reclaim（系統 3・テスト） | **テストスレッド**（メッセージスレッドではない） | `AudioEngine.h:5621-5625`（テスト専用 `driveWorldRetirementReclaimForMeasurement()`）、`V11EngineRetireProtectionTests.cpp:1443`（`std::thread drainer([&e]{ e.tryReclaimResources(); })`）、`DeferredPublicationTestAccess.h:201/230` |
| commit（系統 1） | `commitRuntimePublication()` の呼び出し元に依存 | `AudioEngine.h:3884-3893`（`willRetireRuntimeNonRt` → `onRuntimeRetiredNonRt`）、呼び出し元は `AudioEngine.Processing.PrepareToPlay.cpp:179/309`、`AudioEngine.Processing.ReleaseResources.cpp:223`、`AudioEngine.Timer.cpp:1167` |

**読み方**: 製品経路（Timer コールバック）だけを見れば同一スレッドに見えるが、
ハーネスは `driveWorldRetirementReclaimForMeasurement()` と `std::thread drainer` で
**テストスレッドから reclaim を回す**。assert が観測されたのは Debug のハーネス実行であり、
この並行経路と整合する。製品側でも `ConvolverProcessor` の Timer と `AudioEngine` の Timer が
別インスタンスなら、両者が同時に reclaim を呼び得る点は残る（未確認）。

## 4. 実害の評価

| 観点 | 評価 |
|---|---|
| 二重解放 | **発生しない**。状態遷移は 7 の CAS が単独で成立させ、フリーリストへの push は `freeListMutex_` の下で 1 回だけ行われる |
| Release の挙動 | 安全（`false` を返し、呼び出し側は `continue` / スキップする） |
| Debug の挙動 | **abort**。ローカル検証（F2 / R51 等の確認）が中断される |
| 意味論 | 「既に他者が破棄した」ことを `false` で表現する契約と、assert の「呼び出し前提の検査」が矛盾している |

**結論**: 状態遷移の排他は正しく、欠陥は**検査の位置**にある。assert は「呼び出し前提」を
検査しているが、この関数の契約は「前提が崩れていたら `false` を返す」である。両者が食い違っている。

## 5. 修正案の比較（実装は未承認）

### A1. 呼び出し側での競合対策

呼び出し前に `getSlotState(slot)` で再確認し、Quarantined でなければスキップする。

| 項目 | 評価 |
|---|---|
| 変更範囲 | `AudioEngine.Commit.cpp` / `ReleaseResources.cpp`（2 箇所） |
| 効果 | **限定的**。確認と破棄の間に再び遷移し得るため、TOCTOU の窓は残る |
| リスク | 低（読み取りを 1 回増やすだけ） |
| 判定 | 単独では解決しない |

### A2. 所有側での状態遷移の原子化（assert の位置を遷移の所有者側へ移す）

「assert を削除する」のではなく、**CAS の結果に基づいて検査する**形に変える。

- CAS 成功時: 遷移したのは自分であり、直前状態が Quarantined であったことは CAS が保証する。
  ここで `assert` を維持してもよい（不変条件の文書化として機能する）。
- CAS 失敗時: **他者が既に遷移させた正常系**として扱い、`false` を返す（現行の戻り値契約どおり）。
  このとき「観測した状態が Quarantined のまま」なら本物の異常なので、そこだけ assert する。

| 項目 | 評価 |
|---|---|
| 変更範囲 | `ISRDSPHandle.cpp`（1 関数、数行） |
| 効果 | **競合の正常系を吸収**しつつ、真の異常（CAS 失敗かつ state が Quarantined のまま）は検出し続ける |
| リスク | 低。状態遷移の意味は変えない |
| 判定 | 最小変更で契約と実装を一致させる。ただし「なぜ 2 スレッドが同じ slot を破棄しようとするのか」という構造は残る |

### A3. 書き手の一元化（二重破棄を許容しない契約の維持）

quarantine slot の破棄要求を出すのは commit 経路までとし、**実際の破棄は reclaim tick（系統 3）だけ**が行う。
commit 経路は「破棄要求を立てる」に留める。

| 項目 | 評価 |
|---|---|
| 変更範囲 | `AudioEngine.Commit.cpp`、`RuntimePublicationOrchestrator` / reclaim 経路、場合により `DSPQuarantineManager`（複数ファイル） |
| 効果 | 競合そのものを構造的に消す。単一書き手の原則に合致 |
| リスク | **中〜高**。破棄のタイミングが 1 tick 遅れるため、slot 再利用の遅延・フリーリストの一時的な減少が起きる。grace 条件との相互作用の再検証が必要 |
| 判定 | 望ましい設計だが、影響範囲が広く、測定・回帰の再取得が必要 |

#### A3 の実装形は 2 通りあり、リスクが大きく違う（追記・2026-11-01）

現行コードを読むと、A3 は「全面的な書き手移動」でなくても成立する。

**現行の構造（`AudioEngine.Commit.cpp:655-687`）**: commit 経路は 1 スロットについて
**3 ストアの後片付け**を順に行う。

| 順 | 操作 | 失敗時の扱い |
|---|---|---|
| 1 | `dspHandleRuntime_.destroyQuarantineSlot(qslot, 0)` | `false` なら `continue`（後続 2 は実行しない） |
| 2 | `worldAuthority_.lifetime().reclaim(qslot)`（レーン解放） | — |
| 3 | `dspQuarantineManager_.reclaimSlot(qslot, 0)`（フラグ解放） | — |

**競合の本体**: reclaim tick の `retryDeferredQuarantineDestroy()`（`ISRDSPHandle.cpp:590-614`）は
`deferredQuarantineDestroy_`（**グローバルなカウンタ**）が 0 でなければ、
**Quarantined な全スロット**を走査して 1 を実行する。つまり「commit 経路が今まさに
後片付けしているスロット」も対象になる。これが二重試行の原因である。

| 案 | 内容 | 変更規模 | リスク | 効果 |
|---|---|---|---|---|
| **A3-a** | tick の対象を「commit 経路が pin gate で deferred 計上したスロット」に限定する（`deferredQuarantineDestroy_` を**グローバルカウンタからスロット単位のマーク**へ変える） | 小〜中（`ISRDSPHandle.h/.cpp` と `destroyQuarantineSlot` の defer 計上のみ） | 低〜中。tick が「取りこぼし」を拾えなくなるため、マーク漏れがあると slot リークになる | 競合が構造的に消える。3 ストアの順序は不変 |
| **A3-b** | tick が 3 ストアすべて（1〜3）を所有し、commit 経路は要求のみ | 大（grace 条件の再計算、lane/flag の所有者変更、shutdown 経路も追随） | 高。破棄が 1 tick 遅延し、grace 条件の再評価が必要 | 単一書き手が完全に成立 |

**推奨**: A3 を実施する場合も **A3-a** から入る。A3-b は lane/flag の所有権まで動かすため、
D-02 の測定や既存の retire 契約に影響し得る。

**A3-a の検証観点**（実施する場合）:

| # | 内容 | 合格条件 |
|---|---|---|
| V-4 | `[F2]` 系（`V11EngineRetireProtectionTests`） | 全 PASS を維持 |
| V-5 | quarantine slot の再利用（世代 +1） | `DeferredFlowIntegrationTests` / `STG-8` 群が全 PASS |
| V-6 | シャットダウン経路（`ReleaseResources.cpp:450`） | `retryDeferredQuarantineDestroy()` が 0 を返し、リークしない |
| V-7 | Debug 実行での assert 発火 | 0 回 |

## 6. 推奨（Owner 判断用）

1. **A2 を先行**し、競合の正常系を吸収して Debug の abort を止める（状態遷移の意味は不変）。
2. 併せて **A3-a（tick の対象を deferred 計上済みスロットに限定）から設計評価**する。
   A3-b（tick が 3 ストアすべてを所有）は lane/flag の所有権まで動くため、後段で評価する。
3. **A1 は単独では採用しない**（TOCTOU が残るため）。

A2 を適用する場合の検証は次のとおり。

| # | 内容 | 合格条件 |
|---|---|---|
| V-1 | Debug ビルドで `AudioEngineHarness` を反復 | assert 発火 0 回 |
| V-2 | Release / ASan の全テスト | 51/51 成功を維持 |
| V-3 | `isr-verify` / authority 系 | 全 PASS を維持 |

## 7. 未確定事項

- 系統 1 の実行スレッドは呼び出し元に依存する（`PrepareToPlay` / `Timer` / `ReleaseResources`）。
  競合が実際に起きる組み合わせを、スタックトレースで確定させる必要がある。
- assert 発火の頻度は未計測（Debug 実行でのみ観測）。B の調査と合わせて再現条件を記録する。
