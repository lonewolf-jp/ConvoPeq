# D135-8/9 Gate C-F1 — Release Cycle-1 Deferred Forensic Audit

Date: 2026-08-30
前置き: 本監査は **production/test source 0変更・テスト条件変更なし・修正なし** で実施した read-only forensic investigation。
観測は既存 Diagnostic 機構（`CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`）+ DbgView64（Sysinternals, OutputDebugString 捕獲。JUCE logger 未登録時の `Logger::writeToLog` フォールバック先 = `OutputDebugString`、juce_Logger.cpp:57 を確認済み）のみで行った。

## Verdict（分類: F1-B）

> **F1-B — Admission/Deferred decision 経路。** cycle 1 の `requestRebuild(Structural)` は
> **duplicate suppression で棄却されていない（F1-A 否定）**。generation は新規採番（実測 cycle0=1 → cycle1=2）、
> Builder 実行、admission = **DeferredFadingActive**、`enqueueDeferred` 実行済み（F1-C 否定）。
> その直後、fading クリア前の 1ms coordinator 再駆動が 3回連続で再 Deferred となり、
> **4回目の enqueue で D135-8 の retry accounting（`kMaxDeferredRetries = 2`）が
> `RetryExhaustedDiscard` を発火して deferred 要求を破棄** → `hasDeferred_` が false に復帰し、
> テストの `waitUntil(hasDeferredRequest())` が true を一度も観測できず "cycle 1 did not defer" でタイムアウト。
> **baseline a65ace1（D135 変更前）Release は 3/3 PASS** → D135 変更起因の regression と確定（F1-E 否定）。

## A. 実行条件

| 項目 | 値 |
| --- | --- |
| working tree（D135側） | `C:\VSC_Project\ConvoPeq`（Gate B後と同一、commit a65ace1 + D135 uncommitted changes、変更なし） |
| baseline tree | `C:\VSC_Project\ConvoPeq-f1baseline`（`git worktree add` at **a65ace1**, detached HEAD。JUCE は tracked で自動展開） |
| configuration | Release（両方とも Ninja Multi-Config + cl 14.51.36231） |
| diagnostics | D135側 = `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON`（既存 `build-diag/`、増分ビルド）／baseline = OFF（production 既定） |
| executable | `build-diag/Release/AudioEngineHarness.exe`（D135 diag）／`ConvoPeq-f1baseline/build-f1/Release/AudioEngineHarness.exe`（baseline） |
| runs | D135 diag ×3（exit 1, 1, **0**）／D135 素 Release（Gate C 実測）×3（exit 1×3）／baseline ×3（**exit 0×3**） |
| 観測手段 | DbgView64（`/accepteula /o /l`）— `Logger::writeToLog` → `OutputDebugString` を捕獲。ソース・テストとも無変更 |

## B. cycle trace（失敗ラン run1 / 成功ラン run3 の対比）

### run1（D135 diag, FAIL）— 失敗テスト `testDeferredBacklogDrainsCompletely` の engine

```text
cycle 0 (16:25:12):
  [DIAG] requestRebuild(sr,bs): task queued generation=1          ← 受理（gen=1 採番）
  [DIAG] rebuildThreadLoop: generation=1 build=290.3ms            ← Builder 実行
  [D133] enqueue gen=1 sealed=1                                   ← publish 要求
  [D135] re-defer (new) gen=1 retryCount=0                        ← admission=DeferredFadingActive → defer 1回目
  （テスト poll が hasDeferred=true を観測 → hook=false → Ready → drain 成功）

cycle 1 (16:25:15):
  [DIAG] requestRebuild(sr,bs): task queued generation=2          ← 受理（gen=2 = cycle0+1 実測）★F1-A 否定
  [DIAG] rebuildThreadLoop: generation=2 build=28.3ms             ← Builder 実行
  [D133] enqueue gen=2 sealed=1
  [D135] re-defer (new)   gen=2 retryCount=0                      ← defer 成立（1回目）
  [D135] re-defer (retry) gen=2 retryCount=1                      ← 1ms coordinator 再駆動（fading 未クリア）
  [D135] re-defer (retry) gen=2 retryCount=2
  [D135] re-defer (retry) gen=2 retryCount=3
  [HEALTH] Deferred publish starved gen=2 sequence=4 retryCount=3 reason=RetryExhaustedDiscard
                                                                  ← 4回目 enqueue が kMax=2 超過で破棄
  → hasDeferred_=false に復帰（要求消失・DSP retire）。テスト poll は true を観測できず 45s タイムアウト
  FAIL: cycle 1 did not defer
```

### run3（D135 diag, PASS）— 同一テストの engine

```text
cycle 0: task queued gen=1 → enqueue gen=1 → （drain 成功）
cycle 1: task queued gen=2 → enqueue gen=2
         [D135] re-defer (new) gen=2 retryCount=0                ← defer 1回のみ
         （re-defer (retry) 0回: テスト poll が coordinator 再駆動に先勝ち → hook=false → Ready → drain 成功）
  → INFO: 2x deferred cycles drained (no slot/ownership leak)
```

### 補足（run2, FAIL）

gen=2 で `retryCount=0→1→2→3 → RetryExhaustedDiscard`（run1 と同一機構）。さらに run2 では
test 1（`testDeferredReadyPathDrainsSlot`）の gen=1 が `retryCount=1` まで到達 — レース境界ぎりぎりであることを裏付け。

### run3 の追加観測

gen=3（後続テスト `DeferredPublishViewStateMachineTests` の engine）も `re-defer (new) count=0` で PASS —
本番 path の他の deferred テストは境界に達していない。

## C. requestRebuild 判定

**accepted（duplicate-rejected ではない）。** 証拠: cycle 1 で `[DIAG] requestRebuild(sr,bs): task queued generation=2`
が出力（RebuildDispatch.cpp:745、`queued==true` 時のみ出力）。`BLOCKED duplicate pending task`（:765、
`blockedAsDuplicate` 時のみ出力）は **3ランすべて不検出** → **F1-A 否定**。

## D. generation（実測）

| Cycle | generation 実測 | 判定 |
| --- | --- | --- |
| 0 | 1 | — |
| 1 | **2** | **= cycle0 + 1（新規採番を実測で確定。`++rebuildRequestGeneration`、RebuildDispatch.cpp:655）** |

Gate C 報告時の推論（「generation は新規採番されるはず」）を実測で確定。kMax=2 のサイクル跨ぎ累積説は**実測により不成立**。

## E. Builder

**executed**（cycle 1: `[DIAG] rebuildThreadLoop: generation=2 build=28.3ms` → `[D133] enqueue gen=2`）。

## F. Admission

| Cycle | Admission 実測 | hasFading |
| --- | --- | --- |
| 0 | DeferredFadingActive（→ enqueueDeferred、1回で drain） | true（テストフック、atomic release/acquire） |
| 1 | **DeferredFadingActive**（→ enqueueDeferred 4回: new + retry×3 → 4回目 RetryExhaustedDiscard） | true |

## G. Deferred

**enqueueDeferred = YES**（cycle 1 は 4回実行: `re-defer (new) retryCount=0` + `retry ×3`）。
その後 `RuntimePublicationOrchestrator.cpp:489-502`（kMax=2 超過 branch）が **4回目の enqueue を破棄**
（DSP handle retire + return。`hasDeferred_` を true に戻さない）→ 観測窓から要求が消失。

## H. baseline（a65ace1 Release）

| Run | exit | 主要出力 |
| --- | --- | --- |
| 1 | 0 | `2x deferred cycles drained (no slot/ownership leak)` / `DeferredFlowIntegrationTests: PASS` / `AudioEngineHarness: all publish pipeline tests PASS` |
| 2 | 0 | 同上 |
| 3 | 0 | 同上 |

**a65ace1 Release = 3/3 PASS** → cycle-1 failure は **D135 変更起因（regression）** と確定。
構造的裏付け: a65ace1 には retry accounting 自体が存在しない（D135-1 で追加された未commit変更）ため
`RetryExhaustedDiscard` は発火し得ない。

## I. 結論（分類 + 機構）

**分類: F1-B（Admission/Deferred decision）** — 厳密には:

> admission は正しく `DeferredFadingActive` を返し、defer も成立する。しかし **fading がクリアされる前に
> 1ms coordinator tick が `processDeferredAdmission` を再駆動し、その都度再 Deferred → 再 enqueue
> （re-defer churn）となる。D135-8 で `kMaxDeferredRetries` が 10→2 になった結果、churn 3回で
> `RetryExhaustedDiscard` が発火し、テスト（および観測者）が defer を観測する前に要求が破棄される。**

- 判別子は「1ms coordinator 再駆動 vs テスト poll のレース」。run1/run2 = coordinator 勝ち（3 retry → 破棄 → FAIL）、
  run3 = poll 勝ち（0 retry → drain → PASS）。素 Release は 3/3 失敗、diag Release は 2/3 失敗（diag マクロ有無で
  codegen/タイミングが変わり境界が動く）。
- **D135 regression 確定**（baseline 3/3 PASS + 機構が D135-1/8 追加コードに内在）。
- 準拠事項: Gate C の判定表「D135-8/9関連テストFAIL → FAIL、即停止」および F1 の修正禁止7項目はすべて遵守
  （production/test source 変更 0、テスト変更 0）。

## 修正設計レビューに持ち込む検討事項（実装は未実施・提案に留める）

1. **観測窓と設計の衝突**: テストは「defer が fade クリアまで生存（≥45s）」を要求する一方、D135-1 の設計は
   「同一 obligation の再駆動は 2回許容、3回目で諦める（starvation bound）」。1ms re-drive 前提では
   fading 中の defer は約3-4msで枯渇する。pre-D135（kMax=10）でも約11msで枯渇する構造は同じであり、
   「fade 中の re-defer を枯渇カウントに数えるべきか」自体が設計問題（re-defer は「同一要求の再試行」ではなく
   「同一要求の保持」ではないか、という立論）。
2. 候補方向（いずれも要設計レビュー）: (a) fade-active 中の再 enqueue をカウント対象外とする、
   (b) exhaustion 時も DiscardReason を保持した滞留にする（破棄ではなく保留）、(c) kMax の意味を
   「観測不能時間」ではなく「Ready 判定後の失敗回数」に再定義、(d) テスト側の期待値を設計に合わせる。
   **いずれも本監査では実装していない。**
3. 補足: `RetryExhaustedDiscard` の破棄は DSP handle retire を伴う（D135-3 監査で指摘済みの ownership-loss 系列）。
   破棄が `[HEALTH]` telemetry で観測可能なこと、および run2 で test 1 も count=1 に達したこと（境界の近さ）は
   修正設計の入力情報。

## Artifacts

- `evidence/D135-8-9_GATE_C_F1_diag_run1.dbgview.log` / `run2` / `run3`（DbgView 捕獲の全文）
- `evidence/D135-8-9_GATE_C_F1_diag_run{1,2,3}.console.log`（ハーネス直接出力）
- `evidence/D135-8-9_GATE_C_F1_DIAG_RELEASE_BUILD_LOG.txt` / `..._BASELINE_BUILD_LOG.txt` / `..._LOG2.txt`
- ツール（ソース・テスト不変）: `tools/gatef1_build.bat`（パラメタライズドビルドラッパー）のみ。
  DbgView64 は `%TEMP%` に置きリポジトリ外。
- **残置**: baseline worktree `C:\VSC_Project\ConvoPeq-f1baseline`（再検証用に保持。削除はユーザー判断）。
