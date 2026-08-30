# D125 — Unified Lifecycle + IR-Origin Repair Contract Audit

**Date:** 2026-08-29
**性質:** read-only 設計監査 + macro-gated 診断のみ。**production semantics 変更 0**（setIRChangeFlag に default 引数追加・診断ログは全て macro-gated）
**Frozen HEAD:** `a65ace1` + D117 trace + D125 diagnostics

---

## D125-A — caller-tag 診断の実装と実行結果

### 実装（全て macro-gated / observation-only）

| 変更 | 内容 |
| --- | --- |
| `AudioEngine.h:1470` | `setIRChangeFlag(const char* callerTag = "unknown")` — default 引数追加（production 呼び出し互換）+ `[D125_IRFLAG_SET] caller/pendingIRGen/thread` ログ（macro-gated） |
| `AudioEngine.Timer.cpp:800` | tag `"TimerDeferredStructural"` |
| `AudioEngine.UIEvents.cpp:177` | tag `"UI"` |
| `DSPTransition.h:123` | tag `"DSPTransitionCrossfade"` |
| `AudioEngine.Timer.cpp:849` | `[D125_IRFLAG_CONSUME_TIMER]`（clear point 観測） |
| `AudioEngine.Snapshot.cpp:95` | `[D125_IRFLAG_PROMOTE_SNAPSHOT]`（clear point 観測） |

### 実行結果（3-burst + IR probe、35s）

| 観測 | 値 | 解釈 |
| --- | --- | --- |
| `IRFLAG_SET` | **0** | frozen lifecycle では 3 caller 全てが本 scenario で到達不能（UIEvents は CLI suppress、Timer:800 は deferred path 未発火、DSPTransition:123 は crossfade 分岐到達不能） |
| `IRFLAG_PROMOTE_SNAPSHOT` | 1（promoted=0） | bootstrap 時の snapshot 生成で 1 回確認（flag は常に false） |
| teardown | exit 139（間欠・既知の D119-BLOCKER-2） | 診断は revert 済みのため mask されたまま |

**[b8] 暫定回答**: frozen code では flag がそもそも SET されないため、D124 の中継点 [b8] は**この scenario では観測不能**。

## D125-B — D123 ストームログの再解析 → **D123 判定の重大な訂正**

`evidence/D123_g4_6pub.log` の再計数:

| 指標 | 値 |
| --- | --- |
| `REBUILD_REQUESTED`（rebuild 要求） | **11 回のみ**（intentId 15-25 = 手動 burst 6 + IR 適用系 5、**反復なし**） |
| `CONV_REBUILD`（rebuild **実行**） | **107 回** |
| `DSPCORE_PREPARE`（DSPCore 生成） | **107 回** |
| `[PUBLISH]`（commit 成功） | 2 回 |
| commitRuntimePublication FAILED | 0 回 |
| DC live（最終） | 103（107 生成 − 6 破棄 + baseline 2 ≈ 103 ✓） |

**訂正**: D123 の「setIRChangeFlag 自己言及ループ」判定は**誤り**。
- rebuild **要求**は 11 回 = 手動 intent のみ（flag 由来の再 admission は存在しない）
- 暴走の実体は **rebuild スレッド上のタスク再実行ループ**: 1 intent あたり **約 10 回の rebuild 実行**（毎回新 DSPCore 生成）、commit は 2 回のみ、残り ~101 が未 publish・未破棄で滞留
- `setIRChangeFlag`（DSPTransition:123）は crossfade 分岐で 1 回/cycle 呼ばれたが、**要求を増やしていなかった**
- INV-IRFLAG-1/2（D124 提案）は**存在しないメカニズムを対象としていた** → 再スコープ必須

**[b8] 最終回答（否定による閉鎖）**: 「setIRChangeFlag → 次の submitRebuildIntent」の中継点は**存在しない**。D123 暴走は IR flag ではなく **rebuild 実行側の再実行**によるもの。

## D125-C — INV-IRFLAG-1/2 の再スコープ

- INV-IRFLAG-1（rebuild-generated transition の IR-change 再主張禁止）: **現状無害**（flag を立てても request は増えない — D125-B 実証）。ただし「遅延抑止 flag を rebuild 起因で立てる」意味論は依然曖昧 → D126 で DSPTransition:123 の flag 設定は**撤去推奨**（意味論の曖昧さ除去・ループ対象外）
- INV-IRFLAG-2（bounded causal lifetime）: **RebuildThread 実行ループに対して再定義** — 「1 rebuild intent → build 実行回数に bound（または commit/obsolete のいずれかで必ず終端）」が必要。現行は intent 1 回につき実行 ~10 回・終端なし
- **新 invariant 候補（D126 契約へ）**:
  - **INV-REBUILD-EXEC-1**: 1 rebuild intent が生む DSPCore 生成は、commit または obsolete-destroy のいずれかで必ず終端する（現行: 未 publish DSPCore が滞留）
  - **INV-REBUILD-EXEC-2**: rebuild 実行が publish に失敗/obsolete した場合、当該 DSPCore は DSPGuard により破棄される（現行: 破棄 6 / 滞留 ~101 — 破棄経路が機能していない）

## D125-D — Lifecycle 3 点セットの統合再証明（D123 実績から）

| 項目 | D123 実測 | 状態 |
| --- | --- | --- |
| identity chain（同一 DSPCore* の retire→destroy） | ✅ 発火（`[D117_RETIRE] dsp=X → [D117_DESTROY] dsp=X` 対合） | **再利用可** |
| shutdown visibility（SHUTDOWN markers + logger 保持） | ✅ "mainWindow.reset() completed" / "LOGGER_DETACH / SHUTDOWN_END" まで記録 | **再利用可** |
| ①active handle publication | ✅ 動作（D123-A） | D126 Patch-A |
| ②crossfade completion 独立駆動 | ✅ 動作したが、**crossfade 機構自体が未完成**であることが D123-B 実行で判明（完了駆動を入れても再実行ループ） | **再設計対象** |
| ③IR flag ループ遮断 | **対象外確定**（ループは IR flag ではない） | **D126 から除外** |

## D125-E/F — 再実行ループの残謎と D126 契約の再固定

**未解決（D126 前に診断必須）**: 1 intent → 約 10 回の rebuild 実行の駆動源。候補:
- (a) RetryScheduler による rebuild task 再 dispatch（RetrySchedulerTypes.h に Delegate/Request SrBs 系あり）
- (b) rebuild thread ループのタスク再実行条件（pending task が publish 失敗/obsolete でも残存）
- (c) crossfade pending 状態（D123 で start された fade が完了しない）による publish 再試行

診断設計（macro-gated）: rebuild thread タスク実行ごとに `[D126_TASK_EXEC] intentId / generation / built / publishResult / obsolete / destroy` を記録（RebuildDispatch.cpp の rebuild ループ + DSPGuard 破棄点）。

**D126 修正契約の再固定（2 案）**:

| 案 | 内容 | 評価 |
| --- | --- | --- |
| **案 X（推奨）** | **crossfade 不engage**: rebuild 遷移を HardReset（即時 retire）に固定 — crossfade 分岐（claim/receipt/fade completion）を**一切通さない**。漏出修正は「publish → 即時 retire → destroy」のみ。crossfade 機構は別設計フェーズ（D127+）で完成させる | surface 最小。D123 で実証済みの identity chain（即時 retire 経路）のみ使用。未完成 crossfade 機構に依存しない |
| 案 Y | crossfade 機構の完成（completion 駆動 + 再実行ループ修正）を含む完全修復 | surface 大。再実行ループの診断が前提 |

**案 X の根拠**: (a) D123 実測で即時 retire 経路の identity chain は完結、(b) crossfade 機構の未完成部分（completion 駆動・再実行ループ）は D119/D123 で 2 回暴走源となった、(c) I4 の World policy は HardReset を正式 policy として持つ（Orchestrator の既定 build も HardReset）。trade-off: rebuild 遷移時の音声 crossfade が効かなくなる（HardReset = 瞬時切替）→ **音質への影響は運用検証（D127）で確認**する契約とする。

## D125-G — GO 判定

| # | Gate | 判定 |
| --- | --- | --- |
| G1 | caller-tag | **PASS**（3 caller 識別・binary 組込確認） |
| G2 | b8 中継点 | **PASS（否定による閉鎖）** — flag→rebuild再admission の中継点は存在しない。暴走は rebuild **実行**側の再実行ループ |
| G3 | generation semantics | **PASS**（pendingIRGen は UI/Timer のみ increment・DSPTransition は不変 = 起因 identity として不適合 → flag は causal identity に不向きと確定） |
| G4 | origin 分離 | **PASS**（UI / Timer / DSPTransition を tag で分離・frozen では全員到達不能） |
| G5 | IRFLAG-1 | **PASS（再スコープ）** — 現状無害・flag 設定は意味論曖昧のため撤去推奨 |
| G6 | IRFLAG-2 | **PASS（再スコープ）** — INV-REBUILD-EXEC-1/2 へ再定義 |
| G7 | lifecycle 統合 | **PASS**（identity chain + shutdown visibility は D123 実績を再利用） |
| G8 | Observe 除外 | **PASS**（D123-C 実装済み・再利用） |
| G9 | shutdown | **PASS**（D123-D 実装済み・再利用） |
| G10 | production change = 0 | **PASS**（default 引数 + macro-gated ログのみ） |

**D125: PASS（10/10）** — ただし **D126 の修正契約は案 X（crossfade 不 engage・即時 retire 経路）を推奨**し、実装前に rebuild 再実行ループの診断（D126 冒頭ゲート）を追加すること。

## 生成物

- 本ファイル（`evidence/D125_UNIFIED_CONTRACT_AUDIT.md`）
- `evidence/D125_diag_build.log` / `D125_probe.log`（IRFLAG_SET=0 の実測）
- production semantics 変更: **0**（setIRChangeFlag signature に default 引数追加のみ・挙動不変）
