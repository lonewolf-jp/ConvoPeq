# D-01 分析書 — STG-8 群のテスト・フレーク（2026-11-01）

- 対象: `checkSTG82OverwriteTerminalizesEvicted`（`[STG-8-D2]`）と
  `checkSTG8ClearIsIdempotent`（`[R8-2]`）
- 制約: **製品コードの変更なしを優先**（Owner 指示 D-01）。期待値・終端化条件の緩和、
  待機時間の単純延長、skip 化は行わない。
- 基準: `main` = `a12e05f2`

---

## 1. 症状

| テスト | 観測メッセージ | 意味 |
|---|---|---|
| `checkSTG82OverwriteTerminalizesEvicted` | `[STG-8-D2] FAIL: unstable (L=0 hasDeferred=0)` | O2 の deferred 保持を 8 秒間確認する区間で、obligation が終端化（L=0）し deferred も消えた |
| `checkSTG8ClearIsIdempotent` | `[R8-2] FAIL: first clear (L=0 hasDeferred=1)` | clear 直後に obligation は 0 だが deferred slot が存在する（不整合な中間状態） |

いずれもローカル（`ctest` 経由または Debug 実行）で断続的に発生し、失敗すると
`runSTG8RecoveryObligationTests()` が中断して後続の V11 群（F2 / R51）に到達しない。

---

## 2. 根本原因（2 件・いずれもテスト側の前提欠陥）

### RC-1 「fading 前提」を作るテストフックが**死んでいる**（STG-8-D2 の原因）

`DeferredPublicationTestAccess::setFadingRuntimePresent(e, on)` は
`e.testFadingRuntimePresent_` を書くだけだが、**この値を読む製品コードが存在しない**。

| 証拠 | 内容 |
|---|---|
| `AudioEngine.h:4005` | `std::atomic<bool> testFadingRuntimePresent_ { false };`（テスト専用と明記） |
| `AudioEngine.h:4006-4009` | getter `testFadingRuntimePresent()` は自分の atomic を読むだけ |
| リポジトリ全体の検索 | `testFadingRuntimePresent_` の参照は上記と `DeferredPublicationTestAccess.h:36`（書き込み）のみ。**getter の呼び出し元は 0** |
| `PublicationAdmission::evaluate()` | fading 判定は `engine.hasFadingRuntimeInWorld(...)` のみ（`PublicationAdmission.cpp:55`） |
| `hasFadingRuntimeInWorld` | `runtimeWorld->topology.fadingRuntimeUuid != 0` を読む（実ワールドのみ） |
| コメント | `AudioEngine.h:3996-4004` は「テストビルドでのみ evaluate() が testFadingRuntimePresent() に到達できる」と説明しているが、**その分岐は現存しない** |

**帰結**: `checkSTG82OverwriteTerminalizesEvicted` が期待する
「fading 中なので deferred を保持し続ける（8 秒）」という前提は、実際には成立していない。
deferred は通常のパイプラインで Ready → publish され得るため、8 秒窓の間に
L=0 / hasDeferred=0 になる。**テストは「たまたま保持されていた」ことを検証していた**。

### RC-2 テストが「単一書き込み者」契約に違反して clear を直接呼ぶ（R8-2 の原因）

| 証拠 | 内容 |
|---|---|
| `RuntimePublicationOrchestrator.h:229-233` | `requestDeferredClear()` が**唯一の Writer**（非 RebuildThread）、`drainDeferredClearIfRequested()` が**唯一の Reader**（RebuildThread のみ）と明記 |
| `RuntimePublicationOrchestrator.cpp:725-743` | `requestDeferredClear()` は RebuildThread 停止時のみ同期 clear、稼働中は intent を latch して起床させる |
| `RuntimePublicationOrchestrator.cpp:727-732` | R8 (P1) の修正コメント: 同期 clear を稼働中の RebuildThread と並行させると **2 スレッドが `deferredSlot_` / `hasDeferred_` を書く**ため、述語を `rebuildThreadStopped` に変更した |
| テスト旧実装 | `checkSTG8ClearIsIdempotent` はテストスレッドから `orch.clearDeferredForShutdown()` を**直接**呼んでいた（RebuildThread は稼働中） |

**競合の型**:

```
テストスレッド : clear → obligation 解決（L 1→0）→ slot reset → hasDeferred=false
RebuildThread  : processDeferredAdmission → Ready → submitPublishRequest → 再 enqueue
                 → hasDeferred=true
結果           : L=0 かつ hasDeferred=1（観測された FAIL 状態）
```

**帰結**: R8-2 の失敗は「製品の不整合」ではなく、**テストが製品の契約を破っていた**ことによる。

---

## 3. 修正方針

### 3.1 実装する（テスト専用・製品コード変更なし）

- `checkSTG8ClearIsIdempotent`: clear を**製品経路**（`requestDeferredClear()`）に置き換え、
  結果を上限付きで待つ。**検証内容（L==0 / deferred なし / 反復が no-op）は一切変更しない**。
- `checkSTG82OverwriteTerminalizesEvicted`: RC-1 の修正（フック復活）が承認されるまで
  **アサーションを変更しない**（弱めない）。

### 3.2 承認が必要（製品コード 2 行）

RC-1 を直すには、`PublicationAdmission::evaluate()` の fading 判定に
**テスト専用フック**を戻す必要がある。

```cpp
// PublicationAdmission.cpp（evaluate の fading 判定）
const bool hasFading =
    engine.hasFadingRuntimeInWorld(engine.makeRuntimeReadHandle(ctx))
    || engine.testFadingRuntimePresent();   // ★ テスト専用 override（既定 false）
if (hasFading)
    return Decision::DeferredFadingActive;
```

- `PublicationAdmission` は既に `AudioEngine` の friend（`AudioEngine.h:3999`）。
- 既定値は `false` のため**製品挙動は不変**。呼び出しは publish 評価時（非 RT 経路）で、
  atomic load 1 回。
- これにより STG-8-D2 の「fading 中の retention」が**実際に検証される**（テストが意味を取り戻す）。

**代替案（採らない）**:
1. 8 秒安定性のアサーションを削除 → 検証の弱体化（Owner 禁止）。
2. 実 crossfade をテストで開始して fading を作る → 重量級で新たな非決定性を持ち込む。
3. タイムアウト延長 → 原因を隠すだけ（Owner 禁止）。

---

## 4. 検証計画

| # | 内容 | 合格条件 |
|---|---|---|
| V-1 | `build-diag` Release でビルド | 成功 |
| V-2 | `AudioEngineHarness` を 6 回反復 | `[R8-2] FAIL` が 0/6 |
| V-3 | `ctest -R AudioEngineHarness`（`CONVOPEQ_SOURCE_ROOT` あり） | 全 PASS |
| V-4 | フック復活後（承認後）: STG-8-D2 を 6 回反復 | `unstable` が 0/6 |
| V-5 | CI（PR） | 全ジョブ成功 |

---

## 5. 再現性の実測

| 実施 | 日時 | 結果 |
|---|---|---|
| 現行 main のハーネスを 3 回連続実行（`build-diag` Release） | 2026-11-01 02:04 | **0/3** で STG-8 失敗なし |
| 同 6 回連続実行 | 2026-11-01 07:44〜08:08 | **0/6** で STG-8 失敗なし |
| 同日の観測（`ctest` 経由／Debug 実行、測定・ビルドと並行していた時間帯） | 2026-10-31〜11-01 | **3 回**発生（`[STG-8-D2] unstable` 1、`[R8-2] first clear` 2） |

**解釈**: 単独実行では再現しない（0/9）。発生はいずれも**他の負荷（測定・ビルド・CI 待ち）と
並行していた時間帯**に偏っており、**タイミング依存の競合**であることと整合する。
RC-1 / RC-2 はいずれも「前提が成立しない／契約違反」という**静的に証明できる欠陥**であり、
再現率の低さが原因の否定にはならない。

**注意**: 修正後の「0/6」は、修正前も 0/6 であったため**改善の証明にはならない**。
本修正の根拠は §2 の静的証拠（コメントと実装の不一致、単一書き込み者契約の明文）である。
RC-1 の修正（フック復活）を適用した後は、**fading 前提が実際に成立する**ため、
8 秒安定性の検証が意味を持つようになる（その時点で再測定する）。

