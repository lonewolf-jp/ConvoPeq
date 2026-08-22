# Phase I-T1-C6-R1 — Evidence Re-audit

> **Verdict: T1-C6 = FAIL 継続（G2 = 1件: G2-2 新規昇格）**
> G2-1 は RESOLVED を維持。しかし先入観なしの再分類により、`observedOutstandingMax` writer 経路が
> **harness から到達不能・動的カバレッジゼロ**であることを発見（G2-2）。
> A_max candidate 記録の依存先であるため C6 PASS 条件（G2=0）を満たさない。
> コード変更 0 / measurement run 引き続き禁止 / A_max・R・R_cap・T2 禁止のまま。

## 1. R1 — G2 census 再実行（src/tests 全件・41 ファイル）

| 対象 | 必須結果 | 実測 | 判定 |
|---|---|---|---|
| `releaseObserved()` の test assertion | ≥ 1 | **9件**（全て WorldRetirementMeasurementTests.cpp:321-483） | ✅ |
| `addReleaseObserved()` test-side direct misuse | 0 | **0件** | ✅ |
| `worldReclaimCount` の measurement test | ≥ 1 | **15 refs**（:288-358 ほか） | ✅ |
| `ΔworldReclaimCount == ΔreleaseObserved` assertion | 有 | :416 `if (dRelease != dReclaim) FAIL` | ✅ |
| double-transfer regression assertion | 有 | :461/464/474/487/500/518-520（複数） | ✅ |
| reference → releaseObserved 誤結合 | 0 | **0件** | ✅ |
| production writer | 共有 transfer step のみ | `transferWorldReclaimDeltaForTelemetry` 1実装 | ✅ |
| `lastSampledWorldReclaimCount_` writer | 1 | 共有メソッド内 1箇所 | ✅ |
| transfer helper caller | production + harness | 2 call sites（Timer.cpp:442 / AudioEngine.h:4954） | ✅ |

## 2. R2 — deterministic test の assertion 強度（source 監査）

実行ログではなくソースの assert 文を監査。すべて `return false` による**明示的 failure 分岐**であり diagnostic 出力ではない。

| 要求 | ソース根拠 |
|---|---|
| N=1: Δreclaim==1 / Δrelease==1 / Δreference==1 | :409 `dReclaim != destructions → FAIL`（N=1 で実行）+ :416 + :423 系 |
| N=4: 同上 | 同一 case 関数を `destructions=4` で実行（同一 assert 群が N=4 を強制） |
| second sampler: Δrelease == 0 | :518-520 `rcAfterResample == rcBeforeResample && relAfterResample != relBeforeResample → FAIL`（新規破壊なし条件下での +0 を明示 assert）+ :487 累積整合 `releaseDelta != reclaimDelta → FAIL`（無条件） |

補足: second-sampler の +0 assert に「新規破壊なし」前提条件が付くのは仕様どおり（背景破壊があれば sampler は転送すべき）。その場合も累積整合 assert（:487）が無条件に二重転送を検出する。

## 3. R3 — `observedOutstanding = A - R` の動的確認

- テストは `observedOutstandingEstimate() == (int64)A - (int64)R`（live getter 比較）を assert（:437-444）。
- delta 恒等式: `(A1-A0) - (R1-R0) = dAcquire - dRelease = N - N = 0`（窓内で収支ゼロ）。
- 絶対値 `outstanding = 1` は生存 World 分であり 0 を期待しない（指示どおり・実測も 1 で一貫）。

✅ 動的確認成立。

## 4. R4 — Terminal path coverage の再判定

### G2-1 テストが通る経路（source 確定）

```
publishOnce → commitRuntimePublication
  → onRuntimeRetiredNonRt (Commit.cpp:446)
    → runtimePublicationBridge_.retire(...) (:470)   // ISR retire chain Stage 1
      → DeferredDeletionQueue (D) enqueue
driveWorldRetirementReclaimForMeasurement
  → publishEpoch + tryReclaimResources → router->tryReclaim → D::reclaim
    → worldReclaimCount_++ (EpochDomain/D counter)
    → onRelease() → referenceReleaseCount_
ISRRetireRouter::worldReclaimCount() = provider_(D) + Q + E + Terminal.reclaimCount（集計）
  → shared transfer step → releaseObserved
```

**動的に検証されたのは D 経路の end-to-end。**

### Q/E/Terminal を G2 としない根拠

1. 測定観測経路は集計 counter 下流で**経路非依存**（どの store が壊しても同一 aggregate → 同一 transfer）。
2. 各 store の increment site は 9 LP 全て静的列挙済みで `onRelease()` と 1:1（R2 §2 / C2 §2-3）。
3. storage drain 機構は RetireGraceSemantics / StuckReaderFallbackDrain で動的カバー済み。
4. 残る未動作検証は各 store の trivial な `++` 1 行のみ。

→ per-path 動的観測テストの欠落は **G1 に留める**（G2 ではない）。

## 5. R5 — G1 再分類（先入観なし）

| 旧分類 | 再評価 | 新分類 |
|---|---|---|
| G1-1 observedOutstandingMax monotonic | **昇格**。writer `updateObservedOutstandingMax` の caller は Timer.cpp:444（timerCallback block）**のみ**。harness（`driveWorldRetirementSamplerForMeasurement`）は呼ばない（census で確認: harness calls = false）。→ ヘッドレス全条件で `observedOutstandingMax_` は **0 のまま静止**。15-field contract #4・D91 基準 8（accumulated max）・**A_max candidate 記録の依存先**。measurement run を harness 経由で実施する場合、max が死んだまま候補値を記録することになる | **G2-2（新規）** |
| G1-2 cross-window isolation | 維持。計画中の characterization は 1 engine instance 1 window（normal/burst/jitter それぞれ fresh harness）で成立。Closed→Idle 遷移は C5-4 で静的确認済み。持続観測（D101 以降）で multi-window が必要になる時点で prerequisite 化 | G1 |
| G1-3 JSON 24-field export content | 維持。evidence file の忠実性の問題であり counter 正しさの問題ではない | G1 |
| G1-4 reference observer wiring | **解消**。G2-1 テストが `dRef == dReclaim == N` を実測 → onRelease が実 drain 経路で発火し referenceReleaseCount が進むこと（= wiring）を動的証明済み | G0（解消） |
| G1-5 exactly-1 acquire/release value assertion | **ほぼ解消**。静穏 baseline で `dAcquire==N` / `dRelease==N` を deterministic 実測。残置は背景併走時の単イベント粒度のみ | G1（縮小） |

追加記録: `setWindowTag` も timerCallback のみ（Timer.cpp:446/448）で harness 未接続。ただし tag は診断分類であり counter 値に影響しないため G2-2 の修復時に同時対応すればよい（G1 相当）。

## 6. R6 — full test result の扱い（32 tests / 1 failure）

`#28 HeadlessAudioPathVerification` の失敗を以下の具体根拠をもって measurement contract 非影響と判定する:

1. **失敗内容**: `.github/scripts/cli-smoke-test.ps1` が `build-icx/ConvoPeq_artefacts/Release/ConvoPeq.exe`（**別ビルドツリー・icx ツールチェイン・Debug pipeline では rebuild されない**）を起動。app は既知の static-teardown `0xC0000005` で終了（スクリプトは許容）したが、log window 内に `[CLI_PERF_RAW] callbacks=` の正の callback count を確認できず throw。
2. **変更前からの環境問題**: R3 実装前の full build で既に icx/ipp 系環境失敗（`ipp.h` C1083 × 4 ターゲット等）を確認済み。本ツリーは Debug(MSVC) であり icx Release 成果物とは独立。
3. **G2-1 関連 target は全 PASS**: AudioEngineHarness --measurement=release 3/3、--measurement=all 6/6。
4. **retire/publish regression は全 PASS**: DeferredDeletionQueueReclaim / RetireGraceSemantics / StuckReaderFallbackDrain / ShutdownRetireIntentDrain / InvariantINV3INV5 / publish pipeline 一式。
5. 失敗経路（CLI smoke of icx build）は T1 telemetry / retire counter とコード共有なし。

## 7. C6 final gate 判定

| Gate | 条件 | 判定 |
|---|---|---|
| C6-R1 | G2-1 deterministic test 存在し PASS | ✅（3/3 PASS） |
| C6-R2 | R1 double-count 検出可能 | ✅（:416 guard 常設） |
| C6-R3 | Δreclaim = Δrelease（N=1/N=4） | ✅ |
| C6-R4 | repeated sampler duplicate transfer = 0 | ✅ |
| C6-R5 | observedOutstanding = A-R 動的確認 | ✅ |
| C6-R6 | **G2 candidate = 0** | ❌ **G2-2 発見** |
| C6-R7 | residual G1 明示 | ✅（G1×3: cross-window / JSON export / 単イベント粒度） |
| C6-R8 | production measurement authority 不変 | ✅（mutation/writer census 再確定） |
| C6-R9 | 今回変更による既存 regression = 0 | ✅（31/32、#28 は非関連環境） |

```
T1-C6 = FAIL 継続（G2 = 1: G2-2）
```

## 8. G2-2 の定義と最小修復要件（設計指示のみ・本タスクでは実装しない）

**G2-2**: `observedOutstandingMax`（15-field #4・accumulated）の writer 経路が timerCallback 専用のため、test harness から到達不能かつ動的カバレッジゼロ。A_max candidate の記録源として機能しない。

修復オプション（いずれか）:

- **Option A（推奨）**: production timerCallback と同じ順序（transfer → estimate → max → tag → samplerTick）を共有化し、harness も同一 step を実行する。`updateObservedOutstandingMax(estimate)`（+ `setWindowTag`）を共有 step に含める。production 挙動は不変（同一コード呼び出しへの置換のみ）。
- Option B: A_max candidate の source を windowMax/finalEstimate 系に再定義する設計変更 — D91 基準 8 の accumulated semantic との整合再審査が必要なため重い。

加えて monotonic assertion test（同一窓内で max が減少しないこと）を G2-2 解消の完了条件とする。

## 9. 進行状態

```
C1-R3 PASS → C2 PASS → C3 PASS → C4 PASS → C5 PASS
  → C6-G2-1 RESOLVED → C6-R1 re-audit → G2-2 発見 → C6 FAIL 継続
        ↓
G2-2 repair（shared step へ max/tag 追加 + monotonic test）
        ↓
C6 再判定（G2=0）→ measurement run 解禁 → A_max candidate（R は UNDETERMINED のまま）
```

- 本 re-audit でのコード変更: **0**。
- Tool coverage: node fs census（41 files）/ rg・sed・awk（WSL）/ source 直読（Commit.cpp・Timer.cpp・AudioEngine.h・Telemetry.h・Reference.h）/ serena・AiDex・semble・ccc・graphify は C2〜G2-1 で確定済みのシンボル参照との整合維持 / 文献 9 系統は前 Gate までに 200 OK 済。

---

*Evidence generated: Phase I-T1-C6-R1 — no code change. G2-2 解消まで measurement run には進まない。*
