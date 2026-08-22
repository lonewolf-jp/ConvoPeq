# Phase I-T1-C6-G2-1 — Authoritative Release Observation Test Repair

> **Verdict: G2-1 = RESOLVED**
> C6 の blocking issue（G2-1: authoritative release observation path の動的テスト欠落）を解消した。
> production measurement authority は変更せず、delta-transfer を production/test 共通 step 化し、
> R1 double-count を自動検出する deterministic regression test を追加。
> **measurement run / A_max candidate / R は引き続き禁止**（C6 再判定 → evidence re-audit 後に解禁判定）。

## 1. Step 1 — production path 再確認（変更前・A〜E）

| 項目 | 確認結果 |
|---|---|
| A. delta-transfer の唯一性 | `Timer.cpp` inline block（旧 :423-431）が唯一の `releaseObserved` writer 経路 |
| B. harness の現状 | `driveWorldRetirementSamplerForMeasurement()` は `samplerTick()` + reference 同期のみ（transfer 欠落） |
| C. reclaim helper | `driveWorldRetirementReclaimForMeasurement()` = publishEpoch + tryReclaimResources（terminal deleter 発生用） |
| D. reference observer | `onRelease()` は `referenceReleaseCount_` のみ更新・releaseObserved へ転送しない（R3 確定） |
| E. test/production 区別 | 修正対象は test harness の observation step。Telemetry authority は変更しない |

## 2. Step 2 — shared delta-transfer 共通化

**採用形:** production と test が同一コードを呼ぶ共有メソッド化（コピーによる semantic drift を回避）。

```cpp
// AudioEngine.Timer.cpp（新設）
void AudioEngine::transferWorldReclaimDeltaForTelemetry() noexcept
{
    auto& telemetry = worldRetirementTelemetry();
    const uint64_t worldReclaimed = m_retireRouter ? m_retireRouter->worldReclaimCount() : 0;
    const uint64_t prevReclaimed = convo::consumeAtomic(lastSampledWorldReclaimCount_, acquire);
    if (worldReclaimed > prevReclaimed)
    {
        telemetry.addReleaseObserved(worldReclaimed - prevReclaimed);
        convo::publishAtomic(lastSampledWorldReclaimCount_, worldReclaimed, release);
    }
}
```

- `timerCallback()` は inline block を削除し `transferWorldReclaimDeltaForTelemetry()` 呼び出しに置換（estimate/max/windowTag/samplerTick の順序は不変）。
- `driveWorldRetirementSamplerForMeasurement()` は先頭で同一メソッドを呼ぶように変更。
- **`lastSampledWorldReclaimCount_` の ownership は不変**（AudioEngine member のまま・writer は共有メソッド内の 1箇所のみ）。

### 変更後の静的不変量（全て再検証済み）

| 不変量 | 結果 |
|---|---|
| `releaseObserved_` mutation = Telemetry.h の 2メソッドのみ | ✅（Reference.h 0 / Timer.cpp 直接 0） |
| `lastSampledWorldReclaimCount_` writer 一意 | ✅（共有メソッド内 1箇所） |
| 共有メソッドの call site = 2（timerCallback + harness） | ✅（Timer.cpp:442 / AudioEngine.h:4954） |
| `onReleaseObserved()` production caller = 0 | ✅ |
| `addReleaseObserved()` caller = sampler 共有 step のみ | ✅（Timer.cpp:382 の 1件） |

## 3. Step 3-6 — deterministic regression tests（追加）

追加先: `src/tests/AudioEngineHarness/WorldRetirementMeasurementTests.cpp`。実行条件 `--measurement=release`（`all` にも組み込み）。

| Test | 内容 | assert |
|---|---|---|
| `testReleaseObservationSingle` (G2-1-A) | measurement baseline 安定化 → 1 publish → reclaim 確定待ち → sampler step | `Δreclaim==1`, `Δrelease==Δreclaim`, `Δreference==Δreclaim`, `Δacquire>=1`, outstanding 恒等式 |
| `testReleaseObservationMultiple` (Step5) | N=4 で同様 | `Δreclaim==Δrelease==Δreference==4` |
| `testNoDoubleTransfer` (Step6) | 2 round 各「publish→reclaim→sampler→sampler 再駆動」 | 全時点で `累積転送==累積破壊`、かつ新規破壊なしの再駆動で releaseObserved 不変 |

共通ヘルパー:

- `waitForWorldReclaimCount(e, target, timeout)` — RT reader の epoch 離脱・背景 CoordinatorLoop を吸収して destruction 確定を待つ。
- `stabilizeMeasurementBaseline(e)` — 起動時背景処理（Bootstrap/idle publish 回収・Structural rebuild intent 消化）を測定窓外へ排出し、主要カウンタ連続不変後に sampler を 1 回余分に回して **lastSampled cursor を現在世界破壊数へ同期**（C3 cursor semantics を利用した正規手順）。

### 初回実行で検出された実障害と対処（記録）

初版テストは `dAcquire=2 dReclaim=2 dRelease=3`（N=1）で FAIL。原因はテスト側の測定窓設計:

1. 起動直後は `lastSampledWorldReclaimCount_=0` のまま未転送の破壊が累積しており、最初の sampler step が過去分を一括転送する（production 正常動作）。
2. 背景 Structural rebuild intent が窓内で publish+destroy を追加。

→ production の欠陥ではなく**テストの baseline 設計不備**。`stabilizeMeasurementBaseline` による cursor 同期で解消（production コードは無変更のまま）。

## 4. Step 8 — ビルド・テスト結果

### Debug build

- `AudioEngineHarness` / `ConvoPeq` / retire 系 3 ターゲット: **成功**（警告は既存 C4458/C4996 のみ）

### targeted tests（--measurement=release）

```
[g2-single] N=1 dAcquire=1 dReclaim=1 dReference=1 dRelease=1 outstanding=1
[g2-multi]  N=4 dAcquire=4 dReclaim=4 dReference=4 dRelease=4 outstanding=1
[no-double-transfer] totalReleaseDelta=2 totalReclaimDelta=2
```

**3/3 PASS** — `ΔworldReclaimCount == ΔreferenceReleaseCount == ΔreleaseObserved == N`（2N は出現せず）。

### --measurement=all（既存条件への影響確認）

```
[normal] O_w=2 T_w=3 E_w=1   (T_w >= O_w ✅)
[burst]  O_w=2 T_w=2 E_w=0   ✅
[jitter] O_w=1 T_w=3 E_w=2   ✅
[g2-single]/[g2-multi]/[no-double-transfer] ✅
```

**6/6 PASS** — harness 変更後も normal/burst/jitter の既存 assert は成立。

### full CTest（Soak 除外）

```
97% tests passed, 1 tests failed out of 32
失敗: #28 HeadlessAudioPathVerification のみ
```

- `#28` は `build-icx` ツリーの Release 実行ファイルを起動する CLI-smoke であり、**本変更と無関係の既存環境問題**（R3 実装前のビルドでも同様の icx/ipp 系環境失敗を確認済み）。
- publish pipeline 一式（`AudioEngineHarness` デフォルトモード）: **PASS**（"all publish pipeline tests PASS"）。
- retire 系: DeferredDeletionQueueReclaim / RetireGraceSemantics / StuckReaderFallbackDrain / ShutdownRetireIntentDrain / InvariantINV3INV5 全て **PASS**。

## 5. Step 9 — G2-1 完了判定

| 条件 | 必須 | 判定 |
|---|---|---|
| destruction → worldReclaimCount が実際に増える | ✅ | PASS（dReclaim==N を deterministic に確認） |
| sampler/test step が production と同じ delta semantics | ✅ | PASS（同一メソッド `transferWorldReclaimDeltaForTelemetry` を共有・call site 2） |
| releaseObserved が Δreclaim と 1:1 | ✅ | PASS（N=1/N=4 で dRelease==dReclaim） |
| referenceReleaseCount が releaseObserved に加算されない | ✅ | PASS（dRef==N かつ dRelease==N＝加算なら 2N になるが不発生） |
| N destruction → N releaseObserved | ✅ | PASS（N=4） |
| sampler 再実行で二重加算しない | ✅ | PASS（no-double-transfer 一致・再駆動 +0） |
| `observedOutstanding = A - R` が動的に成立 | ✅ | PASS（恒等式を実測値で確認・outstanding=1 は生存 World 分で妥当） |
| R1 double-count regression を検出可能 | ✅ | PASS（dRelease != dReclaim 即 FAIL の guard を常設） |
| production measurement authority の変更なし | ✅ | PASS（Telemetry/Reference/storage counter は無変更・writer 一意性は静的 census で再確認） |

```
G2-1 = RESOLVED
```

## 6. Tool Coverage

| 系統 | ツール | 用途 |
|---|---|---|
| Sandbox | node fs census | 変更後の静的不変量（mutation/writer/call-site）を全数確認 |
| Build | MSVC 18.9.1 + oneAPI + CMake/Ninja（Debug） | AudioEngineHarness / ConvoPeq / retire 系ターゲット |
| Test | AudioEngineHarness --measurement=release/all、デフォルト publish pipeline、ctest 32 件 | 動的検証 |
| MCP | serena / AiDex / semble / ccc / graphify | C2〜C6 で確定済みシンボル参照との整合維持 |
| WSL | rg / sg / fdfind / ag / fzf / sed / awk | 静的 census 補助 |

## 7. 進行状態（次ゲート）

```
G2-1 RESOLVED（本文書）
        ↓
C6 evidence re-audit（G2=0 確認 → T1-C6 = PASS 再判定・Residual Gap = G1×5）
        ↓
T1 measurement run 解禁判定（normal / burst / jitter）
        ↓
A_max candidate 記録（R は未決定のまま）
```

- 実施しないもの: R / R_cap 決定、A_max 確定、T2、Recovery coalesce、I4 D12-17 実装、shutdown architecture 変更。
- G1（5件: max 単調 assert / cross-window isolation / JSON export 内容 / observer wiring / exactly-1 値 assert）は Residual Gap として C6 再監査時に明示。

---

*Evidence generated: Phase I-T1-C6-G2-1 — G2-1 resolved via shared sampler step + deterministic regression tests. Measurement run awaits C6 re-audit.*
