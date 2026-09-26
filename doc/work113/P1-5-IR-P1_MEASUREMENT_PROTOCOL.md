# P1-5-IR-P1 — Measurement Protocol Freeze

- **作成**: 2026-09-22 / work113 P1-5-IR Phase 1
- **性質**: 実測を行わない。P0(`P1-5-IR-P0_PROTOCOL_RECONCILIATION.md`)で確定した lifecycle を、**既存 instrumentation の観測可能範囲のみ**で記述可能な measurement contract として凍結する。
- **基準 source**: `ConvoPeq.md` Generated 2026-09-22 06:20:11 相当の現行 working tree。
- **絶対条件(凍結時点で確認済み)**: HEAD = 1e9e63e3・production source / CMake / default / calibration / harness / commit = **0**。flag cache = OFF。本ドキュメント作成自体も上記を変更しない。

---

## 1. Test environment

| 項目 | 値 |
| --- | --- |
| repository | `C:\VSC_Project\ConvoPeq`(git・HEAD `1e9e63e3`) |
| build dir | `build\`(Ninja Multi-Config・MSVC 1951 / C++20 / AVX2) |
| authorized binaries | `build\Release\AudioEngineHarness.exe`(OFF K2 build)・`build\Release\AudioEngineHarness_on.exe`(ON K2 build) |
| flag cache | `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=OFF`(維持義務) |
| test IR assets | 既存: `tmp/p15_ir_g0.wav`(Step 5-A で使用・実在確認済み)、`writeH01TempIr`/`writeSR01TempIr` が生成する一時 IR(test 実行時に生成) |
| env setup | `vcvars64.bat`(VS 18 Enterprise) + `oneAPI setvars.bat intel64`(MKL/IPP) |

## 2. Binary identity

- binary identity の evidence は **実行ログの flag_macro 行を原則とする**。`--p1-char` モードは先頭行に `[P1CHAR] flag_macro=0|1 (0=OFF build / 1=ON build)` を出力する(P0〜HR で使用済み)。
- `--buzz` 系ランは flag_macro を出さないため、P2 の identity は次の 3 点組で担保する:
  1. 実行 exe の絶対パス + SHA-256(記録時点)
  2. `build\CMakeCache.txt` の `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=OFF` 値
  3. build provenance(本 protocol に紐づく build ログ `tmp/p1_5_hr_step*_off_build.log` の成功記録)
- **ファイル名・タイムスタンプ単独は evidence としない**(P1-5-HR Step 0 の規律を引き継ぐ)。

## 3. Existing instrumentation used(実在確認済みのみ)

### 3.1 harness 内 settle ヘルパ(`src/tests/AudioEngineHarness/BassBuzzMeasurement.cpp`)

| 関数 | 行 | 機能 | 契約上の扱い |
| --- | --- | --- | --- |
| `waitBacklogZero(e, ms)` | :86 | `getPublicationBacklogCount()==0` 待ち | L2 の必要条件(十分条件ではない) |
| `waitWorldPublished(e, before, ms, tag)` | :107 | commit seq 前進 + backlog 0 + **300ms 安定保持**を要求。`[BUZZ] %s: world seq %lld -> %lld committed` を出力 | L2 の seq 観測の一次手段 |
| `waitIrFinalized(e, ms)` | :134 | **UI 側** `isIRFinalized()` のみ poll | RT 到達の保証なし(WORK104-R2 コメント明言) — 値の記録のみに使用 |

### 3.2 UI Convolver 状態の accessor(L1)

`isIRLoaded` / `isLoadingIR` / `isIRFinalized` / `getIRLength` / `getIRName` / `getPreparedSampleRate`(= `currentSampleRate` atomic)/ `getPreparedBlockSize`(= `currentBufferSize` atomic)/ `getStructuralHash` / `getLastPreparedIRApplyTicks` / `getIrFreqPeakGainDb`(`ConvolverProcessor.h:431-465, 1256-1263`)。

### 3.3 CLI 経路

| 旗 | 位置 | 動作 |
| --- | --- | --- |
| `--cli-ir <path>` | `MainWindow.cpp:777` | 200ms 遅延で `requestConvolverPreset`(= `uiConvolverProcessor.loadIR`= RCU 系統 B)。`[CLI_IR] isIRLoaded/irLen` を直後に記録 |
| `--cli-ir-reload-count` / `--cli-ir-reload-interval-ms` | `MainWindow.cpp:806-819` | IR 再ロード storm(既定 0 / 300ms) |
| `--cli-rebuild` | `MainWindow.cpp:1030` | **500ms** 後に `[CLI_REBUILD] isIRLoaded/irLen` をログして `requestStructuredRebuildIntent(Structural)` を強制発行 |
| `--cli-exit-ms` | `MainWindow.cpp:1059` | IR/rebuild 使用時は最小 3000ms に自動底上げ |
| `--buzz-*` 一式 | `BassBuzzMeasurement.cpp:1579-1614` | `--buzz-sr= --buzz-block= --buzz-ir= --buzz-out=<csv> --buzz-quick --buzz-rigcheck[=bare] --buzz-probe=delta\|boost\|silence --buzz-probe-level= --buzz-probe-signal= --buzz-quiet=<ms>(既定150000) --buzz-dur= --buzz-order=cte\|etc(fail-closed) --buzz-eq=on\|off --buzz-conv=on\|off --buzz-hc= --buzz-lc= --buzz-eqlpf= --buzz-direct= --buzz-flip-*=(hc/lc/eqbypass/eqgain @ flip-t)` |

### 3.4 ログトレース(既存)

| トレース | 生産箇所 | 内容 |
| --- | --- | --- |
| `[IR_TAIL_GEOM] gen=%llu loadedSr=%.0f loadedLen=%d targetLength=%d copySamples=%d` | `ConvolverProcessor.LoaderThread.cpp:759` | LoaderThread の geometry(gen は LoaderThread 側生成 ID) |
| `[L0_WRITE] gen ir=%d val=... / geom part=%d numIR=%d numParts=%d fft=%d imm=%d irLen=%d / slot=%d peak=%.6f bin=%d` | `MKLNonUniformConvolver.cpp:1068-1087` | RT partition 構築の実データ |
| `[DIAG_IR] applyComputedIR: ... / loadIR: ...` | `LoadPipeline.cpp` | payload 生成・SR 不一致・generation mismatch・validation 失敗 |
| `[DIAG] convolverParamsChanged: enter / SUPPRESSED / hash dedup / DEFERRED / requestRebuild Structural hash=` | `AudioEngine.UIEvents.cpp:55-190` | needsStructuralRebuild 判定の全分岐 |
| RebuildTelemetry(`Suppressed / Merged / Deferred / Dispatched / Released`)と理由(`DeferredStructuralRebuildRequested / SnapshotEnqueued / PreparedIRApplyWindow / HashDedup / MixedPhaseIntermediate`) | `AudioEngine.UIEvents.cpp`・`AudioEngine.Timer.cpp:774-810` | structural rebuild の要求〜解放チェーン |
| `[VERIFY] EQ reflection createdHash=... dspReady=...` | `AudioEngine.Timer.cpp:758-767` | world 構築の dspReady 観測 |
| `[BUZZ] ... world seq %lld -> %lld committed / quiet period %d ms ...` | `BassBuzzMeasurement.cpp` | harness 側 publish 追跡 |
| `[H01] diag: loading=%d finalized=%d irLen=%d algo=%d peak...`(`pollH01Peak`)・SR-01 diag(`pollSR01Load`) | `ConvolverStateRoundTripTests.cpp:699, 993` | bare `ConvolverProcessor` 単体の L1 観測 |
| `writeH01TempIr(tag, peakPos)`(:604・4800 samples/48k/2ch)/ `writeSR01TempIr(tag, sr, ch, samples, peakPos)`(:955) | 同 | 既存 IR 生成資産(P0 指示で言及・**実在確認済み**) |

## 4. Measurement matrix(P2 で実行)

新規条件は追加しない。凍結する最小集合:

### 4.A 同一 IR / 同一 geometry / 反復 run(最低 3 run)

```text
IR      : 同一ファイル(実在資産・SHA-256 固定)
OS      : 1x 相当の既存 scenario 条件
SC      : off 相当(--buzz-eq=off 等の既存旗のみ)
input   : 既存 scenario の 2 水準(−20 dBFS 相当 / −6 dBFS 相当)
```

- run N / N+1 / N+2 ごとに `gainDb / irLength / irFinalized / publicationSeq / generation` を**同一観測行**に記録。
- OS=1x の pin に既存旗がない場合、P2 実行前に `BassBuzzMeasurement.cpp:1579-1614` の parser を単一の source of truth として当該条件への割当を確定する(**旗の追加はしない**)。割当不能な条件はその run を UNKNOWN 記録とする。

### 4.B run-order test

```text
順序1: A(-20) → B(-6) → A(-20) → B(-6)
順序2(可能なら): B(-6) → A(-20) → B(-6) → A(-20)
```

- 目的は順序依存性の検出。平均値は出さず、**各 run の seq を含む行を生のまま保存**する。

## 5. Run ordering

- 1 run = 1 プロセス起動(既存旗のみ)。同一 IR・同一旗の連続 run を最低 3 回。
- run の識別: `runId = <binary>_<irAsset>_<level>_<index>`。各行に起動時刻・exe SHA-256・CMakeCache 値を併記。
- **平均・最大・最小による要約は観測行の保存後に行ってもよいが、判定材料は個別行**。

## 6. Required observables(4 層)

| Layer | 対象 | 既存 instrumentation での取得 | 取得不能時 |
| --- | --- | --- | --- |
| L0 | IR input / WAV nominal(パス・SR・ch・samples・nominal peak) | ファイル検査 + `[IR_TAIL_GEOM]`(loadedSr/loadedLen) | — |
| L1 | `isIRLoaded / isIRFinalized / irLength / sampleRate / generation` | 3.2 の accessor + `[CLI_IR]/[CLI_REBUILD]/[IR_TAIL_GEOM]` ログ | **generation の run 直接値は RCU 系統で未ログ → UNKNOWN**(LoaderThread 系は `[IR_TAIL_GEOM] gen=` で観測可) |
| L2 | publication sequence + DSP projection | `waitWorldPublished`(seq 前進+backlog+300ms 安定)+ RebuildTelemetry(Dispatched/Released + Structural 系 reason) | **IR-structural world と通常 world publish の識別が不能な行は UNKNOWN**(P0 既知問題) |
| L3 | 実出力: existing DFT `gainDb` / output peak / RMS / WAV features | 既存 measurement 出力(付加実装なし) | — |

- **L0→L1・L1→L2 の利得を推測で補完しない**。差分が観測されても段階帰属は UNKNOWN。

## 6a. 1.48 dB definition(凍結)

```text
measurement-level delta = OBSERVED
IR internal gain error   = UNKNOWN
```

- 「1.48 dB」= Step 5-A/P0 で記録された **measurement output gainDb の 2 行間差**(同一 IR g0・os=1・sc0・amp=−20 の −14.5035 dB vs amp=−6 の −13.0276 dB)。
- 本 protocol の全工程で **1.48 dB を「IR gain error」と呼ばない**。P2 の matrix A/B が分離する対象は:
  1. measurement artifact
  2. runtime IR state mismatch
  3. actual IR gain difference
  の 3 分岐であり、P2 が完了するまでいずれにも帰属させない。

## 6b. 1-run-lag definition(凍結)

各 run を次の 4 点で観測する(同一観測行):

```text
run N:   requested IR = X / UI finalized = X? / Runtime IR = X? / output = X?
run N+1: requested IR = Y / UI finalized = Y? / Runtime IR = Y? / output = Y?
```

- **Runtime(X) が run N+1 の観測に初めて現れる** → `1-run-lag = OBSERVED`
- 単なる output amplitude 差のみで状態の遅延が示せない → `1-run-lag = NOT REPRODUCED`
- 「そう見える」だけの記述は **INFERRED にしない**。Runtime IR の同定は telemetry reason/class + `[IR_TAIL_GEOM]`/`[L0_WRITE]` の gen 対応で行い、識別不能なら UNKNOWN。

## 6c. isIRFinalized observation rule(凍結)

各 run で最低 4 点を記録: `before load / after request / after finalize / before measurement`。

- 系列 `previous run=true → new IR request → still true` を観測した場合 → `stale candidate = OBSERVED`。
- ただし **`stale == cause of 1-run-lag` は別 claim**(P2/P3 で別途判定)。本 protocol では同一視しない。

## 6d. publication sequence rule(凍結)

- 最低限 `seq_before / seq_after / IR generation / irLength / irFinalized` を**同一観測行**に置く。
- **`publicationSeq advanced` と `IR-specific structural world published` を同一視しない**。IR-structural world の識別は RebuildTelemetry(event=Deferred/Dispatched/Released・reason=DeferredStructuralRebuildRequested 等・class=Structural)と `[IR_TAIL_GEOM]/[L0_WRITE]` の対応で行い、**識別不能な publish は UNKNOWN** として記録する(P0 確認済みの未解決問題)。

## 6e. IR geometry fields(凍結)

各観測行に最低限: `sample rate / processing rate / OS / IR length / block size / FFT・partition geometry / generation` を既存ログから収集する。

- `[IR_TAIL_GEOM]`・`[L0_WRITE]` を第一級 trace とする。
- IR length が run ごとに変わる場合、`same nominal IR ≠ same runtime IR geometry` の可能性を記録する — ただしその事実だけで **gain mismatch の原因とは直ちにしない**(分岐候補として記録)。

## 7. Output measurement(出力側)

- 既存 DFT measurement のみ(追加実装禁止)。
- 各 run: `ampDb / gainDb / output peak / RMS` を取得できるなら保存。
- `gainDb(N) − gainDb(N+1)` の算出は可。ただし **measurement output delta** であって IR gain の差ではない。

## 8. 判定ルール(P1 で使用する 4 分類)

| 分類 | 定義 |
| --- | --- |
| **OBSERVED** | ログまたは deterministic measurement で直接確認できる |
| **INFERRED** | 複数の OBSERVED facts と整合するが、直接測定ではない(必ず根拠行を列挙) |
| **NOT REPRODUCED** | 同一条件を複数回実行して anomaly が再現しなかった |
| **UNKNOWN** | 既存 instrumentation では比較不能 |

## 9. Stop conditions

以下は P1/P2 を通じて即停止する:

1. binary identity の不一致(flag_macro / SHA-256 の不一致)
2. 整定時間・sleep・settle の変更によって anomaly が消えた状態での計測(bug disappeared と protocol masked の区別が不能になるため — **最重要禁止**)
3. 新しい logging/accessor/WAV writer を使わざるを得ない状況の発生
4. 未説明の新 artifact の出現
5. IR 補正・gain compensation・normalization への誘惑
6. Option E / D2 / D4 判断への進入

## 10. PASS 条件

```text
[x] measurement matrix が固定された(§4 — 新規条件なし・既存旗のみ)
[x] 1.48 dB の比較定義が固定された(measurement-level delta = OBSERVED / IR internal gain error = UNKNOWN)
[x] 1-run-lag の観測定義が固定された(4 点チェック・Runtime 適合の trace 遡及)
[x] isIRFinalized の観測方法が固定された(4 時点記録・stale candidate 分離)
[x] publicationSeq の解釈が固定された(同一行記録・seq 前進 ≠ IR world)
[x] IR geometry fields が固定された(§3.3/6e・[IR_TAIL_GEOM]/[L0_WRITE] を trace とする)
[x] existing instrumentation だけで実施可能と確認された(§3 の資産は全て実在確認済み)
[x] production/CMake/harness/commit = 0(本ドキュメントは読み取り監査のみ)
```

**P1-5-IR-P1 = PASS。ここで停止する。** P2(実測)はユーザーの指示を受けてから開始する。P2 の実行形態(正確な旗の組合せ)は、実行直前に `BassBuzzMeasurement.cpp:1579-1614` の parser を単一の source of truth として確定する(旗の追加・変更は行わない)。
