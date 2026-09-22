# P1-1 実装報告書 — polyphase gain convention 補正（案E）の production flag 実装と OFF/ON characterization（2026-09-22）

- **工程**: P1-0 audit（`eeeba325`）→ **P1-1 実装 + OFF/ON characterization（本報告書）** → ユーザー GO（未取得）
- **性格（R12-8）**: 本報告書は **flag ON の実測証拠の取得** までを示す。**案E の採用・default ON・calibration・release handling は未決定（HOLD 継続）**。
- **commit 状態**: **P1-1 は未 commit**（実装と測定結果の分離のため。ユーザー指示待ち）。HEAD = `eeeba325`。
- **snapshot**: `ConvoPeq.md` を production 変更後に再生成（header **2026-09-22 08:12:33** / 5,485,382 B / `--check` **FRESH**）— Phase 1 の build identity 更新フローとして実施（Phase 0 freeze 記録 §6 は不変）。

---

## P1-1-A implementation（PASS）

| 対象 | 変更 | 行 |
|------|------|----|
| production | `src/CustomInputOversampler.cpp` **+3 / −0** | `interpolateStage()`: :557 `convValue *= 2.0;` の直後・denorm 判定の**前**に `#if CONVOPEQ_CORRECT_POLYPHASE_GAIN`（:558）/ `centerValue *= 2.0;`（:559）/ `#endif`（:560） |
| build | `CMakeLists.txt` **+8 / −0** | `option(... OFF)`（:133）＋ genex 伝播 3 箇所（`ConvoPeq` :1320・`AudioEngineHarness` :1945・`PolyphaseGainFidelityTests` :2026）＋ 測定 TU 登録（:1919） |
| test/tooling | `src/tools/build_identity_gate.py` **+15 / −8** | `production_flag` を `CMakeCache.txt` の実値から表示（:446-450）。**identity 判定ロジック（stamp / SDK / zero-deps の fail-closed）は不変** |
| test-only | `src/tests/AudioEngineHarness/P1PolyphaseGainCharacterization.cpp`（新規・untracked） | `--p1-char` で起動（hook: `PublishPipelineIntegrationTests.cpp:1118-1119 / 1129`）。production の既存 tap のみ使用（analyzer output tap = pre-dither/pre-limiter・`getOutputLevel()`・`getTotalLatencySamples()`） |

- **挿入位置の契約遵守**: `convValue *= 2.0` → `centerValue *= 2.0` → denorm clear → output の順序を維持（candidate reference `PolyphaseGainCandidateRef.h:449-451` と同一）。
- **commit 分離（Commit A `d6995b2b`）**: production/build/測定 TU を 1 commit に固定。`PublishPipelineIntegrationTests.cpp` は **P1-1 の 5 行（forward 宣言 3 行＋`--p1-char` dispatch 2 行）のみ** を stage し、同ファイルに既存の**外部変更 8 行（`runEqDirectDriveAttribution` 関連）は stage していない**（未 commit のまま保持）。
- **scope 遵守**: `src/audioengine/**`・`RuntimeBuilder.*`・`AutoGainPlanner.*`・SoftClip / Limiter / Convolver / `build.bat` は無変更（`git diff --name-only` で確認）。`build.bat` は不変。
- **逸脱 1 件（build 手順）**: `build.bat` の `-D` 引数は cmd が `=` で分割するため `-DCONVOPEQ_CORRECT_POLYPHASE_GAIN=OFF` を渡せない（実測: `[INFO] Extra CMake define: -DCONVOPEQ_CORRECT_POLYPHASE_GAIN` と `=OFF` が欠落し CMake が構成エラー）。対策として **flag 値は事前 `cmake -D...=OFF|ON` で CMakeCache に設定し、`build.bat Release nopause` は引数なしで実行**した（cache 実値で OFF/ON を担保・下記記録参照）。
- **full build について**: `cmake --build`（対象を限定した build）は **EXIT=0**。一方 `build.bat` の全 target build は **EXIT=255** で、失敗は **P1-1 と無関係な既存 target**（`ISRSemanticValidationTests` / `PublicationAdmissionTests` / `invariant_INV3_INV5Tests` — `ipp.h` / `mkl.h` の include が該当 target の CMake ブロックに無い既存構成要因）。本測定に必要な 2 target（`PolyphaseGainFidelityTests` / `AudioEngineHarness`）は正常にビルドされる。

---

## P1-1-B OFF baseline（PASS）

| 項目 | 実測 |
|------|------|
| CMakeCache | `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=**OFF**` |
| BUILD-ID | snapshot `ConvoPeq.md 2026-09-22 06:20:11`（測定時点）/ git_head `eeeba32`(parent `de95f33`) / working_tree `PRODUCTION src DIFF: src/CustomInputOversampler.cpp`（＝意図した Phase 1 変更）/ **production_flag = OFF (Phase 1: default OFF)** / shadow_candidate 0 / build_config `Debug;Release;RelWithDebInfo / cl / Ninja` |
| **Phase 0 baseline 再現** | `PolyphaseGainFidelityTests_off.exe` → **PASS=54 FAIL=0（exit 0）** |
| binary | 355,840 B / 2026-09-22 08:31 |

**判定: flag 実装は OFF build で Phase 0 と同一挙動（54/54 再現）** — `#if` が除去される設計の実証。

---

## P1-1-C ON characterization（PASS・実測）

| 項目 | 実測 |
|------|------|
| CMakeCache | `CONVOPEQ_CORRECT_POLYPHASE_GAIN:BOOL=**ON**` |
| BUILD-ID | snapshot `ConvoPeq.md 2026-09-22 08:12:33` / 5,485,382 B / **FRESH** / git_head `eeeba32` / **production_flag = ON (Phase 1: 明示指定 — characterization 用)** / shadow_candidate 0 / build_config 同上 |
| binary | 355,840 B（OFF と同サイズ）/ 2026-09-22 08:36 |

### Level（正弦 → chain 出力 DFT・同一入力で OFF/ON 比較）

| N | 周波数 | ampDb | OFF gainDb | ON gainDb | 差 | 理論値 20·N·log10(4/3) | 判定 |
|---|--------|-------|-----------|-----------|-----|------------------------|------|
| 1 | 50 Hz | −20 / −6 | −3.6824 | −1.1836 | **+2.4988** | +2.4988 | PASS |
| 1 | 1 kHz | −20 / −6 | −3.5063 | −1.0076 | **+2.4987** | +2.4988 | PASS |
| 1 | 10 kHz | −20 / −6 | −3.6207 | −1.1219 | **+2.4988** | +2.4988 | PASS |
| 2 | 50 Hz | −20 / −6 | −6.1789 | −1.1813 | **+4.9976** | +4.9975 | PASS |
| 2 | 1 kHz | −20 / −6 | −6.0048 | −1.0073 | **+4.9975** | +4.9975 | PASS |
| 2 | 10 kHz | −20 / −6 | −6.2211 | −1.2236 | **+4.9975** | +4.9975 | PASS |
| 3 | 50 Hz | −20 | −8.6773 | −1.1809 | **+7.4964** | +7.4963 | PASS |
| 3 | 1 kHz | −20 | −8.5035 | −1.0072 | **+7.4963** | +7.4963 | PASS |
| 3 | 10 kHz | −20 | −8.7511 | −1.2547 | **+7.4964** | +7.4963 | PASS |
| 3 | 50 Hz | −6 | −8.6773 | −1.1809 | +7.4964 | +7.4963 | limiter 接触域（record） |
| 3 | 1 kHz | −6 | −8.5035 | −1.0072 | +7.4963 | +7.4963 | limiter 接触域（record） |
| 3 | 1 kHz | **0** | −8.5035 | −1.4975 | **+7.0060** | +7.4963 | limiter 接触域（record: **−0.49 dB を limiter が吸収**） |
| 3 | 10 kHz | −6 | −8.7511 | −1.2547 | +7.4964 | +7.4963 | limiter 接触域（record） |

- **線形域は理論値 ±0.0003 dB で一致**（gate は ±0.05 dB）。
- **ON は factor に対し level-flat**（1 kHz: −1.0076 / −1.0073 / −1.0072 dB for N=1/2/3）— すなわち **factor 依存の level 変化（OFF の −2.4988 dB/stage）が消える**。
- 0 dBFS 行は limiter 吸収により理論差より 0.49 dB 小さい（**「最終出力が必ず +2.4988N dB」と単純化しない**という P1-0-E の規定どおり record 扱い）。

### Latency（OFF == ON 必須）

| route | reported (OFF/ON) | argmax (OFF/ON) | OFF==ON | argmax ∈ [floor(D), floor(D)+1] |
|-------|-------------------|-----------------|---------|----------------------------------|
| os2-IIR-N1 | 255 / 255 | 256 / 256 | **一致** | ✓ |
| os4-IIR-N2 | 287 / 287 | 287 / 287 | **一致** | ✓ |
| os8-IIR-N3 | 290 / 290 | 291 / 291 | **一致** | ✓ |
| os8-LP-N3 | 582 / 582 | 583 / 583 | **一致** | ✓ |

### SoftClip / Limiter（sat=1.0・1 kHz・1024/4096 サンプル窓）

| 条件 | 指標 | OFF | ON |
|------|------|-----|-----|
| os2 / −6 dBFS | outMax（clip off / on） | 0.3345 / 0.3338 | 0.4460 / 0.4419 |
| os2 / −6 dBFS | limiter 入力 peak（`getOutputLevel`, clip on） | −8.49 dB | −6.02 dB |
| os2 / 0 dBFS | outMax（clip off / on） | 0.6674 / 0.6328 | **0.8414** / 0.7285 |
| os2 / 0 dBFS | **limiter 到達数（clip off / on）** | **0 / 0** | **119 / 0** |
| os8 / −6 dBFS | outMax（clip off / on） | 0.1883 / 0.1901 | 0.4465 / 0.4425 |
| os8 / 0 dBFS | outMax（clip off / on） | 0.3758 / 0.3738 | **0.8414** / 0.7288 |
| os8 / 0 dBFS | **limiter 到達数（clip off / on）** | **0 / 0** | **119 / 0** |
| 全条件 | hard clamp 件数 | 0 | **0** |
| 全条件 | clip 作用の最大差（clipEngMax） | 0.0018〜0.0687 | 0.0182〜0.1473 |

- ON は 0 dBFS で **limiter 到達が発生**（119/4096 = 2.9%）し、出力 peak が limiter 天井 0.8413951287507587 に一致 → **D-5 の「limiter 到達量増加」が production chain で確認**された。
- **SoftClip を有効にすると limiter 到達は 0**（clip が peak を 0.7285 に抑える）→ clip と limiter の相互作用も記録。
- hard clamp は両 build で 0 件（安全鎖は破綻していない）。

### flag 間 sample 差（os8 / −6 dBFS / clip on・1024 サンプル）

- **diffCount = 1024/1024（100%）・maxDiff = 0.2509** — chain では flag 効果が全サンプルに及ぶ（利得差 (4/3)^3 ≈ 2.37 倍）。
- 参考: Phase 0 の OS 内部イベント比 **R_s = 1.013**（candidate/base の corruption event 数比）は OS 単体の指標であり、chain では saturated（100%）のため弁別力がない → chain では **limiter 到達数（0→119）と clipEngMax** を弁別指標とする（測定設計の明示的代替）。

### ON build での Phase 0 battery（正の対照）

`PolyphaseGainFidelityTests_on.exe` → **PASS=19 / FAIL=35**。FAIL はすべて **base 期待値が candidate 挙動になったことによる意図的差分**（例: `R17-4(C) constant-input DC … dcBase=1.000000000000 expect=0.750000000000 dcCand=1.000000000000`）。PASS 19 件は flag 非依存の不変条件（定数 bitwise・係数不変条件・安全性）。**「flag ON が production TU に到達して案E 意味論を実装している」ことの正の対照**として記録する。

---

## P1-1-D OFF→ON→OFF 再現性（PASS）

| 検査 | OFF#1 | ON | OFF#2 |
|------|-------|----|-------|
| CMakeCache | OFF | ON | **OFF** |
| Phase 0 battery | 54/54 | 19/54（意図的） | **54/54（OFF#1 と同一ログ）** |
| binary（Harness） | 41,172,480 B @08:31 | 41,172,992 B @08:36 | **41,172,480 B @08:42** |
| characterization metrics（level/latency/softclip/summary） | — | — | **OFF#1 と byte-identical** |
| sample dump（1024 点） | — | — | max\|diff\| **1.0e-07** / mean 1.3e-09 / 差ゼロ 928/1024（**dither/NS の LSB ノイズのみ**） |

- **CMake cache 汚染検出ゲート**: ON を挟んだ後の OFF#2 が OFF#1 と同一（cache・54/54・metrics 完全一致）→ **cache 切替の残留なし**。
- sample レベルの LSB 差は dither / noise shaper（32 bit・乱数）由来であり、機能差ではない（metric 行は完全一致）。

---

## 判定（4 層）

```text
P1-1-A implementation      = PASS
P1-1-B OFF baseline        = PASS（54/54 再現・cache=OFF）
P1-1-C ON characterization  = PASS（level 理論一致・latency 不変・limiter 到達量増加を確認）
P1-1-D OFF→ON→OFF          = PASS（cache/54/54/metrics 完全再現・LSB ノイズのみ）
```

**停止条件 10 項目の確認（すべて非該当）**: ①挿入位置不変 ②production 変更は `CustomInputOversampler.cpp` のみ（+3/−0）③flag は 3 target のみへ伝播 ④default OFF 維持（option 既定＋OFF cycle の cache 実値）⑤OFF で 54/54 を 2 回再現 ⑥ON の BUILD-ID/cache 確定 ⑦latency OFF==ON ⑧既存 gain compensation の新規発見なし ⑨SoftClip/limiter は予測どおり（破綻なし）⑩snapshot は再生成済みで FRESH（source と一致）。

**これは案E採用の確定ではない。** 以下は **HOLD 継続**（本報告書では判断しない）:

```text
案E の最終採用 / default ON / calibration / release-note の実装
F-3 / F-4 / O-20 / AutoGainPlanner / SoftClip threshold / Limiter threshold
headroom / makeup / Convolver trim
```

---

## 付録: 測定の限界

- `limiter 入力 peak` は `AudioEngine::getOutputLevel()`（**peak 定義**・pre-dither）を使用。真の limiter 入力は post-dither のため、θ 交差数は pre-dither 近似。
- chain 出力 tap（`DSPCoreDouble.cpp:565-566`）は **pre-dither / pre-limiter** であり、最終出力（post-limiter）とは dither・limiter・kOutputHeadroom（−1.0 dB）分だけ異なる。
- up ドメイン pre-clip 内部量（clip 入力 drive・Padé arg clamp 数）は chain から観測不能 → Phase 0 record（tanhClamp 18,200 件）を継承し、chain では clipEngMax と limiter 到達数で代替（P1-0 で承認済みの設計選択）。
- 測定は 48 kHz / block 512 / identity EQ + conv bypass（wet 経路）/ autoGain off + staging 0 dB の固定条件。limiter・SoftClip は非線形のため、他の設定では差分が変わり得る。
- full-project build（`build.bat` 全 target）は無関係な既存 target の include 不足で EXIT=255（本報告の測定は対象を限定した build で実施・EXIT=0）。
- 聴感評価（listening）は未実施。
