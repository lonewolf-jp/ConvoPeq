# D164 — D162-2-I1 Corrected Operational Shutdown Soak（Evidence）

```text
Task: D164 — D162-2-I1 Corrected Operational Shutdown Soak
Date: 2026-09-06
Type: operational validation（production source 変更 0）
Baseline: git 9cacee1f（2026-09-05 23:39）/ CMakeLists.txt = HEAD + H5 +10（/utf-8 RWDI・本作業で未変更）
Binary: build-diag RWDI ConvoPeq.exe
  - v1（無効）: sha256 3642813285859dce…（09-06 02:00・RUNTIME_DIAGNOSTICS=OFF）
  - v2（正式）: sha256 72cce20a759f29fcf2e025d988d88814107d181f1ec4daec2e028224433e3ad2（09-06 10:49・DIAG=ON 復元）
Verdict: **PASS — D120 historical teardown race は I1 profile（A×8/B×6/C×6/D×6）で再現せず**
```

## 0. ★ A0 での発見 — binary config drift（初回 26 runs 無効化）

I0 契約は「binary = build-diag RWDI（G4 と同一 config — lifecycle closure の観測可能性を確保）」を要求。
A0 freeze 後、初回 26 runs を完走したが、channel 4 解析中に **MEM_SNAP / [Seq=] / [TIMER] が全 run で 0 件**
であることが発覚（G4 log では MEM_SNAP 4130 件）。原因調査:

```text
build-diag/CMakeCache.txt: CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS:BOOL=OFF
  ← I3-4-H5（09-06 00:01）の reconfigure が ON フラグを消失させていた
    （D162-1P_diag_build.bat 自体が「cache deletion on compiler change wipes -D options:
      re-run configure to re-apply」と同じ pitfall を文書化済み。H5 は再適用しなかった）
影響: lifecycle closure（EBR pend/ovf・retire/destroy 台帳・remaining）が観測不能
     = I0 §4.3 channel 4 不成立 → 初回 26 runs は測定無効
処置: evidence/D164/invalid_v1/ へ quarantine（26 log 全件保存・破棄しない）
     evidence/D164/d164_diag_restore_build.bat で DIAG=ON 復元（production source/CMake 変更 0、
     build-diag cache のフラグのみ・コンパイラ cache 値は保持）→ RWDI rebuild 137 steps OK
     smoke run: MEM_SNAP 119 / Seq 36 / exit 0x0 / zone clean → 観測性回復確認
     → 全 26 runs 再実行（以下が正式データ）
```

教訓: **binary hash だけでなく compile-time 診断フラグの cache 値も freeze 項目に含めるべき**
（D165 の A0 checklist へ反映推奨）。

## 1. 実行条件（I0 §5 準拠）

- G4 production tree 変更 0（S3=ON / V-D=ON — 現行 src の clearDeferredForShutdown retire + V-D retire-not-destroy）
- exit code authority = PowerShell `$p.ExitCode -band 0xFFFFFFFF`（I0 channel 1・旧 %ERRORLEVEL% パターン不使用）
- crash dump = run 前後 `%LOCALAPPDATA%\CrashDumps` の ConvoPeq.exe.*.dmp スナップショット差分（channel 2）
  - baseline 7（A0 時点）→ 全 52 run 終了後も 7（新規 0）
- shutdown trace = `[D123] SHUTDOWN_BEGIN → mainWindow.reset() completed → LOGGER_DETACH / SHUTDOWN_END`（channel 3）
- lifecycle closure = MEM_SNAP（Ret: pend/ovf）+ D117_RETIRE/DESTROY + DSP_DESTROY_FOOTPRINT + DSP_FOOTPRINT_RELEASED + D162-2B_RETIRE(origin) + D162-2G2_VD_RETIRE（channel 4）
- CLI は既存 flag のみ（MainWindow.cpp:367-400 の定義内）

## 2. 結果サマリ（26 runs・正式 v2 binary）

| Profile | runs | exit 0x0 | new dump | zone clean | E-3 | Sig A/B/C | ovf | final pend | dup destroy | remaining≠0 | XRUN(shutdown窓) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A plain ×8 | 8 | 8/8 | 0 | 8/8 | 0 | 0 | 0 | 0 | 0 | 0 | 0（total 37・全て startup） |
| B IR+rebuild ×6 | 6 | 6/6 | 0 | 6/6 | 0 | 0 | 0 | 0 | 0 | 0 | 0（total 17） |
| C IR reload ×6 | 6 | 6/6 | 0 | 6/6 | 0 | 0 | 0 | 0 | 0 | 0 | 0（total 11） |
| D device cycle ×6 | 6 | 6/6 | 0 | 6/6 | 0 | 0 | 0 | 0 | 0 | 0 | 0（total 11） |
| **合計** | **26** | **26/26** | **0** | **26/26** | **0** | **0** | **0** | **0** | **0** | **0** | **0** |

per-run 明細 = `d164_b_reconciliation.json`。

### lifecycle closure の内訳（channel 4）

- **Profile A**（plain）: startup DSP 1 obj/gen（gen 3-5）→ V-D retire（`D162-2G2_VD_RETIRE target=active-final`）→ destroy 1 → remaining=0。publish なし。
- **Profile B**（IR+rebuild）: 3-4 obj/run（startup gen + rebuild gen 群）。retire 経路 = crossfade retire（retired=1+enqueue）+ handle retire（RETIRE_BY_HANDLE HIT→enqueue）+ **shutdown-clear retire（D162-2B_RETIRE origin=shutdown-clear・S3 発火 1/run）**。全 obj destroy 1 回・remaining=0・final pend=0。
- **Profile C**（reload×3+burst）: B と同型 + REBUILD_MERGED（coalesce）経路。全 closure 成立。
- **Profile D**（device cycle）: Windows Audio / DirectSound 交互（`CLI_AUDIO_DEV_SWITCH` 実測）。DirectSound run は loader placeholder DSP（gen=0・Init.cpp:116 の 48k/512 暫定値）も処分対象となり destroy 2 obj/run。全 closure 成立。

### 既知 benign パターン（G4 PASS log にも存在・STOP 該当でないことの確認済み）

| パターン | D164 | G4 実測 | 意味 |
| --- | --- | --- | --- |
| `[FAULT] coordinator in Faulted after markShutdownComplete` | 1/run | 1/r1, 1/r2 | 15-P-5 post-shutdown 診断（AudioEngine.CtorDtor.cpp:308-312）|
| `[D117_RETIRE] retired=0` | 0-1/run | 6/r2, 8/r1, 1/G3 | 二重 retire 試行（既に erase 済み・early return で安全）|
| `RETIRE_BY_HANDLE lookup=MISS` | 1-2/run | 12 lines | handle 未登録 = 期待される安全側結果 |
| `D162-2B_RETIRE origin=桳瑵…` mojibake | 全該当 run | 同一パターン存在 | `%s` origin ラベルの UTF-16/8 cosmetic（decode="shutdown-clear"）・機能影響なし |
| XRUN | 76 件 total | — | 全件 startup transient・**shutdown 窓 0 件**（I0 §5.1 充足）|

### analyzer 判定訂正記録（false positive 2 件）

1. `genmismatch`（初回 26 件検出）: `[PUBLISH] seq` マーカーは startup/intermediate gen を網羅しない
   （G4 の "destroy 61 = F+1" と同構造）。正 = per-object (pointer, gen) accounting。
2. `dup_destroy`（C_run2）: 同一 pointer の 2 回 destroy = **address reuse**（gen=4 obj 処分後に
   新 obj gen=6 が同アドレス採番・`DSP_FOOTPRINT phase=construct` 挟まる）。(pointer, gen) ペアでは
   重複 0。D run の gen_gap [0,N] も loader placeholder（gen=0）+ startup gen で正常。

## 3. D120 との対比（I0 §5 の統計比較）

| Profile | D120 historical | D164 現行 G4 tree |
| --- | --- | --- |
| A plain | crash 2/5 | **0/8** |
| B IR+rebuild | crash 3/3（最高頻度） | **0/6** |
| C / D | （D120 該当 profile なし） | 0/6・0/6 |

## 4a. ★ 追加発見 — DirectSound device-switch による releaseResources 二重 pass（transitionViolations=7）

change-scope 検証で `evidence/evidence/shutdown_trace.json`（app が shutdown 毎に書く runtime telemetry・
WorkingDirectory=evidence のため evidence/evidence/ に生成）が HEAD 比 `transitionViolations: 0→7` を
検出。I0 STOP リスト外だが lifecycle 観測面のため完全 attribution 実施:

```text
per-profile 帰属（各 1 run + JSON capture・全 exit 0x0）:
  A=0  B=0  C=0  D(Windows Audio)=0  D(DirectSound)=7（再現 2 回目・決定論的）

機構（text log 実測）:
  D1(WA): releaseResources 1 pass（stopRebuildThread×1・VerifyDrained×1・ABOUT_TO_EXIT×1）
  D2(DS): releaseResources 2 pass — 1st pass は SHUTDOWN_BEGIN より前。
    起動時 --cli-device-type DirectSound への実切替（WA→DS）で旧 device close →
    JUCE が releaseResources を呼ぶ → phase が ShutdownComplete(10) まで進行。
    shutdown 時 2nd pass で 7 箇所の backward 再遷移（AudioStopped→VerifyDrained）が
    ガードに却下 = ちょうど 7。最終 ShutdownComplete は t==c で受理。
  root: ISRShutdown.h:335 phase_{Running} は構築時初期化のみ・re-prepare reset 経路なし
       （initiateShutdown に caller なし）

影響評価: telemetry-only。ガードは設計どおり作動（状態破壊なし）、2nd pass でも drain 作業は
  完走（retire→destroy→remaining=0・final pend=0・ovf=0・zone clean・exit 0x0・XRUN=0）。
  26+5 runs で機能異常の証拠なし。D120 型 race（非決定・T1 crash）とは無関係。
  ※ I1-D（09-05）は device-type 固定=Windows Audio のため DS shutdown は初観測面。

分類: 新規 observation（residual register 候補）。修正候補 = prepareToPlay 時の
  ShutdownRuntime phase reset（または releaseResources idempotency 考慮）—
  本作業では実装しない（コード変更 0 遵守・次編集 window/D165 scoping 判断へ）。
```

## 4b. runtime telemetry 生成物（working tree）

soak 実行自体が `evidence/evidence/*.json`（shutdown_trace / retire_timeline / world_lifecycle_audit 等 8 ファイル）
を更新（app の runtime 出力・HEAD にも同種の最終実行値がコミットされている既存慣行）。
ソース変更ではない。DS 二重 pass の証拠は `evidence/D164/d164_trace_D2_ds.json` に copy 保存。

## 5. 結論（D164-C）

> **PASS。** 現行構成（G4 production tree・S3/V-D ON・DIAG binary 復元後）では、
> **D120 historical teardown race は I1 profile において再現しなかった**（26/26 exit 0x0・
> dump 0・zone clean・lifecycle closure 完全）。
>
> これは D120 race の「絶対的不存在」の証明ではない（I0 §3.3 の分類を維持:
> historical probable → current 非再現）。D120-7 の主容疑 zone T1 は 1 回も観測されなかった。

### 分岐

→ **D165 = I2 Long-run / Device-cycle Validation へ**（I0 §6 定義: 60-gen class long soak +
IR reload + rebuild burst + device cycle + repeated start/stop の組み合わせ）。

## 6. 変更範囲（実測）

```text
production source: 0 / test source: 0 / CMakeLists.txt: 0（H5 +10 のまま）/ build.bat: 0 / tool: 0
build-diag cache: CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS OFF→ON（測定機構の G4 config 復元・派生 build tree のみ）
新規 artifacts: evidence/D164/{d164_a0_freeze.json, d164_diag_restore_build.bat, d164_b_reconciliation.json,
  D164_I1_SHUTDOWN_SOAK_EVIDENCE.md, invalid_v1/×26} + evidence/D164_*.log×26 + D164_smoke.log + evidence/D164_soak.ps1
```
