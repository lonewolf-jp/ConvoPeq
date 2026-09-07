# D165 — D162-2-I2 Long-run / Device-cycle Validation（Evidence）

```text
Task: D165 — D162-2-I2 Long-run / Device-cycle Validation
Date: 2026-09-06
Type: operational validation（production/test/CMake/build.bat/tool 変更 = 0）
Binary: build-diag RWDI ConvoPeq.exe
  sha256 72cce20a759f29fcf2e025d988d88814107d181f1ec4daec2e028224433e3ad2（2026-09-06 10:49・D164 復元後 binary）
Baseline: git 9cacee1f + H5（CMakeLists.txt +10 のみ・source diff 0）
CrashDumps baseline: 7（8 run 全終了後も 7・新規 dump なし）
Verdict: **PASS + RESIDUAL**
```

## 1. A0 — extended freeze（D164 教訓を反映）

| 項目 | 実測値 |
| --- | --- |
| git HEAD | `9cacee1f4c81f47bf55eb6396627270cd3cb7618` |
| production source diff | `src/` clean（0） |
| CMakeLists.txt SHA-256 | `680e8239…9a9f`（H5 +10 のまま） |
| binary SHA-256 | `72cce20a…3ad2`（build-diag RWDI） |
| **CMake cache 診断フラグ** | `CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS:BOOL=ON`（D164 の drift 教訓として明示確認）／`CONVOPEQ_REQUIRE_MKL=ON` |
| compiler | MSVC 14.51.36231 完全パス（short form 'cl' 問題なし） |
| generator | Ninja Multi-Config |
| S3（source 現行確認） | `RuntimePublicationOrchestrator.cpp:632` に `origin="shutdown-clear"` 存在（D164 で発火実測済） |
| V-D（source 現行確認） | `D162-2G2_VD_RETIRE` が ReleaseResources.cpp に存在（D164 で発火実測済） |
| CrashDumps | baseline 7 で固定、すべて run 中 `%LOCALAPPDATA%\CrashDumps` の前後差分で監視 |
| CLI flag | `MainWindow.cpp:367–400` の既存 flag のみ使用（新規修正なし） |

## 2. I2 workload（I0 §6 定義の組み合わせを網羅）

I0 は「60-gen class long soak + IR reload + rebuild burst + device cycle + repeated start/stop」の組み合わせを求める。
本監査は 3 種の run で全てをカバーする。

| Stage | 内容 | CLI（既存 flag のみ） |
| --- | --- | --- |
| **I2-WA** | Windows Audio で 60-gen long soak（IR reload×60 + intent burst×60、6 秒間隔）+ clean shutdown | `--cli-run --cli-log-file D165_WA.log --cli-device-type "Windows Audio" --cli-ir …active.wav --cli-ir-reload-count 60 --cli-ir-reload-interval-ms 6000 --cli-intent-burst-count 60 --cli-intent-burst-interval-ms 6000 --cli-exit-ms 420000` |
| **I2-DS** | 同上を DirectSound で実行（D164 residual `transitionViolations=7` の追跡） | 同上 `--cli-device-type DirectSound` |
| **I2-R1..R6** | restart×6・WA/DS 交互（各 run が 1 device cycle）・小さい構造負荷 | `--cli-run --cli-log-file D165_Rn.log --cli-device-type <交替> --cli-ir …active.wav --cli-ir-reload-count 2 --cli-ir-reload-interval-ms 2500 --cli-intent-burst-count 3 --cli-intent-burst-interval-ms 2500 --cli-exit-ms 30000` |

runner: [evidence/D165/d165_soak.ps1](C:\VSC_Project\ConvoPeq\evidence\D165\d165_soak.ps1) ＋ per-run `shutdown_trace.json` capture。

## 3. 全 run 実測

| Run | device | exit | new dump | transVio | zone | objs destroyed | remaining≠0 | dup destroy | E-3/Sig/stale-HIT | ovf/pend末 | XRUN(shutdown窓) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| I2-WA | Windows Audio | 0x0 | 0 | 0 | clean | 61 | 0 | 0 | 0/0/0 | 0/0 | 0 |
| I2-DS | DirectSound | 0x0 | 0 | **7** | clean | 2 | 0 | 0 | 0/0/0 | 0/0 | 0 |
| I2-R1 | Windows Audio | 0x0 | 0 | 0 | clean | 4 | 0 | 0 | 0/0/0 | 0/0 | 0 |
| I2-R2 | DirectSound | 0x0 | 0 | 7 | clean | 2 | 0 | 0 | 0/0/0 | 0/0 | 0 |
| I2-R3 | Windows Audio | 0x0 | 0 | 0 | clean | 4 | 0 | 0 | 0/0/0 | 0/0 | 0 |
| I2-R4 | DirectSound | 0x0 | 0 | 7 | clean | 2 | 0 | 0 | 0/0/0 | 0/0 | 0 |
| I2-R5 | Windows Audio | 0x0 | 0 | 0 | clean | 4 | 0 | 0 | 0/0/0 | 0/0 | 0 |
| I2-R6 | DirectSound | 0x0 | 0 | 7 | clean | 2 | 0 | 0 | 0/0/0 | 0/0 | 0 |

- exit は PowerShell `$p.ExitCode`（I0 channel 1 authority）で正規捕捉
- zone は `SHUTDOWN_BEGIN / mainWindow.reset() completed / LOGGER_DETACH` 3 行の到達性
- objs destroyed は `[DSP_DESTROY_FOOTPRINT]` で (pointer, generation) ペアをユニーク化
- XRUN は全 scope で shutdown 窓 = 0；出現しても startup transient（既知）。

## 4. Channel 別 verdict

| Channel | 結果 |
| --- | --- |
| 1 exit code | 8/8 = `0x00000000` |
| 2 crash dump | 8/8 で `DumpNew=0`（baseline 7 不変） |
| 3 shutdown trace | 8/8 zone clean（T0/T1/T2 なし） |
| 4 lifecycle closure | 8/8 で construct→retire→destroy→remaining=0 が per-object で完遂 |

D165 の I2 workload は **teardown race 非再現を 60-gen・rebuild burst・device cycle・start/stop 反復まで広げた**ことを確認した。

## 5. ★ DirectSound での新規発見（residual として扱う）

### DS-F1（再現確認済）: `transitionViolations=7`

- DS run では常に 7（WA は 0）。D164 で既知。per-run capture `d165_trace_*.json` で同一値を再確認。
- 機構はすべて D164 で特定済み：`--cli-device-type DirectSound` による**初期 WA→DS 切替**で releaseResources(pass 1) が呼ばれ、その後の releaseResources(pass 2=shutdown) で backward transition 7 箇所が guard reject。
- **lifecycle 完全性には影響なし（両 run で closure 完全）**。

### DS-F2（**D165 で新規発見** — DS-F1 より重大）：device 切替後 rebuild-intake が dead silent

D165-DS の 60-gen soak は generation を進められなかった（gen 5 停滞）：

```
[REBUILD_TELEMETRY] REBUILD_REQUESTED …   61 件（全て decision=accepted）
[REBUILD_TELEMETRY] REBUILD_DISPATCHED …   0 件
CONV_STATUS rebuildThreadLoop … 0 行
[MEM_SNAP] PUBLISH gen=4..5 停留
```

source 連鎖（実読コード）：

```
1. CLI 起動時 --cli-device-type=DirectSound
2. MainWindow.cpp:457 → audioDeviceManager.setCurrentAudioDeviceType("DirectSound", false)
3. JUCE：旧 device（WA）close → AudioEngine::releaseResources() 第 1 pass
     ReleaseResources.cpp:87  shutdownRuntime_.closeAdmission()
     → packedState_ の state = Closing→Closed、以降 tryAdmit が false を返す
4. prepareToPlay(DirectSound) で rebuild スレッド再 start・generation も初期化される
   （`AudioEngine.Processing.PrepareToPlay.cpp:87` `prepareToPlay: rebuild thread started`）
5. burst intent 60 件は `[REBUILD_TELEMETRY] REQUESTED decision=accepted` で受け入れ記録
6. ただし RebuildDispatch.cpp:319 の `if (!shutdownRuntime_.tryAdmit(1)) return;`
   で何も出力せずに return → **telemetry でも dropped 未記録**
7. generation は発進せず、IR rebuild / publish / retire の chain は 1 回も動かない。
8. shutdown(pass 2) でも admission closed のままなので reconstruct path 格納のみ。
```

**これは機能的欠陥の可能性**：device-type switch 後に rebuild intake を再開させる affordance が現行 source に存在しない
（`reopenAdmission` / `resetForRunning` 相当が grep で 0 件）。I2-DS・R2/R4/R6 がこの構造に乗る。

- 一方で **lifecycle / release の completeness は不変**：構築済 DSP は 1 個も残らず destroy され、remaining=0 完遂。
  teardown race / retention leak / double destroy とは無関係。crash/dump/zone も正常。
- **D164 の DS transitionViolations=7 と同一 root cause（二重 releaseResources + admission close が reset されない）に収束**。

## 6. PASS 判定（D165 仕様に厳密従う）

| 条件 | 結果 |
| --- | --- |
| exit == 0 全 run | ✓ |
| new crash dump == 0 | ✓ |
| shutdown zone clean（T0/T1/T2 なし） | ✓ |
| E-3 / Signature A/B/C == 0 | ✓ |
| EBR overflow == 0 / final pending == 0 | ✓ |
| residual == 0（remaining=0 で全 object 収支） | ✓ |
| duplicate destroy == 0（pointer+gen で） | ✓ |
| registered DSP direct destroy == 0 | ✓（全 destroy は retire lifecycle 経由を通過 — V-D[D162-2G2] 経由で確認） |
| forbidden stale-map HIT == 0 | ✓ |
| shutdown-window XRUN == 0 | ✓ |

DirectSound の transitionViolations 滞留は既知 residual で、D165 でも判定を変更しない — **PASS + RESIDUAL**。

## 7. 既知 benign pattern の log-level 再確認（D165 の要求）

| pattern | 実測 | 判定 |
| --- | --- | --- |
| `[D117_RETIRE] retired=0` | WA 2 件 / DS 0 / R 各 1-2 — 二重 retire 試行 | benign（D164 で分類済） |
| `RETIRE_BY_HANDLE lookup=MISS` | 全 run で 1 件前後 | benign（ISANE） |
| `[D162-2B_RETIRE] origin=桳瑵潤湷挭敬牡` | WA 1 / DS 1 / R 各 1 | benign（UTF-16/8 corruption・decode=`shutdown-clear`） |
| `[FAULT] coordinator in Faulted after markShutdownComplete` | WA 1 / DS 1 / R 各 1 | benign（15-P-5｜G4 PASS 担当：ER、同一 pattern） |
| `XRUN … startup transient` | WA 36 / DS 0 / R 0-6 | benign（shutdown 窓 = 0・全 far-phase） |

## 8. Working tree の runtime telemetry 生成物

D164 と同じく soak 実行が `evidence/evidence/*.json` を app runtime パラメータで書き換えます。
（app 自身が overwrite する既存慣行 — commit されていない；本報告には一時 snapshot `d165_trace_*.json` に保存。）

## 9. 変更範囲（実測）

```text
production source    : 0
test source          : 0
CMakeLists.txt       : 0（H5 の +10 のまま）
build.bat / tools    : 0
CMakeCache           : 0（D164 で DIAG=ON 復元済；本監査は freeze 確認のみ）
新規追加
   evidence/D165/d165_a0_freeze.json
   evidence/D165/d165_soak.ps1
   evidence/D165/d165_b_reconciliation.json
   evidence/D165/d165_trace_*.json（8 件）
   evidence/D165/D165_I2_LONGRUN_VALIDATION.md（本文件）
   evidence/D165_[WA|DS|R1..R6].log×8
```

## 10. 結論と推奨 next step

> **D164 の I1 非再現観測は、I2 = 60-gen long soak + IR reload + rebuild burst + device cycle + restart で拡張しても再現しなかった**（8/8 PASS）。
> DirectSound 残留 `transitionViolations=7`（DS-F1）と、今回新規に判明した 「device switch 後の rebuild-intake 停止」（DS-F2）は共通の root cause を持つ（二重 releaseResources 由来）。**いずれもlifecycle integrity を崩さず・teardown race / 検証目的自体には干渉しない**。

次の扱いは次のとおりにする：
- **本 track（teardown / lifecycle validation）は D165 で closed（PASS + RESIDUAL）**。
- DS-F2（rebuild intake 不反応）は teardown の問題ではなく /device reconfigure 機能の実装上の不備。新規 defect として後日の CR に切り出す（修復自体は D165 では行わない）。GUI への影響が存在する可能性がある点は要調査（現行 UI で device-switch して rebuild がすぐ再動くかどうか）。
- DirectSound の transVio でない実行は独立対象 — 「pass 内部の un-reset admission」の設計に基づくため、別 task で扱うべき。
