# D162-2-B Evidence — Build / CTest / Soak / EBR

Date: 2026-09-03
Binary: build-diag（RWDI / Debug / Release、全て CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON）
Soak 条件: D162-1P と同一（IR=D162-1P_active.wav 76800@192kHz / block 1024 / OS 2x /
--cli-ir-reload-count 60 --cli-ir-reload-interval-ms 6000 --cli-intent-burst-count 60
--cli-intent-burst-interval-ms 6000 --cli-exit-ms 420000）
Log: evidence/D162-2B_soak.log（199,864 行・EXITCODE=0x00000000・59 enqueued generations gen=4..62）

## 1. Build

| 構成 | script | 結果 |
| --- | --- | --- |
| RelWithDebInfo + DIAG | evidence/D162-1RB_diag_build_rwdi.bat | EXIT=0 |
| Debug + DIAG | evidence/D162-1RB_diag_build_debug.bat | EXIT=0 |
| Release + DIAG | evidence/D162-2B_diag_build_release.bat | EXIT=0 |
| Release + DIAG + ASAN | evidence/D162-1RB_diag_asan_build.bat（実装途中版で PASS・最終版は未再実行） | EXIT=0 |

## 2. CTest

| suite | 結果 |
| --- | --- |
| Debug 全件 | **40/40 PASS**（32.3s） |
| Release 全件（AudioEngineHarness 除く） | **39/39 PASS**（14.6s） |
| Release AudioEngineHarness | 0xC0000374 — **HEAD baseline（362dccd・stash で本修正除外）でも 3/3 再現する pre-existing**（D162-1P 既知の Release DIAG クラッシュと同一 0xC0000374）。本修正に起因しない |
| Debug HeadlessAudioPathVerification | 単独再実行 3/3 PASS（全体実行時 1 回失敗は flake） |

## 3. Soak 実測（60 gen）

### 3.1 カウント対応表

| 項目 | D162-1P（修正前） | D162-2-B（修正後） |
| --- | ---: | ---: |
| [D133] enqueue | 60 | 59（gen 4..62） |
| publish SUCCEEDED | 11 | 11 |
| [D117_DESTROY] | 11 | **54** |
| [D162-2B_RETIRE]（S1/S2 authority 経由） | — | 43（overwrite 40 + discard 3） |
| [DSP_FOOTPRINT] construct/retained | 11/11 | 59/59 |
| [DSP_DESTROY_FOOTPRINT] / [DSP_FOOTPRINT_RELEASED] | 2/2 | 54/54 |
| retained（= enqueued − destroyed、final active 除く） | **49（82%）** | **4（6.8%）** |

### 3.2 メモリ

| 指標 | D162-1P | D162-2-B |
| --- | ---: | ---: |
| DC live（MEM_SNAP DC: live） | 1 → 50 線形増加 | min 1 / **max 7** / last 6 |
| Private | 381 → 7,740 MB（+1,215 MB/min） | 381 → **1,196 MB**（終盤安定） |
| retained DSP × footprint | 49 × 142.48 MB ≒ 6,982 MB | 6 × 142.48 MB ≒ 855 MB（うち 2 件は shutdown 時 residual） |

### 3.3 EBR counters（最終 MEM_SNAP）

- pend=0 / ovf=0 / tr=0/0 / rec=46,298,739（単調増加）— EBR 正常 drain・overflow 無し。

### 3.4 破壊 reconciliation

- construct pointers 59 = destroy 対象の pointer set と一致（54 destroyed + 5 by-design/audit residual）。
- destroyed gens 54、重複 destroy なし（同一アドレスの再使用は construct→destroy の厳密交互で確認）。
- retained 残存 6: gen 60（final active・by design residual）、gen 62（shutdown-clear deferred・residual）、
  gen 9/35/41/53（S1/S2 の未網羅窓口 — D162-2-C で遷移追跡）。

## 4. 分離試験（S3/V-D の exit AV 確定に使用）

| 試験 | S3 | V-D | destroy 数 | exit |
| --- | --- | --- | ---: | --- |
| 初版（S1/S2/S3/S4 + VD、EBR retire） | EBR | EBR | 61 | **0xC0000005** |
| VD 無効化 | EBR | off | 61 | **0xC0000005** |
| + S3 無効化 | off | off | 60 | 0x00000000 |
| + S3 direct destroy | direct | off | 7（active 残存） | 0x00000000 |
| + S3 direct + VD direct（world clear 後） | direct | direct | 7/7 | **0xC0000005** |
| 最終（S3 off + VD off、S1/S2/S4 のみ） | off | off | 54 | **0x00000000** ×2 |

→ shutdown 時の DSP 破壊（S3/V-D のいずれの実装形態でも）が ~AudioEngine 後半 member teardown の
0xC0000005（クラッシュダンプ: `mkl_serv_check_fast_memory_size` READ @ 0xFFFFFFFFFFFFFFFF）を誘発。
破壊しない（D162-1P 同一挙動）場合は clean exit。この因果は 6-gen 反復で再現性確認済み。

## 5. 保存物

- evidence/D162-2B_soak.log（最終 60 gen・exit 0）/ D162-2B_soak6.log（6-gen 反復）
- evidence/D162-2B_ctest_debug.log / D162-2B_ctest_release.log
- evidence/D162-2B_build_rwdi.log / D162-2B_build_debug.log / D162-2B_build_release.log
- Crash dumps: C:\Users\user\AppData\Local\CrashDumps\ConvoPeq.exe.448.dmp ほか 3 件（D162-2-C 解析用）
