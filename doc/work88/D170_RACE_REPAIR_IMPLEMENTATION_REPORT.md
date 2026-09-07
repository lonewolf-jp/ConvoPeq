# D170 — Race Repair Implementation（Work Report）

```text
Task:      D170 — D169-1 Case A（in-flight rebuild × terminal shutdown race）の minimal repair
Date:      2026-09-07
Contract:  evidence/D169/D169_1R_REPAIR_CONTRACT.md（RC-D169-1-1〜5・APPROVED）
Preflight: evidence/D169/D170_1_PREFLIGHT_SOURCE_AUDIT.md（GO）
Verdict:   **実装完了 — D170-1〜9 全項目 PASS・D169-1 defect signature 消滅**
```

---

## 1. 実装内容（D170-2・minimal implementation）

**変更ファイル: `src/audioengine/AudioEngine.Processing.ReleaseResources.cpp` のみ。**

```text
BEFORE                                          AFTER
terminal releaseResources                       terminal releaseResources
    ├─ :172 activeToRelease = slot 値               └─ handle → resolveDSPHandle
    ├─ :178 fadingToRelease = slot CAS                 → dspHandleRuntime_.retire（冪等）
    ├─ :352-359 lifetimeForShutdown.retire(×4)  ← 削除  → tryShutdownQuiescentReclaim
    │    （pointer-value retire・defect 本体）           → V-D-b authority retire
    └─ handle path（:466-505）・V-D-b（:556-576）       （SINGLE AUTHORITY・RC-D169-1-1）
         ← 無変更（RC-D169-1-3）
```

具体的変更（+83/-14 行・コメント主体）:

1. **:352-359 の 4 連 pointer-value retire を削除**し、D169-1R 契約注記 + `juce::ignoreUnused`
   に置換。`lifetimeForShutdown` ローカル（pointer retire 専用）も削除。
2. **`pendingNewToRelease` / `pendingCurrentToRelease` を削除**。D170-1 preflight で
   非 null writer が src 全体に存在しない（死変数）ことを実証済み。
   `pendingTask` consume は hasPendingTask/publishRetryReady の clear 副作用（slot 衛生）のみ維持。
3. **`activeToRelease` / `fadingToRelease` は観測専用として維持**（dtor E-2 前例と同一の
   `juce::ignoreUnused` 抑制）。slot clear（setActiveRuntimeDSP(nullptr) / fading CAS）は
   topology 衛生として無変更。
4. handle path（:466-505）・V-D-b（:556-576）・shutdown FSM・EBR protocol は **無変更**
   （RC-D169-1-3/4 準拠）。RC-D169-1-5 の禁止修復は一切適用していない。

## 2. D170-3/4 — CTest

| Config | 結果 |
| --- | --- |
| Debug（build-diag） | **40/40 PASS**（87s） |
| Release（build-diag） | **40/40 PASS**（40.7s・再実行） |

初回 Release は test #36 HeadlessAudioPathVerification が失敗したが、失敗内容は
「既存 ConvoPeq プロセスの Stop-Process 拒否（アクセス拒否）」— Debug/Release CTest を
並行実行したことによる runner 衝突（ConvoPeq.exe 重複起動）で、ソース起因ではない。
単独再実行で 40/40 PASS（evidence/D170/d170_ctest_release2.log）。過去にも同型の
環境起因 failure が 1 件ある（D162-2B_ctest_debug.log）。

## 3. D170-5 — 直接再現条件（reconfigure → rebuild in-flight → immediate terminal）

RWDI binary で `DirectSound` switch（reconfigure pass）+ IR reload 3 回 + intent burst 8 回 +
16s 即時 exit を **5 回**実行:

```text
REPRO ×5: exit 0x0 ×5 / dump 0 ×5 / transitionViolations 0 ×5
trace:    reconfigure pass 1 回 → REBUILD_DISPATCHED 16/17（差 1 = merge 窓）→
          terminal pass 1 回 → 全 DSP destroy → remaining=0
```

D169-1 で crash した interleaving（rebuild EBR destroy → address reuse → terminal pointer
retire map 再命中 → 二重 destroy）が **1 回も発火せず**、pointer retire が source から
除去されたため構造的に発生不能。

## 4. D170-6〜8 — stress / restart / regression

| Stage | 結果 |
| --- | --- |
| CHURN（address reuse stress・IR reload 30 回 ×1.2s + burst 30） | exit 0x0・dump 0・TV=0・gen 30 |
| R1-R6 restart cycle（WA/DS 交互 ×6） | 全 exit 0x0・dump 0・TV=0 |
| WA long-run（reload 60 + burst 60・424s） | exit 0x0・TV=0・REQ 120/DIS 120・gen 63・admission_closed 0 |
| DS long-run（switch + reload 60 + burst 60・426s） | exit 0x0・TV=0・**REQ 121/DIS 120・gen 60**・admission_closed 0 |

DS long-run は D168 baseline（121/120・gen 60・TV=0）と **完全一致**。regression なし。

## 5. D170-9 — trace ownership accounting

**evidence/D170/D170_9_TRACE_OWNERSHIP_ACCOUNTING.md** 参照。要旨:

- 10 logs 全部で lifecycle（construct → RELEASED）窓内の retire:destroy = **1:1**・
  **二重 destroy 0**・exit 時 open lifecycle 0（leak 0）・`remaining=0` 全行。
- D169-1 の defect signature（stale address の map 再命中 retired=1 → 二重 DESTROY）が
  **全 logs で 0 件**。
- `retired=0` の benign no-op（既に disposition 済み DSP への V-D resolve → map MISS）は
  baseline D168 にも存在する pre-existing 挙動で、D170 では retired=0 の直後に
  DESTROY が発生するケース 0 件（map erase-once invariant が機能）。

## 6. 受入基準照合

```text
[D170-1] preflight source audit（4 問に回答・GO）                      ✅ evidence/D169/D170_1_...
[D170-2] minimal implementation（RC-D169-1-1〜5 準拠・単一ファイル）     ✅ §1
[D170-3] Debug CTest 全 PASS                                            ✅ 40/40
[D170-4] Release CTest 全 PASS                                          ✅ 40/40
[D170-5] reconfigure → rebuild → immediate terminal（×5）               ✅ exit0x0/dump0/TV=0 ×5
[D170-6] address reuse stress                                           ✅ exit0x0/TV=0
[D170-7] repeated restart ×6                                            ✅ 全 0x0/TV=0
[D170-8] DS/WA regression（D168 基準）                                  ✅ DS 121/120・gen 60 完全一致
[D170-9] trace ownership accounting（destroy 2 回なし）                 ✅ 10 logs 全部 1:1
[D117_DESTROY per DSP = 1]                                              ✅ lifecycle 会計 1:1
[DSP_FOOTPRINT_RELEASED remaining = 0]                                  ✅ 全行 remaining=0
[transitionViolations = 0]                                              ✅ 全 run
[exit = 0 / dump = 0]                                                   ✅ 全 run
```

## 7. 成果物

- 実装: src/audioengine/AudioEngine.Processing.ReleaseResources.cpp（+83/-14）
- 契約: evidence/D169/D169_1R_REPAIR_CONTRACT.md + doc/work88/D169_1R_REPAIR_CONTRACT_REPORT.md
- preflight: evidence/D169/D170_1_PREFLIGHT_SOURCE_AUDIT.md
- 会計: evidence/D170/D170_9_TRACE_OWNERSHIP_ACCOUNTING.md
- runner: evidence/D170/d170_build.bat + d170_soak.ps1
- logs: evidence/D170_{REPRO,CHURN,R1..R6,WA,DS}.log + evidence/D170/d170_*.log/json ×10 run
- CTest: evidence/D170/d170_ctest_{debug,release,release2}.log

## 8. 残課題

1. **D169-2（duplicate-prepare collapse abort）**: 予定どおり D170 完了後の独立 track。
   prepare transaction / lifecycle FSM の修復原理（physical lifetime とは別系統）。
2. `~AudioEngine` dtor の `[FAULT] coordinator in Faulted state` ログ 1 行:
   D165/D168 から存在する既知残差（実害なし・exit 0x0）— 本 track では非扱い。
3. 初回 Release CTest の runner 衝突は手順上の教訓（CTest は単独実行）— source 起因ではない。
