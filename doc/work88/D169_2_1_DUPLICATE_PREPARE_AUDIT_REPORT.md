# D169-2-1 — Duplicate-Prepare Collapse / Abort Race Audit（Work Report）

```text
Task:    D169-2-1 — Duplicate-Prepare Collapse / Abort Race Audit
Date:    2026-09-07
Type:    read-only source audit（production/test/CMake/build.bat/tool 変更 0・実装禁止）
Verdict: **Case A — Defect confirmed（availability クラス: collapse → std::abort 0xC0000409）**
```

## 判定

`LifecycleIsolationRuntime::enterPrepare` の **collapse 経路**（phase を Prepared のまま
token 返却・`ISRLifecycle.cpp:27-36`）と、`prepareToPlay` 本体（collapse 時の早期 return
が存在せず全 body を実行）と、`leavePrepare` の `phase==Preparing` 前提違反で
`std::abort()`（`ISRLifecycle.cpp:52-55`）の 3 者契約不整合。

**production 到達経路 4 系統**（同一 SR/BS での device restart）:
1. DeviceSettings recovery 経路（closeAudioDevice → 同一保存 setup reopen）
2. 同一 SR/BS を保った device type 切替
3. ASIO panel 経由の同設定 reopen
4. device hot-plug 再接続

JUCE 側確認: `AudioProcessorPlayer::audioDeviceAboutToStart` は device restart 毎に
`setProcessor` swap で prepareToPlay を 1 回呼ぶ（`juce_AudioProcessorPlayer.cpp:364-372`
→ `:180`）。`AudioDeviceManager::setAudioDeviceSetup` は同一 setup なら early return
（`juce_AudioDeviceManager.cpp:815-818`）なので、trigger は device 再 open が必要。

## 重要な負判定（memory-safety は健全）

- **並行 duplicate prepare は構造的に阻止**（Message Thread 単一入口 + enterPrepare
  overlap abort — LIF-1 設計どおり）。
- **ownership 二重化 / leak / double destroy / stale publish は不成立**: rollback 対象は
  未登録 placeholder のみ（CallerDestroy 契約）、re-prepare は既存 published DSP を
  継続使用（新 placeholder 不作成）、in-flight rebuild は gen gate → S4 authority dispose。
- **transaction identity は存在しない**（LifecycleToken は照合されない）が、保護は
  thread 直列化 + lifecycleState CAS + FSM abort の 3 層で成立している。
- generation は rebuild publish ordering の保護のみで、prepare transaction validity は
  保護しない（R4 = No）。

## R1-R12 判定（要旨）

R1 **Yes**（sequential 形のみ）/ R2 No / R3 No / R4 No / R5 No / R6 No / R7 No /
R8 No / R9 Conditional（memory race 無し・fail-stop abort あり）/ R10 memory-safety No・
**availability Yes**（collapse → abort）/ R11 **No**（テストは SR 交互で collapse を
意図的回避・coverage 0）/ R12 direction のみ（collapse no-op 化 a 案 or collapse 廃止
b 案・block 経路 phase 残留の扱いは D169-2-2 で判定）。

## 修复方向（D169-2-2 契約候補）

- **候補 a**: collapse token 時に prepareToPlay を早期 return（冪等 no-op 化・副作用最小）
- **候補 b**: collapse 廃止・常に完全 re-prepare（leavePrepare 前提と自然整合）
- 禁止: mutex/atomic/generation/transaction ID/prepare lock/cancellation/shutdown FSM 追加

## 成果物

- 正本: [evidence/D169/D169_2_1_DUPLICATE_PREPARE_AUDIT.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_2_1_DUPLICATE_PREPARE_AUDIT.md)
  （caller map / transaction lifetime graph / ownership graph / interleaving 4 形式 /
  generation analysis / test coverage / R1-R12 / repair direction）
- 本報告: doc/work88/D169_2_1_DUPLICATE_PREPARE_AUDIT_REPORT.md

## 変更範囲

production 0 / test 0 / CMake 0 / tool 0 / binary 0。

次: **D169-2-2（repair contract approval）→ D169-2-3 preflight → D169-2-4 minimal
implementation → D169-2-5〜7 targeted race validation / stress / full regression**。
