# D169-1R — Repair Contract Approval（Work Report）

```text
Task:    D169-1R — RC-D169-1-1〜5 repair contract fixation
Date:    2026-09-07
Type:    read-only / contract freeze（production source 変更 0）
Verdict: **APPROVED — 契約固定**
```

## 判定

D170 に先立ち、D169-1（Case A）の修復契約 5 項目を正文として固定した。

- **RC-D169-1-1 Single Destroy Authority**: registered DSPCore の terminal retirement は
  DSPHandle identity による handle-based retirement のみ。terminal `releaseResources()` の
  pointer-value retire 4 系（active/fading/pendingNew/pendingCurrent → `retire(void*)`）を廃止対象とする。
- **RC-D169-1-2 Pointer Slot は Ownership Source ではない**: `activeRuntimeDSPSlot` /
  `fadingRuntimeDSPSlot` は legacy placeholder observation に限定。D162-2-E (E-2) の
  dtor 側修復を releaseResources に横展開。
- **RC-D169-1-3 Handle path 既存 protocol 不変**: resolve → registry retire（冪等）→
  quiescent reclaim → authority retire → map identity removal → requestReclaim →
  EBR enqueue → destroyDSPCoreNode の既存一本路を変更しない。
- **RC-D169-1-4 terminal/reconfigure semantics は D167 のまま**。
- **RC-D169-1-5 禁止修復 9 項目**（generation tag / fake handle / reuse 検出 workaround /
  sleep / rebuild 強制停止 / map・EBR・shutdown FSM 変更等）を明記。

## 契約の前提条件として preflight 確定した source 事実

1. pointer slot の writer は prepareToPlay のみ・placeholder は registration+activate 済みで
   常に handle 系で disposition 可能（PrepareToPlay.cpp:270/284・DSPTransition.h:71/104）。
2. `pendingNewToRelease` は宣言のみで非 null になり得ない（代入 0 件）。
3. `pendingTask.currentDSP` は非 null writer が src 全体に存在しない
   （RebuildDispatch.cpp:602 の nullptr 初期化のみ・死変数）。
4. handle registry retire は冪等・map MISS は正常 no-op — pointer 廃止による未処理 DSP なし。

これらは D169-1 指示の「単純に :352 の retire を削除するだけで十分か」への回答であり、
**全 pointer 系 capture の廃止が安全である**ことを source-level で実証した。

## 成果物

- 正本: [evidence/D169/D169_1R_REPAIR_CONTRACT.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_1R_REPAIR_CONTRACT.md)
- 本報告: doc/work88/D169_1R_REPAIR_CONTRACT_REPORT.md

## 変更範囲

production 0 / test 0 / CMake 0 / tool 0（契約文書のみ）。

次: **D170-1 preflight source audit → D170-2 minimal implementation → D170-3〜9 targeted race validation + full regression**。
D169-2（duplicate-prepare collapse abort）は D170 完了後に独立 track。
