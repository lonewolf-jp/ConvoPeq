# D169-1 — In-flight Rebuild × Terminal Shutdown Race Audit（Work Report）

```text
D169-1 — In-flight Rebuild × Terminal Shutdown Lifetime / Ownership Race Audit
Date: 2026-09-06
Type: read-only architectural / source audit（production/test/CMake/build.bat/tool 変更 0・binary rebuild 0）
Source: 現行 production source 直接読取（ConvoPeq.md snapshot は source authority 不使用）
Verdict: **Case A — Defect confirmed（source-level 成立証明）**
```

## 判定

> **Case A。** terminal `releaseResources()` は同一 DSPCore に対し **handle-based retire
> （epoch 保護・正しい）** と **pointer-value retire（legacy `activeRuntimeDSPSlot`・epoch 保護なし）**
> の **二重 destroy authority** を併存する。後者は **D162-2-E (E-2) が dtor から
> 「address reuse で生存 DSP を誤 lookup し二重破壊する」として廃止済み**の pattern で、
> releaseResources 側にのみ残存。D167 の reconfigure pass が「placeholder を handle 経路で
> retire/destroy 後も pointer slot に address が残る」前条件を実運用で成立可能にし、
> terminal の pointer-value retire が reuse 後 address の生存 DSP を destroy →
> handle 経路 destroy と二重化 → 0xC0000005。

## 主要発見（source-level）

1. **destroy authority 二系統**: `retireDSPHandleForRuntime(DSPCore*)`（`AudioEngine.h:4388`）は
   `runtimeDSPHandleMap_.find(dsp)` = **raw pointer key lookup**。これが pointer-value retire（system B）。
   正系は handle identity の `retireByHandle`（system A・epoch 保護）。
2. **terminal pass 内の併存**: `ReleaseResources.cpp:172` capture → `:352` pointer-value retire と、
   `:444` handle capture → `:534` handle retire が **同一 pass で両方走る**。
3. **dtor は E-2 で pointer-value retire 廃止済み**（`CtorDtor.cpp:202/208` retireByHandle のみ）。
   E-2 コメント（`CtorDtor.cpp:186-196`）が本 race を予見・releaseResources は未適用。
4. **pointer slot は placeholder 専用**（`AudioEngine.h:2265`・writer は prepareToPlay のみ）。
   **rebuild publish は pointer slot を更新しない**ため、placeholder が handle 経路で destroy されても
   `activeRuntimeDSPSlot` は free 済み address を保持し続ける。
5. **generation は logical のみ保護**（R4）: `isRebuildObsolete` は int 比較で stale request を弾くが、
   physical pointer/handle lifetime は未保護。pointer slot に generation tag なし。
6. **interleaving 実測対応**（probe8/9）: `[D117_DESTROY] 69F6E080`（rebuild EBR・:3427）→
   free（:3475）→ terminal `[D117_RETIRE] 69F6E080 retired=1`（:3530・**address reuse で map 再命中**）
   → `[D117_DESTROY] 69F6E080`（:3558・二重破壊）→ 0xC0000005。

## R1〜R5 判定

- **R1 double-destroy**: Yes（C パターン — reuse 後 object を pointer 経路と handle 経路が両方 destroy）。
- **R2 ownership 二重化**: terminal pass 内 :352（pointer）× :534（handle）。
- **R3 linearization**: window あり（join 後も EBR destroy / reuse / 再登録が pending queue で進行）。
- **R4 generation**: physical lifetime を保護していない（logical のみ）。
- **R5 epoch/retire**: system B は ownership→retire→epoch→reclaim→delete 一本路を逸脱。

## repair contract 方向（提示のみ・D170 で確定）

**destroy authority を handle-based 一本路に収束**（E-2 を releaseResources へ横展開）。
terminal の pointer-value retire（:172/:352 の active/fading/pending）を dtor と同一の
handle-based retire に統合し、pointer slot は観測専用化。INV-D162-1/3（registered DSP 処分は
DSPLifetimeManager handle 経由）と整合。**本 audit 中に実装契約は確定しない。**

## 変更範囲（実測）

production 0 / test 0 / CMake 0 / build.bat 0 / tool 0 / binary rebuild 0。

## 成果物

- 正本: [evidence/D169/D169_1_INFLIGHT_REBUILD_TERMINAL_RACE_AUDIT.md](C:\VSC_Project\ConvoPeq\evidence\D169\D169_1_INFLIGHT_REBUILD_TERMINAL_RACE_AUDIT.md)
- 本報告: doc/work88/D169_1_INFLIGHT_REBUILD_TERMINAL_RACE_AUDIT_REPORT.md

次: **D169-1R（repair contract approval）→ D170（race repair 実装）**。
D169-2（duplicate-prepare collapse abort）は **独立監査として分離**（physical lifetime と
prepare transaction state-machine は修復原理が異なる）。
