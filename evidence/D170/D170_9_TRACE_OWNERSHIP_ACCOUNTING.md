# D170-9 — Trace Ownership Accounting（DSP identity 単位 destroy 会計）

```text
D170-9 — per-DSP destroy accounting across all D170 validation logs
Date:     2026-09-07
Binary:   build-diag RelWithDebInfo（D170-2 実装入り・CONVOPEQ_ENABLE_RUNTIME_DIAGNOSTICS=ON）
Method:   [DSP_FOOTPRINT] construct 〜 [DSP_FOOTPRINT_RELEASED] を 1 object lifecycle 窓とし、
          窓内の [D117_RETIRE] retired=1 / [D117_RETIRE_BY_HANDLE] lookup=HIT（retire 1 回）と
          [D117_DESTROY]（destroy 1 回）を突合。address 再利用は隣接 lifecycle として正規化。
Verdict:  **PASS — 全 10 logs で lifecycle 1:1:1・二重 destroy 0・leak 0**
```

## 会計結果

| Log | lifecycles | retire/destroy 不整合 | open at exit | remaining≠0 |
| --- | --- | --- | --- | --- |
| D170_REPRO.log | 8 | NONE | 0 | 0 |
| D170_CHURN.log | 30 | NONE | 0 | 0 |
| D170_R1.log | 3 | NONE | 0 | 0 |
| D170_R2.log | 3 | NONE | 0 | 0 |
| D170_R3.log | 3 | NONE | 0 | 0 |
| D170_R4.log | 3 | NONE | 0 | 0 |
| D170_R5.log | 3 | NONE | 0 | 0 |
| D170_R6.log | 3 | NONE | 0 | 0 |
| D170_WA.log | 60 | NONE | 0 | 0 |
| D170_DS.log | 60 | NONE | 0 | 0 |

- **同一 lifecycle 窓内の DESTROY 2 回 = 0 件**（D169-1 defect signature の不在）。
- **exit 時 open lifecycle = 0**（constructed-not-destroyed 0 = D117_DESTROY per DSP = 1 成立）。
- 全 `[DSP_FOOTPRINT_RELEASED] remaining=0`。

## 2 つの見かけ上の exception（正当性確認済み）

1. **各 log 1 件の「DESTROY outside window」**: CLI log capture 開始前に構築済みの
   bootstrap placeholder DSP（`--cli-log-file` は engine 起動後に開始するため construct
   footprint が log 外）。retire → DESTROY → RELEASED は完全記録されており、
   lifecycle としても 1:1:1（REPRO: L207-210 retire retired=1 → DESTROY → RELEASED remaining=0）。
2. **`[D117_RETIRE] retired=0`（benign no-op）**: terminal V-D-b が既に disposition 済み DSP を
   resolve した場合の map MISS。runtimeDSPHandleMap_ は erase-once のため構造的に
   二重 destroy 不可（D162-2-G0 N-1 確立の invariant）。baseline D168_DS でも同種
   no-op 6 件が存在する pre-existing 挙動であり、D170 では retire(0) 後に
   DESTROY が続くケースは **0 件**（全 retired=0 の直後に destroy なし）。

## D169-1 defect signature との対比

| 項目 | D169-1（修復前 probe8/9） | D170（修復後 10 logs） |
| --- | --- | --- |
| stale address の map 再命中（`retired=1`） | あり（:352 pointer retire・address reuse） | **0 件**（pointer-value retire 廃止） |
| 同一 object の二重 destroy | あり（`[D117_DESTROY]` 69F6E080 ×2） | **0 件** |
| 0xC0000005 / dump | あり | **0 件**（REPRO ×5 含む全 run） |
| reconfigure → rebuild → immediate terminal | crash 条件 | REPRO ×5 全 exit 0x0・TV=0 |

## 補足実測

- REPRO（D169-1 直接再現条件）: reconfigure pass 1 回 → REBUILD_DISPATCHED 16/17
  （差 1 は latest-wins merge 窓）→ terminal pass 1 回 → 全 DSP destroy → remaining=0。
- terminal pass の retire は handle 経路のみで実行（`getActive/FadingRuntimeDSPHandle` →
  `dspHandleRuntime_.retire` → V-D-b authority retire）を trace で確認
  （例 REPRO :8325-8425: active-final retired=1 → DESTROY、fading-final retired=0 no-op、
  RETIRE_BY_HANDLE MISS ×2 = dtor E-2 冪等）。
