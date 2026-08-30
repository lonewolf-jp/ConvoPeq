# D128 — Publish Tail / Active Handle Lifetime Causality Audit

**Date:** 2026-08-29
**性質:** read-only 因果監査 + 既存 macro-gated trace の解析。**production semantics 変更 0**
**対象:** D127-NOGO probe（`evidence/D127_nogo_probe.log` — M1+M3+M5 適用状態で収集・修正済み TASK_WAKE トレース含む）

---

## D128-A — Publish → Tail の 1:1 対応（G1）— **完全解明**

修正済み `[D126_TASK_WAKE]`（wake reason を正しく記録する版）による実測タイムライン:

| task execution | wake 要因 | build | publish | tail (onPublishCompleted) | 結果 |
| --- | --- | --- | --- | --- | --- |
| taskGen=4 | pendingTask | ✅ | ✅ seq=6 (gen=6 world) | ✅ oldDSP 配送（needsCrossfade=1） | crossfade claim → 完了待ち |
| taskGen=5 | pendingTask | ✅ | **✗ — admission が永久 deferred** | ✗ | **DSPCore 滞留** |
| （以降 15,796 回） | **RetryReady のみ** | ✗（M3 ガード） | — | — | wake storm のみ |

- **publish 成功 → tail は 1:1 で実行される**（tail 間引きは存在しなかった）
- **tail が無い publish は、publish 自体が成立していない**（admission が deferred のまま）
- 従って D127 の「6 intent → tail 2 回」の正しい読みは:
  - 6 intent のうち实际に build+publish に到達したのは 2 つ（taskGen 4, 5）
  - 残りは pendingTask レベルで Replaceable 統合（pre-build 破棄 — **benign**、DSPCore 未生成につき漏出なし）
  - taskGen=5 は build まで実行したが **admission deferred で publish 不成立** → 滞留

## D128-B — Replaceable / Phase5-KEEP の意味論（G2）

| 機構 | 統合するもの | 漏出への影響 |
| --- | --- | --- |
| Replaceable（pendingTask 置換） | **build 前のタスク**を置換 — DSPCore 未生成のため **benign** | なし（pre-build 破棄） |
| Phase5-KEEP | duplicate pending task を維持 | 本 probe では 0 発火 |
| **admission deferral**（fade 完了待ち） | **build 済み DSPCore を publish せず滞留** | **漏出の本体（経路 2）** |

**結論**: build の統合（pre-build 破棄）は benign。漏出は **post-build の admission deferral** による。A build/B build/C publish の coalescing は「最後の 1 つだけ publish され、残りは publish 待ちで滞留」——obsolete-destroy 経路が存在しないため漏出になる（INV-REBUILD-EXEC-1 違反の実証）。

## D128-C — `activeRuntimeDSPHandle_` 「消失」の真実（G5）

**D127 の「seq=7 で oldHandle が消失した」という解釈は訂正される。**

| 観測 | 事実 |
| --- | --- |
| seq=6 | rebuild commit。tail で `activate(newHandle)` 実行 → `activeRuntimeDSPHandle_` = handle6 **公開済み**。needsCrossfade=1 → claim + crossfade start |
| seq=7 | **idle publish**（同一 DSP の再 publish。`publishIdleWorldOnly` 系 — oldHandle null 固定は**設計どおり**）→ tail で oldDSP=null → no-op（正当） |

- `activeRuntimeDSPHandle_` の全 writer: ① `activate()`（=handle）② `endCrossfade()`（=toHandle）— **null を書く writer は存在しない**
- seq=7 の oldHandleNull=1 は「handle が消えた」のではなく、**idle publish が設計上 oldHandle=null を渡した**だけ
- 三択の回答: **「消された」でも「読めない」でもなく「oldHandle null は別経路（idle publish）の設計値」**
- 残る微細な確認点: seq=6 の activate 後、seq=7（idle）の前の rebuild（taskGen=5）が getActiveRuntimeDSPHandle を読んだ際の値 — timeline 上 activate 後であり非 null のはず。gen-5 publish の deferral は active handle ではなく **admission（crossfade pending）** が原因

## D128-D — handle identity の二重管理（G6）

| 対 | 確認 |
| --- | --- |
| newDSP ↔ newHandle | `registerDSPHandleForRuntime`（冪等）で 1:1 確立（D127-A 実装済み） |
| oldDSP ↔ oldHandle | seq=6 の tail で `resolve(decision.oldHandle).instance == oldDSP` 成立（oldResolvedValid=1 + old=...4948A080） |
| **activeRuntimeDSPHandle_** | 「直近 commit 済み world の activeDSP handle」（D121-2 の invariant 定義どおり動作 — M1 配線時に確立） |
| **fadingRuntimeDSPHandle_** | crossfade 中の旧 handle。**completion 駆動が存在しないため、一度 claim された fading は永久に解消されない**（D121-C の再確定） |

BUG-054 対策（getActiveRuntimeDSPHandle を crossfade old 判定に使わない）は現行コードでも維持されていることを確認。

## D128-E — 統合因果モデル（最終形）

```text
M1 (activate) 配線
  ↓ needsCrossfade=1（IR 遷移時）
crossfade branch: claim fading slot + crossfadeRuntime_.start
  ↓ 【M2 (completion driver) 不在】
fade 完了せず → fading slot 占有継続
  ↓ 次の publish の admission が永久 deferred（RetryReady wake ≈450Hz）
build 済み DSPCore 未 publish のまま滞留（+ wake storm）
  ↓ M3 がなければ stale task 再 build で増幅（D123 storm）
  ↓ M3 あり（D127）: wake storm のみ・DSPCore 滞留は継続

frozen baseline（M1 なし）:
  crossfade 分岐到達不能 → publish された DSPCore が誰にも retire されない（経路 1 漏出）
```

**D119/D123/D127 の 3 回の失敗が全て説明される**:
- D119: M1 単体 → crossfade branch 到達 → completion 不在 → 暴走
- D123: M1+completion 試行 → storm 判定は誤りだったが M3 で増幅遮断、admission deferral 残存
- D127: M1+M3（M2 意図的除外）→ admission deferral 滞留が Gate 4 で検出

**結論（D129 への唯一の修復契約）**: **M1 + M2 + M3 の 3 点セットが必須であり、単一パッチで実装する。** これに加えて:
- admission deferral の liveness 契約（fade 完了を待つ deferred publish に timeout/強制進行が必要 — 15,796 回の無限 wake は別問題としても監査価値あり）
- M5（Observe 撤去）/ M6（shutdown 診断）は D123 実績どおり併せて適用

## GO/NO-GO（G1-G8）

| Gate | 判定 |
| --- | --- |
| G1 publish→tail 1:1 | **PASS** — 1:1 成立。tail 無し = publish 不成立（admission deferral）と完全対合 |
| G2 Replaceable semantics | **PASS** — pre-build 統合は benign / post-build deferral が漏出本体 |
| G3 未 publish DSP ownership | **PASS** — INV-REBUILD-EXEC-1 違反を line-level で確定（admission deferral → 滞留） |
| G4 active handle writer 完全列挙 | **PASS** — writer 2（activate/endCrossfade）・null writer なし |
| G5 seq=6→7 消失原因 | **PASS** — 「消失」は誤認。seq=7 は idle publish（null 固定は設計どおり） |
| G6 identity 対応 | **PASS**（seq=6 で oldDSP↔oldHandle↔resolve 一致を実測） |
| G7 M3 regression なし | **PASS**（M3 は D127 で実証済み・本監査で再変更なし） |
| G8 production semantics change = 0 | **PASS**（D127 実装は取消済み・現状は診断 trace のみ） |

**D128: PASS（8/8）** → **D129（M1+M2+M3 統合実装）へ進行可。**

## D129 実装スコープ（確定）

1. **M1**: DSPTransition.h — activate 公開（D127-A 実装を再適用・実績あり）
2. **M2**: crossfade completion 駆動（D121-5 方式 A — RT ramp エッジ検出 → notifyRampComplete → Timer consume → endCrossfade → retire。D123-B 実装を再適用）
3. **M3**: stale task ガード（D127-B 実装を再適用 — 15,371 wake 実証済み）
4. **M5/M6**: Observe 撤去 + shutdown 診断（D123-C/D 実装を再適用）
5. **新規**: admission deferral の liveness 監視（fade 完了待ち publish の timeout 診断 — macro-gated）

## 生成物

- 本ファイル（`evidence/D128_PUBLISH_TAIL_CAUSALITY_AUDIT.md`）
- `evidence/D127_nogo_probe.log`（修正済み TASK_WAKE による因果実測）
- production semantics 変更: **0**（D127 実装は取消済み）
