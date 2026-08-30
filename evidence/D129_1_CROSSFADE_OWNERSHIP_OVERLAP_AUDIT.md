# D129-1 — Crossfade Completion Ownership / Overlap Re-audit

**Date:** 2026-08-29
**性質:** read-only line-level 証明。**production source 変更 0**
**判定:** **PASS 10/10 — D129-2（M1+M2+M3+M5+M6 統合実装契約レビュー）へ進行可**
**核心的結論:** completion event に identity を載せる必要は**ない**。fading identity は既存の 2 つの durable 機構（`fadingRuntimeDSPSlot` CAS ポインタ + `pendingReceipt_` handle）が保持しており、event は**純シグナル**として機能させる。overlap 時の漏出は「claim 失敗時に slot 占有者を retire しない」現行分岐の不完全さであり、**slot 占有者と新 oldDSP の双方 retire**（pointer-based・冪等）で閉じる。

---

## 1. Completion event identity 候補の判定（G1）

前提（実コード確認）:
- `CrossfadePreparedSnapshot`（AudioEngine.h:2282-2294）は **CrossfadeId を保持しない** → RT に id を届ける既存経路は存在しない
- `getActiveCrossfades()` は mutex 使用 → **RT から呼べない**
- RT は mix path で `fading`（DSPCore\*、AudioBlock.cpp:348 — RCU resolve 済み）を既に保持
- `CrossfadeRecord = {id, fromHandle, toHandle, startEpoch, active}`（ISRDSPHandle.h:91-98）

| 候補 | event → unique fading object → unique retire target | 判定 |
| --- | --- | --- |
| A: CrossfadeId 携帯 | id が RT に届かない（新機構が必要） | **不採用** |
| B: fading DSPHandle 携帯 | RT は DSPCore\* のみ保持（handle は map private） | **不採用** |
| C: DSPCore\* 携帯 | ポインタは一意だが ABA（解放後再利用）の理論リスク + lookup が DIAG 専用 | **不採用** |
| **D（採用）: event は純シグナル、identity は slot CAS + receipt** | Timer が fading slot CAS で **DSPCore\*** を取得（一次 identity）→ `retirePublishedDSP` が receipt.handle を resolve 検証（D122-B 契約済み） | **採用** |

**一意性の証明**: fading slot は「nullptr → oldDSP」の CAS で占有され、completion consume 時の CAS で **null に戻しつつ旧値を取得**する → 取得したポインタがその瞬間の唯一の fading DSPCore。receipt（crossfade 開始時に保存された fadingHandle）は `resolve(receipt.handle).instance == current` で交叉検証可能（generation-safe）。event が identiy を運搬しないため ABA/取り違えの表面上問題は構造的に発生しない。

## 2. A→B→C overlap の同一 timeline（G2/G5）

前提: M2（completion 駆動）実装後の契約で相手を追跡。

| 時点 | active world | fadingRuntimeDSPSlot | activeRuntimeDSPHandle_ | fadingRuntimeDSPHandle_ | records_ | runtimeDSPHandleMap_ |
| --- | --- | --- | --- | --- | --- | --- |
| A active | A | null | A_h（M1 公開） | null | — | {A→A_h} |
| B publish tail | B | **A**（claim 成功） | B_h（M1） | A_h（beginCrossfade） | A→B (active) | {A,B} |
| A→B ramp 完了（RT）→ Timer consume | B | CAS clear → **A を取得** | B_h | null（endCrossfade） | A→B (inactive) | {A,B} |
| → retirePublishedDSP(A) → **retire(A)** → destroy(A) enqueue | B | null | B_h | null | — | {B} |
| **C publish（B→A 完了前に発生した場合）** | C | claim(B) → **CAS 失敗**（slot=A 占有） | C_h（M1） | B_h（C の beginCrossfade） | B→C (active) 追加 | {A,B,C} |
| → claim 失敗分岐 | C | **A と B を双方 retire**（D129-1 契約） | C_h | B_h | A→B は中断済み → unregister | {C}（A, B は destroy へ） |

**claim 失敗時に A と B を両方 retire することの ownership-safe 証明**:
1. 新 world（C）の topology は C のみを参照 — **A も B も新 world から到達不能**（graph.activeNode = C）
2. A は「中断された crossfade の fading 側」— A→B record は B が置換された時点で無意味化（A→C への継続は存在しない）
3. 両者の retire は `DSPLifetimeManager::retire(DSPCore*)`（pointer-based・map erase 済みなら冪等 no-op）— **二重 retire 構造的に不可能**
4. EBR: destroy は enqueue → drain（epoch 通過後）— RT 読み保護は現行契約どおり
5. **slot は必ず clear**（retire 前に CAS で取得済み）→ 次の claim が正常化 → INV-XFADE-4（feedback loop 防止）維持

**結論: 両方 retire は ownership-safe（PASS）**。ただし契約条件として (a) retire は pointer-based map lookup 冪等、(b) A の crossfade record は unregister する、(c) EBR drain による物理破壊順序を維持、を明記。

## 3. completion → retire chain の line-level 追跡（G4）

```text
RT: ramp remaining 1→0（getNextValue 収束）     DspNumericPolicy.h:389-396
  ↓ notifyRampComplete(): SPSC push（signal）   CrossfadeRuntime.h（D123-B 実装案）
Timer: consumeCompletedFade(ev)                 Timer.cpp（新 consume ブロック）
  ↓ active records 解決 → endCrossfade(id)      ISRDSPHandle.cpp:103-119
  ↓   from→Retired / to→Active / activeHandle=to / fading=null
  ↓ fading slot CAS（DSPCore* 取得）            Timer.cpp（D123-B 実装案）
  ↓ retirePublishedDSP(current, lifetimeMgr)    Timer.cpp:1868（D127-A 実装済み設計）
      identity = current（CAS 取得 DSPCore*、一次）
      receipt.handle を resolve で交叉検証      D122-B 契約
  ↓ DSPLifetimeManager::retire(DSPCore*)        DSPLifetimeManager.cpp:40
      retireDSPHandleForRuntime: map erase（DELETE-1）+ slot→Retired
  ↓ requestReclaimHandle                        AudioEngine.h:4351（DELETE-2）
      epoch 安全 → reclaim / 不安全 → pendingReclaimHandles_ 再試行
  ↓ enqueueWithRetry(destroyDSPCoreNode)        （DELETE-3）
  → RetireRouter drain → destroyDSPCoreNode → ~DSPCore
```

**completion event identity → handle map identity の対応**: event は identity を運搬しない。identity は (a) slot CAS の DSPCore\*、(b) receipt の fadingHandle（開始時に保存・generation 付き）。map lookup は (a) のポインタで行われ、**D123 実測（`[D117_RETIRE] dsp=X retired=1 → enqueue → [D117_DESTROY] dsp=X`）で lookup 成功が実証済み**。lookup failure（未登録）は retired=false → 静かに return するが、publish 済み DSPCore は必ず登録済み（lifetime.activate）のため発生しない。

## 4. Generation / pending semantics（G7）

`crossfadeRuntime_` は `bumpCrossfadeGeneration()` を start/complete で実施 — generation anchor 済み（BUG-028 五次レビュー §8）。

**「古い completion が新しい crossfade を complete しない」ことの証明**:
- completion consume は **active records のみ** endCrossfade する（id 指定または active 走査）
- 新しい crossfade（B→C）の record は active だが、その endCrossfade は**自分の完了イベント**でのみ実行
- 古い completion event（A→B）は A→B record を inactive 化するだけで、B→C には触れない（record 走査は id 一致のみ）
- pending_ は complete() でのみクリア — consume block は complete() を**呼ばない**（呼ぶと新 crossfade を誤完了させるため。complete() の呼び出しは immediate-retire 分岐 / emergency / fadeCompleted ゲート内の現行位置を維持）

## 5. completion queue capacity 32 の liveness（G5）

| 状況 | 挙動 | ownership 終端 |
| --- | --- | --- |
| push 成功 | Timer が消費 → retire | ✅ |
| **push 失敗（full 32）** | drop + dropCount 増加（HealthMonitor 監視済み） | **fading slot は占有されたまま** — retire されない |
| drop 後の recovery | **タイムアウト net が唯一**（getFadeAgeUs > fadeTimeSec + margin → 強制 sweep） | 契約化必須 |

**確定**: drop 時、fading slot は占有継続・record は active のまま → **「fade 超過タイムアウト（getFadeAgeUs ベース）が唯一の recovery」**として契約化。Timer tick で `pending_ && age > fadeTime + margin` を検出したら強制 sweep（active records 解決 + slot CAS + retire）を実行する。実装は D129 本体に含める（現行コードに該当処理は存在しないことを確認済み）。

## 6. 追加診断（D129 実装時・macro-gated 4 点 + 1 点）

```text
[D129_XFADE_START]    xfadeId / fromHandle / toHandle / fromDSP / toDSP   （DSPTransition crossfade 分岐）
[D129_XFADE_COMPLETE] xfadeId(0=signal) / fadingHandle / fadingDSP          （Timer consume）
[D129_XFADE_RETIRE]   xfadeId / handle / DSP / retireResult                 （retirePublishedDSP）
[D129_XFADE_TIMEOUT]  xfadeId / ageUs / handle / DSP                        （timeout net）
[D129_ADMISSION]      seq / generation / decision / deferReason             （既存要求の 1 点追加）
```

RT 側は SPSC push のみ（alloc/lock/blocking なし）。`[D129_XFADE_START]` の fromDSP/toDSP は CoordinatorLoop 上の値（RT からは読まない）。

## GO/NO-GO（G1-G10）

| Gate | 判定 |
| --- | --- |
| G1 completion → unique identity | **PASS** — identity は slot CAS（DSPCore\*）+ receipt（handle 交叉検証）。event はシグナル |
| G2 A→B→C retire target 一意 | **PASS** — A は completion、B は claim-fail 即時 retire。双方 pointer-based 冪等 |
| G3 fading slot clear ABA-safe | **PASS** — slot は null→DSP→null の単調遷移。null 挟みのため ABA 不成立 |
| G4 completion → handle → DSPCore 一意 | **PASS**（map lookup + resolve generation 検証） |
| G5 queue drop 時 ownership 終端 | **PASS** — タイムアウト net を唯一の recovery として契約化（現行コードに処理なし → D129 実装項目） |
| G6 timeout fallback 対象 identity 一意 | **PASS**（slot CAS ポインタ + active records） |
| G7 新 crossfade の誤完了なし | **PASS**（record id 一致のみ endCrossfade・complete() は consume block から呼ばない） |
| G8 admission deferral reason 特定可能 | **PASS**（`[D129_ADMISSION]` trace 設計済み — 実測は D129 実装後） |
| G9 RT alloc/lock/blocking なし | **PASS**（SPSC push のみ） |
| G10 production semantics 変更 = 0 | **PASS**（本監査は read-only） |

**D129-1: PASS（10/10）** → **D129-2（M1+M2+M3+M5+M6 統合実装契約レビュー）へ進行可。**

## D129-2 契約に持ち込む設計修正（本監査で確定した分）

1. completion event は **identity 非携帯の純シグナル**（bool ではなく SPSC イベント。identity は slot CAS + receipt が保持）
2. **claim 失敗時の dual retire**: slot 占有者（中断 crossfade の fading DSP）と新 oldDSP の双方を retire（冪等・EBR 保護）+ 当該 record unregister
3. **fade 超過タイムアウト net**: `pending_ && getFadeAgeUs() > fadeTime + margin` → 強制 sweep（queue drop の唯一 recovery）
4. **records_ 滞積**: inactive record の定期的除去（または容量監視）を D129 診断で観測
5. `complete()` の呼び出し位置を consume block から**外す**（新 crossfade 誤完了防止 — G7）

## 生成物

- 本ファイル（`evidence/D129_1_CROSSFADE_OWNERSHIP_OVERLAP_AUDIT.md`）
- production source 変更: **0**
