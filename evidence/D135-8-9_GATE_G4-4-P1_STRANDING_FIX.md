# D135-8/9 Gate G-4.4-P1 — Deterministic Stranding 修復（Implementation Record）

**Date:** 2026-08-31 (+09:00)
**Type:** implementation（単一欠陥・単一 Gate。wake 修正は意図的に未混入）
**Scope:** `src/audioengine/AudioEngine.RebuildDispatch.cpp` のみ **+11 行**（呼び出し 2 行 + コメント 9 行）。他ファイルへの変更 0。
**Build evidence:** `evidence/g44p1_ctest.log` — Debug full build→CTest `100% tests passed out of 40`（DBG_CTEST_EXIT=0）/ Release 同 `100% tests passed out of 40`（REL_CTEST_EXIT=0）
**ConvoPeq.md:** 実装後再生成 `Generated: 2026-08-31 00:52:36`（`G-4.4-P1` マーカー 2 件 = 実ツリーと同期）

## 修復対象（D136-A 新規発見）
transport recovery の build 失敗（旧 :996-1000）・warmup 失敗（旧 :1009-1023）が `continue` のみで `markTransientFailure()` を呼ばず、pop 済み obligation が **Live+delivery=Transport** に固定 → redrive 候補（`delivery==None` のみ、cpp:1116）から恒久除外 → recovery の決定論的 stranding。durable 側（settle(true)+markTransientFailure）との非対称が原因。

## 実装
両失敗経路に `runtimePublicationBridge_.markTransientFailure(recovery->obligationId);` を追加（warmup 側は未コミット DSP 破棄の**後**、continue の前）。効果の連鎖:
```
transport build/warmup 失敗（intent は pop 済み＝transport 表現は消費）
  → markTransientFailure: delivery=None（cpp:1078）+ consecutiveFailureCount+1（1080-82）
  → 次 tick redrive 候補化（cpp:1112-1118）→ durable free なら Durable / なければ transport 再取得
  → 枯渇（>= kMaxObligationConsecutiveFailures=4）なら ResolvedFailed 終端（cpp:1085-87、唯一の公認 Failed）
```

## 実装前トレース（指示 6 点＋不変条件確認）
1. **transport recovery build/warmup failure**: RebuildDispatch:996-1001/1009-1024（現 996-1008/1016-1036）— pop 直後、durable 不干与。
2. **markTransientFailure**（cpp:1066-1092）: Live のみ作用（1073）、`obligationId==0` no-op（1068）、delivery=None + counter、枯渇で resolve(Failed)。カウンタ操作なし（pendingIntentCount_/liveCount_ 不変）。
3. **redriveDeferredRecoveryObligations**（cpp:1110-1120）: Live ∧ delivery==None のみに再付与。
4. **resolveRecoveryObligation**（cpp:1023-1048）: Retry は Live 維持、terminal は resolve CAS 経由のみ。
5. **transport pop**: popRecoveryRequest（cpp:1250-1263）— reservation −1（1261）、delivery は変えない（消費表現は delivery=Transport のまま残る＝stranding の原因側）。
6. **delivery 全 writer**: cpp:978（Transport/submit）/ 993（None/defer）/ 1008（Durable/submit）/ 1078（None/markTransientFailure）/ 1169（Durable/redrive）/ 1176（Transport/redrive）/ h:402（None/tryInsert）— 全て CoordinatorLoop 単一書込者。

**不変条件「transport failure 経路に durable slot が存在しない」の決着**:
- durable slot 書込元は cpp:998（submit、**push 失敗後にのみ**到達）/ 1159（redrive、**NoAdmission のときのみ**）/ take/settle/discard（consumer 側）に限定。通常の transport 付着（push 成功）は durable を触らない → **通常フローでは transport-resident obligation は durable slot を保持しない**。
- 例外は D136-B で確定済みの既存窓（durable transient failure 後の delivery=None×durable 保持、または coalesce 再 push 時の durable 残留）のみ。**P1 の変更はこの窓で新規相互作用を導入しない** — 同一窓では既存 durable 側 call site（:1091/:1115）が既に同じ `markTransientFailure`（delivery=None 化）を行っており、P1 は同じ挙動を transport 失敗時に適用するだけ。窓自体の修復は P3 の対象（指示分離どおり）。
- 枯渇→ResolvedFailed 時に durable が同一 oblId を保持していても、durable 表現は Builder に一度消費され resolve は冪等 no-op（h:418-419、C7/C9 実証済み）— 台帳破壊なし。

## spin 安全性
transport ループは各反復で queue を 1 消費（有限）＋ obligation counter が 4 で終端。durable ループ側の Builder-local `kMaxRecoveryConsecutiveFailures=4`（spin 防止）は不要（pop 主体のため無限自己再取得が構造的に起きない）。

## 禁止事項の遵守
durable slot 設計 / take / settle / pendingRecoveryAdmission_ memory order / SemanticRecoveryTarget / coalesce / supersession / retry scheduler / capacity 定数 / **wake 機構** — すべて無変更（diff = RebuildDispatch.cpp +11/−0 のみで実証）。

## 残存（意図的に未対応 — 次 Gate）
- P2: redrive→Builder wake（本修正で retry は可能になるが、**付着後の消費開始は依然として無関係 wake 依存** — stranding は「恒久」から「遅延」に改善されただけ。P2 まで liveness は未完了である点を明示）。
- P3: D136-B double representation（delivery uniqueness / INV-X1-5 違反窓）。
- P4: memory-order 契約。P5: Building 中 overwrite 契約。P6: T1/T2/T3。

## STOP
P1 実装 → build/CTest 合格 → 差分提示、まで。**P2 以降には進まない。**
