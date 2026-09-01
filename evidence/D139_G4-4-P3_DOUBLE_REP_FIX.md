# D139 — G-4.4-P3 Double Representation 修復 (Implementation Record)

**Date:** 2026-08-31 (+09:00)
**Type:** implementation（delivery uniqueness 単独 Gate。P4/P5/P6 非touch）
**Production delta（P2 比）:** `ISRRuntimePublicationCoordinator.cpp` +19（same-holder repair 1 ブロックのみ）
**Test delta:** `ISRSemanticValidationTests.cpp` +150（T-P3-1..6 + openDurableWindow ヘルパ + main 登録）
**Build evidence:** `evidence/g44p3_ctest.log` — Debug full build→CTest `100% tests passed out of 40`（DBG_CTEST_EXIT=0）/ Release 同（REL_CTEST_EXIT=0）
**ConvoPeq.md:** 実装後再生成 `Generated: 2026-08-31 09:28:16`（`G-4.4-P3` マーカー 9 件 = 実ツリーと同期確認）
**Verdict: P3 実装完了・AC-P3-1..10 成立（AC-P3-9 は公式 run で成立、ただし §5 の既存断続 AV を別途報告）**

---

## 1. 実装前再確認（指示 §2・§6）

### same-holder 条件の十分性 → 十分と確認
`recoveryObligationId` の書込元は CoordinatorLoop の付着/上書き（cpp:1006/1167）のみで、take（cpp:1208）/settle(true)（cpp:1242）は state だけを変え oblId を保持。`NoAdmission` は構造体まるごとリセット（cpp:1229/1248）で oblId=0。よって redrive の else 分岐（`state != NoAdmission` 確定済み）では **`recoveryObligationId == obligationId` 単体で holder 判定として十分**（state との組合せ明示は不要）。lease 連鎖 `DurablePending(O)→take→Building(O)→settle(true)→DurablePending(O)` は現行コードで保証済み（take/settle が oblId を触らない）。

### wake 必要性（Builder predicate 基準）→ 必要と確定
repair 後の状態は `slot=DurablePending(O) ∧ delivery=Durable`。Builder の wait predicate は `{hasPendingTask, publishRetryReady, recoveryPending, exit}`（RebuildDispatch:855-858）で **durable slot 占有は predicate に現れない**。窓は Builder 自身が開ける（settle(true)+markTransientFailure）が、spin 上限 break（RebuildDispatch:1082-83）で Builder が眠ると durable 表現は放置される。よって **latch は必要**（P2 と同一プロトコル）。Builder がループ内の場合は冗長 wake は 1=no-op cycle で無害（predicate 状態式）。**「毎 tick notify」への逆戻りではない**（実付着/実 repair のときだけ発火）。

## 2. 実装（唯一の変更点）

`redriveDeferredRecovery()` durable-busy 分岐（cpp:1185-1189）:
```cpp
if (pendingRecoveryAdmission_.recoveryObligationId == obligationId) {
    s.delivery = ObligationDeliveryState::Durable;   // metadata re-sync (NOT a new attach)
    redriveWakePending_ = true;
    return;
}
// Else attempt transport queue. (different holder — C13 fallback semantics unchanged)
```
- **実体を生成しない**: queue push なし（`pendingIntentCount_` 不変）、durable 書込なし（state/oblId/buildSource 不変）、tryInsert/resolve 非経由（**ΔL=0・oblId 追加なし**）。
- different-holder は従来どおり transport fallback（AC-P3-4/C13 維持）。
- 禁止リスト（markTransientFailure/settle/take/pendingRecoveryAdmission_ 構造/queue/capacity/pendingIntentCount_/K=4/recoveryGeneration/coalesce/supersession/episode/target/P2 wake プロトコル/memory order/Building overwrite/shutdown/scheduler）— **全て無変更**（diff = 上記 1 ブロック + テストのみで実証、AC-P3-10）。

## 3. 二重住処の生成不能証明（container + delivery + ordering の 3 点）

D136-B/D138 の反例 t0-t4 を再追跡:
- t2 終了時: `slot=DurablePending(O) ∧ delivery=None`（窓は依然開く — markTransientFailure は禁止により不変）。
- t3（redrive）: 旧実装は cpp:1174-1177 で transport push → 二重住処。**新実装は same-holder 検査で repair 分岐に入り transport を生成しない**（push 到達不能）。
- 生成経路の網羅性: 同一 oblId の durable∧transport 同時占有を作り得る書込は (a) redrive transport fallback（本修正で same-holder を排除）、(b) submit push（cpp:976-978）— ただし submit が push に到達する前に同一スレッドで opportunistic redrive（cpp:902）が走り、same-holder 窓は repair で delivery=Durable に復同期される → coalesce 早期 return（cpp:943-944: wasDeferredBefore ∧ delivery!=None）で push 到達不能。(c) durable 上書き（cpp:998-1008）は同一 oblId の payload を**更新するだけで transport を増やさない**。**∎ 生成不能**（AC-P3-1/2/3）。
- delivery==Durable ⇒ slot 保持（INV-P3-3'）: repair が窓を閉じることで恢复。submit 経路は元々同一 oblId 上書きのみで不変条件を壊さない。

## 4. テスト（T-P3-1..6、public API のみ）

| T | 内容 | 結果 |
|---|---|---|
| T-P3-1 | 窓（slot=O ∧ None）→ redrive → **transport 非生成**（pendingIntentCount 不変・pop nullopt）+ delivery 再同期（take が O を返す）+ wake latch 発火 | ✅ |
| T-P3-2 | O1 durable 保持 + O2 None → redrive → O2=Transport（+1 entry）・O1 durable 無傷（C13 維持） | ✅ |
| T-P3-3 | repair 後 `¬(Transport(O) ∧ Durable(O))` を直接検証（pop nullopt ∧ take==O） | ✅ |
| T-P3-4 | redrive ×3 → 初回のみ wake・以後 skip（delivery=Durable）・表現増なし | ✅ |
| T-P3-5 | failure→redrive 循環 ×2 で毎回 durable-XOR（transport 一切なし）、4 回目 mark で ResolvedFailed・以後 wake 源なし | ✅ |
| T-P3-6 | free-slot 付着経路（P2 latch 契約）不変 + settle(false) クリア | ✅ |
既存 P2 テスト（T-P2-1/5）は repair 経路でも latch が立つため成立継続（T-P2-5 の「各周期 1 wake」がそのまま通ることを実測確認）。既存 40 テスト回帰なし。

## 5. ★ 実測で判明した**既存の断続 AV**（P3 と無関係 — 混在させず報告）

公式 CTest 前の負荷試験で、Release テスト exe に**低頻度（約 1/9〜1/15）の断続的アクセス違反（0xC0000005）**を観測。切り分け結果:
- マーカー二分: クラッシュは P2/P3 テスト**完了後**（post-P3 マーカー以降）。
- **bisect2 決定的**: P2+P3 テストブロックをスキップしても再現（RUN_9/40）。この構成では既存テスト（R18_T9/C16 等）は same-holder 窓を作らない（holder は常に異 oblId）ため、**P3 repair 経路はデッドコード** — それでもクラッシュ → **P3 実装・P3 テストは無罪**。
- WER: フォールトは exe 内同一関数形状（double 4 フィールドを NaN ガード付きでコピー、+0x198〜0x1B0 書込、r9 非 NULL だが不正領域へ書込 = 潜在的自由領域参照/ヒープ破損の遅延顕在化が疑われる）。
- 最終クリーンバイナリ: Release 20/20・Debug 20/20 合格、公式フル CTest 40/40×2 合格（再現せず＝低頻度）。
- 帰属: P2 以前の既存テスト/teardown に潜む可能性が高い（P2 era の CTest は各 1 回の実行のみで、~1/10 頻度を検出できていなかった）。**P3 の範囲外**であり、指示の Gate 分離に従い修正しない。→ **P4（memory-order/潜在 UB）の診断対象として正式に引き継ぎ**（WER 記録・bisect データはこのファイルに添付）。

## 6. Acceptance Criteria 判定

| AC | 判定 | 根拠 |
|---|---|---|
| P3-AC1 二重同時存在の生成排除 | ✅ | §3 生成不能証明 + T-P3-1/3 |
| P3-AC2 same-holder で transport 非付与 | ✅ | cpp:1185-1189 + T-P3-1 |
| P3-AC3 既存 durable 実体と delivery 再同期 | ✅ | T-P3-1/3（take==O・pop nullopt） |
| P3-AC4 different-holder fallback 維持 | ✅ | T-P3-2 + C13 合格 |
| P3-AC5 repeated redrive で重複なし | ✅ | T-P3-4 |
| P3-AC6 failure→redrive cycle で二重なし | ✅ | T-P3-5 |
| P3-AC7 P1/P2 wake/liveness 非破壊 | ✅ | T-P2-1/5・T-P3-6・既存 40 合格 |
| P3-AC8 ΔL=0・oblId 追加なし | ✅ | repair は enum+latch のみ（tryInsert/resolve 非経由）+ T-P3-5 liveCount 検証 |
| P3-AC9 Debug/Release build+CTest 全 PASS | ✅ | g44p3_ctest.log 40/40×2（§5 の断続 AV は別建て報告） |
| P3-AC10 diff は delivery uniqueness 限定 | ✅ | cpp +19（1 ブロック）+ tests のみ |

## 7. 残存（次 Gate 入力）
- **P4**: D136-D memory-order 契約 + **§5 の断続 AV 診断**（同一の潜在 UB/所有問題として一体で追うことを推奨）。
- P5: Building 中 overwrite 契約（cpp:991 の sub-state 非照合は不変 — 本修正は窓を作らない）。
- P6: 実 Engine 統合 wake テスト。

## STOP
P3 完了。**P4/P5/P6 には進まない。**
